"""Reload best checkpoints and evaluate Validation + Efficiency for E10/E11."""
from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import DogBreedDataset
from src.data.transforms import build_transforms
from src.evaluation.efficiency import benchmark_model
from src.research.extended_config import load_research_config, research_output_files
from src.research.extended_models import build_research_model, get_model_provenance
from src.training.validator import validate_one_epoch
from src.utils.device import resolve_device
from src.utils.logger import write_json
from src.utils.paths import dataset_root, split_file


DEFAULT_CONFIGS = [
    "configs/experiments/e10a_efficientnet_no_cutmix.yaml",
    "configs/experiments/e10b_efficientnet_cutmix.yaml",
    "configs/experiments/e11a_vit_no_cutmix.yaml",
    "configs/experiments/e11b_vit_cutmix.yaml",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_single(
    config_path: str | Path,
    device: torch.device,
    dataset_dir: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    output_root: Path | None = None,
    warmup: int = 20,
    timed: int = 100,
) -> dict:
    config = load_research_config(config_path)
    exp_id = config["experiment"]["id"]
    seed = int(config["experiment"]["seed"])
    paths = research_output_files(exp_id, seed, output_root=output_root)
    ckpt_path = paths["best_checkpoint"]
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing best checkpoint for {exp_id}: {ckpt_path}")

    ckpt_sha = sha256_file(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    best_epoch = int(ckpt.get("epoch", -1))

    # Build architecture without downloading weights, then load saved checkpoint strictly.
    model = build_research_model(config, force_no_pretrained=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    root = dataset_root(dataset_dir)
    val_ds = DogBreedDataset(split_file("validation"), root / "Images", build_transforms(config, "validation"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    criterion = torch.nn.CrossEntropyLoss()
    val_metrics = validate_one_epoch(model, val_loader, criterion, device)

    eff = benchmark_model(
        model,
        device=device,
        checkpoint_path=None,  # already loaded strictly above
        num_classes=30,
        warmup=warmup,
        timed=timed,
    )

    summary = {
        "experiment_id": exp_id,
        "model_type": config["model"]["type"],
        "cutmix_enabled": bool(config["augmentation"]["cutmix"]["enabled"]),
        "seed": seed,
        "n_samples": len(val_ds),
        "best_epoch": best_epoch,
        "top1_accuracy": val_metrics["top1_accuracy"],
        "top5_accuracy": val_metrics["top5_accuracy"],
        "macro_f1": val_metrics["macro_f1"],
        "validation_loss": val_metrics["loss"],
        "total_params": eff["total_params"],
        "trainable_params": eff["trainable_params"],
        "model_size_mib": eff["size_mib"],
        "median_ms": eff["median_ms"],
        "mean_ms": eff["mean_ms"],
        "std_ms": eff["std_ms"],
        "benchmark_device": str(device),
        "benchmark_platform": platform.platform(),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint_sha256": ckpt_sha,
        "provenance": get_model_provenance(config),
    }
    write_json(paths["evaluation_summary"], summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate E10/E11 checkpoints on Validation + Efficiency")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-csv", default="outputs/research/e10_e11/evaluation_summary.csv")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"[evaluate_research_models] device={device}")
    rows = []
    for cfg_path in args.configs:
        res = evaluate_single(
            cfg_path,
            device=device,
            dataset_dir=args.dataset_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        rows.append(res)
        print(
            f"[{res['experiment_id']}] epoch={res['best_epoch']} | "
            f"top1={res['top1_accuracy']:.4f} | top5={res['top5_accuracy']:.4f} | "
            f"macro_f1={res['macro_f1']:.4f} | val_loss={res['validation_loss']:.4f} | "
            f"params={res['total_params']:,} (train={res['trainable_params']:,}) | "
            f"size={res['model_size_mib']:.2f} MiB | lat={res['median_ms']:.2f} ms"
        )

    csv_path = PROJECT_ROOT / args.output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "model_type",
        "cutmix_enabled",
        "seed",
        "n_samples",
        "best_epoch",
        "top1_accuracy",
        "top5_accuracy",
        "macro_f1",
        "validation_loss",
        "total_params",
        "trainable_params",
        "model_size_mib",
        "median_ms",
        "mean_ms",
        "std_ms",
        "benchmark_device",
        "benchmark_platform",
        "cuda_device_name",
        "checkpoint_sha256",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"Saved summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
