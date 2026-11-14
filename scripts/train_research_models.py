"""Train/Validation runner for Task 07 E10/E11 experiments."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import DogBreedDataset
from src.data.transforms import build_transforms
from src.research.extended_config import load_research_config, research_output_files
from src.research.extended_models import build_research_model, get_model_provenance, validate_frozen_backbone
from src.research.extended_optimizer import build_research_optimizer
from src.training.checkpoint import save_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.trainer import train_one_epoch
from src.training.validator import validate_one_epoch
from src.utils.device import resolve_device, use_mixed_precision
from src.utils.logger import append_history, run_metadata, write_json
from src.utils.paths import dataset_root, split_file
from src.utils.seed import seed_worker, set_seed


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_training(
    config_path: str | Path,
    dataset_dir: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    device_arg: str = "auto",
    dry_run: bool = False,
    output_root: Path | None = None,
) -> dict:
    config = load_research_config(config_path)
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch-size must be positive; num-workers cannot be negative.")
    effective = int(config["training"]["effective_batch_size"])
    if effective % batch_size != 0:
        raise ValueError(f"Physical batch-size {batch_size} must divide effective batch {effective}.")
    accumulation = effective // batch_size

    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    root = dataset_root(dataset_dir)

    train_ds = DogBreedDataset(split_file("train"), root / "Images", build_transforms(config, "train"))
    val_ds = DogBreedDataset(split_file("validation"), root / "Images", build_transforms(config, "validation"))

    train_gen = torch.Generator().manual_seed(seed)
    val_gen = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=train_gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=val_gen,
    )

    exp_id = config["experiment"]["id"]
    print(f"Experiment={exp_id}; seed={seed}; model={config['model']['type']}")
    print(f"Train={len(train_ds)}, Validation={len(val_ds)}, effective_batch={effective}, accumulation={accumulation}")

    if dry_run:
        for name, loader in [("train", train_loader), ("validation", val_loader)]:
            sample = next(iter(loader))
            print(f"{name}: image={tuple(sample['image'].shape)}; label={tuple(sample['label'].shape)}")
        print("RESEARCH DATA DRY-RUN PASSED. Official Test was not accessed.")
        return {"dry_run": True, "experiment_id": exp_id}

    paths = research_output_files(exp_id, seed, output_root=output_root)
    if paths["history"].exists() or paths["best_checkpoint"].exists():
        raise FileExistsError(
            f"Run output already exists. Do not overwrite an experiment silently: {paths['run_dir']}"
        )

    device = resolve_device(device_arg)
    model = build_research_model(config).to(device)
    validate_frozen_backbone(model, config["model"]["type"])
    optimizer = build_research_optimizer(model, config)

    criterion = torch.nn.CrossEntropyLoss()
    scheduler_cfg = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_cfg["mode"],
        factor=float(scheduler_cfg["factor"]),
        patience=int(scheduler_cfg["patience"]),
        min_lr=float(scheduler_cfg["min_lr"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision(device, config))
    stopper = EarlyStopping(patience=int(config["training"]["early_stopping_patience"]))

    metadata = run_metadata(config, device, batch_size, accumulation)
    metadata["provenance"] = get_model_provenance(config)
    write_json(paths["metadata"], metadata)

    cutmix_cfg = config["augmentation"]["cutmix"]
    cutmix_config = cutmix_cfg if cutmix_cfg.get("enabled") else None
    best_epoch = 0
    best_val: dict = {}

    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        training = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            accumulation_steps=accumulation,
            scaler=scaler,
            cutmix_config=cutmix_config,
        )
        validation = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(validation["macro_f1"])
        improved, should_stop = stopper.step(validation["macro_f1"])

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in training.items()},
            **{f"val_{k}": v for k, v in validation.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        append_history(paths["history"], row)

        if improved:
            best_epoch = epoch
            best_val = dict(validation)
            save_checkpoint(paths["best_checkpoint"], model, optimizer, epoch, validation, config)
            write_json(
                paths["validation_metrics"],
                {
                    "experiment_id": exp_id,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    **best_val,
                },
            )

        save_checkpoint(paths["last_checkpoint"], model, optimizer, epoch, validation, config)
        metric_samples = training.get("metric_samples")
        metric_note = f" | train metric samples={metric_samples}" if metric_samples is not None else ""
        print(
            f"Epoch {epoch:02d} | train loss={training['loss']:.4f} | "
            f"val macro-F1={validation['macro_f1']:.4f} | best={stopper.best:.4f}{metric_note}"
        )
        if should_stop:
            print(f"Early stopping after {epoch} epochs (patience={stopper.patience}).")
            break

    best_sha256 = sha256_file(paths["best_checkpoint"])
    summary_metrics = {
        "experiment_id": exp_id,
        "seed": seed,
        "best_epoch": best_epoch,
        **best_val,
        "best_checkpoint_sha256": best_sha256,
    }
    write_json(paths["validation_metrics"], summary_metrics)
    metadata["best_checkpoint_sha256"] = best_sha256
    write_json(paths["metadata"], metadata)
    print(f"Finished {exp_id}. Best epoch={best_epoch}, macro_f1={best_val['macro_f1']:.4f}, sha256={best_sha256}")
    return summary_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="DogBreedID Task 07 E10/E11 Training Runner")
    parser.add_argument("--config", required=True, help="configs/experiments/e10a_...yaml")
    parser.add_argument("--dataset-root", default=None, help="Root containing Images/")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        dataset_dir=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_arg=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
