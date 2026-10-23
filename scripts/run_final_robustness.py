"""Task 04 E8 integration: synthetic fixtures or approved Validation only.

Corruptions and normalization are delegated to the shared RobustnessEvaluator.
Official Test stays blocked while Configuration Lock is a draft.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import numbers
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import yaml

SOURCE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SOURCE_ROOT))

from src.data.dataset import DogBreedDataset
from src.data.transforms import build_transforms
from src.models.model_factory import build_model
from src.robustness.config import E8Config, load_e8_config
from src.robustness.robustness_eval import RobustnessEvaluator
from src.utils.config import load_experiment_config

E4_CHECKPOINT_SHA256 = "73df0cc1c5b16866d4facc910f3e4d3ab481e3f12f1681a59fa7d61e3845c47b"
DEFAULT_CHECKPOINT = Path("outputs/checkpoints/E4/seed_42/best.pt")
DEFAULT_CONFIG = Path("configs/experiments/e8_robustness.yaml")
NUM_CLASSES = 30
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _assert_not_official_test(path: str | Path) -> None:
    text = str(path).replace("\\", "/").lower().replace("-", "_").replace(" ", "_")
    if "official_test" in text or text.rsplit("/", 1)[-1] == "test.csv":
        raise RuntimeError(f"Task 04 runner refuses to use Official Test: {path}")


def _reject_symlinks(path: Path) -> None:
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
           for p in (path, *path.parents)):
        raise ValueError(f"Symlinks/junctions are not allowed for Task 04 inputs/outputs: {path}")


def _resolve(path: str | Path) -> Path:
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = SOURCE_ROOT / result
    _assert_not_official_test(result)
    _reject_symlinks(result.absolute())
    result = result.resolve()
    _assert_not_official_test(result)
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _assert_validation_csv(path: Path) -> None:
    canonical = SOURCE_ROOT / "data/splits/validation.csv"
    if path != canonical.resolve():
        raise RuntimeError("Validation requires the canonical repo data/splits/validation.csv")
    _reject_symlinks(canonical)
    # Path/name checks alone do not establish that this is the approved split.
    try:
        approved = subprocess.check_output(
            ["git", "show", "HEAD:data/splits/validation.csv"], cwd=SOURCE_ROOT,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Cannot verify the committed canonical Validation split") from exc
    if path.read_bytes().replace(b"\r\n", b"\n") != approved.replace(b"\r\n", b"\n"):
        raise RuntimeError("Canonical Validation CSV differs from its committed approved content")


def _load_e4_model(checkpoint_path: Path) -> nn.Module:
    checkpoint_path = _resolve(checkpoint_path)
    actual = _sha256_file(checkpoint_path)
    if actual != E4_CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: expected {E4_CHECKPOINT_SHA256}, got {actual}")
    cfg = load_experiment_config(SOURCE_ROOT / "configs/experiments/e4_cutmix.yaml")
    if (cfg["experiment"]["id"] != "E4" or cfg["experiment"]["seed"] != 42
            or cfg["model"]["type"] != "mobilenet_v2"
            or cfg["model"]["backbone_frozen"] is not True
            or cfg["model"]["num_classes"] != NUM_CLASSES
            or cfg["augmentation"]["cutmix"]["enabled"] is not True):
        raise ValueError("Expected E4 MobileNetV2 Frozen + CutMix, seed 42, 30 classes")
    cfg = copy.deepcopy(cfg)
    cfg["model"]["weights"] = "none"  # Never download pretrained weights.
    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    return model.eval()


def _build_pre_norm_transform(data_payload: dict | None = None) -> T.Compose:
    if data_payload is None:
        data_payload = _yaml(SOURCE_ROOT / "configs/data.yaml")
    evaluation = build_transforms(data_payload, split="validation")
    # Reuse shared transform components, excluding only its final Normalize.
    if (not isinstance(evaluation.transforms[-1], T.Normalize)
            or evaluation.transforms[-1].mean != IMAGENET_MEAN
            or evaluation.transforms[-1].std != IMAGENET_STD
            or data_payload["preprocessing"]["evaluation"] != {"resize": 256, "center_crop": 224}):
        raise ValueError("Shared preprocessing differs from the approved E4 protocol")
    return T.Compose(evaluation.transforms[:-1])


class _PreNormDataset(torch.utils.data.Dataset):
    """Shared RGB dataset with pre-normalization tensors and unchanged CSV IDs."""

    def __init__(self, csv_path: Path, images_dir: Path, *, split: str,
                 data_payload: dict, mapping: dict):
        self.images_dir = images_dir
        self.dataset = DogBreedDataset(csv_path, images_dir, _build_pre_norm_transform(data_payload))
        self.records = self.dataset.rows
        if split == "validation" and len(self.records) != 300:
            raise ValueError("Approved Validation must contain exactly 300 images")
        for row in self.records:
            image_id, label = row["relative_path"], row["class_id"]
            if (not isinstance(image_id, str) or not image_id.strip()
                    or "\\" in image_id or ":" in image_id
                    or PurePosixPath(image_id).is_absolute()
                    or any(part in {"", ".", ".."} for part in image_id.split("/"))):
                raise ValueError("relative_path must be a safe, unchanged, non-empty string ID")
            if (isinstance(label, bool) or not isinstance(label, numbers.Integral)
                    or not 0 <= label < NUM_CLASSES
                    or mapping.get(str(label)) != row["breed_name"]):
                raise ValueError("CSV label/breed_name does not match the approved class mapping")
            expected_source = "official_train" if split == "validation" else "synthetic"
            if row["source_split"] != expected_source:
                raise ValueError(f"{split} CSV source_split must be {expected_source}")
            if split == "synthetic" and not image_id.startswith("synthetic/"):
                raise ValueError("Synthetic fixtures require synthetic/ image IDs")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image_id = self.records[index]["relative_path"]
        path = _resolve(self.images_dir / image_id)
        if not path.is_relative_to(self.images_dir):
            raise ValueError("Image path escapes images_dir")
        sample = self.dataset[index]  # RGB -> shared resize/crop -> [0,1].
        if sample["relative_path"] != image_id:
            raise ValueError("Shared dataset changed the CSV image ID")
        return sample["image"], sample["label"], image_id


@torch.inference_mode()
def _evaluate_loader(model: nn.Module, loader: DataLoader, cfg: E8Config) -> dict:
    evaluator = RobustnessEvaluator(model, cfg)
    counts = {name: 0 for name in cfg.conditions}
    image_ids = []
    seen = set()
    for images, labels, paths in loader:
        paths = list(paths)
        if any(path in seen for path in paths):
            raise ValueError("Duplicate image IDs across DataLoader batches")
        metrics = evaluator.evaluate(images, labels, paths)
        seen.update(paths)
        image_ids.extend(paths)
        for name in counts:
            key = "clean_accuracy" if name == "clean" else f"{name}_accuracy"
            # Shared evaluator returns integer-correct/N. Recover that count,
            # then aggregate by samples, including a short final batch.
            counts[name] += round(metrics[key] * len(labels))
    if not image_ids:
        raise ValueError("DataLoader produced no images")
    accuracies = {name: correct / len(image_ids) for name, correct in counts.items()}
    result = {"clean_accuracy": accuracies["clean"], "conditions": list(cfg.conditions),
              "n_samples": len(image_ids), "image_ids": image_ids}
    for name, accuracy in accuracies.items():
        if name != "clean":
            result[f"{name}_accuracy"] = accuracy
            result[f"{name}_accuracy_drop"] = accuracies["clean"] - accuracy
    return result


def _make_synthetic_model(seed: int) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, NUM_CLASSES))
    return model.eval()


def _smoke_test(cfg: E8Config, batch_size: int = 32) -> dict:
    generator = torch.Generator().manual_seed(cfg.reproducibility.seed)
    images = torch.rand(8, 3, cfg.input_size, cfg.input_size, generator=generator)
    labels = torch.randint(0, NUM_CLASSES, (8,), generator=generator)
    samples = [(image, label, f"synthetic/{i:04d}.jpg")
               for i, (image, label) in enumerate(zip(images, labels))]
    model = _make_synthetic_model(cfg.reproducibility.seed)
    result = _evaluate_loader(model, DataLoader(samples, batch_size=batch_size), cfg)
    return dict(result, mode="smoke_test_synthetic", split="synthetic", synthetic=True,
                official_evaluation=False, checkpoint=None, checkpoint_sha256=None)


def _output_path(path: str | Path, split: str) -> Path:
    path = _resolve(path)
    if path.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {path}")
    dedicated = SOURCE_ROOT / "outputs/robustness/task04" / split
    if not path.is_relative_to(dedicated):
        # Synthetic fixture output can live outside the repository in tmp_path.
        if split != "synthetic" or path.is_relative_to(SOURCE_ROOT):
            raise ValueError(f"Task 04 {split} output must be under {dedicated}")
    return path


def _write_results(path: Path, results: dict) -> None:
    payload = json.dumps(results, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def run_final_robustness(*, csv_path: str | Path, images_dir: str | Path,
                         output_path: str | Path, split: str = "validation",
                         checkpoint_path: str | Path | None = None,
                         config_path: str | Path = DEFAULT_CONFIG,
                         batch_size: int = 32, device: str = "auto") -> dict:
    if split not in {"synthetic", "validation"}:
        raise RuntimeError("Task 04 permits only synthetic or approved Validation; Official Test is blocked")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    csv_path, images_dir = _resolve(csv_path), _resolve(images_dir)
    if split == "validation":
        _assert_validation_csv(csv_path)
    elif checkpoint_path is not None:
        raise ValueError("Synthetic mode uses a dummy model, never an E4 checkpoint")
    if split == "synthetic" and (csv_path.is_relative_to(SOURCE_ROOT)
                                 or images_dir.is_relative_to(SOURCE_ROOT)):
        raise ValueError("Synthetic fixtures must be outside the repository's real data")
    output = _output_path(output_path, split)
    cfg = load_e8_config(_resolve(config_path))
    data_payload = _yaml(SOURCE_ROOT / "configs/data.yaml")
    if cfg.input_size != data_payload["preprocessing"]["evaluation"]["center_crop"]:
        raise ValueError("E8 input_size must match the shared CenterCrop")
    with (SOURCE_ROOT / "data/metadata/class_mapping.json").open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    if (set(mapping) != {str(i) for i in range(NUM_CLASSES)}
            or len(set(mapping.values())) != NUM_CLASSES):
        raise ValueError("Expected the approved 30-class mapping")
    dataset = _PreNormDataset(csv_path, images_dir, split=split,
                              data_payload=data_payload, mapping=mapping)
    if split == "synthetic":
        model = _make_synthetic_model(cfg.reproducibility.seed)
        checkpoint = None
    else:
        checkpoint = _resolve(checkpoint_path or DEFAULT_CHECKPOINT)
        model = _load_e4_model(checkpoint)
    runtime_device = _device(device)
    model = model.to(runtime_device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    result = _evaluate_loader(model, loader, cfg)
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT,
                                          stderr=subprocess.PIPE, text=True).strip()
    except subprocess.CalledProcessError:
        git_sha = "unknown"
    result.update({
        "experiment_id": "E8_TASK04_ROBUSTNESS", "split": split,
        "mode": "synthetic_fixture" if split == "synthetic" else "validation_integration",
        "synthetic": split == "synthetic", "official_evaluation": False,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": E4_CHECKPOINT_SHA256 if checkpoint else None,
        "model_experiment": "E4" if checkpoint else "synthetic_dummy",
        "num_classes": NUM_CLASSES, "device": str(runtime_device),
        "git_sha": git_sha, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split_csv": str(csv_path), "batch_size": batch_size,
        "pytorch_version": torch.__version__, "seed": cfg.reproducibility.seed,
        "preprocessing": data_payload["preprocessing"],
        "corruption_config": {
            name: {key: list(value) if isinstance(value, tuple) else value
                   for key, value in asdict(getattr(cfg, name)).items()}
            for name in ("gaussian_blur", "gaussian_noise", "occlusion")
        },
    })
    _write_results(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="E8 Task 04: synthetic or approved Validation only")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=None)
    parser.add_argument("--split", choices=["synthetic", "validation"], default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--official-test", action="store_true", help="Blocked in Task 04")
    args = parser.parse_args()
    try:
        if args.official_test:
            raise RuntimeError("--official-test is blocked in Task 04; Configuration Lock is not approved")
        if args.smoke_test:
            if args.split == "validation" or args.csv or args.images_dir or args.checkpoint:
                raise ValueError("Smoke test is synthetic only; real dataset/checkpoint arguments are forbidden")
            if args.batch_size <= 0:
                raise ValueError("batch_size must be a positive integer")
            output = _output_path(args.output, "synthetic") if args.output else None
            results = _smoke_test(load_e8_config(_resolve(args.config)), args.batch_size)
            if output:
                _write_results(output, results)
        else:
            split = args.split or "validation"
            if split == "synthetic" and (not args.csv or not args.images_dir):
                raise ValueError("Synthetic fixtures require --csv and --images-dir")
            results = run_final_robustness(
                csv_path=args.csv or "data/splits/validation.csv",
                images_dir=args.images_dir or "data/raw/stanford_dogs/Images",
                output_path=args.output or f"outputs/robustness/task04/{split}/results.json",
                checkpoint_path=args.checkpoint, config_path=args.config,
                split=split, batch_size=args.batch_size, device=args.device,
            )
        print(json.dumps(results, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
