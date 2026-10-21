"""E8 Robustness Integration Runner — Task 04.

Loads the locked E4 checkpoint (MobileNetV2 Frozen, seed 42), builds a
DataLoader from a permitted split CSV, and evaluates accuracy across four
conditions: Clean, Gaussian Blur, Gaussian Noise, Occlusion.

Usage (Validation set — permitted):
    python scripts/run_final_robustness.py \\
        --checkpoint outputs/checkpoints/E4/seed_42/best.pt \\
        --csv        data/splits/validation.csv \\
        --images-dir data/images \\
        --output     outputs/robustness/validation_results.json

Usage (synthetic smoke-test — no checkpoint required):
    python scripts/run_final_robustness.py --smoke-test

Official Test is permanently blocked in Task 04.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robustness.config import load_e8_config
from src.robustness.corruptions import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_occlusion,
)

# ── Constants ─────────────────────────────────────────────────────────────────
E4_CHECKPOINT_SHA256 = "73df0cc1c5b16866d4facc910f3e4d3ab481e3f12f1681a59fa7d61e3845c47b"
OFFICIAL_TEST_KEYWORDS = ("official_test", "test.csv", "official-test")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 30


# ── Guard: Official Test ───────────────────────────────────────────────────────
def _assert_not_official_test(csv_path: str | Path) -> None:
    """Raise RuntimeError if csv_path looks like the Official Test split."""
    p = str(Path(csv_path)).replace("\\", "/").lower()
    for kw in OFFICIAL_TEST_KEYWORDS:
        if kw in p:
            raise RuntimeError(
                f"Task 04 runner refuses to run on Official Test data. "
                f"Matched keyword '{kw}' in path: {csv_path}"
            )


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_e4_model(checkpoint_path: Path) -> nn.Module:
    """Load MobileNetV2 Frozen (E4) from checkpoint_path and verify SHA-256."""
    from src.models.mobilenet_v2 import build_model as build_mobilenet

    # Verify checksum before loading weights
    actual_sha = _sha256_file(checkpoint_path)
    if actual_sha != E4_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch.\n"
            f"  Expected: {E4_CHECKPOINT_SHA256}\n"
            f"  Got:      {actual_sha}\n"
            f"  Path:     {checkpoint_path}\n"
            "Do not use a different checkpoint."
        )

    model_cfg = {
        "type": "mobilenet_v2",
        "weights": "none",   # weights come from checkpoint, not ImageNet download
        "backbone_frozen": True,
        "num_classes": NUM_CLASSES,
    }
    model = build_mobilenet(model_cfg)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Accept raw state_dict or wrapped {"model_state_dict": ...}
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


# ── Dataset ────────────────────────────────────────────────────────────────────
def _build_pre_norm_transform() -> T.Compose:
    """Resize + CenterCrop + ToTensor only — NO normalization.

    Normalization is applied after corruption inside the evaluator.
    """
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),  # [0, 1]
    ])


class _PreNormDataset(torch.utils.data.Dataset):
    """Dataset that returns (image_tensor_[0,1], label, relative_path).

    Wraps src.data.dataset.DogBreedDataset logic but with pre-norm transform.
    """

    def __init__(self, csv_path: Path, images_dir: Path):
        import pandas as pd
        from PIL import Image as _Image
        self._Image = _Image
        self.images_dir = images_dir
        self.transform = _build_pre_norm_transform()
        df = pd.read_csv(csv_path)
        required = {"relative_path", "class_id", "breed_name", "source_split"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        if df.empty:
            raise ValueError("Split CSV is empty.")
        if df["relative_path"].duplicated().any():
            raise ValueError("Duplicate relative_path in split CSV.")
        if not df["class_id"].between(0, 29).all():
            raise ValueError("class_id out of [0, 29] range.")
        self.records = df[["relative_path", "class_id"]].to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        row = self.records[idx]
        rel = str(row["relative_path"]).replace("\\", "/")
        img_path = self.images_dir / rel
        with self._Image.open(img_path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)   # [3, 224, 224] in [0, 1]
        return tensor, int(row["class_id"]), rel


# ── Normalization + inference ─────────────────────────────────────────────────
_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def _normalize(images: torch.Tensor) -> torch.Tensor:
    return (images - _MEAN.to(images.device)) / _STD.to(images.device)


@torch.inference_mode()
def _run_condition(
    model: nn.Module,
    loader: DataLoader,
    corrupt_fn,   # callable(batch_unnorm) -> batch_unnorm  or  None
    device: torch.device,
) -> tuple[float, list[str]]:
    """Return (top1_accuracy, list_of_image_ids)."""
    correct = 0
    total = 0
    image_ids: list[str] = []
    for tensors, labels, paths in loader:
        tensors = tensors.to(device)    # [B, 3, 224, 224] in [0, 1]
        labels = labels.to(device)
        if corrupt_fn is not None:
            tensors = corrupt_fn(tensors)
        normed = _normalize(tensors)
        logits = model(normed)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        image_ids.extend(paths)
    return correct / total, image_ids


# ── Corruption wrappers (batch-level) ─────────────────────────────────────────
def _blur_batch(images: torch.Tensor, cfg) -> torch.Tensor:
    return torch.stack([
        apply_gaussian_blur(img, kernel_size=cfg.kernel_size, sigma=cfg.sigma)
        for img in images
    ])


def _noise_batch(images: torch.Tensor, cfg, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.stack([
        apply_gaussian_noise(
            img.cpu(), mean=cfg.mean, std=cfg.std,
            clamp_range=cfg.clamp_range, generator=gen,
        ).to(images.device)
        for img in images
    ])


def _occlusion_batch(images: torch.Tensor, cfg, image_ids: list[str]) -> torch.Tensor:
    """Deterministic occlusion using stable string image_id (not batch index)."""
    import hashlib
    corrupted = []
    for img, iid in zip(images, image_ids):
        seed_str = f"42:{iid}:occlusion"
        seed_int = int.from_bytes(hashlib.sha256(seed_str.encode()).digest()[:4], "big")
        corrupted.append(
            apply_occlusion(
                img, side_fraction=cfg.side_fraction,
                fill_rgb=cfg.fill_rgb, image_id=seed_int,
            )
        )
    return torch.stack(corrupted)


# ── Synthetic mode (no checkpoint, no real images) ────────────────────────────
def _smoke_test(cfg) -> dict:
    """Run on purely synthetic data — verifies pipeline, no files needed."""
    print("[E8-Task04] Smoke-test mode: synthetic data, dummy model.")
    dummy = nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, NUM_CLASSES)
    ).eval()
    g = torch.Generator()
    g.manual_seed(42)
    images = torch.rand(8, 3, 224, 224, generator=g)   # [0, 1], pre-norm
    labels = torch.randint(0, NUM_CLASSES, (8,), generator=g)
    ids = [f"synthetic/{i:04d}.jpg" for i in range(8)]

    with torch.inference_mode():
        clean_logits = dummy(_normalize(images))
    clean_acc = (clean_logits.argmax(1) == labels).float().mean().item()

    blur_cfg = cfg.gaussian_blur
    noise_cfg = cfg.gaussian_noise
    occ_cfg = cfg.occlusion

    blur_imgs = _blur_batch(images, blur_cfg)
    noise_imgs = _noise_batch(images, noise_cfg, cfg.reproducibility.seed)
    occ_imgs = _occlusion_batch(images, occ_cfg, ids)

    with torch.inference_mode():
        blur_acc = (dummy(_normalize(blur_imgs)).argmax(1) == labels).float().mean().item()
        noise_acc = (dummy(_normalize(noise_imgs)).argmax(1) == labels).float().mean().item()
        occ_acc = (dummy(_normalize(occ_imgs)).argmax(1) == labels).float().mean().item()

    return {
        "mode": "smoke_test_synthetic",
        "n_samples": 8,
        "clean_accuracy": clean_acc,
        "gaussian_blur_accuracy": blur_acc,
        "gaussian_blur_accuracy_drop": clean_acc - blur_acc,
        "gaussian_noise_accuracy": noise_acc,
        "gaussian_noise_accuracy_drop": clean_acc - noise_acc,
        "occlusion_accuracy": occ_acc,
        "occlusion_accuracy_drop": clean_acc - occ_acc,
        "official_evaluation": False,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="E8 Robustness Integration Runner — Task 04")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/E4/seed_42/best.pt")
    parser.add_argument("--csv",         default="data/splits/validation.csv")
    parser.add_argument("--images-dir",  default="data/images")
    parser.add_argument("--config",      default="configs/experiments/e8_robustness.yaml")
    parser.add_argument("--output",      default="outputs/robustness/validation_results.json")
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--smoke-test",  action="store_true",
                        help="Run on synthetic data; no checkpoint or images needed.")
    parser.add_argument("--official-test", action="store_true",
                        help="[BLOCKED] Official Test disabled in Task 04.")
    args = parser.parse_args()

    # ── Hard block: official test flag ────────────────────────────────────────
    if args.official_test:
        print(
            "ERROR: --official-test is blocked in Task 04. "
            "Configuration Lock must be approved by team lead first.",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = load_e8_config(args.config)

    # ── Hard block: official test in config ───────────────────────────────────
    if cfg.restrictions.use_official_test:
        print(
            "ERROR: e8_robustness.yaml use_official_test=true blocked.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Smoke-test mode ───────────────────────────────────────────────────────
    if args.smoke_test:
        results = _smoke_test(cfg)
        print(json.dumps(results, indent=2))
        print("[E8-Task04] Smoke-test PASSED. No official data touched.")
        return

    # ── Real mode: CSV + checkpoint ───────────────────────────────────────────
    csv_path = Path(args.csv)
    _assert_not_official_test(csv_path)

    checkpoint_path = Path(args.checkpoint)
    images_dir = Path(args.images_dir)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if not images_dir.exists():
        print(f"ERROR: Images dir not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[E8-Task04] Loading E4 checkpoint: {checkpoint_path}")
    model = _load_e4_model(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[E8-Task04] Device: {device}")

    dataset = _PreNormDataset(csv_path, images_dir)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    n = len(dataset)
    print(f"[E8-Task04] Split: {csv_path.name} — {n} samples")

    blur_cfg = cfg.gaussian_blur
    noise_cfg = cfg.gaussian_noise
    occ_cfg = cfg.occlusion

    # ── Clean ─────────────────────────────────────────────────────────────────
    print("[E8-Task04] Running: clean")
    clean_acc, _ = _run_condition(model, loader, None, device)

    # ── Gaussian Blur ─────────────────────────────────────────────────────────
    print("[E8-Task04] Running: gaussian_blur")
    blur_acc, _ = _run_condition(
        model, loader,
        lambda imgs: _blur_batch(imgs, blur_cfg),
        device,
    )

    # ── Gaussian Noise ────────────────────────────────────────────────────────
    print("[E8-Task04] Running: gaussian_noise")
    noise_acc, _ = _run_condition(
        model, loader,
        lambda imgs: _noise_batch(imgs, noise_cfg, cfg.reproducibility.seed),
        device,
    )

    # ── Occlusion — needs image_ids from loader, run separately ──────────────
    print("[E8-Task04] Running: occlusion")
    occ_correct = 0
    occ_total = 0
    with torch.inference_mode():
        for tensors, labels, paths in loader:
            tensors = tensors.to(device)
            labels = labels.to(device)
            occ_imgs = _occlusion_batch(tensors, occ_cfg, list(paths))
            logits = model(_normalize(occ_imgs))
            occ_correct += (logits.argmax(1) == labels).sum().item()
            occ_total += labels.size(0)
    occ_acc = occ_correct / occ_total

    results = {
        "split": csv_path.name,
        "n_samples": n,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": E4_CHECKPOINT_SHA256,
        "device": str(device),
        "clean_accuracy": clean_acc,
        "gaussian_blur_accuracy": blur_acc,
        "gaussian_blur_accuracy_drop": clean_acc - blur_acc,
        "gaussian_noise_accuracy": noise_acc,
        "gaussian_noise_accuracy_drop": clean_acc - noise_acc,
        "occlusion_accuracy": occ_acc,
        "occlusion_accuracy_drop": clean_acc - occ_acc,
        "official_evaluation": False,
        "corruption_config": {
            "gaussian_blur": {"kernel_size": blur_cfg.kernel_size, "sigma": blur_cfg.sigma},
            "gaussian_noise": {"mean": noise_cfg.mean, "std": noise_cfg.std},
            "occlusion": {"side_fraction": occ_cfg.side_fraction, "fill_rgb": list(occ_cfg.fill_rgb)},
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"[E8-Task04] Results saved: {output_path}")
    print(f"  clean_accuracy            : {clean_acc:.4f}")
    print(f"  gaussian_blur_accuracy    : {blur_acc:.4f}  drop={clean_acc-blur_acc:+.4f}")
    print(f"  gaussian_noise_accuracy   : {noise_acc:.4f}  drop={clean_acc-noise_acc:+.4f}")
    print(f"  occlusion_accuracy        : {occ_acc:.4f}  drop={clean_acc-occ_acc:+.4f}")


if __name__ == "__main__":
    main()
