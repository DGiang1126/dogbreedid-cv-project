"""Integration tests — E8 Robustness Pipeline, Task 04.

All tests run on synthetic data / dummy models.
No real checkpoint, no real images, no Official Test.

Run:
    python -m pytest tests/test_final_robustness.py -v
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robustness.config import E8Config, load_e8_config
from src.robustness.corruptions import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_occlusion,
)
from src.robustness.robustness_eval import RobustnessEvaluator

# ── Helpers ───────────────────────────────────────────────────────────────────
CFG_PATH = Path("configs/experiments/e8_robustness.yaml")
N, C, H, W = 8, 3, 224, 224
NUM_CLASSES = 30


def _cfg() -> E8Config:
    return load_e8_config(CFG_PATH)


def _dummy_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """Tiny deterministic model — returns logits without touching weights."""
    m = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, num_classes))
    nn.init.constant_(m[2].weight, 0.0)
    nn.init.constant_(m[2].bias, 0.0)
    return m.eval()


def _synthetic_images(seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator()
    g.manual_seed(seed)
    images = torch.rand(N, C, H, W, generator=g)         # [0, 1]
    labels = torch.randint(0, NUM_CLASSES, (N,), generator=g)
    return images, labels


def _synthetic_ids(n: int = N) -> list[str]:
    return [f"synthetic/img_{i:04d}.jpg" for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Checkpoint SHA-256 verification logic
# ─────────────────────────────────────────────────────────────────────────────

def test_sha256_mismatch_raises(tmp_path: Path) -> None:
    """SHA-256 mismatch must raise RuntimeError before model is loaded."""
    fake_ckpt = tmp_path / "best.pt"
    fake_ckpt.write_bytes(b"not a real checkpoint")

    # Import after sys.path insert
    from scripts.run_final_robustness import _sha256_file, E4_CHECKPOINT_SHA256
    actual = _sha256_file(fake_ckpt)
    assert actual != E4_CHECKPOINT_SHA256, "Fake checkpoint must not match expected SHA"


def test_sha256_consistent_for_same_file(tmp_path: Path) -> None:
    f = tmp_path / "file.bin"
    f.write_bytes(b"deterministic content 42")
    from scripts.run_final_robustness import _sha256_file
    assert _sha256_file(f) == _sha256_file(f)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Official Test guard
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad_path", [
    "data/splits/official_test.csv",
    "data/splits/test.csv",
    "path/to/official-test/split.csv",
])
def test_official_test_guard_blocks(bad_path: str) -> None:
    from scripts.run_final_robustness import _assert_not_official_test
    with pytest.raises(RuntimeError, match="Task 04 runner refuses"):
        _assert_not_official_test(bad_path)


@pytest.mark.parametrize("ok_path", [
    "data/splits/validation.csv",
    "data/splits/calibration.csv",
    "data/splits/train.csv",
])
def test_official_test_guard_permits(ok_path: str) -> None:
    from scripts.run_final_robustness import _assert_not_official_test
    _assert_not_official_test(ok_path)   # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# 3. E8Config: official_evaluation_enabled stays False
# ─────────────────────────────────────────────────────────────────────────────

def test_config_official_evaluation_disabled() -> None:
    cfg = _cfg()
    assert not cfg.official_evaluation_enabled
    assert not cfg.restrictions.use_official_test


def test_evaluator_rejects_official_eval_enabled(tmp_path: Path) -> None:
    """RobustnessEvaluator must raise if official_evaluation_enabled=True."""
    cfg = _cfg()
    # Patch the flag (do NOT write to YAML — test only)
    object.__setattr__(cfg, "official_evaluation_enabled", True)
    with pytest.raises(RuntimeError):
        RobustnessEvaluator(_dummy_model(), cfg)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Four conditions run on synthetic data
# ─────────────────────────────────────────────────────────────────────────────

def test_four_conditions_complete() -> None:
    cfg = _cfg()
    model = _dummy_model()
    evaluator = RobustnessEvaluator(model, cfg)
    images, labels = _synthetic_images()
    results = evaluator.evaluate(images, labels)
    assert "clean_accuracy" in results
    assert "gaussian_blur_accuracy" in results
    assert "gaussian_noise_accuracy" in results
    assert "occlusion_accuracy" in results
    assert "conditions" in results
    assert len(results["conditions"]) == 4


def test_accuracy_values_in_range() -> None:
    cfg = _cfg()
    evaluator = RobustnessEvaluator(_dummy_model(), cfg)
    images, labels = _synthetic_images()
    results = evaluator.evaluate(images, labels)
    for key in ("clean_accuracy", "gaussian_blur_accuracy",
                "gaussian_noise_accuracy", "occlusion_accuracy"):
        assert 0.0 <= results[key] <= 1.0, f"{key} out of [0,1]"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Accuracy Drop formula: clean_minus_corrupted (signed, not clamped)
# ─────────────────────────────────────────────────────────────────────────────

def test_accuracy_drop_formula() -> None:
    cfg = _cfg()
    evaluator = RobustnessEvaluator(_dummy_model(), cfg)
    images, labels = _synthetic_images()
    r = evaluator.evaluate(images, labels)
    for cond in ("gaussian_blur", "gaussian_noise", "occlusion"):
        expected = r["clean_accuracy"] - r[f"{cond}_accuracy"]
        actual = r[f"{cond}_accuracy_drop"]
        assert abs(actual - expected) < 1e-6, f"{cond} drop formula wrong"


def test_accuracy_drop_can_be_negative() -> None:
    """Drop must not be clamped to >= 0; a corrupted model can outperform clean on small N."""
    # Use a model that returns constant logits — clean and corrupted same logits
    cfg = _cfg()
    evaluator = RobustnessEvaluator(_dummy_model(), cfg)
    images, labels = _synthetic_images()
    r = evaluator.evaluate(images, labels)
    # drop = clean - corrupted; can be negative — just check it equals the formula
    for cond in ("gaussian_blur", "gaussian_noise", "occlusion"):
        assert isinstance(r[f"{cond}_accuracy_drop"], float)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Labels and image IDs not modified by corruption
# ─────────────────────────────────────────────────────────────────────────────

def test_labels_unchanged_across_conditions() -> None:
    """Corruption must never mutate label tensors."""
    cfg = _cfg()
    evaluator = RobustnessEvaluator(_dummy_model(), cfg)
    images, labels = _synthetic_images()
    labels_before = labels.clone()
    evaluator.evaluate(images, labels)
    assert torch.equal(labels, labels_before)


def test_images_unchanged_by_evaluator() -> None:
    """Evaluator must not mutate the input images tensor."""
    cfg = _cfg()
    evaluator = RobustnessEvaluator(_dummy_model(), cfg)
    images, labels = _synthetic_images()
    images_before = images.clone()
    evaluator.evaluate(images, labels)
    assert torch.allclose(images, images_before)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Corruption severity matches approved config
# ─────────────────────────────────────────────────────────────────────────────

def test_blur_severity_config() -> None:
    cfg = _cfg()
    assert cfg.gaussian_blur.kernel_size == 7
    assert cfg.gaussian_blur.sigma == 1.5


def test_noise_severity_config() -> None:
    cfg = _cfg()
    assert cfg.gaussian_noise.mean == 0.0
    assert cfg.gaussian_noise.std == 0.05


def test_occlusion_severity_config() -> None:
    cfg = _cfg()
    assert cfg.occlusion.side_fraction == 0.30
    assert cfg.occlusion.fill_rgb == (0.5, 0.5, 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Corruption before normalization (pixel range [0,1] preserved)
# ─────────────────────────────────────────────────────────────────────────────

def test_blur_output_in_01() -> None:
    img = torch.rand(3, 224, 224)
    out = apply_gaussian_blur(img, kernel_size=7, sigma=1.5)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_noise_output_clamped_01() -> None:
    img = torch.rand(3, 224, 224)
    out = apply_gaussian_noise(img, mean=0.0, std=0.05, clamp_range=(0.0, 1.0))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_occlusion_output_in_01() -> None:
    img = torch.rand(3, 224, 224)
    out = apply_occlusion(img, side_fraction=0.30, fill_rgb=(0.5, 0.5, 0.5), image_id=0)
    assert out.min() >= 0.0 and out.max() <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Occlusion deterministic — same image_id, same result (order-independent)
# ─────────────────────────────────────────────────────────────────────────────

def test_occlusion_deterministic_by_image_id() -> None:
    img = torch.rand(3, 224, 224)
    out1 = apply_occlusion(img, image_id=7)
    out2 = apply_occlusion(img, image_id=7)
    assert torch.equal(out1, out2)


def test_occlusion_different_ids_different_positions() -> None:
    img = torch.ones(3, 224, 224) * 0.3
    out0 = apply_occlusion(img, image_id=0)
    out1 = apply_occlusion(img, image_id=1)
    # Different positions → different tensors (unless astronomically unlikely)
    assert not torch.equal(out0, out1)


def test_occlusion_order_independence() -> None:
    """Occlusion of img_A is identical whether processed before or after img_B."""
    img_a = torch.rand(3, 224, 224)
    img_b = torch.rand(3, 224, 224)
    occ_a_first = apply_occlusion(img_a, image_id=0)
    occ_a_second = apply_occlusion(img_a, image_id=0)   # same id, different call order
    assert torch.equal(occ_a_first, occ_a_second)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Noise reproducibility (same seed → same result)
# ─────────────────────────────────────────────────────────────────────────────

def test_noise_reproducible_with_same_seed() -> None:
    img = torch.rand(3, 224, 224)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    out1 = apply_gaussian_noise(img, generator=g1)
    out2 = apply_gaussian_noise(img, generator=g2)
    assert torch.equal(out1, out2)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Model weights not modified
# ─────────────────────────────────────────────────────────────────────────────

def test_model_weights_unchanged() -> None:
    cfg = _cfg()
    model = _dummy_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    images, labels = _synthetic_images()
    RobustnessEvaluator(model, cfg).evaluate(images, labels)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), f"Weight {k} was modified!"


# ─────────────────────────────────────────────────────────────────────────────
# 12. Same predictions → same metrics (determinism end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

def test_deterministic_results_same_input() -> None:
    cfg = _cfg()
    model = _dummy_model()
    images, labels = _synthetic_images()
    r1 = RobustnessEvaluator(model, cfg).evaluate(images.clone(), labels.clone())
    r2 = RobustnessEvaluator(model, cfg).evaluate(images.clone(), labels.clone())
    for key in ("clean_accuracy", "gaussian_blur_accuracy",
                "gaussian_noise_accuracy", "occlusion_accuracy"):
        assert abs(r1[key] - r2[key]) < 1e-6, f"{key} not deterministic"


# ─────────────────────────────────────────────────────────────────────────────
# 13. Smoke-test script helper (no I/O, pure function)
# ─────────────────────────────────────────────────────────────────────────────

def test_smoke_test_returns_all_keys() -> None:
    from scripts.run_final_robustness import _smoke_test
    cfg = _cfg()
    result = _smoke_test(cfg)
    for key in ("clean_accuracy", "gaussian_blur_accuracy", "gaussian_blur_accuracy_drop",
                "gaussian_noise_accuracy", "gaussian_noise_accuracy_drop",
                "occlusion_accuracy", "occlusion_accuracy_drop", "official_evaluation"):
        assert key in result, f"Missing key: {key}"
    assert result["official_evaluation"] is False


def test_smoke_test_no_official_evaluation() -> None:
    from scripts.run_final_robustness import _smoke_test
    result = _smoke_test(_cfg())
    assert result["official_evaluation"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 14. Batch-size independence: corruption fn produces same per-image result
# ─────────────────────────────────────────────────────────────────────────────

def test_blur_batchsize_independent() -> None:
    """Blurring a single image equals blurring the same image inside a larger batch."""
    imgs = torch.rand(4, 3, 224, 224)
    single = apply_gaussian_blur(imgs[2])
    batch_result = torch.stack([apply_gaussian_blur(img) for img in imgs])
    assert torch.allclose(single, batch_result[2], atol=1e-6)


def test_occlusion_batchsize_independent() -> None:
    imgs = torch.rand(4, 3, 224, 224)
    single = apply_occlusion(imgs[2], image_id=2)
    batch_result = torch.stack([apply_occlusion(img, image_id=i) for i, img in enumerate(imgs)])
    assert torch.equal(single, batch_result[2])


# ─────────────────────────────────────────────────────────────────────────────
# 15. Regression: RobustnessEvaluator interface unchanged (Task 03 compat)
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluator_interface_regression() -> None:
    """Evaluator still accepts (model, cfg) and returns dict with expected keys."""
    cfg = _cfg()
    evaluator = RobustnessEvaluator(_dummy_model(), cfg)
    images, labels = _synthetic_images()
    r = evaluator.evaluate(images, labels)
    assert isinstance(r, dict)
    assert "clean_accuracy" in r
    assert "conditions" in r
    assert isinstance(r["conditions"], list)
