"""Unit tests for E8 Robustness Tooling.

Tests use synthetic tensors/models only — no real checkpoint, no Official Test.
Run: python -m pytest tests/test_robustness.py -v
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.robustness.config import load_e8_config
from src.robustness.corruptions import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_occlusion,
)
from src.robustness.robustness_eval import RobustnessEvaluator


# ── helpers ───────────────────────────────────────────────────────────────────

def _img(h: int = 224, w: int = 224, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(); g.manual_seed(seed)
    return torch.rand(3, h, w, generator=g)


def _batch(n: int = 4, num_classes: int = 10, seed: int = 42):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.rand(n, 3, 224, 224, generator=g), torch.randint(0, num_classes, (n,), generator=g)


def _dummy_model(num_classes: int = 10) -> nn.Module:
    m = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, num_classes))
    return m.eval()


@pytest.fixture()
def cfg():
    return load_e8_config("configs/experiments/e8_robustness.yaml")


# ── load_e8_config ────────────────────────────────────────────────────────────

def test_config_loads():
    cfg = load_e8_config("configs/experiments/e8_robustness.yaml")
    assert cfg.experiment_id == "E8"
    assert cfg.official_evaluation_enabled is False


def test_config_approved_blur():
    cfg = load_e8_config("configs/experiments/e8_robustness.yaml")
    assert cfg.gaussian_blur.kernel_size == 7
    assert cfg.gaussian_blur.sigma == pytest.approx(1.5)


def test_config_approved_noise():
    cfg = load_e8_config("configs/experiments/e8_robustness.yaml")
    assert cfg.gaussian_noise.mean == pytest.approx(0.0)
    assert cfg.gaussian_noise.std == pytest.approx(0.05)
    assert cfg.gaussian_noise.clamp_range == (0.0, 1.0)


def test_config_approved_occlusion():
    cfg = load_e8_config("configs/experiments/e8_robustness.yaml")
    assert cfg.occlusion.side_fraction == pytest.approx(0.30)
    assert cfg.occlusion.fill_rgb == (0.5, 0.5, 0.5)
    assert cfg.occlusion.position_policy == "deterministic_per_image"


def test_config_restrictions():
    cfg = load_e8_config("configs/experiments/e8_robustness.yaml")
    assert cfg.restrictions.use_official_test is False
    assert cfg.restrictions.train_new_model is False


# ── Gaussian Blur ─────────────────────────────────────────────────────────────

def test_blur_shape_unchanged():
    assert apply_gaussian_blur(_img()).shape == (3, 224, 224)


def test_blur_pixel_range():
    out = apply_gaussian_blur(_img())
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0


def test_blur_deterministic():
    img = _img(seed=7)
    assert torch.allclose(apply_gaussian_blur(img), apply_gaussian_blur(img))


def test_blur_approved_kernel_sigma():
    apply_gaussian_blur(_img(), kernel_size=7, sigma=1.5)


def test_blur_changes_image():
    img = _img()
    assert not torch.allclose(img, apply_gaussian_blur(img))


def test_blur_bad_shape():
    with pytest.raises(ValueError, match="3-D"):
        apply_gaussian_blur(torch.rand(224, 224))


def test_blur_even_kernel_raises():
    with pytest.raises(ValueError):
        apply_gaussian_blur(_img(), kernel_size=6)


def test_blur_bad_sigma_raises():
    with pytest.raises(ValueError):
        apply_gaussian_blur(_img(), sigma=-1.0)


# ── Gaussian Noise ────────────────────────────────────────────────────────────

def test_noise_shape_unchanged():
    assert apply_gaussian_noise(_img()).shape == (3, 224, 224)


def test_noise_clamp_range():
    out = apply_gaussian_noise(_img(), std=10.0)
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0


def test_noise_approved_params():
    out = apply_gaussian_noise(_img(), mean=0.0, std=0.05)
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0


def test_noise_deterministic_with_generator():
    img = _img()
    g1 = torch.Generator(); g1.manual_seed(42)
    g2 = torch.Generator(); g2.manual_seed(42)
    assert torch.allclose(
        apply_gaussian_noise(img, generator=g1),
        apply_gaussian_noise(img, generator=g2),
    )


def test_noise_changes_image():
    img = _img()
    g = torch.Generator(); g.manual_seed(0)
    assert not torch.allclose(img, apply_gaussian_noise(img, std=0.05, generator=g))


def test_noise_bad_shape():
    with pytest.raises(ValueError, match="3-D"):
        apply_gaussian_noise(torch.rand(224, 224))


# ── Occlusion ─────────────────────────────────────────────────────────────────

def test_occlusion_shape_unchanged():
    assert apply_occlusion(_img(), image_id=0).shape == (3, 224, 224)


def test_occlusion_fill_color():
    img = torch.ones(3, 224, 224)
    out = apply_occlusion(img, side_fraction=0.30, fill_rgb=(0.0, 0.0, 0.0), image_id=1)
    black = (out[0] == 0.0) & (out[1] == 0.0) & (out[2] == 0.0)
    assert black.any()


def test_occlusion_approved_params():
    out = apply_occlusion(_img(), side_fraction=0.30, fill_rgb=(0.5, 0.5, 0.5), image_id=0)
    assert out.shape == (3, 224, 224)


def test_occlusion_deterministic_per_image_id():
    img = _img(seed=5)
    assert torch.allclose(apply_occlusion(img, image_id=7), apply_occlusion(img, image_id=7))


def test_occlusion_order_independent():
    img = _img(seed=3)
    out_a = apply_occlusion(img, image_id=2)
    apply_occlusion(_img(seed=9), image_id=99)
    out_b = apply_occlusion(img, image_id=2)
    assert torch.allclose(out_a, out_b)


def test_occlusion_different_ids_differ():
    img = torch.ones(3, 224, 224)
    assert not torch.allclose(apply_occlusion(img, image_id=0), apply_occlusion(img, image_id=999))


def test_occlusion_bad_shape():
    with pytest.raises(ValueError, match="3-D"):
        apply_occlusion(torch.rand(224, 224))


def test_occlusion_bad_fraction():
    with pytest.raises(ValueError):
        apply_occlusion(_img(), side_fraction=1.5)


# ── Corruption before normalization ──────────────────────────────────────────

def test_corruption_before_normalization_pixel_range():
    img = _img()
    for fn in [
        apply_gaussian_blur,
        apply_gaussian_noise,
        lambda x: apply_occlusion(x, image_id=0),
    ]:
        out = fn(img)
        assert out.min().item() >= -1e-6 and out.max().item() <= 1.0 + 1e-6


# ── Label preservation ────────────────────────────────────────────────────────

def test_labels_unchanged_after_corruption():
    _, labels = _batch()
    saved = labels.clone()
    img = _img()
    apply_gaussian_blur(img); apply_gaussian_noise(img); apply_occlusion(img, image_id=0)
    assert torch.equal(labels, saved)


# ── RobustnessEvaluator ───────────────────────────────────────────────────────

def test_evaluator_no_weight_update(cfg):
    model = _dummy_model()
    w_before = [p.clone() for p in model.parameters()]
    RobustnessEvaluator(model, cfg).evaluate(*_batch())
    for before, after in zip(w_before, model.parameters()):
        assert torch.allclose(before, after), "Weights changed!"


def test_evaluator_returns_clean_accuracy(cfg):
    results = RobustnessEvaluator(_dummy_model(), cfg).evaluate(*_batch())
    assert "clean_accuracy" in results
    assert 0.0 <= results["clean_accuracy"] <= 1.0


def test_evaluator_accuracy_drop_formula(cfg):
    results = RobustnessEvaluator(_dummy_model(), cfg).evaluate(*_batch())
    clean = results["clean_accuracy"]
    for name in ("gaussian_blur", "gaussian_noise", "occlusion"):
        if f"{name}_accuracy" in results:
            assert results[f"{name}_accuracy_drop"] == pytest.approx(
                clean - results[f"{name}_accuracy"], abs=1e-6
            )


def test_evaluator_accuracy_in_01(cfg):
    results = RobustnessEvaluator(_dummy_model(), cfg).evaluate(*_batch())
    for key, val in results.items():
        if "accuracy" in key and "drop" not in key:
            assert 0.0 <= val <= 1.0, f"{key}={val}"


def test_evaluator_conditions_list(cfg):
    results = RobustnessEvaluator(_dummy_model(), cfg).evaluate(*_batch())
    for cond in ("clean", "gaussian_blur", "gaussian_noise", "occlusion"):
        assert cond in results["conditions"]


def test_evaluator_no_official_test(cfg):
    assert cfg.official_evaluation_enabled is False
    assert cfg.restrictions.use_official_test is False


def test_evaluator_accuracy_drop_perfect_model(cfg):
    class PerfectModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._dummy = nn.Linear(1, 1)  # gives parameters() something to return

        def forward(self, x):
            out = torch.zeros(x.shape[0], 10)
            out[:, 0] = 100.0
            return out

    labels = torch.zeros(4, dtype=torch.long)
    results = RobustnessEvaluator(PerfectModel(), cfg).evaluate(torch.rand(4, 3, 224, 224), labels)
    assert results["clean_accuracy"] == pytest.approx(1.0)
    for name in ("gaussian_blur", "gaussian_noise", "occlusion"):
        if f"{name}_accuracy_drop" in results:
            assert results[f"{name}_accuracy_drop"] >= -1e-6


def test_evaluator_deterministic(cfg):
    model = _dummy_model()
    images, labels = _batch(seed=99)
    r1 = RobustnessEvaluator(model, cfg).evaluate(images, labels)
    r2 = RobustnessEvaluator(model, cfg).evaluate(images, labels)
    for key in r1:
        if isinstance(r1[key], float):
            assert r1[key] == pytest.approx(r2[key], abs=1e-6), f"Non-det: {key}"


# ── Runner ────────────────────────────────────────────────────────────────────

def test_runner_rejects_official_test():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "scripts/run_robustness.py", "--official-test"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "official" in (result.stderr + result.stdout).lower()


def test_runner_smoke_test_passes():
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "scripts/run_robustness.py", "--smoke-test"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Tooling verified" in result.stdout


