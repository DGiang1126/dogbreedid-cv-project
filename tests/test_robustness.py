"""Unit tests for E8 Robustness Tooling.

Tests use synthetic tensors/models only — no real checkpoint, no Official Test.
Run: python -m pytest tests/test_robustness.py -v
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
import torch
import torch.nn as nn

from src.robustness.config import load_e8_config
from src.robustness.corruptions import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_occlusion,
    per_image_seed,
)
from src.robustness.robustness_eval import RobustnessEvaluator


# ── helpers ───────────────────────────────────────────────────────────────────

def _img(h: int = 32, w: int = 32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(); g.manual_seed(seed)
    return torch.rand(3, h, w, generator=g)


def _batch(n: int = 4, num_classes: int = 10, seed: int = 42):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.rand(n, 3, 32, 32, generator=g), torch.randint(0, num_classes, (n,), generator=g)


def _ids(n):
    return [f"dog/image_{i}.jpg" for i in range(n)]


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
    assert apply_gaussian_blur(_img()).shape == (3, 32, 32)


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
        apply_gaussian_blur(torch.rand(32, 32))


def test_blur_even_kernel_raises():
    with pytest.raises(ValueError):
        apply_gaussian_blur(_img(), kernel_size=6)


def test_blur_bad_sigma_raises():
    with pytest.raises(ValueError):
        apply_gaussian_blur(_img(), sigma=-1.0)


# ── Gaussian Noise ────────────────────────────────────────────────────────────

def test_noise_shape_unchanged():
    assert apply_gaussian_noise(_img(), global_seed=42, image_id="dog/a.jpg").shape == (3, 32, 32)


def test_noise_clamp_range():
    out = apply_gaussian_noise(_img(), std=10.0, global_seed=42, image_id="dog/a.jpg")
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0


def test_noise_approved_params():
    out = apply_gaussian_noise(_img(), mean=0.0, std=0.05, global_seed=42, image_id="dog/a.jpg")
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0


def test_noise_deterministic_per_image():
    img = _img()
    assert torch.allclose(
        apply_gaussian_noise(img, global_seed=42, image_id="dog/a.jpg"),
        apply_gaussian_noise(img, global_seed=42, image_id="dog/a.jpg"),
    )


def test_noise_changes_image():
    img = _img()
    assert not torch.allclose(img, apply_gaussian_noise(img, std=0.05, global_seed=42, image_id="dog/a.jpg"))


def test_noise_bad_shape():
    with pytest.raises(ValueError, match="3-D"):
        apply_gaussian_noise(torch.rand(32, 32), global_seed=42, image_id="dog/a.jpg")


# ── Occlusion ─────────────────────────────────────────────────────────────────

def test_occlusion_shape_unchanged():
    assert apply_occlusion(_img(), global_seed=42, image_id="dog/0.jpg").shape == (3, 32, 32)


def test_occlusion_fill_color():
    img = torch.ones(3, 32, 32)
    out = apply_occlusion(img, side_fraction=0.30, fill_rgb=(0.0, 0.0, 0.0), global_seed=42, image_id="dog/1.jpg")
    black = (out[0] == 0.0) & (out[1] == 0.0) & (out[2] == 0.0)
    assert black.any()


def test_occlusion_approved_params():
    out = apply_occlusion(_img(), side_fraction=0.30, fill_rgb=(0.5, 0.5, 0.5), global_seed=42, image_id="dog/0.jpg")
    assert out.shape == (3, 32, 32)


def test_occlusion_deterministic_per_image_id():
    img = _img(seed=5)
    assert torch.allclose(apply_occlusion(img, global_seed=42, image_id="dog/7.jpg"), apply_occlusion(img, global_seed=42, image_id="dog/7.jpg"))


def test_occlusion_order_independent():
    img = _img(seed=3)
    out_a = apply_occlusion(img, global_seed=42, image_id="dog/2.jpg")
    apply_occlusion(_img(seed=9), global_seed=42, image_id="dog/99.jpg")
    out_b = apply_occlusion(img, global_seed=42, image_id="dog/2.jpg")
    assert torch.allclose(out_a, out_b)


def test_occlusion_different_ids_differ():
    img = torch.ones(3, 32, 32)
    assert not torch.allclose(apply_occlusion(img, global_seed=42, image_id="dog/0.jpg"), apply_occlusion(img, global_seed=42, image_id="dog/999.jpg"))


def test_occlusion_bad_shape():
    with pytest.raises(ValueError, match="3-D"):
        apply_occlusion(torch.rand(32, 32), global_seed=42, image_id="dog/a.jpg")


def test_occlusion_bad_fraction():
    with pytest.raises(ValueError):
        apply_occlusion(_img(), side_fraction=1.5, global_seed=42, image_id="dog/a.jpg")


# ── Corruption before normalization ──────────────────────────────────────────

def test_corruption_before_normalization_pixel_range():
    img = _img()
    for fn in [
        apply_gaussian_blur,
        lambda x: apply_gaussian_noise(x, global_seed=42, image_id="dog/a.jpg"),
        lambda x: apply_occlusion(x, global_seed=42, image_id="dog/0.jpg"),
    ]:
        out = fn(img)
        assert out.min().item() >= -1e-6 and out.max().item() <= 1.0 + 1e-6


# ── Label preservation ────────────────────────────────────────────────────────

def test_labels_unchanged_after_corruption():
    _, labels = _batch()
    saved = labels.clone()
    img = _img()
    apply_gaussian_blur(img); apply_gaussian_noise(img, global_seed=42, image_id="dog/a.jpg"); apply_occlusion(img, global_seed=42, image_id="dog/0.jpg")
    assert torch.equal(labels, saved)


# ── RobustnessEvaluator ───────────────────────────────────────────────────────

def test_evaluator_no_weight_update(cfg):
    model = _dummy_model()
    w_before = [p.clone() for p in model.parameters()]
    RobustnessEvaluator(model, cfg).evaluate(*_batch(), image_ids=_ids(4))
    for before, after in zip(w_before, model.parameters()):
        assert torch.allclose(before, after), "Weights changed!"


def test_evaluator_returns_clean_accuracy(cfg):
    results = RobustnessEvaluator(_dummy_model(), cfg).evaluate(*_batch(), image_ids=_ids(4))
    assert "clean_accuracy" in results
    assert 0.0 <= results["clean_accuracy"] <= 1.0


def test_evaluator_accuracy_drop_formula(cfg):
    results = RobustnessEvaluator(_dummy_model(), cfg).evaluate(*_batch(), image_ids=_ids(4))
    clean = results["clean_accuracy"]
    for name in ("gaussian_blur", "gaussian_noise", "occlusion"):
        if f"{name}_accuracy" in results:
            assert results[f"{name}_accuracy_drop"] == pytest.approx(
                clean - results[f"{name}_accuracy"], abs=1e-6
            )


def test_evaluator_accuracy_in_01(cfg):
    results = RobustnessEvaluator(_dummy_model(), cfg).evaluate(*_batch(), image_ids=_ids(4))
    for key, val in results.items():
        if "accuracy" in key and "drop" not in key:
            assert 0.0 <= val <= 1.0, f"{key}={val}"


def test_evaluator_conditions_list(cfg):
    results = RobustnessEvaluator(_dummy_model(), cfg).evaluate(*_batch(), image_ids=_ids(4))
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
    results = RobustnessEvaluator(PerfectModel(), cfg).evaluate(torch.rand(4, 3, 32, 32), labels, _ids(4))
    assert results["clean_accuracy"] == pytest.approx(1.0)
    for name in ("gaussian_blur", "gaussian_noise", "occlusion"):
        if f"{name}_accuracy_drop" in results:
            assert results[f"{name}_accuracy_drop"] >= -1e-6


def test_evaluator_deterministic(cfg):
    model = _dummy_model()
    images, labels = _batch(seed=99)
    r1 = RobustnessEvaluator(model, cfg).evaluate(images, labels, _ids(len(images)))
    r2 = RobustnessEvaluator(model, cfg).evaluate(images, labels, _ids(len(images)))
    for key in r1:
        if isinstance(r1[key], float):
            assert r1[key] == pytest.approx(r2[key], abs=1e-6), f"Non-det: {key}"


# ── Runner ────────────────────────────────────────────────────────────────────

def test_runner_rejects_official_test():
    result = subprocess.run(
        [sys.executable, "scripts/run_robustness.py", "--official-test"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "official" in (result.stderr + result.stdout).lower()


def test_runner_smoke_test_passes():
    result = subprocess.run(
        [sys.executable, "scripts/run_robustness.py", "--smoke-test"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Tooling verified" in result.stdout



# Validation uses temporary copies; the approved YAML is never edited.
CONFIG_PATH = Path("configs/experiments/e8_robustness.yaml")


def _write_config(tmp_path, raw):
    path = tmp_path / "e8.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _replace_field(raw, name, value):
    parts = name.split(".")
    parent = raw
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = value


@pytest.mark.parametrize("field,value", [
    ("experiment.id", "E7"),
    ("execution.status", "final_evaluation"),
    ("execution.official_evaluation_enabled", True),
    ("execution.official_evaluation_enabled", "false"),
    ("restrictions.use_official_test", True),
    ("restrictions.train_new_model", True),
    ("restrictions.modify_checkpoint", True),
    ("restrictions.use_official_test", "false"),
    ("restrictions.train_new_model", 0),
    ("restrictions.modify_checkpoint", None),
    ("corruptions.gaussian_blur.kernel_size", 0),
    ("corruptions.gaussian_blur.kernel_size", -7),
    ("corruptions.gaussian_blur.kernel_size", 6),
    ("corruptions.gaussian_blur.kernel_size", 7.5),
    ("corruptions.gaussian_blur.kernel_size", True),
    ("corruptions.gaussian_blur.sigma", 0),
    ("corruptions.gaussian_blur.sigma", -1),
    ("corruptions.gaussian_blur.sigma", float("nan")),
    ("corruptions.gaussian_blur.sigma", "1.5"),
    ("corruptions.gaussian_blur.enabled", "true"),
    ("corruptions.gaussian_noise.std", -0.05),
    ("corruptions.gaussian_noise.std", float("inf")),
    ("corruptions.gaussian_noise.std", False),
    ("corruptions.gaussian_noise.mean", "0"),
    ("corruptions.gaussian_noise.mean", float("nan")),
    ("corruptions.gaussian_noise.clamp_range", [1, 0]),
    ("corruptions.gaussian_noise.clamp_range", [0, 0]),
    ("corruptions.gaussian_noise.clamp_range", [-1, 1]),
    ("corruptions.gaussian_noise.clamp_range", [0, 2]),
    ("corruptions.gaussian_noise.clamp_range", [0]),
    ("corruptions.gaussian_noise.clamp_range", "0,1"),
    ("corruptions.gaussian_noise.clamp_range", [0, float("inf")]),
    ("corruptions.gaussian_noise.enabled", 1),
    ("corruptions.occlusion.side_fraction", 0),
    ("corruptions.occlusion.side_fraction", 1),
    ("corruptions.occlusion.side_fraction", float("nan")),
    ("corruptions.occlusion.fill_rgb", [0.5, 0.5]),
    ("corruptions.occlusion.fill_rgb", [0.5, 0.5, 0.5, 0.5]),
    ("corruptions.occlusion.fill_rgb", [-0.1, 0.5, 0.5]),
    ("corruptions.occlusion.fill_rgb", [0.5, 1.1, 0.5]),
    ("corruptions.occlusion.fill_rgb", [0.5, True, 0.5]),
    ("corruptions.occlusion.fill_rgb", [0.5, float("nan"), 0.5]),
    ("corruptions.occlusion.shape", "circle"),
    ("corruptions.occlusion.position_policy", "random"),
    ("corruptions.occlusion.enabled", None),
    ("reproducibility.seed", -1),
    ("reproducibility.seed", 2**64),
    ("reproducibility.seed", 42.5),
    ("reproducibility.seed", "42"),
    ("reproducibility.seed", True),
    ("reproducibility.seed", None),
    ("reproducibility.deterministic", False),
    ("reproducibility.deterministic", "true"),
    ("reproducibility.per_image_seed", "batch_index"),
    ("data.corruption_stage", "after_normalization"),
    ("data.pixel_range", [-1, 1]),
    ("data.pixel_range", [0.1, 1]),
    ("data.preserve_labels", False),
    ("data.input_size", 0),
    ("data.input_size", True),
    ("model.selection_config", ""),
    ("evaluation.conditions", "clean"),
    ("evaluation.conditions", ["clean", "unknown"]),
    ("evaluation.conditions", ["clean", "clean", "gaussian_noise", "occlusion"]),
    ("evaluation.conditions", ["clean", [], "gaussian_noise", "occlusion"]),
    ("evaluation.primary_metric", "unknown"),
    ("evaluation.report_accuracy_drop", "true"),
    ("evaluation.accuracy_drop_formula", "corrupted_minus_clean"),
])
def test_config_rejects_invalid_fields(tmp_path, field, value):
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    _replace_field(raw, field, value)
    with pytest.raises(ValueError, match=field):
        load_e8_config(_write_config(tmp_path, raw))


_REQUIRED_FIELDS = [
    "experiment.id", "execution.status", "execution.official_evaluation_enabled",
    "model.selection_config", "data.input_size", "data.corruption_stage",
    "data.pixel_range", "data.preserve_labels",
    "corruptions.gaussian_blur.enabled", "corruptions.gaussian_blur.kernel_size",
    "corruptions.gaussian_blur.sigma", "corruptions.gaussian_noise.enabled",
    "corruptions.gaussian_noise.mean", "corruptions.gaussian_noise.std",
    "corruptions.gaussian_noise.clamp_range", "corruptions.occlusion.enabled",
    "corruptions.occlusion.shape", "corruptions.occlusion.side_fraction",
    "corruptions.occlusion.fill_rgb", "corruptions.occlusion.position_policy",
    "reproducibility.seed", "reproducibility.per_image_seed", "reproducibility.deterministic",
    "evaluation.conditions", "evaluation.primary_metric", "evaluation.report_accuracy_drop",
    "evaluation.accuracy_drop_formula", "restrictions.train_new_model",
    "restrictions.modify_checkpoint", "restrictions.use_official_test",
]


@pytest.mark.parametrize("field", _REQUIRED_FIELDS)
def test_config_rejects_missing_fields(tmp_path, field):
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    parent = raw
    parts = field.split(".")
    for part in parts[:-1]:
        parent = parent[part]
    del parent[parts[-1]]
    with pytest.raises(ValueError, match=field):
        load_e8_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize("section", [
    "experiment", "execution", "model", "data", "corruptions",
    "reproducibility", "evaluation", "restrictions", "corruptions.gaussian_noise",
])
def test_config_rejects_non_mapping_sections(tmp_path, section):
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    _replace_field(raw, section, [])
    with pytest.raises(ValueError, match="mapping"):
        load_e8_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize("content", ["", "null", "[]", "false", "E8", "{}", "experiment: ["])
def test_config_rejects_empty_or_malformed_yaml(tmp_path, content):
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_e8_config(path)


def test_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_e8_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize("seed", [0, 123, 2**64 - 1])
def test_config_valid_seed_boundaries(tmp_path, seed):
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["reproducibility"]["seed"] = seed
    assert load_e8_config(_write_config(tmp_path, raw)).reproducibility.seed == seed


def test_seed_matches_sha256_and_separates_components():
    payload = json.dumps([123, "dogs/a.jpg", "gaussian_noise"], separators=(",", ":"))
    expected = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
    assert per_image_seed(123, "dogs/a.jpg", "gaussian_noise") == expected
    assert expected != per_image_seed(123, "dogs/a.jpg", "occlusion")
    assert per_image_seed(1, "23", "noise") != per_image_seed(12, "3", "noise")


@pytest.mark.parametrize("fn", [apply_gaussian_noise, apply_occlusion])
def test_random_corruptions_repeat_and_change_with_seed_and_id(fn):
    img = torch.full((3, 64, 64), 0.8)
    first = fn(img, global_seed=123, image_id="dogs/a.jpg")
    torch.rand(20)  # Global RNG activity must not affect per-image streams.
    assert torch.equal(first, fn(img, global_seed=123, image_id="dogs/a.jpg"))
    assert not torch.equal(first, fn(img, global_seed=2026, image_id="dogs/a.jpg"))
    assert not torch.equal(first, fn(img, global_seed=123, image_id="dogs/b.jpg"))


@pytest.mark.parametrize("fn", [apply_gaussian_noise, apply_occlusion])
@pytest.mark.parametrize("image_id", [None, "", "  ", 0, True])
def test_random_corruptions_reject_bad_ids(fn, image_id):
    with pytest.raises(ValueError, match="image_id"):
        fn(_img(), global_seed=123, image_id=image_id)


@pytest.mark.parametrize("fn", [apply_gaussian_noise, apply_occlusion])
def test_random_corruptions_require_explicit_seed_and_id(fn):
    with pytest.raises(TypeError):
        fn(_img())


@pytest.mark.parametrize("seed", [-1, True, "42", 2**64])
def test_per_image_seed_rejects_invalid_seed(seed):
    with pytest.raises(ValueError, match="global_seed"):
        per_image_seed(seed, "dog/a.jpg", "occlusion")


def test_noise_matches_seeded_distribution_and_preserves_rng_and_input():
    image = torch.full((3, 256, 256), 0.5)
    before = image.clone()
    state = torch.random.get_rng_state().clone()
    actual = apply_gaussian_noise(image, global_seed=123, image_id="dog/a.jpg")
    generator = torch.Generator().manual_seed(per_image_seed(123, "dog/a.jpg", "gaussian_noise"))
    noise = torch.empty_like(image).normal_(mean=0, std=0.05, generator=generator)
    assert torch.equal(actual, (image + noise).clamp(0, 1))
    assert abs((actual - image).mean().item()) < 0.001
    assert (actual - image).std().item() == pytest.approx(0.05, abs=0.001)
    assert torch.equal(image, before)
    assert torch.equal(state, torch.random.get_rng_state())


def test_noise_zero_std_and_custom_clamp():
    image = torch.full((3, 8, 8), 0.5)
    actual = apply_gaussian_noise(image, mean=0.1, std=0, clamp_range=(0.2, 0.55),
                                  global_seed=123, image_id="dog/a.jpg")
    assert torch.equal(actual, torch.full_like(image, 0.55))


def test_blur_approved_impulse_response():
    image = torch.zeros(3, 21, 21)
    image[:, 10, 10] = 1
    before = image.clone()
    actual = apply_gaussian_blur(image, kernel_size=7, sigma=1.5)
    coords = torch.arange(-3, 4, dtype=torch.float32)
    expected = torch.exp(-(coords[:, None]**2 + coords[None, :]**2) / (2 * 1.5**2))
    expected /= expected.sum()
    assert torch.allclose(actual[:, 7:14, 7:14], expected.expand(3, -1, -1), atol=1e-7)
    assert torch.allclose(actual.sum(dim=(1, 2)), torch.ones(3))
    assert torch.equal(image, before)


@pytest.mark.parametrize("size", [(64, 80), (224, 224)])
def test_occlusion_exact_square_area_fill_and_input_preservation(size):
    image = torch.ones(3, *size)
    state = torch.random.get_rng_state().clone()
    fill = (0.2, 0.4, 0.6)
    actual = apply_occlusion(image, fill_rgb=fill, global_seed=123, image_id="dog/a.jpg")
    mask = (actual != image).any(dim=0)
    side = int(min(size) * 0.30)
    y, x = mask.nonzero(as_tuple=True)
    assert mask.sum().item() == side**2
    assert y.max() - y.min() + 1 == side
    assert x.max() - x.min() + 1 == side
    assert torch.equal(actual[:, mask], torch.tensor(fill)[:, None].expand(3, side**2))
    assert torch.equal(actual[:, ~mask], image[:, ~mask])
    assert torch.equal(image, torch.ones_like(image))
    assert torch.equal(state, torch.random.get_rng_state())


class _RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, images):
        assert not self.training
        assert torch.is_inference_mode_enabled()
        self.inputs.append(images.clone())
        return torch.zeros(len(images), 10, device=images.device)


def _record_evaluation(cfg, images, labels, image_ids, batch_size=None):
    model = _RecordingModel()
    result = RobustnessEvaluator(model, cfg).evaluate(
        images, labels, image_ids, batch_size=batch_size,
    )
    chunks = (len(images) + (batch_size or len(images)) - 1) // (batch_size or len(images))
    tensors = {
        name: torch.cat(model.inputs[i * chunks:(i + 1) * chunks])
        for i, name in enumerate(cfg.conditions)
    }
    return result, tensors


@pytest.mark.parametrize("name", ["gaussian_noise", "occlusion"])
def test_evaluator_per_image_corruption_independent_of_order(cfg, name):
    images, labels = _batch(n=5)
    ids = _ids(5)
    permutation = torch.tensor([4, 2, 0, 3, 1])
    result, original = _record_evaluation(cfg, images, labels, ids)
    reordered_result, reordered = _record_evaluation(
        cfg, images[permutation], labels[permutation], [ids[i] for i in permutation],
    )
    assert torch.equal(original[name][permutation], reordered[name])
    assert result == reordered_result


@pytest.mark.parametrize("name", ["gaussian_noise", "occlusion"])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
def test_evaluator_per_image_corruption_independent_of_batch_size(cfg, name, batch_size):
    images, labels = _batch(n=5)
    ids = _ids(5)
    result, original = _record_evaluation(cfg, images, labels, ids)
    chunked_result, chunked = _record_evaluation(cfg, images, labels, ids, batch_size)
    assert torch.equal(original[name], chunked[name])
    assert result == chunked_result
    # Also simulate an external DataLoader splitting evaluation calls.
    pieces = []
    for start in range(0, len(images), batch_size):
        _, part = _record_evaluation(cfg, images[start:start + batch_size],
                                     labels[start:start + batch_size], ids[start:start + batch_size])
        pieces.append(part[name])
    assert torch.equal(original[name], torch.cat(pieces))


def test_evaluator_uses_config_seed_before_imagenet_normalization(cfg):
    cfg.reproducibility.seed = 2026
    # Change an in-memory test config to catch hard-coded evaluator parameters.
    cfg.gaussian_blur.kernel_size, cfg.gaussian_blur.sigma = 3, 0.7
    cfg.gaussian_noise.mean, cfg.gaussian_noise.std = 0.01, 0.1
    cfg.gaussian_noise.clamp_range = (0.1, 0.9)
    cfg.occlusion.side_fraction, cfg.occlusion.fill_rgb = 0.2, (0.1, 0.2, 0.3)
    images, labels = _batch(n=2)
    before, labels_before = images.clone(), labels.clone()
    _, captured = _record_evaluation(cfg, images, labels, _ids(2))
    expected = {
        "clean": images,
        "gaussian_blur": torch.stack([apply_gaussian_blur(img, 3, 0.7) for img in images]),
        "gaussian_noise": torch.stack([
            apply_gaussian_noise(img, 0.01, 0.1, (0.1, 0.9), global_seed=2026, image_id=image_id)
            for img, image_id in zip(images, _ids(2))
        ]),
        "occlusion": torch.stack([
            apply_occlusion(img, 0.2, (0.1, 0.2, 0.3), global_seed=2026, image_id=image_id)
            for img, image_id in zip(images, _ids(2))
        ]),
    }
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    for name, tensor in expected.items():
        assert torch.equal(captured[name], (tensor - mean) / std)
    assert torch.equal(images, before)
    assert torch.equal(labels, labels_before)


@pytest.mark.parametrize("image_ids", [None, [], ["a"], ["a", ""], ["a", None],
                                       ["a", 0], ["a", "a"], "ab"])
def test_evaluator_rejects_missing_or_invalid_ids(cfg, image_ids):
    with pytest.raises(ValueError, match="image_id"):
        RobustnessEvaluator(_RecordingModel(), cfg).evaluate(*_batch(n=2), image_ids)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_evaluator_rejects_invalid_batch_size(cfg, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        RobustnessEvaluator(_RecordingModel(), cfg).evaluate(*_batch(n=2), _ids(2), batch_size=batch_size)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_evaluator_rejects_invalid_pixels(cfg, value):
    images, labels = _batch(n=2)
    images[0, 0, 0, 0] = value
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        RobustnessEvaluator(_RecordingModel(), cfg).evaluate(images, labels, _ids(2))


def test_evaluator_eval_inference_and_all_model_state_unchanged(cfg):
    model = nn.Sequential(nn.BatchNorm2d(3), nn.Dropout(0.5), nn.AdaptiveAvgPool2d(1),
                          nn.Flatten(), nn.Linear(3, 10)).train()
    before = copy.deepcopy(model.state_dict())
    images, labels = _batch(n=3)
    RobustnessEvaluator(model, cfg).evaluate(images, labels, _ids(3), batch_size=2)
    assert not model.training
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name
    assert all(p.grad is None for p in model.parameters())


def test_accuracy_drop_hand_computed_including_negative_drop(cfg):
    class Predictions(nn.Module):
        def __init__(self):
            super().__init__()
            self.predictions = iter([[0, 1, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]])

        def forward(self, images):
            return torch.nn.functional.one_hot(torch.tensor(next(self.predictions)), 2).float()

    images, _ = _batch()
    labels = torch.tensor([0, 1, 1, 0])
    result = RobustnessEvaluator(Predictions(), cfg).evaluate(images, labels, _ids(4))
    assert result["clean_accuracy"] == 0.75
    assert result["gaussian_blur_accuracy"] == 0.5
    assert result["gaussian_blur_accuracy_drop"] == 0.25
    assert result["gaussian_noise_accuracy"] == 1.0
    assert result["gaussian_noise_accuracy_drop"] == -0.25
    assert result["occlusion_accuracy"] == 0.0
    assert result["occlusion_accuracy_drop"] == 0.75


def test_runner_blocks_official_before_reading_config():
    result = subprocess.run(
        [sys.executable, "scripts/run_robustness.py", "--smoke-test", "--official-test",
         "--config", "missing.yaml"], capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "Official Test is disabled" in result.stderr
    assert "Traceback" not in result.stderr


def test_runner_reads_seed_dimensions_and_classes_from_yaml(tmp_path, monkeypatch, capsys):
    from scripts import run_robustness as runner

    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["data"]["input_size"] = 16
    raw["reproducibility"]["seed"] = 2026
    selection = tmp_path / "selection.yaml"
    selection.write_text(yaml.safe_dump({"model": {"num_classes": 7}}), encoding="utf-8")
    raw["model"]["selection_config"] = str(selection)
    config = _write_config(tmp_path, raw)
    observed = []
    real_evaluate = runner.RobustnessEvaluator.evaluate

    def record(self, images, labels, image_ids):
        observed.append((images.clone(), labels.clone(), image_ids, self.model[2].weight.clone()))
        assert images.shape == (8, 3, 16, 16)
        assert self.model[2].out_features == 7
        assert self.cfg.reproducibility.seed == 2026
        return real_evaluate(self, images, labels, image_ids)

    monkeypatch.setattr(runner.RobustnessEvaluator, "evaluate", record)
    monkeypatch.setattr(sys, "argv", ["run_robustness.py", "--smoke-test", "--config", str(config)])
    runner.main()
    first_output = capsys.readouterr().out
    torch.rand(10)
    runner.main()
    assert capsys.readouterr().out == first_output
    for index in (0, 1, 3):
        assert torch.equal(observed[0][index], observed[1][index])
    assert observed[0][2] == observed[1][2]

@pytest.mark.parametrize("fn", [apply_gaussian_blur, apply_gaussian_noise, apply_occlusion])
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_corruptions_reject_pixels_outside_pre_normalization_range(fn, value):
    kwargs = {} if fn is apply_gaussian_blur else {"global_seed": 123, "image_id": "dog/a.jpg"}
    image = _img()
    image[0, 0, 0] = value
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        fn(image, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"std": -1}, {"std": float("nan")}, {"mean": float("inf")},
    {"clamp_range": None}, {"clamp_range": [0, 0]}, {"clamp_range": [-1, 1]},
])
def test_noise_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        apply_gaussian_noise(_img(), global_seed=123, image_id="dog/a.jpg", **kwargs)


@pytest.mark.parametrize("fill", [None, [0.5, 0.5], [1.1, 0.5, 0.5], [True, 0.5, 0.5]])
def test_occlusion_rejects_invalid_fill(fill):
    with pytest.raises(ValueError, match="fill_rgb"):
        apply_occlusion(_img(), fill_rgb=fill, global_seed=123, image_id="dog/a.jpg")


@pytest.mark.parametrize("field", [
    "official_evaluation_enabled", "status", "restrictions.use_official_test",
    "restrictions.train_new_model", "restrictions.modify_checkpoint",
])
def test_evaluator_rejects_unsafe_config(cfg, field):
    if "." in field:
        setattr(cfg.restrictions, field.split(".")[1], True)
    else:
        setattr(cfg, field, "official" if field == "status" else True)
    with pytest.raises(RuntimeError, match="tooling-only"):
        RobustnessEvaluator(_dummy_model(), cfg)


def test_evaluator_rejects_empty_batch_and_misaligned_labels(cfg):
    evaluator = RobustnessEvaluator(_dummy_model(), cfg)
    with pytest.raises(ValueError, match="non-empty"):
        evaluator.evaluate(torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long), [])
    images, labels = _batch(n=2)
    with pytest.raises(ValueError, match="labels"):
        evaluator.evaluate(images, labels[:1], _ids(2))


def test_config_supports_zero_noise_std_and_disabled_condition(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["corruptions"]["gaussian_noise"]["std"] = 0
    raw["corruptions"]["occlusion"]["enabled"] = False
    raw["evaluation"]["conditions"].remove("occlusion")
    cfg = load_e8_config(_write_config(tmp_path, raw))
    result = RobustnessEvaluator(_dummy_model(), cfg).evaluate(*_batch(n=2), _ids(2))
    assert result["gaussian_noise_accuracy"] == result["clean_accuracy"]
    assert "occlusion" not in result["conditions"]
    assert "occlusion_accuracy" not in result
