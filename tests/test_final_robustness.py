"""Integration tests — E8 Robustness Pipeline, Task 04.

All tests run on synthetic data / dummy models.
No real checkpoint, no real images, no Official Test.

Run:
    python -m pytest tests/test_final_robustness.py -v
"""
from __future__ import annotations

import csv
import copy
import hashlib
import subprocess
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts import run_final_robustness as runner

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

def test_sha256_mismatch_raises(tmp_path: Path, monkeypatch) -> None:
    """SHA-256 mismatch must raise RuntimeError before model is loaded."""
    fake_ckpt = tmp_path / "best.pt"
    fake_ckpt.write_bytes(b"not a real checkpoint")

    # Import after sys.path insert
    from scripts.run_final_robustness import _sha256_file, E4_CHECKPOINT_SHA256
    actual = _sha256_file(fake_ckpt)
    assert actual != E4_CHECKPOINT_SHA256, "Fake checkpoint must not match expected SHA"
    monkeypatch.setattr(runner, "build_model", lambda *args: pytest.fail("Hash mismatch must precede model build"))
    monkeypatch.setattr(runner.torch, "load", lambda *args, **kwargs: pytest.fail("Hash mismatch must precede torch.load"))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        runner._load_e4_model(fake_ckpt)


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
    results = evaluator.evaluate(images, labels, _synthetic_ids(len(images)))
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
    results = evaluator.evaluate(images, labels, _synthetic_ids(len(images)))
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
    r = evaluator.evaluate(images, labels, _synthetic_ids(len(images)))
    for cond in ("gaussian_blur", "gaussian_noise", "occlusion"):
        expected = r["clean_accuracy"] - r[f"{cond}_accuracy"]
        actual = r[f"{cond}_accuracy_drop"]
        assert abs(actual - expected) < 1e-6, f"{cond} drop formula wrong"


def test_accuracy_drop_can_be_negative() -> None:
    """Drop must not be clamped to >= 0; a corrupted model can outperform clean on small N."""
    class ImprovingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, images):
            self.calls += 1
            logits = torch.zeros(len(images), NUM_CLASSES, device=images.device)
            logits[:, 1 if self.calls == 1 else 0] = 1
            return logits

    cfg = _cfg()
    evaluator = RobustnessEvaluator(ImprovingModel(), cfg)
    images, _ = _synthetic_images()
    labels = torch.zeros(N, dtype=torch.long)
    r = evaluator.evaluate(images, labels, _synthetic_ids(len(images)))
    for cond in ("gaussian_blur", "gaussian_noise", "occlusion"):
        assert isinstance(r[f"{cond}_accuracy_drop"], float)
        assert r[f"{cond}_accuracy_drop"] == -1.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Labels and image IDs not modified by corruption
# ─────────────────────────────────────────────────────────────────────────────

def test_labels_unchanged_across_conditions() -> None:
    """Corruption must never mutate label tensors."""
    cfg = _cfg()
    evaluator = RobustnessEvaluator(_dummy_model(), cfg)
    images, labels = _synthetic_images()
    labels_before = labels.clone()
    evaluator.evaluate(images, labels, _synthetic_ids(len(images)))
    assert torch.equal(labels, labels_before)


def test_images_unchanged_by_evaluator() -> None:
    """Evaluator must not mutate the input images tensor."""
    cfg = _cfg()
    evaluator = RobustnessEvaluator(_dummy_model(), cfg)
    images, labels = _synthetic_images()
    images_before = images.clone()
    evaluator.evaluate(images, labels, _synthetic_ids(len(images)))
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
    out = apply_gaussian_noise(img, mean=0.0, std=0.05, clamp_range=(0.0, 1.0), global_seed=_cfg().reproducibility.seed, image_id="synthetic/a.jpg")
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_occlusion_output_in_01() -> None:
    img = torch.rand(3, 224, 224)
    out = apply_occlusion(img, side_fraction=0.30, fill_rgb=(0.5, 0.5, 0.5), global_seed=_cfg().reproducibility.seed, image_id="synthetic/0.jpg")
    assert out.min() >= 0.0 and out.max() <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Occlusion deterministic — same image_id, same result (order-independent)
# ─────────────────────────────────────────────────────────────────────────────

def test_occlusion_deterministic_by_image_id() -> None:
    img = torch.rand(3, 224, 224)
    out1 = apply_occlusion(img, global_seed=_cfg().reproducibility.seed, image_id="synthetic/7.jpg")
    out2 = apply_occlusion(img, global_seed=_cfg().reproducibility.seed, image_id="synthetic/7.jpg")
    assert torch.equal(out1, out2)


def test_occlusion_different_ids_different_positions() -> None:
    img = torch.ones(3, 224, 224) * 0.3
    out0 = apply_occlusion(img, global_seed=_cfg().reproducibility.seed, image_id="synthetic/0.jpg")
    out1 = apply_occlusion(img, global_seed=_cfg().reproducibility.seed, image_id="synthetic/1.jpg")
    # Different positions → different tensors (unless astronomically unlikely)
    assert not torch.equal(out0, out1)


def test_occlusion_order_independence() -> None:
    """Occlusion of img_A is identical whether processed before or after img_B."""
    img_a = torch.rand(3, 224, 224)
    img_b = torch.rand(3, 224, 224)
    occ_a_first = apply_occlusion(img_a, global_seed=_cfg().reproducibility.seed, image_id="synthetic/0.jpg")
    apply_occlusion(img_b, global_seed=_cfg().reproducibility.seed, image_id="synthetic/b.jpg")
    occ_a_second = apply_occlusion(img_a, global_seed=_cfg().reproducibility.seed, image_id="synthetic/0.jpg")   # same id, different call order
    assert torch.equal(occ_a_first, occ_a_second)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Noise reproducibility (same seed → same result)
# ─────────────────────────────────────────────────────────────────────────────

def test_noise_reproducible_with_same_seed() -> None:
    img = torch.rand(3, 224, 224)
    out1 = apply_gaussian_noise(img, global_seed=_cfg().reproducibility.seed, image_id="synthetic/a.jpg")
    out2 = apply_gaussian_noise(img, global_seed=_cfg().reproducibility.seed, image_id="synthetic/a.jpg")
    assert torch.equal(out1, out2)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Model weights not modified
# ─────────────────────────────────────────────────────────────────────────────

def test_model_weights_unchanged() -> None:
    cfg = _cfg()
    model = _dummy_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    images, labels = _synthetic_images()
    RobustnessEvaluator(model, cfg).evaluate(images, labels, _synthetic_ids(len(images)))
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), f"Weight {k} was modified!"


# ─────────────────────────────────────────────────────────────────────────────
# 12. Same predictions → same metrics (determinism end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

def test_deterministic_results_same_input() -> None:
    cfg = _cfg()
    model = _dummy_model()
    images, labels = _synthetic_images()
    r1 = RobustnessEvaluator(model, cfg).evaluate(images.clone(), labels.clone(), _synthetic_ids(len(images)))
    r2 = RobustnessEvaluator(model, cfg).evaluate(images.clone(), labels.clone(), _synthetic_ids(len(images)))
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
    single = apply_occlusion(imgs[2], global_seed=_cfg().reproducibility.seed, image_id=_synthetic_ids(4)[2])
    batch_result = torch.stack([apply_occlusion(img, global_seed=_cfg().reproducibility.seed, image_id=_synthetic_ids(4)[i]) for i, img in enumerate(imgs)])
    assert torch.equal(single, batch_result[2])


# ─────────────────────────────────────────────────────────────────────────────
# 15. Regression: RobustnessEvaluator interface unchanged (Task 03 compat)
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluator_interface_regression() -> None:
    """Evaluator still accepts (model, cfg) and returns dict with expected keys."""
    cfg = _cfg()
    evaluator = RobustnessEvaluator(_dummy_model(), cfg)
    images, labels = _synthetic_images()
    r = evaluator.evaluate(images, labels, _synthetic_ids(len(images)))
    assert isinstance(r, dict)
    assert "clean_accuracy" in r
    assert "conditions" in r
    assert isinstance(r["conditions"], list)

# Synthetic files stay in pytest tmp_path; integration exercises the real
# dataset -> DataLoader -> shared evaluator -> JSON output path.
@pytest.fixture()
def synthetic_fixture(tmp_path):
    mapping = json.loads((runner.SOURCE_ROOT / "data/metadata/class_mapping.json").read_text(encoding="utf-8"))
    images_dir = tmp_path / "Images"
    rows = []
    rng = np.random.default_rng(2026)
    for index in range(5):
        image_id = f"synthetic/dog_{index}.png"
        path = images_dir / image_id
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(rng.integers(0, 256, (256, 288, 3), dtype=np.uint8))
        if index == 0:
            image = image.convert("L")  # Shared dataset must convert to RGB.
        image.save(path)
        rows.append({"relative_path": image_id, "class_id": index,
                     "breed_name": mapping[str(index)], "source_split": "synthetic"})
    csv_path = tmp_path / "fixture.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"csv_path": csv_path, "images_dir": images_dir,
            "output_path": tmp_path / "run" / "results.json", "rows": rows, "mapping": mapping}


def _run_synthetic(fixture, **kwargs):
    return runner.run_final_robustness(
        csv_path=fixture["csv_path"], images_dir=fixture["images_dir"],
        output_path=fixture["output_path"], split="synthetic", device="cpu", **kwargs,
    )


def test_synthetic_file_runner_reuses_shared_evaluator_and_writes_honest_output(synthetic_fixture, monkeypatch):
    calls = []

    class RecordEvaluator(RobustnessEvaluator):
        def evaluate(self, images, labels, image_ids, **kwargs):
            calls.append((images.shape, list(image_ids)))
            return super().evaluate(images, labels, image_ids, **kwargs)

    monkeypatch.setattr(runner, "RobustnessEvaluator", RecordEvaluator)
    monkeypatch.setattr(runner, "_load_e4_model", lambda *args: pytest.fail("Synthetic must not load E4"))
    result = _run_synthetic(synthetic_fixture, batch_size=2)
    written = json.loads(synthetic_fixture["output_path"].read_text(encoding="utf-8"))
    assert result == written
    assert result["conditions"] == ["clean", "gaussian_blur", "gaussian_noise", "occlusion"]
    assert result["n_samples"] == 5
    assert [shape[0] for shape, _ in calls] == [2, 2, 1]
    assert all(shape[1:] == (3, 224, 224) for shape, _ in calls)
    assert result["image_ids"] == [row["relative_path"] for row in synthetic_fixture["rows"]]
    assert result["split"] == "synthetic" and result["synthetic"] is True
    assert result["model_experiment"] == "synthetic_dummy"
    assert result["checkpoint"] is result["checkpoint_sha256"] is None
    assert result["official_evaluation"] is False
    for name in ("gaussian_blur", "gaussian_noise", "occlusion"):
        assert result[f"{name}_accuracy_drop"] == result["clean_accuracy"] - result[f"{name}_accuracy"]


def test_synthetic_cli_runs_the_actual_file_pipeline(synthetic_fixture, tmp_path):
    command = [sys.executable, str(runner.SOURCE_ROOT / "scripts/run_final_robustness.py"),
               "--split", "synthetic", "--csv", str(synthetic_fixture["csv_path"]),
               "--images-dir", str(synthetic_fixture["images_dir"]),
               "--output", str(synthetic_fixture["output_path"]), "--batch-size", "2", "--device", "cpu"]
    process = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result == json.loads(synthetic_fixture["output_path"].read_text(encoding="utf-8"))
    assert result["synthetic"] is True and result["n_samples"] == 5
    assert len(result["conditions"]) == 4


def _capture_loader(monkeypatch, cfg, batch_size, order):
    generator = torch.Generator().manual_seed(7)
    images = torch.rand(5, 3, 32, 32, generator=generator)
    labels = torch.tensor([0, 1, 2, 0, 1])
    ids = _synthetic_ids(5)
    captured = {}

    class CaptureEvaluator(RobustnessEvaluator):
        def _corrupt(self, tensors, image_ids, name):
            output = super()._corrupt(tensors, image_ids, name)
            for image_id, tensor in zip(image_ids, output):
                captured[name, image_id] = tensor.clone()
            return output

    monkeypatch.setattr(runner, "RobustnessEvaluator", CaptureEvaluator)
    samples = [(images[i], labels[i], ids[i]) for i in order]
    result = runner._evaluate_loader(_dummy_model(), DataLoader(samples, batch_size=batch_size), cfg)
    return result, captured


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_runner_noise_and_occlusion_tensor_invariance_with_batches_and_order(monkeypatch, batch_size):
    cfg = _cfg()
    cfg.reproducibility.seed = 2026
    original, reference = _capture_loader(monkeypatch, cfg, 5, list(range(5)))
    reordered, actual = _capture_loader(monkeypatch, cfg, batch_size, [4, 1, 3, 0, 2])
    for name in ("gaussian_noise", "occlusion"):
        for image_id in _synthetic_ids(5):
            assert torch.equal(reference[name, image_id], actual[name, image_id])
    for key in original:
        if "accuracy" in key:
            assert original[key] == reordered[key]


def test_runner_passes_config_seed_to_both_random_corruptions(monkeypatch):
    cfg = _cfg()
    cfg.reproducibility.seed = 123
    _, first = _capture_loader(monkeypatch, cfg, 2, list(range(5)))
    _, repeated = _capture_loader(monkeypatch, cfg, 3, list(range(5)))
    cfg.reproducibility.seed = 2026
    _, different = _capture_loader(monkeypatch, cfg, 2, list(range(5)))
    for name in ("gaussian_noise", "occlusion"):
        key = name, _synthetic_ids(5)[0]
        assert torch.equal(first[key], repeated[key])
        assert not torch.equal(first[key], different[key])


def test_runner_corrupts_pre_normalization_tensors_and_keeps_clean_untouched(monkeypatch):
    images, labels = _synthetic_images()
    raw, normalized = [], []

    class CaptureEvaluator(RobustnessEvaluator):
        def _corrupt(self, tensors, image_ids, name):
            assert tensors.min() >= 0 and tensors.max() <= 1
            output = super()._corrupt(tensors, image_ids, name)
            if name == "clean":
                assert torch.equal(output, tensors)
            raw.append(output.clone())
            return output

    class CaptureModel(nn.Module):
        def forward(self, tensors):
            assert not self.training and torch.is_inference_mode_enabled()
            normalized.append(tensors.clone())
            return torch.zeros(len(tensors), NUM_CLASSES)

    monkeypatch.setattr(runner, "RobustnessEvaluator", CaptureEvaluator)
    samples = list(zip(images, labels, _synthetic_ids()))
    runner._evaluate_loader(CaptureModel(), DataLoader(samples, batch_size=3), _cfg())
    mean = torch.tensor(runner.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(runner.IMAGENET_STD).view(3, 1, 1)
    for before, after in zip(raw, normalized):
        assert before.min() >= 0 and before.max() <= 1
        assert torch.equal(after, (before - mean) / std)


def test_pre_norm_dataset_reuses_shared_components_and_preserves_csv_id(synthetic_fixture):
    from src.data.transforms import build_transforms
    payload = runner._yaml(runner.SOURCE_ROOT / "configs/data.yaml")
    dataset = runner._PreNormDataset(synthetic_fixture["csv_path"], synthetic_fixture["images_dir"],
                                     split="synthetic", data_payload=payload, mapping=synthetic_fixture["mapping"])
    image, label, image_id = dataset[0]
    assert image.shape == (3, 224, 224) and 0 <= image.min() <= image.max() <= 1
    assert image_id == synthetic_fixture["rows"][0]["relative_path"]
    assert label == 0
    with Image.open(synthetic_fixture["images_dir"] / image_id) as source:
        expected = build_transforms(payload, "validation")(source.convert("RGB"))
    mean = torch.tensor(runner.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(runner.IMAGENET_STD).view(3, 1, 1)
    assert torch.equal((image - mean) / std, expected)


def test_integrated_runner_preserves_all_model_weights_buffers_and_gradients(synthetic_fixture, monkeypatch):
    model = nn.Sequential(nn.BatchNorm2d(3), nn.Dropout(0.5), nn.AdaptiveAvgPool2d(1),
                          nn.Flatten(), nn.Linear(3, NUM_CLASSES)).train()
    before = copy.deepcopy(model.state_dict())
    monkeypatch.setattr(runner, "_make_synthetic_model", lambda seed: model)
    _run_synthetic(synthetic_fixture, batch_size=2)
    assert not model.training
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name
    assert all(parameter.grad is None for parameter in model.parameters())


def test_checkpoint_uses_e4_factory_no_download_weights_only_and_strict(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint; torch.load below is mocked")
    model = _dummy_model()
    state = copy.deepcopy(model.state_dict())
    calls = {}
    real_load_state = model.load_state_dict

    def build(cfg):
        calls["config"] = cfg
        return model

    def load(path, **kwargs):
        calls["torch_load"] = (path, kwargs)
        return {"model_state_dict": state}

    def load_state(value, *, strict):
        calls["strict"] = strict
        return real_load_state(value, strict=strict)

    monkeypatch.setattr(runner, "_sha256_file", lambda path: runner.E4_CHECKPOINT_SHA256)
    monkeypatch.setattr(runner, "build_model", build)
    monkeypatch.setattr(runner.torch, "load", load)
    monkeypatch.setattr(model, "load_state_dict", load_state)
    assert runner._load_e4_model(checkpoint) is model
    assert calls["config"]["experiment"]["id"] == "E4"
    assert calls["config"]["experiment"]["seed"] == 42
    assert calls["config"]["model"] == {"type": "mobilenet_v2", "weights": "none", "backbone_frozen": True, "num_classes": 30}
    assert calls["config"]["augmentation"]["cutmix"]["enabled"] is True
    assert calls["torch_load"] == (checkpoint, {"map_location": "cpu", "weights_only": True})
    assert calls["strict"] is True and not model.training


@pytest.mark.parametrize("path", ["data/splits/train.csv", "data/splits/calibration.csv", "validation.csv"])
def test_validation_rejects_noncanonical_csv_before_checkpoint_io(path, synthetic_fixture, monkeypatch):
    monkeypatch.setattr(runner, "_load_e4_model", lambda *args: pytest.fail("Unapproved CSV must be blocked first"))
    with pytest.raises(RuntimeError, match="canonical"):
        runner.run_final_robustness(csv_path=path, images_dir=synthetic_fixture["images_dir"],
                                     output_path=synthetic_fixture["output_path"], split="validation")


def test_canonical_validation_guard_accepts_committed_split_without_inference():
    runner._assert_validation_csv((runner.SOURCE_ROOT / "data/splits/validation.csv").resolve())


def test_canonical_validation_guard_rejects_replaced_content(tmp_path, monkeypatch):
    csv_path = tmp_path / "data/splits/validation.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_bytes(b"relative_path,class_id\nsynthetic/x.png,9\n")
    monkeypatch.setattr(runner, "SOURCE_ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: b"relative_path,class_id\nsynthetic/x.png,0\n")
    with pytest.raises(RuntimeError, match="committed approved content"):
        runner._assert_validation_csv(csv_path)


@pytest.mark.parametrize("bad_path", ["data/splits/official_test.csv", "renamed/official-test/validation.csv"])
def test_runner_rejects_official_paths_before_any_input_io(synthetic_fixture, monkeypatch, bad_path):
    monkeypatch.setattr(runner, "_yaml", lambda *args: pytest.fail("Official path must be blocked before config I/O"))
    with pytest.raises(RuntimeError, match="Official Test"):
        runner.run_final_robustness(csv_path=bad_path, images_dir=synthetic_fixture["images_dir"],
                                     output_path=synthetic_fixture["output_path"])


def test_cli_official_flag_is_rejected_before_smoke_or_config(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_final_robustness.py", "--official-test", "--smoke-test"])
    monkeypatch.setattr(runner, "load_e8_config", lambda *args: pytest.fail("Must reject official flag first"))
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 1
    assert "blocked in Task 04" in capsys.readouterr().err


def test_runner_refuses_existing_output_without_modifying_it(synthetic_fixture, monkeypatch):
    output = synthetic_fixture["output_path"]
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing output must survive")
    monkeypatch.setattr(runner, "_make_synthetic_model", lambda *args: pytest.fail("Must reject output first"))
    with pytest.raises(FileExistsError):
        _run_synthetic(synthetic_fixture)
    assert output.read_bytes() == b"existing output must survive"


def test_exclusive_output_write_preserves_file_created_after_preflight(synthetic_fixture, monkeypatch):
    original = runner._evaluate_loader

    def race(*args, **kwargs):
        result = original(*args, **kwargs)
        output = synthetic_fixture["output_path"]
        output.parent.mkdir(parents=True)
        output.write_bytes(b"created by another run")
        return result

    monkeypatch.setattr(runner, "_evaluate_loader", race)
    with pytest.raises(FileExistsError):
        _run_synthetic(synthetic_fixture)
    assert synthetic_fixture["output_path"].read_bytes() == b"created by another run"


@pytest.mark.parametrize("relative", [
    "outputs/metrics/E6/new.json", "outputs/metrics/E7/new.json",
    "outputs/predictions/task04_validation_review/new.json", "outputs/robustness/task04/validation/fake.json",
])
def test_synthetic_cannot_write_into_other_members_or_validation_outputs(relative):
    with pytest.raises(ValueError, match="output must be under"):
        runner._output_path(runner.SOURCE_ROOT / relative, "synthetic")


@pytest.mark.parametrize("field,value", [
    ("relative_path", "../escape.png"), ("relative_path", "real/dog.png"),
    ("class_id", 0.5), ("breed_name", "wrong breed"), ("source_split", "official_train"),
])
def test_synthetic_csv_contract_rejects_bad_paths_labels_or_provenance(synthetic_fixture, field, value):
    rows = copy.deepcopy(synthetic_fixture["rows"])
    rows[0][field] = value
    with synthetic_fixture["csv_path"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        _run_synthetic(synthetic_fixture)
    assert not synthetic_fixture["output_path"].exists()


def test_synthetic_cannot_accept_checkpoint_or_claim_validation(synthetic_fixture):
    with pytest.raises(ValueError, match="dummy model"):
        _run_synthetic(synthetic_fixture, checkpoint_path="fake.pt")
    with pytest.raises(RuntimeError, match="canonical"):
        runner.run_final_robustness(csv_path=synthetic_fixture["csv_path"],
                                     images_dir=synthetic_fixture["images_dir"],
                                     output_path=synthetic_fixture["output_path"], split="validation")


def test_smoke_model_and_data_are_seeded_from_cfg_not_global_rng():
    cfg = _cfg()
    cfg.reproducibility.seed = 2026
    first = runner._smoke_test(cfg, batch_size=3)
    torch.rand(20)
    second = runner._smoke_test(cfg, batch_size=2)
    assert first == second


def test_loader_keeps_negative_drop_and_weights_short_batch_by_samples():
    class ImprovingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, images):
            batch_index, condition = divmod(self.calls, 4)
            self.calls += 1
            logits = torch.zeros(len(images), NUM_CLASSES)
            # Only the short final batch is correct when clean; every
            # corrupted sample is correct. A mean of batch means is wrong.
            predicted = 1 if condition == 0 and batch_index < 2 else 0
            logits[:, predicted] = 1
            return logits

    generator = torch.Generator().manual_seed(7)
    samples = [(torch.rand(3, 32, 32, generator=generator), 0, image_id)
               for image_id in _synthetic_ids(5)]
    result = runner._evaluate_loader(ImprovingModel(), DataLoader(samples, batch_size=2), _cfg())
    assert result["clean_accuracy"] == 1 / 5
    for name in ("gaussian_blur", "gaussian_noise", "occlusion"):
        assert result[f"{name}_accuracy"] == 1.0
        assert result[f"{name}_accuracy_drop"] == -0.8


def test_runner_rejects_official_evaluation_config_before_model_or_images(synthetic_fixture, monkeypatch):
    config = copy.deepcopy(_cfg().raw)
    config["execution"]["official_evaluation_enabled"] = True
    config_path = synthetic_fixture["csv_path"].parent / "unsafe_config.yaml"
    config_path.write_text(runner.yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(runner, "_make_synthetic_model", lambda *args: pytest.fail("Unsafe config must be rejected first"))
    monkeypatch.setattr(runner, "_PreNormDataset", lambda *args, **kwargs: pytest.fail("Unsafe config must precede images"))
    with pytest.raises(ValueError, match="official_evaluation_enabled"):
        _run_synthetic(synthetic_fixture, config_path=config_path)
    assert not synthetic_fixture["output_path"].exists()
