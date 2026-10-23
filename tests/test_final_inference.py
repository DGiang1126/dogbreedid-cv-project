"""Unit tests for the portable pieces of Task 04 inference.

Run inside the DogBreedID repository after copying the files:
    python -m pytest tests/test_final_inference.py -q
These tests use synthetic tensors only; they never load Official Test images.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
torch.set_num_threads(min(torch.get_num_threads(), 4))
from torch.utils.data import DataLoader, Dataset

from src.inference import predictor
from src.inference.predictor import (
    InferenceContractError,
    assert_task04_split_allowed,
    collect_predictions,
    compute_task1_metrics,
    load_raw_predictions,
    save_raw_predictions,
    sha256_file,
    validate_raw_predictions,
    verify_checkpoint_hash,
)


def _fake_predictions(n: int = 4):
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(n, 30)).astype(np.float32)
    labels = np.array([0, 1, 2, 3][:n], dtype=np.int64)
    ids = np.array([f"synthetic/image_{i}.jpg" for i in range(n)], dtype="U64")
    return logits, labels, ids


def test_valid_raw_prediction_contract():
    logits, labels, ids = _fake_predictions()
    got_logits, got_labels, got_ids = validate_raw_predictions(logits, labels, ids)
    assert got_logits.shape == (4, 30)
    assert got_labels.shape == (4,)
    assert got_labels.dtype == np.int64
    assert got_ids.shape == (4,)
    assert got_ids.dtype.kind == "U"


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda x, y, z: (x[:, :29], y, z), "logits must have shape"),
        (lambda x, y, z: (x, y[:2], z), "labels must have shape"),
        (lambda x, y, z: (x, y, z[:2]), "image_ids must have shape"),
        (lambda x, y, z: (np.full_like(x, np.nan), y, z), "NaN or Inf"),
        (lambda x, y, z: (x, np.array([0, 1, 2, 30]), z), "labels must be in"),
        (lambda x, y, z: (x, y, np.array(["same", "same", "third", "fourth"])), "unique"),
    ],
)
def test_invalid_raw_contract_is_rejected(mutator, message):
    logits, labels, ids = _fake_predictions()
    with pytest.raises(InferenceContractError, match=message):
        validate_raw_predictions(*mutator(logits, labels, ids))


def test_npz_roundtrip_has_required_fields_and_unicode_ids(tmp_path: Path):
    logits, labels, ids = _fake_predictions()
    path = save_raw_predictions(tmp_path / "sample_predictions.npz", logits, labels, ids)
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"logits", "labels", "image_ids"}
        assert data["image_ids"].dtype.kind == "U"
    loaded = load_raw_predictions(path)
    np.testing.assert_allclose(loaded["logits"], logits)
    np.testing.assert_array_equal(loaded["labels"], labels)
    np.testing.assert_array_equal(loaded["image_ids"], ids)


def test_checkpoint_hash_verification_accepts_match_and_rejects_mismatch(tmp_path: Path):
    path = tmp_path / "tiny.pt"
    path.write_bytes(b"synthetic checkpoint bytes, never a model")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_file(path) == expected
    assert verify_checkpoint_hash(path, expected) == expected
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_checkpoint_hash(path, "0" * 64)


def test_official_test_is_rejected():
    for blocked in ("official_test", "Official Test", "test", "calibration", "train"):
        with pytest.raises(PermissionError, match="only permits split='validation'"):
            assert_task04_split_allowed(blocked)
    assert_task04_split_allowed("validation")


def test_temperature_scaling_changes_probabilities_but_not_top1():
    logits, labels, _ = _fake_predictions()
    original_logits = logits.copy()
    temperature = 0.80246283961825
    result = compute_task1_metrics(logits, labels, temperature=temperature)
    assert result["calibration"]["temperature"] == temperature
    assert result["calibration"]["top1_prediction_unchanged"] is True
    before = result["calibration"]["before_temperature_scaling"]
    after = result["calibration"]["after_temperature_scaling"]
    assert before["ece_bins"] == 15
    assert after["ece_bins"] == 15
    for key in ("nll", "ece", "multiclass_brier"):
        assert np.isfinite(before[key])
        assert np.isfinite(after[key])
    assert len(result["classification"]["per_class"]) == 30
    assert np.asarray(result["classification"]["confusion_matrix"]).shape == (30, 30)
    np.testing.assert_array_equal(logits, original_logits)


def test_calibration_metrics_match_known_uniform_fixture():
    # With 30 identical logits, p(class)=1/30 for every class/sample.
    # The class-0 tie-breaking predicts class 0 for both samples; labels [0,1]
    # therefore give accuracy=1/2, NLL=ln(30), ECE=|1/2-1/30|, Brier=29/30.
    logits = np.zeros((2, 30), dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    result = compute_task1_metrics(logits, labels, temperature=0.80246283961825)
    before = result["calibration"]["before_temperature_scaling"]
    assert before["top1_accuracy"] == pytest.approx(0.5)
    assert before["nll"] == pytest.approx(np.log(30.0), rel=1e-6)
    assert before["ece"] == pytest.approx(0.5 - 1.0 / 30.0, rel=1e-6)
    assert before["multiclass_brier"] == pytest.approx(29.0 / 30.0, rel=1e-6)
    assert result["calibration"]["top1_prediction_unchanged"] is True


class _SyntheticImageDataset(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(11)
        self.images = torch.randn(7, 3, 224, 224, generator=generator)
        self.labels = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long)
        self.ids = [f"synthetic/{i}.jpg" for i in range(7)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return {"image": self.images[index], "label": self.labels[index], "relative_path": self.ids[index]}


class _TinyDeterministicModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(3, 30)
        torch.manual_seed(19)
        torch.nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))


def test_inference_is_batch_size_invariant_and_does_not_update_weights():
    dataset = _SyntheticImageDataset()
    model = _TinyDeterministicModel()
    model.eval()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}

    out_a = collect_predictions(model, DataLoader(dataset, batch_size=2, shuffle=False), device="cpu")
    out_b = collect_predictions(model, DataLoader(dataset, batch_size=4, shuffle=False), device="cpu")

    np.testing.assert_array_equal(out_a[2], out_b[2])
    np.testing.assert_array_equal(out_a[1], out_b[1])
    np.testing.assert_allclose(out_a[0], out_b[0], rtol=1e-5, atol=1e-6)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), f"Model state changed for {key}"


def test_bad_input_shape_is_rejected():
    model = _TinyDeterministicModel()
    bad_batch = [{
        "image": torch.zeros(1, 3, 128, 128),
        "label": torch.tensor([0]),
        "relative_path": ["synthetic/bad.jpg"],
    }]
    with pytest.raises(InferenceContractError, match="Expected input shape"):
        collect_predictions(model, bad_batch, device="cpu")


def test_model_output_wrong_class_count_is_rejected():
    class WrongHead(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((len(x), 29))

    bad_batch = [{
        "image": torch.zeros(1, 3, 224, 224),
        "label": torch.tensor([0]),
        "relative_path": ["synthetic/bad.jpg"],
    }]
    with pytest.raises(InferenceContractError, match="Model output must have shape"):
        collect_predictions(WrongHead(), bad_batch, device="cpu")


def test_approved_temperature_reuses_e6_loader_without_refitting(monkeypatch):
    from src.calibration import temperature_scaling as e6

    repo_root = Path(__file__).resolve().parents[1]
    artifact = repo_root / predictor.DEFAULT_TEMPERATURE_ARTIFACT
    lock_text = (repo_root / "docs/final_evaluation/configuration_lock.md").read_text(encoding="utf-8")
    assert f"`{predictor.EXPECTED_TEMPERATURE!r}`" in lock_text
    original_loader = e6.load_temperature
    calls = []

    def record_load(path):
        calls.append(path)
        return original_loader(path)

    def forbid_fit(*args, **kwargs):
        pytest.fail("Task 04 must not refit E6 Temperature")

    monkeypatch.setattr(e6, "load_temperature", record_load)
    monkeypatch.setattr(e6, "fit_temperature", forbid_fit)
    monkeypatch.setattr(e6.TemperatureScaler, "fit", forbid_fit)
    temperature, payload = predictor._load_temperature(artifact)
    assert calls == [artifact]
    assert temperature == payload["temperature"] == 0.80246283961825


@pytest.mark.parametrize("temperature", [
    1.0, 0.8, 0.80246284, float(np.float32(0.80246283961825)),
    float(np.nextafter(0.80246283961825, 1.0)),
])
def test_temperature_different_from_configuration_lock_is_rejected(tmp_path, temperature):
    repo_root = Path(__file__).resolve().parents[1]
    payload = json.loads((repo_root / predictor.DEFAULT_TEMPERATURE_ARTIFACT).read_text(encoding="utf-8"))
    payload["temperature"] = temperature
    artifact = tmp_path / "temperature.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="E6 Temperature mismatch with Configuration Lock") as error:
        predictor._load_temperature(artifact)
    assert "Expected=0.80246283961825" in str(error.value)
    assert f"actual={temperature!r}" in str(error.value)


@pytest.fixture()
def synthetic_final_pipeline(monkeypatch, tmp_path):
    """Exercise metadata/NPZ export; replace real model and dataset access."""
    from src.calibration import calibration_metrics
    from src.data import dataset

    source_root = Path(__file__).resolve().parents[1]
    config = predictor._load_yaml(source_root / predictor.DEFAULT_DATA_CONFIG)
    repo_root = tmp_path
    # Keep all generated outputs inside an isolated repository, as required by
    # production path checks. No real images or checkpoint are copied or read.
    for relative in (predictor.DEFAULT_DATA_CONFIG, predictor.DEFAULT_TEMPERATURE_ARTIFACT):
        destination = repo_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / relative).read_bytes())
    for relative in ("src/models/model_factory.py", "data/metadata/class_mapping.json"):
        destination = repo_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.touch()
    monkeypatch.syspath_prepend(str(repo_root))
    logits, labels, image_ids = _fake_predictions()
    monkeypatch.setattr(predictor, "_require_validation_csv", lambda *args: repo_root / predictor.DEFAULT_SPLIT_CSV)
    monkeypatch.setattr(predictor, "verify_checkpoint_hash", lambda *args: predictor.EXPECTED_CHECKPOINT_SHA256)
    monkeypatch.setattr(predictor, "sha256_file", lambda *args: "synthetic-mapping-hash")
    monkeypatch.setattr(predictor, "_load_yaml", lambda *args: config)
    monkeypatch.setattr(dataset, "DogBreedDataset", lambda **kwargs: [])
    monkeypatch.setattr(predictor, "load_e4_model", lambda *args: (torch.nn.Identity(), predictor.EXPECTED_CHECKPOINT_SHA256))
    monkeypatch.setattr(predictor, "collect_predictions", lambda *args, **kwargs: (logits, labels, image_ids))
    monkeypatch.setattr(predictor, "_assert_npz_alignment_with_csv", lambda *args: {})
    monkeypatch.setattr(calibration_metrics, "plot_reliability_diagram", lambda *args, **kwargs: None)

    def run():
        return predictor.run_final_inference(repo_root=repo_root, images_dir=tmp_path, device="cpu")

    return run, config, logits


def test_final_metadata_matches_data_yaml_and_preserves_raw_logits(synthetic_final_pipeline):
    run, config, logits = synthetic_final_pipeline
    original_logits = logits.copy()
    result = run()
    metadata = json.loads(Path(result["paths"]["metadata_json"]).read_text(encoding="utf-8"))
    preprocessing = metadata["preprocessing"]
    assert preprocessing == result["metadata"]["preprocessing"]
    assert preprocessing["evaluation_resize"] == config["preprocessing"]["evaluation"]["resize"] == 256
    assert preprocessing["evaluation_center_crop"] == config["preprocessing"]["evaluation"]["center_crop"] == 224
    assert preprocessing["normalization"] == config["preprocessing"]["normalization"]
    assert preprocessing["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert preprocessing["normalization"]["std"] == [0.229, 0.224, 0.225]
    assert metadata["temperature"] == 0.80246283961825
    np.testing.assert_array_equal(logits, original_logits)
    np.testing.assert_array_equal(result["logits"], original_logits)
    with np.load(result["paths"]["raw_predictions_npz"], allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["logits"], original_logits)


@pytest.mark.parametrize("field", [
    ("preprocessing",),
    ("preprocessing", "evaluation", "resize"),
    ("preprocessing", "evaluation", "center_crop"),
    ("preprocessing", "normalization", "mean"),
    ("preprocessing", "normalization", "std"),
])
def test_missing_preprocessing_config_is_rejected_without_fallback(synthetic_final_pipeline, field):
    run, config, _ = synthetic_final_pipeline
    parent = config
    for key in field[:-1]:
        parent = parent[key]
    del parent[field[-1]]
    with pytest.raises(KeyError, match=field[-1]):
        run()
