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
    def synthetic_diagram(*args, output_path, **kwargs):
        from PIL import Image

        Image.new("RGB", (2, 2), "white").save(output_path)
        return Path(output_path)

    monkeypatch.setattr(calibration_metrics, "plot_reliability_diagram", synthetic_diagram)

    def run(**kwargs):
        return predictor.run_final_inference(repo_root=repo_root, images_dir=tmp_path, device="cpu", **kwargs)

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


ARTIFACT_NAMES = (
    "validation_raw_predictions.npz", "validation_metrics.json",
    "validation_metadata.json", "validation_reliability_diagram.png",
)


def _run_output(tmp_path):
    return tmp_path / predictor.DEFAULT_OUTPUT_DIR


def _assert_failed_bundle(output, foreign_files=(), retained_artifacts=()):
    assert (output / ".incomplete").read_text().startswith("INCOMPLETE RUN")
    assert {p.name for p in output.iterdir()} == {
        ".inference_run", ".incomplete", *foreign_files, *retained_artifacts,
    }
    assert not (output / ".complete").exists()
    with pytest.raises(InferenceContractError, match="Incomplete"):
        load_raw_predictions(output / ARTIFACT_NAMES[0])


def test_complete_bundle_is_published_once_without_changing_payloads(synthetic_final_pipeline, tmp_path):
    from PIL import Image

    run, _, logits = synthetic_final_pipeline
    expected_metrics = compute_task1_metrics(logits, _fake_predictions()[1], temperature=predictor.EXPECTED_TEMPERATURE)
    result = run()
    output = _run_output(tmp_path)
    assert {p.name for p in output.iterdir()} == {*ARTIFACT_NAMES, ".inference_run", ".complete"}
    assert (output / ".complete").read_bytes() == (output / ".inference_run").read_bytes()
    predictor.assert_prediction_bundle_complete(output / ARTIFACT_NAMES[0])
    np.testing.assert_array_equal(load_raw_predictions(output / ARTIFACT_NAMES[0])["logits"], logits)
    assert json.loads((output / ARTIFACT_NAMES[1]).read_text(encoding="utf-8")) == expected_metrics
    assert json.loads((output / ARTIFACT_NAMES[2]).read_text(encoding="utf-8")) == result["metadata"]
    assert result["metadata"]["outputs"]["raw_predictions_npz"] == (
        predictor.DEFAULT_OUTPUT_DIR / ARTIFACT_NAMES[0]
    ).as_posix()
    assert {Path(p).parent for p in result["paths"].values()} == {output}
    with np.load(output / ARTIFACT_NAMES[0], allow_pickle=False) as stored:
        assert set(stored.files) == {"logits", "labels", "image_ids"}
        for key, expected in zip(("logits", "labels", "image_ids"), _fake_predictions()):
            np.testing.assert_array_equal(stored[key], expected)
    with Image.open(output / ARTIFACT_NAMES[3]) as image:
        image.verify()
    original = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError, match="already exists"):
        run()
    assert {p.name: p.read_bytes() for p in output.iterdir()} == original
    # A different run directory remains supported.
    second = run(output_dir="outputs/predictions/a_fresh_run")
    assert set(Path(second["outputs_dir"]).iterdir()) != set(output.iterdir())


@pytest.mark.parametrize("existing", [None, "unrelated.txt", *ARTIFACT_NAMES])
def test_existing_output_is_rejected_before_model_or_inference(
    synthetic_final_pipeline, tmp_path, monkeypatch, existing,
):
    run, _, _ = synthetic_final_pipeline
    output = _run_output(tmp_path)
    output.mkdir(parents=True)
    if existing:
        (output / existing).write_bytes(b"previous result must survive")
    before = {p.name: p.read_bytes() for p in output.iterdir()}

    def forbid(*args, **kwargs):
        pytest.fail("Output preflight must precede checkpoint loading and inference")

    for name in ("verify_checkpoint_hash", "load_e4_model", "collect_predictions"):
        monkeypatch.setattr(predictor, name, forbid)
    with pytest.raises(FileExistsError, match="already exists"):
        run()
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_directory_created_after_preflight_is_not_reused(synthetic_final_pipeline, tmp_path, monkeypatch):
    run, _, _ = synthetic_final_pipeline
    safe_output = predictor._safe_output_dir

    def race(*args):
        output = safe_output(*args)
        output.mkdir(parents=True)
        (output / "other_writer.txt").write_bytes(b"foreign")
        return output

    monkeypatch.setattr(predictor, "_safe_output_dir", race)
    monkeypatch.setattr(predictor, "collect_predictions", lambda *a, **k: pytest.fail("Inference started"))
    with pytest.raises(FileExistsError):
        run()
    output = _run_output(tmp_path)
    assert {p.name for p in output.iterdir()} == {"other_writer.txt"}
    assert (output / "other_writer.txt").read_bytes() == b"foreign"


@pytest.mark.parametrize("artifact", ARTIFACT_NAMES)
def test_artifact_created_after_preflight_is_never_overwritten(
    synthetic_final_pipeline, tmp_path, monkeypatch, artifact,
):
    run, _, _ = synthetic_final_pipeline
    collect = predictor.collect_predictions
    output = _run_output(tmp_path)

    def race(*args, **kwargs):
        (output / artifact).write_bytes(b"another writer's artifact")
        return collect(*args, **kwargs)

    monkeypatch.setattr(predictor, "collect_predictions", race)
    with pytest.raises(FileExistsError):
        run()
    _assert_failed_bundle(output, [artifact], ARTIFACT_NAMES[:ARTIFACT_NAMES.index(artifact)])
    assert (output / artifact).read_bytes() == b"another writer's artifact"


@pytest.mark.parametrize("stage", ["npz", "metrics", "metadata", "png", "plot_import", "missing_png"])
def test_staging_failure_cannot_publish_a_partial_bundle(
    synthetic_final_pipeline, tmp_path, monkeypatch, stage,
):
    from src.calibration import calibration_metrics

    run, _, _ = synthetic_final_pipeline
    staged_paths = []
    original_save = predictor.save_raw_predictions

    def save(path, *args, **kwargs):
        staged_paths.append(Path(path))
        if stage == "npz":
            Path(path).write_bytes(b"partial npz")
            raise OSError("injected staging failure")
        return original_save(path, *args, **kwargs)

    monkeypatch.setattr(predictor, "save_raw_predictions", save)
    original_dump = predictor.json.dump

    def dump(payload, handle, *args, **kwargs):
        if Path(handle.name).name == f"validation_{stage}.json":
            handle.write('{"partial":')
            raise OSError("injected staging failure")
        return original_dump(payload, handle, *args, **kwargs)

    monkeypatch.setattr(predictor.json, "dump", dump)

    def plot(*args, output_path, **kwargs):
        if stage == "missing_png":
            return None
        Path(output_path).write_bytes(b"partial png")
        error = ImportError if stage == "plot_import" else OSError
        raise error("injected staging failure")

    if stage in ("png", "plot_import", "missing_png"):
        monkeypatch.setattr(calibration_metrics, "plot_reliability_diagram", plot)
    with pytest.raises((OSError, ImportError, RuntimeError), match="injected staging failure|expected inference output bundle"):
        run()
    output = _run_output(tmp_path)
    _assert_failed_bundle(output)
    assert staged_paths and all(not path.parent.exists() for path in staged_paths)
    with pytest.raises(FileExistsError):
        run()


@pytest.mark.parametrize("fail_at", [1, 2, 3, 4])
def test_publication_write_failure_retains_artifacts_in_a_rejected_bundle(
    synthetic_final_pipeline, tmp_path, monkeypatch, fail_at,
):
    run, _, _ = synthetic_final_pipeline
    original_copy = predictor.shutil.copyfileobj
    calls = 0

    def failing_copy(reader, writer):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            writer.write(b"partial artifact")
            raise OSError("injected publication failure")
        return original_copy(reader, writer)

    monkeypatch.setattr(predictor.shutil, "copyfileobj", failing_copy)
    with pytest.raises(OSError, match="publication failure"):
        run()
    _assert_failed_bundle(_run_output(tmp_path), retained_artifacts=ARTIFACT_NAMES[:fail_at])
    assert (_run_output(tmp_path) / ARTIFACT_NAMES[fail_at - 1]).read_bytes() == b"partial artifact"


def test_rollback_preserves_foreign_file_replacing_a_published_artifact(
    synthetic_final_pipeline, tmp_path, monkeypatch,
):
    run, _, _ = synthetic_final_pipeline

    def fail_finish(transaction):
        artifact = transaction.output / ARTIFACT_NAMES[0]
        artifact.rename(tmp_path / "moved_owned_artifact.npz")
        artifact.write_bytes(b"foreign replacement")
        raise OSError("injected completion failure")

    monkeypatch.setattr(predictor._OutputRun, "finish", fail_finish)
    with pytest.raises(OSError, match="completion failure"):
        run()
    output = _run_output(tmp_path)
    _assert_failed_bundle(output, [ARTIFACT_NAMES[0]], ARTIFACT_NAMES)
    assert (output / ARTIFACT_NAMES[0]).read_bytes() == b"foreign replacement"


def test_exclusive_npz_export_refuses_overwrite_and_preserves_its_partial_file(tmp_path, monkeypatch):
    values = _fake_predictions()
    existing = tmp_path / "existing.npz"
    save_raw_predictions(existing, *values, exclusive=True)
    original = existing.read_bytes()
    with pytest.raises(FileExistsError):
        save_raw_predictions(existing, *values, exclusive=True)
    assert existing.read_bytes() == original

    def partial_write(handle, **kwargs):
        handle.write(b"partial npz")
        raise OSError("injected npz failure")

    monkeypatch.setattr(predictor.np, "savez_compressed", partial_write)
    failed = tmp_path / "failed.npz"
    with pytest.raises(OSError, match="npz failure"):
        save_raw_predictions(failed, *values, exclusive=True)
    assert failed.read_bytes() == b"partial npz"
    assert existing.read_bytes() == original


@pytest.mark.parametrize("relative", [
    "outputs", "outputs/predictions", "outputs/metrics/E6/new_run",
    "outputs/robustness/new_run", "outputs/checkpoints/new_run",
    "outputs/predictions/final/new_run", "outputs/predictions/E6/new_run",
    "outputs/predictions/official_test/new_run", "outside_outputs/new_run",
])
def test_protected_output_locations_are_rejected(tmp_path, relative):
    with pytest.raises(PermissionError):
        predictor._safe_output_dir(tmp_path, relative)
    assert not (tmp_path / relative).exists()


def test_new_run_cannot_be_nested_in_a_previous_bundle(tmp_path):
    old_output = _run_output(tmp_path)
    old_output.mkdir(parents=True)
    (old_output / ARTIFACT_NAMES[0]).write_bytes(b"old result")
    with pytest.raises(PermissionError, match="existing output bundle"):
        predictor._safe_output_dir(tmp_path, old_output / "nested_run")
    assert (old_output / ARTIFACT_NAMES[0]).read_bytes() == b"old result"


def test_runner_still_refuses_official_test_before_creating_outputs(synthetic_final_pipeline, tmp_path):
    run, _, _ = synthetic_final_pipeline
    with pytest.raises(PermissionError, match="Official Test is locked"):
        run(split_name="official_test")
    assert not _run_output(tmp_path).exists()


@pytest.mark.parametrize("artifact", ARTIFACT_NAMES)
def test_rollback_preserves_replacement_after_ownership_check(tmp_path, monkeypatch, artifact):
    """Replay the exact stale-ownership result that formerly authorized unlink."""
    output = tmp_path / "run"
    replacement = b"foreign file installed after identity check"
    original_owned = predictor._owned_file
    with pytest.raises(OSError, match="force rollback"):
        with predictor._exclusive_output_run(output) as transaction:
            staging = transaction.staging
            for name in ARTIFACT_NAMES:
                (staging / name).write_bytes(b"owned artifact")
            transaction.publish()
            target, identity = next(item for item in transaction.published if item[0].name == artifact)
            checked_owned = original_owned(target, identity)
            assert checked_owned
            # The scheduler lets a foreign writer replace the name after stat,
            # before the old cleanup uses that already-computed True result.
            target.rename(tmp_path / "moved_original")
            target.write_bytes(replacement)

            def stale_check(path, expected):
                return checked_owned if path == target else original_owned(path, expected)

            monkeypatch.setattr(predictor, "_owned_file", stale_check)
            raise OSError("force rollback after ownership check")
    assert target.read_bytes() == replacement
    assert (tmp_path / "moved_original").read_bytes() == b"owned artifact"
    assert not staging.exists()
    _assert_failed_bundle(output, retained_artifacts=ARTIFACT_NAMES)
    # Invalidation is idempotent and never deletes an artifact or foreign file.
    transaction.rollback()
    assert target.read_bytes() == replacement


def test_exclusive_file_failure_preserves_replacement_after_ownership_check(tmp_path, monkeypatch):
    target = tmp_path / "artifact.npz"
    replacement = b"foreign replacement"
    with pytest.raises(OSError, match="force writer failure"):
        with predictor._exclusive_binary_file(target) as handle:
            handle.write(b"owned bytes")
            identity = predictor._identity(predictor.os.fstat(handle.fileno()))
            handle.close()  # Release Windows' open-file rename restrictions.
            checked_owned = predictor._owned_file(target, identity)
            assert checked_owned
            target.rename(tmp_path / "owned_backup")
            target.write_bytes(replacement)
            monkeypatch.setattr(predictor, "_owned_file", lambda *args: checked_owned)
            raise OSError("force writer failure after ownership check")
    assert target.read_bytes() == replacement


@pytest.mark.parametrize("fail", [False, True])
def test_no_public_unlink_even_for_completion_markers(synthetic_final_pipeline, tmp_path, monkeypatch, fail):
    run, _, _ = synthetic_final_pipeline
    output = _run_output(tmp_path)
    original_unlink = Path.unlink

    def no_public_unlink(path, *args, **kwargs):
        assert not path.is_relative_to(output), f"Unsafe public cleanup: {path}"
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", no_public_unlink)
    if fail:
        def fail_finish(transaction):
            raise OSError("completion failure")

        monkeypatch.setattr(predictor._OutputRun, "finish", fail_finish)
        with pytest.raises(OSError, match="completion failure"):
            run()
        _assert_failed_bundle(output, retained_artifacts=ARTIFACT_NAMES)
    else:
        run()
        predictor.assert_prediction_bundle_complete(output / ARTIFACT_NAMES[0])


@pytest.mark.parametrize("marker", [".incomplete", ".complete", "foreign_file.txt"])
def test_foreign_file_or_control_marker_is_neither_overwritten_nor_deleted(
    synthetic_final_pipeline, tmp_path, monkeypatch, marker,
):
    run, _, _ = synthetic_final_pipeline
    output = _run_output(tmp_path)
    collect = predictor.collect_predictions

    def inject(*args, **kwargs):
        (output / marker).write_bytes(b"foreign marker")
        return collect(*args, **kwargs)

    monkeypatch.setattr(predictor, "collect_predictions", inject)
    with pytest.raises(RuntimeError, match="Unexpected files"):
        run()
    assert (output / marker).read_bytes() == b"foreign marker"
    assert (output / ".incomplete").is_file()
    with pytest.raises(InferenceContractError, match="Incomplete"):
        load_raw_predictions(output / ARTIFACT_NAMES[0])


def test_run_marker_replaced_after_ownership_check_is_preserved(
    synthetic_final_pipeline, tmp_path, monkeypatch,
):
    run, _, _ = synthetic_final_pipeline
    output = _run_output(tmp_path)
    original_owned = predictor._owned_file
    injected = False

    def swap_marker(path, identity):
        nonlocal injected
        owned = original_owned(path, identity)
        if owned and path == output / ".inference_run" and not injected:
            path.rename(tmp_path / "owned_run_marker")
            path.write_bytes(b"foreign run marker")
            injected = True
        return owned

    monkeypatch.setattr(predictor, "_owned_file", swap_marker)
    with pytest.raises(InferenceContractError, match="Incomplete"):
        run()
    assert injected
    assert (output / ".inference_run").read_bytes() == b"foreign run marker"
    assert (output / ".incomplete").is_file()
    with pytest.raises(InferenceContractError, match="Incomplete"):
        load_raw_predictions(output / ARTIFACT_NAMES[0])


@pytest.mark.parametrize("state", [
    "legacy_incomplete", "pending", "orphan_complete", "mismatched_complete",
    "partial_complete", "complete_but_incomplete", "missing_artifact",
])
def test_prediction_consumer_rejects_unfinished_bundles(tmp_path, state):
    output = tmp_path / "bundle"
    output.mkdir()
    target = save_raw_predictions(output / ARTIFACT_NAMES[0], *_fake_predictions(), exclusive=True)
    for name in ARTIFACT_NAMES[1:]:
        (output / name).write_bytes(b"synthetic companion artifact")
    token = b"a" * 32 + b"\n"
    if state != "legacy_incomplete":
        (output / ".inference_run").write_bytes(token)
    if state not in ("legacy_incomplete", "pending"):
        (output / ".complete").write_bytes(token)
    if state in ("legacy_incomplete", "complete_but_incomplete"):
        (output / ".incomplete").write_bytes(b"failed")
    elif state == "orphan_complete":
        (output / ".inference_run").unlink()
    elif state == "mismatched_complete":
        (output / ".complete").write_bytes(b"b" * 32 + b"\n")
    elif state == "partial_complete":
        (output / ".complete").write_bytes(token[:5])
    elif state == "missing_artifact":
        (output / ARTIFACT_NAMES[-1]).unlink()
    with pytest.raises(InferenceContractError, match="Incomplete"):
        load_raw_predictions(target)


def test_consumer_rechecks_failure_marker_after_npz_read(tmp_path, monkeypatch):
    target = save_raw_predictions(tmp_path / "standalone.npz", *_fake_predictions())
    original_load = predictor.np.load

    def load_and_invalidate(*args, **kwargs):
        archive = original_load(*args, **kwargs)
        (tmp_path / ".incomplete").write_bytes(b"concurrent failure")
        return archive

    monkeypatch.setattr(predictor.np, "load", load_and_invalidate)
    with pytest.raises(InferenceContractError, match="Incomplete"):
        load_raw_predictions(target)


@pytest.mark.parametrize("state", ["incomplete", "pending"])
def test_e7_production_consumer_rejects_incomplete_npz_before_reading_metadata(tmp_path, state):
    from scripts import run_final_selective as selective_runner

    target = save_raw_predictions(tmp_path / "standalone.npz", *_fake_predictions())
    marker = ".incomplete" if state == "incomplete" else ".inference_run"
    (tmp_path / marker).write_bytes(b"a" * 32 + b"\n")
    with pytest.raises(InferenceContractError, match="Incomplete"):
        selective_runner.load_prediction_bundle(
            target, tmp_path / "missing_metadata.json", split="synthetic", repo_root=tmp_path,
        )


def test_legacy_npz_overwrite_mode_is_explicitly_compatible(tmp_path):
    logits, labels, image_ids = _fake_predictions()
    path = save_raw_predictions(tmp_path / "fixture.npz", logits, labels, image_ids)
    changed_logits = logits + 1
    save_raw_predictions(path, changed_logits, labels, image_ids)
    np.testing.assert_array_equal(load_raw_predictions(path)["logits"], changed_logits)
    with pytest.raises(FileExistsError):
        save_raw_predictions(path, logits, labels, image_ids, exclusive=True)
    np.testing.assert_array_equal(load_raw_predictions(path)["logits"], changed_logits)


def test_failure_marker_write_error_preserves_original_error_and_rejects_bundle(
    synthetic_final_pipeline, tmp_path, monkeypatch,
):
    run, _, _ = synthetic_final_pipeline
    output = _run_output(tmp_path)
    exclusive_file = predictor._exclusive_binary_file
    staging_paths = []
    publish = predictor._OutputRun.publish

    def record_staging(transaction):
        staging_paths.append(transaction.staging)
        return publish(transaction)

    def fail_finish(transaction):
        raise OSError("original completion failure")

    def fail_marker(path):
        if path.name == ".incomplete":
            raise PermissionError("cannot create failure marker")
        return exclusive_file(path)

    monkeypatch.setattr(predictor._OutputRun, "publish", record_staging)
    monkeypatch.setattr(predictor._OutputRun, "finish", fail_finish)
    monkeypatch.setattr(predictor, "_exclusive_binary_file", fail_marker)
    with pytest.raises(OSError, match="original completion failure"):
        run()
    assert {path.name for path in output.iterdir()} == {*ARTIFACT_NAMES, ".inference_run"}
    assert staging_paths and all(not path.exists() for path in staging_paths)
    with pytest.raises(InferenceContractError, match="Incomplete"):
        load_raw_predictions(output / ARTIFACT_NAMES[0])
