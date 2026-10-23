"""Synthetic integration tests for the Task 04 final selective runner.

All inputs and artifacts are isolated under pytest's temporary directory. No
Validation predictions, Official Test files, model checkpoints, or images are
opened by this suite.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from scripts import run_final_selective as runner
from src.selective import risk_coverage, threshold


SOURCE_ROOT = Path(__file__).resolve().parents[1]
TEMPERATURE = 0.80246283961825
FROZEN_THRESHOLD = 0.5011672918550012
CHECKPOINT_SHA256 = "73df0cc1c5b16866d4facc910f3e4d3ab481e3f12f1681a59fa7d61e3845c47b"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, n: int = 33, logits: np.ndarray | None = None):
    """Build a synthetic repo root with only approved text reference artifacts."""
    repo = tmp_path / "fixture_repo"
    approved = (
        "configs/data.yaml",
        "configs/model_selection.yaml",
        "configs/experiments/e4_cutmix.yaml",
        "configs/experiments/e6_calibration.yaml",
        "configs/experiments/e7_selective_prediction.yaml",
        "data/metadata/class_mapping.json",
        "outputs/metrics/E6/temperature.json",
        "outputs/metrics/E7/threshold.json",
    )
    for relative in approved:
        source = SOURCE_ROOT / relative
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    ids = np.asarray([f"synthetic/image_{i:03d}.jpg" for i in range(n)], dtype="U64")
    labels = (np.arange(n, dtype=np.int64) % 30)
    mapping = json.loads((repo / "data/metadata/class_mapping.json").read_text(encoding="utf-8"))
    if logits is None:
        # High-margin, all-correct rows, with varied margins so confidence is
        # not tied and may fall above or below the frozen cutoff.
        logits = np.zeros((n, 30), dtype=np.float32)
        logits[np.arange(n), labels] = np.linspace(0.2, 5.0, n, dtype=np.float32)
    logits = np.asarray(logits)
    predictions = tmp_path / "synthetic_predictions.npz"
    np.savez(predictions, logits=logits, labels=labels, image_ids=ids)

    split_csv = tmp_path / "synthetic_split.csv"
    with split_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("relative_path", "class_id", "breed_name", "source_split"))
        writer.writeheader()
        for image_id, label in zip(ids, labels):
            writer.writerow({
                "relative_path": image_id,
                "class_id": int(label),
                "breed_name": mapping[str(int(label))],
                "source_split": "synthetic",
            })

    metadata_path = tmp_path / "synthetic_metadata.json"
    metadata = {
        "experiment_id": "E4_TASK04_FINAL_INFERENCE",
        "split": "synthetic",
        "data_status": "synthetic",
        "synthetic": True,
        "num_samples": n,
        "num_classes": 30,
        "class_mapping_path": "data/metadata/class_mapping.json",
        "class_mapping_sha256": _sha256(repo / "data/metadata/class_mapping.json"),
        "git_sha": "891edf9a77f5f4dcdef35a44fd64f850d1d6a7cc",
        "checkpoint_path": "outputs/checkpoints/E4/seed_42/best.pt",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "temperature_artifact_path": "outputs/metrics/E6/temperature.json",
        "temperature_source_experiment": "E6",
        "temperature_source_split": "calibration",
        "temperature": TEMPERATURE,
        "preprocessing": {
            "config": "configs/data.yaml",
            "split_argument": "synthetic",
            "expected_tensor_shape": [3, 224, 224],
            "evaluation_resize": 256,
            "evaluation_center_crop": 224,
            "normalization": {
                "type": "imagenet",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "runtime": {
            "timestamp_utc": "2026-10-22T00:00:00+00:00",
            "python_version": "3.13.15",
            "pytorch_version": "2.11.0+cpu",
            "torchvision_version": "0.26.0+cpu",
            "device": "cpu",
            "batch_size": 16,
            "num_workers": 0,
        },
        "safety": {
            "official_test_used": False,
            "model_trained_or_updated": False,
            "temperature_refit": False,
        },
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return {
        "repo": repo,
        "predictions": predictions,
        "metadata": metadata_path,
        "csv": split_csv,
        "ids": ids,
        "labels": labels,
        "logits": logits,
        "meta": metadata,
    }


def _load(fixture, *, predictions=None, metadata=None, split_csv=None):
    return runner.load_prediction_bundle(
        predictions or fixture["predictions"],
        metadata or fixture["metadata"],
        split="synthetic",
        split_csv=split_csv or fixture["csv"],
        repo_root=fixture["repo"],
    )


def _readonly_validation_fixture(tmp_path: Path):
    fixture = _fixture(tmp_path, n=300)
    repo = fixture["repo"]
    canonical_csv = repo / "data/splits/validation.csv"
    canonical_csv.parent.mkdir(parents=True, exist_ok=True)
    with fixture["csv"].open("r", encoding="utf-8", newline="") as source, canonical_csv.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row["source_split"] = "official_train"
            writer.writerow(row)

    metadata = dict(fixture["meta"])
    metadata.pop("synthetic")
    metadata.update({
        "split": "validation",
        "data_status": "validation",
        "checkpoint_path": "outputs/checkpoints/E4/seed_42/best.pt",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "temperature_source_experiment": "E6",
        "temperature_source_split": "calibration",
        "preprocessing": {
            "config": "configs/data.yaml",
            "transform_builder": "src.data.transforms.build_transforms",
            "split_argument": "validation",
            "expected_tensor_shape": [3, 224, 224],
            "evaluation_resize": 256,
            "evaluation_center_crop": 224,
            "normalization": {
                "type": "imagenet", "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "runtime": {
            "timestamp_utc": "2026-10-22T00:00:00+00:00",
            "python_version": "3.13.15",
            "pytorch_version": "2.11.0+cpu",
            "torchvision_version": "0.26.0+cpu",
            "device": "cpu",
            "cuda_available": False,
            "batch_size": 16,
            "num_workers": 0,
        },
        "outputs": {"raw_predictions_npz": "outputs/predictions/final_validation/clean_predictions.npz"},
        "safety": {
            "official_test_used": False,
            "model_trained_or_updated": False,
            "temperature_refit": False,
            "official_test_guard": "validation-only runner",
        },
    })
    metadata_path = tmp_path / "validation_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return {**fixture, "metadata": metadata_path, "csv": canonical_csv, "meta": metadata}


def test_reads_valid_synthetic_bundle_and_frozen_artifacts(tmp_path):
    fixture = _fixture(tmp_path)
    bundle = _load(fixture)
    assert bundle["logits"].shape == (33, 30)
    np.testing.assert_array_equal(bundle["labels"], fixture["labels"])
    np.testing.assert_array_equal(bundle["image_ids"], fixture["ids"])
    assert bundle["temperature"] == TEMPERATURE
    assert bundle["threshold"] == FROZEN_THRESHOLD
    assert bundle["input_order_matches_csv"] is True
    assert bundle["mapping_hash_mode"] == "exact"


def test_readonly_validation_bundle_accepts_producer_metadata_and_canonical_temp_csv(tmp_path):
    fixture = _readonly_validation_fixture(tmp_path)
    bundle = runner.load_prediction_bundle(
        fixture["predictions"], fixture["metadata"], split="validation", repo_root=fixture["repo"],
    )
    assert bundle["split"] == "validation"
    assert bundle["logits"].shape == (300, 30)
    assert bundle["metadata"]["preprocessing"]["config"] == "configs/data.yaml"
    assert bundle["upstream_provenance"]["status"] == "historical execution not independently verified"
    with pytest.raises(ValueError, match="canonical repo_root"):
        runner.load_prediction_bundle(
            fixture["predictions"], fixture["metadata"], split="validation",
            split_csv=tmp_path / "arbitrary_validation.csv", repo_root=fixture["repo"],
        )


@pytest.mark.parametrize("mutation", [
    "preprocessing_not_object", "contradictory_model", "wrong_transform_builder",
    "missing_transform_builder", "wrong_split_argument", "missing_split_argument",
])
def test_validation_reader_rejects_inconsistent_producer_metadata(tmp_path, mutation):
    fixture = _readonly_validation_fixture(tmp_path)
    metadata = json.loads(fixture["metadata"].read_text(encoding="utf-8"))
    if mutation == "preprocessing_not_object":
        metadata["preprocessing"] = []
    elif mutation == "contradictory_model":
        metadata["model_experiment"] = "E7"
    elif mutation == "wrong_transform_builder":
        metadata["preprocessing"]["transform_builder"] = "some.other.transform"
    elif mutation == "missing_transform_builder":
        del metadata["preprocessing"]["transform_builder"]
    elif mutation == "wrong_split_argument":
        metadata["preprocessing"]["split_argument"] = "calibration"
    elif mutation == "missing_split_argument":
        del metadata["preprocessing"]["split_argument"]
    fixture["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.load_prediction_bundle(
            fixture["predictions"], fixture["metadata"], split="validation", repo_root=fixture["repo"],
        )


@pytest.mark.parametrize("scenario", ["clean", "missing_head", "untracked", "changed"])
def test_validation_execution_guard_requires_head_tracked_clean_sources(tmp_path, monkeypatch, scenario):
    commit = "a" * 40

    def fake_git(args, **kwargs):
        if "rev-parse" in args:
            if scenario == "missing_head":
                if kwargs.get("check"):
                    raise subprocess.CalledProcessError(128, args)
                return subprocess.CompletedProcess(args, 128, stdout="", stderr="missing HEAD")
            return subprocess.CompletedProcess(args, 0, stdout=commit + "\n", stderr="")
        if "ls-files" in args:
            code = 1 if scenario == "untracked" else 0
            return subprocess.CompletedProcess(args, code, stdout="", stderr="")
        if "diff" in args:
            code = 1 if scenario == "changed" else 0
            return subprocess.CompletedProcess(args, code, stdout="", stderr="")
        pytest.fail(f"unexpected git command: {args}")

    monkeypatch.setattr(runner.subprocess, "run", fake_git)
    bundle = {"repo_root": runner.SOURCE_ROOT, "split": "validation"}
    if scenario == "clean":
        assert runner._execution_guard(bundle, tmp_path / "unused-output", None) is None
    else:
        with pytest.raises(ValueError, match="tracked and clean at HEAD"):
            runner._execution_guard(bundle, tmp_path / "unused-output", None)


@pytest.mark.parametrize("mutation", [
    "missing_field", "extra_field", "object_array", "nan", "inf", "float16_logits",
    "integer_logits", "shape", "label", "int32_labels", "uint64_labels", "duplicate_id",
    "byte_ids", "unsafe_id", "swapped_label", "changed_id", "short_labels", "short_ids",
    "labels_2d", "ids_2d", "empty_n",
])
def test_rejects_invalid_or_misaligned_prediction_bundles(tmp_path, mutation):
    fixture = _fixture(tmp_path)
    with np.load(fixture["predictions"], allow_pickle=False) as original:
        arrays = {key: original[key] for key in original.files}
    if mutation == "missing_field":
        del arrays["image_ids"]
    elif mutation == "extra_field":
        arrays["unexpected"] = np.asarray([1])
    elif mutation == "object_array":
        arrays["image_ids"] = arrays["image_ids"].astype(object)
    elif mutation == "nan":
        arrays["logits"][0, 0] = np.nan
    elif mutation == "inf":
        arrays["logits"][0, 0] = np.inf
    elif mutation == "float16_logits":
        arrays["logits"] = arrays["logits"].astype(np.float16)
    elif mutation == "integer_logits":
        arrays["logits"] = arrays["logits"].astype(np.int32)
    elif mutation == "shape":
        arrays["logits"] = arrays["logits"][:, :29]
    elif mutation == "label":
        arrays["labels"][0] = 30
    elif mutation == "int32_labels":
        arrays["labels"] = arrays["labels"].astype(np.int32)
    elif mutation == "uint64_labels":
        arrays["labels"] = arrays["labels"].astype(np.uint64)
    elif mutation == "duplicate_id":
        arrays["image_ids"][1] = arrays["image_ids"][0]
    elif mutation == "byte_ids":
        arrays["image_ids"] = arrays["image_ids"].astype("S64")
    elif mutation == "unsafe_id":
        arrays["image_ids"][0] = "synthetic/../escape.jpg"
    elif mutation == "swapped_label":
        arrays["labels"][0], arrays["labels"][1] = arrays["labels"][1], arrays["labels"][0]
    elif mutation == "changed_id":
        arrays["image_ids"][0] = "synthetic/not-in-csv.jpg"
    elif mutation == "short_labels":
        arrays["labels"] = arrays["labels"][:-1]
    elif mutation == "short_ids":
        arrays["image_ids"] = arrays["image_ids"][:-1]
    elif mutation == "labels_2d":
        arrays["labels"] = arrays["labels"][:, None]
    elif mutation == "ids_2d":
        arrays["image_ids"] = arrays["image_ids"][:, None]
    elif mutation == "empty_n":
        arrays = {
            "logits": np.empty((0, 30), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int64),
            "image_ids": np.empty((0,), dtype="U1"),
        }
    bad = tmp_path / "bad.npz"
    np.savez(bad, **arrays)
    with pytest.raises(ValueError):
        _load(fixture, predictions=bad)


@pytest.mark.parametrize("malformation", ["npy", "corrupt_zip", "raw_zip_members"])
def test_malformed_prediction_containers_fail_cleanly_in_api_and_cli(tmp_path, malformation):
    fixture = _fixture(tmp_path)
    suffix = ".npy" if malformation == "npy" else ".npz"
    bad = tmp_path / f"malformed_{malformation}{suffix}"
    if malformation == "npy":
        with bad.open("wb") as handle:
            np.save(handle, np.zeros((33, 30), dtype=np.float32))
    elif malformation == "corrupt_zip":
        bad.write_bytes(bytes.fromhex("504b030462726f6b656e"))
    else:
        with zipfile.ZipFile(bad, "w") as archive:
            archive.writestr("logits", b"not an NPY array")
            archive.writestr("labels", b"not an NPY array")
            archive.writestr("image_ids", b"not an NPY array")

    with pytest.raises(ValueError, match="prediction NPZ|NPZ archive"):
        _load(fixture, predictions=bad)

    output = tmp_path / f"malformed_{malformation}_output"
    command = [
        sys.executable, str(SOURCE_ROOT / "scripts/run_final_selective.py"),
        "--predictions", str(bad), "--metadata", str(fixture["metadata"]),
        "--output-dir", str(output), "--split", "synthetic",
        "--split-csv", str(fixture["csv"]), "--repo-root", str(fixture["repo"]),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 2
    assert "run_final_selective:" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not output.exists()


def test_accepts_float64_logits_without_downcasting(tmp_path):
    logits = np.zeros((33, 30), dtype=np.float64)
    fixture = _fixture(tmp_path, logits=logits)
    bundle = _load(fixture)
    assert bundle["logits"].dtype == np.dtype("float64")
    np.testing.assert_array_equal(bundle["logits"], logits)


def test_inference_writer_predictions_feed_selective_runner_with_alignment_and_dtype(tmp_path):
    from src.inference.predictor import load_raw_predictions, save_raw_predictions

    margins = np.asarray([6, 6, 4, 4, 1, 1], dtype=np.float64)
    logits = np.zeros((6, 30), dtype=np.float64)
    logits[:, 0] = margins
    fixture = _fixture(tmp_path, n=6, logits=logits.astype(np.float32))
    permutation = np.asarray([5, 2, 0, 4, 1, 3])
    saved_path = save_raw_predictions(
        fixture["predictions"],
        logits[permutation],
        fixture["labels"][permutation],
        fixture["ids"][permutation],
    )
    upstream = load_raw_predictions(saved_path)
    assert upstream["logits"].dtype == np.dtype("float64")
    np.testing.assert_array_equal(upstream["image_ids"], fixture["ids"][permutation])

    bundle = _load(fixture)
    assert bundle["input_order_matches_csv"] is False
    np.testing.assert_array_equal(bundle["image_ids"], fixture["ids"])
    np.testing.assert_array_equal(bundle["labels"], fixture["labels"])
    np.testing.assert_array_equal(bundle["logits"], logits)
    assert bundle["logits"].dtype == np.dtype("float64")

    result = runner.run_final_selective(
        saved_path, fixture["metadata"], tmp_path / "upstream-writer-output",
        split="synthetic", split_csv=fixture["csv"], repo_root=fixture["repo"],
    )
    assert result["n_total"] == 6
    assert result["selected_metrics"]["n_accept"] == 4
    assert result["selected_metrics"]["coverage"] == pytest.approx(2 / 3)
    assert result["selected_metrics"]["selective_accuracy"] == pytest.approx(1 / 4)
    assert result["selected_metrics"]["selective_risk"] == pytest.approx(3 / 4)
    assert result["aurc"] == pytest.approx(25 / 36)
    assert all(Path(path).is_file() for path in result["output_paths"].values())


def test_row_permutation_preserves_selected_metrics(tmp_path):
    fixture = _fixture(tmp_path)
    baseline = runner.run_final_selective(
        fixture["predictions"], fixture["metadata"], tmp_path / "out_a",
        split="synthetic", split_csv=fixture["csv"], repo_root=fixture["repo"],
    )
    permutation = np.random.default_rng(902).permutation(len(fixture["ids"]))
    permuted = tmp_path / "permuted.npz"
    np.savez(permuted,
             logits=fixture["logits"][permutation],
             labels=fixture["labels"][permutation],
             image_ids=fixture["ids"][permutation])
    permuted_result = runner.run_final_selective(
        permuted, fixture["metadata"], tmp_path / "out_b",
        split="synthetic", split_csv=fixture["csv"], repo_root=fixture["repo"],
    )
    for section in ("selected_metrics", "aurc", "aurc_convention"):
        assert baseline[section] == permuted_result[section]
    assert Path(baseline["output_paths"]["risk_coverage_csv"]).read_bytes() == Path(
        permuted_result["output_paths"]["risk_coverage_csv"]
    ).read_bytes()


def test_runner_uses_frozen_cutoff_without_selecting_new_threshold(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, n=33)
    def forbidden_selection(*args, **kwargs):
        pytest.fail("final selective evaluation must not select a new threshold")

    monkeypatch.setattr(threshold, "select_threshold", forbidden_selection)
    monkeypatch.setattr(runner, "select_threshold", forbidden_selection, raising=False)
    result = runner.run_final_selective(
        fixture["predictions"], fixture["metadata"], tmp_path / "out",
        split="synthetic", split_csv=fixture["csv"], repo_root=fixture["repo"],
    )
    assert result["threshold"] == FROZEN_THRESHOLD


def test_runner_accepts_confidence_exactly_equal_to_threshold(tmp_path, monkeypatch):
    logits = np.zeros((1, 30), dtype=np.float32)
    logits[0, 0] = 8.0
    fixture = _fixture(tmp_path, n=1, logits=logits)
    original_loader = runner.load_prediction_bundle
    confidence = {}

    def set_exact_threshold(*args, **kwargs):
        bundle = original_loader(*args, **kwargs)
        confidence["value"] = float(runner.calibrated_predictions(
            bundle["logits"], bundle["temperature"]
        )["confidences"][0])
        bundle["threshold"] = confidence["value"]
        return bundle

    monkeypatch.setattr(runner, "load_prediction_bundle", set_exact_threshold)
    result = runner.run_final_selective(
        fixture["predictions"], fixture["metadata"], tmp_path / "inclusive-boundary-output",
        split="synthetic", split_csv=fixture["csv"], repo_root=fixture["repo"],
    )
    assert result["threshold"] == confidence["value"]
    assert result["selected_metrics"]["n_accept"] == 1


@pytest.mark.parametrize("mode, expected_accept", [("uniform", 0), ("confident", 5)])
def test_runner_handles_no_accept_and_all_accept_without_dropping_batch_tail(tmp_path, mode, expected_accept):
    n = 5  # deliberately not a multiple of common inference batch sizes
    logits = np.zeros((n, 30), dtype=np.float32)
    if mode == "confident":
        logits[np.arange(n), np.arange(n)] = 20.0
    fixture = _fixture(tmp_path, n=n, logits=logits)
    # All rows in this fixture have matching ground-truth top-1 labels.
    if mode == "uniform":
        fixture["labels"][:] = 0  # deterministic argmax tie break
        # Update both prediction labels and CSV labels for the aligned fixture.
        np.savez(fixture["predictions"], logits=logits, labels=fixture["labels"], image_ids=fixture["ids"])
        with fixture["csv"].open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with fixture["csv"].open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("relative_path", "class_id", "breed_name", "source_split"))
            writer.writeheader()
            for row in rows:
                row["class_id"] = "0"
                row["breed_name"] = json.loads((fixture["repo"] / "data/metadata/class_mapping.json").read_text())["0"]
                writer.writerow(row)
    result = runner.run_final_selective(
        fixture["predictions"], fixture["metadata"], tmp_path / f"out_{mode}",
        split="synthetic", split_csv=fixture["csv"], repo_root=fixture["repo"],
    )
    assert result["n_total"] == n
    assert result["selected_metrics"]["n_accept"] == expected_accept
    if expected_accept == 0:
        assert result["selected_metrics"]["selective_accuracy"] is None
        assert result["selected_metrics"]["selective_risk"] is None
    else:
        assert result["selected_metrics"]["n_reject"] == 0
        assert result["selected_metrics"]["selective_accuracy"] == 1.0


def test_run_outputs_are_synthetic_marked_and_do_not_modify_inputs_or_frozen_artifacts(tmp_path):
    fixture = _fixture(tmp_path)
    tracked = [fixture["predictions"], fixture["metadata"], fixture["csv"],
               fixture["repo"] / "outputs/metrics/E6/temperature.json",
               fixture["repo"] / "outputs/metrics/E7/threshold.json"]
    before = {path: path.read_bytes() for path in tracked}
    result = runner.run_final_selective(
        fixture["predictions"], fixture["metadata"], tmp_path / "generated",
        split="synthetic", split_csv=fixture["csv"], repo_root=fixture["repo"],
    )
    assert {path: path.read_bytes() for path in tracked} == before
    payload = json.loads(Path(result["output_paths"]["metrics"]).read_text(encoding="utf-8"))
    assert payload["synthetic"] is True
    assert payload["official_final_evaluation"] is False
    assert payload["threshold_reselected"] is False
    assert payload["runtime"]["inference_performed"] is False
    assert payload["upstream_provenance"]["status"].startswith("synthetic fixture")
    assert payload["source_commit_status"] == "source checkout HEAD recorded; tracked clean unchecked"
    assert payload["source_git_sha"] == runner._git_status(SOURCE_ROOT)[0]
    csv_text = Path(result["output_paths"]["risk_coverage_csv"]).read_text(encoding="utf-8")
    assert "n_accept, n_reject" not in csv_text  # conventional compact header
    assert csv_text.splitlines()[0] == "n_total,n_accept,n_reject,n_correct_accept,coverage,selective_accuracy,selective_risk,threshold"
    assert "" in csv_text.splitlines()[1].split(",")  # origin selective accuracy/risk are null
    png = Path(result["output_paths"]["risk_coverage_curve"]).read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", png[16:24])
    assert width > 0 and height > 0


def test_synthetic_csv_order_is_restored_and_mapping_newline_hash_is_accepted(tmp_path):
    fixture = _fixture(tmp_path)
    with np.load(fixture["predictions"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    permutation = np.arange(len(fixture["ids"]))[::-1]
    permuted = tmp_path / "reverse.npz"
    np.savez(permuted, logits=arrays["logits"][permutation], labels=arrays["labels"][permutation], image_ids=arrays["image_ids"][permutation])
    bundle = _load(fixture, predictions=permuted)
    assert bundle["input_order_matches_csv"] is False
    np.testing.assert_array_equal(bundle["image_ids"], fixture["ids"])
    mapping_path = fixture["repo"] / "data/metadata/class_mapping.json"
    mapping_text = mapping_path.read_text(encoding="utf-8")
    crlf_hash = hashlib.sha256(mapping_text.replace("\n", "\r\n").encode("utf-8")).hexdigest()
    metadata = dict(fixture["meta"], class_mapping_sha256=crlf_hash)
    altered_metadata = tmp_path / "crlf_metadata.json"
    altered_metadata.write_text(json.dumps(metadata), encoding="utf-8")
    assert _load(fixture, metadata=altered_metadata)["mapping_hash_mode"] == "newline_variant_crlf"
    metadata["class_mapping_sha256"] = "0" * 64
    altered_metadata.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="class mapping hash"):
        _load(fixture, metadata=altered_metadata)


def test_crlf_reference_files_accept_lf_counterpart_hashes_without_changing_values(tmp_path):
    fixture = _fixture(tmp_path)
    for relative in (
        "configs/data.yaml", "configs/model_selection.yaml",
        "configs/experiments/e4_cutmix.yaml", "configs/experiments/e6_calibration.yaml",
        "configs/experiments/e7_selective_prediction.yaml",
    ):
        path = fixture["repo"] / relative
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))

    temperature_path = fixture["repo"] / "outputs/metrics/E6/temperature.json"
    temperature_path.write_bytes(temperature_path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    mapping_path = fixture["repo"] / "data/metadata/class_mapping.json"
    mapping_lf = mapping_path.read_bytes()
    mapping_path.write_bytes(mapping_lf.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    # Metadata records the digest of the semantically identical LF mapping.
    fixture["meta"]["class_mapping_sha256"] = hashlib.sha256(mapping_lf).hexdigest()
    fixture["metadata"].write_text(json.dumps(fixture["meta"]), encoding="utf-8")

    bundle = _load(fixture)
    assert bundle["temperature"] == TEMPERATURE
    assert bundle["threshold"] == FROZEN_THRESHOLD
    assert bundle["mapping_hash_mode"] in {"newline_variant_lf", "newline_variant"}
    assert bundle["temperature_hash_mode"] in {"newline_variant_lf", "newline_variant"}

    temperature_payload = json.loads(temperature_path.read_text(encoding="utf-8"))
    temperature_payload["temperature"] = TEMPERATURE + 0.01
    temperature_path.write_text(json.dumps(temperature_payload), encoding="utf-8", newline="\r\n")
    with pytest.raises(ValueError, match="referenced bytes|Temperature|temperature"):
        _load(fixture)


def test_official_test_split_is_rejected_before_any_input_io(tmp_path, monkeypatch):
    def fail_io(*args, **kwargs):
        pytest.fail("official-test rejection must happen before input I/O")

    monkeypatch.setattr(Path, "open", fail_io)
    monkeypatch.setattr(np, "load", fail_io)
    with pytest.raises(ValueError, match="split must be"):
        runner.run_final_selective(
            tmp_path / "official_test" / "predictions.npz",
            tmp_path / "metadata.json", tmp_path / "out", split="official_test",
        )


def test_blocked_path_variants_and_symlink_inputs_are_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    blocked_path = tmp_path / "official_test_archive" / "predictions.npz"
    with pytest.raises(ValueError, match="official_test"):
        _load(fixture, predictions=blocked_path)
    symlink = tmp_path / "predictions_link.npz"
    try:
        symlink.symlink_to(fixture["predictions"])
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    with pytest.raises(ValueError, match="symlink"):
        _load(fixture, predictions=symlink)

    with pytest.raises(ValueError, match="official_test"):
        runner.run_final_selective(
            fixture["predictions"], fixture["metadata"], tmp_path / "official_test_output",
            split="synthetic", split_csv=fixture["csv"], repo_root=fixture["repo"],
        )


@pytest.mark.parametrize("relative", [
    "outputs/metrics/E6/temperature.json",
    "data/metadata/class_mapping.json",
])
def test_symlinked_frozen_references_are_rejected_before_target_read(tmp_path, monkeypatch, relative):
    fixture = _fixture(tmp_path)
    target = fixture["repo"] / relative
    original = target.read_bytes()
    target.unlink()
    external = tmp_path / f"external_{target.name}"
    external.write_bytes(original)
    target.symlink_to(external)
    original_read_bytes = Path.read_bytes
    reads = []

    def record_read(path):
        if path == target:
            reads.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record_read)
    with pytest.raises(ValueError, match="symlink"):
        _load(fixture)
    assert reads == []


def test_outputs_cannot_replace_existing_or_frozen_locations(tmp_path):
    fixture = _fixture(tmp_path)
    existing = tmp_path / "already-there"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        runner.run_final_selective(fixture["predictions"], fixture["metadata"], existing,
                                  split="synthetic", split_csv=fixture["csv"], repo_root=fixture["repo"])
    protected = fixture["repo"] / "outputs/metrics/final"
    with pytest.raises(ValueError, match="already exists|collides"):
        runner.run_final_selective(fixture["predictions"], fixture["metadata"], protected,
                                  split="synthetic", split_csv=fixture["csv"], repo_root=fixture["repo"])


def test_partial_output_write_failure_rolls_back_new_files(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    output = tmp_path / "write-failure"
    original_open = Path.open

    class PartialWriter:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self.handle.__exit__(*exc)

        def write(self, data):
            self.handle.write(data[:8])
            raise OSError("injected output failure after partial write")

    def fail_on_figure(path, *args, **kwargs):
        if path.name == "risk_coverage_curve.png" and args and args[0] == "xb":
            return PartialWriter(original_open(path, *args, **kwargs))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_on_figure)
    with pytest.raises(OSError, match="injected output failure after partial write"):
        runner.run_final_selective(fixture["predictions"], fixture["metadata"], output,
                                   split="synthetic", split_csv=fixture["csv"], repo_root=fixture["repo"])
    assert not output.exists()


def test_output_directory_race_does_not_remove_foreign_empty_directory(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    output = tmp_path / "raced-output"
    original_mkdir = Path.mkdir

    def race_mkdir(path, *args, **kwargs):
        if path == output:
            original_mkdir(path, *args, **kwargs)
            # Simulate another process winning the directory creation between
            # preflight and this runner's mkdir(exist_ok=False).
            return original_mkdir(path, *args, **kwargs)
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", race_mkdir)
    with pytest.raises(FileExistsError):
        runner.run_final_selective(fixture["predictions"], fixture["metadata"], output,
                                   split="synthetic", split_csv=fixture["csv"], repo_root=fixture["repo"])
    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_risk_coverage_plot_uses_grouped_tie_vertices(monkeypatch):
    import matplotlib.axes

    captured = {}
    original_stairs = matplotlib.axes.Axes.stairs

    def capture_stairs(axis, values, edges=None, *args, **kwargs):
        captured["edges"] = np.asarray(edges, dtype=np.float64)
        captured["values"] = np.asarray(values, dtype=np.float64)
        return original_stairs(axis, values, edges=edges, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "stairs", capture_stairs)
    rows = risk_coverage.risk_coverage_curve(
        np.asarray([0.9, 0.8, 0.8, 0.2]),
        np.asarray([True, True, False, False]),
    )
    png = runner._curve_png(rows, "synthetic")
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    np.testing.assert_allclose(captured["edges"], [0.0, 0.25, 0.75, 1.0])
    np.testing.assert_allclose(captured["values"], [0.0, 1 / 3, 0.5])


def test_cli_runs_synthetic_fixture_from_unrelated_working_directory(tmp_path):
    fixture = _fixture(tmp_path)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    command = [
        sys.executable, str(SOURCE_ROOT / "scripts/run_final_selective.py"),
        "--predictions", str(fixture["predictions"]), "--metadata", str(fixture["metadata"]),
        "--output-dir", str(tmp_path / "cli-output"), "--split", "synthetic",
        "--split-csv", str(fixture["csv"]), "--repo-root", str(fixture["repo"]),
    ]
    completed = subprocess.run(command, cwd=unrelated, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "SYNTHETIC Task 04 preparation complete" in completed.stdout


def test_end_to_end_mixed_errors_ties_and_aurc_match_hand_calculation(tmp_path):
    margins = np.asarray([6, 6, 4, 4, 1, 1], dtype=np.float32)
    logits = np.zeros((len(margins), 30), dtype=np.float32)
    logits[:, 0] = margins
    fixture = _fixture(tmp_path, n=len(margins), logits=logits)
    # Labels remain [0,1,2,3,4,5] from the synthetic CSV; class 0 is top-1
    # for every row. The locked cutoff accepts both margin-6 and margin-4 ties.
    result = runner.run_final_selective(
        fixture["predictions"], fixture["metadata"], tmp_path / "mixed-errors-output",
        split="synthetic", split_csv=fixture["csv"], repo_root=fixture["repo"],
    )
    selected = result["selected_metrics"]
    assert result["threshold"] == FROZEN_THRESHOLD
    assert selected["n_total"] == 6
    assert selected["n_accept"] == 4
    assert selected["n_reject"] == 2
    assert selected["coverage"] == pytest.approx(2 / 3)  # new-data coverage is not forced to 90%
    assert selected["n_correct_accept"] == 1
    assert selected["selective_accuracy"] == pytest.approx(1 / 4)
    assert selected["selective_risk"] == pytest.approx(3 / 4)
    # Three confidence tie groups have post-group risks 1/2, 3/4, 5/6.
    assert result["aurc"] == pytest.approx(25 / 36)
    with Path(result["output_paths"]["risk_coverage_csv"]).open(encoding="utf-8") as handle:
        curve_rows = list(csv.DictReader(handle))
    assert len(curve_rows) == 4  # zero boundary plus three complete tie groups
    assert curve_rows[0]["coverage"] == "0.0"
    assert curve_rows[0]["selective_accuracy"] == "" and curve_rows[0]["selective_risk"] == ""
    assert [float(row["coverage"]) for row in curve_rows[1:]] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert [float(row["selective_risk"]) for row in curve_rows[1:]] == pytest.approx([1 / 2, 3 / 4, 5 / 6])
