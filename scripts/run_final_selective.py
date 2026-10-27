"""Prepare E7 selective-prediction metrics from a Task 04 prediction bundle.

This runner consumes raw predictions only. It never fits temperature or selects
an E7 threshold; both values are read from the frozen E6/E7 artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from src.selective.config import load_e7_config
from src.selective.threshold import accept_mask, calibrated_predictions
from src.selective.risk_coverage import compute_aurc, risk_coverage_curve, selective_metrics
from src.inference.predictor import assert_prediction_bundle_complete

OFFICIAL_CHECKPOINT_SHA256 = "73df0cc1c5b16866d4facc910f3e4d3ab481e3f12f1681a59fa7d61e3845c47b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _newline_hash_mode(recorded_hash: Any, raw: bytes, name: str) -> str:
    if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(recorded_hash)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from exc
    if hashlib.sha256(raw).hexdigest() == recorded_hash.lower():
        return "exact"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} bytes are not UTF-8") from exc
    normalized = text.replace("\r\n", "\n")
    candidates = {normalized.encode("utf-8"), normalized.replace("\n", "\r\n").encode("utf-8")}
    for candidate in candidates:
        if hashlib.sha256(candidate).hexdigest() == recorded_hash.lower():
            return "newline_variant_lf" if b"\r\n" not in candidate else "newline_variant_crlf"
    raise ValueError(f"{name} does not match the referenced bytes or an LF/CRLF-only variant")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _reject_unsafe_path(path: Path, label: str) -> None:
    lexical = str(path).lower()
    if "official_test" in lexical:
        raise ValueError(f"{label} path is forbidden because it contains 'official_test'")
    # Check every existing component before resolving, so symlinks cannot hide
    # a forbidden location or redirect a path after the lexical check.
    absolute = path if path.is_absolute() else Path.cwd() / path
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} path contains a symlink component: {cursor}")
    if "official_test" in str(path.resolve(strict=False)).lower():
        raise ValueError(f"{label} resolved path is forbidden because it contains 'official_test'")


def _safe_id(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        return False
    if "\\" in value or value.startswith("/") or "\x00" in value or ":" in value:
        return False
    parts = value.split("/")
    return all(part not in ("", ".", "..") for part in parts)


def _mapping(root: Path) -> tuple[dict[int, str], str, str]:
    path = root / "data/metadata/class_mapping.json"
    _reject_unsafe_path(path, "class mapping reference")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate class mapping key: {key}")
            result[key] = value
        return result
    try:
        raw = path.read_bytes()
        mapping_raw = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read class mapping {path}: {exc}") from exc
    if not isinstance(mapping_raw, dict) or len(mapping_raw) != 30:
        raise ValueError("class mapping must contain exactly 30 classes")
    if set(mapping_raw) != {str(index) for index in range(30)}:
        raise ValueError("class mapping keys must be the unique strings '0' through '29'")
    mapping = {int(key): value for key, value in mapping_raw.items()}
    if set(mapping) != set(range(30)) or any(not isinstance(v, str) or not v.strip() for v in mapping.values()) or len(set(mapping.values())) != 30:
        raise ValueError("class mapping must map every class ID 0..29 to a name")
    return mapping, hashlib.sha256(raw).hexdigest(), raw.decode("utf-8")


def _mapping_hash_mode(root: Path, metadata_hash: Any) -> str:
    if not isinstance(metadata_hash, str) or len(metadata_hash) != 64:
        raise ValueError("metadata.class_mapping_sha256 must be a SHA-256 hex digest")
    mapping_path = root / "data/metadata/class_mapping.json"
    raw = mapping_path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    wanted = metadata_hash.lower()
    if wanted == actual:
        return "exact"
    text = raw.decode("utf-8")
    lf_text = text.replace("\r\n", "\n")
    variants = {lf_text.encode("utf-8"), lf_text.replace("\n", "\r\n").encode("utf-8")}
    for candidate in variants:
        if hashlib.sha256(candidate).hexdigest() == wanted:
            return "newline_variant_lf" if b"\r\n" not in candidate else "newline_variant_crlf"
    raise ValueError("metadata class mapping hash does not match the repository mapping or its newline variant")


def _csv_rows(path: Path, mapping: dict[int, str], split: str) -> list[tuple[str, int]]:
    required = {"relative_path", "class_id", "breed_name", "source_split"}
    rows: list[tuple[str, int]] = []
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("split CSV must include relative_path,class_id,breed_name,source_split")
            for line, row in enumerate(reader, 2):
                image_id = row.get("relative_path")
                if not _safe_id(image_id):
                    raise ValueError(f"split CSV row {line} has an unsafe image ID")
                if image_id in seen:
                    raise ValueError(f"split CSV contains duplicate image ID: {image_id}")
                seen.add(image_id)
                try:
                    label = int(row["class_id"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"split CSV row {line} has an invalid class_id") from exc
                if str(label) != row["class_id"].strip() or label not in mapping:
                    raise ValueError(f"split CSV row {line} class_id must be an integer from 0 to 29")
                if row["breed_name"] != mapping[label]:
                    raise ValueError(f"split CSV row {line} breed_name disagrees with class mapping")
                expected_source = "official_train" if split == "validation" else "synthetic"
                if row["source_split"] != expected_source:
                    raise ValueError(f"split CSV row {line} source_split must be {expected_source!r}")
                rows.append((image_id, label))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read split CSV {path}: {exc}") from exc
    if not rows or (split == "validation" and len(rows) != 300):
        raise ValueError("split CSV row count is invalid for the requested split")
    return rows


def _load_references(root: Path) -> dict[str, Any]:
    reference_paths = [
        root / "configs/experiments/e7_selective_prediction.yaml",
        root / "configs/model_selection.yaml", root / "configs/data.yaml",
        root / "configs/experiments/e4_cutmix.yaml", root / "configs/experiments/e6_calibration.yaml",
        root / "outputs/metrics/E6/temperature.json", root / "outputs/metrics/E7/threshold.json",
    ]
    for path in reference_paths:
        _reject_unsafe_path(path, "frozen reference")
    cfg_path = root / "configs/experiments/e7_selective_prediction.yaml"
    try:
        config = load_e7_config(cfg_path)
        raw_config = cfg_path.read_bytes()
        selection = yaml.safe_load((root / "configs/model_selection.yaml").read_text(encoding="utf-8"))
        data_config = yaml.safe_load((root / "configs/data.yaml").read_text(encoding="utf-8"))
        e4_config = yaml.safe_load((root / "configs/experiments/e4_cutmix.yaml").read_text(encoding="utf-8"))
        e6_config = yaml.safe_load((root / "configs/experiments/e6_calibration.yaml").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"Could not load frozen configuration references: {exc}") from exc
    if any(not isinstance(value, dict) for value in (selection, data_config, e4_config, e6_config)):
        raise ValueError("frozen YAML references must have object roots")
    for parent, key in ((selection, "selection"), (selection, "model"), (selection, "data"),
                        (data_config, "data"), (data_config, "preprocessing"),
                        (e4_config, "experiment"), (e4_config, "model"), (e6_config, "experiment")):
        if not isinstance(parent.get(key), dict):
            raise ValueError(f"frozen YAML section {key!r} must be an object")
    if (selection.get("selection", {}).get("status") != "approved"
            or selection.get("selection", {}).get("selected_experiment") != "E4"
            or selection.get("selection", {}).get("selected_seed") != 42):
        raise ValueError("frozen model selection must be E4 seed 42")
    if (selection.get("model", {}).get("architecture") != "mobilenet_v2"
            or selection.get("model", {}).get("num_classes") != 30
            or selection.get("model", {}).get("checkpoint_path") != "outputs/checkpoints/E4/seed_42/best.pt"):
        raise ValueError("frozen model selection must be MobileNetV2 with 30 classes")
    if e4_config.get("experiment", {}).get("id") != "E4" or e4_config.get("experiment", {}).get("seed") != 42:
        raise ValueError("E4 config must identify the selected seed 42 experiment")
    if e6_config.get("experiment", {}).get("id") != "E6":
        raise ValueError("E6 config must identify E6")
    if data_config.get("data", {}).get("num_classes") != 30 or e4_config.get("model", {}).get("num_classes") != 30:
        raise ValueError("shared data and E4 configs must declare 30 classes")
    preprocess = data_config.get("preprocessing", {})
    if not isinstance(preprocess.get("evaluation"), dict) or not isinstance(preprocess.get("normalization"), dict):
        raise ValueError("shared preprocessing evaluation and normalization sections must be objects")
    if preprocess.get("evaluation", {}).get("resize") != 256 or preprocess.get("evaluation", {}).get("center_crop") != 224:
        raise ValueError("shared evaluation preprocessing must be Resize(256), CenterCrop(224)")
    norm = preprocess.get("normalization", {})
    if norm != {"type": "imagenet", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}:
        raise ValueError("shared ImageNet normalization does not match the approved values")
    temp_path = root / config["calibration"]["temperature_file"]
    threshold_path = root / config["outputs"]["threshold"]
    _reject_unsafe_path(temp_path, "E6 temperature artifact")
    _reject_unsafe_path(threshold_path, "E7 threshold artifact")
    temp_artifact = _json(temp_path)
    threshold_artifact = _json(threshold_path)
    temperature = temp_artifact.get("temperature")
    threshold = threshold_artifact.get("threshold")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("E6 artifact temperature must be positive and finite")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("E7 artifact threshold must be finite and in [0,1]")
    if (temp_artifact.get("experiment") != "E6" or temp_artifact.get("model_experiment") != "E4"
            or temp_artifact.get("synthetic", False) is not False
            or temp_artifact.get("checkpoint_seed") != 42 or temp_artifact.get("split") != "calibration"):
        raise ValueError("E6 artifact does not identify E4 seed 42 Calibration")
    if (threshold_artifact.get("experiment") != "E7" or threshold_artifact.get("split") != "calibration"
            or threshold_artifact.get("synthetic") is not False
            or threshold_artifact.get("accept_operator") != ">="
            or threshold_artifact.get("tie_policy") != "accept_all_equal_to_threshold"
            or threshold_artifact.get("selection_rule") != "highest_feasible_threshold"):
        raise ValueError("E7 artifact is not the approved nonsynthetic Calibration threshold")
    threshold_n_total = threshold_artifact.get("n_total")
    if isinstance(threshold_n_total, bool) or not isinstance(threshold_n_total, int) or threshold_n_total < 1:
        raise ValueError("E7 threshold artifact n_total must be a positive integer")
    if threshold_artifact.get("target_coverage") != 0.90:
        raise ValueError("E7 threshold artifact does not record the approved Calibration selection context")
    if threshold_artifact.get("temperature") != temperature:
        raise ValueError("E7 artifact temperature differs from the E6 artifact")
    threshold_hash_inputs = threshold_artifact.get("input_sha256")
    if not isinstance(threshold_hash_inputs, dict):
        raise ValueError("E7 threshold artifact input_sha256 must be an object")
    recorded_t_hash = threshold_hash_inputs.get("e6_temperature")
    temperature_hash_mode = _newline_hash_mode(recorded_t_hash, temp_path.read_bytes(), "E7 E6-temperature input hash")
    return {
        "config": config, "config_path": cfg_path, "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "temperature": float(temperature), "threshold": float(threshold),
        "temperature_path": temp_path, "threshold_path": threshold_path,
        "temperature_sha256": _sha256(temp_path), "threshold_sha256": _sha256(threshold_path),
        "temperature_hash_mode": temperature_hash_mode,
        "data_config_path": root / "configs/data.yaml", "data_config_sha256": _sha256(root / "configs/data.yaml"),
        "selection_config_path": root / "configs/model_selection.yaml", "selection_config_sha256": _sha256(root / "configs/model_selection.yaml"),
        "e4_config_path": root / "configs/experiments/e4_cutmix.yaml", "e4_config_sha256": _sha256(root / "configs/experiments/e4_cutmix.yaml"),
        "e6_config_path": root / "configs/experiments/e6_calibration.yaml", "e6_config_sha256": _sha256(root / "configs/experiments/e6_calibration.yaml"),
    }


def load_prediction_bundle(predictions_path, metadata_path, *, split="validation", split_csv=None, repo_root=None) -> dict[str, Any]:
    """Read and validate a prediction bundle, aligning rows to its approved CSV."""
    if split not in {"validation", "synthetic"}:
        raise ValueError("split must be 'validation' or 'synthetic'")
    predictions_path, metadata_path = Path(predictions_path), Path(metadata_path)
    csv_path = Path(split_csv) if split_csv is not None else None
    for path, label in ((predictions_path, "predictions"), (metadata_path, "metadata")):
        _reject_unsafe_path(path, label)
        assert_prediction_bundle_complete(path)
    if csv_path is not None:
        _reject_unsafe_path(csv_path, "split CSV")
    root = Path(repo_root).resolve() if repo_root is not None else SOURCE_ROOT
    if repo_root is not None:
        _reject_unsafe_path(Path(repo_root), "repo root")
    if "official_test" in str(root).lower():
        raise ValueError("repo_root is forbidden because it contains 'official_test'")
    metadata = _json(metadata_path)
    if metadata.get("split") != split or metadata.get("data_status") != split:
        raise ValueError("metadata split and data_status must match the requested split")
    if metadata.get("experiment_id") != "E4_TASK04_FINAL_INFERENCE":
        raise ValueError("metadata experiment_id must identify the approved E4 Task 04 inference bundle")
    if "model_experiment" in metadata and metadata["model_experiment"] != "E4":
        raise ValueError("metadata model_experiment must identify E4")
    if split == "synthetic":
        if metadata.get("synthetic") is not True:
            raise ValueError("synthetic metadata must explicitly set synthetic=true")
        if root == SOURCE_ROOT:
            # Read-only tests may validate a fixture under a temp reference root;
            # synthetic execution has a stricter check in run_final_selective.
            raise ValueError("synthetic bundle validation requires an alternate fixture/reference root")
    elif metadata.get("synthetic", False) is not False:
        raise ValueError("validation metadata must not be marked synthetic")
    expected_csv = root / "data/splits/validation.csv" if split == "validation" else csv_path
    if split == "validation":
        if csv_path is not None and csv_path.resolve() != expected_csv.resolve():
            raise ValueError("validation split CSV must be the canonical repo_root/data/splits/validation.csv")
        csv_path = expected_csv
    elif csv_path is None:
        raise ValueError("synthetic split requires --split-csv")
    assert csv_path is not None
    _reject_unsafe_path(csv_path, "split CSV")
    mapping, mapping_hash, _ = _mapping(root)
    mapping_mode = _mapping_hash_mode(root, metadata.get("class_mapping_sha256"))
    csv_rows = _csv_rows(csv_path, mapping, split)
    try:
        with predictions_path.open("rb") as input_handle:
            archive = np.load(input_handle, allow_pickle=False)
            if not isinstance(archive, np.lib.npyio.NpzFile):
                raise ValueError("prediction file must be an NPZ archive")
            with archive:
                if set(archive.files) != {"logits", "labels", "image_ids"} or len(archive.files) != 3:
                    raise ValueError("prediction NPZ must contain exactly logits, labels, and image_ids")
                logits, labels, image_ids = archive["logits"], archive["labels"], archive["image_ids"]
                if not all(isinstance(array, np.ndarray) for array in (logits, labels, image_ids)):
                    raise ValueError("prediction NPZ entries must be NumPy arrays")
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("prediction NPZ"):
            raise
        raise ValueError(f"Could not read prediction NPZ {predictions_path}: {exc}") from exc
    if logits.dtype.kind != "f" or logits.dtype not in (np.dtype("float32"), np.dtype("float64")) or logits.ndim != 2 or logits.shape[1] != 30:
        raise ValueError("logits must be float32 or float64 with shape [N,30]")
    if labels.dtype != np.dtype("int64") or labels.ndim != 1:
        raise ValueError("labels must be signed int64 with shape [N]")
    if image_ids.dtype.kind != "U" or image_ids.ndim != 1:
        raise ValueError("image_ids must be a Unicode string vector")
    n = len(labels)
    if n == 0 or len(logits) != n or len(image_ids) != n or n != len(csv_rows):
        raise ValueError("prediction arrays and split CSV must have the same nonzero sample count")
    if not np.isfinite(logits).all() or np.any((labels < 0) | (labels > 29)):
        raise ValueError("logits must be finite and labels must be in [0,29]")
    ids = [str(value) for value in image_ids.tolist()]
    if any(not _safe_id(value) for value in ids) or len(set(ids)) != n:
        raise ValueError("image_ids must be unique, safe, nonempty relative paths")
    csv_pairs = dict(csv_rows)
    prediction_pairs = dict(zip(ids, labels.tolist()))
    if prediction_pairs != csv_pairs:
        raise ValueError("prediction image_id/label pairs do not exactly match the split CSV")
    order_matches = ids == [item[0] for item in csv_rows]
    row_index = {image_id: i for i, image_id in enumerate(ids)}
    order = np.asarray([row_index[image_id] for image_id, _ in csv_rows], dtype=np.int64)
    refs = _load_references(root)
    if metadata.get("num_samples") != n or metadata.get("num_classes") != 30:
        raise ValueError("metadata sample/class counts do not match the prediction bundle")
    if split == "validation" and metadata.get("num_samples") != 300:
        raise ValueError("validation metadata must declare 300 samples")
    if isinstance(metadata.get("num_samples"), bool) or not isinstance(metadata.get("num_samples"), int):
        raise ValueError("metadata num_samples must be an integer")
    if isinstance(metadata.get("num_classes"), bool) or not isinstance(metadata.get("num_classes"), int):
        raise ValueError("metadata num_classes must be an integer")
    if metadata.get("class_mapping_path") != "data/metadata/class_mapping.json":
        raise ValueError("metadata class_mapping_path must use the shared mapping")
    upstream_sha = metadata.get("git_sha")
    if not isinstance(upstream_sha, str) or len(upstream_sha) != 40 or any(c not in "0123456789abcdefABCDEF" for c in upstream_sha):
        raise ValueError("metadata.git_sha must be a 40-character hexadecimal commit")
    declared_temperature = metadata.get("temperature")
    if declared_temperature is not None:
        if isinstance(declared_temperature, bool) or not isinstance(declared_temperature, (int, float)) or not math.isfinite(declared_temperature):
            raise ValueError("metadata temperature must be a finite numeric scalar")
        if declared_temperature != refs["temperature"]:
            raise ValueError("metadata temperature differs from the frozen E6 artifact")
    for field, expected in (("temperature_source_experiment", "E6"), ("temperature_source_split", "calibration")):
        if field in metadata and metadata[field] != expected:
            raise ValueError(f"metadata {field} must identify E6 Calibration")
    if split == "validation":
        if metadata.get("temperature_source_experiment") != "E6" or metadata.get("temperature_source_split") != "calibration" or metadata.get("temperature") != refs["temperature"]:
            raise ValueError("validation metadata temperature source/value disagrees with frozen E6")
        if metadata.get("temperature_artifact_path") != "outputs/metrics/E6/temperature.json":
            raise ValueError("validation metadata must identify the frozen E6 temperature artifact")
        safety = metadata.get("safety", {})
        if not isinstance(safety, dict) or any(safety.get(key) is not False for key in ("official_test_used", "model_trained_or_updated", "temperature_refit")):
            raise ValueError("validation metadata safety flags must all be false")
        if metadata.get("checkpoint_sha256") != OFFICIAL_CHECKPOINT_SHA256:
            raise ValueError("validation metadata checkpoint hash does not match the approved E4 checkpoint")
        if metadata.get("checkpoint_path") != "outputs/checkpoints/E4/seed_42/best.pt":
            raise ValueError("validation metadata must identify the approved E4 seed 42 checkpoint")
        preprocessing = metadata.get("preprocessing")
        if not isinstance(preprocessing, dict):
            raise ValueError("validation metadata must include preprocessing details")
        if preprocessing.get("config") != "configs/data.yaml":
            raise ValueError("validation metadata must identify shared preprocessing config")
        if metadata.get("model_experiment") not in (None, "E4"):
            raise ValueError("validation metadata must identify the approved E4 experiment")
        runtime = metadata.get("runtime")
        if not isinstance(runtime, dict):
            raise ValueError("validation metadata must include runtime details")
        for field in ("python_version", "pytorch_version", "torchvision_version", "device"):
            if not isinstance(runtime.get(field), str) or not runtime[field]:
                raise ValueError(f"validation metadata runtime.{field} must be a nonempty string")
        for field in ("batch_size", "num_workers"):
            value = runtime.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if field == "batch_size" else 0):
                raise ValueError(f"validation metadata runtime.{field} is invalid")
        if preprocessing.get("transform_builder") != "src.data.transforms.build_transforms":
            raise ValueError("validation metadata must identify the shared transform builder")
        if preprocessing.get("split_argument") != "validation":
            raise ValueError("validation metadata preprocessing split_argument must be validation")
        if preprocessing.get("expected_tensor_shape", preprocessing.get("shape")) != [3, 224, 224] or preprocessing.get("evaluation_resize") != 256 or preprocessing.get("evaluation_center_crop", preprocessing.get("crop")) != 224:
            raise ValueError("validation metadata preprocessing dimensions disagree with shared config")
        norm = preprocessing.get("normalization", {})
        if norm != {"type": "imagenet", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}:
            raise ValueError("validation metadata normalization disagrees with shared config")
        timestamp = runtime.get("timestamp_utc", metadata.get("timestamp_utc"))
        if not isinstance(timestamp, str) or not timestamp:
            raise ValueError("validation metadata must include timestamp_utc")
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("validation runtime timestamp_utc must be ISO 8601") from exc
        if parsed_timestamp.tzinfo is None:
            raise ValueError("validation runtime timestamp_utc must include a timezone")
    assert_prediction_bundle_complete(predictions_path)
    assert_prediction_bundle_complete(metadata_path)
    return {
        "logits": np.array(logits[order], copy=True), "labels": np.array(labels[order], dtype=np.int64, copy=True),
        "image_ids": np.asarray([item[0] for item in csv_rows], dtype=str), "metadata": metadata,
        "temperature": refs["temperature"], "threshold": refs["threshold"],
        "input_sha256": {"predictions": _sha256(predictions_path), "metadata": _sha256(metadata_path),
                         "split_csv": _sha256(csv_path), "class_mapping": mapping_hash,
                         "temperature": refs["temperature_sha256"], "threshold": refs["threshold_sha256"],
                         "e7_config": refs["config_sha256"], "data_config": refs["data_config_sha256"],
                         "e4_config": refs["e4_config_sha256"], "e6_config": refs["e6_config_sha256"],
                         "model_selection": refs["selection_config_sha256"]},
        "temperature_hash_mode": refs["temperature_hash_mode"],
        "mapping_hash_mode": mapping_mode, "input_order_matches_csv": order_matches,
        "upstream_provenance": {"status": "historical execution not independently verified" if split == "validation" else "synthetic fixture; no model inference claimed",
                                "metadata_git_sha": upstream_sha,
                                "checkpoint_verification": "metadata-only; no checkpoint loaded or independently verified" if split == "validation" else "no actual checkpoint used",
                                "local_git_source_presence": "diagnostic only; does not verify historical execution"},
        "split": split, "repo_root": root,
    }


def _git_status(root: Path, required_paths: list[str] | None = None) -> tuple[str | None, bool | None]:
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        if required_paths is None:
            return head, None
        tracked = subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", *required_paths], capture_output=True).returncode == 0
        changed = subprocess.run(["git", "-C", str(root), "diff", "HEAD", "--quiet", "--", *required_paths], capture_output=True).returncode != 0
        return head, tracked and not changed
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _execution_guard(bundle: dict[str, Any], output_dir: Path, split_csv: Path | None) -> None:
    root = bundle["repo_root"]
    if bundle["split"] == "validation":
        if root != SOURCE_ROOT:
            raise ValueError("validation execution requires repo_root to be this source checkout")
        required = ["scripts/run_final_selective.py", "src/selective/config.py", "src/selective/threshold.py", "src/selective/risk_coverage.py",
                    "configs/data.yaml", "configs/model_selection.yaml", "configs/experiments/e4_cutmix.yaml",
                    "configs/experiments/e6_calibration.yaml", "configs/experiments/e7_selective_prediction.yaml",
                    "outputs/metrics/E6/temperature.json", "outputs/metrics/E7/threshold.json",
                    "data/metadata/class_mapping.json", "data/splits/validation.csv"]
        head, clean_at_head = _git_status(root, required)
        if not head or not clean_at_head:
            raise ValueError("validation execution requires runner and frozen references tracked and clean at HEAD")
    else:
        if root == SOURCE_ROOT:
            raise ValueError("synthetic execution requires an alternate fixture/reference root")
        if output_dir.resolve(strict=False).is_relative_to(SOURCE_ROOT):
            raise ValueError("synthetic execution output must be outside the source checkout")
        if split_csv is None or split_csv.resolve(strict=False).is_relative_to(SOURCE_ROOT):
            raise ValueError("synthetic fixture CSV must be outside the source checkout")


def _output_location_guard(output_dir: Path, root: Path, input_paths: list[Path]) -> None:
    _reject_unsafe_path(output_dir, "output directory")
    resolved = output_dir.resolve(strict=False)
    if resolved.exists():
        raise ValueError(f"output directory already exists: {resolved}")
    if resolved.is_relative_to(root):
        relative = resolved.relative_to(root).as_posix()
        if not relative.startswith("outputs/"):
            raise ValueError("repository output directories must be under outputs/")
        protected = ("outputs/metrics/E6", "outputs/metrics/E7", "outputs/figures/E7",
                     "outputs/predictions/E6", "outputs/checkpoints",
                     "outputs/predictions/final", "outputs/metrics/final", "outputs/figures/final")
        if any(relative == item or relative.startswith(item + "/") for item in protected):
            raise ValueError("output directory collides with frozen E6/E7 or reserved Final Evaluation paths")
    for input_path in input_paths:
        resolved_input = input_path.resolve(strict=False)
        if resolved == resolved_input or resolved.is_relative_to(resolved_input) or resolved_input.is_relative_to(resolved):
            raise ValueError("output directory overlaps an input or reference path")


def run_final_selective(predictions_path, metadata_path, output_dir, *, split="validation", split_csv=None, repo_root=None) -> dict[str, Any]:
    if split not in {"validation", "synthetic"}:
        raise ValueError("split must be 'validation' or 'synthetic'")
    paths = [Path(predictions_path), Path(metadata_path), Path(output_dir)]
    if split_csv is not None:
        paths.append(Path(split_csv))
    for path, label in zip(paths, ("predictions", "metadata", "output directory", "split CSV")):
        _reject_unsafe_path(path, label)
    if repo_root is not None:
        _reject_unsafe_path(Path(repo_root), "repo root")
    resolved_root = Path(repo_root).resolve() if repo_root is not None else SOURCE_ROOT
    if split == "validation" and resolved_root != SOURCE_ROOT:
        raise ValueError("validation execution requires repo_root to be this source checkout")
    if split == "synthetic":
        if resolved_root == SOURCE_ROOT:
            raise ValueError("synthetic execution requires an alternate fixture/reference root")
        if Path(output_dir).resolve(strict=False).is_relative_to(SOURCE_ROOT):
            raise ValueError("synthetic execution output must be outside the source checkout")
        if split_csv is None or Path(split_csv).resolve(strict=False).is_relative_to(SOURCE_ROOT):
            raise ValueError("synthetic fixture CSV must be outside the source checkout")
    bundle = load_prediction_bundle(predictions_path, metadata_path, split=split, split_csv=split_csv, repo_root=repo_root)
    out = Path(output_dir).resolve(strict=False)
    _execution_guard(bundle, out, None if split_csv is None else Path(split_csv))
    _output_location_guard(out, bundle["repo_root"], [Path(predictions_path), Path(metadata_path), Path(split_csv) if split_csv else bundle["repo_root"] / "data/splits/validation.csv", bundle["repo_root"] / "outputs/metrics/E6/temperature.json", bundle["repo_root"] / "outputs/metrics/E7/threshold.json"])
    calibrated = calibrated_predictions(bundle["logits"], bundle["temperature"])
    correct = calibrated["predictions"] == bundle["labels"]
    accepted = accept_mask(calibrated["confidences"], bundle["threshold"])
    selected = selective_metrics(correct, accepted)
    curve = risk_coverage_curve(calibrated["confidences"], correct)
    aurc = compute_aurc(calibrated["confidences"], correct)
    head, _ = _git_status(SOURCE_ROOT)
    synthetic = split == "synthetic"
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "experiment_id": "E4_TASK04_FINAL_SELECTIVE", "experimental_status": "task04_preparation",
        "official_final_evaluation": False, "split": split, "synthetic": synthetic,
        "model": {"experiment": "E4", "architecture": "mobilenet_v2", "seed": 42, "num_classes": 30,
                  "checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256 if not synthetic else None,
                  "checkpoint_evidence": "metadata-only; not independently verified" if not synthetic else "no actual checkpoint used"},
        "temperature": bundle["temperature"], "temperature_hash_mode": bundle["temperature_hash_mode"], "threshold": bundle["threshold"],
        "threshold_reselected": False, "accept_operator": ">=", "selected_metrics": selected,
        "aurc": aurc, "aurc_convention": "grouped right-step over confidence tie groups",
        "n_total": int(len(bundle["labels"])), "input_order_matches_csv": bundle["input_order_matches_csv"],
        "mapping_hash_mode": bundle["mapping_hash_mode"], "input_sha256": bundle["input_sha256"],
        "upstream_provenance": bundle["upstream_provenance"],
        "source_git_sha": head, "source_commit_status": ("source checkout HEAD recorded; tracked clean unchecked" if head else "source checkout HEAD unavailable; tracked clean unchecked") if synthetic else "required source and reference files tracked and clean at HEAD",
        "upstream_git_sha": bundle["metadata"].get("git_sha"), "timestamp_utc": timestamp,
        "runtime": {"python_version": platform.python_version(), "numpy_version": np.__version__, "device": "CPU", "inference_performed": False},
        "output_paths": {"metrics": str(out / "selective_metrics.json"), "risk_coverage_csv": str(out / "risk_coverage.csv"), "risk_coverage_curve": str(out / "risk_coverage_curve.png")},
    }
    csv_text = _curve_csv(curve)
    figure = _curve_png(curve, split)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    created: list[Path] = []
    created_dir = False
    try:
        out.mkdir(parents=True, exist_ok=False)
        created_dir = True
        for name, data in (("selective_metrics.json", encoded.encode("utf-8")), ("risk_coverage.csv", csv_text.encode("utf-8")), ("risk_coverage_curve.png", figure)):
            target = out / name
            with target.open("xb") as handle:
                created.append(target)
                handle.write(data)
    except Exception:
        for path in created:
            try:
                path.unlink()
            except OSError:
                pass
        if created_dir:
            try:
                out.rmdir()
            except OSError:
                pass
        raise
    return payload


def _curve_csv(rows: list[dict[str, Any]]) -> str:
    import io
    buffer = io.StringIO(newline="")
    fields = ["n_total", "n_accept", "n_reject", "n_correct_accept", "coverage", "selective_accuracy", "selective_risk", "threshold"]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row.get(key) is None else row[key] for key in fields})
    return buffer.getvalue()


def _curve_png(rows: list[dict[str, Any]], split: str) -> bytes:
    import io
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    coverage = [row["coverage"] for row in rows]
    risks = [row["selective_risk"] for row in rows[1:]]
    if risks:
        ax.stairs(risks, edges=[0.0, *coverage[1:]], baseline=None, color="#245b9e", linewidth=2)
    ax.annotate("undefined at zero coverage", (0, 0), xytext=(8, 10), textcoords="offset points", fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Coverage", ylabel="Selective risk", title="Risk–coverage curve")
    ax.grid(True, alpha=0.25)
    ax.text(0.5, 0.5, "SYNTHETIC" if split == "synthetic" else "VALIDATION", transform=ax.transAxes,
            ha="center", va="center", fontsize=24, color="#a22", alpha=0.22, rotation=25, weight="bold")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    plt.close(fig)
    return buffer.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("validation", "synthetic"), default="validation")
    parser.add_argument("--split-csv")
    parser.add_argument("--repo-root")
    args = parser.parse_args(argv)
    try:
        result = run_final_selective(args.predictions, args.metadata, args.output_dir, split=args.split,
                                     split_csv=args.split_csv, repo_root=args.repo_root)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"run_final_selective: {exc}", file=sys.stderr)
        return 2
    label = "SYNTHETIC" if result["synthetic"] else "VALIDATION"
    metrics = result["selected_metrics"]
    print(f"{label} Task 04 preparation complete: n={result['n_total']}, coverage={metrics['coverage']:.6f}, AURC={result['aurc']:.6f}")
    print(f"Outputs: {result['output_paths']['metrics']}, {result['output_paths']['risk_coverage_csv']}, {result['output_paths']['risk_coverage_curve']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
