#!/usr/bin/env python3
"""Run the approved E7 selective-prediction evaluation on Calibration only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[1]
# E7 YAML deliberately omits these two fixed E6 handoff artifact paths.
E6_LOGITS_PATH = "outputs/predictions/E6/calibration_logits.npz"
E6_METADATA_PATH = "outputs/predictions/E6/calibration_metadata.json"
MODEL_SELECTION_PATH = "configs/model_selection.yaml"
DATA_CONFIG_PATH = "configs/data.yaml"
CLASS_MAPPING_PATH = "data/metadata/class_mapping.json"
E4_CONFIG_PATH = "configs/experiments/e4_cutmix.yaml"
E7_CONFIG_PATH = "configs/experiments/e7_selective_prediction.yaml"
E4_CHECKPOINT_PATH = "outputs/checkpoints/E4/seed_42/best.pt"
PREPROCESSING_MEAN = [0.485, 0.456, 0.406]
PREPROCESSING_STD = [0.229, 0.224, 0.225]
HEX_SHA1 = re.compile(r"^[0-9a-fA-F]{40}$")


def _source_imports() -> tuple[Any, Any, Any, Any]:
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from src.selective.config import load_e7_config, validate_e7_config
    from src.selective.risk_coverage import compute_aurc, risk_coverage_curve, selective_metrics
    from src.selective.threshold import accept_mask, calibrated_predictions, select_threshold
    return load_e7_config, validate_e7_config, (compute_aurc, risk_coverage_curve, selective_metrics), (
        accept_mask, calibrated_predictions, select_threshold
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Could not read {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must be a mapping")
    return value


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    import yaml

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except Exception as exc:
        raise ValueError(f"Could not read {label} YAML at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} YAML must be a mapping")
    return value


def _safe_repo_path(repo_root: Path, relative: str, *, must_exist: bool = True) -> Path:
    """Resolve a fixed repository-relative path without traversing symlinks."""
    if not isinstance(relative, str) or not relative:
        raise ValueError("Repository paths must be nonempty relative strings")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in relative:
        raise ValueError(f"Unsafe repository-relative path: {relative!r}")
    root = repo_root.resolve()
    current = root
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink path components are prohibited: {current}")
    resolved = current.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {relative!r}") from exc
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"Required E7 input file not found: {resolved}")
    return resolved


def _validate_mode_root(repo_root: Path, synthetic: bool) -> Path:
    root = Path(repo_root).resolve()
    source = SOURCE_ROOT.resolve()
    if type(synthetic) is not bool:
        raise ValueError("synthetic must be a boolean")
    if synthetic and root == source:
        raise ValueError("Synthetic mode requires an alternate temporary root")
    if not synthetic and root != source:
        raise ValueError("Official mode may only use the source repository root")
    if not root.is_dir():
        raise FileNotFoundError(f"Repository root does not exist: {root}")
    return root


def _source_head(repo_root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Could not determine source Git HEAD") from exc
    if not HEX_SHA1.fullmatch(value):
        raise ValueError("Source Git HEAD is not a 40-character SHA")
    return value


def _require_tracked_clean_sources(repo_root: Path) -> str:
    paths = [
        "src/selective/threshold.py",
        "src/selective/risk_coverage.py",
        "src/selective/config.py",
        "scripts/run_selective_prediction.py",
    ]
    try:
        subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", "--", *paths],
            cwd=repo_root, stderr=subprocess.DEVNULL, text=True,
        )
        changed = subprocess.check_output(
            ["git", "diff", "HEAD", "--name-only", "--", *paths],
            cwd=repo_root, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("E7 source files must be tracked before an official run") from exc
    if changed:
        raise ValueError("E7 source files differ from Git HEAD; commit them before an official run")
    return _source_head(repo_root)


def _valid_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not HEX_SHA1.fullmatch(value):
        raise ValueError(f"{name} must be a 40-character hexadecimal Git SHA")
    return value.lower()


def _strict_int(value: Any, expected: int, name: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{name} must be {expected}")


def _path_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty checkpoint identifier")
    normalized = value.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if ".." in parts or len(parts) < 3 or parts[-3:] != ("E4", "seed_42", "best.pt"):
        raise ValueError(f"{name} must identify E4/seed_42/best.pt")
    return value


def _validate_preprocessing(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label}.preprocessing must be a mapping")
    expected = {
        "color_mode": "RGB",
        "resize": 256,
        "center_crop": 224,
        "normalization_mean": PREPROCESSING_MEAN,
        "normalization_std": PREPROCESSING_STD,
    }
    for key, expected_value in expected.items():
        actual = value.get(key)
        if key in ("resize", "center_crop"):
            _strict_int(actual, expected_value, f"{label}.preprocessing.{key}")
        elif key in ("normalization_mean", "normalization_std"):
            if not isinstance(actual, list) or len(actual) != len(expected_value):
                raise ValueError(f"{label}.preprocessing.{key} does not match the shared transform")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x))
                   for x in actual) or [float(x) for x in actual] != expected_value:
                raise ValueError(f"{label}.preprocessing.{key} does not match the shared transform")
        elif actual != expected_value or type(actual) is not type(expected_value):
            raise ValueError(f"{label}.preprocessing.{key} must be {expected_value!r}")


def _validate_static_configs(repo_root: Path, config: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    """Check model, data, split, and class mapping before opening the NPZ."""
    model_selection_path = _safe_repo_path(repo_root, MODEL_SELECTION_PATH)
    data_config_path = _safe_repo_path(repo_root, DATA_CONFIG_PATH)
    mapping_path = _safe_repo_path(repo_root, CLASS_MAPPING_PATH)
    e4_config_path = _safe_repo_path(repo_root, E4_CONFIG_PATH)
    split_rel = config["data"]["split_file"]
    split_path = _safe_repo_path(repo_root, split_rel)

    selection = _read_yaml(model_selection_path, "model selection")
    expected_samples = config["data"]["expected_samples"]
    num_classes = config["data"]["num_classes"]
    selection_block = selection.get("selection")
    model_block = selection.get("model")
    data_block = selection.get("data")
    if not all(isinstance(block, dict) for block in (selection_block, model_block, data_block)):
        raise ValueError("model_selection.yaml must contain selection, model, and data mappings")
    if selection_block.get("status") != "approved" or selection_block.get("selected_experiment") != "E4":
        raise ValueError("model_selection.yaml must approve E4")
    _strict_int(selection_block.get("selected_seed"), 42, "model_selection.selection.selected_seed")
    if model_block.get("architecture") != "mobilenet_v2":
        raise ValueError("Approved model selection must use MobileNetV2")
    _strict_int(model_block.get("num_classes"), num_classes, "model_selection.model.num_classes")
    if model_block.get("config_path") != E4_CONFIG_PATH:
        raise ValueError("Approved E4 config path must match configs/experiments/e4_cutmix.yaml")
    if model_block.get("checkpoint_path") != E4_CHECKPOINT_PATH:
        raise ValueError("Approved checkpoint must be E4 seed 42 best.pt")
    if data_block.get("class_mapping") != CLASS_MAPPING_PATH:
        raise ValueError("Approved class mapping path is required")
    if data_block.get("calibration_split") != split_rel:
        raise ValueError("Model selection calibration split does not match E7 config")
    _strict_int(data_block.get("calibration_samples"), expected_samples,
                "model_selection.data.calibration_samples")
    if data_block.get("official_test_locked") is not True:
        raise ValueError("model_selection.yaml must keep Official Test locked")

    e4 = _read_yaml(e4_config_path, "E4 experiment")
    e4_experiment = e4.get("experiment")
    e4_model = e4.get("model")
    if not isinstance(e4_experiment, dict) or e4_experiment.get("id") != "E4" or not isinstance(e4_model, dict):
        raise ValueError("E4 config identity is invalid")
    if e4_model.get("type") != "mobilenet_v2" or e4_model.get("backbone_frozen") is not True:
        raise ValueError("E4 config must specify frozen MobileNetV2")
    _strict_int(e4_model.get("num_classes"), num_classes, "E4 model.num_classes")

    data_config = _read_yaml(data_config_path, "shared data")
    data_values = data_config.get("data")
    preprocessing = data_config.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("Shared data config preprocessing must be a mapping")
    evaluation = preprocessing.get("evaluation")
    normalization = preprocessing.get("normalization")
    if not isinstance(data_values, dict) or not isinstance(evaluation, dict) or not isinstance(normalization, dict):
        raise ValueError("Shared data config is missing data/preprocessing fields")
    _strict_int(data_values.get("num_classes"), num_classes, "data.num_classes")
    _strict_int(data_values.get("image_size"), 224, "data.image_size")
    official_test = data_values.get("official_test")
    if not isinstance(official_test, dict) or official_test.get("locked_until_final_evaluation") is not True:
        raise ValueError("Shared data config must lock Official Test")
    _strict_int(evaluation.get("resize"), 256, "preprocessing.evaluation.resize")
    _strict_int(evaluation.get("center_crop"), 224, "preprocessing.evaluation.center_crop")
    if normalization.get("type") != "imagenet":
        raise ValueError("Shared normalization must be ImageNet")
    if normalization.get("mean") != PREPROCESSING_MEAN or normalization.get("std") != PREPROCESSING_STD:
        raise ValueError("Shared ImageNet normalization values do not match the approved protocol")

    mapping = _read_json(mapping_path, "class mapping")
    expected_ids = {str(index) for index in range(num_classes)}
    if set(mapping) != expected_ids or any(not isinstance(name, str) or not name for name in mapping.values()):
        raise ValueError("Class mapping must contain exactly class IDs 0 through 29")
    if len(set(mapping.values())) != num_classes:
        raise ValueError("Class mapping breed names must be unique")
    rows = _read_calibration_csv(split_path, expected_samples, mapping)

    hashed_paths = {
        MODEL_SELECTION_PATH: model_selection_path,
        DATA_CONFIG_PATH: data_config_path,
        CLASS_MAPPING_PATH: mapping_path,
        split_rel: split_path,
        E4_CONFIG_PATH: e4_config_path,
    }
    return {key: _sha256(path) for key, path in hashed_paths.items()}, {
        "rows": rows,
        "class_mapping": mapping,
        "model_selection": selection,
        "paths": hashed_paths,
    }


def _read_calibration_csv(path: Path, expected_samples: int,
                          mapping: dict[str, str]) -> list[dict[str, Any]]:
    required = {"relative_path", "class_id", "breed_name", "source_split"}
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise ValueError("Calibration CSV must contain relative_path,class_id,breed_name,source_split")
            for row_number, row in enumerate(reader, start=2):
                relative = row.get("relative_path")
                if not isinstance(relative, str) or not relative or "\\" in relative:
                    raise ValueError(f"Calibration CSV row {row_number} has an unsafe relative_path")
                pure = PurePosixPath(relative)
                if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
                    raise ValueError(f"Calibration CSV row {row_number} has an unsafe relative_path")
                if "official_test" in relative.lower():
                    raise ValueError("Calibration rows must not reference Official Test")
                raw_class = row.get("class_id")
                if not isinstance(raw_class, str) or not raw_class.isdigit():
                    raise ValueError(f"Calibration CSV row {row_number} class_id must be an integer")
                class_id = int(raw_class)
                if not 0 <= class_id < len(mapping):
                    raise ValueError(f"Calibration CSV row {row_number} class_id is out of range")
                if row.get("breed_name") != mapping[str(class_id)]:
                    raise ValueError(f"Calibration CSV row {row_number} breed does not match class mapping")
                if row.get("source_split") != "official_train":
                    raise ValueError(f"Calibration CSV row {row_number} must come from official_train")
                rows.append({"relative_path": relative, "class_id": class_id,
                             "breed_name": row["breed_name"], "source_split": row["source_split"]})
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read Calibration CSV at {path}: {exc}") from exc
    if len(rows) != expected_samples:
        raise ValueError(f"Expected exactly {expected_samples} Calibration rows; found {len(rows)}")
    image_ids = [row["relative_path"] for row in rows]
    if len(set(image_ids)) != expected_samples:
        raise ValueError("Calibration CSV relative_path values must be unique")
    if expected_samples % len(mapping):
        raise ValueError("Expected sample count must divide evenly across the configured classes")
    per_class = expected_samples // len(mapping)
    counts = np.bincount([row["class_id"] for row in rows], minlength=len(mapping))
    if counts.shape != (len(mapping),) or np.any(counts != per_class):
        raise ValueError(f"Calibration CSV must contain exactly {per_class} samples per class")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"Could not hash input file {path}: {exc}") from exc
    return digest.hexdigest()


def _validate_e6_jsons(temperature_json: dict[str, Any], metadata: dict[str, Any],
                       synthetic: bool, expected_samples: int,
                       num_classes: int) -> tuple[float, str, str]:
    for label, obj in (("temperature", temperature_json), ("metadata", metadata)):
        marker = obj.get("synthetic", False)
        if type(marker) is not bool:
            raise ValueError(f"E6 {label}.synthetic must be a boolean when present")
        if synthetic and marker is not True:
            raise ValueError(f"Synthetic mode requires synthetic=true in E6 {label} JSON")
        if not synthetic and marker is True:
            raise ValueError("Official mode rejects synthetic-marked E6 inputs")

    if temperature_json.get("experiment") != "E6" or temperature_json.get("model_experiment") != "E4":
        raise ValueError("E6 temperature artifact must identify E6 and source model E4")
    if metadata.get("experiment") != "E6":
        raise ValueError("E6 metadata.experiment must be E6")
    if metadata.get("source_model") != "E4 MobileNetV2 Frozen + CutMix":
        raise ValueError("E6 metadata must identify E4 MobileNetV2 Frozen + CutMix")
    for obj, label in ((temperature_json, "temperature"), (metadata, "metadata")):
        _strict_int(obj.get("checkpoint_seed"), 42, f"E6 {label}.checkpoint_seed")
        if obj.get("split") != "calibration":
            raise ValueError(f"E6 {label} artifact must use the calibration split")
        _strict_int(obj.get("n_samples"), expected_samples, f"E6 {label}.n_samples")

    temperature = temperature_json.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("E6 temperature must be a real numeric scalar")
    try:
        temperature = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("E6 temperature must be finite and positive") from exc
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("E6 temperature must be finite and positive")

    checkpoint_temp = _path_identifier(temperature_json.get("checkpoint_path"), "temperature.checkpoint_path")
    checkpoint_meta = _path_identifier(metadata.get("checkpoint_identifier"), "metadata.checkpoint_identifier")
    if checkpoint_temp != checkpoint_meta:
        raise ValueError("E6 temperature and metadata checkpoint identifiers must match exactly")
    if metadata.get("checkpoint_filename") != "best.pt":
        raise ValueError("E6 metadata.checkpoint_filename must be best.pt")
    source_sha = _valid_sha(temperature_json.get("git_sha"), "temperature.git_sha")
    metadata_sha = _valid_sha(metadata.get("git_sha"), "metadata.git_sha")
    if source_sha != metadata_sha:
        raise ValueError("E6 temperature and metadata Git SHA values must match")
    if metadata.get("weights_updated") is not False:
        raise ValueError("E6 metadata must confirm weights_updated=false")
    if metadata.get("official_test_used") is not False:
        raise ValueError("E6 metadata must confirm official_test_used=false")
    if temperature_json.get("weights_updated", False) is not False:
        raise ValueError("E6 temperature artifact must not report weight updates")
    if temperature_json.get("official_test_used", False) is not False:
        raise ValueError("E6 temperature artifact must not report Official Test use")
    expected_shapes = {
        "logits_shape": [expected_samples, num_classes],
        "labels_shape": [expected_samples],
        "image_ids_shape": [expected_samples],
    }
    for key, expected in expected_shapes.items():
        actual = metadata.get(key)
        if (not isinstance(actual, list) or len(actual) != len(expected)
                or any(type(value) is not int for value in actual) or actual != expected):
            raise ValueError(f"E6 metadata.{key} must be {expected}")
    _validate_preprocessing(metadata.get("preprocessing"), "E6 metadata")
    return temperature, source_sha, checkpoint_temp


def _decode_image_ids(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "U":
        decoded = values.astype(str, copy=True)
    elif values.dtype.kind == "S":
        try:
            decoded = np.asarray([value.decode("utf-8", errors="strict") for value in values], dtype=str)
        except UnicodeDecodeError as exc:
            raise ValueError("E6 image_ids contain invalid UTF-8 bytes") from exc
    else:
        raise ValueError("E6 image_ids must be Unicode or byte-string arrays without objects")
    if np.any(decoded == ""):
        raise ValueError("E6 image_ids must be nonempty")
    return decoded


def load_e6_artifacts(config: dict[str, Any], repo_root: str | Path, *,
                      synthetic: bool = False) -> dict[str, Any]:
    """Validate E6 handoff inputs and return aligned arrays without scoring them."""
    _, validate_config, _, _ = _source_imports()
    config = validate_config(config)
    root = _validate_mode_root(Path(repo_root), synthetic)

    temp_rel = config["calibration"]["temperature_file"]
    metadata_path = _safe_repo_path(root, E6_METADATA_PATH)
    logits_path = _safe_repo_path(root, E6_LOGITS_PATH)
    temperature_path = _safe_repo_path(root, temp_rel)
    static_hashes, static_context = _validate_static_configs(root, config)

    # Check JSON provenance and mode markers before opening the compressed array artifact.
    temperature_json = _read_json(temperature_path, "E6 temperature")
    metadata = _read_json(metadata_path, "E6 calibration metadata")
    temperature, upstream_sha, checkpoint_identifier = _validate_e6_jsons(
        temperature_json, metadata, synthetic,
        config["data"]["expected_samples"], config["data"]["num_classes"],
    )
    expected_samples = config["data"]["expected_samples"]
    num_classes = config["data"]["num_classes"]
    try:
        with np.load(logits_path, allow_pickle=False) as archive:
            if set(archive.files) != {"logits", "labels", "image_ids"}:
                raise ValueError("E6 NPZ must contain exactly logits, labels, and image_ids")
            logits = np.array(archive["logits"], copy=True)
            labels = np.array(archive["labels"], copy=True)
            image_ids_raw = np.array(archive["image_ids"], copy=True)
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("E6 NPZ"):
            raise
        raise ValueError(f"Could not safely load E6 logits NPZ: {exc}") from exc

    if logits.dtype.kind != "f" or logits.shape != (expected_samples, num_classes):
        raise ValueError(f"E6 logits must be floating with shape ({expected_samples}, {num_classes})")
    if not np.all(np.isfinite(logits)):
        raise ValueError("E6 logits contain NaN or infinity")
    if labels.dtype.kind not in "iu" or labels.shape != (expected_samples,):
        raise ValueError(f"E6 labels must be an integer vector of length {expected_samples}")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("E6 labels contain class IDs outside the configured range")
    if image_ids_raw.shape != (expected_samples,):
        raise ValueError(f"E6 image_ids must have shape ({expected_samples},)")
    image_ids = _decode_image_ids(image_ids_raw)
    if image_ids.shape != (expected_samples,) or len(set(image_ids.tolist())) != expected_samples:
        raise ValueError("E6 image_ids must be unique strings with the configured sample count")
    csv_rows = static_context["rows"]
    csv_ids = np.asarray([row["relative_path"] for row in csv_rows], dtype=str)
    csv_labels = np.asarray([row["class_id"] for row in csv_rows], dtype=np.int64)
    if not np.array_equal(image_ids, csv_ids):
        raise ValueError("E6 image_ids/order do not exactly match Calibration CSV relative_path values")
    if not np.array_equal(labels.astype(np.int64, copy=False), csv_labels):
        raise ValueError("E6 labels/order do not exactly match Calibration CSV class_id values")

    shape_checks = {"logits_shape": list(logits.shape), "labels_shape": list(labels.shape),
                    "image_ids_shape": list(image_ids.shape)}
    for key, actual in shape_checks.items():
        if metadata.get(key) != actual:
            raise ValueError(f"E6 metadata.{key} does not match the loaded NPZ")

    hashes = {
        **static_hashes,
        temp_rel: _sha256(temperature_path),
        E6_METADATA_PATH: _sha256(metadata_path),
        E6_LOGITS_PATH: _sha256(logits_path),
    }
    return {
        "logits": logits,
        "labels": labels.astype(np.int64, copy=False),
        "image_ids": image_ids,
        "temperature": temperature,
        "input_sha256": hashes,
        "upstream_metadata": metadata,
        "upstream_git_sha": upstream_sha,
        "checkpoint_identifier": checkpoint_identifier,
    }


def _safe_input_bundle(repo_root: Path, config_path: Path, config: dict[str, Any]) -> dict[str, str]:
    paths = {
        "e7_config": config_path,
        **{name: _safe_repo_path(repo_root, rel) for name, rel in (
            ("model_selection", MODEL_SELECTION_PATH),
            ("data_config", DATA_CONFIG_PATH),
            ("class_mapping", CLASS_MAPPING_PATH),
            ("calibration_csv", config["data"]["split_file"]),
            ("e4_config", E4_CONFIG_PATH),
            ("e6_temperature", config["calibration"]["temperature_file"]),
            ("e6_metadata", E6_METADATA_PATH),
            ("e6_logits", E6_LOGITS_PATH),
        )},
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _coverage_rows(confidences: np.ndarray, correct: np.ndarray, config: dict[str, Any],
                   select_threshold: Any, accept_mask: Any, selective_metrics: Any) -> list[dict[str, Any]]:
    rows = []
    for requested in config["evaluation"]["coverage_points"]:
        threshold = select_threshold(confidences, requested)
        accepted = accept_mask(confidences, threshold)
        metrics = selective_metrics(correct, accepted)
        rows.append({"requested_coverage": float(requested), "threshold": threshold, **metrics})
    return rows


def _json_bytes(value: dict[str, Any]) -> bytes:
    try:
        return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not serialize E7 JSON output: {exc}") from exc


def _csv_bytes(curve: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    columns = ["threshold", "n_total", "n_accept", "n_reject", "n_correct_accept",
               "coverage", "selective_accuracy", "selective_risk"]
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in curve:
        writer.writerow({key: "" if row[key] is None else row[key] for key in columns})
    return output.getvalue().encode("utf-8")


def _figure_bytes(curve: list[dict[str, Any]], synthetic: bool) -> bytes:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ValueError("Matplotlib is required to create the E7 risk-coverage figure") from exc
    coverages = [0.0] + [float(row["coverage"]) for row in curve[1:]]
    risks = [float(row["selective_risk"]) for row in curve[1:]]
    figure = Figure(figsize=(7.2, 5.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.stairs(risks, coverages, color="#245a9b", linewidth=2.2, fill=False)
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0), xlabel="Coverage", ylabel="Selective Risk")
    title = "Risk–Coverage Curve"
    if synthetic:
        title += " (SYNTHETIC SMOKE TEST)"
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.text(0.02, 0.03, "Coverage 0: risk undefined", transform=axis.transAxes,
              fontsize=8, color="#555555")
    buffer = io.BytesIO()
    figure.canvas.print_png(buffer)
    figure.clear()
    return buffer.getvalue()


def _output_paths(repo_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    return {key: _safe_repo_path(repo_root, relative, must_exist=False)
            for key, relative in config["outputs"].items()}


def _preflight_outputs(repo_root: Path, config: dict[str, Any]) -> dict[str, Path]:
    paths = _output_paths(repo_root, config)
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"E7 output already exists; refusing to overwrite: {path}")
    return paths


def _write_outputs_atomically(outputs: dict[Path, bytes], repo_root: Path) -> None:
    for path in outputs:
        # Recheck symlink components immediately before the exclusive create.
        relative = path.relative_to(repo_root.resolve()).as_posix()
        checked = _safe_repo_path(repo_root, relative, must_exist=False)
        if checked != path or path.exists():
            raise FileExistsError(f"E7 output already exists or path changed: {path}")
    created: list[Path] = []
    try:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Validate again after mkdir in case any path component changed.
            relative = path.relative_to(repo_root.resolve()).as_posix()
            _safe_repo_path(repo_root, relative, must_exist=False)
            with path.open("xb") as handle:
                created.append(path)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
    except Exception:
        for path in created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def run_selective_prediction(config_path: str | Path | None = None,
                              repo_root: str | Path | None = None, *,
                              split: str = "calibration", synthetic: bool = False) -> dict[str, Any]:
    """Validate inputs, calculate E7 metrics, and create four exclusive outputs."""
    if split != "calibration":
        raise ValueError("E7 runner accepts only split='calibration'; Official Test is prohibited")
    root = _validate_mode_root(Path(repo_root) if repo_root is not None else SOURCE_ROOT, synthetic)
    load_config, validate_config, risk_functions, threshold_functions = _source_imports()
    compute_aurc, risk_coverage_curve, selective_metrics = risk_functions
    accept_mask, calibrated_predictions, select_threshold = threshold_functions

    raw_config_path = Path(config_path) if config_path is not None else Path(E7_CONFIG_PATH)
    if raw_config_path.is_absolute():
        try:
            lexical_path = Path(os.path.abspath(str(raw_config_path)))
            config_rel = lexical_path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("E7 config path must be inside the selected repository root") from exc
    else:
        config_rel = raw_config_path.as_posix()
    config_file = _safe_repo_path(root, config_rel)
    config = validate_config(load_config(config_file))

    if synthetic:
        source_git_sha = _source_head(SOURCE_ROOT)
        source_commit_status = "source HEAD recorded; tracked/clean status not checked in synthetic mode"
    else:
        source_git_sha = _require_tracked_clean_sources(root)
        source_commit_status = "tracked and clean at source HEAD; remote publication not checked"

    artifacts = load_e6_artifacts(config, root, synthetic=synthetic)
    output_paths = _preflight_outputs(root, config)
    computed = calibrated_predictions(artifacts["logits"], artifacts["temperature"])
    predictions = computed["predictions"]
    confidences = computed["confidences"]
    correct = predictions == artifacts["labels"]

    target = config["threshold_selection"]["target_coverage"]
    threshold = select_threshold(confidences, target)
    accepted = accept_mask(confidences, threshold)
    selected = selective_metrics(correct, accepted)
    if selected["coverage"] < target:
        raise RuntimeError("Selected E7 threshold does not achieve the configured target coverage")
    coverage_points = _coverage_rows(confidences, correct, config, select_threshold,
                                     accept_mask, selective_metrics)
    curve = risk_coverage_curve(confidences, correct)
    aurc = compute_aurc(confidences, correct)
    timestamp = datetime.now(timezone.utc).isoformat()
    runtime = {"python": platform.python_version(), "numpy": np.__version__,
               "platform": platform.platform()}
    input_hashes = _safe_input_bundle(root, config_file, config)
    upstream_provenance = {
        "temperature_git_sha": artifacts["upstream_git_sha"],
        "checkpoint_identifier": artifacts["checkpoint_identifier"],
        "provenance_status": "historical E6 provenance recorded; not independently verified",
    }
    common = {
        "experiment": "E7",
        "execution_kind": "synthetic_smoke_test" if synthetic else "official_calibration_execution",
        "official_e7_execution": not synthetic,
        "synthetic": synthetic,
        "model": {"experiment": "E4", "architecture": "mobilenet_v2", "seed": 42,
                  "checkpoint_identifier": artifacts["checkpoint_identifier"]},
        "split": config["data"]["split"],
        "n_total": int(len(artifacts["labels"])),
        "temperature": float(artifacts["temperature"]),
        "selected_threshold": float(threshold),
        "target_coverage": float(target),
        "selected_metrics": selected,
        "n_accept": selected["n_accept"],
        "n_reject": selected["n_reject"],
        "n_correct_accept": selected["n_correct_accept"],
        "coverage": selected["coverage"],
        "selective_accuracy": selected["selective_accuracy"],
        "selective_risk": selected["selective_risk"],
        "coverage_points": coverage_points,
        "aurc": float(aurc),
        "aurc_convention": {
            "sort": "confidence descending",
            "ties": "accept entire exact-equal-confidence group at once",
            "integral": "right-step sum of post-group risk times the coverage increment",
            "zero_coverage": "mathematical origin; risk and accuracy undefined and omitted from area",
        },
        "risk_coverage_rows": len(curve),
        "rejection_message": config["rejection"]["message"],
        "source_git_sha": source_git_sha,
        "source_commit_status": source_commit_status,
        "runtime": runtime,
        "timestamp_utc": timestamp,
        "input_sha256": input_hashes,
        "upstream_provenance": upstream_provenance,
    }
    threshold_payload = {
        "experiment": "E7",
        "execution_kind": common["execution_kind"],
        "official_e7_execution": common["official_e7_execution"],
        "synthetic": synthetic,
        "split": config["data"]["split"],
        "temperature": float(artifacts["temperature"]),
        "threshold": float(threshold),
        "target_coverage": float(target),
        "selection_rule": config["threshold_selection"]["selection_rule"],
        "accept_operator": config["threshold_selection"]["accept_operator"],
        "tie_policy": config["threshold_selection"]["tie_policy"],
        "n_total": selected["n_total"],
        "n_accept": selected["n_accept"],
        "n_reject": selected["n_reject"],
        "achieved_coverage": selected["coverage"],
        "source_git_sha": source_git_sha,
        "source_commit_status": common["source_commit_status"],
        "input_sha256": input_hashes,
        "upstream_provenance": upstream_provenance,
    }
    payloads = {
        config["outputs"]["metrics"]: _json_bytes(common),
        config["outputs"]["threshold"]: _json_bytes(threshold_payload),
        config["outputs"]["risk_coverage_data"]: _csv_bytes(curve),
        config["outputs"]["risk_coverage_figure"]: _figure_bytes(curve, synthetic),
    }
    resolved_outputs = {output_paths[key]: payloads[config["outputs"][key]]
                        for key in output_paths}
    _write_outputs_atomically(resolved_outputs, root)
    common["output_paths"] = {relative: str(_safe_repo_path(root, relative, must_exist=False))
                              for relative in payloads}
    return common


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="E7 YAML path (default: frozen repository config)")
    parser.add_argument("--repo-root", default=None, help="Repository root (alternate root required for synthetic mode)")
    parser.add_argument("--split", default="calibration", help="Only calibration is permitted")
    parser.add_argument("--synthetic", action="store_true", help="Run a clearly marked fixture smoke test")
    args = parser.parse_args(argv)
    try:
        result = run_selective_prediction(
            config_path=args.config,
            repo_root=args.repo_root,
            split=args.split,
            synthetic=args.synthetic,
        )
    except Exception as exc:
        print(f"E7 selective-prediction failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"experiment": "E7", "execution_kind": result["execution_kind"],
                      "output_paths": result["output_paths"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
