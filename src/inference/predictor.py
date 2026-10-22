"""Final E4 inference, raw-prediction export, and E6 calibration evaluation.

This module reuses DogBreedID project components:
- src.models.model_factory.build_model
- src.data.dataset.DogBreedDataset
- src.data.transforms.build_transforms
- src.evaluation.metrics.classification_metrics
- src.calibration.calibration_metrics.compute_calibration_metrics
- src.calibration.temperature_scaling.load_temperature

Task 04 safety: this runner only permits the approved Validation split. It does
not fit Temperature, select E7 thresholds, train the model, or use Official Test.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

NUM_CLASSES = 30
INPUT_SIZE = 224
ECE_BINS = 15
EXPECTED_CHECKPOINT_SHA256 = (
    "73df0cc1c5b16866d4facc910f3e4d3ab481e3f12f1681a59fa7d61e3845c47b"
)
# Approved E6 value recorded in docs/final_evaluation/configuration_lock.md.
EXPECTED_TEMPERATURE = 0.80246283961825
DEFAULT_CHECKPOINT = Path("outputs/checkpoints/E4/seed_42/best.pt")
DEFAULT_TEMPERATURE_ARTIFACT = Path("outputs/metrics/E6/temperature.json")
DEFAULT_SPLIT_CSV = Path("data/splits/validation.csv")
DEFAULT_E4_CONFIG = Path("configs/experiments/e4_cutmix.yaml")
DEFAULT_DATA_CONFIG = Path("configs/data.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/predictions/task04_validation")


class InferenceContractError(ValueError):
    """Raised when an input/output violates the Task 04 data contract."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA-256 without reading a large checkpoint into memory at once."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for SHA-256 verification: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_hash(
    checkpoint_path: str | Path,
    expected_sha256: str = EXPECTED_CHECKPOINT_SHA256,
) -> str:
    """Return the verified SHA-256, refusing any checkpoint that does not match."""
    actual = sha256_file(checkpoint_path)
    if actual.lower() != str(expected_sha256).lower():
        raise RuntimeError(
            "E4 checkpoint SHA-256 mismatch. Refusing to load the checkpoint. "
            f"Expected={expected_sha256}; actual={actual}; path={checkpoint_path}"
        )
    return actual


def assert_task04_split_allowed(split_name: str) -> None:
    """Block Official Test and every split except Validation in Task 04."""
    normalized = str(split_name).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized != "validation":
        raise PermissionError(
            f"Task 04 runner only permits split='validation'; received {split_name!r}. "
            "Official Test is locked and must not be accessed in this task."
        )


def _require_validation_csv(repo_root: Path, csv_path: str | Path) -> Path:
    path = Path(csv_path).expanduser().resolve()
    approved = (repo_root / DEFAULT_SPLIT_CSV).resolve()
    if path != approved:
        raise PermissionError(
            "This Task 04 runner only reads the approved validation split CSV at "
            f"{approved}. Refusing: {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Validation split CSV not found: {path}")
    # Avoid accidentally accepting a different file renamed to look like Validation.
    if path.name.lower() != "validation.csv":
        raise PermissionError(f"Expected validation.csv, got {path.name!r}")
    return path


def _safe_output_dir(repo_root: Path, output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    outputs_root = (repo_root / "outputs").resolve()
    try:
        path.relative_to(outputs_root)
    except ValueError as exc:
        raise PermissionError(
            f"Task 04 output directory must be inside {outputs_root}; got {path}"
        ) from exc
    lowered_parts = {part.lower().replace("-", "_") for part in path.parts}
    if "official_test" in lowered_parts or "official" in lowered_parts:
        raise PermissionError(
            f"Refusing to write Task 04 outputs to a location named Official Test: {path}"
        )
    if path == outputs_root or path == (outputs_root / "predictions" / "final").resolve():
        raise PermissionError(f"Refusing to write to a protected output directory: {path}")
    return path


def validate_raw_predictions(
    logits: Any,
    labels: Any,
    image_ids: Any,
    *,
    num_classes: int = NUM_CLASSES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate alignment, shapes, dtypes, finite values, labels and unique IDs."""
    logits_np = np.asarray(logits)
    labels_np = np.asarray(labels)
    ids_np = np.asarray(image_ids)

    if logits_np.ndim != 2 or logits_np.shape[1] != num_classes:
        raise InferenceContractError(
            f"logits must have shape [N,{num_classes}], got {logits_np.shape}"
        )
    n = logits_np.shape[0]
    if n < 1:
        raise InferenceContractError("Raw predictions cannot be empty")
    if labels_np.ndim != 1 or labels_np.shape[0] != n:
        raise InferenceContractError(
            f"labels must have shape [{n}], got {labels_np.shape}"
        )
    if ids_np.ndim != 1 or ids_np.shape[0] != n:
        raise InferenceContractError(
            f"image_ids must have shape [{n}], got {ids_np.shape}"
        )
    if not np.issubdtype(logits_np.dtype, np.number):
        raise InferenceContractError("logits must be numeric")
    if not np.isfinite(logits_np).all():
        raise InferenceContractError("logits contain NaN or Inf")
    if not np.issubdtype(labels_np.dtype, np.integer):
        if not np.issubdtype(labels_np.dtype, np.number):
            raise InferenceContractError("labels must be integer class IDs")
        if not np.isfinite(labels_np).all() or not np.equal(labels_np, np.floor(labels_np)).all():
            raise InferenceContractError("labels must contain finite integer class IDs")
    labels_np = labels_np.astype(np.int64, copy=False)
    if labels_np.min() < 0 or labels_np.max() >= num_classes:
        raise InferenceContractError(f"labels must be in [0, {num_classes - 1}]")

    # Convert to Unicode, never object arrays (the NPZ reader uses allow_pickle=False).
    try:
        ids_as_list = [str(item) for item in ids_np.tolist()]
    except Exception as exc:  # pragma: no cover - defensive conversion path
        raise InferenceContractError("image_ids must be convertible to strings") from exc
    if any(not image_id.strip() for image_id in ids_as_list):
        raise InferenceContractError("image_ids cannot contain empty strings")
    if len(set(ids_as_list)) != len(ids_as_list):
        raise InferenceContractError("image_ids must be unique within the evaluated split")

    logits_np = logits_np.astype(np.float32 if logits_np.dtype.itemsize <= 4 else np.float64, copy=False)
    ids_unicode = np.asarray(ids_as_list, dtype=np.str_)
    return logits_np, labels_np, ids_unicode


def save_raw_predictions(
    output_path: str | Path,
    logits: Any,
    labels: Any,
    image_ids: Any,
) -> Path:
    """Write exactly the three required NPZ fields, using Unicode image IDs."""
    output_path = Path(output_path)
    logits_np, labels_np, ids_np = validate_raw_predictions(logits, labels, image_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        logits=logits_np,
        labels=labels_np.astype(np.int64, copy=False),
        image_ids=ids_np,
    )
    return output_path


def load_raw_predictions(input_path: str | Path) -> dict[str, np.ndarray]:
    """Read and validate a raw-prediction NPZ without enabling pickle."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Raw-prediction NPZ not found: {input_path}")
    with np.load(input_path, allow_pickle=False) as stored:
        required = {"logits", "labels", "image_ids"}
        missing = required - set(stored.files)
        if missing:
            raise InferenceContractError(f"NPZ is missing required fields: {sorted(missing)}")
        logits, labels, image_ids = (stored["logits"], stored["labels"], stored["image_ids"])
    logits, labels, image_ids = validate_raw_predictions(logits, labels, image_ids)
    return {"logits": logits, "labels": labels, "image_ids": image_ids}


def collect_predictions(
    model: torch.nn.Module,
    data_loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    num_classes: int = NUM_CLASSES,
    input_size: int = INPUT_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on batches with the project's dictionary-based dataset API."""
    device = torch.device(device)
    model.eval()
    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    image_ids: list[str] = []

    with torch.inference_mode():
        for batch_number, batch in enumerate(data_loader):
            if not isinstance(batch, Mapping):
                raise InferenceContractError(
                    "Each dataset batch must be a mapping with image, label and relative_path fields"
                )
            missing = {"image", "label", "relative_path"} - set(batch.keys())
            if missing:
                raise InferenceContractError(f"Dataset batch missing fields: {sorted(missing)}")

            images = batch["image"]
            labels = batch["label"]
            ids = batch["relative_path"]
            if not isinstance(images, torch.Tensor) or images.ndim != 4:
                got_shape = getattr(images, "shape", None)
                raise InferenceContractError(
                    f"image batch must be a Tensor [B,3,{input_size},{input_size}], got {got_shape}"
                )
            if tuple(images.shape[1:]) != (3, input_size, input_size):
                raise InferenceContractError(
                    f"Expected input shape [B,3,{input_size},{input_size}], got {tuple(images.shape)}"
                )
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels)
            if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
                raise InferenceContractError(
                    f"Batch labels must have shape [{images.shape[0]}], got {tuple(labels.shape)}"
                )
            if isinstance(ids, (str, bytes)):
                ids_list = [ids.decode("utf-8") if isinstance(ids, bytes) else ids]
            else:
                ids_list = [str(item) for item in list(ids)]
            if len(ids_list) != images.shape[0]:
                raise InferenceContractError(
                    f"Batch image IDs count {len(ids_list)} differs from image count {images.shape[0]}"
                )

            output = model(images.to(device, non_blocking=(device.type == "cuda")))
            if not isinstance(output, torch.Tensor):
                raise InferenceContractError("E4 model must return a Tensor of raw logits")
            if output.ndim != 2 or tuple(output.shape) != (images.shape[0], num_classes):
                raise InferenceContractError(
                    f"Model output must have shape [{images.shape[0]},{num_classes}], got {tuple(output.shape)}"
                )
            if not torch.isfinite(output).all().item():
                raise InferenceContractError(f"Model generated NaN/Inf logits in batch {batch_number}")

            logits_parts.append(output.detach().to(device="cpu", dtype=torch.float32).numpy())
            label_parts.append(labels.detach().to(device="cpu", dtype=torch.int64).numpy())
            image_ids.extend(ids_list)

    if not logits_parts:
        raise InferenceContractError("DataLoader produced no samples")
    logits_np = np.concatenate(logits_parts, axis=0)
    labels_np = np.concatenate(label_parts, axis=0)
    logits_np, labels_np, ids_np = validate_raw_predictions(logits_np, labels_np, image_ids, num_classes=num_classes)
    return logits_np, labels_np, ids_np


def compute_task1_metrics(
    logits: Any,
    labels: Any,
    *,
    temperature: float,
    class_names: Mapping[int, str] | None = None,
    num_classes: int = NUM_CLASSES,
) -> dict[str, Any]:
    """Compute classification and E6 calibration metrics without fitting anything."""
    logits_np, labels_np, _unused_ids = validate_raw_predictions(
        logits, labels, [f"sample_{i}" for i in range(len(labels))], num_classes=num_classes
    )
    if not np.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("Temperature loaded from E6 must be finite and positive")

    metrics_module = importlib.import_module("src.evaluation.metrics")
    calibration_module = importlib.import_module("src.calibration.calibration_metrics")
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    logits_tensor = torch.from_numpy(logits_np)
    labels_tensor = torch.from_numpy(labels_np)
    core_classification = metrics_module.classification_metrics(
        logits_tensor, labels_tensor, num_classes=num_classes
    )
    predictions = logits_np.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_np,
        predictions,
        labels=np.arange(num_classes),
        average=None,
        zero_division=0,
    )
    per_class = []
    for class_id in range(num_classes):
        row: dict[str, Any] = {
            "class_id": class_id,
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "support": int(support[class_id]),
        }
        if class_names is not None and class_id in class_names:
            row["class_name"] = str(class_names[class_id])
        per_class.append(row)

    before = calibration_module.compute_calibration_metrics(logits_np, labels_np, ece_bins=ECE_BINS)
    scaled_logits = logits_np / float(temperature)
    after = calibration_module.compute_calibration_metrics(scaled_logits, labels_np, ece_bins=ECE_BINS)
    top1_before = logits_np.argmax(axis=1)
    top1_after = scaled_logits.argmax(axis=1)
    top1_unchanged = bool(np.array_equal(top1_before, top1_after))
    if not top1_unchanged:
        raise AssertionError("Positive scalar Temperature Scaling unexpectedly changed Top-1 predictions")

    matrix = confusion_matrix(labels_np, predictions, labels=np.arange(num_classes))
    return {
        "classification": {
            "top1_accuracy": float(core_classification["top1_accuracy"]),
            "top5_accuracy": float(core_classification["top5_accuracy"]),
            "macro_f1": float(core_classification["macro_f1"]),
            "per_class": per_class,
            "confusion_matrix": matrix.astype(int).tolist(),
        },
        "calibration": {
            "temperature": float(temperature),
            "ece_bins": ECE_BINS,
            "before_temperature_scaling": before,
            "after_temperature_scaling": after,
            "top1_prediction_unchanged": top1_unchanged,
        },
        "n_samples": int(len(labels_np)),
        "n_classes": int(num_classes),
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("PyYAML is required. Install project requirements first.") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Required YAML config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return config


def find_image_root(repo_root: str | Path, split_csv: str | Path) -> Path:
    """Find the Stanford Dogs Images root by checking known paths and sample files."""
    repo_root = Path(repo_root).resolve()
    import pandas as pd

    frame = pd.read_csv(split_csv)
    if frame.empty or "relative_path" not in frame:
        raise ValueError(f"Cannot determine image root from invalid CSV: {split_csv}")
    sample_relatives = [str(value).replace("\\", "/") for value in frame["relative_path"].head(5)]
    candidates = [
        repo_root / "data" / "raw" / "Images",
        repo_root / "data" / "StanfordDogs" / "Images",
        repo_root / "data" / "stanford_dogs" / "Images",
        repo_root / "data" / "Stanford Dogs" / "Images",
        repo_root / "data" / "images",
        repo_root / "data" / "Images",
        repo_root / "data" / "raw" / "stanford_dogs" / "Images",
        repo_root / "data" / "raw" / "stanford-dogs" / "Images",
        repo_root / "data" / "external" / "Images",
        repo_root / "Images",
        repo_root / "stanford_dogs" / "Images",
        repo_root / "StanfordDogs" / "Images",
    ]
    for candidate in candidates:
        if all((candidate / Path(rel)).is_file() for rel in sample_relatives):
            return candidate.resolve()

    # Last-resort search under data/ for the first sample. This avoids assuming
    # one particular local extraction layout for Stanford Dogs.
    data_root = repo_root / "data"
    if data_root.is_dir():
        first_rel = Path(sample_relatives[0])
        searched = 0
        for possible in data_root.rglob(first_rel.name):
            searched += 1
            if searched > 200:
                break
            if not possible.is_file():
                continue
            try:
                base = possible
                for _ in first_rel.parts:
                    base = base.parent
                if all((base / Path(rel)).is_file() for rel in sample_relatives):
                    return base.resolve()
            except OSError:
                continue
    raise FileNotFoundError(
        "Could not auto-detect the Stanford Dogs Images directory. Set images_dir "
        "explicitly to the directory directly containing the breed folders, for example "
        ".../Images. Searched common locations under the repository's data/ directory."
    )


def _get_git_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown (git metadata unavailable)"


def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Support a raw state_dict or common wrapper keys while keeping strict=True."""
    candidate = checkpoint
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping) and value and all(isinstance(k, str) for k in value):
                candidate = value
                break
    if not isinstance(candidate, Mapping) or not candidate:
        raise TypeError("Checkpoint must contain a non-empty model state_dict")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in candidate.items()):
        raise TypeError(
            "Could not identify a pure tensor state_dict inside checkpoint; refusing to load ambiguously"
        )
    return candidate


def load_e4_model(repo_root: str | Path, checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, str]:
    """Build E4 with the project model factory, verify hash, then load strictly."""
    repo_root = Path(repo_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    verified_hash = verify_checkpoint_hash(checkpoint_path)

    # Make repository modules importable whether this is called from root or notebook.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    e4_config_path = repo_root / DEFAULT_E4_CONFIG
    e4_config = _load_yaml(e4_config_path)
    experiment_id = str(e4_config.get("experiment", {}).get("id", "")).upper()
    if experiment_id != "E4":
        raise ValueError(f"Expected E4 config at {e4_config_path}, got experiment={experiment_id!r}")
    model_cfg = e4_config.get("model", {})
    if model_cfg.get("type") != "mobilenet_v2" or int(model_cfg.get("num_classes", -1)) != NUM_CLASSES:
        raise ValueError("E4 config does not describe the approved MobileNetV2 30-class model")

    # Build the same architecture without downloading ImageNet weights: the approved
    # E4 checkpoint contains the full learned state and is loaded immediately below.
    # This is an in-memory initialization-only override; no YAML/config file is changed.
    architecture_config = copy.deepcopy(e4_config)
    architecture_config["model"]["weights"] = "none"
    factory_module = importlib.import_module("src.models.model_factory")
    model = factory_module.build_model(architecture_config)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Project model builder did not return torch.nn.Module")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This task requires a PyTorch version supporting torch.load(..., weights_only=True). "
            "Upgrade to a compatible project environment; do not disable weights_only."
        ) from exc
    state_dict = _extract_state_dict(checkpoint)
    # The strict flag is required by the Task 04 assignment.
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, verified_hash


def _load_temperature(artifact_path: Path) -> tuple[float, dict[str, Any]]:
    if not artifact_path.is_file():
        raise FileNotFoundError(f"E6 Temperature artifact not found: {artifact_path}")
    with artifact_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("E6 Temperature artifact must be a JSON object")
    if payload.get("experiment") != "E6":
        raise ValueError("Temperature artifact does not identify experiment E6")
    if payload.get("model_experiment") not in (None, "E4"):
        raise ValueError("E6 Temperature artifact was not produced for E4")
    if payload.get("checkpoint_seed") not in (None, 42):
        raise ValueError("E6 Temperature artifact does not match E4 seed 42")
    if payload.get("split") not in (None, "calibration"):
        raise ValueError("Expected the approved E6 artifact fit on Calibration split")
    calibration_module = importlib.import_module("src.calibration.temperature_scaling")
    temperature = float(calibration_module.load_temperature(artifact_path))
    if temperature != EXPECTED_TEMPERATURE:
        raise ValueError(
            "E6 Temperature mismatch with Configuration Lock. "
            f"Expected={EXPECTED_TEMPERATURE!r}; actual={temperature!r}; path={artifact_path}"
        )
    return temperature, payload


def _assert_npz_alignment_with_csv(
    split_csv: Path, image_ids: np.ndarray, labels: np.ndarray
) -> dict[int, str]:
    import pandas as pd

    frame = pd.read_csv(split_csv)
    if not {"relative_path", "class_id", "breed_name", "source_split"}.issubset(frame.columns):
        raise InferenceContractError("Validation CSV does not match the project's required split schema")
    csv_ids = frame["relative_path"].astype(str).tolist()
    csv_labels = frame["class_id"].astype(np.int64).to_numpy()
    if len(csv_ids) != len(set(csv_ids)):
        raise InferenceContractError("Validation CSV contains duplicate relative_path values")
    if csv_ids != image_ids.astype(str).tolist():
        raise InferenceContractError(
            "Predicted image_ids do not preserve the approved Validation CSV order/values"
        )
    if not np.array_equal(csv_labels, labels):
        raise InferenceContractError("Predicted labels do not align with class_id in Validation CSV")
    names: dict[int, str] = {}
    pairs = frame[["class_id", "breed_name"]].drop_duplicates()
    for _, row in pairs.iterrows():
        class_id = int(row["class_id"])
        name = str(row["breed_name"])
        if class_id in names and names[class_id] != name:
            raise InferenceContractError(f"Validation CSV maps class_id={class_id} to multiple breed names")
        names[class_id] = name
    return names


def run_final_inference(
    *,
    repo_root: str | Path,
    split_name: str = "validation",
    split_csv: str | Path | None = None,
    images_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    temperature_artifact: str | Path | None = None,
    output_dir: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run the complete Task 1 pipeline on the approved Validation split."""
    repo_root = Path(repo_root).expanduser().resolve()
    assert_task04_split_allowed(split_name)
    if str(split_name).strip().lower() != "validation":
        raise PermissionError("End-to-end model inference in Task 04 is configured for Validation only")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if not (repo_root / DEFAULT_DATA_CONFIG).is_file():
        raise FileNotFoundError(f"Repository root is incorrect or missing {DEFAULT_DATA_CONFIG}: {repo_root}")
    if not (repo_root / "src" / "models" / "model_factory.py").is_file():
        raise FileNotFoundError("Could not find src/models/model_factory.py under repo_root")

    split_csv_path = _require_validation_csv(
        repo_root,
        split_csv if split_csv is not None else repo_root / DEFAULT_SPLIT_CSV,
    )
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else repo_root / DEFAULT_CHECKPOINT
    if not checkpoint.is_absolute():
        checkpoint = repo_root / checkpoint
    checkpoint = checkpoint.resolve()
    temp_path = Path(temperature_artifact) if temperature_artifact is not None else repo_root / DEFAULT_TEMPERATURE_ARTIFACT
    if not temp_path.is_absolute():
        temp_path = repo_root / temp_path
    temp_path = temp_path.resolve()
    outputs = _safe_output_dir(repo_root, output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR)

    if device is None or str(device).lower() == "auto":
        runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        runtime_device = torch.device(device)
        if runtime_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    # Check all artifacts before touching the model or dataset.
    actual_checkpoint_hash = verify_checkpoint_hash(checkpoint)
    temperature, temperature_payload = _load_temperature(temp_path)
    class_mapping_path = repo_root / "data" / "metadata" / "class_mapping.json"
    if not class_mapping_path.is_file():
        raise FileNotFoundError(f"Approved class mapping not found: {class_mapping_path}")
    class_mapping_sha256 = sha256_file(class_mapping_path)

    data_payload = _load_yaml(repo_root / DEFAULT_DATA_CONFIG)
    data_config = data_payload.get("data", data_payload)
    if int(data_config.get("num_classes", -1)) != NUM_CLASSES:
        raise ValueError("configs/data.yaml must define exactly 30 classes")

    if images_dir is None:
        images_root = find_image_root(repo_root, split_csv_path)
    else:
        images_root = Path(images_dir).expanduser()
        if not images_root.is_absolute():
            images_root = repo_root / images_root
        images_root = images_root.resolve()
        if not images_root.is_dir():
            raise FileNotFoundError(f"Images directory not found: {images_root}")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    dataset_module = importlib.import_module("src.data.dataset")
    transforms_module = importlib.import_module("src.data.transforms")
    transform = transforms_module.build_transforms(data_payload, split="validation")
    dataset = dataset_module.DogBreedDataset(
        csv_path=split_csv_path,
        images_dir=images_root,
        transform=transform,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(runtime_device.type == "cuda"),
        drop_last=False,
    )

    model, loaded_checkpoint_hash = load_e4_model(repo_root, checkpoint, runtime_device)
    if loaded_checkpoint_hash != actual_checkpoint_hash:
        raise RuntimeError("Checkpoint SHA-256 unexpectedly changed during model setup")
    logits, labels, image_ids = collect_predictions(
        model,
        data_loader,
        device=runtime_device,
        num_classes=NUM_CLASSES,
        input_size=INPUT_SIZE,
    )
    class_names = _assert_npz_alignment_with_csv(split_csv_path, image_ids, labels)
    metrics = compute_task1_metrics(
        logits,
        labels,
        temperature=temperature,
        class_names=class_names,
        num_classes=NUM_CLASSES,
    )

    outputs.mkdir(parents=True, exist_ok=True)
    npz_path = save_raw_predictions(outputs / "validation_raw_predictions.npz", logits, labels, image_ids)

    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        relative_checkpoint = checkpoint.relative_to(repo_root).as_posix()
    except ValueError:
        relative_checkpoint = str(checkpoint)
    try:
        relative_temperature_artifact = temp_path.relative_to(repo_root).as_posix()
    except ValueError:
        relative_temperature_artifact = str(temp_path)
    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = "unavailable"

    metadata = {
        "experiment_id": "E4_TASK04_FINAL_INFERENCE",
        "task": "Task 04 / Task 1 - Final Classification & Calibration Inference",
        "split": "validation",
        "data_status": "validation",
        "num_samples": int(len(labels)),
        "num_classes": NUM_CLASSES,
        "class_mapping_path": "data/metadata/class_mapping.json",
        "class_mapping_sha256": class_mapping_sha256,
        "git_sha": _get_git_sha(repo_root),
        "checkpoint_path": relative_checkpoint,
        "checkpoint_sha256": loaded_checkpoint_hash,
        "model_initialization": "E4 architecture built with weights='none'; full approved checkpoint then loaded with strict=True",
        "temperature_artifact_path": relative_temperature_artifact,
        "temperature_source_experiment": temperature_payload.get("experiment"),
        "temperature_source_split": temperature_payload.get("split"),
        "temperature": float(temperature),
        "preprocessing": {
            "config": DEFAULT_DATA_CONFIG.as_posix(),
            "transform_builder": "src.data.transforms.build_transforms",
            "split_argument": "validation",
            "expected_tensor_shape": [3, INPUT_SIZE, INPUT_SIZE],
            "evaluation_resize": data_payload["preprocessing"]["evaluation"]["resize"],
            "evaluation_center_crop": data_payload["preprocessing"]["evaluation"]["center_crop"],
            "normalization": data_payload["preprocessing"]["normalization"],
        },
        "runtime": {
            "timestamp_utc": timestamp,
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "torchvision_version": torchvision_version,
            "device": str(runtime_device),
            "cuda_available": bool(torch.cuda.is_available()),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
        },
        "outputs": {
            "raw_predictions_npz": str(npz_path.relative_to(repo_root).as_posix()),
        },
        "safety": {
            "official_test_used": False,
            "model_trained_or_updated": False,
            "temperature_refit": False,
            "official_test_guard": "validation-only runner",
        },
    }

    metrics_path = outputs / "validation_metrics.json"
    metadata_path = outputs / "validation_metadata.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")

    reliability_path: Path | None = None
    try:
        calibration_module = importlib.import_module("src.calibration.calibration_metrics")
        reliability_path = calibration_module.plot_reliability_diagram(
            logits,
            logits / float(temperature),
            labels,
            n_bins=ECE_BINS,
            output_path=outputs / "validation_reliability_diagram.png",
            title="E4 Validation: Before vs After E6 Temperature Scaling",
        )
    except ImportError:
        # Metrics are still complete if optional matplotlib is unavailable.
        reliability_path = None

    return {
        "logits": logits,
        "labels": labels,
        "image_ids": image_ids,
        "metrics": metrics,
        "metadata": metadata,
        "temperature": temperature,
        "checkpoint_sha256": loaded_checkpoint_hash,
        "images_dir": str(images_root),
        "outputs_dir": str(outputs),
        "paths": {
            "raw_predictions_npz": str(npz_path),
            "metrics_json": str(metrics_path),
            "metadata_json": str(metadata_path),
            "reliability_diagram": str(reliability_path) if reliability_path else None,
        },
    }
