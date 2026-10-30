"""E9 error-analysis utilities for DogBreedID.

This module is intentionally model-agnostic. It consumes saved logits/labels/image_ids
and computes aggregate error-analysis artifacts without rerunning model inference.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

VALIDATION_SPLIT_CSV = Path(__file__).resolve().parents[2] / "data/splits/validation.csv"


def _as_int_labels(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must have shape [N], got {labels.shape}")
    if labels.size == 0:
        raise ValueError("labels must not be empty")
    if labels.dtype.kind not in "iu":
        raise TypeError(f"labels must use an integer dtype, got {labels.dtype}")
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(f"labels must be in range [0, {num_classes - 1}]")
    return labels


def validate_predictions(
    logits: np.ndarray,
    labels: np.ndarray,
    image_ids: np.ndarray,
    num_classes: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return logits, labels and string IDs without changing row order."""
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise ValueError(f"logits must have shape [N, {num_classes}], got {logits.shape}")
    if logits.shape[0] == 0:
        raise ValueError("logits must contain at least one sample")
    if logits.dtype.kind not in "fiu":
        raise TypeError(f"logits must be numeric, got {logits.dtype}")
    logits = logits.astype(np.float64, copy=False)
    if not np.isfinite(logits).all():
        raise ValueError("logits contain NaN or Inf")

    labels = _as_int_labels(labels, num_classes)
    if labels.shape[0] != logits.shape[0]:
        raise ValueError(f"labels length {labels.shape[0]} does not match logits N={logits.shape[0]}")

    image_ids = np.asarray(image_ids)
    if image_ids.ndim != 1 or image_ids.shape[0] != logits.shape[0]:
        raise ValueError(f"image_ids must have shape [{logits.shape[0]}], got {image_ids.shape}")
    if image_ids.dtype.kind not in "US":
        raise TypeError(
            "image_ids must be a non-object string array (dtype kind U or S); "
            f"got {image_ids.dtype}. Save NPZ string IDs without pickle."
        )
    if image_ids.dtype.kind == "S":
        image_ids = np.char.decode(image_ids, encoding="utf-8")
    image_ids = image_ids.astype(str, copy=False)
    stripped = np.char.strip(image_ids)
    if np.any(stripped == ""):
        raise ValueError("image_ids must not contain empty strings")
    if np.unique(image_ids).size != image_ids.size:
        raise ValueError("image_ids must be unique")
    return logits, labels, image_ids


def load_predictions_npz(path: str | Path, num_classes: int = 30) -> dict[str, np.ndarray]:
    """Load E9 NPZ contract with allow_pickle=False and validate every required field."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction NPZ not found: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            required = {"logits", "labels", "image_ids"}
            missing = sorted(required - keys)
            if missing:
                raise ValueError(
                    f"NPZ is missing required keys {missing}. Available keys: {sorted(keys)}. "
                    "Do not guess key mappings; inspect the artifact schema and get approval before adapting it."
                )
            logits = np.array(archive["logits"], copy=True)
            labels = np.array(archive["labels"], copy=True)
            image_ids = np.array(archive["image_ids"], copy=True)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not safely read NPZ with allow_pickle=False: {exc}") from exc

    logits, labels, image_ids = validate_predictions(logits, labels, image_ids, num_classes)
    return {"logits": logits, "labels": labels, "image_ids": image_ids}


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 30,
) -> np.ndarray:
    """Return an integer matrix where rows=true labels and columns=predictions."""
    true = _as_int_labels(np.asarray(y_true), num_classes)
    pred = _as_int_labels(np.asarray(y_pred), num_classes)
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred shapes differ: {true.shape} vs {pred.shape}")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def per_class_metrics(confusion: np.ndarray, class_names: Sequence[str]) -> list[dict[str, Any]]:
    """Compute per-class precision, recall, F1 and support; zero division -> 0."""
    cm = np.asarray(confusion)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("confusion matrix must be square")
    n_classes = cm.shape[0]
    if len(class_names) != n_classes:
        raise ValueError(f"Expected {n_classes} class names, received {len(class_names)}")
    if cm.dtype.kind not in "iu" or np.any(cm < 0):
        raise ValueError("confusion matrix must contain non-negative integer counts")

    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    true_positive = np.diag(cm)
    rows: list[dict[str, Any]] = []
    for idx in range(n_classes):
        tp = int(true_positive[idx])
        fp = int(col_sum[idx] - tp)
        fn = int(row_sum[idx] - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "class_id": idx,
            "class_name": str(class_names[idx]),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "support": int(row_sum[idx]),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        })
    return rows


def top_confusion_pairs(
    confusion: np.ndarray,
    class_names: Sequence[str],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rank directional True -> Predicted errors; ties sort by true then predicted ID."""
    cm = np.asarray(confusion)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("confusion matrix must be square")
    if len(class_names) != cm.shape[0]:
        raise ValueError("class_names length must match confusion matrix")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    support = cm.sum(axis=1)
    pairs: list[dict[str, Any]] = []
    for true_id in range(cm.shape[0]):
        for pred_id in range(cm.shape[1]):
            if true_id == pred_id or int(cm[true_id, pred_id]) == 0:
                continue
            count = int(cm[true_id, pred_id])
            pairs.append({
                "true_id": true_id,
                "true_breed": str(class_names[true_id]),
                "predicted_id": pred_id,
                "predicted_breed": str(class_names[pred_id]),
                "error_count": count,
                "true_class_support": int(support[true_id]),
                "error_rate_by_true_support": float(count / support[true_id]) if support[true_id] else 0.0,
            })
    pairs.sort(key=lambda row: (-row["error_count"], row["true_id"], row["predicted_id"]))
    return pairs[:top_k]


def stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax(logits / temperature)."""
    if not isinstance(temperature, (float, int)) or not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("logits must have shape [N, C] with N>0 and C>=2")
    if not np.isfinite(scores).all():
        raise ValueError("logits contain NaN or Inf")
    scaled = scores / float(temperature)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp_scores = np.exp(scaled)
    denom = exp_scores.sum(axis=1, keepdims=True)
    probs = exp_scores / denom
    if not np.isfinite(probs).all():
        raise ValueError("softmax probabilities contain NaN or Inf")
    return probs


def e7_error_breakdown(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    threshold: float,
) -> dict[str, Any]:
    """Aggregate correctness x ACCEPT/REJECT counts using the locked E6/E7 values."""
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or not np.isfinite(logits).all():
        raise ValueError("logits must be a finite [N, C] array")
    labels = _as_int_labels(np.asarray(labels), logits.shape[1])
    if labels.shape[0] != logits.shape[0]:
        raise ValueError("labels length must match logits")
    probs = stable_softmax(logits, temperature)
    predictions = np.argmax(logits, axis=1)
    confidence = probs.max(axis=1)
    accepted = confidence >= float(threshold)
    correct = predictions == labels

    groups = {
        "correct_accept": int(np.sum(correct & accepted)),
        "incorrect_accept": int(np.sum(~correct & accepted)),
        "correct_reject": int(np.sum(correct & ~accepted)),
        "incorrect_reject": int(np.sum(~correct & ~accepted)),
    }
    n = int(labels.size)
    n_accept = groups["correct_accept"] + groups["incorrect_accept"]
    n_correct_accept = groups["correct_accept"]
    return {
        "temperature": float(temperature),
        "threshold": float(threshold),
        "num_samples": n,
        "num_accept": int(n_accept),
        "num_reject": int(n - n_accept),
        "coverage": float(n_accept / n),
        "selective_accuracy": float(n_correct_accept / n_accept) if n_accept else None,
        "selective_risk": float(1.0 - n_correct_accept / n_accept) if n_accept else None,
        "groups": groups,
    }


def select_representative_cases(
    logits: np.ndarray,
    labels: np.ndarray,
    image_ids: np.ndarray,
    class_names: Sequence[str],
    temperature: float,
    threshold: float,
    max_cases: int = 15,
) -> list[dict[str, Any]]:
    """Select deterministic validation/synthetic examples; caller must not export for Official."""
    if max_cases < 1:
        raise ValueError("max_cases must be >= 1")
    logits, labels, image_ids = validate_predictions(logits, labels, image_ids, len(class_names))
    probs = stable_softmax(logits, temperature)
    pred = np.argmax(logits, axis=1)
    conf = probs.max(axis=1)
    accepted = conf >= threshold
    correct = pred == labels

    categories: list[tuple[str, np.ndarray, bool]] = [
        ("incorrect_high_confidence", (~correct) & (conf >= 0.8), True),
        ("incorrect_low_confidence", (~correct) & (conf < threshold), False),
        ("correct_high_confidence", correct & (conf >= 0.8), False),
        ("incorrect_accepted", (~correct) & accepted, True),
        ("rejected", ~accepted, False),
    ]
    pools: dict[str, list[int]] = {}
    for category, mask, descending_conf in categories:
        candidates = np.flatnonzero(mask)
        pools[category] = sorted(candidates.tolist(), key=lambda i: ((-conf[i] if descending_conf else conf[i]), str(image_ids[i])))

    # Reserve one distinct example per category when possible. Reassign overlapping
    # candidates so an early category cannot steal a later category's only example.
    owners: dict[int, str] = {}

    def reserve(category: str, seen: set[int]) -> bool:
        for idx in pools[category]:
            if idx in seen:
                continue
            seen.add(idx)
            if idx not in owners or reserve(owners[idx], seen):
                owners[idx] = category
                return True
        return False

    for category in pools:
        if len(owners) >= max_cases:
            break
        reserve(category, set())

    selected_category = dict(owners)
    selected = [idx for category in pools for idx in owners if owners[idx] == category]
    # Fill the remaining quota in rounds, skipping exhausted categories and IDs
    # already reserved by another category. Every ordering tie is broken by ID.
    while len(selected) < max_cases:
        added = False
        for category, pool in pools.items():
            idx = next((i for i in pool if i not in selected_category), None)
            if idx is None:
                continue
            selected.append(idx)
            selected_category[idx] = category
            added = True
            if len(selected) >= max_cases:
                break
        if not added:
            break

    rows: list[dict[str, Any]] = []
    for idx in selected:
        rows.append({
            "image_id": str(image_ids[idx]),
            "true_id": int(labels[idx]),
            "true_label": str(class_names[int(labels[idx])]),
            "predicted_id": int(pred[idx]),
            "predicted_label": str(class_names[int(pred[idx])]),
            "calibrated_confidence": float(conf[idx]),
            "e7_status": "ACCEPT" if accepted[idx] else "REJECT",
            "correct": bool(correct[idx]),
            "case_category": selected_category[idx],
        })
    return rows


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def validate_ids_against_split(
    image_ids: np.ndarray,
    labels: np.ndarray,
    split_csv: str | Path,
    id_column: str = "image_id",
    label_column: str = "label",
) -> dict[str, Any]:
    """Compare saved prediction IDs/labels to an explicitly supplied split CSV.

    This is a validation helper only. It never writes image IDs to output files.
    Column names must be supplied/verified against the repository's actual CSV.
    """
    import csv

    path = Path(split_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Split reference CSV not found: {path}")
    ids = np.asarray(image_ids).astype(str)
    y = np.asarray(labels)
    if ids.ndim != 1 or y.ndim != 1 or ids.size != y.size:
        raise ValueError("image_ids and labels must be 1D arrays with matching lengths")
    if not ids.size or y.dtype.kind not in "iu":
        raise ValueError("Predictions must contain non-empty integer labels")
    if any(not key or key != key.strip() for key in ids.tolist()):
        raise ValueError("Prediction IDs must be non-empty and have no surrounding whitespace")
    lookup: dict[str, int] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or id_column not in reader.fieldnames or label_column not in reader.fieldnames:
            raise ValueError(f"Split CSV must contain columns {id_column!r} and {label_column!r}; found {reader.fieldnames}")
        for line_no, row in enumerate(reader, start=2):
            key = str(row[id_column]).strip()
            if not key:
                raise ValueError(f"Empty image ID in split CSV at line {line_no}")
            if key in lookup:
                raise ValueError(f"Duplicate image ID in split CSV: {key!r}")
            try:
                value = int(row[label_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-integer label in split CSV at line {line_no}") from exc
            lookup[key] = value
    missing = [str(key) for key in ids if str(key) not in lookup]
    mismatches = [str(key) for key, label in zip(ids, y) if str(key) in lookup and lookup[str(key)] != int(label)]
    unpredicted = set(lookup) - set(ids.tolist())
    duplicates = int(ids.size - np.unique(ids).size)
    order_matches = ids.tolist() == list(lookup)
    return {
        "num_prediction_rows": int(ids.size),
        "num_split_rows": int(len(lookup)),
        "num_missing_ids": int(len(missing)),
        "num_label_mismatches": int(len(mismatches)),
        "num_unpredicted_ids": len(unpredicted),
        "num_duplicate_ids": duplicates,
        "order_matches": order_matches,
        "match": not missing and not mismatches and not unpredicted and not duplicates and order_matches,
        # Individual IDs are deliberately not returned or serialized.
    }


def validate_analysis_split(
    image_ids: np.ndarray,
    labels: np.ndarray,
    split: str,
    split_reference_csv: str | Path | None = None,
) -> dict[str, Any] | None:
    """Fail before exporting any per-image rows unless their split is established.

    Synthetic IDs use the explicit ``synthetic/`` fixture namespace, never real
    dataset relative paths. This is a fixture contract, not proof of how logits
    were produced. Validation always binds the full canonical CSV, in its order.
    """
    if split == "validation":
        canonical = VALIDATION_SPLIT_CSV
        if canonical.resolve() != canonical.absolute():
            raise PermissionError("Canonical Validation reference must not redirect outside its repository path")
        if split_reference_csv is not None and Path(split_reference_csv).resolve() != canonical.resolve():
            raise PermissionError("Validation requires the canonical repository split reference")
        check = validate_ids_against_split(image_ids, labels, canonical, "relative_path", "class_id")
        if not check["match"]:
            raise ValueError(f"Predictions do not match canonical Validation IDs, labels, count or order: {check}")
        return check
    if split == "synthetic":
        if split_reference_csv is not None:
            raise ValueError("Synthetic fixtures must not use a real split reference CSV")
        for image_id in image_ids:
            value = str(image_id)
            parts = PurePosixPath(value).parts
            if (not value.startswith("synthetic/") or len(parts) < 2
                    or any(part in {"", ".", ".."} for part in value.split("/"))
                    or "\\" in value or ":" in value
                    or re.search(r"n\d{8}", value, flags=re.IGNORECASE)):
                raise PermissionError("Synthetic mode requires artificial IDs in the synthetic/ fixture namespace")
        return None
    if split == "official":
        if split_reference_csv is not None:
            raise PermissionError("Do not provide an Official split CSV; use only approved saved predictions")
        return None
    raise ValueError("split must be one of: synthetic, validation, official")


def load_class_names(path: str | Path, num_classes: int = 30) -> list[str]:
    """Read common class_mapping.json layouts without changing class order."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Class mapping not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    def names_from_sequence(items: Any) -> list[str] | None:
        if not isinstance(items, list) or len(items) != num_classes:
            return None
        names: list[str] = []
        for idx, item in enumerate(items):
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                value = item.get("class_name", item.get("name", item.get("breed", item.get("label"))))
                if value is None:
                    return None
                names.append(str(value))
            else:
                return None
        return names

    if isinstance(raw, list):
        result = names_from_sequence(raw)
        if result:
            return result
    if isinstance(raw, dict):
        for key in ("classes", "class_names", "breeds"):
            result = names_from_sequence(raw.get(key))
            if result:
                return result
        for key in ("class_to_idx", "class_to_index", "breed_to_idx"):
            mapping = raw.get(key)
            if isinstance(mapping, dict) and len(mapping) == num_classes:
                try:
                    ordered = sorted(((int(v), str(k)) for k, v in mapping.items()), key=lambda x: x[0])
                    if [idx for idx, _ in ordered] == list(range(num_classes)):
                        return [name for _, name in ordered]
                except (TypeError, ValueError):
                    pass
        for key in ("idx_to_class", "index_to_class", "idx_to_breed"):
            mapping = raw.get(key)
            if isinstance(mapping, dict) and len(mapping) == num_classes:
                try:
                    ordered = sorted(((int(k), str(v)) for k, v in mapping.items()), key=lambda x: x[0])
                    if [idx for idx, _ in ordered] == list(range(num_classes)):
                        return [name for _, name in ordered]
                except (TypeError, ValueError):
                    pass
        # Also support a direct {class_name: integer_index} mapping.
        if len(raw) == num_classes:
            try:
                ordered = sorted(((int(v), str(k)) for k, v in raw.items()), key=lambda x: x[0])
                if [idx for idx, _ in ordered] == list(range(num_classes)):
                    return [name for _, name in ordered]
            except (TypeError, ValueError):
                pass
            # Or direct {integer_index: class_name} mapping.
            try:
                ordered = sorted(((int(k), str(v)) for k, v in raw.items()), key=lambda x: x[0])
                if [idx for idx, _ in ordered] == list(range(num_classes)):
                    return [name for _, name in ordered]
            except (TypeError, ValueError):
                pass
    raise ValueError(
        f"Unsupported class mapping schema in {path}. Expected a 30-class list or a mapping with "
        "class_to_idx/idx_to_class/classes; inspect this file before adapting the loader."
    )


def save_confusion_matrix_figures(
    confusion: np.ndarray,
    class_names: Sequence[str],
    output_dir: str | Path,
    prefix: str = "confusion_matrix",
) -> list[str]:
    """Save count and row-normalized heatmaps; imports matplotlib lazily."""
    import matplotlib.pyplot as plt

    cm = np.asarray(confusion)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or len(class_names) != cm.shape[0]:
        raise ValueError("confusion matrix and class_names dimensions do not match")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    normalized = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) != 0)
    for suffix, matrix, title, fmt in (
        ("", cm, "Confusion Matrix — counts", "d"),
        ("_normalized", normalized, "Confusion Matrix — row normalized", ".2f"),
    ):
        fig, ax = plt.subplots(figsize=(17, 15))
        image = ax.imshow(matrix, interpolation="nearest", aspect="auto")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ticks = np.arange(len(class_names))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        fig.tight_layout()
        path = output_dir / f"{prefix}{suffix}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def save_e9_artifacts(
    logits: np.ndarray,
    labels: np.ndarray,
    image_ids: np.ndarray,
    class_names: Sequence[str],
    output_dir: str | Path,
    split: str,
    temperature: float,
    threshold: float,
    reference_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write aggregate E9 outputs; per-image IDs are emitted only for non-Official splits."""
    num_classes = len(class_names)
    logits, labels, image_ids = validate_predictions(logits, labels, image_ids, num_classes)
    split = split.lower().strip()
    split_check = validate_analysis_split(image_ids, labels, split)
    predictions = np.argmax(logits, axis=1)
    cm = compute_confusion_matrix(labels, predictions, num_classes)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Always store aggregate outputs. Never export raw Official per-image rows.
    cm_csv = out / "confusion_matrix.csv"
    with cm_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label / predicted_label", *class_names])
        for idx, row in enumerate(cm):
            writer.writerow([class_names[idx], *[int(v) for v in row]])

    metrics_rows = per_class_metrics(cm, class_names)
    metrics_fields = ["class_id", "class_name", "precision", "recall", "f1_score", "support", "true_positive", "false_positive", "false_negative"]
    write_csv(out / "per_class_metrics.csv", metrics_rows, metrics_fields)

    top_pairs = top_confusion_pairs(cm, class_names, top_k=5)
    pair_fields = ["true_id", "true_breed", "predicted_id", "predicted_breed", "error_count", "true_class_support", "error_rate_by_true_support"]
    write_csv(out / "top5_confusion_pairs.csv", top_pairs, pair_fields)

    breakdown = e7_error_breakdown(logits, labels, temperature, threshold)
    (out / "e7_error_breakdown.json").write_text(json.dumps(breakdown, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    figures = save_confusion_matrix_figures(cm, class_names, out)

    n = int(labels.size)
    correct_count = int(np.trace(cm))
    summary: dict[str, Any] = {
        "experiment": "E9",
        "dataset_split": split,
        "num_samples": n,
        "num_classes": num_classes,
        "top1_correct": correct_count,
        "top1_incorrect": int(n - correct_count),
        "top1_accuracy": float(correct_count / n),
        "macro_precision": float(np.mean([r["precision"] for r in metrics_rows])),
        "macro_recall": float(np.mean([r["recall"] for r in metrics_rows])),
        "macro_f1": float(np.mean([r["f1_score"] for r in metrics_rows])),
        "temperature": float(temperature),
        "threshold": float(threshold),
        "artifacts": [
            "confusion_matrix.csv", "confusion_matrix.png", "confusion_matrix_normalized.png",
            "per_class_metrics.csv", "top5_confusion_pairs.csv", "e7_error_breakdown.json",
        ],
        "reference_checks": {},
        "privacy_note": "Aggregate outputs only; per-image IDs/predictions are not written for Official split.",
    }
    if split_check is not None:
        summary["split_reference_check"] = split_check
    if reference_metrics:
        actuals = {
            "num_samples": n,
            "top1_correct": correct_count,
            "top1_incorrect": int(n - correct_count),
            "top1_accuracy": float(correct_count / n),
            "macro_f1": summary["macro_f1"],
            "e7_groups": breakdown["groups"],
        }
        for key, expected in reference_metrics.items():
            actual = actuals.get(key)
            if actual is None:
                continue
            if isinstance(actual, dict) and isinstance(expected, dict):
                match = actual == expected
            elif isinstance(actual, (float, np.floating)):
                match = math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-4)
            else:
                match = actual == expected
            summary["reference_checks"][key] = {"expected": expected, "actual": actual, "match": bool(match)}
        if any(not check["match"] for check in summary["reference_checks"].values()):
            summary["warnings"] = ["One or more values differ from Task 06 reference values; inspect the source artifact and protocol. No values were modified to force a match."]

    # Representative cases are useful for validation reports only. Official IDs are never exported.
    if split.lower() in {"validation", "synthetic"}:
        cases = select_representative_cases(logits, labels, image_ids, class_names, temperature, threshold, max_cases=15)
        case_fields = ["image_id", "true_id", "true_label", "predicted_id", "predicted_label", "calibrated_confidence", "e7_status", "correct", "case_category"]
        write_csv(out / "representative_cases.csv", cases, case_fields)
        summary["artifacts"].append("representative_cases.csv")
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
