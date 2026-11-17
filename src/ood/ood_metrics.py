"""ID/OOD metrics for selective classification with an OOD rejection step.

Conventions
- ``accepted``: final ACCEPT decision of the method being evaluated.
- ID metrics use only in-distribution samples (labels 0-29 known).
- OOD False Accept Rate = accepted OOD / all OOD; OOD Rejection Rate = 1 - FAR.
- AUROC treats OOD as the POSITIVE class; higher score => more OOD.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _bool_vector(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if array.dtype != bool:
        raise ValueError(f"{name} must be boolean")
    return array


def id_selective_metrics(accepted, predictions, labels) -> dict:
    accepted = _bool_vector(accepted, "accepted")
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if not (accepted.shape == predictions.shape == labels.shape) or accepted.size == 0:
        raise ValueError("accepted, predictions and labels must be equal-length nonempty vectors")
    n_accept = int(accepted.sum())
    correct = predictions == labels
    selective_accuracy = float(correct[accepted].mean()) if n_accept else None
    return {
        "n": int(accepted.size),
        "n_accept": n_accept,
        "id_coverage": n_accept / accepted.size,
        "id_selective_accuracy": selective_accuracy,
        "id_selective_risk": None if selective_accuracy is None else 1.0 - selective_accuracy,
        "id_full_accuracy": float(correct.mean()),
    }


def ood_rejection_metrics(accepted) -> dict:
    accepted = _bool_vector(accepted, "accepted")
    if accepted.size == 0:
        raise ValueError("accepted must be nonempty")
    n_accept = int(accepted.sum())
    far = n_accept / accepted.size
    return {"n": int(accepted.size), "n_accept": n_accept, "n_reject": int(accepted.size) - n_accept,
            "ood_false_accept_rate": far, "ood_rejection_rate": 1.0 - far}


def ood_auroc(id_scores, ood_scores) -> float:
    """AUROC with OOD as positive; scores must be oriented so higher = more OOD."""
    id_scores = np.asarray(id_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    if id_scores.ndim != 1 or ood_scores.ndim != 1 or not id_scores.size or not ood_scores.size:
        raise ValueError("score vectors must be nonempty and 1D")
    scores = np.concatenate([id_scores, ood_scores])
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    labels = np.concatenate([np.zeros(id_scores.size), np.ones(ood_scores.size)])
    return float(roc_auc_score(labels, scores))


def score_summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    q = np.quantile(values, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {"mean": float(values.mean()), "std": float(values.std()),
            **{f"p{p}": float(v) for p, v in zip((0, 10, 25, 50, 75, 90, 100), q)}}
