"""Calibration metrics and reliability diagram for multiclass classifiers.

Definitions used here:
- NLL: mean cross-entropy on logits and integer labels.
- ECE: top-label expected calibration error, with right-closed equal-width bins
  (0, 1/B], (1/B, 2/B], ..., ((B-1)/B, 1].
- Multiclass Brier: mean over samples of the SUM of squared errors across all
  classes (not divided by number of classes).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _as_numpy_logits_labels(logits: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    logits_np = np.asarray(logits)
    labels_np = np.asarray(labels)
    if logits_np.ndim != 2:
        raise ValueError(f"logits must have shape [N, C], got {logits_np.shape}")
    if logits_np.shape[0] < 1 or logits_np.shape[1] < 2:
        raise ValueError("logits must contain at least one sample and two classes")
    if not np.issubdtype(logits_np.dtype, np.number) or not np.isfinite(logits_np).all():
        raise ValueError("logits must be numeric and finite")
    if labels_np.ndim != 1:
        raise ValueError(f"labels must have shape [N], got {labels_np.shape}")
    if labels_np.shape[0] != logits_np.shape[0]:
        raise ValueError("logits and labels have different numbers of samples")
    if labels_np.size == 0:
        raise ValueError("labels must not be empty")
    if not np.issubdtype(labels_np.dtype, np.integer):
        if not np.isfinite(labels_np).all() or not np.equal(labels_np, np.floor(labels_np)).all():
            raise ValueError("labels must contain finite integer class IDs")
    labels_np = labels_np.astype(np.int64, copy=False)
    if labels_np.min() < 0 or labels_np.max() >= logits_np.shape[1]:
        raise ValueError("labels contain class IDs outside [0, C-1]")
    return logits_np.astype(np.float64, copy=False), labels_np


def softmax_numpy(logits: Any) -> np.ndarray:
    """Numerically stable softmax for a finite [N, C] logits array."""
    logits_np = np.asarray(logits, dtype=np.float64)
    if logits_np.ndim != 2 or not np.isfinite(logits_np).all():
        raise ValueError("logits must be a finite [N, C] array")
    shifted = logits_np - np.max(logits_np, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    denom = exp_values.sum(axis=1, keepdims=True)
    probs = exp_values / denom
    if not np.isfinite(probs).all():
        raise ValueError("softmax produced NaN or Inf")
    return probs


def reliability_bins(
    logits: Any, labels: Any, *, n_bins: int = 15
) -> list[dict[str, float | int]]:
    """Return count, mean confidence, and accuracy for each equal-width bin."""
    if isinstance(n_bins, bool) or int(n_bins) < 1:
        raise ValueError("n_bins must be an integer >= 1")
    logits_np, labels_np = _as_numpy_logits_labels(logits, labels)
    probs = softmax_numpy(logits_np)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels_np).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    result: list[dict[str, float | int]] = []
    for index in range(int(n_bins)):
        lower, upper = float(edges[index]), float(edges[index + 1])
        # All bins are right-closed; softmax confidence is strictly positive.
        mask = (confidences > lower) & (confidences <= upper)
        count = int(mask.sum())
        result.append({
            "bin_index": index,
            "lower_bound": lower,
            "upper_bound": upper,
            "count": count,
            "mean_confidence": float(confidences[mask].mean()) if count else 0.0,
            "accuracy": float(correct[mask].mean()) if count else 0.0,
        })
    return result


def compute_calibration_metrics(
    logits: Any, labels: Any, *, ece_bins: int = 15
) -> dict[str, float | int]:
    """Compute top-1 accuracy, NLL, ECE and multiclass Brier score."""
    if isinstance(ece_bins, bool) or int(ece_bins) < 1:
        raise ValueError("ece_bins must be an integer >= 1")
    logits_np, labels_np = _as_numpy_logits_labels(logits, labels)
    probs = softmax_numpy(logits_np)
    predictions = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    correct = (predictions == labels_np).astype(np.float64)

    # Stable equivalent of mean cross-entropy/NLL.
    shifted = logits_np - logits_np.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1)) + logits_np.max(axis=1)
    nll = float(np.mean(logsumexp - logits_np[np.arange(len(labels_np)), labels_np]))

    one_hot = np.zeros_like(probs, dtype=np.float64)
    one_hot[np.arange(len(labels_np)), labels_np] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    bins = reliability_bins(logits_np, labels_np, n_bins=int(ece_bins))
    n = len(labels_np)
    ece = float(sum(
        (int(item["count"]) / n)
        * abs(float(item["accuracy"]) - float(item["mean_confidence"]))
        for item in bins
    ))

    metrics = {
        "n_samples": int(n),
        "n_classes": int(logits_np.shape[1]),
        "ece_bins": int(ece_bins),
        "top1_accuracy": float(correct.mean()),
        "nll": nll,
        "ece": ece,
        "multiclass_brier": brier,
        "mean_top_confidence": float(confidences.mean()),
    }
    if not all(math.isfinite(float(v)) for k, v in metrics.items() if k not in {"n_samples", "n_classes", "ece_bins"}):
        raise ValueError("one or more calibration metrics are not finite")
    return metrics


def plot_reliability_diagram(
    logits_before: Any,
    logits_after: Any,
    labels: Any,
    *,
    n_bins: int = 15,
    output_path: str | Path,
    title: str = "E6 Reliability Diagram",
) -> Path:
    """Save a before/after reliability diagram and return its path."""
    # Import lazily so metric-only use and unit tests do not require a GUI.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins_before = reliability_bins(logits_before, labels, n_bins=n_bins)
    bins_after = reliability_bins(logits_after, labels, n_bins=n_bins)

    def nonempty(bins: list[dict[str, float | int]]) -> tuple[list[float], list[float]]:
        active = [b for b in bins if int(b["count"]) > 0]
        return (
            [float(b["mean_confidence"]) for b in active],
            [float(b["accuracy"]) for b in active],
        )

    conf_b, acc_b = nonempty(bins_before)
    conf_a, acc_a = nonempty(bins_after)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    if conf_b:
        ax.plot(conf_b, acc_b, marker="o", label="Before calibration")
    if conf_a:
        ax.plot(conf_a, acc_a, marker="s", label="After temperature scaling")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
