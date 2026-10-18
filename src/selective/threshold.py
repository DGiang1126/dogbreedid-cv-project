"""Confidence validation and threshold selection for selective prediction."""

from fractions import Fraction
from numbers import Real

import numpy as np


def _real_scalar(value, name):
    """Return a finite Python float, rejecting booleans and non-real inputs."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real numeric scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_confidences(values):
    """Validate confidences and return an independent float64 1D copy."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidences must be a one-dimensional numeric vector") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError("confidences must be a nonempty one-dimensional vector")
    if array.dtype.kind not in "iuf":
        raise ValueError("confidences must contain real numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError("confidences must be finite")
    if np.any((array < 0) | (array > 1)):
        raise ValueError("confidences must be in [0, 1]")
    try:
        result = np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("confidences must contain real numeric values") from exc
    return result


def _validate_logits(logits):
    try:
        array = np.asarray(logits)
    except (TypeError, ValueError) as exc:
        raise ValueError("logits must be a nonempty real numeric matrix") from exc
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 2:
        raise ValueError("logits must have shape (N, C), with N > 0 and C >= 2")
    if array.dtype.kind not in "iuf":
        raise ValueError("logits must contain real numeric values")
    try:
        result = np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("logits must contain real numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("logits must be finite")
    if array.dtype.kind in "iu" and any(
        int(original) != int(converted)
        for original, converted in zip(array.flat, result.flat)
    ):
        raise ValueError("integer logits must be exactly representable as float64")
    return result


def calibrated_predictions(logits, temperature):
    """Apply stable temperature-scaled softmax and return predictions/confidence.

    Predictions use argmax on the original logits. Centering before scaling
    preserves small differences under large shared offsets. If subtraction
    overflows for opposite-sign extremes, those cells are evaluated by
    separately scaling their operands. Logits are represented internally as
    float64, so higher-precision floating inputs are normalized to that dtype.
    """
    values = _validate_logits(logits)
    temp = _real_scalar(temperature, "temperature")
    if temp <= 0.0:
        raise ValueError("temperature must be greater than zero")

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        row_max = np.max(values, axis=1, keepdims=True)
        centered = values - row_max
        shifted = centered / temp
        overflowed = np.isneginf(centered)
        if np.any(overflowed):
            # Only these entries overflowed subtraction. Their operands have
            # opposite signs, so separately scaling them avoids inf - inf.
            row_max_broadcast = np.broadcast_to(row_max, values.shape)
            shifted[overflowed] = (
                values[overflowed] / temp - row_max_broadcast[overflowed] / temp
            )
        exponentials = np.exp(shifted)
        probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)

    if not np.all(np.isfinite(probabilities)) or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("temperature-scaled probabilities could not be normalized")
    predictions = np.argmax(values, axis=1).astype(np.int64, copy=False)
    confidences = np.max(probabilities, axis=1).astype(np.float64, copy=False)
    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "confidences": confidences,
    }


def accept_mask(confidences, threshold):
    """Return the inclusive ACCEPT mask for a confidence threshold in [0, 1]."""
    values = validate_confidences(confidences)
    cutoff = _real_scalar(threshold, "threshold")
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be in [0, 1]")
    return values >= cutoff


def select_threshold(confidences, target_coverage):
    """Choose the largest threshold accepting at least the requested coverage.

    The required count is ceil(N * target_coverage), with the decimal spelling
    of the supplied real scalar used for the multiplication. Equal-confidence
    samples are all accepted at the selected threshold.
    """
    values = validate_confidences(confidences)
    target = _real_scalar(target_coverage, "target_coverage")
    if target_coverage <= 0 or target_coverage > 1:
        raise ValueError("target_coverage must be in (0, 1]")
    fraction = Fraction(str(target))
    count = (values.size * fraction.numerator + fraction.denominator - 1) // fraction.denominator
    descending = np.sort(values)[::-1]
    return float(descending[count - 1])
