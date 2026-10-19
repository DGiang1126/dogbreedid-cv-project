"""Selective metrics, tie-aware risk-coverage curves, and discrete AURC."""

from math import fsum

import numpy as np

from .threshold import validate_confidences


def _validate_boolean_vector(values, name):
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonempty one-dimensional boolean vector") from exc
    if array.ndim != 1 or array.size == 0 or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a nonempty one-dimensional boolean vector")
    return np.array(array, dtype=np.bool_, copy=True)


def _validate_pair(correct, accepted):
    correct_values = _validate_boolean_vector(correct, "correct")
    accepted_values = _validate_boolean_vector(accepted, "accepted")
    if correct_values.shape != accepted_values.shape:
        raise ValueError("correct and accepted must have the same shape")
    return correct_values, accepted_values


def _metric_row(n_total, n_accept, n_correct_accept):
    n_reject = n_total - n_accept
    if n_accept:
        accuracy = n_correct_accept / n_accept
        risk = (n_accept - n_correct_accept) / n_accept
    else:
        accuracy = None
        risk = None
    return {
        "n_total": int(n_total),
        "n_accept": int(n_accept),
        "n_reject": int(n_reject),
        "n_correct_accept": int(n_correct_accept),
        "coverage": float(n_accept / n_total),
        "selective_accuracy": None if accuracy is None else float(accuracy),
        "selective_risk": None if risk is None else float(risk),
    }


def selective_metrics(correct, accepted):
    """Compute selective metrics for matching nonempty boolean vectors."""
    correct_values, accepted_values = _validate_pair(correct, accepted)
    n_total = correct_values.size
    n_accept = int(np.count_nonzero(accepted_values))
    n_correct_accept = int(np.count_nonzero(correct_values & accepted_values))
    return _metric_row(n_total, n_accept, n_correct_accept)


def _validate_curve_inputs(confidences, correct):
    confidence_values = validate_confidences(confidences)
    correct_values = _validate_boolean_vector(correct, "correct")
    if confidence_values.shape != correct_values.shape:
        raise ValueError("confidences and correct must have the same shape")
    return confidence_values, correct_values


def _grouped_prefixes(confidences, correct):
    """Return descending tie-group boundary indices and prefix correctness."""
    order = np.argsort(-confidences, kind="stable")
    sorted_confidences = confidences[order]
    sorted_correct = correct[order]
    ends = np.flatnonzero(
        np.r_[sorted_confidences[1:] != sorted_confidences[:-1], True]
    ) + 1
    prefix_correct = np.cumsum(sorted_correct, dtype=np.int64)[ends - 1]
    return sorted_confidences[ends - 1], ends, prefix_correct


def risk_coverage_curve(confidences, correct):
    """Build an ascending-coverage curve, adding each exact-confidence tie group.

    The initial row is the mathematical zero-coverage boundary. Its risk and
    accuracy are undefined because no examples have been accepted.
    """
    confidence_values, correct_values = _validate_curve_inputs(confidences, correct)
    n_total = confidence_values.size
    thresholds, ends, prefix_correct = _grouped_prefixes(confidence_values, correct_values)

    origin = _metric_row(n_total, 0, 0)
    origin["threshold"] = None
    rows = [origin]
    for threshold, n_accept, n_correct_accept in zip(thresholds, ends, prefix_correct):
        row = _metric_row(n_total, int(n_accept), int(n_correct_accept))
        row["threshold"] = float(threshold)
        rows.append(row)
    return rows


def compute_aurc(confidences, correct):
    """Compute right-step AURC over tie-group coverage increments.

    For each distinct confidence group g, integrate its post-group risk over
    the coverage increment: sum_g risk_g * (coverage_g - coverage_previous).
    The zero-coverage origin has undefined risk and contributes no area.
    The final value is bounded to [0, 1] to absorb floating-point roundoff.
    """
    confidence_values, correct_values = _validate_curve_inputs(confidences, correct)
    n_total = confidence_values.size
    _, ends, prefix_correct = _grouped_prefixes(confidence_values, correct_values)
    group_risk = (ends - prefix_correct) / ends
    coverage_increments = np.diff(np.r_[0, ends]) / n_total
    area = fsum(float(risk * increment) for risk, increment in zip(group_risk, coverage_increments))
    return float(min(1.0, max(0.0, area)))
