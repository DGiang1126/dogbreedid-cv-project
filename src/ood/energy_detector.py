"""Energy-based OOD score and decision rule for the closed-set E4 classifier.

E(x) = -T * logsumexp(z / T) over the 30 raw E4 logits z (Liu et al., 2020).
Lower energy tends to indicate in-distribution inputs; higher energy tends to
indicate OOD. This is a statistical signal, not a guarantee.

The Energy temperature ``T`` is a separate research parameter fixed before any
evaluation (default 1.0). It is NOT the E6 calibration temperature.
"""
from __future__ import annotations

import math
from fractions import Fraction
from numbers import Real

import numpy as np
import torch

NUM_CLASSES = 30
DEFAULT_ENERGY_TEMPERATURE = 1.0
OOD_REJECT = "OOD_REJECT"
PASS_OOD_CHECK = "PASS_OOD_CHECK"


def _finite_scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_temperature(temperature) -> float:
    value = _finite_scalar(temperature, "energy temperature")
    if value <= 0:
        raise ValueError("energy temperature must be greater than zero")
    return value


def validate_logits(logits, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Return a float64 CPU tensor of shape [N, num_classes] with finite values."""
    if isinstance(logits, torch.Tensor):
        tensor = logits.detach().to("cpu")
    else:
        try:
            array = np.asarray(logits)
        except (TypeError, ValueError) as exc:
            raise ValueError("logits must be a numeric matrix") from exc
        if array.dtype.kind not in "iuf":
            raise ValueError("logits must contain real numeric values")
        tensor = torch.from_numpy(np.array(array, dtype=np.float64, copy=True))
    if tensor.dtype.is_complex or tensor.dtype == torch.bool:
        raise ValueError("logits must contain real numeric values")
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] != num_classes:
        raise ValueError(f"logits must have shape [N, {num_classes}] with N > 0")
    tensor = tensor.to(torch.float64)
    if not torch.isfinite(tensor).all():
        raise ValueError("logits must not contain NaN or Inf")
    return tensor


def energy_score(logits, temperature=DEFAULT_ENERGY_TEMPERATURE) -> np.ndarray:
    """Return E(x) = -T * logsumexp(z / T) per row as a float64 numpy vector."""
    values = validate_logits(logits)
    temp = _validate_temperature(temperature)
    scores = -temp * torch.logsumexp(values / temp, dim=1)
    if not torch.isfinite(scores).all():
        raise ValueError("energy score overflowed; check logits and temperature")
    return scores.numpy().astype(np.float64, copy=True)


def _validate_scores(scores) -> np.ndarray:
    array = np.asarray(scores)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iuf":
        raise ValueError("energy scores must be a nonempty 1D numeric vector")
    array = array.astype(np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError("energy scores must be finite")
    return array


def is_ood(scores, threshold) -> np.ndarray:
    """OOD when energy_score > threshold (strict). Equal scores pass the check."""
    values = _validate_scores(scores)
    cutoff = _finite_scalar(threshold, "ood threshold")
    return values > cutoff


def ood_decisions(scores, threshold) -> list[str]:
    return [OOD_REJECT if flag else PASS_OOD_CHECK for flag in is_ood(scores, threshold)]


def select_energy_threshold(id_scores, target_id_acceptance=0.9) -> float:
    """Pick the smallest threshold so that at least the target fraction of ID scores pass.

    Pass rule is ``energy <= threshold``. With N ID scores the required count is
    k = ceil(N * target); the threshold is the k-th smallest score, so every score
    tied with it also passes (consistent tie handling, acceptance may exceed target).
    """
    values = _validate_scores(id_scores)
    target = _finite_scalar(target_id_acceptance, "target_id_acceptance")
    if not 0 < target <= 1:
        raise ValueError("target_id_acceptance must be in (0, 1]")
    fraction = Fraction(str(target_id_acceptance))
    count = (values.size * fraction.numerator + fraction.denominator - 1) // fraction.denominator
    return float(np.sort(values)[count - 1])


class EnergyOODDetector:
    """A locked Energy detector: fixed temperature and threshold."""

    def __init__(self, threshold, temperature=DEFAULT_ENERGY_TEMPERATURE):
        self.threshold = _finite_scalar(threshold, "ood threshold")
        self.temperature = _validate_temperature(temperature)

    def score(self, logits) -> np.ndarray:
        return energy_score(logits, self.temperature)

    def predict(self, logits) -> dict:
        scores = self.score(logits)
        flags = is_ood(scores, self.threshold)
        return {"energy": scores, "is_ood": flags,
                "decisions": [OOD_REJECT if f else PASS_OOD_CHECK for f in flags]}
