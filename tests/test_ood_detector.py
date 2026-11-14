import math

import numpy as np
import pytest
import torch

from src.ood import (
    OOD_REJECT,
    PASS_OOD_CHECK,
    EnergyOODDetector,
    energy_score,
    is_ood,
    ood_decisions,
    select_energy_threshold,
)


def _logits(n=4, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 30))


def test_energy_matches_reference_formula():
    z = _logits()
    for t in (1.0, 0.5, 2.0):
        expected = [-t * math.log(sum(math.exp(v / t) for v in row)) for row in z]
        assert np.allclose(energy_score(z, t), expected, rtol=1e-12, atol=1e-12)


def test_energy_is_numerically_stable_for_large_logits():
    z = np.zeros((2, 30))
    z[0, 0] = 1e4
    z[1, :] = -1e4
    scores = energy_score(z)
    assert np.all(np.isfinite(scores))
    assert scores[0] == pytest.approx(-1e4)
    assert scores[1] == pytest.approx(1e4 - math.log(30))


def test_energy_accepts_torch_and_numpy_identically():
    z = _logits()
    assert np.allclose(energy_score(torch.tensor(z, dtype=torch.float32)), energy_score(z.astype(np.float32)))


def test_confident_logits_have_lower_energy():
    confident = np.zeros((1, 30)); confident[0, 3] = 15.0
    flat = np.zeros((1, 30))
    assert energy_score(confident)[0] < energy_score(flat)[0]


@pytest.mark.parametrize("bad", [np.full((1, 30), np.nan), np.full((1, 30), np.inf),
                                 np.zeros((1, 29)), np.zeros((30,)), np.zeros((0, 30)),
                                 np.array([["a"] * 30])])
def test_invalid_logits_are_rejected(bad):
    with pytest.raises(ValueError):
        energy_score(bad)


@pytest.mark.parametrize("t", [0, -1, float("nan"), float("inf"), True])
def test_invalid_temperature_is_rejected(t):
    with pytest.raises(ValueError):
        energy_score(_logits(), t)


def test_decision_direction_and_tie_rule():
    scores = np.array([-5.0, 0.0, 0.0, 3.0])
    assert is_ood(scores, 0.0).tolist() == [False, False, False, True]
    assert ood_decisions(scores, 0.0) == [PASS_OOD_CHECK, PASS_OOD_CHECK, PASS_OOD_CHECK, OOD_REJECT]


def test_threshold_selection_reaches_target_with_ties():
    scores = np.arange(10, dtype=float)  # 0..9
    thr = select_energy_threshold(scores, 0.9)
    assert thr == 8.0 and (scores <= thr).mean() == pytest.approx(0.9)
    tied = np.array([1, 2, 2, 2, 5], dtype=float)
    thr = select_energy_threshold(tied, 0.5)  # k = 3 -> third smallest = 2
    assert thr == 2.0 and (tied <= thr).mean() == pytest.approx(0.8)


@pytest.mark.parametrize("target", [0, 1.5, float("nan")])
def test_threshold_target_validation(target):
    with pytest.raises(ValueError):
        select_energy_threshold([1.0, 2.0], target)


def test_detector_predict_uses_locked_parameters():
    det = EnergyOODDetector(threshold=-5.0, temperature=1.0)
    confident = np.zeros((1, 30)); confident[0, 0] = 20.0
    flat = np.zeros((1, 30))
    out = det.predict(np.vstack([confident, flat]))
    assert out["decisions"] == [PASS_OOD_CHECK, OOD_REJECT]
    with pytest.raises(ValueError):
        EnergyOODDetector(threshold=float("nan"))
