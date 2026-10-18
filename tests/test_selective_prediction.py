"""Synthetic unit tests for E7 selective prediction algorithms.

These tests exercise only pure array algorithms.  They deliberately do not read
E6 artifacts or any evaluation split.
"""

from fractions import Fraction
import math

import numpy as np
import pytest

from src.selective.threshold import (
    accept_mask,
    calibrated_predictions,
    select_threshold,
    validate_confidences,
)
from src.selective.risk_coverage import (
    compute_aurc,
    risk_coverage_curve,
    selective_metrics,
)


_BAD_VECTOR_CASES = [
    pytest.param(np.array([], dtype=float), id="empty"),
    pytest.param(np.array([[0.2, 0.8]]), id="two-dimensional"),
    pytest.param(np.array([True, False]), id="boolean-dtype"),
    pytest.param(np.array([0.2, "0.8"], dtype=object), id="object-dtype"),
    pytest.param(np.array(["0.2", "0.8"]), id="string-dtype"),
    pytest.param(np.array([0.2 + 0j]), id="complex-dtype"),
    pytest.param(np.array([np.nan]), id="nan"),
    pytest.param(np.array([np.inf]), id="positive-infinity"),
    pytest.param(np.array([-np.inf]), id="negative-infinity"),
    pytest.param(np.array([-0.01]), id="below-zero"),
    pytest.param(np.array([1.01]), id="above-one"),
]


def _assert_error(call):
    with pytest.raises(ValueError):
        call()


def test_validate_confidences_accepts_real_numeric_vectors_and_returns_copy():
    original = np.array([0, 0.25, 1], dtype=np.float32)
    result = validate_confidences(original)

    assert result.dtype == np.float64
    assert result.tolist() == [0.0, 0.25, 1.0]
    assert result is not original
    assert not np.shares_memory(result, original)
    result[0] = 0.5
    assert original[0] == 0
    assert validate_confidences([0, 1]).tolist() == [0.0, 1.0]


def test_validate_confidences_rejects_extended_precision_value_above_one():
    extended = np.finfo(np.longdouble)
    if extended.eps == np.finfo(np.float64).eps:
        pytest.skip("platform longdouble has no precision beyond float64")
    value = np.longdouble(1) + extended.eps
    assert value > 1
    with pytest.raises(ValueError):
        validate_confidences(np.array([value], dtype=np.longdouble))


@pytest.mark.parametrize("values", _BAD_VECTOR_CASES)
def test_validate_confidences_rejects_invalid_vectors(values):
    _assert_error(lambda: validate_confidences(values))


@pytest.mark.parametrize("logits", [
    np.empty((0, 2)),
    np.ones((2, 1)),
    np.ones((2, 2, 1)),
    np.array([[0.0, np.nan]]),
    np.array([[0.0, np.inf]]),
    np.array([["1", "2"]]),
])
def test_calibrated_predictions_rejects_invalid_logits(logits):
    _assert_error(lambda: calibrated_predictions(logits, 1.0))


@pytest.mark.parametrize("temperature", [0, -1, np.nan, np.inf, -np.inf, True, "1"])
def test_calibrated_predictions_rejects_invalid_temperature(temperature):
    _assert_error(lambda: calibrated_predictions(np.array([[0.0, 1.0]]), temperature))


def test_calibrated_predictions_softmax_and_confidence_at_known_temperature():
    logits = np.array([[0.0, math.log(3.0)], [3.0, 0.0]])
    before = logits.copy()
    at_one = calibrated_predictions(logits, 1.0)
    at_two = calibrated_predictions(logits, 2.0)

    assert set(at_one) == {"probabilities", "predictions", "confidences"}
    np.testing.assert_allclose(at_one["probabilities"][0], [0.25, 0.75], rtol=0, atol=1e-15)
    root3 = math.sqrt(3.0)
    np.testing.assert_allclose(
        at_two["probabilities"][0], [1 / (1 + root3), root3 / (1 + root3)],
        rtol=0, atol=1e-15,
    )
    assert at_one["predictions"].tolist() == [1, 0]
    np.testing.assert_allclose(at_one["confidences"], at_one["probabilities"].max(axis=1))
    np.testing.assert_allclose(at_two["confidences"], at_two["probabilities"].max(axis=1))
    np.testing.assert_allclose(at_one["probabilities"].sum(axis=1), 1.0)
    np.testing.assert_array_equal(logits, before)


@pytest.mark.parametrize("offset", [1e16, -1e16])
def test_calibrated_predictions_is_invariant_to_large_representable_logit_offset(offset):
    logits = np.array([[offset, offset + 2.0]])
    assert logits[0, 1] - logits[0, 0] == 2.0
    before = logits.copy()
    result = calibrated_predictions(logits, 3.0)
    e = math.exp(-2.0 / 3.0)
    expected = np.array([e / (1.0 + e), 1.0 / (1.0 + e)])
    np.testing.assert_allclose(result["probabilities"][0], expected, rtol=0, atol=1e-15)
    assert result["predictions"].tolist() == [1]
    np.testing.assert_array_equal(logits, before)


def test_calibrated_predictions_is_stable_for_extreme_logits_and_temperature():
    logits = np.array([[-1e308, 1e308], [7e307, 7e307], [1e308, 1e308 - 1e292]])
    before = logits.copy()
    result = calibrated_predictions(logits, 1e308)

    assert np.isfinite(result["probabilities"]).all()
    np.testing.assert_allclose(result["probabilities"].sum(axis=1), 1.0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(result["probabilities"][0], [math.exp(-2) / (1 + math.exp(-2)), 1 / (1 + math.exp(-2))])
    assert result["predictions"].tolist() == [1, 0, 0]
    np.testing.assert_array_equal(logits, before)


def test_calibrated_predictions_handles_normal_tiny_logits_and_temperature():
    logits = np.array([[0.0, np.nextafter(0.0, 1.0)]])
    result = calibrated_predictions(logits, np.nextafter(0.0, 1.0))
    assert result["predictions"].tolist() == [1]
    assert np.isfinite(result["probabilities"]).all()
    np.testing.assert_allclose(result["probabilities"].sum(axis=1), 1.0)



def test_calibrated_predictions_huge_temperature_rounding_tie_keeps_original_argmax():
    smallest = np.nextafter(0.0, 1.0)
    logits = np.array([[0.0, smallest]])
    result = calibrated_predictions(logits, np.finfo(np.float64).max)
    np.testing.assert_array_equal(result["probabilities"], [[0.5, 0.5]])
    assert result["predictions"].tolist() == [1]


def test_calibrated_predictions_rejects_integer_logits_when_float64_would_lose_rank():
    logits = [[2**63 - 2, 2**63 - 1]]
    with pytest.raises(ValueError):
        calibrated_predictions(logits, 1.0)


def test_calibrated_predictions_accepts_small_integer_logits():
    result = calibrated_predictions(np.array([[1, 2]], dtype=np.int64), 1.0)
    assert result["predictions"].tolist() == [1]
    np.testing.assert_allclose(result["probabilities"][0], [1 / (1 + math.e), math.e / (1 + math.e)])


def test_calibrated_predictions_rejects_huge_integer_temperature_as_value_error():
    with pytest.raises(ValueError):
        calibrated_predictions([[0.0, 1.0]], 10**1000)


def test_accept_mask_includes_exact_boundary_and_does_not_mutate_inputs():
    confidences = np.array([0.2, 0.5, 0.8])
    original = confidences.copy()
    accepted = accept_mask(confidences, 0.5)
    assert accepted.dtype == np.bool_
    assert accepted.tolist() == [False, True, True]
    np.testing.assert_array_equal(confidences, original)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, np.nan, np.inf, -np.inf, True, "0.5"])
def test_accept_mask_rejects_invalid_threshold(threshold):
    _assert_error(lambda: accept_mask([0.5], threshold))


def test_accept_mask_rejects_huge_integer_threshold_as_value_error():
    with pytest.raises(ValueError):
        accept_mask([0.5], 10**1000)


def test_select_threshold_selects_270_of_300_synthetic_examples():
    # A deterministic synthetic calibration sample with exactly the requested
    # 90% coverage; no E6 or project prediction artifact is loaded.
    confidences = np.array([0.8] * 270 + [0.2] * 30)
    threshold = select_threshold(confidences, 0.9)
    assert threshold == 0.8
    assert int(accept_mask(confidences, threshold).sum()) == 270


def test_select_threshold_includes_whole_tied_boundary_group():
    confidences = np.array([0.9, 0.7, 0.7, 0.1])
    threshold = select_threshold(confidences, 0.5)
    assert threshold == 0.7
    assert accept_mask(confidences, threshold).tolist() == [True, True, True, False]


def test_select_threshold_rejects_huge_integer_target_as_value_error():
    with pytest.raises(ValueError):
        select_threshold([0.2, 0.8], 10**1000)


def test_select_threshold_handles_full_coverage_and_decimal_rounding():
    assert select_threshold([0.2, 0.9, 0.5], 1.0) == 0.2
    # Binary floating multiplication can represent 50 * .14 just above 7.
    # The intended decimal target requires seven accepted samples, not eight.
    confidence = np.linspace(0.01, 0.50, 50)
    threshold = select_threshold(confidence, 0.14)
    assert threshold == confidence[-7]
    assert int(accept_mask(confidence, threshold).sum()) == 7


@pytest.mark.parametrize("target", [0, -0.1, 1.1, np.nan, np.inf, True, "0.5"])
def test_select_threshold_rejects_invalid_target(target):
    _assert_error(lambda: select_threshold([0.2, 0.8], target))


def _decimal_ceil_product(n, target):
    """Independent oracle that interprets target using its written decimal."""
    ratio = Fraction(str(target))
    return (n * ratio.numerator + ratio.denominator - 1) // ratio.denominator


def _threshold_oracle(values, target):
    values = list(map(float, values))
    k = _decimal_ceil_product(len(values), target)
    return sorted(values, reverse=True)[k - 1]


@pytest.mark.parametrize("values,target", [
    ([0.1, 0.2, 0.2, 0.9, 0.9], 0.4),
    ([0.25, 0.25, 0.5, 0.75], 0.5),
    ([0.0, 0.1, 0.4, 1.0], 0.75),
    ([0.4] * 7, 0.42),
])
def test_select_threshold_matches_independent_feasible_threshold_oracle(values, target):
    got = select_threshold(values, target)
    expected = _threshold_oracle(values, target)
    assert got == expected
    # Exact predicate semantics verify coverage and maximality independently.
    coverage_count = sum(value >= got for value in values)
    assert coverage_count >= _decimal_ceil_product(len(values), target)
    larger_feasible = [
        candidate for candidate in set(values)
        if candidate > got and sum(value >= candidate for value in values) >= _decimal_ceil_product(len(values), target)
    ]
    assert not larger_feasible


def test_select_threshold_property_on_small_deterministic_synthetic_grids():
    rng = np.random.default_rng(7107)
    targets = [0.14, 0.25, 0.5, 0.73, 1.0]
    for n in range(1, 18):
        # Small integer grid guarantees repeated/tied confidences.
        values = rng.integers(0, 6, size=n).astype(float) / 5
        for target in targets:
            got = select_threshold(values, target)
            assert got == _threshold_oracle(values, target)
            assert np.count_nonzero(values >= got) >= _decimal_ceil_product(n, target)


def test_selective_metrics_reports_counts_and_undefined_empty_accept_set():
    correct = np.array([True, False, True, False])
    accepted = np.array([True, True, False, False])
    result = selective_metrics(correct, accepted)
    assert result == {
        "n_total": 4,
        "n_accept": 2,
        "n_reject": 2,
        "n_correct_accept": 1,
        "coverage": 0.5,
        "selective_accuracy": 0.5,
        "selective_risk": 0.5,
    }
    empty = selective_metrics(correct, np.zeros(4, dtype=bool))
    assert empty["n_accept"] == 0
    assert empty["selective_accuracy"] is None
    assert empty["selective_risk"] is None
    assert empty["coverage"] == 0.0


@pytest.mark.parametrize("correct,accepted", [
    ([], []),
    ([True, False], [True]),
    ([[True, False]], [[True, False]]),
    ([1, 0], [True, False]),
    ([True, False], [1, 0]),
])
def test_selective_metrics_rejects_invalid_vectors(correct, accepted):
    _assert_error(lambda: selective_metrics(correct, accepted))


def test_risk_coverage_curve_starts_at_origin_and_groups_exact_ties():
    curve = risk_coverage_curve(
        np.array([0.9, 0.9, 0.5, 0.1]),
        np.array([True, False, True, False]),
    )
    assert len(curve) == 4
    origin, first, second, final = curve
    assert origin["threshold"] is None
    assert origin["n_accept"] == 0
    assert origin["coverage"] == 0.0
    assert origin["selective_accuracy"] is None
    assert origin["selective_risk"] is None
    assert first["threshold"] == 0.9
    assert first["n_accept"] == 2
    assert first["coverage"] == 0.5
    assert first["selective_risk"] == 0.5
    assert second["threshold"] == 0.5
    assert second["n_accept"] == 3
    assert second["coverage"] == 0.75
    assert second["selective_risk"] == pytest.approx(1 / 3)
    assert final["threshold"] == 0.1
    assert final["n_accept"] == 4
    assert final["coverage"] == 1.0
    assert final["selective_risk"] == 0.5


def test_risk_coverage_keeps_adjacent_representable_confidences_as_distinct_groups():
    lower = 0.5
    upper = np.nextafter(lower, 1.0)
    confidences = np.array([lower, upper])
    correct = np.array([True, False])
    confidence_before = confidences.copy()
    correct_before = correct.copy()

    curve = risk_coverage_curve(confidences, correct)
    assert len(curve) == 3
    assert curve[1]["threshold"] == upper
    assert curve[1]["n_accept"] == 1
    assert curve[2]["threshold"] == lower
    assert curve[2]["n_accept"] == 2
    assert compute_aurc(confidences, correct) == pytest.approx(0.75)
    np.testing.assert_array_equal(confidences, confidence_before)
    np.testing.assert_array_equal(correct, correct_before)


def test_risk_coverage_curve_and_aurc_for_no_ties():
    confidences = np.array([0.9, 0.8, 0.7])
    correct = np.array([True, False, True])
    curve = risk_coverage_curve(confidences, correct)
    assert [row["selective_risk"] for row in curve[1:]] == [0.0, 0.5, 1 / 3]
    assert compute_aurc(confidences, correct) == pytest.approx(5 / 18)
    # With one sample added per group, the right-step integral equals the
    # arithmetic mean of all prefix risks.
    assert compute_aurc(confidences, correct) == pytest.approx(np.mean([0.0, 0.5, 1 / 3]))


def test_aurc_for_tied_example_uses_post_group_right_steps():
    confidences = np.array([0.9, 0.9, 0.5, 0.1])
    correct = np.array([True, False, True, False])
    assert compute_aurc(confidences, correct) == pytest.approx(11 / 24)


def test_risk_coverage_is_permutation_invariant_inside_exact_ties():
    confidence = np.array([0.9, 0.9, 0.5, 0.1])
    correct = np.array([True, False, True, False])
    expected = risk_coverage_curve(confidence, correct)
    permutation = np.array([1, 0, 2, 3])
    assert risk_coverage_curve(confidence[permutation], correct[permutation]) == expected
    assert compute_aurc(confidence[permutation], correct[permutation]) == compute_aurc(confidence, correct)


@pytest.mark.parametrize("correct,expected", [
    ([True, True, True], 0.0),
    ([False, False, False], 1.0),
    ([True, False, True, False], 0.5),
])
def test_all_tied_confidences_reduce_aurc_to_global_error(correct, expected):
    curve = risk_coverage_curve(np.full(len(correct), 0.4), np.array(correct))
    assert len(curve) == 2  # origin plus the sole complete tie group
    assert curve[-1]["coverage"] == 1.0
    assert compute_aurc(np.full(len(correct), 0.4), np.array(correct)) == expected


@pytest.mark.parametrize("confidences,correct", [
    ([], []),
    ([0.5, 0.7], [True]),
    ([[0.5, 0.7]], [True, False]),
    ([0.5, np.nan], [True, False]),
    ([0.5, 1.1], [True, False]),
    ([0.5, 0.7], [1, 0]),
])
def test_risk_coverage_and_aurc_reject_invalid_inputs(confidences, correct):
    _assert_error(lambda: risk_coverage_curve(confidences, correct))
    _assert_error(lambda: compute_aurc(confidences, correct))


@pytest.mark.parametrize("is_correct,expected", [(True, 0.0), (False, 1.0)])
def test_singleton_selective_metrics_and_curve_boundaries(is_correct, expected):
    correct = np.array([is_correct], dtype=bool)
    accepted = np.array([True], dtype=bool)
    metrics = selective_metrics(correct, accepted)
    assert metrics["coverage"] == 1.0
    assert metrics["selective_accuracy"] == float(is_correct)
    assert metrics["selective_risk"] == expected

    curve = risk_coverage_curve([0.6], correct)
    assert curve[0]["threshold"] is None
    assert curve[0]["n_accept"] == 0
    assert curve[0]["selective_risk"] is None
    assert curve[1]["threshold"] == 0.6
    assert curve[1]["coverage"] == 1.0
    assert curve[1]["selective_risk"] == expected
    assert compute_aurc([0.6], correct) == expected


def test_singleton_rejected_example_has_undefined_selective_metrics_and_origin_curve():
    metrics = selective_metrics([False], [False])
    assert metrics["n_accept"] == 0
    assert metrics["coverage"] == 0.0
    assert metrics["selective_accuracy"] is None
    assert metrics["selective_risk"] is None
    # The curve's origin is explicitly the zero-coverage state; the actual
    # confidence group then covers the sole example.
    curve = risk_coverage_curve([0.6], [False])
    assert curve[0]["coverage"] == 0.0
    assert curve[0]["selective_risk"] is None
    assert curve[1]["coverage"] == 1.0
    assert curve[1]["selective_risk"] == 1.0


def test_aurc_exact_bounds_for_300_distinct_all_correct_and_all_wrong():
    confidences = np.arange(1, 301, dtype=float) / 301.0
    for correct, expected in [
        (np.ones(300, dtype=bool), 0.0),
        (np.zeros(300, dtype=bool), 1.0),
    ]:
        value = compute_aurc(confidences, correct)
        assert 0.0 <= value <= 1.0
        assert value == expected


@pytest.mark.parametrize("operation", ["threshold", "target"])
def test_scalar_validators_reject_extended_precision_value_above_one(operation):
    extended = np.finfo(np.longdouble)
    if extended.eps == np.finfo(np.float64).eps:
        pytest.skip("platform longdouble has no precision beyond float64")
    value = np.longdouble(1) + extended.eps
    assert value > 1
    with pytest.raises(ValueError):
        if operation == "threshold":
            accept_mask([0.5], value)
        else:
            select_threshold([0.2, 0.8], value)
