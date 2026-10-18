"""Synthetic unit and integration tests for E7 selective prediction.

Algorithm tests use pure arrays; stage 3 tests build an isolated temporary repo
from approved text configuration and generated synthetic artifacts.  No E6
artifacts or real evaluation split are read.
"""

import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml

from scripts.run_selective_prediction import (
    load_e6_artifacts,
    main as selective_main,
    run_selective_prediction,
)
from src.selective.config import load_e7_config

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


# Stage 3 configuration/runner tests use a separate temporary repository tree.
# Only approved text configuration is copied; all E6 arrays and CSV rows below
# are generated synthetic fixtures.


_REPO_ROOT = Path(__file__).resolve().parents[1]
_E7_CONFIG = _REPO_ROOT / "configs/experiments/e7_selective_prediction.yaml"
_MODEL_CONFIG = _REPO_ROOT / "configs/model_selection.yaml"
_DATA_CONFIG = _REPO_ROOT / "configs/data.yaml"
_CLASS_MAPPING = _REPO_ROOT / "data/metadata/class_mapping.json"


def _write_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _synthetic_repo(tmp_path):
    """Create a fresh fake repo with approved text configs and synthetic E6 inputs."""
    root = tmp_path / "synthetic-repo"
    for source, relative in [
        (_E7_CONFIG, "configs/experiments/e7_selective_prediction.yaml"),
        (_MODEL_CONFIG, "configs/model_selection.yaml"),
        (_DATA_CONFIG, "configs/data.yaml"),
        (_REPO_ROOT / "configs/experiments/e4_cutmix.yaml", "configs/experiments/e4_cutmix.yaml"),
        (_CLASS_MAPPING, "data/metadata/class_mapping.json"),
    ]:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    mapping = json.loads((root / "data/metadata/class_mapping.json").read_text(encoding="utf-8"))
    breeds = [mapping[str(index)] for index in range(30)]
    paths = [f"synthetic/{index % 30:02d}/犬_{index:03d}.jpg" for index in range(300)]
    split = root / "data/splits/calibration.csv"
    split.parent.mkdir(parents=True, exist_ok=True)
    with split.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "class_id", "breed_name", "source_split"])
        writer.writeheader()
        for index, relative_path in enumerate(paths):
            class_id = index % 30
            writer.writerow({
                "relative_path": relative_path,
                "class_id": class_id,
                "breed_name": breeds[class_id],
                "source_split": "official_train",
            })

    # Increasing margins give a deterministic ranking with >90% feasible coverage.
    logits = np.zeros((300, 30), dtype=np.float32)
    labels = (np.arange(300) % 30).astype(np.int64)
    for index, label in enumerate(labels):
        logits[index, label] = np.float32(4.0 + (299 - index) / 300.0)
    arrays = root / "outputs/predictions/E6"
    arrays.mkdir(parents=True, exist_ok=True)
    npz = arrays / "calibration_logits.npz"
    np.savez_compressed(npz, logits=logits, labels=labels, image_ids=np.asarray(paths, dtype=np.str_))

    checkpoint = "/synthetic/E4/seed_42/best.pt"
    sha = "a" * 40
    temperature = {
        "experiment": "E6",
        "method": "scalar_temperature_scaling",
        "temperature": 2.0,
        "model_experiment": "E4",
        "checkpoint_seed": 42,
        "checkpoint_path": checkpoint,
        "split": "calibration",
        "n_samples": 300,
        "git_sha": sha,
        "synthetic": True,
    }
    metadata = {
        "experiment": "E6",
        "source_model": "E4 MobileNetV2 Frozen + CutMix",
        "checkpoint_seed": 42,
        "checkpoint_identifier": checkpoint,
        "checkpoint_filename": "best.pt",
        "git_sha": sha,
        "split": "calibration",
        "n_samples": 300,
        "logits_shape": [300, 30],
        "labels_shape": [300],
        "image_ids_shape": [300],
        "preprocessing": {
            "color_mode": "RGB",
            "resize": 256,
            "center_crop": 224,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
        "official_test_used": False,
        "weights_updated": False,
        "synthetic": True,
    }
    temperature_path = root / "outputs/metrics/E6/temperature.json"
    temperature_path.parent.mkdir(parents=True, exist_ok=True)
    temperature_path.write_text(json.dumps(temperature, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    metadata_path = arrays / "calibration_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return {
        "root": root,
        "config_path": root / "configs/experiments/e7_selective_prediction.yaml",
        "split_path": split,
        "npz_path": npz,
        "temperature_path": temperature_path,
        "metadata_path": metadata_path,
        "logits": logits,
        "labels": labels,
        "image_ids": np.asarray(paths, dtype=np.str_),
        "temperature": temperature,
        "metadata": metadata,
    }


def _mutated_config_file(tmp_path, mutate):
    raw = yaml.safe_load(_E7_CONFIG.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "bad-e7.yaml"
    _write_yaml(path, raw)
    return path


def test_load_e7_config_accepts_approved_text_config_roundtrip(tmp_path):
    fixture = _synthetic_repo(tmp_path)
    loaded = load_e7_config(fixture["config_path"])
    assert loaded["experiment"]["id"] == "E7"
    assert loaded["data"] == {"split": "calibration", "split_file": "data/splits/calibration.csv", "expected_samples": 300, "num_classes": 30}
    assert loaded["threshold_selection"]["target_coverage"] == 0.9
    assert loaded["outputs"]["metrics"] == "outputs/metrics/E7/selective_metrics.json"


@pytest.mark.parametrize("section", [
    "experiment", "model", "data", "calibration", "threshold_selection",
    "evaluation", "rejection", "outputs", "restrictions",
])
def test_load_e7_config_rejects_missing_required_sections(tmp_path, section):
    path = _mutated_config_file(tmp_path, lambda raw: raw.pop(section))
    with pytest.raises(ValueError):
        load_e7_config(path)


@pytest.mark.parametrize("section", [
    "experiment", "model", "data", "calibration", "threshold_selection",
    "evaluation", "rejection", "outputs", "restrictions",
])
def test_load_e7_config_rejects_nonmapping_sections(tmp_path, section):
    path = _mutated_config_file(tmp_path, lambda raw: raw.__setitem__(section, []))
    with pytest.raises(ValueError):
        load_e7_config(path)


@pytest.mark.parametrize("section,key", [
    ("experiment", "id"), ("model", "selection_config"), ("data", "expected_samples"),
    ("calibration", "required"), ("threshold_selection", "target_coverage"),
    ("evaluation", "coverage_points"), ("rejection", "enabled"),
    ("outputs", "metrics"), ("restrictions", "use_official_test"),
])
def test_load_e7_config_rejects_missing_required_fields(tmp_path, section, key):
    path = _mutated_config_file(tmp_path, lambda raw: raw[section].pop(key))
    with pytest.raises(ValueError):
        load_e7_config(path)


@pytest.mark.parametrize("section,key,value", [
    ("experiment", "id", "E6"),
    ("data", "split", "validation"),
    ("data", "split_file", "data/splits/official_test.csv"),
    ("model", "selection_config", "other.yaml"),
    ("calibration", "temperature_file", "elsewhere.json"),
    ("threshold_selection", "selection_rule", "lowest_feasible_threshold"),
    ("threshold_selection", "accept_operator", ">"),
    ("threshold_selection", "tie_policy", "partial_ties"),
    ("evaluation", "coverage_points", [1.0, 0.95, 0.8]),
    ("rejection", "message", "unknown breed"),
    ("outputs", "metrics", "elsewhere.json"),
    ("restrictions", "use_official_test", True),
])
def test_load_e7_config_rejects_changed_frozen_settings(tmp_path, section, key, value):
    path = _mutated_config_file(tmp_path, lambda raw: raw[section].__setitem__(key, value))
    with pytest.raises(ValueError):
        load_e7_config(path)


@pytest.mark.parametrize("section,key,value", [
    ("calibration", "required", "true"),
    ("rejection", "enabled", "true"),
    ("restrictions", "requires_e6_completed", "true"),
    ("data", "expected_samples", True),
    ("data", "num_classes", True),
    ("threshold_selection", "target_coverage", True),
    ("evaluation", "generate_risk_coverage_curve", "true"),
])
def test_load_e7_config_rejects_string_booleans_and_bool_numeric_values(tmp_path, section, key, value):
    path = _mutated_config_file(tmp_path, lambda raw: raw[section].__setitem__(key, value))
    with pytest.raises(ValueError):
        load_e7_config(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0.89, 0.91])
def test_load_e7_config_rejects_nonfinite_or_changed_target_coverage(tmp_path, value):
    path = _mutated_config_file(tmp_path, lambda raw: raw["threshold_selection"].__setitem__("target_coverage", value))
    with pytest.raises(ValueError):
        load_e7_config(path)


def test_load_e7_config_rejects_nonmapping_root_and_missing_file(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- not-a-map\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_e7_config(path)
    with pytest.raises(ValueError):
        load_e7_config(tmp_path / "missing.yaml")


@pytest.mark.parametrize("section,key", [
    ("threshold_selection", "target_coverage"),
    ("evaluation", "coverage_points"),
])
def test_load_e7_config_rejects_huge_numeric_values_without_overflow(tmp_path, section, key):
    huge = 10**1000
    value = huge if key == "target_coverage" else [1.0, huge, 0.8]
    path = _mutated_config_file(tmp_path, lambda raw: raw[section].__setitem__(key, value))
    with pytest.raises(ValueError):
        load_e7_config(path)



def _loaded_synthetic_artifacts(fixture):
    config = load_e7_config(fixture["config_path"])
    return load_e6_artifacts(config, fixture["root"], synthetic=True)


def _save_npz(fixture, *, logits=None, labels=None, image_ids=None, keys=None):
    if logits is None:
        logits = fixture["logits"]
    if labels is None:
        labels = fixture["labels"]
    if image_ids is None:
        image_ids = fixture["image_ids"]
    values = {"logits": logits, "labels": labels, "image_ids": image_ids}
    if keys is not None:
        values = {key: values[key] for key in keys}
    np.savez_compressed(fixture["npz_path"], **values)


def _edit_json(path, mutate):
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def test_load_e6_artifacts_accepts_aligned_synthetic_fixture(tmp_path):
    fixture = _synthetic_repo(tmp_path)
    result = _loaded_synthetic_artifacts(fixture)
    assert result["logits"].shape == (300, 30)
    assert result["logits"].dtype.kind == "f"
    assert result["labels"].shape == (300,)
    assert result["labels"].dtype == np.int64
    assert result["image_ids"].shape == (300,)
    assert result["image_ids"].dtype.kind == "U"
    assert result["temperature"] == 2.0
    assert result["checkpoint_identifier"] == fixture["temperature"]["checkpoint_path"]
    assert set(result["input_sha256"]) >= {
        "outputs/predictions/E6/calibration_logits.npz",
        "outputs/predictions/E6/calibration_metadata.json",
        "outputs/metrics/E6/temperature.json",
        "data/splits/calibration.csv",
    }
    assert all(len(digest) == 64 for digest in result["input_sha256"].values())
    np.testing.assert_array_equal(result["logits"], fixture["logits"])
    np.testing.assert_array_equal(result["labels"], fixture["labels"])
    np.testing.assert_array_equal(result["image_ids"], fixture["image_ids"])


@pytest.mark.parametrize("field,value", [
    ("split", "validation"),
    ("checkpoint_seed", 41),
    ("n_samples", 299),
    ("git_sha", "not-a-sha"),
    ("temperature", 0.0),
    ("temperature", float("nan")),
    ("temperature", True),
    ("checkpoint_path", "/synthetic/E4/seed_41/best.pt"),
    ("synthetic", False),
])
def test_load_e6_artifacts_rejects_invalid_temperature_provenance(tmp_path, field, value):
    fixture = _synthetic_repo(tmp_path)
    _edit_json(fixture["temperature_path"], lambda obj: obj.__setitem__(field, value))
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)


@pytest.mark.parametrize("field,value", [
    ("split", "validation"),
    ("checkpoint_seed", 41),
    ("n_samples", 299),
    ("git_sha", "b" * 40),
    ("checkpoint_identifier", "/synthetic/E4/seed_41/best.pt"),
    ("official_test_used", True),
    ("weights_updated", True),
    ("synthetic", False),
    ("logits_shape", [299, 30]),
    ("labels_shape", [300, 1]),
    ("image_ids_shape", [299]),
])
def test_load_e6_artifacts_rejects_invalid_metadata_provenance(tmp_path, field, value):
    fixture = _synthetic_repo(tmp_path)
    _edit_json(fixture["metadata_path"], lambda obj: obj.__setitem__(field, value))
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)


@pytest.mark.parametrize("field,value", [
    ("color_mode", "BGR"),
    ("resize", 255),
    ("center_crop", 223),
    ("normalization_mean", [0.0, 0.0, 0.0]),
    ("normalization_std", [1.0, 1.0, 1.0]),
])
def test_load_e6_artifacts_rejects_preprocessing_mismatch(tmp_path, field, value):
    fixture = _synthetic_repo(tmp_path)
    _edit_json(fixture["metadata_path"], lambda obj: obj["preprocessing"].__setitem__(field, value))
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)


@pytest.mark.parametrize("case", ["missing-key", "shape", "integer-logits", "nan-logits", "infinite-logits",
                                  "float-labels", "label-out-of-range", "id-order", "duplicate-ids", "object-ids"])
def test_load_e6_artifacts_rejects_malformed_synthetic_npz(tmp_path, case):
    fixture = _synthetic_repo(tmp_path)
    logits, labels, ids = fixture["logits"].copy(), fixture["labels"].copy(), fixture["image_ids"].copy()
    keys = None
    if case == "missing-key":
        keys = ["logits", "labels"]
    elif case == "shape":
        logits = logits[:-1]
    elif case == "integer-logits":
        logits = logits.astype(np.int32)
    elif case == "nan-logits":
        logits[0, 0] = np.nan
    elif case == "infinite-logits":
        logits[0, 0] = np.inf
    elif case == "float-labels":
        labels = labels.astype(np.float64)
    elif case == "label-out-of-range":
        labels[0] = 30
    elif case == "id-order":
        ids[[0, 1]] = ids[[1, 0]]
    elif case == "duplicate-ids":
        ids[1] = ids[0]
    elif case == "object-ids":
        ids = np.asarray(ids, dtype=object)
    _save_npz(fixture, logits=logits, labels=labels, image_ids=ids, keys=keys)
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)


def test_load_e6_artifacts_rejects_label_order_mismatch(tmp_path):
    fixture = _synthetic_repo(tmp_path)
    labels = fixture["labels"].copy()
    labels[[0, 1]] = labels[[1, 0]]
    _save_npz(fixture, labels=labels)
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)


@pytest.mark.parametrize("case", ["row-count", "duplicate-id", "class-count", "mapping-mismatch", "official-path"])
def test_load_e6_artifacts_rejects_invalid_synthetic_calibration_csv(tmp_path, case):
    fixture = _synthetic_repo(tmp_path)
    with fixture["split_path"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if case == "row-count":
        rows.pop()
    elif case == "duplicate-id":
        rows[1]["relative_path"] = rows[0]["relative_path"]
    elif case == "class-count":
        rows[0]["class_id"] = "1"
        rows[0]["breed_name"] = rows[1]["breed_name"]
    elif case == "mapping-mismatch":
        rows[0]["breed_name"] = "wrong breed"
    elif case == "official-path":
        rows[0]["relative_path"] = "synthetic/official_test/marker.jpg"
    with fixture["split_path"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "class_id", "breed_name", "source_split"])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)


def _e7_output_paths(fixture):
    return [
        fixture["root"] / "outputs/metrics/E7/selective_metrics.json",
        fixture["root"] / "outputs/metrics/E7/threshold.json",
        fixture["root"] / "outputs/metrics/E7/risk_coverage.csv",
        fixture["root"] / "outputs/figures/E7/risk_coverage_curve.png",
    ]


def _strict_json(path):
    def reject_constant(value):
        raise AssertionError(f"non-standard JSON number: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _input_bytes(fixture):
    paths = [
        fixture["config_path"],
        fixture["root"] / "configs/model_selection.yaml",
        fixture["root"] / "configs/data.yaml",
        fixture["root"] / "configs/experiments/e4_cutmix.yaml",
        fixture["root"] / "data/metadata/class_mapping.json",
        fixture["split_path"],
        fixture["temperature_path"],
        fixture["metadata_path"],
        fixture["npz_path"],
    ]
    return {path: path.read_bytes() for path in paths}


def test_synthetic_e7_run_writes_valid_auditable_four_outputs_and_refuses_overwrite(tmp_path):
    fixture = _synthetic_repo(tmp_path)
    before_inputs = _input_bytes(fixture)
    result = run_selective_prediction(
        fixture["config_path"], fixture["root"], split="calibration", synthetic=True,
    )
    output_paths = _e7_output_paths(fixture)
    assert all(path.is_file() and path.stat().st_size > 0 for path in output_paths)
    metrics_path, threshold_path, curve_path, figure_path = output_paths
    metrics = _strict_json(metrics_path)
    threshold_record = _strict_json(threshold_path)

    assert result["execution_kind"] == metrics["execution_kind"] == "synthetic_smoke_test"
    assert metrics["experiment"] == "E7"
    assert metrics["synthetic"] is True
    assert metrics["official_e7_execution"] is False
    assert metrics["split"] == "calibration"
    assert metrics["n_total"] == 300
    assert metrics["temperature"] == 2.0
    assert metrics["model"]["experiment"] == "E4"
    assert metrics["model"]["seed"] == 42
    assert metrics["source_git_sha"] and len(metrics["source_git_sha"]) == 40
    assert metrics["upstream_provenance"]["temperature_git_sha"] == "a" * 40
    assert metrics["runtime"]["python"]
    assert metrics["timestamp_utc"].endswith("+00:00")
    assert metrics["input_sha256"] and all(len(value) == 64 for value in metrics["input_sha256"].values())

    assert metrics["target_coverage"] == 0.9
    assert metrics["n_accept"] >= 270
    assert metrics["n_reject"] <= 30
    assert metrics["n_accept"] + metrics["n_reject"] == metrics["n_total"]
    assert metrics["selected_threshold"] == threshold_record["threshold"]
    assert threshold_record["synthetic"] is True
    assert threshold_record["official_e7_execution"] is False
    assert threshold_record["achieved_coverage"] == metrics["coverage"]
    assert metrics["selected_metrics"]["n_accept"] == metrics["n_accept"]

    confidences = calibrated_predictions(fixture["logits"], 2.0)["confidences"]
    required = _decimal_ceil_product(300, 0.9)
    expected_threshold = sorted(confidences.tolist(), reverse=True)[required - 1]
    assert metrics["selected_threshold"] == expected_threshold
    assert np.count_nonzero(confidences >= metrics["selected_threshold"]) >= required
    assert not any(
        candidate > metrics["selected_threshold"]
        and np.count_nonzero(confidences >= candidate) >= required
        for candidate in np.unique(confidences)
    )

    points = metrics["coverage_points"]
    assert [row["requested_coverage"] for row in points] == [1.0, 0.9, 0.8]
    for row, target in zip(points, [1.0, 0.9, 0.8]):
        count = _decimal_ceil_product(300, target)
        expected = sorted(confidences.tolist(), reverse=True)[count - 1]
        assert row["threshold"] == expected
        assert row["n_accept"] >= count

    with curve_path.open("r", encoding="utf-8", newline="") as handle:
        curve = list(csv.DictReader(handle))
    assert len(curve) == metrics["risk_coverage_rows"]
    assert curve[0]["threshold"] == ""
    assert curve[0]["selective_risk"] == ""
    assert curve[0]["selective_accuracy"] == ""
    assert float(curve[0]["coverage"]) == 0.0
    area = 0.0
    previous_coverage = 0.0
    for row in curve[1:]:
        coverage = float(row["coverage"])
        risk = float(row["selective_risk"])
        area += risk * (coverage - previous_coverage)
        previous_coverage = coverage
    assert area == pytest.approx(metrics["aurc"], rel=1e-12, abs=1e-15)
    assert 0.0 <= metrics["aurc"] <= 1.0
    assert metrics["aurc_convention"]["zero_coverage"].startswith("mathematical origin")

    from PIL import Image
    with Image.open(figure_path) as image:
        assert image.format == "PNG"
        image.verify()
    assert _input_bytes(fixture) == before_inputs
    output_before = {path: path.read_bytes() for path in output_paths}
    with pytest.raises(FileExistsError):
        run_selective_prediction(fixture["config_path"], fixture["root"], synthetic=True)
    assert {path: path.read_bytes() for path in output_paths} == output_before


def test_invalid_synthetic_input_fails_without_creating_any_e7_outputs(tmp_path):
    fixture = _synthetic_repo(tmp_path)
    _edit_json(fixture["metadata_path"], lambda obj: obj.__setitem__("weights_updated", True))
    with pytest.raises(ValueError):
        run_selective_prediction(fixture["config_path"], fixture["root"], synthetic=True)
    assert not any(path.exists() for path in _e7_output_paths(fixture))


def test_symlinked_input_is_rejected_without_creating_outputs(tmp_path):
    fixture = _synthetic_repo(tmp_path)
    outside = tmp_path / "external-synthetic.npz"
    shutil.copyfile(fixture["npz_path"], outside)
    fixture["npz_path"].unlink()
    fixture["npz_path"].symlink_to(outside)
    with pytest.raises(ValueError):
        run_selective_prediction(fixture["config_path"], fixture["root"], synthetic=True)
    assert not any(path.exists() for path in _e7_output_paths(fixture))
    assert outside.is_file()


def test_output_collision_or_symlink_is_rejected_without_overwriting_target(tmp_path):
    fixture = _synthetic_repo(tmp_path)
    metrics_path = _e7_output_paths(fixture)[0]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel = tmp_path / "external-sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    metrics_path.symlink_to(sentinel)
    with pytest.raises(ValueError):
        run_selective_prediction(fixture["config_path"], fixture["root"], synthetic=True)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not any(path.exists() for path in _e7_output_paths(fixture)[1:])


def test_cli_runs_synthetic_fixture_from_unrelated_current_directory(tmp_path):
    fixture = _synthetic_repo(tmp_path)
    unrelated = tmp_path / "unrelated-cwd"
    unrelated.mkdir()
    process = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts/run_selective_prediction.py"),
         "--config", str(fixture["config_path"]), "--repo-root", str(fixture["root"]),
         "--split", "calibration", "--synthetic"],
        cwd=unrelated, capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    cli_result = json.loads(process.stdout)
    assert cli_result["execution_kind"] == "synthetic_smoke_test"
    assert len(cli_result["output_paths"]) == 4
    assert all(path.is_file() for path in _e7_output_paths(fixture))


def test_official_test_split_is_rejected_before_config_or_input_files_open(tmp_path, capsys):
    fixture = _synthetic_repo(tmp_path)
    missing_config = fixture["root"] / "missing.yaml"
    status = selective_main([
        "--config", str(missing_config), "--repo-root", str(fixture["root"]),
        "--split", "official_test", "--synthetic",
    ])
    captured = capsys.readouterr()
    assert status != 0
    assert "calibration" in captured.err.lower()
    assert not any(path.exists() for path in _e7_output_paths(fixture))


def test_cli_rejects_synthetic_mode_at_actual_root_before_loading_inputs(tmp_path):
    process = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts/run_selective_prediction.py"),
         "--config", str(_E7_CONFIG), "--repo-root", str(_REPO_ROOT), "--synthetic"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert process.returncode != 0
    assert "alternate temporary root" in process.stderr.lower()


@pytest.mark.parametrize("section,key,value", [
    ("threshold_selection", "target_coverage", "0.9"),
    ("threshold_selection", "target_coverage", None),
    ("evaluation", "metrics", ["coverage", "wrong_metric"]),
    ("evaluation", "generate_risk_coverage_curve", False),
    ("evaluation", "coverage_points", [1.0, float("nan"), 0.8]),
    ("evaluation", "coverage_points", [1.0, float("inf"), 0.8]),
    ("rejection", "enabled", False),
    ("rejection", "message", ""),
    ("outputs", "risk_coverage_data", "../outside.csv"),
    ("outputs", "risk_coverage_figure", "/tmp/curve.png"),
    ("calibration", "temperature_file", "outputs/metrics/E6/../../elsewhere.json"),
])
def test_load_e7_config_rejects_invalid_metric_curve_point_rejection_or_path(tmp_path, section, key, value):
    path = _mutated_config_file(tmp_path, lambda raw: raw[section].__setitem__(key, value))
    with pytest.raises(ValueError):
        load_e7_config(path)


def _mutate_fixture_yaml(fixture, relative, mutate):
    path = fixture["root"] / relative
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(value)
    _write_yaml(path, value)


@pytest.mark.parametrize("relative,mutate", [
    ("configs/model_selection.yaml", lambda value: value["selection"].__setitem__("selected_seed", 7)),
    ("configs/model_selection.yaml", lambda value: value["selection"].__setitem__("status", "pending")),
    ("configs/model_selection.yaml", lambda value: value["model"].__setitem__("architecture", "resnet18")),
    ("configs/model_selection.yaml", lambda value: value["data"].__setitem__("calibration_samples", 299)),
    ("configs/model_selection.yaml", lambda value: value["model"].__setitem__("checkpoint_path", "elsewhere.pt")),
    ("configs/data.yaml", lambda value: value["data"].__setitem__("num_classes", 29)),
    ("configs/data.yaml", lambda value: value["preprocessing"]["evaluation"].__setitem__("resize", 255)),
    ("configs/data.yaml", lambda value: value["preprocessing"]["normalization"].__setitem__("type", "custom")),
    ("configs/experiments/e4_cutmix.yaml", lambda value: value["model"].__setitem__("backbone_frozen", False)),
])
def test_load_e6_artifacts_rejects_mutated_approved_static_configs(tmp_path, relative, mutate):
    fixture = _synthetic_repo(tmp_path)
    _mutate_fixture_yaml(fixture, relative, mutate)
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)


@pytest.mark.parametrize("case", ["source-split", "unsafe-path", "nonnumeric-class"])
def test_load_e6_artifacts_rejects_calibration_rows_outside_protocol(tmp_path, case):
    fixture = _synthetic_repo(tmp_path)
    with fixture["split_path"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if case == "source-split":
        rows[0]["source_split"] = "calibration"
    elif case == "unsafe-path":
        rows[0]["relative_path"] = "../outside.jpg"
    else:
        rows[0]["class_id"] = "1.5"
    with fixture["split_path"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "class_id", "breed_name", "source_split"])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)


def test_load_e6_artifacts_rejects_missing_synthetic_markers_and_sha_mismatch(tmp_path):
    fixture = _synthetic_repo(tmp_path)
    _edit_json(fixture["temperature_path"], lambda obj: obj.pop("synthetic"))
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)

    fixture = _synthetic_repo(tmp_path / "second")
    _edit_json(fixture["metadata_path"], lambda obj: obj.pop("synthetic"))
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)

    fixture = _synthetic_repo(tmp_path / "third")
    _edit_json(fixture["metadata_path"], lambda obj: obj.__setitem__("git_sha", "b" * 40))
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)


def test_regular_existing_output_file_causes_clean_collision_failure(tmp_path):
    fixture = _synthetic_repo(tmp_path)
    metrics_path = _e7_output_paths(fixture)[0]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("sentinel", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_selective_prediction(fixture["config_path"], fixture["root"], synthetic=True)
    assert metrics_path.read_text(encoding="utf-8") == "sentinel"
    assert not any(path.exists() for path in _e7_output_paths(fixture)[1:])


def test_cli_official_test_split_is_rejected_before_opening_config(tmp_path):
    fixture = _synthetic_repo(tmp_path)
    process = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts/run_selective_prediction.py"),
         "--config", str(fixture["root"] / "missing.yaml"), "--repo-root", str(fixture["root"]),
         "--split", "official_test", "--synthetic"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert process.returncode != 0
    assert "calibration" in process.stderr.lower()
    assert not any(path.exists() for path in _e7_output_paths(fixture))


def test_output_write_failure_rolls_back_partial_files(tmp_path, monkeypatch):
    fixture = _synthetic_repo(tmp_path)
    paths = _e7_output_paths(fixture)
    failing_path = paths[1]
    original_open = Path.open

    def fail_on_second_output(path, mode="r", *args, **kwargs):
        if path == failing_path and mode == "xb":
            raise OSError("synthetic injected write failure")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_on_second_output)
    with pytest.raises(OSError, match="synthetic injected write failure"):
        run_selective_prediction(fixture["config_path"], fixture["root"], synthetic=True)
    assert not any(path.exists() for path in paths)


@pytest.mark.parametrize("field,value", [
    ("model_experiment", "E3"),
    ("checkpoint_path", "/synthetic/E4/seed_42/other.pt"),
])
def test_load_e6_artifacts_rejects_wrong_upstream_model_or_checkpoint_path(tmp_path, field, value):
    fixture = _synthetic_repo(tmp_path)
    _edit_json(fixture["temperature_path"], lambda obj: obj.__setitem__(field, value))
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)


def test_load_e6_artifacts_requires_best_checkpoint_filename_in_metadata(tmp_path):
    fixture = _synthetic_repo(tmp_path)
    _edit_json(fixture["metadata_path"], lambda obj: obj.__setitem__("checkpoint_filename", "last.pt"))
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)


def test_load_e6_artifacts_rejects_extra_npz_keys(tmp_path):
    fixture = _synthetic_repo(tmp_path)
    np.savez_compressed(
        fixture["npz_path"], logits=fixture["logits"], labels=fixture["labels"],
        image_ids=fixture["image_ids"], ignored=np.zeros(1),
    )
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)


def test_load_e6_artifacts_rejects_wrong_or_missing_metadata_experiment(tmp_path):
    fixture = _synthetic_repo(tmp_path)
    _edit_json(fixture["metadata_path"], lambda obj: obj.__setitem__("experiment", "E8"))
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)

    fixture = _synthetic_repo(tmp_path / "missing-experiment")
    _edit_json(fixture["metadata_path"], lambda obj: obj.pop("experiment"))
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)


def test_load_e6_artifacts_rejects_checkpoint_with_e4_as_substring_not_path_segment(tmp_path):
    fixture = _synthetic_repo(tmp_path)
    evil_checkpoint = "/synthetic/EvilE4/seed_42/best.pt"
    _edit_json(fixture["temperature_path"], lambda obj: obj.__setitem__("checkpoint_path", evil_checkpoint))
    _edit_json(fixture["metadata_path"], lambda obj: obj.__setitem__("checkpoint_identifier", evil_checkpoint))
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)


def test_load_e6_artifacts_rejects_byte_image_ids_with_wrong_shape_before_decoding(tmp_path):
    fixture = _synthetic_repo(tmp_path)
    encoded = np.asarray([value.encode("utf-8") for value in fixture["image_ids"]], dtype="S64").reshape(300, 1)
    _save_npz(fixture, image_ids=encoded)
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)


def test_load_e6_artifacts_decodes_valid_utf8_byte_image_ids(tmp_path):
    fixture = _synthetic_repo(tmp_path)
    encoded = np.asarray([value.encode("utf-8") for value in fixture["image_ids"]], dtype="S64")
    _save_npz(fixture, image_ids=encoded)
    result = _loaded_synthetic_artifacts(fixture)
    np.testing.assert_array_equal(result["image_ids"], fixture["image_ids"])
    assert result["image_ids"].dtype.kind == "U"


def test_load_e7_config_rejects_unrecognized_output_key(tmp_path):
    path = _mutated_config_file(
        tmp_path, lambda raw: raw["outputs"].__setitem__("extra_output", "outputs/metrics/E7/extra.json")
    )
    with pytest.raises(ValueError):
        load_e7_config(path)


@pytest.mark.parametrize("relative,mutate", [
    ("configs/experiments/e4_cutmix.yaml", lambda value: value.__setitem__("experiment", "E4")),
    ("configs/data.yaml", lambda value: value.__setitem__("preprocessing", "invalid")),
])
def test_load_e6_artifacts_rejects_scalar_nested_static_config_blocks_as_value_error(tmp_path, relative, mutate):
    fixture = _synthetic_repo(tmp_path)
    _mutate_fixture_yaml(fixture, relative, mutate)
    with pytest.raises(ValueError):
        _loaded_synthetic_artifacts(fixture)
