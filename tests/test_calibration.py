from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import yaml

from src.calibration.config import load_e6_config
from src.calibration.temperature_scaling import (
    TemperatureScaler, fit_temperature, save_temperature, load_temperature
)
from src.calibration.calibration_metrics import compute_calibration_metrics, reliability_bins


def _valid_config() -> dict:
    return {
        "experiment": {"id": "E6", "name": "calibration", "phase": "reliability_development"},
        "model": {"selection_config": "configs/model_selection.yaml", "checkpoint_experiment": "E4", "checkpoint_seed": 42},
        "data": {"split": "calibration", "split_file": "data/splits/calibration.csv", "expected_samples": 300, "preprocessing": "evaluation", "num_classes": 30},
        "temperature_scaling": {"method": "scalar_temperature", "initial_temperature": 1.0, "parameterization": "log_temperature", "objective": "nll", "optimizer": "lbfgs", "learning_rate": 0.1, "max_iterations": 100},
        "evaluation": {"ece_bins": 15, "metrics": ["ece", "multiclass_brier", "nll", "top1_accuracy"], "verify_top1_unchanged": True},
        "outputs": {"metrics": "outputs/metrics/E6/calibration_metrics.json", "temperature": "outputs/metrics/E6/temperature.json", "reliability_diagram": "outputs/figures/E6/reliability_diagram.png"},
        "restrictions": {"update_classifier_weights": False, "use_official_test": False},
    }


def _write_config(tmp_path, config: dict):
    path = tmp_path / "e6.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_temperature_is_positive_and_initial_is_valid():
    scaler = TemperatureScaler(1.0)
    assert scaler.temperature.item() > 0
    with pytest.raises(ValueError):
        TemperatureScaler(0.0)
    with pytest.raises(ValueError):
        TemperatureScaler(float("nan"))


def test_softmax_probabilities_are_finite_and_sum_to_one():
    logits = torch.tensor([[2.0, 1.0, 0.0], [-4.0, 3.0, 1.0]])
    scaler = TemperatureScaler(2.5)
    probabilities = torch.softmax(scaler(logits), dim=1)
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-6)
    assert tuple(scaler(logits).shape) == tuple(logits.shape)


def test_temperature_scaling_preserves_top1_predictions():
    logits = torch.tensor([[5.0, 2.0, -1.0], [0.2, 0.3, 0.1], [-2.0, 1.0, 1.5]])
    for temperature in (0.25, 0.7, 1.0, 2.5, 10.0):
        calibrated = logits / temperature
        assert torch.equal(logits.argmax(dim=1), calibrated.argmax(dim=1))


def test_fit_temperature_uses_lbfgs_and_reports_nll():
    # Deliberately overconfident errors make a softer distribution useful.
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [3.0, 0.0], [0.0, 3.0]])
    labels = torch.tensor([0, 0, 1, 1])
    result = fit_temperature(logits, labels, initial_temperature=1.0, learning_rate=0.1, max_iterations=100)
    assert result["temperature"] > 0
    assert math.isfinite(result["temperature"])
    assert result["nll_after"] <= result["nll_before"] + 1e-12
    assert result["optimizer"] == "LBFGS"
    assert result["parameterization"] == "log_temperature"


def test_fit_temperature_rejects_empty_or_bad_shapes():
    with pytest.raises(ValueError):
        fit_temperature(torch.empty((0, 3)), torch.empty((0,), dtype=torch.long))
    with pytest.raises(ValueError):
        fit_temperature(torch.zeros((2, 3)), torch.tensor([0]))
    with pytest.raises(ValueError):
        fit_temperature(torch.zeros((2, 3)), torch.tensor([0, 3]))


def test_fit_temperature_rejects_nan_inf():
    with pytest.raises(ValueError):
        fit_temperature(torch.tensor([[1.0, float("nan")]]), torch.tensor([0]))
    with pytest.raises(ValueError):
        fit_temperature(torch.tensor([[1.0, 0.0]]), torch.tensor([float("inf")]))


def test_calibration_metrics_match_hand_calculation():
    probs = np.array([[0.8, 0.2], [0.6, 0.4]], dtype=np.float64)
    logits = np.log(probs)
    labels = np.array([0, 1], dtype=np.int64)
    result = compute_calibration_metrics(logits, labels, ece_bins=1)
    assert result["top1_accuracy"] == pytest.approx(0.5)
    assert result["ece"] == pytest.approx(0.2)
    assert result["multiclass_brier"] == pytest.approx(0.4)
    assert result["nll"] == pytest.approx((-math.log(0.8) - math.log(0.4)) / 2)


def test_reliability_bins_counts_all_samples_once():
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [0.3, 0.1]])
    labels = np.array([0, 1, 0])
    bins = reliability_bins(logits, labels, n_bins=15)
    assert sum(int(item["count"]) for item in bins) == 3
    assert all(0 <= float(item["accuracy"]) <= 1 for item in bins)


@pytest.mark.parametrize("mutate", [
    lambda c: c["experiment"].update(id="E7"),
    lambda c: c["data"].update(split="official_test", split_file="data/splits/official_test.csv"),
    lambda c: c["data"].update(split_file="data/splits/official_test.csv"),
    lambda c: c["restrictions"].update(use_official_test=True),
    lambda c: c["restrictions"].update(update_classifier_weights=True),
    lambda c: c["evaluation"].update(ece_bins=10),
    lambda c: c["temperature_scaling"].update(learning_rate=0.5),
    lambda c: c["temperature_scaling"].update(max_iterations=10),
    lambda c: c["temperature_scaling"].update(initial_temperature=2.0),
    lambda c: c["data"].update(split_file="data/splits/validation.csv"),
])
def test_e6_loader_rejects_invalid_or_forbidden_config(tmp_path, mutate):
    config = _valid_config()
    mutate(config)
    path = _write_config(tmp_path, config)
    with pytest.raises(ValueError):
        load_e6_config(path)


def test_e6_loader_accepts_valid_config(tmp_path):
    path = _write_config(tmp_path, _valid_config())
    result = load_e6_config(path)
    assert result["experiment"]["id"] == "E6"
    assert result["temperature_scaling"]["optimizer"] == "lbfgs"


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError):
        compute_calibration_metrics(np.array([[1.0, np.inf]]), np.array([0]))
    with pytest.raises(ValueError):
        compute_calibration_metrics(np.zeros((2, 3)), np.array([0]))
    with pytest.raises(ValueError):
        compute_calibration_metrics(np.zeros((2, 3)), np.array([0, 3]))


def test_temperature_save_and_load_round_trip(tmp_path):
    path = tmp_path / "temperature.json"
    value = 1.23456789012345
    save_temperature(path, value, metadata={"experiment": "E6"})
    loaded = load_temperature(path)
    assert loaded == value
    assert json_load(path)["experiment"] == "E6"


def json_load(path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def test_temperature_loader_rejects_nonpositive_or_nonfinite(tmp_path):
    path = tmp_path / "temperature.json"
    path.write_text('{"temperature": 0}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_temperature(path)
    path.write_text('{"temperature": NaN}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_temperature(path)


def test_e6_loader_rejects_missing_required_section(tmp_path):
    config = _valid_config()
    config.pop("temperature_scaling")
    path = _write_config(tmp_path, config)
    with pytest.raises(ValueError):
        load_e6_config(path)


def test_temperature_fitting_does_not_update_model_weights():
    model = torch.nn.Linear(4, 3)
    inputs = torch.tensor([[1.0, 0.0, 0.5, -0.5], [0.1, 1.0, -0.4, 0.2], [0.0, -1.0, 0.3, 0.8]])
    labels = torch.tensor([0, 1, 2])
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.inference_mode():
        logits = model(inputs)
    fit_temperature(logits, labels, max_iterations=10)
    after = model.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)
