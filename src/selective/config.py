"""Strict loader for the frozen E7 selective-prediction configuration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"E7 config field {name!r} must be a mapping")
    return value


def _same(value: Any, expected: Any, name: str) -> None:
    # YAML booleans compare equal to 0/1 in Python, so require exact types.
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"E7 config field {name!r} must be {expected!r}")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"E7 config field {name!r} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"E7 config field {name!r} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"E7 config field {name!r} must be a finite number")
    return result


def validate_e7_config(config: Any) -> dict[str, Any]:
    """Validate the approved E7 mapping and return that same plain mapping."""
    root = _mapping(config, "<root>")
    required_sections = (
        "experiment", "model", "data", "calibration", "threshold_selection",
        "evaluation", "rejection", "outputs", "restrictions",
    )
    for section in required_sections:
        if section not in root:
            raise ValueError(f"E7 config is missing required section {section!r}")
        _mapping(root[section], section)

    experiment = root["experiment"]
    for key, expected in (("id", "E7"), ("name", "selective_prediction"),
                          ("phase", "reliability_development")):
        if key not in experiment:
            raise ValueError(f"E7 config is missing experiment.{key}")
        _same(experiment[key], expected, f"experiment.{key}")

    model = root["model"]
    if "selection_config" not in model:
        raise ValueError("E7 config is missing model.selection_config")
    _same(model["selection_config"], "configs/model_selection.yaml", "model.selection_config")

    data = root["data"]
    for key, expected in (("split", "calibration"), ("split_file", "data/splits/calibration.csv")):
        if key not in data:
            raise ValueError(f"E7 config is missing data.{key}")
        _same(data[key], expected, f"data.{key}")
    for key, expected in (("expected_samples", 300), ("num_classes", 30)):
        if key not in data:
            raise ValueError(f"E7 config is missing data.{key}")
        _same(data[key], expected, f"data.{key}")

    calibration = root["calibration"]
    for key, expected in (
        ("required", True),
        ("temperature_file", "outputs/metrics/E6/temperature.json"),
        ("confidence_source", "calibrated_softmax"),
        ("confidence_method", "max_probability"),
    ):
        if key not in calibration:
            raise ValueError(f"E7 config is missing calibration.{key}")
        _same(calibration[key], expected, f"calibration.{key}")

    selection = root["threshold_selection"]
    fixed_selection = {
        "method": "target_coverage",
        "selection_rule": "highest_feasible_threshold",
        "accept_operator": ">=",
        "tie_policy": "accept_all_equal_to_threshold",
    }
    for key, expected in fixed_selection.items():
        if key not in selection:
            raise ValueError(f"E7 config is missing threshold_selection.{key}")
        _same(selection[key], expected, f"threshold_selection.{key}")
    if "target_coverage" not in selection:
        raise ValueError("E7 config is missing threshold_selection.target_coverage")
    target = _finite_number(selection["target_coverage"], "threshold_selection.target_coverage")
    if target != 0.90:
        raise ValueError("E7 target_coverage is frozen at 0.90")

    evaluation = root["evaluation"]
    if "coverage_points" not in evaluation or not isinstance(evaluation["coverage_points"], list):
        raise ValueError("E7 evaluation.coverage_points must be a list")
    points = [_finite_number(point, "evaluation.coverage_points[]") for point in evaluation["coverage_points"]]
    if points != [1.0, 0.9, 0.8]:
        raise ValueError("E7 coverage_points are frozen at [1.0, 0.9, 0.8]")
    if evaluation.get("metrics") != ["coverage", "selective_accuracy", "selective_risk", "aurc"]:
        raise ValueError("E7 evaluation.metrics must contain the approved metric list")
    if "metrics" not in evaluation:
        raise ValueError("E7 config is missing evaluation.metrics")
    if "generate_risk_coverage_curve" not in evaluation:
        raise ValueError("E7 config is missing evaluation.generate_risk_coverage_curve")
    _same(evaluation["generate_risk_coverage_curve"], True, "evaluation.generate_risk_coverage_curve")

    rejection = root["rejection"]
    if "enabled" not in rejection:
        raise ValueError("E7 config is missing rejection.enabled")
    _same(rejection["enabled"], True, "rejection.enabled")
    if "message" not in rejection:
        raise ValueError("E7 config is missing rejection.message")
    _same(rejection["message"], "Mô hình chưa đủ tự tin để đưa ra kết luận.", "rejection.message")

    outputs = root["outputs"]
    fixed_outputs = {
        "metrics": "outputs/metrics/E7/selective_metrics.json",
        "threshold": "outputs/metrics/E7/threshold.json",
        "risk_coverage_data": "outputs/metrics/E7/risk_coverage.csv",
        "risk_coverage_figure": "outputs/figures/E7/risk_coverage_curve.png",
    }
    if set(outputs) != set(fixed_outputs):
        raise ValueError(f"E7 outputs must contain exactly {sorted(fixed_outputs)}")
    for key, expected in fixed_outputs.items():
        if key not in outputs:
            raise ValueError(f"E7 config is missing outputs.{key}")
        _same(outputs[key], expected, f"outputs.{key}")

    restrictions = root["restrictions"]
    for key, expected in (("requires_e6_completed", True), ("use_official_test", False)):
        if key not in restrictions:
            raise ValueError(f"E7 config is missing restrictions.{key}")
        _same(restrictions[key], expected, f"restrictions.{key}")
    return root


def load_e7_config(config_path: str | Path = "configs/experiments/e7_selective_prediction.yaml") -> dict[str, Any]:
    """Read E7 YAML safely and validate its frozen values and types."""
    path = Path(config_path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"E7 configuration not found: {path}") from exc
    except (yaml.YAMLError, UnicodeError, OSError) as exc:
        raise ValueError(f"Could not read E7 configuration {path}: {exc}") from exc
    return validate_e7_config(config)
