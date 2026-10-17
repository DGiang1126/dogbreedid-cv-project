"""Dedicated E6 YAML loader; does not alter the shared E0-E4 config loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_REQUIRED_TOP_LEVEL = ("experiment", "model", "data", "temperature_scaling", "evaluation", "outputs", "restrictions")


def _read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"E6 configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ValueError("E6 YAML must be a mapping")
    return content


def load_e6_config(config_path: str | Path = "configs/experiments/e6_calibration.yaml") -> dict[str, Any]:
    """Read and strictly validate the approved E6 configuration."""
    config = _read_yaml(config_path)
    missing_top = [key for key in _REQUIRED_TOP_LEVEL if key not in config]
    if missing_top:
        raise ValueError(f"E6 config missing top-level sections: {missing_top}")
    for section in _REQUIRED_TOP_LEVEL:
        if not isinstance(config[section], dict):
            raise ValueError(f"E6 config section {section!r} must be a mapping")

    if config["experiment"].get("id") != "E6":
        raise ValueError("experiment.id must be E6")
    if config["model"].get("checkpoint_experiment") != "E4":
        raise ValueError("E6 must use checkpoint_experiment=E4")
    if int(config["model"].get("checkpoint_seed", -1)) != 42:
        raise ValueError("E6 must use E4 checkpoint seed 42")
    if config["data"].get("split") != "calibration":
        raise ValueError("E6 may only use the calibration split")
    split_file = str(config["data"].get("split_file", "")).replace("\\", "/").lower()
    if split_file != "data/splits/calibration.csv":
        raise ValueError("E6 must use the approved data/splits/calibration.csv; other splits are prohibited")
    if int(config["data"].get("expected_samples", -1)) != 300:
        raise ValueError("E6 expects exactly 300 Calibration samples")
    if int(config["data"].get("num_classes", -1)) != 30:
        raise ValueError("E6 expects 30 classes")
    if config["data"].get("preprocessing") != "evaluation":
        raise ValueError("E6 must use deterministic evaluation preprocessing")

    ts = config["temperature_scaling"]
    if ts.get("method") != "scalar_temperature":
        raise ValueError("E6 method must be scalar_temperature")
    if ts.get("parameterization") != "log_temperature":
        raise ValueError("E6 parameterization must be log_temperature")
    if ts.get("objective") != "nll" or str(ts.get("optimizer", "")).lower() != "lbfgs":
        raise ValueError("E6 must optimize NLL using LBFGS")
    try:
        initial_temperature = float(ts["initial_temperature"])
        learning_rate = float(ts["learning_rate"])
        max_iterations = int(ts["max_iterations"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("E6 temperature scaling numeric fields are missing/invalid") from exc
    if not (initial_temperature > 0 and initial_temperature < float("inf")):
        raise ValueError("initial_temperature must be finite and > 0")
    if abs(initial_temperature - 1.0) > 1e-12:
        raise ValueError("Approved E6 initial_temperature is fixed at 1.0")
    if not (learning_rate > 0 and learning_rate < float("inf")):
        raise ValueError("learning_rate must be finite and > 0")
    if abs(learning_rate - 0.1) > 1e-12:
        raise ValueError("Approved E6 LBFGS learning_rate is fixed at 0.1")
    if max_iterations != 100:
        raise ValueError("Approved E6 max_iterations is fixed at 100")

    bins = int(config["evaluation"].get("ece_bins", -1))
    if bins != 15:
        raise ValueError("E6 must use exactly 15 ECE bins")
    metrics = config["evaluation"].get("metrics")
    required_metrics = {"ece", "multiclass_brier", "nll", "top1_accuracy"}
    if not isinstance(metrics, list) or not required_metrics.issubset(set(metrics)):
        raise ValueError(f"E6 metrics must include {sorted(required_metrics)}")
    if config["evaluation"].get("verify_top1_unchanged") is not True:
        raise ValueError("verify_top1_unchanged must be true")
    if config["restrictions"].get("update_classifier_weights") is not False:
        raise ValueError("E6 must not update classifier weights")
    if config["restrictions"].get("use_official_test") is not False:
        raise ValueError("E6 must prohibit Official Test usage")

    output_keys = {"metrics", "temperature", "reliability_diagram"}
    if not output_keys.issubset(config["outputs"]):
        raise ValueError(f"E6 outputs missing keys: {sorted(output_keys - set(config['outputs']))}")
    for key, value in config["outputs"].items():
        if isinstance(value, str) and "official_test" in value.lower():
            raise ValueError(f"E6 output path {key} must not reference Official Test")

    # Return a regular dict as loaded; the runner resolves relative paths from project root.
    return config
