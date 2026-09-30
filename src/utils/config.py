"""Load the frozen base/data/experiment configs with recursive merging."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing configuration: {path}")
    with path.open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle) or {}
    if not isinstance(result, dict):
        raise ValueError(f"Expected a YAML mapping at {path}")
    return result


def deep_merge(left: dict, right: dict) -> dict:
    """Return a fresh recursively merged dictionary; right takes precedence."""
    result = deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_experiment_config(experiment_path: str | Path) -> dict[str, Any]:
    experiment_path = Path(experiment_path)
    if not experiment_path.is_absolute():
        experiment_path = PROJECT_ROOT / experiment_path
    config = deep_merge(
        _read_yaml(PROJECT_ROOT / "configs/base.yaml"),
        _read_yaml(PROJECT_ROOT / "configs/data.yaml"),
    )
    config = deep_merge(config, _read_yaml(experiment_path))
    required = ("project", "data", "preprocessing", "experiment", "model", "training", "scheduler")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    if int(config["project"]["num_classes"]) != 30 or int(config["model"]["num_classes"]) != 30:
        raise ValueError("Project/model class count must match the frozen 30-class mapping.")
    if int(config["data"]["split_seed"]) != 42:
        raise ValueError("Dataset split seed is frozen at 42.")
    expected_model = {"E0": "custom_cnn", "E1": "mobilenet_v2",
                      "E2": "mobilenet_v2", "E3": "resnet18"}
    experiment_id = config["experiment"]["id"]
    if experiment_id not in expected_model:
        raise ValueError("This foundation runner handles only E0–E3; later experiments need approved runners.")
    if config["model"]["type"] != expected_model[experiment_id]:
        raise ValueError(f"{experiment_id} must use model type {expected_model[experiment_id]}.")
    return config
