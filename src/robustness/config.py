"""E8 config loader — reads and validates e8_robustness.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GaussianBlurConfig:
    enabled: bool
    kernel_size: int
    sigma: float


@dataclass
class GaussianNoiseConfig:
    enabled: bool
    mean: float
    std: float
    clamp_range: tuple[float, float]


@dataclass
class OcclusionConfig:
    enabled: bool
    shape: str
    side_fraction: float
    fill_rgb: tuple[float, float, float]
    position_policy: str


@dataclass
class ReproducibilityConfig:
    seed: int
    per_image_seed: str
    deterministic: bool


@dataclass
class RestrictionsConfig:
    train_new_model: bool
    modify_checkpoint: bool
    use_official_test: bool


@dataclass
class E8Config:
    experiment_id: str
    status: str
    official_evaluation_enabled: bool
    corruption_stage: str
    pixel_range: tuple[float, float]
    input_size: int
    model_selection_config: str
    gaussian_blur: GaussianBlurConfig
    gaussian_noise: GaussianNoiseConfig
    occlusion: OcclusionConfig
    reproducibility: ReproducibilityConfig
    restrictions: RestrictionsConfig
    conditions: list[str]
    primary_metric: str
    report_accuracy_drop: bool
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def load_e8_config(path: str | Path = "configs/experiments/e8_robustness.yaml") -> E8Config:
    """Load tooling config; reject missing fields, coercions and unsafe execution.

    Invalid YAML raises ValueError with its field path; a missing file raises
    FileNotFoundError. Only parameters supported by this evaluator are accepted.
    """
    p = Path(path)
    try:
        with p.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid E8 YAML in {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"E8 config in {p} must be a non-empty YAML mapping")

    def get(name: str, kind: type):
        value = raw
        for part in name.split("."):
            if not isinstance(value, dict):
                raise ValueError(f"{name}: parent must be a YAML mapping")
            if part not in value:
                raise ValueError(f"Missing required field '{name}'")
            value = value[part]
        valid = type(value) is kind
        if kind is float:
            valid = type(value) in (int, float) and math.isfinite(value)
        if not valid:
            raise ValueError(f"{name}: expected {kind.__name__}"
                             " (numeric values must be finite)")
        return value

    def expected(name: str, value):
        actual = get(name, type(value))
        if actual != value:
            raise ValueError(f"{name} must be {value!r} in Task 03")
        return actual

    def number(name: str, predicate, requirement: str):
        value = get(name, float)
        if not predicate(value):
            raise ValueError(f"{name} must be {requirement}")
        return float(value)

    def rgb_or_range(name: str, length: int):
        values = get(name, list)
        if (len(values) != length or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
            for v in values
        )):
            raise ValueError(f"{name} must contain exactly {length} finite numbers in [0,1]")
        if length == 2 and values[0] >= values[1]:
            raise ValueError(f"{name} must have lower < upper within [0,1]")
        return tuple(float(v) for v in values)

    experiment_id = expected("experiment.id", "E8")
    status = expected("execution.status", "tooling_only")
    official = expected("execution.official_evaluation_enabled", False)
    restrictions = RestrictionsConfig(**{
        key: expected(f"restrictions.{key}", False)
        for key in ("train_new_model", "modify_checkpoint", "use_official_test")
    })
    stage = expected("data.corruption_stage", "after_crop_before_normalization")
    pixel_range = rgb_or_range("data.pixel_range", 2)
    if pixel_range != (0.0, 1.0):
        raise ValueError("data.pixel_range must be [0,1] before normalization")
    expected("data.preserve_labels", True)
    input_size = get("data.input_size", int)
    if input_size <= 0:
        raise ValueError("data.input_size must be positive")
    selection_config = get("model.selection_config", str)
    if not selection_config.strip():
        raise ValueError("model.selection_config must be non-empty")

    kernel = get("corruptions.gaussian_blur.kernel_size", int)
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("corruptions.gaussian_blur.kernel_size must be a positive odd integer")
    blur = GaussianBlurConfig(
        get("corruptions.gaussian_blur.enabled", bool), kernel,
        number("corruptions.gaussian_blur.sigma", lambda v: v > 0, "positive"),
    )
    noise = GaussianNoiseConfig(
        get("corruptions.gaussian_noise.enabled", bool),
        number("corruptions.gaussian_noise.mean", lambda v: True, "finite"),
        number("corruptions.gaussian_noise.std", lambda v: v >= 0, "non-negative"),
        rgb_or_range("corruptions.gaussian_noise.clamp_range", 2),
    )
    occlusion = OcclusionConfig(
        get("corruptions.occlusion.enabled", bool),
        expected("corruptions.occlusion.shape", "square"),
        number("corruptions.occlusion.side_fraction", lambda v: 0 < v < 1, "in (0,1)"),
        rgb_or_range("corruptions.occlusion.fill_rgb", 3),
        expected("corruptions.occlusion.position_policy", "deterministic_per_image"),
    )
    seed = get("reproducibility.seed", int)
    if not 0 <= seed < 2**64:
        raise ValueError("reproducibility.seed must be an integer in [0, 2**64)")
    repro = ReproducibilityConfig(
        seed, expected("reproducibility.per_image_seed", "stable_image_identifier"),
        expected("reproducibility.deterministic", True),
    )
    conditions = get("evaluation.conditions", list)
    enabled = ["clean"] + [name for name, cfg in (
        ("gaussian_blur", blur), ("gaussian_noise", noise), ("occlusion", occlusion)
    ) if cfg.enabled]
    if (any(type(v) is not str for v in conditions)
            or len(conditions) != len(enabled) or set(conditions) != set(enabled)):
        raise ValueError("evaluation.conditions must list clean and each enabled corruption once")
    metric = expected("evaluation.primary_metric", "top1_accuracy")
    report_drop = expected("evaluation.report_accuracy_drop", True)
    expected("evaluation.accuracy_drop_formula", "clean_minus_corrupted")

    return E8Config(
        experiment_id=experiment_id, status=status, official_evaluation_enabled=official,
        corruption_stage=stage, pixel_range=pixel_range, input_size=input_size,
        model_selection_config=selection_config, gaussian_blur=blur,
        gaussian_noise=noise, occlusion=occlusion, reproducibility=repro,
        restrictions=restrictions, conditions=conditions, primary_metric=metric,
        report_accuracy_drop=report_drop, raw=raw,
    )
