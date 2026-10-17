"""E8 config loader — reads and validates e8_robustness.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
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
    """Load and validate e8_robustness.yaml.

    Raises:
        FileNotFoundError: YAML not found.
        ValueError: Required fields missing or invalid.
        RuntimeError: Official evaluation is enabled (must stay false in Task 03).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"E8 config not found: {p}")

    with p.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    # ── required top-level keys ──────────────────────────────────────────────
    for key in ("experiment", "execution", "data", "corruptions", "reproducibility",
                "evaluation", "restrictions"):
        if key not in raw:
            raise ValueError(f"Missing required key '{key}' in {p}")

    exec_cfg = raw["execution"]
    if exec_cfg.get("official_evaluation_enabled", True):
        raise RuntimeError(
            "official_evaluation_enabled must be false in Task 03. "
            "Do not run E8 Official Evaluation."
        )

    data_cfg = raw["data"]
    pr = data_cfg["pixel_range"]

    blur_raw = raw["corruptions"]["gaussian_blur"]
    noise_raw = raw["corruptions"]["gaussian_noise"]
    occ_raw = raw["corruptions"]["occlusion"]
    repro_raw = raw["reproducibility"]
    eval_cfg = raw["evaluation"]
    restr_raw = raw["restrictions"]

    # ── validate approved severity values ────────────────────────────────────
    if blur_raw["kernel_size"] % 2 == 0:
        raise ValueError("gaussian_blur.kernel_size must be odd")
    if noise_raw["std"] < 0:
        raise ValueError("gaussian_noise.std must be non-negative")
    if not (0.0 < occ_raw["side_fraction"] < 1.0):
        raise ValueError("occlusion.side_fraction must be in (0, 1)")

    return E8Config(
        experiment_id=raw["experiment"]["id"],
        status=exec_cfg.get("status", "tooling_only"),
        official_evaluation_enabled=False,
        corruption_stage=data_cfg["corruption_stage"],
        pixel_range=(float(pr[0]), float(pr[1])),
        gaussian_blur=GaussianBlurConfig(
            enabled=blur_raw["enabled"],
            kernel_size=int(blur_raw["kernel_size"]),
            sigma=float(blur_raw["sigma"]),
        ),
        gaussian_noise=GaussianNoiseConfig(
            enabled=noise_raw["enabled"],
            mean=float(noise_raw["mean"]),
            std=float(noise_raw["std"]),
            clamp_range=(float(noise_raw["clamp_range"][0]), float(noise_raw["clamp_range"][1])),
        ),
        occlusion=OcclusionConfig(
            enabled=occ_raw["enabled"],
            shape=occ_raw["shape"],
            side_fraction=float(occ_raw["side_fraction"]),
            fill_rgb=(
                float(occ_raw["fill_rgb"][0]),
                float(occ_raw["fill_rgb"][1]),
                float(occ_raw["fill_rgb"][2]),
            ),
            position_policy=occ_raw["position_policy"],
        ),
        reproducibility=ReproducibilityConfig(
            seed=int(repro_raw["seed"]),
            per_image_seed=repro_raw["per_image_seed"],
            deterministic=bool(repro_raw["deterministic"]),
        ),
        restrictions=RestrictionsConfig(
            train_new_model=bool(restr_raw["train_new_model"]),
            modify_checkpoint=bool(restr_raw["modify_checkpoint"]),
            use_official_test=bool(restr_raw["use_official_test"]),
        ),
        conditions=list(eval_cfg["conditions"]),
        primary_metric=eval_cfg["primary_metric"],
        report_accuracy_drop=bool(eval_cfg["report_accuracy_drop"]),
        raw=raw,
    )
