"""Config loader and validator for Task 07 E10/E11 experiments."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.utils.config import PROJECT_ROOT, _read_yaml, deep_merge

SUPPORTED_EXPERIMENTS: dict[str, dict[str, Any]] = {
    "E10-A": {
        "model_type": "efficientnet_b0",
        "source": "torchvision",
        "cutmix_enabled": False,
    },
    "E10-B": {
        "model_type": "efficientnet_b0",
        "source": "torchvision",
        "cutmix_enabled": True,
    },
    "E11-A": {
        "model_type": "vit_tiny_patch16_224",
        "source": "timm",
        "cutmix_enabled": False,
    },
    "E11-B": {
        "model_type": "vit_tiny_patch16_224",
        "source": "timm",
        "cutmix_enabled": True,
    },
}


def load_research_config(experiment_path: str | Path) -> dict[str, Any]:
    """Load and validate an E10/E11 experiment config without altering E0-E4 loader."""
    path = Path(experiment_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    config = deep_merge(
        _read_yaml(PROJECT_ROOT / "configs/base.yaml"),
        _read_yaml(PROJECT_ROOT / "configs/data.yaml"),
    )
    config = deep_merge(config, _read_yaml(path))

    required = (
        "project",
        "data",
        "preprocessing",
        "experiment",
        "model",
        "training",
        "scheduler",
        "augmentation",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")

    if int(config["project"]["num_classes"]) != 30 or int(config["model"]["num_classes"]) != 30:
        raise ValueError("Project/model class count must match the frozen 30-class mapping.")
    if int(config["data"]["split_seed"]) != 42:
        raise ValueError("Dataset split seed is frozen at 42.")

    exp_id = config["experiment"].get("id")
    if exp_id not in SUPPORTED_EXPERIMENTS:
        raise ValueError(f"Unsupported research experiment ID: {exp_id}")

    if int(config["experiment"].get("seed", -1)) != 42:
        raise ValueError(f"{exp_id} must use frozen seed 42.")

    spec = SUPPORTED_EXPERIMENTS[exp_id]
    model_cfg = config["model"]
    if model_cfg.get("type") != spec["model_type"]:
        raise ValueError(f"{exp_id} must use model type {spec['model_type']}.")
    if model_cfg.get("source") != spec["source"]:
        raise ValueError(f"{exp_id} must use model source {spec['source']}.")
    if model_cfg.get("backbone_frozen") is not True:
        raise ValueError(f"{exp_id} must have backbone_frozen: true (no partial/full fine-tuning).")

    train_cfg = config["training"]
    if int(train_cfg.get("effective_batch_size", 0)) != 32:
        raise ValueError("effective_batch_size must be frozen at 32.")
    if int(train_cfg.get("max_epochs", 0)) != 40:
        raise ValueError("max_epochs must be frozen at 40.")
    if int(train_cfg.get("early_stopping_patience", 0)) != 7:
        raise ValueError("early_stopping_patience must be frozen at 7.")
    if train_cfg.get("checkpoint_metric") != "macro_f1":
        raise ValueError("checkpoint_metric must be macro_f1.")

    clr = float(train_cfg.get("classifier_lr", 0.0))
    wd = float(train_cfg.get("weight_decay", -1.0))
    if not math.isfinite(clr) or clr <= 0:
        raise ValueError("training.classifier_lr must be finite and positive.")
    if not math.isfinite(wd) or wd < 0:
        raise ValueError("training.weight_decay must be finite and non-negative.")

    cutmix = config.get("augmentation", {}).get("cutmix")
    if not isinstance(cutmix, dict) or not isinstance(cutmix.get("enabled"), bool):
        raise ValueError(f"{exp_id} must define boolean augmentation.cutmix.enabled.")
    if cutmix["enabled"] is not spec["cutmix_enabled"]:
        raise ValueError(
            f"{exp_id} requires augmentation.cutmix.enabled={spec['cutmix_enabled']}."
        )
    alpha = float(cutmix.get("alpha", 1.0))
    prob = float(cutmix.get("probability", 0.5))
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("CutMix alpha must be finite and greater than zero.")
    if not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
        raise ValueError("CutMix probability must be in [0, 1].")

    return config


def research_output_files(experiment_id: str, seed: int, output_root: Path | None = None) -> dict[str, Path]:
    """Return isolated output paths for E10/E11 so E0-E4 and Official artifacts are untouched."""
    if experiment_id not in SUPPORTED_EXPERIMENTS:
        raise ValueError(f"Unsupported research experiment ID: {experiment_id}")
    base = Path(output_root) if output_root is not None else (PROJECT_ROOT / "outputs" / "research" / "e10_e11")
    run_dir = base / experiment_id / f"seed_{seed}"
    return {
        "run_dir": run_dir,
        "best_checkpoint": run_dir / "best.pt",
        "last_checkpoint": run_dir / "last.pt",
        "history": run_dir / "history.csv",
        "metadata": run_dir / "run_metadata.json",
        "validation_metrics": run_dir / "validation.json",
        "evaluation_summary": run_dir / "evaluation_summary.json",
    }
