"""Robustness CLI runner — Task 03 tooling-only mode.

Usage:
    python scripts/run_robustness.py --config configs/experiments/e8_robustness.yaml
    python scripts/run_robustness.py --config configs/experiments/e8_robustness.yaml --smoke-test

Official Test mode is permanently disabled in Task 03.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.robustness.config import load_e8_config
from src.robustness.robustness_eval import RobustnessEvaluator


def _make_dummy_model(num_classes: int = 120) -> nn.Module:
    """Tiny linear model for smoke tests — no real checkpoint needed in Task 03."""
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, num_classes))
    model.eval()
    return model


def _make_synthetic_batch(
    n: int = 8,
    h: int = 224,
    w: int = 224,
    num_classes: int = 120,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator()
    g.manual_seed(seed)
    images = torch.rand(n, 3, h, w, generator=g)
    labels = torch.randint(0, num_classes, (n,), generator=g)
    return images, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="E8 Robustness runner (tooling-only)")
    parser.add_argument(
        "--config",
        default="configs/experiments/e8_robustness.yaml",
        help="Path to e8_robustness.yaml",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on synthetic data to verify tooling (no checkpoint required)",
    )
    parser.add_argument(
        "--official-test",
        action="store_true",
        help="[BLOCKED] Official evaluation is disabled in Task 03.",
    )
    args = parser.parse_args()

    # ── Hard block on official test ───────────────────────────────────────────
    if args.official_test:
        print(
            "ERROR: Official Test is disabled in Task 03. "
            "Do not pass --official-test.",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = load_e8_config(args.config)

    if cfg.restrictions.use_official_test:
        print(
            "ERROR: e8_robustness.yaml has use_official_test=true. "
            "Reset to false.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[E8] Config loaded — experiment: {cfg.experiment_id}, status: {cfg.status}")
    print(f"[E8] Conditions: {cfg.conditions}")
    print(f"[E8] official_evaluation_enabled: {cfg.official_evaluation_enabled}")

    model = _make_dummy_model()
    images, labels = _make_synthetic_batch(seed=cfg.reproducibility.seed)

    evaluator = RobustnessEvaluator(model, cfg)
    results = evaluator.evaluate(images, labels)

    print("\n[E8] Synthetic smoke-test results (NOT official):")
    print(json.dumps(results, indent=2, default=str))
    print("\n[E8] Tooling verified. Task 03 complete — Official Evaluation: Not Started.")


if __name__ == "__main__":
    main()
