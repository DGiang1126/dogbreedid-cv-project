"""Efficiency Benchmark E0–E4.

Usage:
    python scripts/benchmark_efficiency.py --device auto
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.efficiency import benchmark_model  # noqa: E402
from src.models.model_factory import build_model  # noqa: E402
from src.utils.config import load_experiment_config  # noqa: E402

EXPERIMENTS: list[tuple[str, str, str]] = [
    ("E0", "configs/experiments/e0_custom_cnn.yaml",        "outputs/checkpoints/E0/seed_42/best.pt"),
    ("E1", "configs/experiments/e1_mobilenet_frozen.yaml",  "outputs/checkpoints/E1/seed_42/best.pt"),
    ("E2", "configs/experiments/e2_mobilenet_partial.yaml", "outputs/checkpoints/E2/seed_42/best.pt"),
    ("E3", "configs/experiments/e3_resnet18_partial.yaml",  "outputs/checkpoints/E3/seed_42/best.pt"),
    ("E4", "configs/experiments/e4_cutmix.yaml",            "outputs/checkpoints/E4/seed_42/best.pt"),
]

VAL_MACRO_F1: dict[str, str] = {
    "E0": "19.29%", "E1": "79.96%", "E2": "76.75%", "E3": "74.88%", "E4": "81.51%",
}

OUT_DIR = ROOT / "outputs" / "metrics" / "efficiency"
WARMUP, TIMED = 20, 100


def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def collect_environment(device: torch.device) -> dict:
    env: dict = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "device": str(device),
        "warmup_runs": WARMUP,
        "timed_runs": TIMED,
        "input_shape": [1, 3, 224, 224],
        "precision": "float32",
    }
    if device.type == "cuda":
        env["cuda_device_name"] = torch.cuda.get_device_name(device)
        env["cuda_version"] = torch.version.cuda
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Efficiency Benchmark E0–E4")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"[benchmark] device={device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for exp_id, config_file, ckpt_rel in EXPERIMENTS:
        config_path = ROOT / config_file
        ckpt_path   = ROOT / ckpt_rel
        print(f"\n[{exp_id}] config={config_file}")

        config = load_experiment_config(str(config_path))
        ckpt_exists = ckpt_path.exists()
        if not ckpt_exists:
            warnings.warn(
                f"[{exp_id}] checkpoint not found at {ckpt_path}. "
                "Params/size/latency measured from architecture only. "
                "Download best.pt from shared Drive and re-run for checkpoint_loaded=true.",
                stacklevel=1,
            )

        model = build_model(config)
        result = benchmark_model(
            model, device=device,
            checkpoint_path=str(ckpt_path) if ckpt_exists else None,
            num_classes=30, warmup=WARMUP, timed=TIMED,
        )

        row = {
            "experiment":        exp_id,
            "val_macro_f1":      VAL_MACRO_F1[exp_id],
            "total_params":      result["total_params"],
            "trainable_params":  result["trainable_params"],
            "size_mib":          result["size_mib"],
            "median_ms":         result["median_ms"],
            "mean_ms":           result["mean_ms"],
            "std_ms":            result["std_ms"],
            "checkpoint_loaded": result["checkpoint_loaded"],
        }
        rows.append(row)
        print(
            f"  params={row['total_params']:,}  trainable={row['trainable_params']:,}"
            f"  size={row['size_mib']} MiB  latency={row['median_ms']} ms  ckpt={row['checkpoint_loaded']}"
        )

    csv_path = OUT_DIR / "benchmark.csv"
    fieldnames = ["experiment", "val_macro_f1", "total_params", "trainable_params",
                  "size_mib", "median_ms", "mean_ms", "std_ms", "checkpoint_loaded"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[benchmark] saved -> {csv_path}")

    env_path = OUT_DIR / "environment.json"
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(collect_environment(device), f, indent=2)
    print(f"[benchmark] saved -> {env_path}")

    print("\n=== EFFICIENCY BENCHMARK SUMMARY ===")
    print(f"{'Exp':<5} {'Val F1':>8} {'Total Params':>14} {'Trainable':>12} "
          f"{'Size MiB':>10} {'Median ms':>11} {'Mean+/-Std ms':>18} {'Ckpt':>5}")
    print("-" * 80)
    for r in rows:
        print(
            f"{r['experiment']:<5} {r['val_macro_f1']:>8} {r['total_params']:>14,} "
            f"{r['trainable_params']:>12,} {r['size_mib']:>10.4f} "
            f"{r['median_ms']:>11.4f} {r['mean_ms']:.4f}+/-{r['std_ms']:.4f} "
            f"{'Y' if r['checkpoint_loaded'] else 'N':>5}"
        )


if __name__ == "__main__":
    main()

