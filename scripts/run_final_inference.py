"""CLI entry point for Task 04, Task 1 (Validation only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run E4 final classification/calibration inference on the approved "
            "Validation split. Official Test is intentionally not supported."
        )
    )
    parser.add_argument(
        "--root", type=Path, default=None,
        help="Repository root (auto-detected from this script when omitted).",
    )
    parser.add_argument(
        "--split", choices=["validation"], default="validation",
        help="Task 04 permits only the Validation split.",
    )
    parser.add_argument("--split-csv", type=Path, default=None, help="Optional explicit path to data/splits/validation.csv")
    parser.add_argument("--images-dir", type=Path, default=None, help="Image root directly containing breed folders; auto-detect by default")
    parser.add_argument("--checkpoint", type=Path, default=None, help="E4 checkpoint path; defaults to the approved checkpoint")
    parser.add_argument("--temperature-artifact", type=Path, default=None, help="E6 temperature JSON; defaults to outputs/metrics/E6/temperature.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory under repository outputs/; defaults to outputs/predictions/task04_validation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 for safest Windows/Jupyter execution")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    script_path = Path(__file__).resolve()
    default_root = script_path.parent.parent
    root = (args.root or default_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from src.inference.predictor import run_final_inference

        result = run_final_inference(
            repo_root=root,
            split_name=args.split,
            split_csv=args.split_csv,
            images_dir=args.images_dir,
            checkpoint_path=args.checkpoint,
            temperature_artifact=args.temperature_artifact,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
    except Exception as exc:
        print(f"[TASK04-INFERENCE-ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    summary = {
        "n_samples": result["metrics"]["n_samples"],
        "classification": {
            "top1_accuracy": result["metrics"]["classification"]["top1_accuracy"],
            "top5_accuracy": result["metrics"]["classification"]["top5_accuracy"],
            "macro_f1": result["metrics"]["classification"]["macro_f1"],
        },
        "calibration_before": result["metrics"]["calibration"]["before_temperature_scaling"],
        "calibration_after": result["metrics"]["calibration"]["after_temperature_scaling"],
        "temperature": result["temperature"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "paths": result["paths"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nTask 04 runner completed on Validation only. Official Test was not used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
