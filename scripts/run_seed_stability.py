"""E5 CLI: inventory, independent Shared Runner execution, and strict summary."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.seed_stability import collect_summary, plan_runs, run_seed, write_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="E5 tooling; supply the two team-leader-approved original configs.")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Inventory six runs without training or writing outputs.")
    plan.add_argument("--configs", nargs=2, required=True)
    run = commands.add_parser("run", help="Run one independent seed via the unchanged scripts/train.py.")
    run.add_argument("--config", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--dataset-root")
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--num-workers", type=int, default=0)
    run.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    mode = run.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="DataLoader only; no training or run artifacts.")
    mode.add_argument("--execute", action="store_true", help="Explicitly start official training after prechecks.")
    run.add_argument("--drive-upload-confirmed", action="store_true", help="Operator verified target folder and upload permission on the mandatory shared Drive.")
    summary = commands.add_parser("summarize", help="Require all five artifacts for every approved config/seed.")
    summary.add_argument("--configs", nargs=2, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = plan_runs(args.configs)
        elif args.command == "run":
            result = run_seed(args.config, args.seed, dataset_root=args.dataset_root,
                              batch_size=args.batch_size, num_workers=args.num_workers,
                              device=args.device, dry_run=args.dry_run,
                              drive_upload_confirmed=args.drive_upload_confirmed)
        else:
            result = collect_summary(args.configs, project_root=PROJECT_ROOT)
            paths = write_summary(result, PROJECT_ROOT / "outputs/metrics/E5")
            result = {"files": list(map(str, paths)), **result}
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"E5 ERROR: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
