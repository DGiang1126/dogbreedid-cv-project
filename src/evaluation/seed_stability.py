"""E5 orchestration and statistics; all training stays in scripts/train.py."""
from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import yaml

from src.training.early_stopping import EarlyStopping
from src.utils.config import load_experiment_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 123, 2026)
EXPERIMENTS = frozenset({"E0", "E1", "E2", "E3", "E4"})
CANONICAL_CONFIGS = {
    "E0": "e0_custom_cnn.yaml",
    "E1": "e1_mobilenet_frozen.yaml",
    "E2": "e2_mobilenet_partial.yaml",
    "E3": "e3_resnet18_partial.yaml",
    "E4": "e4_cutmix.yaml",
}
STD_DDOF = 1  # Sample standard deviation over the three independent runs.


def validate_seed(seed: int) -> int:
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError(f"E5 seed must be one of {SEEDS}; received {seed!r}.")
    return seed


def config_for_seed(config: dict, seed: int) -> dict:
    """Copy a resolved base experiment, changing only the training RNG seed."""
    validate_seed(seed)
    if config["experiment"]["id"] not in EXPERIMENTS:
        raise ValueError("Use the approved original E0-E4 config, not the E5 placeholder.")
    if config["data"]["split_seed"] != 42:
        raise ValueError("Dataset split_seed must remain 42.")
    result = deepcopy(config)
    result["experiment"]["seed"] = seed
    result["training"]["seed"] = seed
    return result


def artifact_paths(experiment_id: str, seed: int, project_root: Path = PROJECT_ROOT) -> dict:
    validate_seed(seed)
    if experiment_id not in EXPERIMENTS:
        raise ValueError(f"Invalid original experiment ID: {experiment_id!r}.")
    root = Path(project_root) / "outputs"
    part = Path(experiment_id) / f"seed_{seed}"
    return {
        "best_checkpoint": root / "checkpoints" / part / "best.pt",
        "last_checkpoint": root / "checkpoints" / part / "last.pt",
        "history": root / "logs" / part / "history.csv",
        "metadata": root / "logs" / part / "run_metadata.json",
        "validation": root / "metrics" / part / "validation.json",
    }


def load_original_config(config_path: str | Path) -> dict:
    """Load only an unchanged, canonical E0-E4 config for official tooling.

    Arbitrary overlays can pass the shared loader while silently changing
    experimental settings. Accept copies/alternate paths only when their
    fully merged contents still equal the frozen repository config.
    """
    config = load_experiment_config(config_path)
    experiment_id = config["experiment"]["id"]
    if experiment_id not in CANONICAL_CONFIGS:
        raise ValueError("Use an original frozen E0-E4 experiment config; E5 is a placeholder.")
    canonical = load_experiment_config(PROJECT_ROOT / "configs/experiments" / CANONICAL_CONFIGS[experiment_id])
    if config != canonical:
        raise ValueError(f"{experiment_id} config must match its canonical frozen config exactly before seed override.")
    return config


def _assert_unused(paths: dict) -> None:
    existing = [str(p) for p in paths.values() if p.exists() or p.is_symlink()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing/partial run: " + ", ".join(existing))


def _number(value, label: str, *, probability: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, not {value!r}.")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (probability and value > 1):
        raise ValueError(f"Invalid {label}: {value!r}.")
    return value


def _integer(value, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def read_run(config: dict, seed: int, project_root: Path = PROJECT_ROOT) -> dict:
    """Require five nonempty artifacts and matching config/metadata/history.

    Presence checks do not certify checkpoint strict loading or Drive upload.
    Those are separate Part B checks before experiment delivery.
    """
    experiment_id = config["experiment"]["id"]
    paths = artifact_paths(experiment_id, seed, project_root)
    missing = [str(p) for p in paths.values() if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise ValueError(f"{experiment_id}/seed_{seed}: missing/empty artifacts: " + ", ".join(missing))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metrics = json.loads(paths["validation"].read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metrics, dict):
        raise ValueError("Metadata and validation must be JSON objects.")
    if metadata.get("experiment_id") != experiment_id or type(metadata.get("seed")) is not int or metadata["seed"] != seed:
        raise ValueError(f"{experiment_id}/seed_{seed}: metadata experiment/seed mismatch.")
    if metadata.get("config") != config_for_seed(config, seed):
        raise ValueError(f"{experiment_id}/seed_{seed}: configuration differs beyond the permitted seed override.")
    sha = metadata.get("git_commit", "")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise ValueError("Training metadata must contain an actual full Git SHA.")
    batch = _integer(metadata.get("physical_batch_size"), "physical_batch_size")
    accumulation = _integer(metadata.get("gradient_accumulation_steps"), "gradient_accumulation_steps")
    if batch * accumulation != 32 or metadata.get("effective_batch_size") != 32:
        raise ValueError("Run did not preserve effective batch size 32.")
    best_epoch = _integer(metrics.get("best_epoch"), "best_epoch")
    checked = {key: _number(metrics.get(key), key, probability=key != "loss")
               for key in ("macro_f1", "top1_accuracy", "top5_accuracy", "loss")}
    if checked["top5_accuracy"] < checked["top1_accuracy"]:
        raise ValueError("Top-5 accuracy cannot be lower than Top-1.")
    with paths["history"].open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    max_epochs = int(config["training"]["max_epochs"])
    if not history or len(history) > max_epochs:
        raise ValueError("History is empty or exceeds max_epochs.")
    stopper = EarlyStopping(int(config["training"]["early_stopping_patience"]))
    selected = None
    should_stop = False
    for epoch, row in enumerate(history, 1):
        if row.get("epoch") != str(epoch):
            raise ValueError("History epochs must be consecutive starting at 1.")
        values = {}
        for key in checked:
            try:
                value = float(row["val_" + key])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Missing/invalid history val_{key} at epoch {epoch}.") from error
            values[key] = _number(value, "history val_" + key, probability=key != "loss")
        if values["top5_accuracy"] < values["top1_accuracy"]:
            raise ValueError(f"History Top-5 accuracy is below Top-1 at epoch {epoch}.")
        improved, should_stop = stopper.step(values["macro_f1"])
        if improved:
            selected = (epoch, values)
        if should_stop and epoch != len(history):
            raise ValueError("History continued beyond the first early-stopping condition.")
    if len(history) < max_epochs and not should_stop:
        raise ValueError("Run is incomplete: neither max_epochs nor early stopping reached.")
    if best_epoch != selected[0] or any(not math.isclose(checked[k], selected[1][k], abs_tol=1e-9, rel_tol=1e-9) for k in checked):
        raise ValueError("Best epoch/validation metrics do not match the best history row.")
    return {"experiment_id": experiment_id, "configuration": config["experiment"]["name"],
            "seed": seed, "best_epoch": best_epoch, **checked, "git_commit": sha,
            "validation_path": str(paths["validation"].relative_to(project_root))}


def _load_pair(config_paths) -> list[dict]:
    if len(config_paths) != 2:
        raise ValueError("Supply exactly two configurations approved by the team leader.")
    configs = [load_original_config(path) for path in config_paths]
    ids = [config["experiment"]["id"] for config in configs]
    if len(set(ids)) != 2:
        raise ValueError("Top-2 configurations must have distinct original experiment IDs.")
    for config in configs:
        config_for_seed(config, 42)
    return configs


def plan_runs(config_paths, project_root: Path = PROJECT_ROOT) -> list[dict]:
    """Inventory all six runs without starting training or writing any output."""
    rows = []
    for config in _load_pair(config_paths):
        for seed in SEEDS:
            paths = artifact_paths(config["experiment"]["id"], seed, project_root)
            present = [key for key, p in paths.items() if p.exists() or p.is_symlink()]
            row = {"experiment_id": config["experiment"]["id"], "seed": seed,
                   "artifacts": {key: str(p) for key, p in paths.items()}}
            if not present:
                row.update(status="not_started", reason="No artifacts; independent training required.")
            else:
                try:
                    read_run(config, seed, project_root)
                except (ValueError, KeyError, OSError) as error:
                    row.update(status="existing_invalid_or_incomplete", reason=str(error))
                else:
                    row.update(status="complete_artifacts", reason="Verify checkpoint load and Drive delivery before reuse.")
            rows.append(row)
    return rows


def collect_summary(config_paths, project_root: Path = PROJECT_ROOT) -> dict:
    """Strictly collect six completed runs; never skip invalid/missing seeds."""
    runs, errors, summaries = [], [], []
    for config in _load_pair(config_paths):
        group = []
        for seed in SEEDS:
            try:
                group.append(read_run(config, seed, project_root))
            except (ValueError, KeyError, OSError) as error:
                errors.append(f"{config['experiment']['id']}/seed_{seed}: {error}")
        runs.extend(group)
        if len(group) == len(SEEDS):
            summaries.append({"experiment_id": config["experiment"]["id"],
                              "configuration": config["experiment"]["name"], "n": len(group),
                              "macro_f1_mean": statistics.mean(r["macro_f1"] for r in group),
                              "macro_f1_std": statistics.stdev(r["macro_f1"] for r in group),
                              "top1_mean": statistics.mean(r["top1_accuracy"] for r in group),
                              "top1_std": statistics.stdev(r["top1_accuracy"] for r in group)})
    if errors:
        raise ValueError("Cannot produce an official E5 summary:\n" + "\n".join(errors))
    return {"seeds": list(SEEDS), "metric_unit": "fraction", "std_ddof": STD_DDOF,
            "std_convention": "sample standard deviation (n-1)", "runs": runs,
            "summary": summaries,
            "delivery_note": "Checkpoint strict-load and shared Drive access/upload require separate verification."}


def write_summary(summary: dict, output_dir: Path) -> tuple[Path, Path]:
    """Write CSV/JSON once; exclusive creates protect completed summaries."""
    output_dir = Path(output_dir)
    csv_path = output_dir / "seed_stability_summary.csv"
    json_path = output_dir / "seed_stability_summary.json"
    _assert_unused({"csv": csv_path, "json": json_path})
    buffer = io.StringIO(newline="")
    fields = ["experiment_id", "configuration", "n", "macro_f1_mean", "macro_f1_std", "top1_mean", "top1_std"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary["summary"])
    json_text = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with csv_path.open("x", encoding="utf-8", newline="") as csv_file:
            created.append(csv_path)
            with json_path.open("x", encoding="utf-8") as json_file:
                created.append(json_path)
                csv_file.write(buffer.getvalue())
                json_file.write(json_text)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return csv_path, json_path


def _published_source(project_root: Path) -> str:
    status = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "scripts", "configs", "data/splits", "data/metadata"], cwd=project_root, text=True)
    if status.strip():
        raise ValueError("Commit/push source first; source/config/split working tree must be clean.")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/kiet-e5-stability"], cwd=project_root, check=True)
    return sha


def run_seed(config_path: str | Path, seed: int, *, dataset_root: str | Path | None = None,
             batch_size: int = 8, num_workers: int = 0, device: str = "auto",
             dry_run: bool = True, drive_upload_confirmed: bool = False,
             project_root: Path = PROJECT_ROOT) -> dict:
    """Invoke the unchanged Shared Runner in a fresh process and temporary config.

    This function neither resumes a checkpoint nor modifies frozen config files.
    Official execution requires the operator's prior shared Drive verification.
    """
    project_root = Path(project_root).resolve()
    config = load_original_config(config_path)
    run_config = config_for_seed(config, seed)
    if type(batch_size) is not int or batch_size < 1 or 32 % batch_size:
        raise ValueError("Physical batch size must be a positive divisor of 32.")
    if type(num_workers) is not int or num_workers < 0 or device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("Invalid num_workers or device.")
    paths = artifact_paths(config["experiment"]["id"], seed, project_root)
    if not dry_run:
        if not drive_upload_confirmed:
            raise ValueError("Confirm target folder and upload permission on the shared Drive before training.")
        _assert_unused(paths)
        source_sha = _published_source(project_root)
    else:
        source_sha = None
    lock_path = paths["history"].parent / ".seed_stability.lock"
    lock_owned = False
    started = time.perf_counter()
    try:
        if not dry_run:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("x", encoding="utf-8") as handle:
                lock_owned = True
                handle.write(json.dumps({"experiment_id": config["experiment"]["id"], "seed": seed, "git_commit": source_sha}))
            _assert_unused(paths)
        with tempfile.TemporaryDirectory(prefix="dogbreedid-e5-") as temp:
            temporary_config = Path(temp) / "seed_config.yaml"
            temporary_config.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
            command = [sys.executable, "-u", str(project_root / "scripts/train.py"),
                       "--config", str(temporary_config), "--batch-size", str(batch_size),
                       "--num-workers", str(num_workers), "--device", device]
            if dataset_root is not None:
                command.extend(["--dataset-root", str(dataset_root)])
            if dry_run:
                command.append("--dry-run")
            subprocess.run(command, cwd=project_root, check=True)
        if not dry_run:
            result = read_run(config, seed, project_root)
            if result["git_commit"] != source_sha:
                raise ValueError("Run metadata SHA differs from the actual source used.")
        return {"experiment_id": config["experiment"]["id"], "seed": seed,
                "dry_run": dry_run, "git_commit": source_sha,
                "duration_seconds": time.perf_counter() - started,
                "artifacts": {key: str(p) for key, p in paths.items()}}
    finally:
        if lock_owned:
            lock_path.unlink(missing_ok=True)
