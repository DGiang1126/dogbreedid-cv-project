#!/usr/bin/env python3
"""Run official DogBreedID E6 calibration on the approved Calibration Set.

This script never trains or updates E4 model weights and rejects Official Test
configuration. Runtime paths can be supplied from Colab through CLI arguments.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader


def _deep_merge(left: dict, right: dict) -> dict:
    result = deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required YAML not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def _load_e4_merged_config(project_root: Path) -> dict[str, Any]:
    """Use shared loader when it supports E4; fallback locally without editing it.

    The public main branch observed when this runner was prepared had a shared
    loader guarded to E0-E3, although E4's approved YAML exists. In that case,
    merge base.yaml + data.yaml + e4_cutmix.yaml locally and validate the E4
    identity. No repository-shared loader or YAML is changed.
    """
    e4_path = "configs/experiments/e4_cutmix.yaml"
    try:
        from src.utils.config import load_experiment_config
        return load_experiment_config(e4_path)
    except ValueError as exc:
        expected_guard = "handles only E0" in str(exc) or "later experiments need approved runners" in str(exc)
        if not expected_guard:
            raise
        print(
            "WARNING: shared load_experiment_config currently rejects E4; "
            "using E6-local YAML merge (configs/base.yaml + configs/data.yaml + e4_cutmix.yaml). "
            "src/utils/config.py is not modified.",
            file=sys.stderr,
        )
        combined = _deep_merge(_read_yaml(project_root / "configs/base.yaml"),
                                _read_yaml(project_root / "configs/data.yaml"))
        combined = _deep_merge(combined, _read_yaml(project_root / e4_path))
        if combined.get("experiment", {}).get("id") != "E4":
            raise ValueError("Expected E4 experiment config after local merge")
        return combined


def _get_git_sha(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unavailable"


def _resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _resolve_checkpoint(project_root: Path, cli_checkpoint: str | None) -> Path:
    if cli_checkpoint:
        return Path(cli_checkpoint).expanduser().resolve()
    selection = _read_yaml(project_root / "configs/model_selection.yaml")
    model = selection.get("model", {})
    if selection.get("selection", {}).get("selected_experiment") != "E4":
        raise ValueError("model_selection.yaml does not select E4")
    if int(selection.get("selection", {}).get("selected_seed", -1)) != 42:
        raise ValueError("model_selection.yaml must select E4 seed 42")
    return _resolve_path(project_root, model.get("checkpoint_path", "outputs/checkpoints/E4/seed_42/best.pt"))


def _build_e4_model(project_root: Path, e4_config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    from src.models.model_factory import build_model
    model_config = deepcopy(e4_config)
    if model_config.get("experiment", {}).get("id") != "E4":
        raise ValueError("Wrong experiment config: expected E4")
    if int(model_config.get("model", {}).get("num_classes", -1)) != 30:
        raise ValueError("E4 model must have exactly 30 output classes")
    if not bool(model_config["model"].get("backbone_frozen", False)):
        raise ValueError("Approved E4 config must declare backbone_frozen=true")

    # The checkpoint fully supplies the learned weights. Constructing without
    # pretrained downloads is safer in Colab; strict checkpoint loading below
    # verifies that every parameter matches the selected E4 checkpoint.
    model_config["model"]["weights"] = "none"
    model = build_model(model_config)
    return model.to(device)


def _load_checkpoint_strict(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"E4 seed 42 checkpoint not found: {checkpoint_path}\n"
            "Mount Google Drive in Colab, add a shortcut to the shared checkpoint folder, "
            "and pass --checkpoint /content/drive/MyDrive/.../best.pt."
        )
    # Checkpoints in this project are trusted artifacts from the project team.
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # compatibility with older PyTorch versions
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must be a dict containing model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)


def _validate_split_csv(split_csv: Path, expected_samples: int) -> pd.DataFrame:
    if not split_csv.is_file():
        raise FileNotFoundError(f"Calibration split not found: {split_csv}")
    # Explicitly block accidental use of an Official Test path regardless of CLI flags.
    if "official_test" in str(split_csv).replace("\\", "/").lower():
        raise ValueError("Official Test is prohibited for E6")
    frame = pd.read_csv(split_csv)
    required = {"relative_path", "class_id", "breed_name", "source_split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Calibration CSV missing required columns: {sorted(missing)}")
    if frame.empty or len(frame) != expected_samples:
        raise ValueError(f"Expected exactly {expected_samples} Calibration rows, found {len(frame)}")
    if frame["relative_path"].isna().any() or frame["class_id"].isna().any():
        raise ValueError("Calibration split contains missing paths or labels")
    if frame["relative_path"].duplicated().any():
        raise ValueError("Calibration split contains duplicate relative_path values")
    classes = pd.to_numeric(frame["class_id"], errors="raise")
    if not classes.between(0, 29).all() or not np.equal(classes, np.floor(classes)).all():
        raise ValueError("Calibration class_id values must be integers from 0 through 29")
    # source_split identifies the original Stanford Dogs source split (usually
    # 'official_train'), not the experiment split; do not mistake it for E6 split.
    return frame


def _collect_calibration_logits(
    *, project_root: Path,
    e4_config: dict[str, Any],
    split_csv: Path,
    dataset_root: Path,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    from src.data.dataset import DogBreedDataset
    from src.data.transforms import build_transforms

    frame = _validate_split_csv(split_csv, 300)
    images_dir = dataset_root / "Images"
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset root must contain an Images/ directory: {dataset_root}\n"
            "Pass --dataset-root to the Stanford Dogs folder whose child is Images/."
        )

    transform = build_transforms(e4_config, split="calibration")
    dataset = DogBreedDataset(csv_path=split_csv, images_dir=images_dir, transform=transform)
    if len(dataset) != 300:
        raise ValueError(f"Dataset object yielded {len(dataset)} samples; expected 300")
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = _build_e4_model(project_root, e4_config, device)
    _load_checkpoint_strict(model, checkpoint_path, device)
    model.eval()

    logits_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    image_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=(device.type == "cuda"))
            labels = batch["label"].to(device, non_blocking=(device.type == "cuda"))
            logits = model(images)
            if logits.ndim != 2 or tuple(logits.shape[1:]) != (30,):
                raise ValueError(f"E4 logits must be [N, 30], got {tuple(logits.shape)}")
            if not torch.isfinite(logits).all().item():
                raise ValueError("E4 produced NaN/Inf logits")
            logits_parts.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))
            labels_parts.append(labels.detach().cpu().numpy().astype(np.int64, copy=False))
            image_ids.extend([str(item) for item in batch["relative_path"]])

    logits_np = np.concatenate(logits_parts, axis=0)
    labels_np = np.concatenate(labels_parts, axis=0)
    image_ids_np = np.asarray(image_ids, dtype=np.str_)
    if logits_np.shape != (300, 30) or labels_np.shape != (300,) or image_ids_np.shape != (300,):
        raise ValueError(
            f"Unexpected aligned output shapes: logits={logits_np.shape}, "
            f"labels={labels_np.shape}, image_ids={image_ids_np.shape}"
        )
    if not np.array_equal(labels_np, frame["class_id"].to_numpy(dtype=np.int64)):
        raise RuntimeError("Dataset labels/order do not match Calibration CSV")
    if not np.array_equal(image_ids_np, frame["relative_path"].astype(str).to_numpy()):
        raise RuntimeError("Dataset image order does not match Calibration CSV")
    if not np.isfinite(logits_np).all():
        raise ValueError("Collected logits contain NaN/Inf")

    preprocessing = e4_config.get("preprocessing", {}).get("evaluation", {})
    normalization = e4_config.get("preprocessing", {}).get("normalization", {})
    meta = {
        "model": model,
        "preprocessing": {
            "color_mode": "RGB",
            "resize": preprocessing.get("resize", 256),
            "center_crop": preprocessing.get("center_crop", 224),
            "normalization_mean": normalization.get("mean", [0.485, 0.456, 0.406]),
            "normalization_std": normalization.get("std", [0.229, 0.224, 0.225]),
        },
    }
    return logits_np, labels_np, image_ids_np, meta


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Ensure package imports resolve whether run from shell or from Colab.
    project_root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)

    from src.calibration.config import load_e6_config
    from src.calibration.temperature_scaling import fit_temperature, save_temperature
    from src.calibration.calibration_metrics import (
        compute_calibration_metrics,
        plot_reliability_diagram,
    )

    config_path = _resolve_path(project_root, args.config)
    config = load_e6_config(config_path)
    if config["data"].get("split") != "calibration":
        raise ValueError("Runner only supports the Calibration split")
    if config["restrictions"].get("use_official_test") is not False:
        raise ValueError("Official Test use is prohibited")

    e4_config = _load_e4_merged_config(project_root)
    selection = _read_yaml(project_root / config["model"].get("selection_config", "configs/model_selection.yaml"))
    if selection.get("selection", {}).get("selected_experiment") != "E4" or int(selection.get("selection", {}).get("selected_seed", -1)) != 42:
        raise ValueError("Approved model selection must be E4 seed 42")

    checkpoint_path = _resolve_checkpoint(project_root, args.checkpoint)
    split_csv = _resolve_path(project_root, config["data"]["split_file"])
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else project_root / "data/raw/stanford_dogs"
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    logits_np, labels_np, image_ids_np, infer_meta = _collect_calibration_logits(
        project_root=project_root,
        e4_config=e4_config,
        split_csv=split_csv,
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_path,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    logits = torch.from_numpy(logits_np)
    labels = torch.from_numpy(labels_np)
    initial_temperature = float(config["temperature_scaling"]["initial_temperature"])
    fit = fit_temperature(
        logits,
        labels,
        initial_temperature=initial_temperature,
        learning_rate=float(config["temperature_scaling"]["learning_rate"]),
        max_iterations=int(config["temperature_scaling"]["max_iterations"]),
    )
    temperature = float(fit["temperature"])
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("Learned Temperature must be finite and strictly positive")

    calibrated_logits = logits / temperature
    predictions_before = logits.argmax(dim=1).cpu().numpy()
    predictions_after = calibrated_logits.argmax(dim=1).cpu().numpy()
    top1_unchanged = bool(np.array_equal(predictions_before, predictions_after))
    if config["evaluation"].get("verify_top1_unchanged", True) and not top1_unchanged:
        raise RuntimeError("Top-1 predictions changed after Scalar Temperature Scaling")

    ece_bins = int(config["evaluation"]["ece_bins"])
    metrics_before = compute_calibration_metrics(logits_np, labels_np, ece_bins=ece_bins)
    metrics_after = compute_calibration_metrics(calibrated_logits.numpy(), labels_np, ece_bins=ece_bins)
    metrics_payload = {
        "experiment": "E6",
        "model": "E4 MobileNetV2 Frozen + CutMix",
        "checkpoint_seed": 42,
        "split": "calibration",
        "n_samples": 300,
        "temperature": temperature,
        "optimization": fit,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "top1_predictions_unchanged": top1_unchanged,
        "top1_accuracy_unchanged": bool(
            abs(float(metrics_before["top1_accuracy"]) - float(metrics_after["top1_accuracy"])) < 1e-12
        ),
        "metric_definitions": {
            "nll": "mean multiclass negative log-likelihood / cross-entropy from logits",
            "ece": "top-label ECE; 15 equal-width bins, right-closed intervals; empty bins contribute zero",
            "multiclass_brier": "mean over samples of sum_c (p_ic - one_hot(y_i)_c)^2; not divided by C",
            "top1_accuracy": "fraction of samples whose argmax logit equals the label",
        },
    }

    # Output locations follow the assignment even though prediction paths are not in the YAML.
    temp_path = _resolve_path(project_root, config["outputs"]["temperature"])
    metrics_path = _resolve_path(project_root, config["outputs"]["metrics"])
    reliability_path = _resolve_path(project_root, config["outputs"]["reliability_diagram"])
    logits_path = project_root / "outputs/predictions/E6/calibration_logits.npz"
    metadata_path = project_root / "outputs/predictions/E6/calibration_metadata.json"

    save_temperature(temp_path, temperature, metadata={
        "experiment": "E6",
        "method": "scalar_temperature_scaling",
        "initial_temperature": initial_temperature,
        "parameterization": "log_temperature",
        "objective": "nll",
        "optimizer": "LBFGS",
        "nll_before": fit["nll_before"],
        "nll_after": fit["nll_after"],
        "nll_improved": fit["nll_improved"],
        "model_experiment": "E4",
        "checkpoint_seed": 42,
        "checkpoint_path": str(checkpoint_path),
        "split": "calibration",
        "n_samples": 300,
        "git_sha": _get_git_sha(project_root),
    })
    _write_json(metrics_path, metrics_payload)
    logits_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        logits_path,
        logits=logits_np.astype(np.float32, copy=False),
        labels=labels_np.astype(np.int64, copy=False),
        image_ids=image_ids_np.astype(np.str_, copy=False),
    )

    git_sha = _get_git_sha(project_root)
    metadata = {
        "experiment": "E6",
        "source_model": "E4 MobileNetV2 Frozen + CutMix",
        "checkpoint_seed": 42,
        "checkpoint_identifier": str(checkpoint_path),
        "checkpoint_filename": checkpoint_path.name,
        "git_sha": git_sha,
        "split": "calibration",
        "n_samples": int(len(labels_np)),
        "logits_shape": list(logits_np.shape),
        "labels_shape": list(labels_np.shape),
        "image_ids_shape": list(image_ids_np.shape),
        "preprocessing": infer_meta["preprocessing"],
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
        "ordering": "DataLoader shuffle=False; logits, labels and image_ids preserve calibration.csv row order",
        "official_test_used": False,
        "weights_updated": False,
        "source_code_commit_note": "Record/commit the final source Git SHA per project protocol before final submission.",
    }
    _write_json(metadata_path, metadata)
    plot_reliability_diagram(
        logits_np,
        calibrated_logits.detach().cpu().numpy(),
        labels_np,
        n_bins=ece_bins,
        output_path=reliability_path,
        title="DogBreedID E6: Before vs After Temperature Scaling",
    )

    # Generate a factual task report only after the real run completed.
    report_dir = project_root / "docs/task_reports/thuan"
    report_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    report_path = report_dir / f"{now:%d_%m}_task.md"
    report_text = f"""# Task Report — E6 Temperature Scaling

## A. Thông tin thực hiện
- Task / Experiment: Task 01 / E6 — Temperature Scaling
- Model: E4 — MobileNetV2 Frozen + CutMix, seed 42
- Branch / Git SHA hiện tại: `{git_sha}` (hãy cập nhật nếu commit source sau khi chạy)
- Config: `{config_path.relative_to(project_root) if config_path.is_relative_to(project_root) else config_path}`
- Checkpoint: `{checkpoint_path}`
- Split: Calibration, 300 ảnh; Official Test không được sử dụng
- Device: `{device}`
- Command: `python scripts/run_calibration.py --config configs/experiments/e6_calibration.yaml --device {device}`

## B. Kết quả thực nghiệm
- Temperature: `{temperature:.10g}`
- NLL trước / sau: `{fit['nll_before']:.8f}` / `{fit['nll_after']:.8f}`
- ECE trước / sau: `{metrics_before['ece']:.8f}` / `{metrics_after['ece']:.8f}`
- Multiclass Brier trước / sau: `{metrics_before['multiclass_brier']:.8f}` / `{metrics_after['multiclass_brier']:.8f}`
- Top-1 Accuracy trước / sau: `{metrics_before['top1_accuracy']:.8f}` / `{metrics_after['top1_accuracy']:.8f}`
- Top-1 predictions unchanged: `{top1_unchanged}`
- NLL improved: `{fit['nll_improved']}`
- Device and runtime environment are recorded in `outputs/predictions/E6/calibration_metadata.json`.

## C. Artifacts
- `outputs/metrics/E6/temperature.json`
- `outputs/metrics/E6/calibration_metrics.json`
- `outputs/predictions/E6/calibration_logits.npz`
- `outputs/predictions/E6/calibration_metadata.json`
- `outputs/figures/E6/reliability_diagram.png`
- GitHub PR: **add after manual submission**

## D. Kiểm thử và giới hạn
- Unit tests: run `python -m pytest tests/test_calibration.py -v` and record final status.
- Regression tests: run `python -m pytest -q` and record final status.
- No E4 retraining or model-weight updates were performed.
- If `NLL improved` is false, report the outcome honestly; do not change the approved optimizer/config without approval.
- Before submitting, commit the source and tests, record that commit SHA, and update this report if it differs from the runtime SHA.
"""
    report_path.write_text(report_text, encoding="utf-8")

    result = {
        "temperature": temperature,
        "fit": fit,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "top1_predictions_unchanged": top1_unchanged,
        "files": {
            "temperature": str(temp_path),
            "metrics": str(metrics_path),
            "logits": str(logits_path),
            "metadata": str(metadata_path),
            "reliability_diagram": str(reliability_path),
            "task_report": str(report_path),
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/e6_calibration.yaml")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--checkpoint", default=None, help="Path to E4 seed 42 best.pt; overrides model_selection.yaml")
    parser.add_argument("--dataset-root", default=None, help="Stanford Dogs directory whose child is Images/")
    parser.add_argument("--project-root", default=None, help="Root of dogbreedid-cv-project")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.batch_size < 1 or parsed.num_workers < 0:
        raise SystemExit("--batch-size must be >=1 and --num-workers must be >=0")
    run(parsed)
