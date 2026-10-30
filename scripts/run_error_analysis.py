#!/usr/bin/env python3
"""Run E9 aggregate error analysis on an approved saved prediction NPZ.

Never performs model inference. Official mode is disabled by default and requires
both a config opt-in and the explicit --allow-official-analysis CLI flag.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.confusion_matrix import (
    load_class_names, load_predictions_npz, save_e9_artifacts, validate_analysis_split,
)

LOCKED_TEMPERATURE = 0.80246283961825
LOCKED_THRESHOLD = 0.5011672918550012


def load_e9_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"E9 config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("E9 YAML must contain a mapping at its root")
    exp = config.get("experiment")
    if not isinstance(exp, dict) or exp.get("id") != "E9" or exp.get("name") != "error_analysis":
        raise ValueError("experiment.id/name must be E9/error_analysis")
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("model section is required")
    if model.get("experiment_id") != "E4" or model.get("seed") != 42 or model.get("num_classes") != 30:
        raise ValueError("E9 must use locked E4 seed 42 and exactly 30 classes")
    e6, e7 = config.get("e6"), config.get("e7")
    if not isinstance(e6, dict) or not isinstance(e7, dict):
        raise ValueError("e6 and e7 sections are required")
    temperature, threshold = e6.get("temperature"), e7.get("threshold")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("e6.temperature must be finite and > 0")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)) or not 0 <= threshold <= 1:
        raise ValueError("e7.threshold must be finite and in [0,1]")
    if not math.isclose(float(temperature), LOCKED_TEMPERATURE, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"E6 Temperature is locked to {LOCKED_TEMPERATURE}; do not refit or change it")
    if not math.isclose(float(threshold), LOCKED_THRESHOLD, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"E7 threshold is locked to {LOCKED_THRESHOLD}; do not retune or change it")
    if e7.get("accept_operator") != ">=":
        raise ValueError("E7 accept_operator is locked to >=")
    restrictions = config.get("restrictions", {})
    for flag in ("retraining", "official_evaluation_rerun", "refit_temperature", "retune_threshold", "export_official_per_image_rows", "export_official_image_ids"):
        if restrictions.get(flag) is not False:
            raise ValueError(f"restrictions.{flag} must remain false")
    return config


def _reference_metrics(config: dict[str, Any]) -> dict[str, Any] | None:
    official = config.get("official_analysis", {})
    refs = official.get("reference_metrics")
    return refs if isinstance(refs, dict) else None


def run_analysis(
    config_path: str | Path,
    input_path: str | Path,
    split: str,
    output_dir: str | Path | None = None,
    class_mapping_path: str | Path | None = None,
    allow_official_analysis: bool = False,
    split_reference_csv: str | Path | None = None,
    split_id_column: str = "relative_path",
    split_label_column: str = "class_id",
) -> dict[str, Any]:
    config = load_e9_config(config_path)
    split = split.lower().strip()
    if split not in {"synthetic", "validation", "official"}:
        raise ValueError("split must be one of: synthetic, validation, official")

    input_path = Path(input_path)
    if split != "official":
        # Check both the supplied name and a resolved alias before opening NPZ.
        for candidate in (input_path, input_path.resolve()):
            if candidate.name.casefold() == "clean_predictions.npz":
                raise PermissionError("clean_predictions.npz is an Official saved artifact; it cannot be used as Validation or Synthetic")
            if any(part.casefold() in {"final_evaluation_b1", "final_evaluation_b2", "final_evaluation_controls"} for part in candidate.parts):
                raise PermissionError("Evaluation bundles cannot be used as Validation or Synthetic inputs")
    if split == "validation" and (split_id_column, split_label_column) != ("relative_path", "class_id"):
        raise ValueError("Canonical Validation columns are relative_path and class_id")
    if split == "official":
        gate = config.get("official_analysis", {})
        if gate.get("enabled", False) is not True:
            raise PermissionError(
                "Official E9 analysis is disabled in config. Enable it only after the required review/approval."
            )
        if allow_official_analysis is not True:
            raise PermissionError("Official analysis also requires explicit --allow-official-analysis")
        required_name = gate.get("input_filename", "clean_predictions.npz")
        if input_path.name != required_name:
            raise ValueError(f"Official mode only accepts the approved saved artifact named {required_name!r}; no inference is run")
        if split_reference_csv is not None:
            raise PermissionError("Do not provide an Official split CSV; E9 Official analysis must use only the approved saved predictions artifact.")

    num_classes = int(config["model"]["num_classes"])
    predictions = load_predictions_npz(input_path, num_classes=num_classes)
    # Establish split membership before loading display metadata or writing any
    # artifact. A caller-supplied CSV cannot replace the canonical Validation CSV.
    split_check = validate_analysis_split(
        predictions["image_ids"], predictions["labels"], split, split_reference_csv,
    )
    class_path = Path(class_mapping_path) if class_mapping_path else Path(config["class_mapping"]["path"])
    if not class_path.is_absolute():
        class_path = Path.cwd() / class_path
    class_names = load_class_names(class_path, num_classes=num_classes)

    out_path = Path(output_dir) if output_dir else Path(config["paths"]["default_output_dir"])
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path

    ref_metrics = _reference_metrics(config) if split == "official" else None
    summary = save_e9_artifacts(
        logits=predictions["logits"],
        labels=predictions["labels"],
        image_ids=predictions["image_ids"],
        class_names=class_names,
        output_dir=out_path,
        split=split,
        temperature=float(config["e6"]["temperature"]),
        threshold=float(config["e7"]["threshold"]),
        reference_metrics=ref_metrics,
    )
    summary["input_artifact"] = input_path.name if split == "official" else str(input_path)
    if split_check is not None:
        summary["split_reference_check"] = split_check
    summary["official_analysis_enabled_by_config"] = bool(config.get("official_analysis", {}).get("enabled", False))
    (out_path / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nE9 aggregate outputs saved to: {out_path}")
    if split == "official":
        print("Privacy check: official per-image IDs/predictions were not written to output artifacts.")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/e9_error_analysis.yaml")
    parser.add_argument("--input", required=True, help="Saved NPZ containing logits, labels and image_ids")
    parser.add_argument("--split", choices=["synthetic", "validation", "official"], required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--class-mapping", default=None)
    parser.add_argument("--split-reference-csv", default=None, help="Optional canonical data/splits/validation.csv; Validation is always checked against it")
    parser.add_argument("--split-id-column", default="relative_path")
    parser.add_argument("--split-label-column", default="class_id")
    parser.add_argument("--allow-official-analysis", action="store_true", help="Explicit confirmation; still requires config opt-in")
    args = parser.parse_args()
    try:
        run_analysis(
            args.config, args.input, args.split, args.output_dir, args.class_mapping,
            args.allow_official_analysis, args.split_reference_csv, args.split_id_column, args.split_label_column,
        )
    except Exception as exc:
        print(f"E9 ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
