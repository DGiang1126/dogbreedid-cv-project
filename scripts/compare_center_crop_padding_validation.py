"""E4 CenterCrop vs square-padding ablation on the locked Validation CSV only.

No training, calibration, Official Test access, or changes to frozen project files.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import DogBreedDataset
from src.data.transforms import build_transforms
from src.inference import predictor
from app.inference_service import FIT_MODE, PADDING_RGB, TEMPERATURE, THRESHOLD, _artifacts, prepare_image
from src.calibration.calibration_metrics import compute_calibration_metrics
from src.selective.threshold import accept_mask, calibrated_predictions
from src.selective.risk_coverage import selective_metrics


def evaluate(logits: np.ndarray, labels: np.ndarray):
    predicted = logits.argmax(axis=1)
    top5 = torch.from_numpy(logits).topk(5, dim=1).indices.numpy()
    correct_top1 = predicted == labels
    correct_top5 = (top5 == labels[:, None]).any(axis=1)
    calibrated = calibrated_predictions(logits, TEMPERATURE)
    if not np.array_equal(predicted, calibrated["predictions"]):
        raise RuntimeError("Temperature unexpectedly changed Top-1 predictions")
    accepted = accept_mask(calibrated["confidences"], THRESHOLD)
    selective = selective_metrics(correct_top1, accepted)
    ece = compute_calibration_metrics(logits / TEMPERATURE, labels, ece_bins=15)["ece"]
    return {
        "top1_correct": int(correct_top1.sum()),
        "top5_correct": int(correct_top5.sum()),
        "top1_accuracy": float(correct_top1.mean()),
        "top5_accuracy": float(correct_top5.mean()),
        "macro_f1": float(f1_score(labels, predicted, labels=np.arange(30), average="macro", zero_division=0)),
        "ece_after_fixed_e6_temperature": float(ece),
        "accepted_count": int(selective["n_accept"]),
        "coverage_at_fixed_e7_threshold": float(selective["coverage"]),
        "selective_accuracy_at_fixed_e7_threshold": selective["selective_accuracy"],
    }


def main():
    parser = argparse.ArgumentParser(description="Validation only: E4 CenterCrop vs padding")
    parser.add_argument("--run-id", default="e4-crop-padding-val-01")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,63}", args.run_id):
        parser.error("Invalid run ID")
    if args.batch_size < 1:
        parser.error("Batch size must be >= 1")

    output = ROOT / "outputs/research/center_crop_vs_padding_validation" / args.run_id
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists; choose a NEW run ID: {output}")

    csv_path = predictor._require_validation_csv(ROOT, ROOT / "data/splits/validation.csv")
    checkpoint = ROOT / predictor.DEFAULT_CHECKPOINT
    checkpoint_sha = predictor.verify_checkpoint_hash(checkpoint)
    class_names, checked_temperature, checked_threshold = _artifacts(ROOT)
    if len(class_names) != 30 or (checked_temperature, checked_threshold) != (TEMPERATURE, THRESHOLD):
        raise RuntimeError("Locked E6/E7 artifacts are inconsistent")
    data_config = predictor._load_yaml(ROOT / "configs/data.yaml")
    image_root = predictor.find_image_root(ROOT, csv_path)
    ev = data_config["preprocessing"]["evaluation"]
    resize, crop = int(ev["resize"]), int(ev["center_crop"])
    if (resize, crop) != (256, 224):
        raise RuntimeError("Unexpected locked evaluation preprocessing")

    center_transform = build_transforms(data_config, split="validation")
    # Exactly mirror Gradio's FIT mode, then apply the unchanged E4 transform.
    padding_transform = T.Compose([
        T.Lambda(lambda image: prepare_image(image, FIT_MODE, resize=resize, crop=crop)),
        *center_transform.transforms,
    ])
    dataset_center = DogBreedDataset(csv_path, image_root, center_transform)
    dataset_padding = DogBreedDataset(csv_path, image_root, padding_transform)
    if len(dataset_center) != 300 or len(dataset_padding) != 300:
        raise RuntimeError("Expected exactly 300 Validation images")

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model, actual_sha = predictor.load_e4_model(ROOT, checkpoint, device)
    if actual_sha != checkpoint_sha:
        raise RuntimeError("Checkpoint changed during load")

    results = {}
    for name, dataset in (("center_crop", dataset_center), ("padding", dataset_padding)):
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            drop_last=False, pin_memory=(device.type == "cuda"))
        logits, labels, ids = predictor.collect_predictions(model, loader, device=device)
        if len(labels) != 300 or len(np.unique(labels)) != 30:
            raise RuntimeError("Unexpected Validation label distribution")
        results[name] = (logits, labels, ids)
        print(f"Completed {name}: {len(labels)} Validation images", flush=True)

    center, labels, ids = results["center_crop"]
    padding, labels_pad, ids_pad = results["padding"]
    if not np.array_equal(ids, ids_pad) or not np.array_equal(labels, labels_pad):
        raise RuntimeError("Paired image IDs or labels do not match")
    center_metrics = evaluate(center, labels)
    padding_metrics = evaluate(padding, labels)

    # Known E4 seed-42 Validation result, used to catch incorrect checkpoints/transforms.
    if ((center_metrics["top1_correct"], center_metrics["top5_correct"]) != (245, 296)
            or abs(center_metrics["macro_f1"] - 0.815099216821925) > 1e-6):
        raise RuntimeError(f"CenterCrop failed baseline replay: {center_metrics}; no output written")

    c = center.argmax(axis=1)
    p = padding.argmax(axis=1)
    c_ok = c == labels
    p_ok = p == labels
    only_c = int(np.count_nonzero(c_ok & ~p_ok))
    only_p = int(np.count_nonzero(~c_ok & p_ok))
    both_ok = int(np.count_nonzero(c_ok & p_ok))
    both_bad_same = int(np.count_nonzero(~c_ok & ~p_ok & (c == p)))
    both_bad_different = int(np.count_nonzero(~c_ok & ~p_ok & (c != p)))
    if both_ok + only_c + only_p + both_bad_same + both_bad_different != 300:
        raise RuntimeError("Pair counts do not sum to 300")
    different = p_ok.astype(np.int8) - c_ok.astype(np.int8)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(labels), size=(20_000, len(labels)))
    bootstrap = different[indices].mean(axis=1) * 100
    ci = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    discordant = only_c + only_p
    p_value = float(binomtest(only_p, discordant, 0.5).pvalue) if discordant else 1.0

    output.mkdir(parents=True, exist_ok=False)  # never overwrite prior runs
    predictor.save_raw_predictions(output / "center_crop_predictions.npz", center, labels, ids, exclusive=True)
    predictor.save_raw_predictions(output / "padding_predictions.npz", padding, labels, ids, exclusive=True)

    with (output / "paired_predictions.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "true_label", "center_pred", "padding_pred", "center_correct", "padding_correct"])
        for name, label, pred_c, pred_p, ok_c, ok_p in zip(ids, labels, c, p, c_ok, p_ok):
            writer.writerow([name, int(label), int(pred_c), int(pred_p), int(ok_c), int(ok_p)])

    metrics = {
        "split": "validation", "model": "E4 MobileNetV2 Frozen + CutMix, seed 42",
        "n_samples": int(len(labels)), "checkpoint_sha256": checkpoint_sha,
        "batch_size_inference": args.batch_size, "device": str(device),
        "temperature_e6_unchanged": TEMPERATURE,
        "threshold_e7_unchanged": THRESHOLD,
        "center_crop": center_metrics, "padding": padding_metrics,
        "delta_top1_percentage_points_padding_minus_center": float(different.mean() * 100),
        "paired": {"both_correct": both_ok, "only_center_correct": only_c,
                   "only_padding_correct": only_p, "both_wrong_same_class": both_bad_same,
                   "both_wrong_different_class": both_bad_different},
        "mcnemar_exact_two_sided_p": p_value,
        "paired_bootstrap": {"resamples": 20_000, "seed": 42, "ci_95_percentage_points": ci},
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "run_id": args.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": source_sha,
        "script_sha256": predictor.sha256_file(Path(__file__)),
        "split_csv_sha256": predictor.sha256_file(csv_path),
        "data_config_sha256": predictor.sha256_file(ROOT / "configs/data.yaml"),
        "app_inference_service_sha256": predictor.sha256_file(ROOT / "app/inference_service.py"),
        "class_mapping_sha256": predictor.sha256_file(ROOT / "data/metadata/class_mapping.json"),
        "e6_temperature_artifact_sha256": predictor.sha256_file(ROOT / "outputs/metrics/E6/temperature.json"),
        "e7_threshold_artifact_sha256": predictor.sha256_file(ROOT / "outputs/metrics/E7/threshold.json"),
        "checkpoint_sha256": checkpoint_sha,
        "preprocessing": {
            "center_crop": "Resize(256 short side) -> CenterCrop(224) -> ToTensor -> ImageNet Normalize",
            "padding": "app.inference_service.prepare_image(mode='fit', BICUBIC longest=224, canvas=256) -> unchanged E4 transform",
            "padding_rgb": list(PADDING_RGB),
        },
        "files_sha256": {name: predictor.sha256_file(output / name) for name in (
            "center_crop_predictions.npz", "padding_predictions.npz", "paired_predictions.csv", "metrics.json")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Validation comparison saved: {output}")


if __name__ == "__main__":
    main()
