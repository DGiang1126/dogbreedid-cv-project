"""Task 07 OOD-200 runner: Energy threshold selection and final ID/OOD evaluation.

Two subcommands, run in order:

1. select-threshold
   E4 logits on ID **Calibration** (300 images, CenterCrop) -> Energy score with the
   pre-chosen Energy temperature -> smallest threshold giving >= 90% ID acceptance
   (pass rule energy <= threshold). Writes outputs/metrics/OOD/energy_threshold.json.
   OOD-200 is never read here.

2. evaluate
   Requires data/ood_200/LOCK.json (manifest hash must match) and the locked threshold
   artifact (checksum recorded). Scores ID **Validation** (300) and OOD-200 (200) with
   three methods: E7 Baseline, Energy Detector, integrated pipeline (Energy -> E6/E7).
   Main results use CenterCrop; the Gradio Padding ("fit") mode is evaluated as a
   separate supplementary run with the same locked threshold.
   Raw per-image outputs stay in outputs/research/ood_200/<run_id>/ (local only);
   aggregate metrics go to the same run folder for publication.

Official Test is never read. Existing output folders are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference_service import FIT_MODE, STANDARD_MODE, prepare_image  # noqa: E402
from src.data.transforms import build_transforms  # noqa: E402
from src.inference.predictor import load_e4_model  # noqa: E402
from src.ood import (  # noqa: E402
    DEFAULT_ENERGY_TEMPERATURE, energy_score, id_selective_metrics, is_ood, ood_auroc,
    ood_rejection_metrics, score_summary, select_energy_threshold,
)
from src.selective.threshold import accept_mask, calibrated_predictions  # noqa: E402

E6_TEMPERATURE = 0.80246283961825
E7_THRESHOLD = 0.5011672918550012
TARGET_ID_ACCEPTANCE = 0.9
ALLOWED_ID_SPLITS = {"calibration", "validation"}
THRESHOLD_ARTIFACT = Path("outputs/metrics/OOD/energy_threshold.json")
OOD_ROOT = Path("data/ood_200")
CHECKPOINT = Path("outputs/checkpoints/E4/seed_42/best.pt")
GROUPS = ["unknown_breed", "human", "cat", "other_animal", "object_scene"]


class RunnerError(RuntimeError):
    pass


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        return True


def id_split_rows(split: str) -> list[dict]:
    if split not in ALLOWED_ID_SPLITS:
        raise RunnerError(f"split {split!r} is not allowed (Official Test is locked for Task 07)")
    return list(csv.DictReader(open(ROOT / f"data/splits/{split}.csv", encoding="utf-8")))


class E4Scorer:
    """Batch E4 logits for image paths with CenterCrop (standard) or Padding (fit)."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.checkpoint_hash = load_e4_model(ROOT, ROOT / CHECKPOINT, self.device)
        cfg = yaml.safe_load(open(ROOT / "configs/data.yaml", encoding="utf-8"))
        self.transform = build_transforms(cfg, "inference")
        ev = cfg["preprocessing"]["evaluation"]
        self.resize, self.crop = ev["resize"], ev["center_crop"]

    @torch.inference_mode()
    def logits(self, paths: list[Path], mode: str = STANDARD_MODE, batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(paths), batch_size):
            batch = []
            for p in paths[i:i + batch_size]:
                with Image.open(p) as im:
                    rgb = im.convert("RGB")
                batch.append(self.transform(prepare_image(rgb, mode, resize=self.resize, crop=self.crop)))
            out.append(self.model(torch.stack(batch).to(self.device)).cpu().numpy())
        logits = np.concatenate(out).astype(np.float64)
        if logits.shape != (len(paths), 30) or not np.all(np.isfinite(logits)):
            raise RunnerError("E4 produced invalid logits")
        return logits


def new_run_dir(base: Path, prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = ROOT / base / f"{prefix}_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    return run


def cmd_select_threshold(a) -> int:
    artifact = ROOT / THRESHOLD_ARTIFACT
    if artifact.exists():
        raise RunnerError(f"{THRESHOLD_ARTIFACT} already exists; the threshold is locked")
    images = ROOT / a.images_dir
    rows = id_split_rows("calibration")
    scorer = E4Scorer(a.device)
    logits = scorer.logits([images / r["relative_path"] for r in rows], STANDARD_MODE, a.batch_size)
    scores = energy_score(logits, a.energy_temperature)
    threshold = select_energy_threshold(scores, TARGET_ID_ACCEPTANCE)
    achieved = float((scores <= threshold).mean())

    run = new_run_dir(Path("outputs/research/ood_200"), "threshold")
    np.savez(run / "calibration_logits.npz", logits=logits, energy=scores,
             image_ids=np.array([r["relative_path"] for r in rows]))
    record = {
        "experiment": "OOD-Energy", "task": "Task 07 OOD-200", "method": "energy_score",
        "formula": "E(x) = -T * logsumexp(z / T) over 30 raw E4 logits",
        "energy_temperature": float(a.energy_temperature),
        "energy_temperature_note": "chosen a priori (standard T=1 from Liu et al. 2020); not E6, not tuned",
        "split": "calibration", "n_samples": len(rows), "preprocessing": "Resize(256)->CenterCrop(224)",
        "target_id_acceptance": TARGET_ID_ACCEPTANCE, "selection_rule": "kth_smallest_k=ceil(N*target)",
        "decision_rule": "OOD_REJECT if energy_score > threshold else PASS_OOD_CHECK",
        "tie_policy": "scores equal to threshold pass", "threshold": threshold,
        "achieved_id_acceptance": achieved, "n_pass": int((scores <= threshold).sum()),
        "calibration_energy_summary": score_summary(scores),
        "checkpoint_sha256": scorer.checkpoint_hash,
        "calibration_csv_sha256": sha256_bytes(ROOT / "data/splits/calibration.csv"),
        "e6_temperature_unchanged": E6_TEMPERATURE, "e7_threshold_unchanged": E7_THRESHOLD,
        "ood_200_used": False, "git_sha": git_sha(), "git_dirty": git_dirty(),
        "created_at": run.name.split("_", 1)[1], "raw_logits_run": run.relative_to(ROOT).as_posix(),
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(record, indent=2), encoding="utf-8")
    (run / "energy_threshold.json.sha256").write_text(sha256_bytes(artifact) + "\n", encoding="utf-8")
    print(f"threshold={threshold:.6f} achieved_id_acceptance={achieved:.4f} -> {THRESHOLD_ARTIFACT}")
    print(f"artifact sha256 {sha256_bytes(artifact)}")
    return 0


def load_locked_threshold() -> tuple[dict, str]:
    path = ROOT / THRESHOLD_ARTIFACT
    if not path.is_file():
        raise RunnerError("run select-threshold first; no locked Energy threshold")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("split") != "calibration" or record.get("ood_200_used") is not False:
        raise RunnerError("threshold artifact was not derived from Calibration only")
    return record, sha256_bytes(path)


def verify_ood_lock() -> tuple[list[dict], dict]:
    lock_path = ROOT / OOD_ROOT / "LOCK.json"
    if not lock_path.is_file():
        raise RunnerError("OOD-200 is not locked; run scripts/audit_ood_200.py --lock first")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = ROOT / OOD_ROOT / "manifest.csv"
    if sha256_bytes(manifest) != lock["manifest_sha256"]:
        raise RunnerError("OOD-200 manifest changed after lock")
    return list(csv.DictReader(open(manifest, encoding="utf-8"))), lock


def method_decisions(logits: np.ndarray, energy_threshold: float, energy_temperature: float) -> dict:
    cal = calibrated_predictions(logits, E6_TEMPERATURE)
    conf = cal["confidences"]
    e7_accept = accept_mask(conf, E7_THRESHOLD)
    energy = energy_score(logits, energy_temperature)
    pass_ood = ~is_ood(energy, energy_threshold)
    return {"pred": cal["predictions"], "confidence": conf, "energy": energy,
            "e7_baseline": e7_accept, "energy_detector": pass_ood, "pipeline": pass_ood & e7_accept}


METHODS = ("e7_baseline", "energy_detector", "pipeline")


def summarize(id_d: dict, id_labels: np.ndarray, ood_d: dict, ood_groups: np.ndarray) -> dict:
    result = {"id": {}, "ood_all": {}, "ood_by_group": {}, "auroc": {}}
    for m in METHODS:
        result["id"][m] = id_selective_metrics(id_d[m], id_d["pred"], id_labels)
        result["ood_all"][m] = ood_rejection_metrics(ood_d[m])
    for g in GROUPS:
        sel = ood_groups == g
        result["ood_by_group"][g] = {
            "n": int(sel.sum()),
            "mean_confidence": float(ood_d["confidence"][sel].mean()),
            "confidence_summary": score_summary(ood_d["confidence"][sel]),
            "energy_summary": score_summary(ood_d["energy"][sel]),
            **{m: ood_rejection_metrics(ood_d[m][sel]) for m in METHODS},
            "auroc_confidence": ood_auroc(1 - id_d["confidence"], 1 - ood_d["confidence"][sel]),
            "auroc_energy": ood_auroc(id_d["energy"], ood_d["energy"][sel]),
        }
    result["auroc"] = {
        "positive_class": "OOD",
        "e7_confidence_score": "1 - calibrated_confidence (higher = more OOD)",
        "energy_score": "energy_score (higher = more OOD)",
        "e7_confidence": ood_auroc(1 - id_d["confidence"], 1 - ood_d["confidence"]),
        "energy": ood_auroc(id_d["energy"], ood_d["energy"]),
    }
    result["id_confidence_summary"] = score_summary(id_d["confidence"])
    result["id_energy_summary"] = score_summary(id_d["energy"])
    result["ood_confidence_summary"] = score_summary(ood_d["confidence"])
    result["ood_energy_summary"] = score_summary(ood_d["energy"])
    return result


def write_group_csv(path: Path, results: dict) -> None:
    rows = []
    for mode, res in results.items():
        for m in METHODS:
            idm = res["id"][m]
            rows.append({"preprocessing": mode, "method": m, "group": "ID_validation", "n": idm["n"],
                         "n_accept": idm["n_accept"], "accept_rate": round(idm["id_coverage"], 4),
                         "reject_rate": round(1 - idm["id_coverage"], 4),
                         "id_selective_accuracy": None if idm["id_selective_accuracy"] is None else round(idm["id_selective_accuracy"], 4),
                         "mean_confidence": round(res["id_confidence_summary"]["mean"], 4),
                         "mean_energy": round(res["id_energy_summary"]["mean"], 4)})
            for g in ["ALL"] + GROUPS:
                gm = res["ood_all"][m] if g == "ALL" else res["ood_by_group"][g][m]
                conf = res["ood_confidence_summary"]["mean"] if g == "ALL" else res["ood_by_group"][g]["mean_confidence"]
                en = res["ood_energy_summary"]["mean"] if g == "ALL" else res["ood_by_group"][g]["energy_summary"]["mean"]
                rows.append({"preprocessing": mode, "method": m, "group": f"OOD_{g}", "n": gm["n"],
                             "n_accept": gm["n_accept"], "accept_rate": round(gm["ood_false_accept_rate"], 4),
                             "reject_rate": round(gm["ood_rejection_rate"], 4), "id_selective_accuracy": None,
                             "mean_confidence": round(conf, 4), "mean_energy": round(en, 4)})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cmd_evaluate(a) -> int:
    threshold_record, threshold_sha = load_locked_threshold()
    ood_rows, lock = verify_ood_lock()
    thr, temp = float(threshold_record["threshold"]), float(threshold_record["energy_temperature"])
    images = ROOT / a.images_dir
    id_rows = id_split_rows("validation")
    id_labels = np.array([int(r["class_id"]) for r in id_rows])
    id_paths = [images / r["relative_path"] for r in id_rows]
    ood_paths = [ROOT / OOD_ROOT / r["relative_path"] for r in ood_rows]
    ood_groups = np.array([r["ood_group"] for r in ood_rows])

    scorer = E4Scorer(a.device)
    run = new_run_dir(Path("outputs/research/ood_200"), "eval")
    results, raw_rows = {}, []
    for mode, label in ((STANDARD_MODE, "center_crop"), (FIT_MODE, "padding")):
        id_logits = scorer.logits(id_paths, mode, a.batch_size)
        ood_logits = scorer.logits(ood_paths, mode, a.batch_size)
        id_d = method_decisions(id_logits, thr, temp)
        ood_d = method_decisions(ood_logits, thr, temp)
        results[label] = summarize(id_d, id_labels, ood_d, ood_groups)
        np.savez(run / f"raw_{label}.npz", id_logits=id_logits, ood_logits=ood_logits)
        for src, rows, d, extra in (("ID_validation", id_rows, id_d, None), ("OOD", ood_rows, ood_d, ood_groups)):
            for i, r in enumerate(rows):
                raw_rows.append({
                    "preprocessing": label, "set": src,
                    "image_id": r.get("image_id", r.get("relative_path")),
                    "group": extra[i] if extra is not None else "id", "label": r.get("class_id", r.get("label")),
                    "pred_class": int(d["pred"][i]), "confidence": float(d["confidence"][i]),
                    "energy": float(d["energy"][i]), "e7_accept": bool(d["e7_baseline"][i]),
                    "energy_pass": bool(d["energy_detector"][i]), "pipeline_accept": bool(d["pipeline"][i]),
                })
    with open(run / "per_image_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
        w.writeheader()
        w.writerows(raw_rows)

    summary = {
        "task": "Task 07 OOD-200", "main_preprocessing": "center_crop",
        "supplementary_preprocessing": "padding (Gradio fit mode; same threshold, not re-tuned)",
        "id_set": "Validation (300)", "ood_set": "OOD-200 locked",
        "ood_manifest_sha256": lock["manifest_sha256"], "energy_threshold": thr,
        "energy_temperature": temp, "energy_threshold_artifact_sha256": threshold_sha,
        "e6_temperature": E6_TEMPERATURE, "e7_threshold": E7_THRESHOLD,
        "checkpoint_sha256": scorer.checkpoint_hash, "official_test_accessed": False,
        "git_sha": git_sha(), "git_dirty": git_dirty(), "run_id": run.name, "results": results,
    }
    (run / "ood_200_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_group_csv(run / "ood_200_group_metrics.csv", results)
    cc = results["center_crop"]
    print(f"run -> {run.relative_to(ROOT).as_posix()}")
    for m in METHODS:
        i, o = cc["id"][m], cc["ood_all"][m]
        print(f"{m:<16} ID cov={i['id_coverage']:.3f} selAcc={i['id_selective_accuracy']:.3f} | "
              f"OOD reject={o['ood_rejection_rate']:.3f} FAR={o['ood_false_accept_rate']:.3f}")
    print(f"AUROC confidence={cc['auroc']['e7_confidence']:.4f} energy={cc['auroc']['energy']:.4f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Task 07 OOD-200 Energy detector runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("select-threshold", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--images-dir", default="data/raw/stanford_dogs/Images")
        p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
        p.add_argument("--batch-size", type=int, default=32)
        if name == "select-threshold":
            p.add_argument("--energy-temperature", type=float, default=DEFAULT_ENERGY_TEMPERATURE)
    a = ap.parse_args(argv)
    try:
        return cmd_select_threshold(a) if a.cmd == "select-threshold" else cmd_evaluate(a)
    except RunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
