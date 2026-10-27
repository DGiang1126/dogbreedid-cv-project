"""Phase A readiness audit of existing synthetic/Validation artifacts only.

No model inference, training, fitting, threshold selection or Official Test access.
PASS means this audit passed, never authorization for Official Final Evaluation.
Use load_preflight_report(), not a bare JSON read, to require completed output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid

SOURCE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = SOURCE_ROOT
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
from PIL import Image
import torch
import torchvision

from scripts import run_final_selective as selective
from src.inference import predictor
from src.robustness.config import load_e8_config

PREDICTIONS = Path("outputs/predictions/task05_validation_replay_01")
SELECTIVE = Path("outputs/predictions/task05_e7_validation_replay_01")
ROBUSTNESS = Path("outputs/robustness/task04/validation/e8_validation_committed.json")
E8_REPORT = "docs/task_reports/duc-anh/21_10_task.md"
# Acceptance evidence, not a new corruption/metrics algorithm or authorization.
E8_COMMIT = "cf522ffa0b7d582ddb6557b6ff15b891e51fc117"
E8_HASH = "1014bd8bfd4bd67136230122858d67a6654e55e6fe2cf709f882b53b29caac3f"
THRESHOLD = 0.5011672918550012
SOURCE_FILES = (
    "scripts/run_final_evaluation_preflight.py", "tests/test_final_evaluation_preflight.py",
    "scripts/run_final_inference.py", "scripts/run_final_selective.py", "scripts/run_final_robustness.py",
    "src/inference/predictor.py", "src/evaluation/metrics.py",
    "src/calibration/calibration_metrics.py", "src/calibration/temperature_scaling.py",
    "src/selective/config.py", "src/selective/threshold.py", "src/selective/risk_coverage.py",
    "src/robustness/config.py", "src/robustness/corruptions.py", "src/robustness/robustness_eval.py",
    "src/data/dataset.py", "src/data/transforms.py", "src/models/model_factory.py",
)
REFERENCE_FILES = (
    "configs/data.yaml", "configs/model_selection.yaml", "configs/experiments/e4_cutmix.yaml",
    "configs/experiments/e6_calibration.yaml", "configs/experiments/e7_selective_prediction.yaml",
    "configs/experiments/e8_robustness.yaml", "outputs/metrics/E6/temperature.json",
    "outputs/metrics/E7/threshold.json", "data/metadata/class_mapping.json",
    "docs/final_evaluation/data_contract.md", "docs/final_evaluation/configuration_lock.md",
    "docs/final_evaluation/task05_validation_reproducibility.md", E8_REPORT,
)
REPORT_FILES = ("preflight_summary.json", "preflight_checks.json", "preflight_manifest.json")
RUN_MARKER = ".preflight_run"
FIXTURE_MARKER = ".synthetic_fixture.json"
FIXTURE_FORMAT = "dogbreedid.preflight.synthetic.v1"
REQUIRED_CHECKS = frozenset(("source_integrity", "frozen_reference_integrity", "protocol_references",
    "prediction_bundle", "checkpoint_sha256", "e4_e6_metrics", "e7_metrics", "e8_artifact",
    "e8_execution_provenance", "input_snapshot_stable"))


class Blocked(ValueError):
    """Required evidence is unavailable; do not invent a passing result."""


def _safe_path(value, root: Path) -> Path:
    """Reject lexical redirects and inspect ancestors before following children."""
    path = Path(value)
    if ".." in path.parts:
        raise PermissionError("Parent traversal is forbidden")
    path = path if path.is_absolute() else root / path
    for part in path.parts:
        normalized = part.lower().replace("-", "_").replace(" ", "_")
        if "official_test" in normalized or normalized in {"official", "test"}:
            raise PermissionError("Official Test paths are forbidden in Phase A")
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor /= part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise PermissionError(f"Symlink/junction paths are forbidden: {cursor}")
    path = path.resolve(strict=False)
    for part in path.parts:
        normalized = part.lower().replace("-", "_").replace(" ", "_")
        if "official_test" in normalized or normalized in {"official", "test"}:
            raise PermissionError("Resolved Official Test aliases are forbidden in Phase A")
    if path.is_file() and path.stat().st_nlink > 1:
        raise PermissionError(f"Hard-linked input aliases are forbidden: {path}")
    return path


def _require_files(paths):
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact/reference missing: {path}")


def _fingerprint(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read_once(path, destination=None, limit=None):
    """Hash exactly the bytes returned/copied from one checked regular-file handle.

    Checks before reading reject path/link substitution; handle metadata changes
    during capture fail closed. This is not an OS-wide, atomic filesystem snapshot.
    """
    path = _safe_path(path, SOURCE_ROOT)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as reader:
        initial = os.fstat(reader.fileno())
        if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
            raise PermissionError(f"Snapshot input must be a regular, unaliased file: {path}")
        if limit is not None and initial.st_size > limit:
            raise Blocked(f"Input exceeds the bounded fixture/marker size: {path}")
        checked = _safe_path(path, SOURCE_ROOT)
        if predictor._identity(checked.stat()) != predictor._identity(initial):
            raise Blocked(f"Input was replaced while opening: {path}")
        digest, chunks, size = hashlib.sha256(), [], 0
        while chunk := reader.read(1024 * 1024):
            size += len(chunk)
            if limit is not None and size > limit:
                raise Blocked(f"Input grew beyond its size limit: {path}")
            digest.update(chunk)
            if destination is None:
                chunks.append(chunk)
            else:
                destination.write(chunk)
        if _fingerprint(initial) != _fingerprint(os.fstat(reader.fileno())) or size != initial.st_size:
            raise Blocked(f"Input changed while being captured: {path}")
    return dict(sha256=digest.hexdigest(), size_bytes=size, data=b"".join(chunks) if destination is None else None)


def _parse_json(raw, label):
    def invalid(value):
        raise ValueError(f"Non-finite JSON value in {label}: {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {label}: {key}")
            result[key] = value
        return result
    value = json.loads(raw, parse_constant=invalid, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {label}")
    return value


def _json(path):
    _require_files([path])
    return _parse_json(_read_once(path)["data"], str(path))


def _exact(actual, expected, location):
    """Protocol parameters are not measurements: no numerical tolerance."""
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{location}: frozen protocol value/type differs ({actual!r} != {expected!r})")


def _equal(actual, expected, location="metrics"):
    """Compare schema/counts exactly; numeric metrics at explicit tight tolerance."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"{location}: object schema differs")
        for key in expected:
            _equal(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{location}: list shape differs")
        for index, (a, b) in enumerate(zip(actual, expected)):
            _equal(a, b, f"{location}[{index}]")
    elif type(expected) is float:
        if (type(actual) not in (int, float) or not math.isfinite(actual)
                or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)):
            raise ValueError(f"{location}: numeric metric differs ({actual!r} != {expected!r})")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{location}: value/type differs ({actual!r} != {expected!r})")


def _git_state(root, files):
    """Inspect only the explicitly enumerated source/reference paths, not datasets."""
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)
    try:
        head = git("rev-parse", "HEAD").decode().strip()
        tracked = set(git("ls-files", "-z", "--", *files).decode().split("\0"))
        modified = sorted(filter(None, git("diff", "HEAD", "--name-only", "-z", "--", *files).decode().split("\0")))
        untracked = sorted(set(files) - tracked)
        status = "FAIL" if modified else "BLOCKED" if untracked else "PASS"
        return dict(head=head, status=status, modified=modified, untracked=untracked, audited_files=list(files))
    except (OSError, subprocess.CalledProcessError) as error:
        return dict(head=None, status="BLOCKED", detail=f"Git provenance unavailable: {error}", audited_files=list(files))


class Checks:
    def __init__(self):
        self.rows = []
        self.values = {}

    def run(self, name, function, needs=()):
        try:
            missing = [key for key in needs if key not in self.values]
            if missing:
                raise Blocked(f"Required checks did not pass: {missing}")
            value = function()
            self.values[name] = value
            self.rows.append(dict(id=name, status="PASS", required=True, detail="Evidence validated"))
        except (FileNotFoundError, Blocked) as error:
            self.rows.append(dict(id=name, status="BLOCKED", required=True, detail=f"{type(error).__name__}: {error}"))
        except Exception as error:
            self.rows.append(dict(id=name, status="FAIL", required=True, detail=f"{type(error).__name__}: {error}"))

    def audit(self, name, value):
        self.rows.append(dict(id=name, status=value["status"], required=True, evidence=value))

    def overall(self):
        states = {row["status"] for row in self.rows if row["required"]}
        return "FAIL" if "FAIL" in states else "BLOCKED" if "BLOCKED" in states else "PASS"


def _inputs(mode, root, predictions_dir, e7_dir, e8_artifact, output_dir):
    if mode not in {"validation", "synthetic"}:
        raise PermissionError("Phase A accepts only synthetic or validation")
    root = _safe_path(root, SOURCE_ROOT)
    if mode == "validation" and root != SOURCE_ROOT:
        raise PermissionError("Validation must use this source checkout and its canonical references")
    if mode == "synthetic":
        temporary_root = _safe_path(tempfile.gettempdir(), SOURCE_ROOT)
        if (root == SOURCE_ROOT or root.is_relative_to(SOURCE_ROOT) or SOURCE_ROOT.is_relative_to(root)
                or root == temporary_root or not root.is_relative_to(temporary_root)):
            raise PermissionError("Synthetic fixtures require an isolated child of the OS temporary directory, disjoint from the checkout")
        _fixture_declaration(root)
    prediction = _safe_path(predictions_dir or (PREDICTIONS if mode == "validation" else "outputs/predictions/synthetic"), root)
    e7 = _safe_path(e7_dir or (SELECTIVE if mode == "validation" else "outputs/selective/synthetic"), root)
    e8 = _safe_path(e8_artifact or (ROBUSTNESS if mode == "validation" else "outputs/robustness/synthetic/e8.json"), root)
    if mode == "validation":
        allowed = (root / PREDICTIONS, root / "outputs/predictions/task04_validation_review")
        if prediction not in allowed or e7 != root / SELECTIVE or e8 != root / ROBUSTNESS:
            raise PermissionError("Validation inputs must be the explicitly approved artifact locations")
    if any(not path.is_relative_to(root) for path in (prediction, e7, e8)):
        raise PermissionError("Artifact paths must stay inside the approved reference root")
    output = _safe_path(output_dir, root)
    output_root = root / "outputs/preflight"
    if output == output_root or not output.is_relative_to(output_root):
        raise PermissionError("Output must be a new run beneath outputs/preflight/")
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    for parent in output.parents:
        if not parent.is_relative_to(output_root):
            break
        if os.path.lexists(parent / RUN_MARKER):
            raise PermissionError("Cannot nest a new run inside another preflight output")
    csv = root / ("data/splits/validation.csv" if mode == "validation" else "data/splits/synthetic.csv")
    paths = {"npz": prediction / "validation_raw_predictions.npz", "metadata": prediction / "validation_metadata.json",
             "metrics": prediction / "validation_metrics.json", "diagram": prediction / "validation_reliability_diagram.png",
             "e7": e7 / "selective_metrics.json", "e7_csv": e7 / "risk_coverage.csv", "e7_png": e7 / "risk_coverage_curve.png",
             "e8": e8, "csv": csv, "checkpoint": root / predictor.DEFAULT_CHECKPOINT}
    if mode == "synthetic":
        paths["fixture_manifest"] = root / FIXTURE_MARKER
    for label, path in paths.items():
        paths[label] = _safe_path(path, root)
        if not paths[label].is_relative_to(root):
            raise PermissionError("Every input must remain within the approved root")
        if mode == "synthetic" and paths[label].is_relative_to(SOURCE_ROOT):
            raise PermissionError("Synthetic inputs cannot reference the real checkout")
        if output == path or path.is_relative_to(output) or output.is_relative_to(path):
            raise PermissionError("Output cannot overlap an input")
    for relative in REFERENCE_FILES:
        _safe_path(root / relative, root)
    for relative in SOURCE_FILES:
        _safe_path(CODE_ROOT / relative, CODE_ROOT)
    return root, paths, output


def _fixture_declaration(root):
    marker = _safe_path(root / FIXTURE_MARKER, root)
    if not marker.is_file():
        raise PermissionError("Synthetic mode requires an explicitly generated fixture declaration")
    declaration = _parse_json(_read_once(marker, limit=65536)["data"], str(marker))
    if (declaration.get("format") != FIXTURE_FORMAT or declaration.get("synthetic") is not True
            or declaration.get("generator") != "tests/test_final_evaluation_preflight.py"):
        raise PermissionError("Unsupported synthetic fixture declaration")
    return declaration


def _manifest(paths):
    records = {}
    for path in sorted(set(paths), key=str):
        checked = _safe_path(path, SOURCE_ROOT)
        captured = _read_once(checked) if checked.is_file() else None
        records[str(path)] = dict(path=str(path), sha256=captured["sha256"] if captured else None)
    return records


def _head_reference_bytes(root, commit, relative):
    return subprocess.check_output(["git", "-C", str(root), "show", f"{commit}:{relative}"], stderr=subprocess.PIPE)


class InputSnapshot:
    """Private read-only copies; validators never reopen mutable public inputs.

    Content hashes are generated while copying the same stream, not by reopening
    its source path. Files are captured individually, not at one atomic instant.
    The private directory assumes no hostile same-user/admin tampering.
    """
    def __init__(self, original_root, paths, private_root, mode):
        self.root = private_root
        self.root.mkdir(mode=0o700)
        self.original_root = original_root
        self.records, self.errors, self.copies = {}, [], {}
        inputs = set(paths.values()) | {original_root / name for name in REFERENCE_FILES}
        inputs |= {directory / name for directory in {paths["npz"].parent, paths["e7"].parent, paths["e8"].parent}
                   for name in (".inference_run", ".complete", ".incomplete")}
        for original in sorted(inputs, key=str):
            relative = original.relative_to(original_root)
            destination = self.root / relative
            record = dict(path=str(original), sha256=None, snapshot_relative_path=relative.as_posix(), state="missing")
            self.records[str(original)] = record
            try:
                checked = _safe_path(original, original_root)
                if not checked.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as writer:
                    limit = 1024 if mode == "synthetic" and original == paths["checkpoint"] else None
                    captured = _read_once(checked, destination=writer, limit=limit)
                record.update(sha256=captured["sha256"], size_bytes=captured["size_bytes"], state="captured")
                self.copies[original] = destination
                destination.chmod(stat.S_IREAD)
            except (OSError, ValueError) as error:
                record.update(state="capture_failed", error=f"{type(error).__name__}: {error}")
                self.errors.append(record["error"])
        self.paths = {key: self.root / path.relative_to(original_root) for key, path in paths.items()}
        # Source hashes are observations only; Python modules were imported from
        # the real checkout and Git provenance is audited separately.
        for relative in SOURCE_FILES:
            source = CODE_ROOT / relative
            try:
                captured = _read_once(source)
                self.records[str(source)] = dict(path=str(source), sha256=captured["sha256"], state="observed_source")
            except (OSError, ValueError) as error:
                self.errors.append(f"Source capture unavailable: {source}: {error}")

    def verify(self):
        if self.errors:
            raise Blocked("Input snapshot capture failed: " + "; ".join(self.errors))
        for original, copy in self.copies.items():
            if _read_once(copy)["sha256"] != self.records[str(original)]["sha256"]:
                raise ValueError(f"Private snapshot content changed: {original}")
        return True

    def verify_references(self, commit, git_root=None, include_csv=True):
        if not commit:
            raise Blocked("No Git commit available to bind frozen snapshot references")
        names = (*REFERENCE_FILES, self.paths["csv"].relative_to(self.root).as_posix()) if include_csv else REFERENCE_FILES
        for relative in names:
            path = self.root / relative
            _require_files([path])
            expected = _head_reference_bytes(git_root or self.original_root, commit, relative)
            actual = _read_once(path)["data"]
            if actual.replace(b"\r\n", b"\n") != expected.replace(b"\r\n", b"\n"):
                raise ValueError(f"Captured frozen reference differs from Git {commit}: {relative}")
        return True

    def verify_live_drift(self):
        # Supplemental only: ABA on original files cannot change what validators
        # read from self.root, even if this best-effort observation misses it.
        for name, record in self.records.items():
            path = _safe_path(name, SOURCE_ROOT)
            current = _read_once(path)["sha256"] if path.is_file() else None
            if current != record["sha256"]:
                raise ValueError(f"Original input changed after snapshot capture: {path}")
        return True


def _artifact_markers(directory):
    """Legacy E7/E8 have no markers; any managed bundle must be completed."""
    if os.path.lexists(directory / ".incomplete"):
        raise Blocked(f"Incomplete input artifact directory: {directory}")
    markers = [directory / name for name in (".inference_run", ".complete")]
    if any(os.path.lexists(path) for path in markers):
        try:
            tokens = [_safe_path(path, directory).read_bytes() for path in markers]
        except FileNotFoundError as error:
            raise Blocked(f"Managed input has no completion record: {directory}") from error
        if (len(tokens[0]) != 33 or tokens[0][-1:] != b"\n" or tokens[0] != tokens[1]
                or any(c not in b"0123456789abcdef" for c in tokens[0][:-1])):
            raise Blocked(f"Invalid managed input completion record: {directory}")


def _protocol(root):
    _require_files([root / p for p in REFERENCE_FILES])
    # The shared E7 loader follows these config values. Reject redirects before it reads them.
    config = predictor._load_yaml(root / "configs/experiments/e7_selective_prediction.yaml")
    if (config.get("calibration", {}).get("temperature_file") != "outputs/metrics/E6/temperature.json"
            or config.get("outputs", {}).get("threshold") != "outputs/metrics/E7/threshold.json"):
        raise PermissionError("Frozen artifact configuration redirects outside the canonical paths")
    refs = selective._load_references(root)
    e4 = predictor._load_yaml(root / "configs/experiments/e4_cutmix.yaml")
    if e4["model"].get("backbone_frozen") is not True or e4.get("augmentation", {}).get("cutmix", {}).get("enabled") is not True:
        raise ValueError("Approved E4 must be Frozen + CutMix")
    predictor._load_temperature(refs["temperature_path"])
    if refs["threshold"] != THRESHOLD:
        raise ValueError("E7 threshold differs from Configuration Lock")
    lock = (root / "docs/final_evaluation/configuration_lock.md").read_text(encoding="utf-8")
    if "**Trạng thái:** BẢN NHÁP — CHƯA KHÓA CẤU HÌNH" not in lock:
        raise Blocked("Phase A requires review of the Configuration Lock state; it cannot grant authorization")
    return refs


def _bundle(paths, root, mode):
    _require_files([paths[k] for k in ("npz", "metadata", "csv", "diagram")])
    try:
        predictor.assert_prediction_bundle_complete(paths["npz"])
    except predictor.InferenceContractError as error:
        raise Blocked(str(error)) from error
    bundle = selective.load_prediction_bundle(paths["npz"], paths["metadata"], split=mode, split_csv=paths["csv"], repo_root=root)
    raw = predictor.load_raw_predictions(paths["npz"])
    preprocessing = predictor._load_yaml(root / "configs/data.yaml")["preprocessing"]
    expected_preprocessing = dict(config="configs/data.yaml", transform_builder="src.data.transforms.build_transforms",
                                  split_argument=mode, expected_tensor_shape=[3, 224, 224],
                                  evaluation_resize=preprocessing["evaluation"]["resize"],
                                  evaluation_center_crop=preprocessing["evaluation"]["center_crop"],
                                  normalization=preprocessing["normalization"])
    _exact(bundle["metadata"].get("preprocessing"), expected_preprocessing, "E4.preprocessing")
    for key, value in dict(temperature=bundle["temperature"], temperature_source_experiment="E6",
                           temperature_source_split="calibration", temperature_artifact_path="outputs/metrics/E6/temperature.json",
                           checkpoint_path=predictor.DEFAULT_CHECKPOINT.as_posix()).items():
        _exact(bundle["metadata"].get(key), value, f"E4.metadata.{key}")
    for key in ("official_test_used", "model_trained_or_updated", "temperature_refit"):
        _equal(bundle["metadata"].get("safety", {}).get(key), False, f"E4.safety.{key}")
    bundle["class_names"] = predictor._assert_npz_alignment_with_csv(paths["csv"], raw["image_ids"], raw["labels"])
    if mode == "validation" and (len(raw["labels"]) != 300 or len(np.unique(raw["labels"])) != 30):
        raise ValueError("Validation must contain the approved 300 samples and 30 classes")
    if mode == "synthetic" and any(not str(value).startswith("synthetic/") for value in raw["image_ids"]):
        raise ValueError("Synthetic IDs must use the synthetic/ prefix")
    with Image.open(paths["diagram"]) as image:
        image.verify()
    return bundle


def _classification(paths, bundle):
    actual = _json(paths["metrics"])
    expected = predictor.compute_task1_metrics(bundle["logits"], bundle["labels"], temperature=bundle["temperature"], class_names=bundle["class_names"])
    _equal(actual, expected, "E4/E6")
    return {"classification": {k: v for k, v in actual["classification"].items() if k not in {"per_class", "confusion_matrix"}},
            "calibration": actual["calibration"], "n_samples": actual["n_samples"], "n_classes": actual["n_classes"]}


def _selective(paths, bundle, mode):
    _artifact_markers(paths["e7"].parent)
    _require_files([paths[k] for k in ("e7", "e7_csv", "e7_png")])
    actual = _json(paths["e7"])
    for key, expected in {"experiment_id": "E4_TASK04_FINAL_SELECTIVE", "split": mode,
                          "synthetic": mode == "synthetic", "official_final_evaluation": False,
                          "threshold_reselected": False, "accept_operator": ">=",
                          "aurc_convention": "grouped right-step over confidence tie groups",
                          "temperature": bundle["temperature"], "threshold": bundle["threshold"],
                          "upstream_git_sha": bundle["metadata"]["git_sha"], "n_total": len(bundle["labels"])}.items():
        _exact(actual.get(key), expected, f"E7.{key}")
    producer = actual.get("source_git_sha")
    if not isinstance(producer, str) or len(producer) != 40 or any(c not in "0123456789abcdefABCDEF" for c in producer):
        raise Blocked("E7 does not identify its producer source commit")
    for key, expected in {"experiment": "E4", "architecture": "mobilenet_v2", "seed": 42, "num_classes": 30,
                          "checkpoint_sha256": predictor.EXPECTED_CHECKPOINT_SHA256 if mode == "validation" else None}.items():
        _equal(actual.get("model", {}).get(key), expected, f"E7.model.{key}")
    _equal(actual.get("input_sha256"), bundle["input_sha256"], "E7.input_sha256")
    calibrated = selective.calibrated_predictions(bundle["logits"], bundle["temperature"])
    correct = calibrated["predictions"] == bundle["labels"]
    accepted = selective.accept_mask(calibrated["confidences"], bundle["threshold"])
    _equal(actual.get("selected_metrics"), selective.selective_metrics(correct, accepted), "E7.selected_metrics")
    _equal(actual.get("aurc"), selective.compute_aurc(calibrated["confidences"], correct), "E7.aurc")
    expected_rows = selective.risk_coverage_curve(calibrated["confidences"], correct)
    with paths["e7_csv"].open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_rows = list(reader)
        if reader.fieldnames != list(expected_rows[0]) or len(actual_rows) != len(expected_rows):
            raise ValueError("E7 risk-coverage CSV schema/count differs")
    for index, (actual_row, expected_row) in enumerate(zip(actual_rows, expected_rows)):
        if actual_row.keys() != expected_row.keys():
            raise ValueError("E7 risk-coverage CSV row schema differs")
        for key, expected in expected_row.items():
            raw = actual_row[key]
            actual_value = None if raw == "" else int(raw) if type(expected) is int else float(raw)
            _equal(actual_value, expected, f"E7.curve[{index}].{key}")
    with Image.open(paths["e7_png"]) as image:
        image.verify()
    return {k: actual[k] for k in ("selected_metrics", "aurc", "temperature", "threshold")}


def _robustness(paths, root, bundle, mode, original_paths=None, original_root=None):
    _artifact_markers(paths["e8"].parent)
    cfg = load_e8_config(root / "configs/experiments/e8_robustness.yaml")
    if (cfg.input_size != 224 or cfg.model_selection_config != "configs/model_selection.yaml"
            or cfg.raw["model"].get("checkpoint_experiment") != "E4"
            or cfg.raw["model"].get("checkpoint_seed") != 42
            or cfg.conditions != ["clean", "gaussian_blur", "gaussian_noise", "occlusion"]):
        raise ValueError("E8 does not use the approved E4 input/model/conditions")
    actual = _json(paths["e8"])
    expected_fields = dict(experiment_id="E8_TASK04_ROBUSTNESS", split=mode, synthetic=mode == "synthetic",
                           official_evaluation=False, num_classes=30, n_samples=len(bundle["labels"]),
                           conditions=cfg.conditions, seed=cfg.reproducibility.seed,
                           model_experiment="E4" if mode == "validation" else "synthetic_dummy")
    for key, value in expected_fields.items():
        if key not in actual:
            raise Blocked(f"E8 metadata cannot verify {key}")
        _exact(actual[key], value, f"E8.{key}")
    for key in ("preprocessing", "corruption_config", "checkpoint_sha256", "split_csv", "image_ids", "checkpoint"):
        if key not in actual:
            raise Blocked(f"E8 metadata cannot verify {key}")
    _exact(actual["preprocessing"], predictor._load_yaml(root / "configs/data.yaml")["preprocessing"], "E8.preprocessing")
    _exact(actual["corruption_config"], json.loads(json.dumps({name: asdict(getattr(cfg, name)) for name in cfg.conditions if name != "clean"})), "E8.corruption_config")
    _equal(actual["checkpoint_sha256"], predictor.EXPECTED_CHECKPOINT_SHA256 if mode == "validation" else None, "E8.checkpoint_sha256")
    _equal(actual["image_ids"], bundle["image_ids"].tolist(), "E8.image_ids")
    original_paths = paths if original_paths is None else original_paths
    original_root = root if original_root is None else original_root
    def declared_path(value):
        # Metadata is a pathname claim, never a request to open that pathname.
        path = Path(value)
        if ".." in path.parts or any("official_test" in part.lower().replace("-", "_") for part in path.parts):
            raise PermissionError("Forbidden metadata path")
        return path if path.is_absolute() else original_root / path
    if mode == "validation":
        if declared_path(actual["checkpoint"]) != original_paths["checkpoint"]:
            raise ValueError("E8 checkpoint path differs from the approved E4 checkpoint")
    elif actual["checkpoint"] is not None:
        raise ValueError("Synthetic E8 must not claim a real checkpoint")
    if declared_path(actual["split_csv"]) != original_paths["csv"]:
        raise ValueError("E8 split CSV does not match the approved canonical CSV")
    clean = actual.get("clean_accuracy")
    if type(clean) not in (int, float) or not math.isfinite(clean) or not 0 <= clean <= 1:
        raise ValueError("Invalid E8 Clean accuracy")
    if mode == "validation":
        _equal(clean, float(np.mean(bundle["logits"].argmax(1) == bundle["labels"])), "E8.clean_accuracy")
    for name in cfg.conditions[1:]:
        accuracy = actual.get(name + "_accuracy")
        if type(accuracy) not in (int, float) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(f"Invalid E8 accuracy: {name}")
        _equal(actual.get(name + "_accuracy_drop"), float(clean - accuracy), f"E8.{name}_accuracy_drop")
    return {k: v for k, v in actual.items() if k == "clean_accuracy" or k.endswith("_accuracy") or k.endswith("_accuracy_drop")}


def _e8_provenance(paths, root, mode, git_root=None):
    if mode == "synthetic":
        return {"scope": "synthetic fixture; no historical model execution claimed"}
    actual = _json(paths["e8"])
    report = (root / E8_REPORT).read_text(encoding="utf-8")
    if predictor.sha256_file(paths["e8"]) != E8_HASH or actual.get("git_sha") != E8_COMMIT:
        raise ValueError("E8 artifact does not match the committed Validation acceptance evidence")
    if any(value not in report for value in (E8_HASH, E8_COMMIT, ROBUSTNESS.as_posix())):
        raise Blocked("E8 acceptance report does not bind the artifact hash and execution source")
    subprocess.run(["git", "-C", str(git_root or root), "cat-file", "-e", E8_COMMIT + "^{commit}"], check=True, capture_output=True)
    return {"source_git_sha": E8_COMMIT, "artifact_sha256": E8_HASH, "basis": "accepted report and immutable artifact hash; no inference rerun"}


def _completion(output):
    if os.path.lexists(output / ".incomplete"):
        raise Blocked(f"Incomplete preflight output: {output}")
    for name in (RUN_MARKER, ".complete", *REPORT_FILES):
        _safe_path(output / name, output)
    try:
        token = _read_once(output / RUN_MARKER, limit=64)["data"]
        completed = _parse_json(_read_once(output / ".complete", limit=65536)["data"], ".complete")
    except OSError as error:
        raise Blocked(f"Preflight output has no valid completion record: {output}") from error
    if (len(token) != 33 or token[-1:] != b"\n" or any(c not in b"0123456789abcdef" for c in token[:-1])
            or completed.get("run_token") != token.decode("ascii")
            or not all((output / name).is_file() for name in REPORT_FILES)):
        raise Blocked("Preflight output completion record is inconsistent")
    expected = completed.get("output_sha256")
    captured = {name: _read_once(output / name) for name in REPORT_FILES}
    if expected != {name: record["sha256"] for name, record in captured.items()}:
        raise Blocked("Preflight output artifacts changed or were only partially written")
    if os.path.lexists(output / ".incomplete"):
        raise Blocked("Preflight output was invalidated while reading")
    return {name: record["data"] for name, record in captured.items()}


def load_preflight_report(output_dir):
    output = _safe_path(output_dir, SOURCE_ROOT)
    buffers = _completion(output)
    reports = {name: _parse_json(raw, name) for name, raw in buffers.items()}
    summary, checks = reports[REPORT_FILES[0]], reports[REPORT_FILES[1]]["checks"]
    if not isinstance(checks, list) or not checks or any(not isinstance(row, dict) or row.get("status") not in {"PASS", "FAIL", "BLOCKED"} or type(row.get("required")) is not bool for row in checks):
        raise ValueError("Preflight checks have invalid status/schema")
    ids = [row.get("id") for row in checks]
    if any(not isinstance(name, str) for name in ids) or len(set(ids)) != len(ids):
        raise ValueError("Preflight checks have missing/duplicate IDs")
    mandatory = {row["id"] for row in checks if row["required"] is True}
    if not REQUIRED_CHECKS <= mandatory:
        raise ValueError("Preflight report omits or disables mandatory checks")
    states = {row["status"] for row in checks if row["required"]}
    expected = "FAIL" if "FAIL" in states else "BLOCKED" if "BLOCKED" in states else "PASS"
    if summary["overall_status"] != expected:
        raise ValueError("Preflight summary does not match its checks")
    counts = {state: sum(row["status"] == state for row in checks) for state in ("PASS", "FAIL", "BLOCKED")}
    recorded_counts = summary.get("check_counts")
    if (recorded_counts != counts or not isinstance(recorded_counts, dict)
            or any(type(value) is not int for value in recorded_counts.values())
            or summary.get("execution_status") != "complete"):
        raise ValueError("Preflight summary counts/completion are inconsistent")
    if (summary.get("phase") != "task05_phase_a_preflight" or summary.get("mode") not in {"synthetic", "validation"}
            or summary.get("synthetic") is not (summary["mode"] == "synthetic")):
        raise ValueError("Preflight report has invalid Phase A mode/schema")
    manifest = reports[REPORT_FILES[2]]
    for report in (summary, manifest):
        for key in ("official_test_used", "configuration_lock_approved"):
            if report.get(key) is not False:
                raise ValueError("Phase A reports cannot grant Official Test/Configuration Lock approval")
    if summary.get("official_evaluation_authorized") is not False or summary.get("inference_performed") is not False:
        raise ValueError("Phase A report has invalid execution/authorization flags")
    if expected == "PASS":
        binding = manifest.get("input_binding", {})
        frozen = next(row for row in checks if row["id"] == "frozen_reference_integrity")
        if (not isinstance(binding, dict) or binding.get("private_snapshot_verified") is not True
                or binding.get("capture_errors") != []
                or binding.get("strategy") != "private read-only per-file snapshot"
                or binding.get("hash_basis") != "same bytes copied and validated"
                or frozen.get("snapshot_content_bound_to_git") is not True):
            raise ValueError("PASS requires verified content-bound snapshot and frozen-reference evidence; legacy PASS needs a new audit")
    if os.path.lexists(output / ".incomplete"):
        raise Blocked("Preflight output was invalidated while parsing")
    return summary


@contextmanager
def _output_run(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Exclusive reservation even when another writer wins after preflight.
    directory_id = predictor._identity(output.stat())
    token = (uuid.uuid4().hex + "\n").encode("ascii")
    with predictor._exclusive_binary_file(output / RUN_MARKER) as handle:
        marker_id = predictor._identity(os.fstat(handle.fileno()))
        handle.write(token)
    def check_directory():
        _safe_path(output, output.parent)
        if predictor._identity(output.stat()) != directory_id:
            raise RuntimeError("Preflight output directory was replaced")
    published = []
    try:
        with tempfile.TemporaryDirectory(prefix="dogbreedid-preflight-") as temporary:
            staging = Path(temporary)
            yield staging
            for name in REPORT_FILES:
                check_directory()
                destination = output / name
                with (staging / name).open("rb") as reader, predictor._exclusive_binary_file(destination) as writer:
                    published.append((destination, predictor._identity(os.fstat(writer.fileno()))))
                    shutil.copyfileobj(reader, writer)
            hashes = {name: predictor.sha256_file(staging / name) for name in REPORT_FILES}
        check_directory()
        if {p.name for p in output.iterdir()} != {*REPORT_FILES, RUN_MARKER}:
            raise RuntimeError("Unexpected files appeared in the preflight output")
        if not all(predictor._owned_file(path, identity) for path, identity in published + [(output / RUN_MARKER, marker_id)]):
            raise RuntimeError("Preflight output files were replaced")
        with predictor._exclusive_binary_file(output / ".complete") as handle:
            handle.write(json.dumps(dict(run_token=token.decode("ascii"), output_sha256=hashes)).encode("utf-8"))
        _completion(output)
    except BaseException as error:
        try:
            check_directory()
            with predictor._exclusive_binary_file(output / ".incomplete") as handle:
                handle.write(b"INCOMPLETE PREFLIGHT: do not consume as a completed report.\n")
        except FileExistsError:
            pass
        except (OSError, RuntimeError) as marker_error:
            if hasattr(error, "add_note"):
                error.add_note(f"Failure marker unavailable: {marker_error}")
        raise  # Never unlink public files, including foreign replacements or markers.


def run_preflight(*, output_dir, mode="validation", repo_root=SOURCE_ROOT,
                  predictions_dir=None, e7_dir=None, e8_artifact=None):
    original_root, original_paths, output = _inputs(mode, repo_root, predictions_dir, e7_dir, e8_artifact, output_dir)
    with _output_run(output) as staging:
        checks = Checks()
        source = _git_state(CODE_ROOT, SOURCE_FILES)
        reference_names = (*REFERENCE_FILES, original_paths["csv"].relative_to(original_root).as_posix()) if mode == "validation" else REFERENCE_FILES
        reference_git_root = original_root if mode == "validation" else CODE_ROOT
        refs_state = _git_state(reference_git_root, reference_names)
        checks.audit("source_integrity", source)
        checks.audit("frozen_reference_integrity", refs_state)
        snapshot = InputSnapshot(original_root, original_paths, staging / "inputs", mode)
        paths, root = snapshot.paths, snapshot.root
        checks.run("input_snapshot_stable", snapshot.verify)
        frozen_row = next(row for row in checks.rows if row["id"] == "frozen_reference_integrity")
        try:
            snapshot.verify_references(refs_state["head"], reference_git_root, include_csv=mode == "validation")
            frozen_row["snapshot_content_bound_to_git"] = True
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            status = "BLOCKED" if isinstance(error, (FileNotFoundError, Blocked, subprocess.CalledProcessError)) else "FAIL"
            frozen_row["status"] = "FAIL" if frozen_row["status"] == "FAIL" else status
            frozen_row["detail"] = f"Frozen snapshot content: {type(error).__name__}: {error}"
            frozen_row["snapshot_content_bound_to_git"] = False
        def protocol():
            if mode == "synthetic":
                _fixture_declaration(root)
            return _protocol(root)
        checks.run("protocol_references", protocol, needs=("input_snapshot_stable",))
        checks.run("prediction_bundle", lambda: _bundle(paths, root, mode), needs=("protocol_references",))
        def checkpoint():
            expected = predictor.EXPECTED_CHECKPOINT_SHA256
            if mode == "synthetic":
                if "prediction_bundle" not in checks.values:
                    raise Blocked("Synthetic checksum needs fixture metadata")
                if _read_once(paths["checkpoint"], limit=1024)["data"] != b"synthetic checksum fixture, never model weights":
                    raise ValueError("Synthetic checkpoint must be the controlled fixture stub, never E4 weights")
                expected = checks.values["prediction_bundle"]["metadata"]["checkpoint_sha256"]
                if not isinstance(expected, str) or len(expected) != 64:
                    raise ValueError("Synthetic checksum must be a full SHA-256")
            return predictor.verify_checkpoint_hash(paths["checkpoint"], expected)
        checks.run("checkpoint_sha256", checkpoint, needs=("input_snapshot_stable",))
        bundle = checks.values.get("prediction_bundle")
        checks.run("e4_e6_metrics", lambda: _classification(paths, bundle), needs=("prediction_bundle", "checkpoint_sha256"))
        checks.run("e7_metrics", lambda: _selective(paths, bundle, mode), needs=("prediction_bundle",))
        checks.run("e8_artifact", lambda: _robustness(paths, root, bundle, mode, original_paths, original_root), needs=("prediction_bundle", "checkpoint_sha256"))
        checks.run("e8_execution_provenance", lambda: _e8_provenance(paths, root, mode, original_root), needs=("e8_artifact",))
        stable_row = next(row for row in checks.rows if row["id"] == "input_snapshot_stable")
        private_verified = False
        try:
            snapshot.verify()
            private_verified = True
            snapshot.verify_live_drift()
        except (OSError, ValueError) as error:
            if stable_row["status"] == "PASS":
                stable_row.update(status="FAIL", detail=f"{type(error).__name__}: {error}")
        # A commit or tracked working-tree change during the audit is not silently
        # attributed to the initial HEAD. Content binding itself comes from copies.
        for row, audit_root, names in ((checks.rows[0], CODE_ROOT, SOURCE_FILES),
                                      (frozen_row, reference_git_root, reference_names)):
            latest = _git_state(audit_root, names)
            if latest["head"] != row["evidence"]["head"] or latest["status"] == "FAIL":
                row.update(status="FAIL", detail="Git HEAD or tracked source/reference state changed during audit")
                row["final_git_evidence"] = latest
        timestamp = datetime.now(timezone.utc).isoformat()
        summary = dict(phase="task05_phase_a_preflight", mode=mode, synthetic=mode == "synthetic",
                       overall_status=checks.overall(), execution_status="complete", timestamp_utc=timestamp,
                       git_sha=source["head"], official_test_used=False, configuration_lock_approved=False,
                       official_evaluation_authorized=False, inference_performed=False,
                       status_scope="artifact snapshot preflight only; PASS never grants Official Test access",
                       check_counts={state: sum(row["status"] == state for row in checks.rows) for state in ("PASS", "FAIL", "BLOCKED")},
                       metrics={key: checks.values[key] for key in ("e4_e6_metrics", "e7_metrics", "e8_artifact") if key in checks.values})
        manifest = dict(timestamp_utc=timestamp, git_sha=source["head"], source=source, frozen_references=refs_state,
                        runtime=dict(python=platform.python_version(), pytorch=torch.__version__, torchvision=torchvision.__version__,
                                     numpy=np.__version__, cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available()),
                        inputs=snapshot.records, official_test_used=False, configuration_lock_approved=False,
                        comparison_tolerance=dict(relative=1e-9, absolute=1e-12, integer_counts="exact", protocol_parameters="exact"),
                        input_binding=dict(strategy="private read-only per-file snapshot", hash_basis="same bytes copied and validated",
                                           cross_file_atomic=False, original_drift_check="supplemental only",
                                           private_snapshot_verified=private_verified, capture_errors=snapshot.errors,
                                           reference_git_root=str(reference_git_root),
                                           synthetic_origin="declared fixture, synthetic IDs and bounded stub; not cryptographic origin attestation" if mode == "synthetic" else None),
                        provenance=checks.values.get("e8_execution_provenance"), output_dir=str(output))
        manifest["inference_producer_metadata"] = ({key: bundle["metadata"].get(key) for key in
            ("git_sha", "checkpoint_sha256", "runtime", "split", "synthetic")} if bundle is not None else None)
        for name, payload in zip(REPORT_FILES, (summary, {"checks": checks.rows}, manifest)):
            with (staging / name).open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "validation"), default="validation")
    parser.add_argument("--repo-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--e7-dir", type=Path)
    parser.add_argument("--e8-artifact", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = run_preflight(**vars(args))
        print(json.dumps({k: summary[k] for k in ("overall_status", "check_counts", "official_test_used", "configuration_lock_approved")}, indent=2))
        return {"PASS": 0, "FAIL": 1, "BLOCKED": 2}[summary["overall_status"]]
    except Exception as error:
        print(f"preflight: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
