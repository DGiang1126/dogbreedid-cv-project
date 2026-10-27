"""Phase B1: synthetic or canonical Validation execution; no Official Test route.

Validation requires committed, clean audited source/references. Synthetic creates
its own images/model in private temporary storage, never accepts dataset inputs.
Untracked B1 files are permitted only for synthetic development, explicitly
BLOCKED source provenance (never a clean committed execution or experiment PASS).

Bundles are complete only through load_execution_bundle(), not the presence of
an NPZ. The legacy .inference_run sentinel also makes Task 04 readers reject an
in-flight B1 NPZ; their token-only completion API intentionally rejects B1's new
JSON completion protocol even after publication. Use the B1 consumer instead.
Publication uses same-volume hard links with atomic no-clobber creation;
on unsupported filesystems it fails closed. No public path is unlinked. Private
staging assumes a trusted same-user boundary; snapshots are per-file, not an
atomic filesystem/bytecode attestation. Hash markers are not digital signatures.
Consumers require this repository's immutable Git objects at the recorded commit;
missing objects or legacy evidence fail closed, without reopening dataset inputs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import tempfile
import uuid

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision

from scripts import run_final_evaluation_preflight as preflight
from scripts import run_final_robustness as robustness
from scripts import run_final_selective as selective
from src.calibration.calibration_metrics import plot_reliability_diagram
from src.data.dataset import DogBreedDataset
from src.data.transforms import build_transforms
from src.inference import predictor
from src.robustness.config import load_e8_config

NEW_SOURCE = ("scripts/run_final_evaluation.py", "tests/test_final_evaluation_execution.py")
SOURCE_FILES = (*preflight.SOURCE_FILES, "src/models/mobilenet_v2.py", "src/utils/config.py",
                "src/calibration/__init__.py", "src/inference/__init__.py", "src/robustness/__init__.py", *NEW_SOURCE)
REFERENCES = preflight.REFERENCE_FILES
FORMAT = "dogbreedid.execution.b1.v1"
RUN_MARKER = ".inference_run"
PAYLOADS = ("clean_predictions.npz", "e4_e6_metrics.json", "metadata.json",
            "selective_metrics.json", "risk_coverage.csv", "risk_coverage_curve.png",
            "e8_metrics.json", "reliability_diagram.png", "split.csv",
            "class_mapping.json", "protocol.json", "manifest.json")
COMPONENTS = ("E4", "E6", "E7", "E8", "data_contract", "weights_unchanged", "clean_consistency")
IMAGES = Path("data/raw/stanford_dogs/Images")
# Validation constraints only. Execution parameters still come from frozen YAML.
APPROVED_CORRUPTIONS = {
    "gaussian_blur": dict(enabled=True, kernel_size=7, sigma=1.5),
    "gaussian_noise": dict(enabled=True, mean=0.0, std=0.05, clamp_range=[0.0, 1.0]),
    "occlusion": dict(enabled=True, shape="square", side_fraction=0.30, fill_rgb=[0.5, 0.5, 0.5], position_policy="deterministic_per_image"),
}
COMPARISON_TOLERANCE = dict(metrics_relative=1e-9, metrics_absolute=1e-12, classification_vs_counts_relative=1e-7,
                            classification_vs_counts_absolute=1e-8, counts="exact", protocol_parameters="exact")


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _write(path, data):
    with path.open("xb") as handle:
        handle.write(data)


def _json(path, value):
    _write(path, _json_bytes(value))


def _safe_path(value):
    path = Path(value)
    if str(value).replace("\\", "/").startswith("//"):
        raise PermissionError("Network/device path aliases are not supported")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    for part in path.parts:
        if part == path.anchor:
            continue
        if (":" in part or part.rstrip(" .") != part
                or part.split(".")[0].upper() in reserved):
            raise PermissionError("Device, stream and ambiguous trailing-character path aliases are forbidden")
    return preflight._safe_path(path, SOURCE_ROOT)


def _isolated(path):
    temporary = preflight._safe_path(tempfile.gettempdir(), SOURCE_ROOT)
    if (path == temporary or not path.is_relative_to(temporary)
            or path.is_relative_to(SOURCE_ROOT) or SOURCE_ROOT.is_relative_to(path)):
        raise PermissionError("Synthetic output must be an isolated child of OS temp, disjoint from checkout")


def _output_path(split, value):
    if split not in {"synthetic", "validation"}:
        raise PermissionError("B1 supports only synthetic or validation; Official Test is unavailable")
    output = _safe_path(value)
    if split == "synthetic":
        _isolated(output)
    else:
        allowed = SOURCE_ROOT / "outputs/final_evaluation_b1/validation"
        if output == allowed or not output.is_relative_to(allowed):
            raise PermissionError("Validation output must be a fresh run beneath outputs/final_evaluation_b1/validation/")
    if os.path.lexists(output):
        raise FileExistsError(f"Output directory already exists, even if empty: {output}")
    for parent in output.parents:
        if any(os.path.lexists(parent / name) for name in (RUN_MARKER, ".incomplete", ".complete")):
            raise PermissionError("Cannot nest a run inside an existing bundle")
        if parent == SOURCE_ROOT or parent == Path(tempfile.gettempdir()):
            break
    return output


def _source_audit(split):
    for name in SOURCE_FILES:
        _safe_path(SOURCE_ROOT / name)
    evidence = preflight._git_state(SOURCE_ROOT, SOURCE_FILES)
    # Development smoke is honest about new files; tracked dirt always fails.
    development = (split == "synthetic" and evidence["status"] == "BLOCKED"
                   and set(evidence.get("untracked", [])) <= set(NEW_SOURCE)
                   and bool(evidence.get("head")))
    if evidence["status"] != "PASS" and not development:
        raise preflight.Blocked(f"Source integrity required before execution: {evidence}")
    tracked = [name for name in SOURCE_FILES if name not in evidence["untracked"]]
    blobs = _git_blobs(evidence["head"], tracked)
    content = {}
    for name in SOURCE_FILES:
        try:
            captured = preflight._read_once(SOURCE_ROOT / name)
        except OSError as error:
            raise preflight.Blocked(f"Source bytes unavailable: {name}") from error
        if name in blobs:
            content[name] = _bind_bytes(captured["data"], blobs[name], name)
        else:
            content[name] = dict(sha256=captured["sha256"], size_bytes=captured["size_bytes"],
                                 git_blob_sha256=None, match_mode="untracked")
    evidence["content"] = content
    return evidence


def _git_blobs(commit, names):
    """Read only enumerated immutable blobs, never a manifest-supplied pathname.

    One batch avoids spawning a Git process for every file. The returned buffers
    are used both for hashing and parsing; there is no mutable checkout reread.
    """
    if not _hex_digest(commit, 40):
        raise preflight.Blocked("A full Git commit is required for provenance")
    names = tuple(names)
    allowed = {*SOURCE_FILES, *REFERENCES, "data/splits/validation.csv"}
    if len(set(names)) != len(names) or not set(names) <= allowed:
        raise ValueError("Only enumerated source/canonical reference blobs may be read")
    specs = [f"{commit}:{name}" for name in names]
    try:
        result = subprocess.run(["git", "--no-replace-objects", "-C", str(SOURCE_ROOT), "cat-file", "--batch"],
                                input=("\n".join(specs) + "\n").encode("utf-8"),
                                check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise preflight.Blocked("Canonical Git blobs are unavailable") from error
    reader, blobs = io.BytesIO(result.stdout), {}
    for name in names:
        header = reader.readline().split()
        if len(header) != 3 or header[1] != b"blob" or not header[2].isdigit():
            raise preflight.Blocked(f"Missing or invalid Git blob at {commit}: {name}")
        size = int(header[2])
        raw = reader.read(size)
        if len(raw) != size or reader.read(1) != b"\n":
            raise preflight.Blocked(f"Truncated Git blob: {name}")
        blobs[name] = raw
    if reader.read():
        raise preflight.Blocked("Unexpected Git batch response")
    return blobs


def _hex_digest(value, length=64):
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _newline_variants(expected):
    lf = expected.replace(b"\r\n", b"\n")
    return (("exact", expected), ("newline_variant_lf", lf),
            ("newline_variant_crlf", lf.replace(b"\n", b"\r\n")))


def _bind_bytes(actual, expected, label):
    for mode, candidate in _newline_variants(expected):
        if actual == candidate:
            return dict(sha256=_hash(actual), size_bytes=len(actual), git_blob_sha256=_hash(expected), match_mode=mode)
    raise ValueError(f"Content differs from canonical Git blob (including code/metadata): {label}")


def _check_binding(record, expected, label):
    if not isinstance(record, dict) or set(record) != {"sha256", "size_bytes", "git_blob_sha256", "match_mode"}:
        raise ValueError(f"Missing source content binding: {label}")
    for _, candidate in _newline_variants(expected):
        if record == _bind_bytes(candidate, expected, label):
            if type(record["size_bytes"]) is not int:
                break
            return
    raise ValueError(f"Source hash/newline binding differs from Git blob: {label}")


class ExecutionInputs:
    """All validators and model/image loaders consume private copied bytes."""

    def __init__(self, private, split, source):
        self.root = private
        self.records = {}
        self.paths = {}
        self.split = split
        self.source = source
        names = (*REFERENCES, "data/splits/validation.csv") if split == "validation" else REFERENCES
        for name in names:
            _safe_path(SOURCE_ROOT / name)
        state = preflight._git_state(SOURCE_ROOT, names)
        if state["status"] != "PASS" or state["head"] != source["head"]:
            raise preflight.Blocked(f"Frozen reference integrity required: {state}")
        self.reference_state = state
        reference_blobs = _git_blobs(source["head"], names)
        for name in names:
            path = self.capture(SOURCE_ROOT / name, Path(name))
            _bind_bytes(path.read_bytes(), reference_blobs[name], name)
        # These hashes come from the same captured source bytes checked by the
        # audit, not a later mutable-path read. Untracked smoke stays BLOCKED.
        self.source_hashes = {name: source["content"][name]["sha256"] for name in SOURCE_FILES}
        self.refs = preflight._protocol(private)
        self.cfg = load_e8_config(private / "configs/experiments/e8_robustness.yaml")
        self.data = predictor._load_yaml(private / "configs/data.yaml")
        # Reuse the approved shared preprocessing validation before loading E4.
        robustness._build_pre_norm_transform(self.data)
        if (self.cfg.conditions != ["clean", "gaussian_blur", "gaussian_noise", "occlusion"]
                or self.cfg.input_size != 224 or self.cfg.reproducibility.seed != 42):
            raise ValueError("E8 conditions/input/seed differ from the frozen protocol")
        preflight._exact(json.loads(json.dumps({name: asdict(getattr(self.cfg, name)) for name in self.cfg.conditions[1:]})),
                         APPROVED_CORRUPTIONS, "Approved E8 severity")
        self.mapping, _, _ = selective._mapping(private)
        if split == "synthetic":
            self.csv, self.images = self._synthetic()
            self.checkpoint = None
            self.checkpoint_hash = None
        else:
            self.csv = private / "data/splits/validation.csv"
            self.rows = selective._csv_rows(self.csv, self.mapping, split)
            if len(self.rows) != 300 or len({label for _, label in self.rows}) != 30:
                raise ValueError("Canonical Validation must contain 300 images and 30 classes")
            image_root = _safe_path(SOURCE_ROOT / IMAGES)
            if image_root != SOURCE_ROOT / IMAGES:
                raise PermissionError("Only the canonical image root is allowed")
            self.images = private / "images"
            for image_id, _ in self.rows:
                if not selective._safe_id(image_id):
                    raise PermissionError("Unsafe canonical image ID")
                original = _safe_path(image_root / image_id)
                if not original.is_relative_to(image_root):
                    raise PermissionError("Image path escaped canonical image root")
                self.capture(original, Path("images") / image_id)
            self.checkpoint = self.capture(SOURCE_ROOT / predictor.DEFAULT_CHECKPOINT, predictor.DEFAULT_CHECKPOINT)
            self.checkpoint_hash = predictor.verify_checkpoint_hash(self.checkpoint, predictor.EXPECTED_CHECKPOINT_SHA256)
        self.rows = selective._csv_rows(self.csv, self.mapping, split)
        self.verify()

    def capture(self, original, relative):
        original = _safe_path(original)
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as writer:
            record = preflight._read_once(original, destination=writer)
        self.records[str(original)] = dict(sha256=record["sha256"], size_bytes=record["size_bytes"],
                                          snapshot_path=relative.as_posix(), state="captured")
        destination.chmod(stat.S_IREAD)
        self.paths[destination] = record["sha256"]
        return destination

    def _synthetic(self):
        images = self.root / "images"
        csv_path = self.root / "data/splits/synthetic.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        generator = np.random.default_rng(self.cfg.reproducibility.seed)
        with csv_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("relative_path", "class_id", "breed_name", "source_split"))
            writer.writeheader()
            for i in range(8):
                image_id = f"synthetic/generated_{i:03d}.png"
                path = images / image_id
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(generator.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(path)
                writer.writerow(dict(relative_path=image_id, class_id=i, breed_name=self.mapping[i], source_split="synthetic"))
                self._generated(path)
        self._generated(csv_path)
        return csv_path, images

    def _generated(self, path):
        record = preflight._read_once(path)
        self.paths[path] = record["sha256"]
        self.records["generated:" + path.relative_to(self.root).as_posix()] = dict(
            sha256=record["sha256"], size_bytes=record["size_bytes"], state="generated", snapshot_path=path.relative_to(self.root).as_posix())
        path.chmod(stat.S_IREAD)

    def verify(self):
        for path, digest in self.paths.items():
            if preflight._read_once(path)["sha256"] != digest:
                raise ValueError(f"Private execution input changed: {path}")

    def verify_source(self):
        latest = _source_audit(self.split)
        if latest != self.source:
            raise preflight.Blocked("Source Git state changed during execution")
        if preflight._git_state(SOURCE_ROOT, self.reference_state["audited_files"]) != self.reference_state:
            raise preflight.Blocked("Reference Git state changed during execution")
        for name, digest in self.source_hashes.items():
            if preflight._read_once(SOURCE_ROOT / name)["sha256"] != digest:
                raise preflight.Blocked("Observed source changed during execution: " + name)


def _validate_arrays(logits, labels, ids, rows):
    if (logits.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or labels.dtype != np.dtype("int64") or ids.dtype.kind != "U"):
        raise ValueError("NPZ dtypes must be float32/float64, int64, Unicode")
    predictor.validate_raw_predictions(logits, labels, ids)
    if ids.tolist() != [row[0] for row in rows] or labels.tolist() != [row[1] for row in rows]:
        raise ValueError("Image IDs or labels differ from split CSV order")


def _selective(logits, labels, temperature, threshold):
    calibrated = selective.calibrated_predictions(logits, temperature)
    correct = calibrated["predictions"] == labels
    mask = selective.accept_mask(calibrated["confidences"], threshold)
    return dict(temperature=temperature, threshold=threshold, accept_operator=">=", threshold_reselected=False,
                n_total=len(labels), selected_metrics=selective.selective_metrics(correct, mask),
                aurc=selective.compute_aurc(calibrated["confidences"], correct),
                aurc_convention="grouped right-step over confidence tie groups"), selective.risk_coverage_curve(calibrated["confidences"], correct)


def _consistency(e4, e8, logits, labels, ids):
    n = len(labels)
    expected_clean = float(np.mean(logits.argmax(1) == labels))
    preflight._equal(e8["clean_accuracy"], expected_clean, "E8 vs E4 raw Clean Accuracy")
    if not math.isclose(e4["classification"]["top1_accuracy"], expected_clean, rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError("E4 classification accuracy differs from raw logits beyond float32 tolerance")
    if e8["image_ids"] != ids.tolist() or e8["labels"] != labels.tolist() or e8["n_samples"] != n:
        raise ValueError("E8 IDs/labels/count differ from E4")
    if e8["conditions"] != ["clean", "gaussian_blur", "gaussian_noise", "occlusion"]:
        raise ValueError("E8 conditions differ")
    for name in e8["conditions"]:
        accuracy = e8["clean_accuracy" if name == "clean" else name + "_accuracy"]
        if type(accuracy) not in (int, float) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("E8 accuracy must be finite in [0,1]")
        preflight._equal(accuracy * n, float(round(accuracy * n)), "E8 integer correct count")
        preflight._exact(e8["correct_counts"][name], round(accuracy * n), "E8 correct count")
        if name != "clean":
            preflight._equal(e8[name + "_accuracy_drop"], expected_clean - accuracy, "E8 signed drop")


@contextmanager
def _output_run(output):
    _safe_path(output.parent).mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Exclusive reservation BEFORE snapshots/checkpoint/model setup.
    identity = predictor._identity(output.stat())
    token = (uuid.uuid4().hex + "\n").encode("ascii")
    _write(output / RUN_MARKER, token)

    def check_directory():
        _safe_path(output)
        if predictor._identity(output.stat()) != identity:
            raise RuntimeError("Reserved output directory was replaced")

    try:
        # Same filesystem is required for no-replace hardlink publication.
        with tempfile.TemporaryDirectory(prefix=".dogbreedid-b1-private-", dir=output.parent) as temporary:
            staging = Path(temporary)
            yield staging
            payload = staging / "payload"
            hashes = {name: preflight._read_once(payload / name)["sha256"] for name in PAYLOADS}
            for name in PAYLOADS:
                check_directory()
                os.link(payload / name, output / name)  # Atomic EEXIST; never follows/replaces destination.
            check_directory()
            if {path.name for path in output.iterdir()} != {*PAYLOADS, RUN_MARKER}:
                raise RuntimeError("Unexpected files appeared during publication")
        check_directory()  # Private links gone; public input readers require nlink=1.
        _write(output / ".complete", _json_bytes(dict(format=FORMAT, run_token=token.decode("ascii"), output_sha256=hashes)))
        load_execution_bundle(output)
    except BaseException as error:
        try:
            check_directory()
            _write(output / ".incomplete", b"INCOMPLETE B1 EXECUTION: never consume this bundle.\n")
        except FileExistsError:
            pass
        except (OSError, ValueError, RuntimeError) as marker_error:
            if hasattr(error, "add_note"):
                error.add_note(f"Failure marker unavailable: {marker_error}")
        raise  # Never unlink public files, including replaced files or foreign markers.


def _verified_buffers(output):
    if os.path.lexists(output / ".incomplete"):
        raise preflight.Blocked("Incomplete B1 execution bundle")
    token = preflight._read_once(output / RUN_MARKER, limit=64)["data"]
    completed = preflight._parse_json(preflight._read_once(output / ".complete", limit=65536)["data"], ".complete")
    if (completed.get("format") != FORMAT or len(token) != 33 or token[-1:] != b"\n"
            or any(c not in b"0123456789abcdef" for c in token[:-1])
            or completed.get("run_token") != token.decode("ascii")):
        raise ValueError("Invalid B1 completion token")
    records = {name: preflight._read_once(output / name) for name in PAYLOADS}
    if completed.get("output_sha256") != {name: value["sha256"] for name, value in records.items()}:
        raise ValueError("B1 bundle hash mismatch")
    if {path.name for path in output.iterdir()} != {*PAYLOADS, RUN_MARKER, ".complete"}:
        raise ValueError("Unexpected or incomplete B1 bundle files")
    return {name: record["data"] for name, record in records.items()}


def _evidence(value, names, commit, *, source=False, split=None):
    required = {"head", "status", "modified", "untracked", "audited_files"} | ({"content"} if source else set())
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Missing/invalid source or frozen-reference evidence schema")
    audited, untracked = value["audited_files"], value["untracked"]
    if (value["head"] != commit or value["modified"] != []
            or not isinstance(audited, list) or any(not isinstance(v, str) for v in audited)
            or len(audited) != len(names) or set(audited) != set(names)
            or not isinstance(untracked, list) or any(not isinstance(v, str) for v in untracked)
            or len(untracked) != len(set(untracked))):
        raise ValueError("Inconsistent or dirty source/frozen-reference evidence")
    development = (source and split == "synthetic" and value["status"] == "BLOCKED"
                   and bool(untracked) and set(untracked) <= set(NEW_SOURCE))
    if not development and (value["status"] != "PASS" or untracked):
        raise ValueError("Unresolved source or frozen-reference evidence cannot be PASS")


def _input_record_matches(record, raw, label, *, newline=False):
    candidates = [candidate for _, candidate in _newline_variants(raw)] if newline else [raw]
    if not any(record["sha256"] == _hash(candidate) and record["size_bytes"] == len(candidate) for candidate in candidates):
        raise ValueError(f"Snapshot hash/size differs from verified content: {label}")


def _provenance(manifest):
    """Fail closed against immutable Git, independent of rehashed bundle claims.

    Records' original path strings are labels only. Git reads use this checkout
    and compiled path allowlists; a package cannot redirect reads to real data.
    """
    essential = {"git_sha", "source", "frozen_references", "source_sha256", "inputs", "runtime", "timestamp_utc",
                 "command", "input_binding", "comparison_tolerance", "output_sha256", "components"}
    if not essential <= manifest.keys() or not _hex_digest(manifest["git_sha"], 40):
        raise ValueError("Missing essential execution provenance")
    split, commit = manifest["split"], manifest["git_sha"]
    refs = (*REFERENCES, "data/splits/validation.csv") if split == "validation" else REFERENCES
    _evidence(manifest["source"], SOURCE_FILES, commit, source=True, split=split)
    _evidence(manifest["frozen_references"], refs, commit)
    source = manifest["source"]
    hashes, content = manifest["source_sha256"], source["content"]
    if (not isinstance(hashes, dict) or set(hashes) != set(SOURCE_FILES)
            or not isinstance(content, dict) or set(content) != set(SOURCE_FILES)
            or any(not _hex_digest(value) for value in hashes.values())):
        raise ValueError("Missing/invalid source hashes or content bindings")
    tracked = [name for name in SOURCE_FILES if name not in source["untracked"]]
    blobs = _git_blobs(commit, (*tracked, *refs))
    for name in SOURCE_FILES:
        record = content[name]
        if not isinstance(record, dict) or record.get("sha256") != hashes[name]:
            raise ValueError(f"Source hash differs from audited source bytes: {name}")
        if name in blobs:
            _check_binding(record, blobs[name], name)
        elif (set(record) != {"sha256", "size_bytes", "git_blob_sha256", "match_mode"}
              or record["match_mode"] != "untracked" or record["git_blob_sha256"] is not None
              or type(record["size_bytes"]) is not int or record["size_bytes"] < 1):
            raise ValueError("Invalid untracked synthetic development evidence")
    binding = manifest["input_binding"]
    expected_binding = dict(strategy="private per-file copies; same captured bytes hashed and consumed",
                            private_snapshot_verified=True, cross_file_atomic=False,
                            source_scope="Git-bound captured source bytes; no bytecode attestation",
                            source_content_bound_to_git=source["status"] == "PASS",
                            frozen_snapshot_content_bound_to_git=True)
    preflight._exact(binding, expected_binding, "Input/source binding")
    preflight._exact(manifest["comparison_tolerance"], COMPARISON_TOLERANCE, "Recorded comparison tolerance")
    inputs, records = manifest["inputs"], {}
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("Missing input snapshot records")
    for origin, record in inputs.items():
        if (not isinstance(origin, str) or not origin or not isinstance(record, dict)
                or set(record) != {"sha256", "size_bytes", "snapshot_path", "state"}
                or not _hex_digest(record["sha256"]) or type(record["size_bytes"]) is not int or record["size_bytes"] < 1
                or not selective._safe_id(record["snapshot_path"]) or record["state"] not in {"captured", "generated"}
                or record["snapshot_path"] in records):
            raise ValueError("Invalid/duplicate input snapshot record")
        if (record["state"] == "generated") != origin.startswith("generated:"):
            raise ValueError("Input origin/state is inconsistent")
        if record["state"] == "generated" and origin != "generated:" + record["snapshot_path"]:
            raise ValueError("Invalid generated fixture origin")
        records[record["snapshot_path"]] = record
    for name in refs:
        if name not in records or records[name]["state"] != "captured":
            raise ValueError(f"Missing canonical snapshot record: {name}")
        _input_record_matches(records[name], blobs[name], name, newline=True)
    runtime = manifest["runtime"]
    if (not isinstance(runtime, dict) or set(runtime) != {"python", "pytorch", "torchvision", "numpy", "cuda_build", "cuda_available",
                                                        "device", "batch_size", "num_workers", "torch_num_threads"}
            or any(not isinstance(runtime[key], str) or not runtime[key] for key in ("python", "pytorch", "torchvision", "numpy", "device"))
            or type(runtime["cuda_available"]) is not bool
            or runtime["cuda_build"] is not None and not isinstance(runtime["cuda_build"], str)
            or any(type(runtime[key]) is not int or runtime[key] < 1 for key in ("batch_size", "torch_num_threads"))
            or type(runtime["num_workers"]) is not int or runtime["num_workers"] != 0):
        raise ValueError("Missing/invalid runtime evidence")
    try:
        timestamp = datetime.fromisoformat(manifest["timestamp_utc"])
        if timestamp.tzinfo is None:
            raise ValueError("Timestamp must include timezone")
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid execution timestamp") from error
    command = manifest["command"]
    if not ((isinstance(command, list) and command and all(isinstance(arg, str) and arg for arg in command))
            or (isinstance(command, dict) and set(command) == {"api", "split", "output_dir", "batch_size", "device"}
                and command["api"] == "run_final_evaluation" and command["split"] == split
                and type(command["batch_size"]) is int and command["batch_size"] == runtime["batch_size"]
                and isinstance(command["output_dir"], str) and command["output_dir"]
                and isinstance(command["device"], str)
                and command["device"] in {"cpu", "cuda", "auto"})):
        raise ValueError("Missing/invalid execution command")
    return blobs, records


def _data_records(records, rows, split, buffers):
    csv_name = f"data/splits/{split}.csv"
    required = {*REFERENCES, csv_name, *("images/" + image_id for image_id, _ in rows)}
    if split == "validation":
        checkpoint = predictor.DEFAULT_CHECKPOINT.as_posix()
        required.add(checkpoint)
        if checkpoint not in records or records[checkpoint]["sha256"] != predictor.EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("Missing/wrong captured E4 checkpoint hash")
    if set(records) != required:
        raise ValueError("Snapshot records do not cover exactly the canonical inputs/images")
    for name, record in records.items():
        generated = split == "synthetic" and (name == csv_name or name.startswith("images/"))
        if record["state"] != ("generated" if generated else "captured"):
            raise ValueError("Input record state disagrees with execution mode")
    _input_record_matches(records[csv_name], buffers["split.csv"], "published split CSV")
    _input_record_matches(records["data/metadata/class_mapping.json"], buffers["class_mapping.json"], "published mapping")


def _validate_bundle(buffers):
    parsed = {name: preflight._parse_json(raw, name) for name, raw in buffers.items() if name.endswith(".json")}
    if any(not isinstance(value, dict) for value in parsed.values()):
        raise ValueError("Bundle JSON documents must have object roots")
    manifest, metadata, protocol = (parsed[name] for name in ("manifest.json", "metadata.json", "protocol.json"))
    if manifest.get("output_sha256") != {name: _hash(raw) for name, raw in buffers.items() if name != "manifest.json"}:
        raise ValueError("Manifest payload hashes differ from verified bytes")
    if manifest.get("format") != FORMAT or manifest.get("execution_status") != "complete":
        raise ValueError("B1 manifest is not complete")
    split = manifest.get("split")
    if split not in {"synthetic", "validation"} or manifest.get("synthetic") is not (split == "synthetic"):
        raise ValueError("Invalid B1 split/synthetic declaration")
    preflight._exact(manifest.get("scientific_status"), "synthetic_fixture_only" if split == "synthetic" else "validation_only", "B1 scientific scope")
    for doc in (manifest, metadata):
        for name in ("official_test_used", "official_final_evaluation", "configuration_lock_approved", "official_evaluation_authorized"):
            if doc.get(name) is not False:
                raise ValueError("B1 cannot grant Official Evaluation authorization")
    if manifest.get("components") != dict.fromkeys(COMPONENTS, "PASS"):
        raise ValueError("A required B1 component is not PASS")
    blobs, records = _provenance(manifest)
    if metadata.get("git_sha") != manifest["git_sha"]:
        raise ValueError("Inconsistent execution source commit")
    preflight._exact(protocol["temperature"], predictor.EXPECTED_TEMPERATURE, "E6 frozen Temperature")
    preflight._exact(protocol["threshold"], preflight.THRESHOLD, "E7 frozen threshold")
    preflight._exact(protocol["corruption_config"], APPROVED_CORRUPTIONS, "Frozen E8 severity")
    preflight._exact(protocol["seed"], 42, "Frozen E8 seed")
    robustness._build_pre_norm_transform(dict(preprocessing=protocol["preprocessing"]))
    preflight._exact(metadata["preprocessing"], protocol["preprocessing"], "Metadata preprocessing")
    with np.load(io.BytesIO(buffers["clean_predictions.npz"]), allow_pickle=False) as data:
        if set(data.files) != {"logits", "labels", "image_ids"}:
            raise ValueError("Unexpected NPZ schema")
        logits, labels, ids = (data[name] for name in ("logits", "labels", "image_ids"))
    # Canonical references are parsed from the same immutable Git buffers that
    # were hashed above. Public CSV/mapping bytes must agree with those buffers.
    with tempfile.TemporaryDirectory(prefix="dogbreedid-b1-consumer-") as temp:
        root = Path(temp)
        for name in REFERENCES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            _write(path, blobs[name])
        _bind_bytes(buffers["class_mapping.json"], blobs["data/metadata/class_mapping.json"], "canonical mapping")
        csv_raw = buffers["split.csv"]
        if split == "validation":
            csv_raw = blobs["data/splits/validation.csv"]
            _bind_bytes(buffers["split.csv"], csv_raw, "canonical Validation CSV")
        _write(root / "split.csv", csv_raw)
        mapping, _, _ = selective._mapping(root)
        rows = selective._csv_rows(root / "split.csv", mapping, split)
        refs = preflight._protocol(root)
        cfg = load_e8_config(root / "configs/experiments/e8_robustness.yaml")
        canonical_protocol = dict(temperature=refs["temperature"], threshold=refs["threshold"],
                                  preprocessing=predictor._load_yaml(root / "configs/data.yaml")["preprocessing"],
                                  seed=cfg.reproducibility.seed,
                                  corruption_config={name: asdict(getattr(cfg, name)) for name in cfg.conditions[1:]})
        preflight._exact(protocol, json.loads(json.dumps(canonical_protocol)), "Protocol vs frozen Git artifacts")
    _data_records(records, rows, split, buffers)
    if split == "validation" and (len(rows) != 300 or len({v for _, v in rows}) != 30):
        raise ValueError("Invalid Validation counts")
    _validate_arrays(logits, labels, ids, rows)
    if split == "synthetic" and any(not value.startswith("synthetic/") for value in ids.tolist()):
        raise ValueError("Synthetic Image IDs must use the controlled synthetic/ prefix")
    if metadata.get("num_samples") != len(rows) or metadata.get("split") != split or metadata.get("synthetic") is not (split == "synthetic"):
        raise ValueError("Metadata count/split differs")
    if metadata.get("raw_predictions_sha256") != _hash(buffers["clean_predictions.npz"]):
        raise ValueError("Metadata does not bind the verified Clean logits")
    for key, name in (("csv_sha256", "split.csv"), ("class_mapping_sha256", "class_mapping.json")):
        if metadata.get(key) != _hash(buffers[name]):
            raise ValueError("Metadata data-contract hash differs")
    e4 = predictor.compute_task1_metrics(logits, labels, temperature=protocol["temperature"], class_names=mapping)
    preflight._exact(parsed["e4_e6_metrics.json"]["calibration"]["temperature"], protocol["temperature"], "E6 metrics temperature")
    preflight._equal(parsed["e4_e6_metrics.json"], e4, "E4/E6 metrics")
    e7, curve = _selective(logits, labels, protocol["temperature"], protocol["threshold"])
    preflight._exact(parsed["selective_metrics.json"]["temperature"], protocol["temperature"], "E7 temperature")
    preflight._exact(parsed["selective_metrics.json"]["threshold"], protocol["threshold"], "E7 threshold")
    preflight._equal(parsed["selective_metrics.json"], e7, "E7 metrics")
    if buffers["risk_coverage.csv"] != selective._curve_csv(curve).encode("utf-8"):
        raise ValueError("E7 curve differs from Clean logits")
    e8 = parsed["e8_metrics.json"]
    preflight._exact(e8["corruption_config"], protocol["corruption_config"], "E8 configuration")
    preflight._exact(e8["preprocessing"], protocol["preprocessing"], "E8 preprocessing")
    _consistency(e4, e8, logits, labels, ids)
    expected_hash = predictor.EXPECTED_CHECKPOINT_SHA256 if split == "validation" else None
    if e8.get("checkpoint_sha256") != expected_hash or metadata.get("checkpoint_sha256") != expected_hash:
        raise ValueError("E4/E8 checkpoint provenance differs")
    preflight._exact(manifest.get("model"), dict(experiment="synthetic_dummy" if split == "synthetic" else "E4",
        architecture="dummy_pool_linear" if split == "synthetic" else "mobilenet_v2", seed=protocol["seed"], num_classes=30,
        checkpoint_path=None if split == "synthetic" else predictor.DEFAULT_CHECKPOINT.as_posix(), checkpoint_sha256=expected_hash), "Model protocol")
    if metadata.get("model_weights_unchanged") is not True:
        raise ValueError("Model weights changed")
    for name in ("risk_coverage_curve.png", "reliability_diagram.png"):
        with Image.open(io.BytesIO(buffers[name])) as image:
            image.verify()
    return dict(manifest=manifest, metadata=metadata, metrics=e4, selective=e7, robustness=e8,
                logits=logits, labels=labels, image_ids=ids)


def load_execution_bundle(output_dir):
    """Read one set of verified buffers; completion never grants authorization."""
    output = _safe_path(output_dir)
    buffers = _verified_buffers(output)
    result = _validate_bundle(buffers)
    if os.path.lexists(output / ".incomplete"):
        raise preflight.Blocked("B1 bundle invalidated while being consumed")
    return result


def _execute(inputs, payload, split, batch_size, device):
    if split == "synthetic":
        model = robustness._make_synthetic_model(inputs.cfg.reproducibility.seed).to(device)
    else:
        model, loaded_hash = predictor.load_e4_model(inputs.root, inputs.checkpoint, device)
        if loaded_hash != inputs.checkpoint_hash:
            raise ValueError("Checkpoint hash changed during model load")
    model.eval()
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    dataset = DogBreedDataset(inputs.csv, inputs.images, build_transforms(inputs.data, "validation"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits, labels, ids = predictor.collect_predictions(model, loader, device=device)
    _validate_arrays(logits, labels, ids, inputs.rows)
    predictor.save_raw_predictions(payload / "clean_predictions.npz", logits, labels, ids, exclusive=True)
    # E7 receives the same immutable arrays read back from the staged NPZ.
    verified = preflight._read_once(payload / "clean_predictions.npz")
    with np.load(io.BytesIO(verified["data"]), allow_pickle=False) as data:
        if set(data.files) != {"logits", "labels", "image_ids"}:
            raise ValueError("Staged raw NPZ schema differs")
        clean = {name: data[name] for name in data.files}
    _validate_arrays(clean["logits"], clean["labels"], clean["image_ids"], inputs.rows)
    if any(not np.array_equal(clean[name], value) for name, value in (("logits", logits), ("labels", labels), ("image_ids", ids))):
        raise ValueError("Staged NPZ changed raw predictions")
    e4 = predictor.compute_task1_metrics(clean["logits"], labels, temperature=inputs.refs["temperature"], class_names=inputs.mapping)
    e7, curve = _selective(clean["logits"], labels, inputs.refs["temperature"], inputs.refs["threshold"])
    pre_norm = robustness._PreNormDataset(inputs.csv, inputs.images, split=split,
                                        data_payload=inputs.data, mapping={str(k): v for k, v in inputs.mapping.items()})
    e8 = robustness._evaluate_loader(model, DataLoader(pre_norm, batch_size=batch_size, shuffle=False), inputs.cfg)
    e8.update(labels=[int(row["class_id"]) for row in pre_norm.records], checkpoint_sha256=inputs.checkpoint_hash,
              correct_counts={name: round(e8["clean_accuracy" if name == "clean" else name + "_accuracy"] * len(labels)) for name in inputs.cfg.conditions},
              corruption_config=json.loads(json.dumps({name: asdict(getattr(inputs.cfg, name)) for name in inputs.cfg.conditions[1:]})),
              preprocessing=inputs.data["preprocessing"])
    _consistency(e4, e8, logits, labels, ids)
    current = model.state_dict()
    if set(current) != set(initial) or any(not torch.equal(initial[key], current[key].detach().cpu()) for key in initial):
        raise ValueError("Model weights/buffers changed during inference")
    _json(payload / "e4_e6_metrics.json", e4)
    _json(payload / "selective_metrics.json", e7)
    _write(payload / "risk_coverage.csv", selective._curve_csv(curve).encode("utf-8"))
    _write(payload / "risk_coverage_curve.png", selective._curve_png(curve, split))
    _json(payload / "e8_metrics.json", e8)
    plot_reliability_diagram(logits, logits / inputs.refs["temperature"], labels, n_bins=predictor.ECE_BINS,
                             output_path=payload / "reliability_diagram.png", title=f"B1 {split.upper()}: E4/E6")
    _write(payload / "split.csv", inputs.csv.read_bytes())
    _write(payload / "class_mapping.json", (inputs.root / "data/metadata/class_mapping.json").read_bytes())
    _json(payload / "protocol.json", dict(temperature=inputs.refs["temperature"], threshold=inputs.refs["threshold"],
                                         preprocessing=inputs.data["preprocessing"], seed=inputs.cfg.reproducibility.seed,
                                         corruption_config={name: asdict(getattr(inputs.cfg, name)) for name in inputs.cfg.conditions[1:]}))
    _json(payload / "metadata.json", dict(split=split, synthetic=split == "synthetic", num_samples=len(labels), num_classes=30,
         git_sha=inputs.source["head"], checkpoint_sha256=inputs.checkpoint_hash, model_weights_unchanged=True,
         preprocessing=inputs.data["preprocessing"], raw_predictions_sha256=verified["sha256"],
         csv_sha256=_hash(inputs.csv.read_bytes()), class_mapping_sha256=_hash((payload / "class_mapping.json").read_bytes()),
         official_test_used=False, official_final_evaluation=False, configuration_lock_approved=False, official_evaluation_authorized=False))


def run_final_evaluation(*, split, output_dir, batch_size=32, device="cpu", command=None):
    output = _output_path(split, output_dir)  # Before all expensive work.
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be cpu/cuda/auto")
    runtime_device = robustness._device(device)
    source = _source_audit(split)
    with _output_run(output) as staging:
        private = staging / "inputs"
        private.mkdir(mode=0o700)
        inputs = ExecutionInputs(private, split, source)
        payload = staging / "payload"
        payload.mkdir(mode=0o700)
        _execute(inputs, payload, split, batch_size, runtime_device)
        inputs.verify()
        inputs.verify_source()
        manifest = dict(format=FORMAT, execution_status="complete", split=split, synthetic=split == "synthetic",
             scientific_status="synthetic_fixture_only" if split == "synthetic" else "validation_only",
             model=dict(experiment="synthetic_dummy" if split == "synthetic" else "E4",
                        architecture="dummy_pool_linear" if split == "synthetic" else "mobilenet_v2",
                        seed=inputs.cfg.reproducibility.seed, num_classes=30,
                        checkpoint_path=None if split == "synthetic" else predictor.DEFAULT_CHECKPOINT.as_posix(),
                        checkpoint_sha256=inputs.checkpoint_hash),
             timestamp_utc=datetime.now(timezone.utc).isoformat(), git_sha=source["head"], source=source,
             frozen_references=inputs.reference_state, source_sha256=inputs.source_hashes, inputs=inputs.records,
             components=dict.fromkeys(COMPONENTS, "PASS"), official_test_used=False, official_final_evaluation=False,
             configuration_lock_approved=False, official_evaluation_authorized=False,
             command=command or dict(api="run_final_evaluation", split=split, output_dir=str(output), batch_size=batch_size, device=device),
             runtime=dict(python=platform.python_version(), pytorch=torch.__version__, torchvision=torchvision.__version__,
                          numpy=np.__version__, cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                          device=str(runtime_device), batch_size=batch_size, num_workers=0, torch_num_threads=torch.get_num_threads()),
             input_binding=dict(strategy="private per-file copies; same captured bytes hashed and consumed", private_snapshot_verified=True,
                                cross_file_atomic=False, source_scope="Git-bound captured source bytes; no bytecode attestation",
                                source_content_bound_to_git=source["status"] == "PASS", frozen_snapshot_content_bound_to_git=True),
             comparison_tolerance=COMPARISON_TOLERANCE,
             output_sha256={name: preflight._read_once(payload / name)["sha256"] for name in PAYLOADS if name != "manifest.json"})
        _json(payload / "manifest.json", manifest)
        _validate_bundle({name: preflight._read_once(payload / name)["data"] for name in PAYLOADS})
    return load_execution_bundle(output)["manifest"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--split", choices=["synthetic", "validation"], required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = parser.parse_args(argv)
    try:
        result = run_final_evaluation(split=args.split, output_dir=args.output_dir, batch_size=args.batch_size, device=args.device,
                                      command=[sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)])
        print(json.dumps(dict(execution_status=result["execution_status"], split=result["split"], synthetic=result["synthetic"],
                              source_integrity=result["source"]["status"], official_final_evaluation=False), indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[B1-EXECUTION-ERROR] {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
