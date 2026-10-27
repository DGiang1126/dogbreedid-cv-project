"""B2 execution boundary; no dataset access before approval, claim and reservation.

ReviewedRelease must be pinned independently by the leader/trusted caller, not
obtained from an incoming bundle. No approved controls or CLI are created here.
Synthetic fixtures retain a separately pinned dummy checkpoint and explicit
fixture-only results. The Official adapter reuses the accepted E4 loader and
scientific functions without modifying Task 04/B1 guards. Its canonical filename
and schema come from committed scripts/prepare_data.py and the dataset protocol.
The leader must still approve the canonical CSV revision in the descriptor.
Only after claim/reservation may its Git blob, live CSV and images be opened.

Private staging and persistent claim roots retain the accepted same-user trust
boundary. Snapshots are per-file, not cross-file atomic. Consumers need the pinned
Git objects, matching source checkout and claim ledger; no signed attestation is
claimed. Producer and consumer do not reopen original dataset files for metrics.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
import csv
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import tempfile
import uuid

from scripts import run_final_evaluation as b1
from src.final_evaluation import authorization as auth

REAL_ROOT = Path(__file__).resolve().parents[2]
LOCK = "docs/final_evaluation/configuration_lock.md"
CONTROL_ROOT = "docs/final_evaluation/controls/"
CONTROL_INDEX = CONTROL_ROOT + "b2_release.json"
CONTROL_FILES = dict(approval=CONTROL_ROOT + "lock_approval.json",
                     authorization=CONTROL_ROOT + "run_authorization.json",
                     protocol=CONTROL_ROOT + "protocol.json")
EVIDENCE_FILES = {"leader-lock": CONTROL_ROOT + "lock_decision.txt",
                  "leader-grant": CONTROL_ROOT + "grant_decision.txt",
                  "phase-a": "docs/final_evaluation/task05_phase_a_preflight_acceptance.md",
                  "phase-b1": "docs/final_evaluation/task05_phase_b1_validation_acceptance.md"}
INTEGRATION_SOURCE = ("src/final_evaluation/official_execution.py", "tests/test_official_evaluation_integration.py",
                      "scripts/prepare_data.py", "docs/protocol/dataset_protocol.md")
OFFICIAL_CSV = "data/splits/official_test.csv"  # prepare_data.py: save_split(official_test, ...)
GIT_PATHS = {CONTROL_INDEX, LOCK, *CONTROL_FILES.values(), *EVIDENCE_FILES.values(),
             *auth.FROZEN_REFERENCES, *INTEGRATION_SOURCE}
FORMAT = "dogbreedid.b2.synthetic-integration.v1"
OFFICIAL_FORMAT = "dogbreedid.b2.official-execution.v1"
PAYLOADS = (*b1.PAYLOADS, "controls.json", "claim.json", "dataset.json")
FALSE_FLAGS = ("official_test_used", "official_final_evaluation", "configuration_lock_approved", "official_evaluation_authorized")


@dataclass(frozen=True)
class ReviewedRelease:
    """An independently reviewed full Git revision, never self-reported trust."""
    revision: str


@dataclass(frozen=True)
class DatasetDescriptor:
    kind: str
    reference: str
    revision: str
    fixture_root: str | None = None  # Metadata only until authorized/reserved.


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome, not a global grant-availability query.

    claim_consumed includes known/possibly spent state observed at the claim
    step. False on an earlier failure does not authorize a retry.
    """
    status: str
    code: str
    claim_consumed: bool = False
    output_dir: str | None = None
    detail: str = ""


class _Stop(ValueError):
    def __init__(self, status, code):
        self.status, self.code = status, code
        super().__init__(code)


def _git_buffers(revision, names, *, authorized_dataset=False):
    """B1 batch-reading pattern, with a separate fixed control/source allowlist."""
    if not auth._hex(revision, 40):
        raise ValueError("A full trusted Git revision is required")
    names = tuple(names)
    allowed = GIT_PATHS | ({OFFICIAL_CSV} if authorized_dataset else set())
    if len(set(names)) != len(names) or not set(names) <= allowed:
        raise ValueError("Unapproved Git input path")
    try:
        result = subprocess.run(["git", "--no-replace-objects", "-C", str(b1.SOURCE_ROOT), "cat-file", "--batch"],
                                input=("\n".join(f"{revision}:{n}" for n in names) + "\n").encode(),
                                capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise b1.preflight.Blocked("Trusted control/source objects unavailable") from error
    reader, blobs = io.BytesIO(result.stdout), {}
    for name in names:
        header = reader.readline().split()
        if len(header) != 3 or header[1] != b"blob" or not header[2].isdigit():
            raise b1.preflight.Blocked("Missing trusted control/source blob: " + name)
        raw = reader.read(int(header[2]))
        if len(raw) != int(header[2]) or reader.read(1) != b"\n":
            raise ValueError("Invalid Git batch response")
        blobs[name] = raw
    if reader.read():
        raise ValueError("Unexpected Git batch bytes")
    return blobs


def _variant(blob, digest):
    for _, candidate in b1._newline_variants(blob):
        if b1._hash(candidate) == digest:
            return candidate
    raise ValueError("Reviewed digest does not bind to Git bytes")


def observe_runtime(batch_size, device="cpu"):
    """Declared device + observed dependencies; no probe or device allocation."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    return dict(python=platform.python_version(), pytorch=str(b1.torch.__version__),
                torchvision=str(b1.torchvision.__version__), numpy=b1.np.__version__,
                cuda_build=str(b1.torch.version.cuda or "none"), device=device, batch_size=batch_size,
                num_workers=0, torch_num_threads=b1.torch.get_num_threads(), os=platform.platform(),
                **{name.replace('-', '_').lower(): version(name) for name in
                   ("Pillow", "scikit-learn", "scipy", "PyYAML", "matplotlib", "pandas")})


def _controls(release, *, live, runtime):
    if not isinstance(release, ReviewedRelease):
        raise _Stop("BLOCKED", "TRUSTED_RELEASE_REQUIRED")
    buffers = _git_buffers(release.revision, (CONTROL_INDEX, LOCK, *CONTROL_FILES.values(), *EVIDENCE_FILES.values()))
    index = b1.preflight._parse_json(buffers[CONTROL_INDEX], "control index")
    fields = {"schema", "source_revision", "approval_sha256", "authorization_sha256", "protocol_sha256",
              "claim_root", "output_root", "profile"}
    if set(index) != fields or index["schema"] != "dogbreedid.reviewed-controls.v1" or not auth._hex(index["source_revision"], 40):
        raise ValueError("Invalid reviewed control index")
    records = {name: _variant(buffers[path], index[name + "_sha256"]) for name, path in CONTROL_FILES.items()}
    approval = b1.preflight._parse_json(records["approval"], "approval")
    expected_frozen = _git_buffers(index["source_revision"], auth.FROZEN_REFERENCES)
    if live:
        frozen = {name: b1.preflight._read_once(b1.SOURCE_ROOT / name)["data"] for name in auth.FROZEN_REFERENCES}
        for name in frozen:
            b1._bind_bytes(frozen[name], expected_frozen[name], name)
        lock = b1.preflight._read_once(b1.SOURCE_ROOT / LOCK)["data"]
    else:
        frozen = {name: _variant(expected_frozen[name], approval["frozen_sha256"][name]) for name in auth.FROZEN_REFERENCES}
        lock = _variant(buffers[LOCK], approval["lock_sha256"])
    # Paths are compiled roles, never a record's evidence.reference as a path.
    evidence = {name: buffers[path] for name, path in EVIDENCE_FILES.items()}
    anchors = auth.TrustAnchors(index["approval_sha256"], index["authorization_sha256"], index["source_revision"],
                               index["protocol_sha256"], index["claim_root"], index["output_root"])
    inputs = auth.AuthorizationInputs(lock, records["approval"], records["authorization"], records["protocol"],
                                      frozen, evidence, runtime)
    profile = index["profile"]
    if (not isinstance(profile, dict) or set(profile) != {"kind", "checkpoint_sha256", "declaration_sha256", "n_samples"}
            or profile["kind"] not in {"synthetic_fixture", "official"}):
        raise ValueError("Invalid execution profile")
    if profile["kind"] == "synthetic_fixture":
        if (not auth._hex(profile["checkpoint_sha256"]) or not auth._hex(profile["declaration_sha256"])
                or type(profile["n_samples"]) is not int or not 0 < profile["n_samples"] <= 30):
            raise ValueError("Invalid bounded synthetic fixture profile")
    elif (profile["checkpoint_sha256"] != auth.CHECKPOINT_SHA256
          or type(profile["n_samples"]) is not int or profile["n_samples"] != 1934
          or profile["declaration_sha256"] is not None):
        raise ValueError("Official profile differs from frozen protocol")
    return index, inputs, anchors


def _source_hashes(revision):
    hashes = auth.capture_source_binding(revision)
    state = b1.preflight._git_state(b1.SOURCE_ROOT, INTEGRATION_SOURCE)
    if state["status"] != "PASS":
        raise ValueError("Integration source must be tracked and clean")
    expected = _git_buffers(revision, INTEGRATION_SOURCE)
    for name in INTEGRATION_SOURCE:
        raw = b1.preflight._read_once(b1.SOURCE_ROOT / name)["data"]
        hashes[name] = b1._bind_bytes(raw, expected[name], name)["sha256"]
    return hashes


def _verify_live(inputs, source_hashes, revision, batch_size):
    auth._exact(_source_hashes(revision), source_hashes)
    auth._exact(observe_runtime(batch_size, inputs.runtime["device"]), dict(inputs.runtime))
    for name, expected in {LOCK: inputs.lock_document, **inputs.frozen_references}.items():
        if b1.preflight._read_once(b1.SOURCE_ROOT / name)["data"] != expected:
            raise ValueError("Live control/frozen reference drift before dataset access")


class _Snapshot:
    def __init__(self, root):
        self.root, self.records = root, {}

    def write(self, name, raw):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        b1._write(path, raw)
        self.records[name] = dict(sha256=b1._hash(raw), size_bytes=len(raw))
        return path

    def capture(self, original, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as writer:
            captured = b1.preflight._read_once(original, destination=writer)
        self.records[name] = dict(sha256=captured["sha256"], size_bytes=captured["size_bytes"])
        return path

    def verify(self):
        for name, record in self.records.items():
            b1.preflight._exact(b1.preflight._read_once(self.root / name)["sha256"], record["sha256"], "Private snapshot")


def _settings(snapshot, inputs):
    for name, raw in inputs.frozen_references.items():
        snapshot.write(name, raw)
    root = snapshot.root
    refs = b1.selective._load_references(root)
    b1.predictor._load_temperature(refs["temperature_path"])
    protocol = b1.preflight._parse_json(inputs.protocol, "frozen protocol")
    auth._exact(protocol, auth.approved_protocol())
    auth._exact(refs["temperature"], protocol["temperature"])
    auth._exact(refs["threshold"], protocol["threshold"])
    data = b1.predictor._load_yaml(root / "configs/data.yaml")
    e4 = b1.predictor._load_yaml(root / "configs/experiments/e4_cutmix.yaml")
    auth._exact(e4["model"]["backbone_frozen"], True)
    auth._exact(e4["augmentation"]["cutmix"]["enabled"], True)
    b1.robustness._build_pre_norm_transform(data)
    cfg = b1.load_e8_config(root / "configs/experiments/e8_robustness.yaml")
    auth._exact(cfg.reproducibility.seed, protocol["e8"]["seed"])
    auth._exact(cfg.conditions, list(auth.SCOPE_E8))
    # Same JSON representation used by B1 for dataclass tuple/list fields.
    severity = json.loads(json.dumps({name: asdict(getattr(cfg, name)) for name in cfg.conditions[1:]}))
    auth._exact(severity, b1.APPROVED_CORRUPTIONS)
    mapping, _, _ = b1.selective._mapping(root)
    return data, cfg, mapping


class _SyntheticReader:
    def __init__(self, descriptor):
        self.descriptor = descriptor  # No filesystem access at construction.

    def capture(self, snapshot, profile, mapping):
        path = Path(auth._output(self.descriptor.fixture_root))
        temporary = Path(tempfile.gettempdir()).absolute()
        if (path == temporary or not path.is_relative_to(temporary)
                or path.is_relative_to(REAL_ROOT) or REAL_ROOT.is_relative_to(path)
                or path.is_relative_to(b1.SOURCE_ROOT) or b1.SOURCE_ROOT.is_relative_to(path)):
            raise PermissionError("Synthetic input must be isolated OS-temp fixture data")
        root = b1._safe_path(path)
        declaration = snapshot.capture(root / "fixture_manifest.json", "dataset/declaration.json")
        if snapshot.records["dataset/declaration.json"]["sha256"] != profile["declaration_sha256"]:
            raise ValueError("Fixture declaration differs from independently reviewed profile")
        declared = b1.preflight._parse_json(declaration.read_bytes(), "fixture declaration")
        if set(declared) != {"synthetic", "n_samples", "split_sha256", "images"} or declared["synthetic"] is not True:
            raise ValueError("Only declared synthetic inputs are supported")
        csv_path = snapshot.capture(root / "split.csv", "dataset/split.csv")
        if snapshot.records["dataset/split.csv"]["sha256"] != declared["split_sha256"]:
            raise ValueError("Fixture CSV differs from captured declaration")
        rows = b1.selective._csv_rows(csv_path, mapping, "synthetic")
        if len(rows) != profile["n_samples"] or declared["n_samples"] != len(rows) or set(declared["images"]) != {n for n, _ in rows}:
            raise ValueError("Fixture counts/IDs differ from reviewed profile")
        for image_id, _ in rows:
            if not image_id.startswith("synthetic/"):
                raise ValueError("Fixture ID namespace is required")
            snapshot.capture(root / "images" / image_id, "dataset/images/" + image_id)
            if snapshot.records["dataset/images/" + image_id]["sha256"] != declared["images"][image_id]:
                raise ValueError("Fixture image differs from captured declaration")
        return csv_path, snapshot.root / "dataset/images", rows, declared


def _official_rows(raw, mapping, expected_samples):
    """Parse the generator's schema from the same captured bytes that are hashed."""
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    if reader.fieldnames != ["relative_path", "class_id", "breed_name", "source_split"]:
        raise ValueError("Official CSV must use the committed generator schema")
    rows, seen = [], set()
    for row in reader:
        name = row["relative_path"]
        if None in row or not b1.selective._safe_id(name) or name in seen:
            raise ValueError("Unsafe/duplicate Official image ID")
        try:
            label = int(row["class_id"])
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid Official class_id") from error
        if (str(label) != row["class_id"] or label not in mapping
                or row["breed_name"] != mapping[label] or row["source_split"] != "official_test"):
            raise ValueError("Official label/mapping/source_split mismatch")
        seen.add(name)
        rows.append((name, label))
    if len(rows) != expected_samples or {label for _, label in rows} != set(mapping):
        raise ValueError("Official sample count/classes differ from approved contract")
    return rows


def _official_descriptor(descriptor):
    # Lexical metadata only: do not stat/resolve/open a dataset path here.
    if (descriptor.kind != "official" or descriptor.reference != OFFICIAL_CSV
            or not auth._hex(descriptor.revision, 40) or descriptor.fixture_root is not None):
        raise _Stop("BLOCKED", "OFFICIAL_DESCRIPTOR_UNCONFIRMED")


def _dataset_path(root, relative):
    """Official-only path policy; caller must already own a claim/reservation.

    Phase A/B1's path helper intentionally rejects the canonical CSV filename,
    so it cannot be used as this adapter's dataset policy. Their guard is intact.
    """
    if not b1.selective._safe_id(relative):
        raise ValueError("Unsafe canonical dataset path")
    path = root / relative
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor /= part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError("Dataset symlink/junction is forbidden")
    return path


def _capture_dataset(snapshot, root, relative, name):
    """Single-handle snapshot under the authorized dataset policy, no reopen/hash."""
    path = _dataset_path(root, relative)
    destination = snapshot.root / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as reader, destination.open("xb") as writer:
        initial = os.fstat(reader.fileno())
        if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
            raise ValueError("Dataset input must be a regular, unaliased file")
        if b1.predictor._identity(_dataset_path(root, relative).stat()) != b1.predictor._identity(initial):
            raise ValueError("Dataset changed while opening")
        digest, size = b1.hashlib.sha256(), 0
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        fingerprint = b1.preflight._fingerprint
        # Compare mutation timestamps on the same open handle, as in B1.
        # Windows path.stat().st_ctime and fstat().st_ctime can differ even for
        # an unchanged file; path identity is checked separately for replacement.
        if (fingerprint(os.fstat(reader.fileno())) != fingerprint(initial)
                or b1.predictor._identity(_dataset_path(root, relative).stat()) != b1.predictor._identity(initial)
                or size != initial.st_size):
            raise ValueError("Dataset changed during capture")
    snapshot.records[name] = dict(sha256=digest.hexdigest(), size_bytes=size)
    return destination


class _OfficialReader:
    """Constructed only inside the claimed/reserved run; never aliases Validation."""
    def __init__(self, descriptor):
        _official_descriptor(descriptor)
        self.descriptor = descriptor

    def capture(self, snapshot, profile, mapping):
        canonical = _git_buffers(self.descriptor.revision, (OFFICIAL_CSV,), authorized_dataset=True)[OFFICIAL_CSV]
        csv_path = _capture_dataset(snapshot, b1.SOURCE_ROOT, OFFICIAL_CSV, "dataset/split.csv")
        raw = csv_path.read_bytes()
        binding = b1._bind_bytes(raw, canonical, "canonical Official split")
        rows = _official_rows(raw, mapping, profile["n_samples"])
        for image_id, _ in rows:
            _capture_dataset(snapshot, b1.SOURCE_ROOT, b1.IMAGES.as_posix() + "/" + image_id,
                             "dataset/images/" + image_id)
        declaration = dict(synthetic=False, reference=OFFICIAL_CSV, revision=self.descriptor.revision,
                           canonical_csv_sha256=b1._hash(canonical), newline_binding=binding["match_mode"],
                           n_samples=len(rows), split_sha256=b1._hash(raw),
                           images={name: snapshot.records["dataset/images/" + name]["sha256"] for name, _ in rows})
        snapshot.write("dataset/declaration.json", b1._json_bytes(declaration))
        return csv_path, snapshot.root / "dataset/images", rows, declaration


def _make_reader(descriptor):
    return _SyntheticReader(descriptor) if descriptor.kind == "synthetic_fixture" else _OfficialReader(descriptor)


def _output_location(request, anchors):
    path = b1._safe_path(auth._output(request.output_dir))
    approved_root = b1._safe_path(auth._output(anchors.output_root))
    if path == approved_root or not path.is_relative_to(approved_root):
        raise PermissionError("Output escaped trusted root")
    if path.is_relative_to(b1.SOURCE_ROOT):
        allowed = b1.SOURCE_ROOT / "outputs/final_evaluation_b2"
        if path == allowed or not path.is_relative_to(allowed):
            raise PermissionError("Historical/source directories cannot be output destinations")
    elif not path.is_relative_to(Path(tempfile.gettempdir()).absolute()) or path.is_relative_to(REAL_ROOT):
        raise PermissionError("External synthetic outputs must remain in OS temp")
    if os.path.lexists(path):
        raise FileExistsError("Output directory already exists, including an empty directory")
    for parent in path.parents:
        if any(os.path.lexists(parent / marker) for marker in (b1.RUN_MARKER, ".complete", ".incomplete")):
            raise PermissionError("Cannot nest output in an existing bundle")
        if parent == approved_root:
            break
    return path


@contextmanager
def _publish(output, *, release):
    """B1 no-clobber publication pattern, isolated format/payloads/consumer."""
    b1._safe_path(output.parent).mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Must happen after the persistent authorization claim.
    identity = b1.predictor._identity(output.stat())
    token = (uuid.uuid4().hex + "\n").encode()

    def check():
        b1._safe_path(output)
        if b1.predictor._identity(output.stat()) != identity:
            raise RuntimeError("Reserved directory was replaced")

    try:
        b1._write(output / b1.RUN_MARKER, token)
        with tempfile.TemporaryDirectory(prefix=".dogbreedid-b2-publish-", dir=output.parent) as temporary:
            payload = Path(temporary)
            yield payload
            hashes = {name: b1.preflight._read_once(payload / name)["sha256"] for name in PAYLOADS}
            for name in PAYLOADS:
                check()
                os.link(payload / name, output / name)
            if {p.name for p in output.iterdir()} != {*PAYLOADS, b1.RUN_MARKER}:
                raise ValueError("Foreign files appeared during publication")
            bundle_format = b1.preflight._parse_json((payload / "manifest.json").read_bytes(), "manifest")["format"]
        check()
        b1._json(output / ".complete", dict(format=bundle_format, run_token=token.decode(), output_sha256=hashes))
        load_authorized_bundle(output, release=release)
    except BaseException:
        try:
            check()
            b1._write(output / ".incomplete", b"INCOMPLETE B2 INTEGRATION; do not consume.\n")
        except (OSError, ValueError, RuntimeError):
            pass
        raise  # Never delete public artifacts, foreign files or the claim.


def _load_backend(snapshot, checkpoint, cfg, *, official, device="cpu"):
    if not official:
        model = b1.robustness._make_synthetic_model(cfg.reproducibility.seed)
        model.load_state_dict(b1.torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        return model.to(device).eval()
    # The full approved SHA identifies the E4 seed-42 epoch-25 checkpoint, not
    # merely its tensor shapes. The accepted loader verifies it before loading,
    # builds MobileNetV2 without downloads, and uses weights_only/strict/eval.
    model, digest = b1.predictor.load_e4_model(snapshot.root, checkpoint, b1.torch.device(device))
    auth._exact(digest, auth.CHECKPOINT_SHA256)
    if (model.classifier[-1].out_features != 30
            or any(parameter.requires_grad for parameter in model.features.parameters()) or model.training):
        raise ValueError("E4 architecture/frozen-backbone/eval contract differs")
    return model


class _OfficialConditionDataset(b1.torch.utils.data.Dataset):
    """Authorized policy over private snapshots; Task 03/04 guards stay intact.

    Reuse only the evaluator's pure corruption dispatcher, not its tooling-only
    constructor/evaluate policy. No condition algorithm or severity is redefined.
    collect_predictions supplies the accepted inference-mode/model semantics.
    """
    def __init__(self, dataset, cfg, condition, data):
        self.dataset, self.cfg, self.condition = dataset, cfg, condition
        normalization = data["preprocessing"]["normalization"]
        self.normalize = b1.torchvision.transforms.Normalize(normalization["mean"], normalization["std"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        image = sample["image"]
        if not b1.torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Corruption requires finite pre-normalization [0,1] tensors")
        corrupted = b1.robustness.RobustnessEvaluator._corrupt(
            self, image.unsqueeze(0), [sample["relative_path"]], self.condition)[0]
        sample["image"] = self.normalize(corrupted)
        return sample


def _official_e8(model, csv_path, images, data, cfg, rows, batch_size, device):
    pre_norm = b1.DogBreedDataset(csv_path, images, b1.robustness._build_pre_norm_transform(data))
    counts = {}
    for condition in cfg.conditions:
        dataset = _OfficialConditionDataset(pre_norm, cfg, condition, data)
        logits, labels, ids = b1.predictor.collect_predictions(
            model, b1.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0), device=device)
        b1._validate_arrays(logits, labels, ids, rows)
        counts[condition] = int((logits.argmax(1) == labels).sum())
    accuracies = {name: count / len(rows) for name, count in counts.items()}
    result = dict(clean_accuracy=accuracies["clean"], conditions=list(cfg.conditions), n_samples=len(rows),
                  image_ids=[name for name, _ in rows], correct_counts=counts)
    for name in cfg.conditions[1:]:
        result[name + "_accuracy"] = accuracies[name]
        result[name + "_accuracy_drop"] = accuracies["clean"] - accuracies[name]
    return result


def _compute(snapshot, payload, checkpoint, data, cfg, mapping, csv_path, images, rows, batch_size, *, official=False, device="cpu"):
    model = _load_backend(snapshot, checkpoint, cfg, official=official, device=device)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    dataset = b1.DogBreedDataset(csv_path, images, b1.build_transforms(data, "inference"))
    logits, labels, ids = b1.predictor.collect_predictions(model, b1.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0), device=device)
    b1._validate_arrays(logits, labels, ids, rows)
    b1.predictor.save_raw_predictions(payload / "clean_predictions.npz", logits, labels, ids, exclusive=True)
    e4 = b1.predictor.compute_task1_metrics(logits, labels, temperature=auth.approved_protocol()["temperature"], class_names=mapping)
    e7, curve = b1._selective(logits, labels, auth.approved_protocol()["temperature"], auth.approved_protocol()["threshold"])
    if official:
        e8 = _official_e8(model, csv_path, images, data, cfg, rows, batch_size, device)
    else:
        pre_norm = b1.robustness._PreNormDataset(csv_path, images, split="synthetic", data_payload=data,
                                              mapping={str(k): v for k, v in mapping.items()})
        e8 = b1.robustness._evaluate_loader(model, b1.DataLoader(pre_norm, batch_size=batch_size, shuffle=False, num_workers=0), cfg)
    e8.update(labels=labels.tolist(), correct_counts={c: round(e8["clean_accuracy" if c == "clean" else c + "_accuracy"] * len(labels)) for c in cfg.conditions},
              corruption_config={n: asdict(getattr(cfg, n)) for n in cfg.conditions[1:]}, preprocessing=data["preprocessing"])
    b1._consistency(e4, e8, logits, labels, ids)
    if any(not b1.torch.equal(v, model.state_dict()[k]) for k, v in initial.items()):
        raise ValueError("Model weights/buffers changed")
    b1._json(payload / "e4_e6_metrics.json", e4)
    b1._json(payload / "selective_metrics.json", e7)
    b1._json(payload / "e8_metrics.json", e8)
    b1._write(payload / "risk_coverage.csv", b1.selective._curve_csv(curve).encode())
    b1._write(payload / "risk_coverage_curve.png", b1.selective._curve_png(curve, "official_test" if official else "synthetic"))
    b1.plot_reliability_diagram(logits, logits / auth.approved_protocol()["temperature"], labels, n_bins=b1.predictor.ECE_BINS,
                                output_path=payload / "reliability_diagram.png", title="B2 Official" if official else "B2 synthetic fixture only")
    b1._write(payload / "split.csv", csv_path.read_bytes())
    b1._write(payload / "class_mapping.json", (snapshot.root / "data/metadata/class_mapping.json").read_bytes())


def _control_receipt(release, index, inputs):
    return dict(reviewed_release=release.revision, source_revision=index["source_revision"],
                approval_sha256=index["approval_sha256"], authorization_sha256=index["authorization_sha256"],
                protocol_sha256=index["protocol_sha256"], lock_sha256=b1._hash(inputs.lock_document),
                frozen_sha256={n: b1._hash(v) for n, v in inputs.frozen_references.items()},
                evidence_sha256={n: b1._hash(v) for n, v in inputs.evidence.items()}, profile=index["profile"])


def _result_flags(official):
    return dict(synthetic=not official, scientific_status="official_evaluation" if official else "synthetic_fixture_only",
                **dict.fromkeys(FALSE_FLAGS, official))


def _metadata(official, revision, count, checkpoint_hash):
    return dict(**_result_flags(official), model_kind="e4_mobilenet_v2" if official else "synthetic_dummy",
                source_revision=revision, n_samples=count, approved_e4_checkpoint_sha256=auth.CHECKPOINT_SHA256,
                **({"executed_checkpoint_sha256": checkpoint_hash} if official else
                   {"executed_fixture_checkpoint_sha256": checkpoint_hash}), model_weights_unchanged=True)


def _validate_buffers(buffers, release):
    parsed = {n: b1.preflight._parse_json(raw, n) for n, raw in buffers.items() if n.endswith(".json")}
    manifest, metadata = parsed["manifest.json"], parsed["metadata.json"]
    if manifest.get("execution_status") != "complete":
        raise ValueError("Invalid integration completion")
    timestamp = datetime.fromisoformat(manifest["timestamp_utc"])
    if timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
        raise ValueError("Execution timestamp must be UTC")
    if manifest.get("output_sha256") != {n: b1._hash(v) for n, v in buffers.items() if n != "manifest.json"}:
        raise ValueError("Manifest payload hashes differ")
    index, inputs, anchors = _controls(release, live=False, runtime=manifest["runtime"])
    official = index["profile"]["kind"] == "official"
    if manifest.get("format") != (OFFICIAL_FORMAT if official else FORMAT):
        raise ValueError("Bundle format differs from reviewed profile")
    for document in (manifest, metadata):
        auth._exact({key: document.get(key) for key in _result_flags(official)}, _result_flags(official))
    request = auth.RunRequest(manifest["run_id"], manifest["output_dir"], tuple(manifest["scope"]))
    decision = auth.validate_authorization(inputs, anchors, request)
    if decision.status != "PASS":
        raise ValueError("Invalid authorization linkage: " + str(decision.reasons))
    auth._exact(manifest["invocation"], dict(api="run_authorized_evaluation", batch_size=manifest["runtime"]["batch_size"],
                                           device=manifest["runtime"]["device"], reviewed_release=release.revision))
    auth._exact(parsed["controls.json"], _control_receipt(release, index, inputs))
    auth._exact(parsed["protocol.json"], auth.approved_protocol())
    auth._exact(manifest["source_sha256"], _source_hashes(index["source_revision"]))
    descriptor = manifest["dataset_descriptor"]
    approval = b1.preflight._parse_json(inputs.approval, "approval")
    if (not isinstance(descriptor, dict) or set(descriptor) != {"kind", "reference", "revision", "fixture_root"}
            or descriptor["kind"] != index["profile"]["kind"]
            or (not official and not auth._text(descriptor["fixture_root"]))
            or descriptor["reference"] != approval["dataset_descriptor"]["reference"]
            or descriptor["revision"] != approval["dataset_descriptor"]["revision"]):
        raise ValueError("Dataset descriptor is not bound to reviewed approval")
    if manifest["git_sha"] != index["source_revision"] or metadata["source_revision"] != index["source_revision"]:
        raise ValueError("Wrong source provenance")
    if official:
        _official_descriptor(DatasetDescriptor(**descriptor))
    claim_path = Path(anchors.claim_root) / (decision.grant.authorization_id + ".claim.json")
    claim_raw = b1.preflight._read_once(claim_path)["data"]
    if buffers["claim.json"] != claim_raw:
        raise ValueError("Claim receipt differs from persistent control ledger")
    claim = parsed["claim.json"]
    auth._exact(claim, dict(schema="dogbreedid.authorization-claim.v1", authorization_id=decision.grant.authorization_id,
                           run_id=request.run_id, output_dir=decision.grant.output_dir, scope=list(request.scope),
                           approval_sha256=anchors.approval_sha256, authorization_sha256=anchors.authorization_sha256,
                           source_revision=anchors.source_revision, state="CLAIMED", automatic_retry=False))
    with b1.np.load(io.BytesIO(buffers["clean_predictions.npz"]), allow_pickle=False) as stored:
        if set(stored.files) != {"logits", "labels", "image_ids"}:
            raise ValueError("Wrong NPZ schema")
        logits, labels, ids = (stored[n] for n in ("logits", "labels", "image_ids"))
    with tempfile.TemporaryDirectory(prefix="dogbreedid-b2-consumer-") as temp:
        snapshot = _Snapshot(Path(temp))
        data, cfg, mapping = _settings(snapshot, inputs)
        csv_path = snapshot.write("split.csv", buffers["split.csv"])
        rows = (_official_rows(buffers["split.csv"], mapping, index["profile"]["n_samples"]) if official
                else b1.selective._csv_rows(csv_path, mapping, "synthetic"))
    b1._validate_arrays(logits, labels, ids, rows)
    declaration = parsed["dataset.json"]
    profile = index["profile"]
    declaration_hash = b1._hash(buffers["dataset.json"])
    if official:
        canonical = _git_buffers(descriptor["revision"], (OFFICIAL_CSV,), authorized_dataset=True)[OFFICIAL_CSV]
        binding = b1._bind_bytes(buffers["split.csv"], canonical, "canonical Official split")
        auth._exact({k: declaration.get(k) for k in ("reference", "revision", "canonical_csv_sha256", "newline_binding")},
                    dict(reference=OFFICIAL_CSV, revision=descriptor["revision"],
                         canonical_csv_sha256=b1._hash(canonical), newline_binding=binding["match_mode"]))
    elif declaration_hash != profile["declaration_sha256"]:
        raise ValueError("Fixture declaration differs from reviewed hash")
    if (declaration["synthetic"] is not (not official)
            or declaration["n_samples"] != len(rows) or len(rows) != profile["n_samples"]
            or declaration["split_sha256"] != b1._hash(buffers["split.csv"])
            or set(declaration["images"]) != {n for n, _ in rows}):
        raise ValueError("Dataset provenance differs from reviewed descriptor")
    records = manifest["inputs"]
    expected_hashes = {**{n: b1._hash(v) for n, v in inputs.frozen_references.items()},
                       "checkpoint.pt": profile["checkpoint_sha256"], "dataset/declaration.json": declaration_hash,
                       "dataset/split.csv": declaration["split_sha256"],
                       **{"dataset/images/" + n: h for n, h in declaration["images"].items()}}
    if set(records) != set(expected_hashes):
        raise ValueError("Missing/extra snapshot provenance")
    for name, digest in expected_hashes.items():
        record = records[name]
        if (not isinstance(record, dict) or set(record) != {"sha256", "size_bytes"} or record["sha256"] != digest
                or not auth._hex(digest) or type(record["size_bytes"]) is not int or record["size_bytes"] <= 0):
            raise ValueError("Invalid captured input hash/size")
    known_bytes = {**inputs.frozen_references, "dataset/declaration.json": buffers["dataset.json"],
                   "dataset/split.csv": buffers["split.csv"]}
    if any(records[n]["size_bytes"] != len(v) for n, v in known_bytes.items()):
        raise ValueError("Captured input size differs from verified buffers")
    b1._input_record_matches(records["data/metadata/class_mapping.json"], buffers["class_mapping.json"], "published mapping")
    auth._exact(metadata, _metadata(official, anchors.source_revision, len(labels), profile["checkpoint_sha256"]))
    e4 = b1.predictor.compute_task1_metrics(logits, labels, temperature=auth.approved_protocol()["temperature"], class_names=mapping)
    e7, curve = b1._selective(logits, labels, auth.approved_protocol()["temperature"], auth.approved_protocol()["threshold"])
    auth._exact(parsed["e4_e6_metrics.json"]["calibration"]["temperature"], auth.approved_protocol()["temperature"])
    b1.preflight._equal(parsed["e4_e6_metrics.json"], e4)
    b1.preflight._equal(parsed["selective_metrics.json"], e7)
    auth._exact(parsed["selective_metrics.json"]["temperature"], auth.approved_protocol()["temperature"])
    auth._exact(parsed["selective_metrics.json"]["threshold"], auth.approved_protocol()["threshold"])
    if buffers["risk_coverage.csv"] != b1.selective._curve_csv(curve).encode():
        raise ValueError("Risk-coverage curve differs")
    e8 = parsed["e8_metrics.json"]
    auth._exact(e8["corruption_config"], b1.APPROVED_CORRUPTIONS)
    auth._exact(e8["preprocessing"], data["preprocessing"])
    b1._consistency(e4, e8, logits, labels, ids)
    for name in ("reliability_diagram.png", "risk_coverage_curve.png"):
        with b1.Image.open(io.BytesIO(buffers[name])) as image:
            image.verify()
    return dict(manifest=manifest, metadata=metadata, metrics=e4, selective=e7, robustness=e8,
                logits=logits, labels=labels, image_ids=ids)


def load_authorized_bundle(output_dir, *, release):
    """Completion plus externally pinned authorization/provenance, never self-trust."""
    output = b1._safe_path(output_dir)
    if os.path.lexists(output / ".incomplete"):
        raise ValueError("Incomplete integration bundle")
    completed = b1.preflight._parse_json(b1.preflight._read_once(output / ".complete")["data"], "completion")
    token = b1.preflight._read_once(output / b1.RUN_MARKER, limit=64)["data"]
    if (completed.get("format") not in {FORMAT, OFFICIAL_FORMAT} or len(token) != 33 or token[-1:] != b"\n"
            or any(c not in b"0123456789abcdef" for c in token[:-1]) or completed.get("run_token") != token.decode()):
        raise ValueError("Invalid completion token")
    records = {n: b1.preflight._read_once(output / n) for n in PAYLOADS}
    if completed.get("output_sha256") != {n: v["sha256"] for n, v in records.items()}:
        raise ValueError("Completion hashes differ from captured bytes")
    if {p.name for p in output.iterdir()} != {*PAYLOADS, b1.RUN_MARKER, ".complete"}:
        raise ValueError("Incomplete/foreign bundle files")
    result = _validate_buffers({n: v["data"] for n, v in records.items()}, release)
    if completed["format"] != result["manifest"]["format"]:
        raise ValueError("Completion format differs from verified manifest")
    if os.path.lexists(output / ".incomplete"):
        raise ValueError("Bundle invalidated while consuming")
    return result


def run_authorized_evaluation(request, descriptor, *, release=None, batch_size=32, device="cpu"):
    """Central API boundary. Actual repository stays denied; no CLI/bypass exists."""
    claimed, output = False, None
    try:
        if not isinstance(request, auth.RunRequest) or not isinstance(descriptor, DatasetDescriptor):
            raise ValueError("Invalid request/descriptor")
        live_lock = b1.preflight._read_once(b1.SOURCE_ROOT / LOCK)["data"]
        states = [line for line in live_lock.decode("utf-8").splitlines() if line.startswith("**Trạng thái:**")]
        if states == ["**Trạng thái:** BẢN NHÁP — CHƯA KHÓA CẤU HÌNH"]:
            raise _Stop("BLOCKED", "LOCK_DRAFT")
        if states != ["**Trạng thái:** ĐÃ KHÓA CẤU HÌNH"]:
            raise _Stop("FAIL", "LOCK_STATE_INVALID")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Invalid batch size")
        runtime = observe_runtime(batch_size, device)
        index, inputs, anchors = _controls(release, live=True, runtime=runtime)
        decision = auth.validate_authorization(inputs, anchors, request)
        if decision.status != "PASS":
            return ExecutionResult(decision.status, ",".join(r.code for r in decision.reasons))
        approval = b1.preflight._parse_json(inputs.approval, "approval")
        if (descriptor.reference != approval["dataset_descriptor"]["reference"]
                or descriptor.revision != approval["dataset_descriptor"]["revision"]
                or descriptor.kind != index["profile"]["kind"]):
            raise ValueError("Dataset descriptor is not the reviewed descriptor")
        official = descriptor.kind == "official"
        if official:
            _official_descriptor(descriptor)
        source_hashes = _source_hashes(anchors.source_revision)
        with tempfile.TemporaryDirectory(prefix="dogbreedid-b2-inputs-") as temporary:
            snapshot = _Snapshot(Path(temporary))
            data, cfg, mapping = _settings(snapshot, inputs)
            checkpoint = snapshot.capture(b1.SOURCE_ROOT / b1.predictor.DEFAULT_CHECKPOINT, "checkpoint.pt")
            b1.predictor.verify_checkpoint_hash(checkpoint, index["profile"]["checkpoint_sha256"])
            output = _output_location(request, anchors)
            _verify_live(inputs, source_hashes, anchors.source_revision, batch_size)
            claim = auth.claim_authorization(decision)
            if claim.status != "PASS":
                # The accepted helper retains partial claims and cannot report
                # whether every write failure happened before or after allocation.
                spent_or_uncertain = claim.code in {"CLAIM_WRITE_FAILED", "AUTHORIZATION_ALREADY_CLAIMED"}
                return ExecutionResult(claim.status, claim.code, claim_consumed=spent_or_uncertain)
            claimed = True
            with _publish(output, release=release) as payload:
                # The first reader construction/call is strictly after both operations.
                _verify_live(inputs, source_hashes, anchors.source_revision, batch_size)
                reader = _make_reader(descriptor)
                csv_path, images, rows, declaration = reader.capture(snapshot, index["profile"], mapping)
                _compute(snapshot, payload, checkpoint, data, cfg, mapping, csv_path, images, rows, batch_size,
                         official=official, device=device)
                snapshot.verify()
                auth._exact(_source_hashes(anchors.source_revision), source_hashes)
                b1._json(payload / "protocol.json", auth.approved_protocol())
                b1._json(payload / "controls.json", _control_receipt(release, index, inputs))
                b1._write(payload / "claim.json", b1.preflight._read_once(claim.claim_path)["data"])
                b1._write(payload / "dataset.json", (snapshot.root / "dataset/declaration.json").read_bytes())
                b1._json(payload / "metadata.json", _metadata(official, anchors.source_revision, len(rows),
                                                              index["profile"]["checkpoint_sha256"]))
                b1._json(payload / "manifest.json", dict(format=OFFICIAL_FORMAT if official else FORMAT,
                    execution_status="complete", **_result_flags(official),
                    git_sha=anchors.source_revision, source_sha256=source_hashes, runtime=runtime, inputs=snapshot.records,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    invocation=dict(api="run_authorized_evaluation", batch_size=batch_size, device=device, reviewed_release=release.revision),
                    dataset_descriptor=asdict(descriptor),
                    run_id=request.run_id, scope=list(request.scope), output_dir=str(output),
                    output_sha256={n: b1.preflight._read_once(payload / n)["sha256"] for n in PAYLOADS if n != "manifest.json"}))
                _validate_buffers({n: b1.preflight._read_once(payload / n)["data"] for n in PAYLOADS}, release)
        return ExecutionResult("PASS", "OFFICIAL_EXECUTION_COMPLETE" if official else "SYNTHETIC_INTEGRATION_COMPLETE", True, str(output))
    except _Stop as error:
        return ExecutionResult(error.status, error.code, claimed, str(output) if output else None)
    except FileExistsError as error:
        return ExecutionResult("BLOCKED", "OUTPUT_ALREADY_EXISTS", claimed, str(output) if output else None, str(error))
    except (FileNotFoundError, b1.preflight.Blocked) as error:
        return ExecutionResult("BLOCKED", "EVIDENCE_UNAVAILABLE", claimed, str(output) if output else None, str(error))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        return ExecutionResult("FAIL", "INTEGRATION_INVALID", claimed, str(output) if output else None, str(error))
