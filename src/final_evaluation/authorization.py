"""B2.2-A: validate externally trusted records and claim one grant, never data.

There is no Official runner, data adapter or CLI here. PASS means component
validation/claim succeeded, not that an integrated Official I/O gate exists.
TrustAnchors must come from a leader-reviewed context independent of the records;
computing anchors from arbitrary incoming records does not establish authority.

Source verification reuses B1's bytes/Git checks, extended only for this package
and its tests. Frozen/evidence buffers are supplied by a trusted future adapter;
this component binds their exact bytes, not their filesystem origin or the full
meaning of historical acceptance reports. Runtime is also caller-observed input.
Integration must establish those origins and check/reserve the actual output
before any Official I/O. Dataset descriptors are opaque metadata, never paths
opened here. Claim roots must be stable, private control directories, retained
across processes/restarts. No claim or public file is ever removed on failure.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

APPROVAL_SCHEMA = "dogbreedid.configuration-approval.v1"
AUTHORIZATION_SCHEMA = "dogbreedid.official-authorization.v1"
SCOPE = ("E4", "E6", "E7", "E8")
EXTRA_SOURCE = ("src/final_evaluation/__init__.py", "src/final_evaluation/authorization.py",
                "tests/test_final_evaluation_authorization.py")
FROZEN_REFERENCES = (
    "configs/data.yaml", "configs/model_selection.yaml", "configs/experiments/e4_cutmix.yaml",
    "configs/experiments/e6_calibration.yaml", "configs/experiments/e7_selective_prediction.yaml",
    "configs/experiments/e8_robustness.yaml", "outputs/metrics/E6/temperature.json",
    "outputs/metrics/E7/threshold.json", "data/metadata/class_mapping.json",
)
CHECKPOINT_SHA256 = "73df0cc1c5b16866d4facc910f3e4d3ab481e3f12f1681a59fa7d61e3845c47b"


def approved_protocol() -> dict:
    """Return a fresh description of the unchanged, approved scientific values."""
    return dict(
        model=dict(experiment="E4", architecture="mobilenet_v2", backbone_frozen=True,
                   cutmix=True, seed=42, epoch=25, num_classes=30),
        checkpoint_sha256=CHECKPOINT_SHA256, temperature=0.80246283961825,
        threshold=0.5011672918550012, accept_operator=">=",
        preprocessing=dict(color="RGB", resize=256, center_crop=224, to_tensor=True,
                           mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        e8=dict(seed=42, stage="after_crop_before_normalization", pixel_range=[0.0, 1.0],
                conditions=list(SCOPE_E8),
                gaussian_blur=dict(kernel_size=7, sigma=1.5),
                gaussian_noise=dict(mean=0.0, std=0.05, clamp_range=[0.0, 1.0]),
                occlusion=dict(shape="square", side_fraction=0.30, fill_rgb=[0.5, 0.5, 0.5],
                               position_policy="deterministic_per_image")),
    )


SCOPE_E8 = ("clean", "gaussian_blur", "gaussian_noise", "occlusion")


@dataclass(frozen=True)
class TrustAnchors:
    """Out-of-band reviewed hashes/revision and trusted control/output locations."""

    approval_sha256: str | None
    authorization_sha256: str | None
    source_revision: str | None
    protocol_sha256: str | None
    claim_root: str
    output_root: str


@dataclass(frozen=True)
class RunRequest:
    run_id: str
    output_dir: str
    scope: tuple[str, ...] = SCOPE


@dataclass(frozen=True)
class AuthorizationInputs:
    lock_document: bytes | None
    approval: bytes | None
    authorization: bytes | None
    protocol: bytes | None
    frozen_references: Mapping[str, bytes]
    evidence: Mapping[str, bytes]
    runtime: Mapping[str, Any]


@dataclass(frozen=True)
class Reason:
    code: str
    status: str


@dataclass(frozen=True)
class _ValidatedGrant:
    authorization_id: str
    run_id: str
    output_dir: str
    scope: tuple[str, ...]
    approval_sha256: str
    authorization_sha256: str
    source_revision: str
    claim_root: str
    claim_root_identity: tuple[int, int]


@dataclass(frozen=True)
class Decision:
    status: str
    reasons: tuple[Reason, ...]
    grant: _ValidatedGrant | None = None


@dataclass(frozen=True)
class ClaimResult:
    status: str
    code: str
    claim_path: Path | None = None


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex(value, length=64):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{%d}" % length, value))


def _identifier(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value))


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _helpers():
    # Lazy import; importing the component alone loads no Torch/model/dataset.
    from scripts import run_final_evaluation as b1
    return b1


def _exact(actual, expected):
    """Reuse exact leaf checks, recursively preserving types (including bool)."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError("Object schema differs")
        for key in expected:
            _exact(actual[key], expected[key])
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError("List schema differs")
        for a, b in zip(actual, expected):
            _exact(a, b)
    else:
        _helpers().preflight._exact(actual, expected, "Authorization binding")


def _output(value):
    """Lexical output validation only: never resolve/stat an input descriptor."""
    if not _text(value) or value.replace("\\", "/").startswith("//"):
        raise ValueError("Invalid output")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("Output must be absolute")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    for part in path.parts:
        if part == path.anchor:
            continue
        if (part in {".", ".."} or ":" in part or part.rstrip(" .") != part
                or part.split(".")[0].upper() in reserved
                or part.lower().replace("-", "_") in {"test", "official", "official_test"}):
            raise ValueError("Unsafe output")
    return path.as_posix()


def capture_source_binding(approved_revision: str) -> dict[str, str]:
    """Audit current checkout against a pinned revision; no dataset references.

    B1's audit verifies its catalog against actual bytes, including hidden index
    changes. A doc-only HEAD after the approved revision is allowed only if all
    source content still binds to the approved blobs. Additional B2 paths are
    fixed here; no record may supply arbitrary Git/file paths.
    """
    b1 = _helpers()
    state = b1.preflight._git_state(b1.SOURCE_ROOT, (*b1.SOURCE_FILES, *EXTRA_SOURCE))
    if state["status"] == "FAIL":
        raise ValueError("Source differs from HEAD")
    if state["status"] != "PASS":
        raise b1.preflight.Blocked("Source is missing, untracked or Git is unavailable")
    observed = b1._source_audit("validation")
    blobs = b1._git_blobs(approved_revision, b1.SOURCE_FILES)
    hashes = {}
    for name in b1.SOURCE_FILES:
        b1._check_binding(observed["content"][name], blobs[name], name)
        hashes[name] = observed["content"][name]["sha256"]
    for name in EXTRA_SOURCE:
        captured = b1.preflight._read_once(b1.SOURCE_ROOT / name)
        try:
            expected = subprocess.check_output(
                ["git", "--no-replace-objects", "-C", str(b1.SOURCE_ROOT), "show", f"{approved_revision}:{name}"],
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise b1.preflight.Blocked("Additional source blob unavailable") from error
        hashes[name] = b1._bind_bytes(captured["data"], expected, name)["sha256"]
    return hashes


def validate_authorization(inputs: AuthorizationInputs, anchors: TrustAnchors, request: RunRequest) -> Decision:
    """Validate records/bindings only. Never access the described dataset.

    A future orchestrator must additionally validate actual runtime observation,
    checkpoint snapshot, frozen-buffer origins, output filesystem safety and
    exclusive output reservation before it may access Official inputs.
    """
    if (not isinstance(inputs, AuthorizationInputs) or not isinstance(anchors, TrustAnchors)
            or not isinstance(request, RunRequest)):
        return Decision("FAIL", (Reason("API_INPUT_INVALID", "FAIL"),))
    if not all(isinstance(value, Mapping) for value in (inputs.evidence, inputs.frozen_references, inputs.runtime)):
        return Decision("FAIL", (Reason("BINDING_INPUT_INVALID", "FAIL"),))
    if inputs.lock_document is not None and not isinstance(inputs.lock_document, bytes):
        return Decision("FAIL", (Reason("LOCK_STATE_INVALID", "FAIL"),))
    reasons = []

    def add(code, status="FAIL"):
        reason = Reason(code, status)
        if reason not in reasons:
            reasons.append(reason)

    def check(code, function):
        try:
            function()
        except (TypeError, ValueError, KeyError):
            add(code)

    def parsed(raw, label):
        if raw is None:
            add(label + "_MISSING", "BLOCKED")
            return None
        if not isinstance(raw, bytes):
            add(label + "_MALFORMED")
            return None
        try:
            return _helpers().preflight._parse_json(raw, label)
        except (ValueError, UnicodeError, TypeError):
            add(label + "_MALFORMED")
            return None

    approval = parsed(inputs.approval, "APPROVAL")
    authorization = parsed(inputs.authorization, "AUTHORIZATION")
    protocol = parsed(inputs.protocol, "PROTOCOL")
    locked = False
    if inputs.lock_document is None:
        add("LOCK_MISSING", "BLOCKED")
    else:
        try:
            states = [line for line in inputs.lock_document.decode("utf-8").splitlines() if line.startswith("**Trạng thái:**")]
            if states == ["**Trạng thái:** BẢN NHÁP — CHƯA KHÓA CẤU HÌNH"]:
                add("LOCK_DRAFT", "BLOCKED")
            elif states == ["**Trạng thái:** ĐÃ KHÓA CẤU HÌNH"]:
                locked = True
            else:
                add("LOCK_STATE_INVALID")
        except (AttributeError, UnicodeError):
            add("LOCK_STATE_INVALID")
    for name, value, length in (
        ("APPROVAL", anchors.approval_sha256, 64), ("AUTHORIZATION", anchors.authorization_sha256, 64),
        ("SOURCE", anchors.source_revision, 40), ("PROTOCOL", anchors.protocol_sha256, 64),
    ):
        if value is None:
            add(name + "_ANCHOR_MISSING", "BLOCKED")
        elif not _hex(value, length):
            add(name + "_ANCHOR_INVALID")
    for name, raw, digest in (("APPROVAL", inputs.approval, anchors.approval_sha256),
                              ("AUTHORIZATION", inputs.authorization, anchors.authorization_sha256),
                              ("PROTOCOL", inputs.protocol, anchors.protocol_sha256)):
        if isinstance(raw, bytes) and _hex(digest) and _digest(raw) != digest:
            add(name + "_HASH_MISMATCH")

    if protocol is not None:
        check("FROZEN_PARAMETERS_MISMATCH", lambda: _exact(protocol, approved_protocol()))

    def evidence(record):
        if not isinstance(record, dict) or set(record) != {"reference", "sha256"} or not _text(record["reference"]) or not _hex(record["sha256"]):
            raise ValueError("Invalid evidence reference")
        raw = inputs.evidence.get(record["reference"])
        if raw is None:
            add("EVIDENCE_MISSING", "BLOCKED")
        elif not isinstance(raw, bytes) or _digest(raw) != record["sha256"]:
            add("EVIDENCE_HASH_MISMATCH")

    valid_approval = False
    if approval is not None:
        fields = {"schema", "approval_id", "status", "approver", "approval_evidence", "source_revision",
                  "source_sha256", "protocol_sha256", "lock_sha256", "frozen_sha256", "acceptance", "runtime", "dataset_descriptor"}
        try:
            if (set(approval) != fields or approval["schema"] != APPROVAL_SCHEMA
                    or not _identifier(approval["approval_id"]) or not _text(approval["approver"])
                    or approval["status"] not in {"APPROVED", "DRAFT", "REVOKED"}
                    or not _hex(approval["source_revision"], 40) or not _hex(approval["protocol_sha256"])
                    or not _hex(approval["lock_sha256"])):
                raise ValueError("Invalid approval schema")
            if approval["status"] != "APPROVED":
                add("APPROVAL_NOT_APPROVED", "BLOCKED")
            if approval["source_revision"] != anchors.source_revision and anchors.source_revision is not None:
                add("SOURCE_REVISION_MISMATCH")
            if approval["protocol_sha256"] != anchors.protocol_sha256 and anchors.protocol_sha256 is not None:
                add("PROTOCOL_BINDING_MISMATCH")
            if locked and _digest(inputs.lock_document) != approval["lock_sha256"]:
                add("LOCK_HASH_MISMATCH")
            source = approval["source_sha256"]
            if not isinstance(source, dict) or set(source) != {*_helpers().SOURCE_FILES, *EXTRA_SOURCE} or any(not _hex(v) for v in source.values()):
                raise ValueError("Missing source coverage")
            frozen = approval["frozen_sha256"]
            if not isinstance(frozen, dict) or set(frozen) != set(FROZEN_REFERENCES) or any(not _hex(v) for v in frozen.values()):
                raise ValueError("Missing frozen reference coverage")
            if set(inputs.frozen_references) != set(frozen):
                add("FROZEN_REFERENCES_MISSING", "BLOCKED")
            for name, digest in frozen.items():
                if name not in inputs.frozen_references:
                    continue  # Coverage check above already blocks missing keys.
                raw = inputs.frozen_references[name]
                if not isinstance(raw, bytes) or _digest(raw) != digest or not raw:
                    add("FROZEN_REFERENCE_HASH_MISMATCH")
            evidence(approval["approval_evidence"])
            if not isinstance(approval["acceptance"], dict) or set(approval["acceptance"]) != {"phase_a", "phase_b1"}:
                raise ValueError("Missing acceptance references")
            for record in approval["acceptance"].values():
                evidence(record)
            runtime = approval["runtime"]
            required = {"python", "pytorch", "torchvision", "numpy", "cuda_build", "device"}
            if (not isinstance(runtime, dict) or not required <= runtime.keys()
                    or any(not _text(runtime[k]) for k in required)
                    or any(type(v) not in (str, int, bool, type(None)) for v in runtime.values())):
                raise ValueError("Missing runtime profile")
            check("RUNTIME_MISMATCH", lambda: _exact(dict(inputs.runtime), runtime))
            descriptor = approval["dataset_descriptor"]
            if (not isinstance(descriptor, dict) or set(descriptor) != {"reference", "revision", "expected_samples", "num_classes"}
                    or not _text(descriptor["reference"]) or not _hex(descriptor["revision"], 40)
                    or type(descriptor["expected_samples"]) is not int or descriptor["expected_samples"] != 1934
                    or type(descriptor["num_classes"]) is not int or descriptor["num_classes"] != 30):
                raise ValueError("Missing trusted dataset descriptor")
            valid_approval = True
        except (TypeError, ValueError, KeyError):
            add("APPROVAL_SCHEMA_INVALID")

    output = None
    valid_authorization = False
    if authorization is not None:
        fields = {"schema", "authorization_id", "status", "grantor", "grant_evidence", "approval_id",
                  "approval_sha256", "run_id", "scope", "output_dir", "claim_root", "max_runs", "automatic_retry"}
        try:
            if (set(authorization) != fields or authorization["schema"] != AUTHORIZATION_SCHEMA
                    or not _identifier(authorization["authorization_id"]) or not _identifier(authorization["run_id"])
                    or not _text(authorization["grantor"]) or not _identifier(authorization["approval_id"])
                    or not _hex(authorization["approval_sha256"])
                    or authorization["status"] not in {"GRANTED", "PENDING", "REVOKED"}
                    or type(authorization["max_runs"]) is not int or authorization["max_runs"] != 1
                    or authorization["automatic_retry"] is not False):
                raise ValueError("Invalid grant schema")
            evidence(authorization["grant_evidence"])
            if authorization["status"] != "GRANTED":
                add("AUTHORIZATION_NOT_GRANTED", "BLOCKED")
            if valid_approval and (authorization["approval_id"] != approval["approval_id"] or authorization["approval_sha256"] != _digest(inputs.approval)):
                add("APPROVAL_BINDING_MISMATCH")
            _exact(authorization["scope"], list(SCOPE))
            if not _identifier(request.run_id) or authorization["run_id"] != request.run_id or request.scope != SCOPE:
                add("RUN_SCOPE_MISMATCH")
            output = _output(authorization["output_dir"])
            root = Path(_output(anchors.output_root))
            if output != _output(request.output_dir) or Path(output) == root or not Path(output).is_relative_to(root):
                add("OUTPUT_BINDING_MISMATCH")
            if _output(authorization["claim_root"]) != _output(anchors.claim_root):
                add("CLAIM_ROOT_BINDING_MISMATCH")
            valid_authorization = True
        except (TypeError, ValueError, KeyError):
            add("AUTHORIZATION_SCHEMA_INVALID")

    identity = None
    claim_root = None
    if not reasons and valid_approval and valid_authorization:
        try:
            actual = capture_source_binding(anchors.source_revision)
            _exact(actual, approval["source_sha256"])
        except _helpers().preflight.Blocked:
            add("SOURCE_UNVERIFIED", "BLOCKED")
        except OSError:
            add("SOURCE_UNAVAILABLE", "BLOCKED")
        except (ValueError, TypeError, KeyError):
            add("SOURCE_INTEGRITY_MISMATCH")
        if not reasons:
            try:
                claim_root = _helpers()._safe_path(_output(anchors.claim_root))
                if not claim_root.is_dir():
                    add("CLAIM_ROOT_UNAVAILABLE", "BLOCKED")
                else:
                    st = claim_root.stat()
                    identity = (st.st_dev, st.st_ino)
            except (OSError, ValueError):
                add("CLAIM_ROOT_INVALID")
    status = "FAIL" if any(r.status == "FAIL" for r in reasons) else "BLOCKED" if reasons else "PASS"
    if status != "PASS":
        return Decision(status, tuple(reasons))
    grant = _ValidatedGrant(authorization["authorization_id"], request.run_id, output, request.scope,
                            anchors.approval_sha256, anchors.authorization_sha256, anchors.source_revision,
                            str(claim_root), identity)
    return Decision("PASS", (Reason("RECORDS_VALIDATED", "PASS"),), grant)


def claim_authorization(decision: Decision) -> ClaimResult:
    """Claim a validator-produced PASS exactly once. No output/dataset access.

    Validation and claim should be adjacent in the future orchestrator. This is
    not a serializable security token against callers able to forge Python state.
    Changes/revocation after validation require integration revalidation before
    data access. A claimed grant stays spent even if writing its receipt fails.
    """
    if not isinstance(decision, Decision):
        return ClaimResult("FAIL", "CLAIM_REQUIRES_VALIDATION")
    if decision.status != "PASS" or not isinstance(decision.grant, _ValidatedGrant):
        return ClaimResult(decision.status if decision.status in {"FAIL", "BLOCKED"} else "FAIL", "CLAIM_REQUIRES_VALIDATION")
    grant = decision.grant
    root = Path(grant.claim_root)
    path = root / (grant.authorization_id + ".claim.json")
    try:
        # Never follow a replacement control directory or a claim-file symlink.
        for part in (root, *root.parents):
            if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
                return ClaimResult("FAIL", "CLAIM_ROOT_CHANGED")
        st = root.stat()
        if (st.st_dev, st.st_ino) != grant.claim_root_identity:
            return ClaimResult("FAIL", "CLAIM_ROOT_CHANGED")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        with os.fdopen(os.open(path, flags, 0o600), "wb") as writer:
            receipt = dict(schema="dogbreedid.authorization-claim.v1", authorization_id=grant.authorization_id,
                           run_id=grant.run_id, output_dir=grant.output_dir, scope=list(grant.scope),
                           approval_sha256=grant.approval_sha256, authorization_sha256=grant.authorization_sha256,
                           source_revision=grant.source_revision, state="CLAIMED", automatic_retry=False)
            writer.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
            writer.flush()
            os.fsync(writer.fileno())
    except FileExistsError:
        return ClaimResult("BLOCKED", "AUTHORIZATION_ALREADY_CLAIMED", path)
    except OSError:
        # An existing empty/partial receipt also consumes the grant. No unlink.
        return ClaimResult("FAIL", "CLAIM_WRITE_FAILED", path)
    return ClaimResult("PASS", "AUTHORIZATION_CLAIMED", path)
