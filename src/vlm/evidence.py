"""Contract 06 wire records and validators for caller-supplied research evidence.

No filesystem access, scoring, or approval decisions occur in this module.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields
from datetime import date, datetime
from enum import Enum
import re
from types import MappingProxyType
from typing import Annotated, Literal, TypeVar
from uuid import uuid4

from . import contracts as c


class AuditStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ImageGroup(str, Enum):
    IN_SCOPE_DOG = "in_scope_dog"
    OUT_OF_SCOPE_DOG = "out_of_scope_dog"
    NON_DOG = "non_dog"


class QuestionCategory(str, Enum):
    VISIBLE_ATTRIBUTES = "visible_attributes"
    VISIBILITY_UNCERTAINTY = "visibility_uncertainty"
    BREED_KNOWLEDGE = "breed_knowledge"
    BREED_COMPARISON = "breed_comparison"
    GENERAL_VISUAL_OBSERVATION = "general_visual_observation"
    OUT_OF_SCOPE_HANDLING = "out_of_scope_handling"
    NON_DOG_HANDLING = "non_dog_handling"


class Answerability(str, Enum):
    ANSWERABLE = "answerable"
    ABSTENTION_EXPECTED = "abstention_expected"


class PredictionStatus(str, Enum):
    OK = "OK"
    FAIL = "FAIL"
    ABSTAIN = "ABSTAIN"


class ParseStatus(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    OK = "ok"
    INVALID_JSON = "invalid_json"


class SchemaValidationStatus(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    VALID = "valid"
    INVALID = "invalid"


class RunPhase(str, Enum):
    PILOT = "pilot"
    FORMAL = "formal"
    EXPLORATORY = "exploratory"


class RunExecutionStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


def _utc(value: str) -> datetime:
    c._require(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)", value)
               is not None, "expected ISO 8601 UTC timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise c.ContractError("invalid UTC timestamp") from exc


def _date(value: str) -> None:
    c._require(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None, "expected ISO 8601 date")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise c.ContractError("invalid collection date") from exc


_Utc = Annotated[str, _utc]
_Date = Annotated[str, _date]
_Missing = c._Missing
MISSING = c.MISSING
_Slot = tuple[str, str]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Envelope(c._WireRecord):
    schema_version: c._SchemaVersion = c.SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class PilotImage(_Envelope):
    image_id: str
    sample_id: str
    image_ref: str
    image_digest: c.Sha256
    source_ref: str
    usage_permission: str
    audit_status: AuditStatus
    perceptual_hash: str | _Missing = MISSING
    record_type: Literal["pilot_image"] = "pilot_image"


@dataclass(frozen=True, slots=True, kw_only=True)
class VlmEvalImage(_Envelope):
    image_id: str
    sample_id: str
    image_ref: str
    source_url: str
    usage_permission: str
    collection_date: _Date
    class_id: c.ClassId | None
    group: ImageGroup
    image_digest: c.Sha256
    perceptual_hash: str
    audit_status: AuditStatus
    question_ids: tuple[str, ...]
    annotation_provenance: str
    record_type: Literal["vlm_eval_image"] = "vlm_eval_image"

    def _validate(self) -> None:
        c._require((self.class_id is not None) == (self.group is ImageGroup.IN_SCOPE_DOG),
                   "class_id is present only for in_scope_dog")
        c._unique(self.question_ids, "image question IDs")


ImageRecord = PilotImage | VlmEvalImage


@dataclass(frozen=True, slots=True, kw_only=True)
class Question(_Envelope):
    image_id: str
    question_id: str
    sample_id: str
    question: str
    category: QuestionCategory
    record_type: Literal["question"] = "question"


@dataclass(frozen=True, slots=True, kw_only=True)
class AnnotationProvenance(c._WireRecord):
    annotator_id: str
    reviewed_at: _Utc
    revision: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Annotation(_Envelope):
    image_id: str
    question_id: str
    rubric_version: str
    target_attributes: tuple[c.AttributeObservation, ...]
    expected_answer: str | None
    answerability: Answerability
    source_required: bool
    provenance: AnnotationProvenance
    expected_source_ids: tuple[str, ...] | _Missing = MISSING
    record_type: Literal["annotation"] = "annotation"

    def _validate(self) -> None:
        c._unique_observations(self.target_attributes)
        if self.expected_source_ids is not MISSING:
            c._unique(self.expected_source_ids, "expected source IDs")


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassificationPrediction(_Envelope):
    eval_ref: c.EvalRef
    image_id: str
    image_digest: c.Sha256
    prediction_status: PredictionStatus
    model_result: c.ZeroShotResult | None
    reason_code: c.ErrorCode | _Missing = MISSING
    record_type: Literal["classification_prediction"] = "classification_prediction"

    def _validate(self) -> None:
        c._require(self.eval_ref.experiment_id in c._ZERO_SHOT_EXPERIMENTS, "classification requires E12/E13")
        ok = self.prediction_status is PredictionStatus.OK
        if ok:
            c._require(self.model_result is not None and self.reason_code is MISSING,
                       "OK prediction requires a result and excludes failure reason")
        if self.model_result is not None:
            c._require((self.model_result.status is c.BranchOutcome.OK) == ok, "prediction and result outcomes disagree")
            c._require((self.model_result.eval_ref, self.model_result.image_digest) == (self.eval_ref, self.image_digest),
                       "classification result run or digest mismatch")
            if not ok and self.reason_code is not MISSING:
                c._require(self.reason_code is self.model_result.error.code, "prediction and result error codes disagree")


@dataclass(frozen=True, slots=True, kw_only=True)
class InvocationRef(c._WireRecord):
    eval_ref: c.EvalRef
    invocation_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class AttributeInputRef(c._WireRecord):
    producer: InvocationRef
    attributes_sha256: c.Sha256


@dataclass(frozen=True, slots=True, kw_only=True)
class VqaOutputTrace(_Envelope):
    eval_ref: c.EvalRef
    image_id: str
    question_id: str
    response_id: str
    raw_model_output: c._RawText | None
    validated_model_output: c.AssistantResult | None
    controller_generated_response: str | None
    policy_blocked: bool
    retrieval_status: c.RetrievalStatus
    invocation_refs: tuple[InvocationRef, ...]
    reason_code: c.ErrorCode | _Missing = MISSING
    record_type: Literal["vqa_output_trace"] = "vqa_output_trace"

    def _validate(self) -> None:
        experiment = self.eval_ref.experiment_id
        c._require(experiment in c._ASSISTANT_EXPERIMENTS or experiment == "E15", "VQA trace requires E14/E15")
        c._unique(self.invocation_refs, "trace invocation references")
        result = self.validated_model_output
        if self.policy_blocked:
            c._require(self.reason_code is c.ErrorCode.POLICY_BLOCKED and self.raw_model_output is None
                       and result is None, "blocked trace requires POLICY_BLOCKED and null VQA output")
        elif result is None:
            c._require(self.reason_code is not MISSING and self.reason_code is not c.ErrorCode.POLICY_BLOCKED,
                       "failed trace requires a non-policy reason")
            c._require(self.invocation_refs or self.raw_model_output is None or experiment == "E15",
                       "E14 raw answer requires an invocation")
        else:
            c._require(result.status is c.BranchOutcome.OK and self.reason_code is MISSING
                       and self.raw_model_output is not None
                       and result.generated.disposition is not c.Disposition.NOT_APPLICABLE,
                       "validated trace requires successful VQA model output and raw text")
            expected_ref = MISSING if experiment == "E15" else self.eval_ref
            c._require(result.eval_ref == expected_ref and result.context_provenance.retrieval_status is self.retrieval_status,
                       "trace and result run or retrieval status mismatch")
            c._require(self.invocation_refs or experiment == "E15", "E14 model result requires an invocation")


@dataclass(frozen=True, slots=True, kw_only=True)
class E14ModelInvocation(_Envelope):
    eval_ref: c.EvalRef
    invocation_id: str
    stage_type: c.Operation
    image_id: str
    question_id: str | None
    model: c.ModelRef
    profile_id: str
    raw_output: c._RawText | None
    parse_status: ParseStatus
    schema_validation_status: SchemaValidationStatus
    terminal_outcome: c.BranchOutcome
    error_code: c.ErrorCode | None
    latency_ms: c._NonNegativeNumber
    context_provenance: c.ContextProvenance
    validated_stage_output: tuple[c.AttributeObservation, ...] | c.GeneratedContent | None
    produced_attributes_sha256: c.Sha256 | None
    consumed_attributes: AttributeInputRef | None
    record_type: Literal["e14_model_invocation"] = "e14_model_invocation"

    def _validate(self) -> None:
        c._require(self.eval_ref.experiment_id in c._ASSISTANT_EXPERIMENTS, "stage invocation requires E14")
        attributes_only = self.stage_type is c.Operation.ATTRIBUTES
        c._require(attributes_only or self.question_id is not None, "VQA stage requires question_id")
        c._require(self.consumed_attributes is None or self.stage_type is c.Operation.VQA,
                   "only a VQA stage can consume a produced attribute snapshot")
        if self.parse_status is not ParseStatus.NOT_ATTEMPTED:
            c._require(self.raw_output is not None, "attempted parse requires raw output")
        if self.parse_status is not ParseStatus.OK:
            c._require(self.schema_validation_status is SchemaValidationStatus.NOT_ATTEMPTED,
                       "schema validation requires parse ok")
        if self.terminal_outcome is c.BranchOutcome.OK:
            c._require(self.parse_status is ParseStatus.OK and self.schema_validation_status is SchemaValidationStatus.VALID
                       and self.error_code is None, "ok invocation requires valid parse/schema and null error")
            output = self.validated_stage_output
            if attributes_only:
                c._require(type(output) is tuple, "attributes stage requires an observation array")
                c._unique_observations(output)
            else:
                c._require(type(output) is c.GeneratedContent and output.disposition is not c.Disposition.NOT_APPLICABLE,
                           "VQA stage requires a VQA GeneratedContent")
        else:
            c._require(self.error_code is not None and self.validated_stage_output is None,
                       "failed invocation requires an error and null validated output")
        if self.produced_attributes_sha256 is not None:
            c._require(self.terminal_outcome is c.BranchOutcome.OK and self.stage_type is not c.Operation.VQA,
                       "only successful extraction can publish an attribute snapshot")

    @property
    def ref(self) -> InvocationRef:
        return InvocationRef(eval_ref=self.eval_ref, invocation_id=self.invocation_id)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseIndex(_Envelope):
    response_id: str
    eval_ref: c.EvalRef
    image_id: str
    question_id: str
    record_type: Literal["response_index"] = "response_index"

    def _validate(self) -> None:
        c._require(self.eval_ref.experiment_id in c._ASSISTANT_EXPERIMENTS or self.eval_ref.experiment_id == "E15",
                   "response index requires E14/E15")


@dataclass(frozen=True, slots=True, kw_only=True)
class BlindedResponse(_Envelope):
    response_id: str
    review_image_ref: str
    question: str
    content_for_review: c.GeneratedContent | None
    rubric_version: str
    record_type: Literal["blinded_response"] = "blinded_response"


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewerLabel(_Envelope):
    reviewer_label_id: str
    response_id: str
    reviewer_id: str
    rubric_version: str
    labels: Mapping[str, c._JsonValue]
    submitted_at: _Utc
    notes: str | _Missing = MISSING
    record_type: Literal["reviewer_label"] = "reviewer_label"


@dataclass(frozen=True, slots=True, kw_only=True)
class AdjudicatedLabel(_Envelope):
    response_id: str
    rubric_version: str
    reviewer_label_refs: tuple[str, ...]
    final_labels: Mapping[str, c._JsonValue]
    resolved_by: str
    resolution_note: str
    record_type: Literal["adjudicated_label"] = "adjudicated_label"

    def _validate(self) -> None:
        c._require(len(self.reviewer_label_refs) == 2, "adjudication requires two label references")
        c._unique(self.reviewer_label_refs, "adjudication label references")


@dataclass(frozen=True, slots=True, kw_only=True)
class ConfidenceInterval(c._WireRecord):
    lower: c._Number
    upper: c._Number
    level: c._Number

    def _validate(self) -> None:
        c._require(self.lower <= self.upper and 0 < self.level < 1, "invalid confidence interval bounds or level")


@dataclass(frozen=True, slots=True, kw_only=True)
class Metric(_Envelope):
    eval_ref: c.EvalRef
    metric_id: str
    scope: str
    value: c._Number | None
    unit: str
    n_evaluated: c._NonNegativeInt
    numerator: c._Number | _Missing = MISSING
    denominator: c._NonNegativeNumber | _Missing = MISSING
    ci: ConfidenceInterval | _Missing = MISSING
    na_reason: str | _Missing = MISSING
    record_type: Literal["metric"] = "metric"

    def _validate(self) -> None:
        c._require((self.value is None) == (self.na_reason is not MISSING), "metric null value requires na_reason")
        if self.value is None:
            c._require(self.ci is MISSING, "unavailable metric cannot carry a confidence interval")
        if self.denominator == 0:
            c._require(self.value is None, "zero denominator requires an unavailable metric")


@dataclass(frozen=True, slots=True, kw_only=True)
class Benchmark(_Envelope):
    eval_ref: c.EvalRef
    profile_id: str
    hardware_ref: str
    measurement: str
    scenario: str
    warmup_count: c._NonNegativeInt
    measured_count: c._NonNegativeInt
    unit: str
    values: tuple[c._Number, ...]
    failure_count: c._NonNegativeInt
    record_type: Literal["benchmark"] = "benchmark"

    def _validate(self) -> None:
        c._require(len(self.values) + self.failure_count == self.measured_count, "benchmark measurement count mismatch")


@dataclass(frozen=True, slots=True, kw_only=True)
class RunProfileRef(c._WireRecord):
    profile_id: str
    model: c.ModelRef


@dataclass(frozen=True, slots=True, kw_only=True)
class RunProvenance(_Envelope):
    eval_ref: c.EvalRef
    phase: RunPhase
    execution_status: RunExecutionStatus
    git_sha: c._CommitSha
    protocol_revision: str
    lock_ref: str | None
    config_sha256: c.Sha256
    environment_ref: str
    mapping_sha256: c.Sha256
    manifest_sha256: c.Sha256
    profiles: tuple[RunProfileRef, ...]
    artifact_hashes: Mapping[str, c.Sha256]
    started_at: _Utc
    finished_at: _Utc | None
    prompt_bank_sha256: c.Sha256 | _Missing = MISSING
    kb_sha256: c.Sha256 | _Missing = MISSING
    rubric_version: str | _Missing = MISSING
    paired_group_id: str | _Missing = MISSING
    record_type: Literal["run_provenance"] = "run_provenance"

    def _validate(self) -> None:
        experiment = self.eval_ref.experiment_id
        c._require(self.phase.value == self.eval_ref.run_id.split("_")[1], "run phase does not match run ID")
        c._require((self.finished_at is None) == (self.execution_status is RunExecutionStatus.RUNNING),
                   "finished_at is null only for RUNNING")
        if self.finished_at is not None:
            c._require(_utc(self.started_at) <= _utc(self.finished_at), "run started after it finished")
        c._unique(tuple(profile.profile_id for profile in self.profiles), "run profile IDs")
        if self.phase is RunPhase.FORMAL:
            c._require(self.lock_ref is not None, "formal run requires lock_ref")
            if experiment in c._ZERO_SHOT_EXPERIMENTS:
                c._require(self.prompt_bank_sha256 is not MISSING, "formal classifier requires a prompt bank pin")
            if experiment in c._ASSISTANT_EXPERIMENTS:
                c._require(self.rubric_version is not MISSING, "formal E14 requires a rubric version")
            if experiment == "E14-B":
                c._require(self.kb_sha256 is not MISSING, "formal E14-B requires a KB pin")
        if experiment in c._ASSISTANT_EXPERIMENTS:
            c._require(self.prompt_bank_sha256 is MISSING, "E14 has no classifier prompt bank")
        if experiment in ("E12-A", "E13-A", "E14-A"):
            c._require(self.kb_sha256 is MISSING, "this experiment has no KB pin")


EvidenceRecord = (
    PilotImage | VlmEvalImage | Question | Annotation | ClassificationPrediction | VqaOutputTrace
    | E14ModelInvocation | ResponseIndex | BlindedResponse | ReviewerLabel | AdjudicatedLabel
    | Metric | Benchmark | RunProvenance
)
_RECORD_TYPES = MappingProxyType({
    "pilot_image": PilotImage, "vlm_eval_image": VlmEvalImage, "question": Question, "annotation": Annotation,
    "classification_prediction": ClassificationPrediction, "vqa_output_trace": VqaOutputTrace,
    "e14_model_invocation": E14ModelInvocation, "response_index": ResponseIndex, "blinded_response": BlindedResponse,
    "reviewer_label": ReviewerLabel, "adjudicated_label": AdjudicatedLabel, "metric": Metric, "benchmark": Benchmark,
    "run_provenance": RunProvenance,
})


def decode_evidence_record(raw: bytes) -> EvidenceRecord:
    """Decode one strict JSON record. Run and corpus binding are separate."""
    obj = c.decode_json(raw)
    c._require(type(obj) is dict and type(obj.get("record_type")) is str, "evidence requires record_type")
    record_type = _RECORD_TYPES.get(obj["record_type"])
    c._require(record_type is not None, "unknown evidence record_type")
    return c._wire_record(record_type, obj)


def validate_evidence_record(record: EvidenceRecord) -> None:
    """Check the exact record type. Field invariants are checked on construction."""
    c._require(type(record) in _RECORD_TYPES.values(), "expected a validated evidence record")


# The caller supplies the allowed references and independent byte pins.

@dataclass(frozen=True, slots=True, kw_only=True)
class UsageGrant(c._Record):
    source: str
    usage_permission: str


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationScope(c._Record):
    scope_ref: str
    phase: RunPhase
    image_grants: Mapping[str, UsageGrant]


@dataclass(frozen=True, slots=True, kw_only=True)
class ApprovedManifest(c._Record):
    """In-memory manifest and companion records checked against caller pins."""

    scope: EvaluationScope
    manifest_sha256: c.Sha256
    images: Mapping[str, ImageRecord]
    questions: Mapping[_Slot, Question]
    annotations: Mapping[_Slot, Annotation]
    questions_sha256: c.Sha256 | None
    annotations_sha256: c.Sha256 | None


_T = TypeVar("_T")
_K = TypeVar("_K")


def _index(records: Iterable[_T], key: Callable[[_T], _K], kind: str) -> dict[_K, _T]:
    result = {}
    for record in records:
        record_id = key(record)
        c._require(record_id not in result, f"duplicate {kind}", c.ErrorCode.INVALID_SNAPSHOT)
        result[record_id] = record
    return result


def _slot(record: Question | Annotation | ResponseIndex | VqaOutputTrace) -> _Slot:
    return record.image_id, record.question_id


def _decode_rows(raw: bytes, expected_sha256: c.Sha256) -> tuple[EvidenceRecord, ...]:
    c.verify_sha256(raw, expected_sha256)
    lines = raw.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    return tuple(decode_evidence_record(line) for line in lines)


def _pinned_rows(raw: bytes | None, digest: c.Sha256 | None, record_type: type[_T]) -> tuple[_T, ...]:
    c._require((raw is None) == (digest is None), "companion artifact requires bytes and hash")
    if raw is None:
        return ()
    rows = _decode_rows(raw, digest)
    c._require(all(type(row) is record_type for row in rows), f"expected {record_type.__name__} records")
    return rows


def validate_manifest(
    raw: bytes,
    *,
    expected_sha256: c.Sha256,
    scope: EvaluationScope,
    questions_raw: bytes | None = None,
    questions_sha256: c.Sha256 | None = None,
    annotations_raw: bytes | None = None,
    annotations_sha256: c.Sha256 | None = None,
) -> ApprovedManifest:
    """Bind evidence NDJSON bytes and companion joins to an already approved scope."""
    c._expect_record(scope, EvaluationScope)
    rows = _decode_rows(raw, expected_sha256)
    c._require(rows and all(type(row) in (PilotImage, VlmEvalImage) for row in rows),
               "manifest requires image records", c.ErrorCode.INVALID_SNAPSHOT)
    c._require(len({type(row) for row in rows}) == 1, "manifest cannot mix pilot and evaluation image formats")
    images = _index(rows, lambda row: row.image_id, "image ID")
    _index(rows, lambda row: row.sample_id, "manifest sample ID")
    _index(rows, lambda row: row.image_digest, "manifest image digest")
    slots: set[_Slot] = set()
    for row in rows:
        c._require(row.audit_status is AuditStatus.APPROVED, "image audit is not approved", c.ErrorCode.POLICY_BLOCKED)
        grant = scope.image_grants.get(row.image_ref)
        source = row.source_ref if type(row) is PilotImage else row.source_url
        c._require(grant is not None and (grant.source, grant.usage_permission) == (source, row.usage_permission),
                   "image scope or usage permission mismatch", c.ErrorCode.POLICY_BLOCKED)
        if type(row) is PilotImage:
            c._require(scope.phase is RunPhase.PILOT, "pilot image requires pilot scope", c.ErrorCode.POLICY_BLOCKED)
        else:
            slots.update((row.image_id, question_id) for question_id in row.question_ids)
    question_rows = _pinned_rows(questions_raw, questions_sha256, Question)
    questions = _index(question_rows, _slot, "image/question pair")
    _index(question_rows, lambda row: row.sample_id, "question sample ID")
    c._require(questions.keys() == slots, "question records must match manifest question_ids",
               c.ErrorCode.INVALID_SNAPSHOT)
    annotations = _index(_pinned_rows(annotations_raw, annotations_sha256, Annotation), _slot, "annotation slot")
    if annotations_raw is not None:
        c._require(annotations.keys() == slots, "annotation records must match manifest question_ids",
                   c.ErrorCode.INVALID_SNAPSHOT)
    return ApprovedManifest(scope=scope, manifest_sha256=expected_sha256, images=images, questions=questions,
                            annotations=annotations, questions_sha256=questions_sha256,
                            annotations_sha256=annotations_sha256)


def validate_disjoint_manifests(pilot: ApprovedManifest, formal: ApprovedManifest) -> None:
    """Check exact overlap and matching supplied perceptual hashes; no image access."""
    c._expect_record(pilot, ApprovedManifest)
    c._expect_record(formal, ApprovedManifest)
    c._require(pilot.scope.phase is RunPhase.PILOT and formal.scope.phase is RunPhase.FORMAL,
               "disjointness requires a pilot and formal manifest")
    c._require(not (pilot.images.keys() & formal.images.keys()), "pilot and formal share image IDs")
    for field in ("sample_id", "image_digest", "perceptual_hash"):
        pilot_values = {getattr(row, field) for row in pilot.images.values()} - {MISSING}
        formal_values = {getattr(row, field) for row in formal.images.values()} - {MISSING}
        c._require(not (pilot_values & formal_values), f"pilot/formal overlap flagged by {field}",
                   c.ErrorCode.INVALID_SNAPSHOT)


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeEvalInput(c._Record):
    image: c.ValidatedRgb
    operation: c.Operation
    question: str | None


def project_runtime_input(
    manifest: ApprovedManifest,
    *,
    image_id: str,
    image: c.ValidatedRgb,
    operation: c.Operation,
    question_id: str | None = None,
) -> RuntimeEvalInput:
    """Project an already decoded approved image and locked question to model inputs."""
    c._expect_record(manifest, ApprovedManifest)
    c._expect_record(image, c.ValidatedRgb)
    item = manifest.images.get(image_id)
    c._require(item is not None and item.image_digest == image.image_digest,
               "RGB does not match the approved image slot", c.ErrorCode.HASH_MISMATCH)
    question = None
    if operation is c.Operation.ATTRIBUTES:
        c._require(question_id is None or (image_id, question_id) in manifest.questions, "unknown extraction slot")
    else:
        row = manifest.questions.get((image_id, question_id))
        c._require(row is not None, "unknown VQA question slot")
        question = row.question
    return RuntimeEvalInput(image=image, operation=operation, question=question)


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricDefinition(c._Record):
    unit: str
    requires_ratio: bool = False
    requires_ci: bool = False
    min_value: c._Number | None = None
    max_value: c._Number | None = None
    ci_level: c._Number | None = None
    max_evaluated: c._NonNegativeInt | None = None

    def _validate(self) -> None:
        if self.min_value is not None and self.max_value is not None:
            c._require(self.min_value <= self.max_value, "metric range is inverted")
        if self.ci_level is not None:
            c._require(0 < self.ci_level < 1, "invalid metric interval level")


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkScope(c._Record):
    profile_id: str
    hardware_ref: str
    measurement: str
    scenario: str
    unit: str


def validate_metric(metric: Metric, *, definitions: Mapping[_Slot, MetricDefinition]) -> None:
    c._expect_record(metric, Metric)
    definition = definitions.get((metric.metric_id, metric.scope))
    c._require(type(definition) is MetricDefinition and metric.unit == definition.unit, "metric ID, scope or unit mismatch")
    if definition.max_evaluated is not None:
        c._require(metric.n_evaluated <= definition.max_evaluated, "metric count exceeds approved scope")
    has_counts = metric.numerator is not MISSING or metric.denominator is not MISSING
    if definition.requires_ratio and (metric.value is not None or has_counts):
        c._require(metric.numerator is not MISSING and metric.denominator is not MISSING,
                   "metric definition requires numerator and denominator")
        c._require(0 <= metric.numerator <= metric.denominator, "metric proportion counts are inconsistent")
    if metric.value is not None:
        c._require(not definition.requires_ci or metric.ci is not MISSING, "metric definition requires confidence interval")
        values = (metric.value,) if metric.ci is MISSING else (metric.value, metric.ci.lower, metric.ci.upper)
        c._require(all((definition.min_value is None or item >= definition.min_value)
                       and (definition.max_value is None or item <= definition.max_value) for item in values),
                   "metric or CI exceeds approved range")
        if metric.ci is not MISSING and definition.ci_level is not None:
            c._require(metric.ci.level == definition.ci_level, "metric CI level does not match its definition")


def validate_benchmark(benchmark: Benchmark, *, scopes: Iterable[BenchmarkScope]) -> None:
    c._expect_record(benchmark, Benchmark)
    key = BenchmarkScope(profile_id=benchmark.profile_id, hardware_ref=benchmark.hardware_ref,
                         measurement=benchmark.measurement, scenario=benchmark.scenario, unit=benchmark.unit)
    c._require(key in scopes, "benchmark profile/hardware/measurement/scenario/unit is outside scope")


@dataclass(frozen=True, slots=True, kw_only=True)
class TrustedRun(c._Record):
    """Pins selected independently by the caller for this run.

    provenance carries the expected pins. Execution status and times are observations.
    approved_lock_ref comes from the caller's approval authority.
    """

    provenance: RunProvenance
    manifest: ApprovedManifest
    approved_lock_ref: str | None
    knowledge: c.KnowledgeSnapshot | None = None


_OBSERVED_RUN_FIELDS = frozenset({"execution_status", "started_at", "finished_at"})


def _validate_formal_e14_inputs(manifest: ApprovedManifest) -> None:
    code = c.ErrorCode.INVALID_SNAPSHOT
    c._require(manifest.images and all(type(image) is VlmEvalImage for image in manifest.images.values()),
               "formal E14 requires VLM evaluation image records", code)
    c._require(bool(manifest.questions), "formal E14 requires at least one question slot", code)
    c._require(all(image.question_ids for image in manifest.images.values()),
               "formal E14 requires question slots for every image", code)
    slots = {(image.image_id, question_id) for image in manifest.images.values() for question_id in image.question_ids}
    c._require(manifest.questions.keys() == slots, "formal E14 question slots must match the image manifest", code)
    c._require(manifest.questions_sha256 is not None, "formal E14 requires a questions artifact hash", code)
    c._require(manifest.annotations_sha256 is not None, "formal E14 requires an annotations artifact hash", code)
    c._require(manifest.annotations.keys() == slots, "formal E14 annotation slots must match the question slots", code)


def validate_run_provenance(run: RunProvenance, *, trusted: TrustedRun) -> None:
    c._expect_record(run, RunProvenance)
    c._expect_record(trusted, TrustedRun)
    if run.phase is RunPhase.FORMAL:
        c._require(trusted.approved_lock_ref is not None and run.lock_ref == trusted.approved_lock_ref,
                   "formal lock is unknown or unapproved", c.ErrorCode.POLICY_BLOCKED)
    for field in fields(run):
        if field.name not in _OBSERVED_RUN_FIELDS:
            c._require(getattr(run, field.name) == getattr(trusted.provenance, field.name),
                       f"run {field.name} differs from trusted provenance", c.ErrorCode.INVALID_SNAPSHOT)
    manifest = trusted.manifest
    c._require(manifest.scope.phase is run.phase and run.manifest_sha256 == manifest.manifest_sha256,
               "run manifest hash or approved phase mismatch", c.ErrorCode.INVALID_SNAPSHOT)
    if run.phase is RunPhase.FORMAL and run.eval_ref.experiment_id in c._ASSISTANT_EXPERIMENTS:
        _validate_formal_e14_inputs(manifest)
    if trusted.knowledge is not None:
        c._require(run.kb_sha256 == trusted.knowledge.kb_sha256
                   and run.mapping_sha256 == trusted.knowledge.data.mapping_sha256,
                   "run KB or mapping differs from approved snapshot", c.ErrorCode.HASH_MISMATCH)
    if run.eval_ref.experiment_id == "E14-B" and run.kb_sha256 is not MISSING:
        c._require(trusted.knowledge is not None, "E14-B KB pin requires an approved snapshot",
                   c.ErrorCode.INVALID_SNAPSHOT)
    if run.rubric_version is not MISSING:
        c._require(all(row.rubric_version == run.rubric_version for row in manifest.annotations.values()),
                   "manifest annotation rubric differs from run rubric", c.ErrorCode.INVALID_SNAPSHOT)


def verify_run_artifacts(run: RunProvenance, artifacts: Mapping[str, bytes]) -> None:
    """Verify in-memory artifact bytes against an already bound run."""
    c._expect_record(run, RunProvenance)
    c._require(artifacts.keys() == run.artifact_hashes.keys(), "artifact references differ from the run inventory",
               c.ErrorCode.INVALID_SNAPSHOT)
    for reference, expected in run.artifact_hashes.items():
        c.verify_sha256(artifacts[reference], expected)


def _bind_model(profile_id: str, model: c.ModelRef, run: RunProvenance) -> None:
    c._require(any(profile.profile_id == profile_id and profile.model == model for profile in run.profiles),
               "result model/profile absent from trusted run", c.ErrorCode.INVALID_SNAPSHOT)


def _bind_image(image_id: str, manifest: ApprovedManifest) -> ImageRecord:
    item = manifest.images.get(image_id)
    c._require(item is not None, "image ID is outside approved cohort", c.ErrorCode.INVALID_SNAPSHOT)
    return item


def _bind_run_record(record: EvidenceRecord, run: RunProvenance, trusted: TrustedRun) -> None:
    c._require(getattr(record, "eval_ref", None) == run.eval_ref, "evidence run reference mismatch")
    manifest = trusted.manifest
    if type(record) is ClassificationPrediction:
        image = _bind_image(record.image_id, manifest)
        c._require(record.image_digest == image.image_digest, "prediction image digest mismatch", c.ErrorCode.HASH_MISMATCH)
        if record.model_result is not None:
            result = record.model_result
            _bind_model(result.profile_id, result.model, run)
            c._require((result.mapping_sha256, result.prompt_bank_sha256) == (run.mapping_sha256, run.prompt_bank_sha256),
                       "prediction mapping or prompt pin mismatch", c.ErrorCode.HASH_MISMATCH)
    elif type(record) in (VqaOutputTrace, ResponseIndex, E14ModelInvocation):
        image = _bind_image(record.image_id, manifest)
        if record.question_id is not None:
            c._require(_slot(record) in manifest.questions, "question slot is outside approved cohort")
        if type(record) is E14ModelInvocation:
            _bind_model(record.profile_id, record.model, run)
            _bind_e14_context(record.context_provenance, trusted, extraction=record.stage_type is c.Operation.ATTRIBUTES)
        elif type(record) is VqaOutputTrace:
            if run.eval_ref.experiment_id == "E14-A":
                c._require(record.retrieval_status is c.RetrievalStatus.NOT_APPLICABLE, "E14-A trace cannot report KB retrieval")
            if record.validated_model_output is not None:
                result = record.validated_model_output
                _bind_model(result.profile_id, result.model, run)
                c._require(result.image_digest == image.image_digest, "trace result image digest mismatch",
                           c.ErrorCode.HASH_MISMATCH)
                if run.eval_ref.experiment_id in c._ASSISTANT_EXPERIMENTS:
                    _bind_e14_context(result.context_provenance, trusted, extraction=False)
    elif type(record) is Benchmark:
        c._require(any(profile.profile_id == record.profile_id for profile in run.profiles),
                   "benchmark profile is outside run")


def validate_evidence_run_binding(
    record: EvidenceRecord, *, run: RunProvenance, trusted: TrustedRun,
) -> None:
    """Bind one run-scoped record. Stage joins and metric/benchmark definitions have separate validators."""
    validate_evidence_record(record)
    validate_run_provenance(run, trusted=trusted)
    if type(record) is RunProvenance:
        c._require(record == run, "run evidence does not match the observed run")
    else:
        _bind_run_record(record, run, trusted)


def validate_classification_cohort(
    predictions: Iterable[ClassificationPrediction], *, run: RunProvenance, trusted: TrustedRun,
) -> None:
    validate_run_provenance(run, trusted=trusted)
    c._require(run.eval_ref.experiment_id in c._ZERO_SHOT_EXPERIMENTS, "classification cohort requires E12/E13")
    rows = _index(predictions, lambda row: row.image_id, "classification prediction image ID")
    for row in rows.values():
        c._expect_record(row, ClassificationPrediction)
        _bind_run_record(row, run, trusted)
    if run.execution_status is RunExecutionStatus.COMPLETED:
        c._require(rows.keys() == trusted.manifest.images.keys(), "completed predictions must cover the approved cohort",
                   c.ErrorCode.INVALID_SNAPSHOT)


# Actual E14 calls are supplied independently of the evidence rows.

@dataclass(frozen=True, slots=True, kw_only=True)
class InvocationCapture(c._Record):
    """Trusted in-memory capture for a dispatched model call."""

    request: c.AssistantRequest
    profile: c.ApprovedProfile
    policy_mode: c.PolicyMode
    raw_output: c._RawText | None
    terminal_outcome: c.BranchOutcome
    error_code: c.ErrorCode | None
    mapping: c.CanonicalMapping | None = None
    retrieval: c.ValidatedRetrieval | None = None
    published_attributes: bytes | None = None
    shared_inputs_sha256: c.Sha256 | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class E14PairLock(c._Record):
    """Caller-approved pair identity, shared call inputs and stage design."""

    paired_group_id: str
    a: c.EvalRef
    b: c.EvalRef
    answer_stage: c.Operation
    shared_extraction: bool
    shared_inputs: Mapping[_Slot, c.Sha256]

    def _validate(self) -> None:
        c._require((self.a.experiment_id, self.b.experiment_id) == ("E14-A", "E14-B"),
                   "paired lock requires E14-A and E14-B")
        c._require(self.answer_stage in (c.Operation.VQA, c.Operation.ATTRIBUTES_AND_VQA),
                   "paired lock requires a VQA answer stage")
        if self.shared_extraction:
            c._require(self.answer_stage is c.Operation.VQA, "shared extraction requires separate stages")


def _bind_e14_context(context: c.ContextProvenance, trusted: TrustedRun, *, extraction: bool) -> None:
    if trusted.provenance.eval_ref.experiment_id == "E14-A":
        c._require(context.kb_sha256 is None, "E14-A model evidence cannot use KB")
        return
    if extraction and context.kb_sha256 is None:
        return
    kb = trusted.knowledge
    c._require(kb is not None and context.kb_version == kb.data.kb_version and context.kb_sha256 == kb.kb_sha256,
               "E14-B model evidence requires the approved KB version and hash", c.ErrorCode.HASH_MISMATCH)


def _attributes(stage: E14ModelInvocation) -> tuple[c.AttributeObservation, ...]:
    output = stage.validated_stage_output
    return output if type(output) is tuple else output.attributes


def _validate_capture(
    stage: E14ModelInvocation,
    capture: InvocationCapture,
    trusted: TrustedRun,
    serialize_attributes: Callable[[tuple[c.AttributeObservation, ...]], bytes] | None,
) -> None:
    c._expect_record(capture, InvocationCapture)
    request = capture.request
    manifest = trusted.manifest
    image = manifest.images[stage.image_id]
    c._require((request.eval_ref, request.operation, request.profile_id, request.image_digest)
               == (stage.eval_ref, stage.stage_type, stage.profile_id, image.image_digest),
               "invocation does not match captured request, stage or image")
    c._require(stage.model == capture.profile.model and stage.context_provenance == request.context_provenance,
               "invocation model or context differs from captured request")
    c._require((stage.raw_output, stage.terminal_outcome, stage.error_code)
               == (capture.raw_output, capture.terminal_outcome, capture.error_code),
               "invocation raw output, outcome or error differs from captured call")
    if request.operation is not c.Operation.ATTRIBUTES:
        c._require(request.question == manifest.questions[_slot(stage)].question, "captured VQA question mismatch")
        c._require((stage.consumed_attributes is None) == (request.intermediate_attributes is None),
                   "captured intermediate attributes require an explicit producer reference")
    if capture.mapping is not None:
        c._require(capture.mapping.mapping_sha256 == trusted.provenance.mapping_sha256,
                   "captured canonical mapping differs from trusted run")
    c.validate_assistant_request(request, profile=capture.profile, policy_mode=capture.policy_mode,
                                 mapping=capture.mapping, retrieval=capture.retrieval, offline=True)
    if stage.parse_status is not ParseStatus.NOT_ATTEMPTED:
        raw = stage.raw_output.encode("utf-8")
        try:
            c.decode_json(raw)
        except c.ContractError:
            parsed = False
        else:
            parsed = True
        c._require(parsed == (stage.parse_status is ParseStatus.OK), "invocation parse status disagrees with raw output")
        if stage.schema_validation_status is not SchemaValidationStatus.NOT_ATTEMPTED:
            try:
                generated = c.decode_generated_content(raw, request=request)
            except c.ContractError:
                c._require(stage.schema_validation_status is SchemaValidationStatus.INVALID,
                           "claimed valid output fails schema or source validation")
            else:
                c._require(stage.schema_validation_status is SchemaValidationStatus.VALID,
                           "claimed invalid schema passes validation")
                if stage.terminal_outcome is c.BranchOutcome.OK:
                    output = generated.attributes if stage.stage_type is c.Operation.ATTRIBUTES else generated
                    c._require(output == stage.validated_stage_output, "raw and validated stage output disagree")

    digest = stage.produced_attributes_sha256
    c._require((digest is None) == (capture.published_attributes is None),
               "published attribute snapshot and digest must appear together")
    if digest is not None:
        c._require(serialize_attributes is not None, "published attributes require a pinned serializer")
        c.verify_sha256(capture.published_attributes, digest)
        c._require(capture.published_attributes == serialize_attributes(_attributes(stage)),
                   "published bytes do not serialize the validated attributes", c.ErrorCode.HASH_MISMATCH)


def _validate_attribute_input(
    stage: E14ModelInvocation,
    capture: InvocationCapture,
    invocations: Mapping[InvocationRef, E14ModelInvocation],
    pair: E14PairLock | None,
) -> None:
    link = stage.consumed_attributes
    if link is None:
        return
    producer = invocations.get(link.producer)
    c._require(producer is not None and producer.stage_type is c.Operation.ATTRIBUTES
               and producer.terminal_outcome is c.BranchOutcome.OK, "attribute input requires a successful extraction")
    c._require(producer.image_id == stage.image_id
               and (producer.question_id is None or producer.question_id == stage.question_id),
               "attribute producer belongs to another image or question")
    if producer.eval_ref != stage.eval_ref:
        c._require(pair is not None and pair.shared_extraction and {producer.eval_ref, stage.eval_ref} == {pair.a, pair.b}
                   and producer.context_provenance.kb_sha256 is None,
                   "cross-run attribute input is outside approved paired sharing", c.ErrorCode.POLICY_BLOCKED)
    c._require(producer.produced_attributes_sha256 == link.attributes_sha256
               and _attributes(producer) == capture.request.intermediate_attributes,
               "consumed attribute snapshot hash or content mismatch", c.ErrorCode.HASH_MISMATCH)


def _bind_e14_trace(
    trace: VqaOutputTrace, invocations: Mapping[InvocationRef, E14ModelInvocation],
) -> E14ModelInvocation | None:
    stages = []
    for ref in trace.invocation_refs:
        stage = invocations.get(ref)
        c._require(stage is not None and stage.image_id == trace.image_id
                   and (stage.question_id is None or stage.question_id == trace.question_id),
                   "trace has an unknown or mismatched invocation")
        stages.append(stage)

    extraction = stages[0] if stages and stages[0].stage_type is c.Operation.ATTRIBUTES else None
    answer_stages = stages[1:] if extraction is not None else stages
    c._require(len(answer_stages) <= 1 and not any(stage.stage_type is c.Operation.ATTRIBUTES for stage in answer_stages),
               "trace stage sequence must be extraction and/or one answer invocation")
    answer = answer_stages[0] if answer_stages else None

    if answer is None:
        c._require(trace.raw_model_output is None and trace.validated_model_output is None, "trace has no VQA invocation")
        if extraction is not None and not trace.policy_blocked:
            c._require(extraction.terminal_outcome is not c.BranchOutcome.OK
                       and trace.reason_code is extraction.error_code,
                       "trace stopped before VQA without a matching extraction failure")
        return None

    c._require(not trace.policy_blocked and answer.eval_ref == trace.eval_ref,
               "blocked trace or different run cannot supply a VQA invocation")
    if extraction is not None:
        c._require(answer.stage_type is c.Operation.VQA and extraction.terminal_outcome is c.BranchOutcome.OK,
                   "VQA cannot follow a failed or incompatible extraction")
    if answer.consumed_attributes is not None:
        c._require(extraction is not None and extraction.ref == answer.consumed_attributes.producer,
                   "trace must include the consumed attribute producer")
    c._require(trace.raw_model_output == answer.raw_output
               and trace.retrieval_status is answer.context_provenance.retrieval_status,
               "trace raw output or retrieval status differs from answer invocation")
    result = trace.validated_model_output
    if answer.terminal_outcome is c.BranchOutcome.OK:
        c._require(result is not None and (result.profile_id, result.model, result.context_provenance, result.generated)
                   == (answer.profile_id, answer.model, answer.context_provenance, answer.validated_stage_output),
                   "trace result differs from validated answer invocation")
    else:
        c._require(result is None and trace.reason_code is answer.error_code, "trace failure differs from invocation")
    return answer


def _validate_pair_runs(
    pair: E14PairLock, runs: Mapping[c.EvalRef, RunProvenance], trusted_runs: Mapping[c.EvalRef, TrustedRun],
) -> None:
    c._expect_record(pair, E14PairLock)
    c._require({pair.a, pair.b} <= runs.keys(), "paired validation requires both approved runs")
    a, b = runs[pair.a], runs[pair.b]
    for run in (a, b):
        c._require(run.paired_group_id == pair.paired_group_id, "run group differs from trusted paired group")
    for name in ("phase", "lock_ref", "git_sha", "protocol_revision", "environment_ref", "mapping_sha256",
                 "manifest_sha256", "profiles", "rubric_version"):
        c._require(getattr(a, name) == getattr(b, name), f"E14 shared run pin differs: {name}")
    ma, mb = trusted_runs[pair.a].manifest, trusted_runs[pair.b].manifest
    c._require(ma.questions == mb.questions and ma.questions_sha256 == mb.questions_sha256
               and ma.annotations == mb.annotations and ma.annotations_sha256 == mb.annotations_sha256
               and ma.images == mb.images and pair.shared_inputs.keys() == ma.questions.keys(),
               "paired lock and approved image/question/annotation inputs differ")


def _validate_pair_trace(
    pair: E14PairLock,
    trace: VqaOutputTrace,
    answer: E14ModelInvocation | None,
    invocations: Mapping[InvocationRef, E14ModelInvocation],
    captures: Mapping[InvocationRef, InvocationCapture],
) -> None:
    for ref in trace.invocation_refs:
        stage = invocations[ref]
        if stage.stage_type is c.Operation.ATTRIBUTES:
            c._require(pair.answer_stage is c.Operation.VQA, "paired single-pass lock excludes separate extraction")
            if pair.shared_extraction:
                c._require(stage.context_provenance.kb_sha256 is None, "shared extraction must precede KB intervention")
        else:
            c._require(stage.stage_type is pair.answer_stage, "answer stage does not match paired pipeline")
    if answer is not None:
        capture = captures[answer.ref]
        c._require(capture.shared_inputs_sha256 == pair.shared_inputs[_slot(trace)],
                   "captured shared VQA inputs differ from paired lock", c.ErrorCode.HASH_MISMATCH)
        if pair.shared_extraction:
            first = invocations[trace.invocation_refs[0]]
            c._require(first.stage_type is c.Operation.ATTRIBUTES and first.produced_attributes_sha256 is not None,
                       "paired pipeline requires a published shared attribute snapshot")


def validate_e14_evidence(
    traces: Iterable[VqaOutputTrace],
    invocations: Iterable[E14ModelInvocation],
    *,
    runs: Iterable[RunProvenance],
    trusted_runs: Mapping[c.EvalRef, TrustedRun],
    captures: Mapping[InvocationRef, InvocationCapture],
    serialize_attributes: Callable[[tuple[c.AttributeObservation, ...]], bytes] | None = None,
    pair: E14PairLock | None = None,
) -> None:
    """Validate stage joins and completed cohort coverage using independent calls and pins."""
    run_by_ref = _index(runs, lambda run: run.eval_ref, "E14 run")
    c._require(run_by_ref and run_by_ref.keys() == trusted_runs.keys(), "E14 run scopes do not match")
    for ref, run in run_by_ref.items():
        c._require(ref.experiment_id in c._ASSISTANT_EXPERIMENTS, "stage joins are defined for E14")
        validate_run_provenance(run, trusted=trusted_runs[ref])
    if pair is not None:
        _validate_pair_runs(pair, run_by_ref, trusted_runs)

    stages = _index(invocations, lambda stage: stage.ref, "invocation ID in run")
    c._require(stages.keys() == captures.keys(), "invocation IDs differ from captured calls")
    for ref, stage in stages.items():
        c._expect_record(stage, E14ModelInvocation)
        run = run_by_ref.get(ref.eval_ref)
        c._require(run is not None, "invocation is outside approved runs")
        trusted = trusted_runs[ref.eval_ref]
        _bind_run_record(stage, run, trusted)
        _validate_capture(stage, captures[ref], trusted, serialize_attributes)
    for ref, stage in stages.items():
        _validate_attribute_input(stage, captures[ref], stages, pair)

    rows = _index(traces, lambda trace: (trace.eval_ref, *_slot(trace)), "trace slots")
    _index(rows.values(), lambda trace: trace.response_id, "response ID")
    seen: dict[c.EvalRef, set[_Slot]] = {ref: set() for ref in run_by_ref}
    answers = {}
    extractions = {}
    used_refs: set[InvocationRef] = set()
    for trace in rows.values():
        c._expect_record(trace, VqaOutputTrace)
        run = run_by_ref.get(trace.eval_ref)
        c._require(run is not None, "trace is outside approved runs")
        _bind_run_record(trace, run, trusted_runs[trace.eval_ref])
        for ref in trace.invocation_refs:
            stage = stages.get(ref)
            if ref.eval_ref != trace.eval_ref:
                c._require(pair is not None and pair.shared_extraction and {ref.eval_ref, trace.eval_ref} == {pair.a, pair.b}
                           and stage is not None and stage.stage_type is c.Operation.ATTRIBUTES
                           and stage.context_provenance.kb_sha256 is None,
                           "trace cross-run reference is outside approved paired sharing", c.ErrorCode.POLICY_BLOCKED)
        answer = _bind_e14_trace(trace, stages)
        used_refs.update(trace.invocation_refs)
        slot = _slot(trace)
        seen[trace.eval_ref].add(slot)
        if pair is not None and trace.eval_ref in (pair.a, pair.b):
            _validate_pair_trace(pair, trace, answer, stages, captures)
            if answer is not None:
                answers[trace.eval_ref, slot] = answer
            if trace.invocation_refs and stages[trace.invocation_refs[0]].stage_type is c.Operation.ATTRIBUTES:
                extractions[trace.eval_ref, slot] = trace.invocation_refs[0]

    for ref, run in run_by_ref.items():
        if run.execution_status is RunExecutionStatus.COMPLETED:
            c._require(seen[ref] == trusted_runs[ref].manifest.questions.keys(),
                       "completed traces must cover the approved cohort, including blocked and failed slots")
            c._require(all(stage_ref in used_refs for stage_ref in stages if stage_ref.eval_ref == ref),
                       "completed run has a captured invocation without a trace")

    if pair is not None:
        for slot in seen[pair.a] & seen[pair.b]:
            a_extraction, b_extraction = (extractions.get((ref, slot)) for ref in (pair.a, pair.b))
            if pair.shared_extraction and a_extraction is not None and b_extraction is not None:
                c._require(a_extraction == b_extraction, "paired traces must reuse the same extraction invocation")
            a = answers.get((pair.a, slot))
            b = answers.get((pair.b, slot))
            if a is not None and b is not None:
                c._require(a.consumed_attributes == b.consumed_attributes,
                           "paired answers do not reference the same attribute snapshot")
                ca, cb = captures[a.ref], captures[b.ref]
                c._require(ca.profile == cb.profile and ca.policy_mode == cb.policy_mode
                           and ca.request.cnn_context == cb.request.cnn_context,
                           "paired profile, CNN context or validation policy differ")


# The private index stays with the evaluator. A packet is rebuilt from validated trace output.

@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewRubric:
    rubric_version: str
    field_checks: Mapping[str, Callable[[object], bool]]
    reviewer_ids: tuple[str, ...]
    adjudicator_id: str

    def __post_init__(self) -> None:
        for value in (self.rubric_version, self.adjudicator_id):
            c._field_value(value, str, "rubric identity", wire=False)
        reviewers = c._field_value(self.reviewer_ids, tuple[str, ...], "reviewer IDs", wire=False)
        c._require(len(reviewers) == 2 and len(set(reviewers)) == 2, "rubric requires two distinct reviewers")
        checks = dict(self.field_checks)
        for name, check in checks.items():
            c._field_value(name, str, "rubric field", wire=False)
            c._require(callable(check), "rubric field requires a trusted checker")
        object.__setattr__(self, "reviewer_ids", reviewers)
        object.__setattr__(self, "field_checks", MappingProxyType(checks))


def new_response_id() -> str:
    """Generate an opaque ID; corpus validation still checks uniqueness."""
    return uuid4().hex


def make_blinded_packet(
    trace: VqaOutputTrace, *, rubric: ReviewRubric, question: Question, review_image_ref: str,
) -> BlindedResponse:
    """Use only the approved projection inputs and validated model content."""
    c._expect_record(trace, VqaOutputTrace)
    c._expect_record(rubric, ReviewRubric)
    c._expect_record(question, Question)
    c._require(_slot(trace) == _slot(question), "question does not match blinded response slot")
    result = trace.validated_model_output
    return BlindedResponse(response_id=trace.response_id, review_image_ref=review_image_ref, question=question.question,
                           content_for_review=None if result is None else result.generated,
                           rubric_version=rubric.rubric_version)


def _check_labels(labels: Mapping[str, object], rubric: ReviewRubric) -> None:
    c._require(labels.keys() == rubric.field_checks.keys(), "label fields differ from approved rubric")
    for key, value in labels.items():
        c._require(rubric.field_checks[key](value) is True, f"label fails the approved rule for {key}")


def validate_review_joins(
    traces: Iterable[VqaOutputTrace],
    indexes: Iterable[ResponseIndex],
    blinded: Iterable[BlindedResponse],
    labels: Iterable[ReviewerLabel],
    adjudications: Iterable[AdjudicatedLabel],
    *,
    rubric: ReviewRubric,
    manifest: ApprovedManifest,
    review_image_refs: Mapping[str, str],
) -> None:
    """Validate one review scope against already bound traces and the approved rubric."""
    c._expect_record(rubric, ReviewRubric)
    c._expect_record(manifest, ApprovedManifest)
    by_response = _index(traces, lambda trace: trace.response_id, "review response ID")
    _index(by_response.values(), lambda trace: (trace.eval_ref, *_slot(trace)), "private response slot")
    private = _index(indexes, lambda index: index.response_id, "private response ID")
    packets = _index(blinded, lambda packet: packet.response_id, "blinded response ID")
    c._require(private.keys() == packets.keys() == by_response.keys(), "trace/index/blinded response IDs differ")
    for response_id, trace in by_response.items():
        c._expect_record(trace, VqaOutputTrace)
        entry = private[response_id]
        c._expect_record(entry, ResponseIndex)
        c._expect_record(packets[response_id], BlindedResponse)
        c._require(entry.eval_ref == trace.eval_ref and _slot(entry) == _slot(trace), "private response slot mismatch")
        question = manifest.questions.get(_slot(trace))
        image_ref = review_image_refs.get(trace.image_id)
        c._require(question is not None and image_ref is not None, "review image or question is outside approved projection")
        packet = make_blinded_packet(trace, rubric=rubric, question=question, review_image_ref=image_ref)
        c._require(packet == packets[response_id], "blinded response differs from validated model projection")

    originals = _index(labels, lambda label: label.reviewer_label_id, "reviewer-label ID")
    by_reviewer = _index(originals.values(), lambda label: (label.response_id, label.reviewer_id),
                         "reviewer for response")
    expected = {(response, reviewer) for response in by_response for reviewer in rubric.reviewer_ids}
    c._require(by_reviewer.keys() == expected, "each response requires its two independent reviewers")
    for label in originals.values():
        c._expect_record(label, ReviewerLabel)
        c._require(label.rubric_version == rubric.rubric_version, "reviewer rubric version mismatch")
        _check_labels(label.labels, rubric)
    finals = _index(adjudications, lambda row: row.response_id, "adjudicated response ID")
    for final in finals.values():
        c._expect_record(final, AdjudicatedLabel)
        c._require(final.response_id in by_response and final.rubric_version == rubric.rubric_version
                   and final.resolved_by == rubric.adjudicator_id, "adjudication outside approved response/rubric scope")
        expected_refs = {by_reviewer[final.response_id, reviewer].reviewer_label_id for reviewer in rubric.reviewer_ids}
        c._require(set(final.reviewer_label_refs) == expected_refs, "adjudication must reference both original labels")
        _check_labels(final.final_labels, rubric)


__all__ = [
    "AuditStatus", "ImageGroup", "QuestionCategory", "Answerability", "PredictionStatus", "ParseStatus",
    "SchemaValidationStatus", "RunPhase", "RunExecutionStatus", "EvidenceRecord", "PilotImage", "VlmEvalImage",
    "Question", "Annotation", "AnnotationProvenance", "ClassificationPrediction", "VqaOutputTrace",
    "E14ModelInvocation", "InvocationRef", "AttributeInputRef", "ResponseIndex", "BlindedResponse",
    "ReviewerLabel", "AdjudicatedLabel", "Metric", "Benchmark", "ConfidenceInterval", "RunProvenance",
    "RunProfileRef", "UsageGrant", "EvaluationScope", "ApprovedManifest", "RuntimeEvalInput", "TrustedRun",
    "MetricDefinition", "BenchmarkScope", "InvocationCapture", "E14PairLock", "ReviewRubric",
    "decode_evidence_record", "validate_evidence_record", "validate_manifest", "validate_disjoint_manifests",
    "project_runtime_input", "validate_run_provenance", "verify_run_artifacts", "validate_evidence_run_binding",
    "validate_classification_cohort", "validate_e14_evidence", "validate_metric", "validate_benchmark",
    "new_response_id", "make_blinded_packet", "validate_review_joins",
]
