"""Deterministic evidence fixtures. Paths, reviews, locks and calls are synthetic."""
from __future__ import annotations

from dataclasses import dataclass, replace

from src.vlm import contracts as c, evidence as e

from . import fixtures as f, mocks as m


IMAGE_ID = "fixture-image"
QUESTION_ID = "fixture-question"
RUBRIC_VERSION = "synthetic-rubric-1"
START = "2099-01-01T12:00:00Z"
FINISH = "2099-01-01T12:01:00Z"
PERMISSION = "fixture-license-review"
REVIEW_REF = "fixture/review/image"
SHARED_INPUT = f.digest(b"synthetic independently pinned shared input")
OBSERVATIONS = (c.AttributeObservation(attribute_id="ear_shape", visibility=c.Visibility.UNKNOWN, value=None),)


def ndjson(*records: e.EvidenceRecord) -> bytes:
    return b"".join(c.encode_record(record) + b"\n" for record in records)


def make_image(image_id: str = IMAGE_ID) -> c.ValidatedRgb:
    image = f.make_image()
    return image if image_id == IMAGE_ID else replace(image, image_digest=f.digest(image_id.encode("utf-8")))


def image_row(image_id: str = IMAGE_ID, *, with_question: bool = True) -> e.VlmEvalImage:
    return e.VlmEvalImage(
        image_id=image_id, sample_id=f"{image_id}-sample", image_ref=f"fixture/eval/{image_id}",
        source_url=f"https://example.invalid/{image_id}", usage_permission=PERMISSION,
        collection_date="2098-12-31", class_id=7, group=e.ImageGroup.IN_SCOPE_DOG,
        image_digest=make_image(image_id).image_digest, perceptual_hash=f"synthetic-phash-{image_id}",
        audit_status=e.AuditStatus.APPROVED, question_ids=(QUESTION_ID,) if with_question else (),
        annotation_provenance="synthetic-annotation-review",
    )


def pilot_row() -> e.PilotImage:
    return e.PilotImage(image_id="fixture-pilot", sample_id="pilot-sample", image_ref="fixture/dev/pilot",
                        source_ref="synthetic-pilot-origin", usage_permission=PERMISSION,
                        image_digest=f.digest(b"synthetic independent pilot bytes"), audit_status=e.AuditStatus.APPROVED,
                        perceptual_hash="synthetic-pilot-phash")


def question_row(image_id: str = IMAGE_ID) -> e.Question:
    return e.Question(image_id=image_id, question_id=QUESTION_ID, sample_id=f"{image_id}-question-sample",
                      question=f.QUESTION, category=e.QuestionCategory.VISIBLE_ATTRIBUTES)


def annotation_row(image_id: str = IMAGE_ID) -> e.Annotation:
    return e.Annotation(image_id=image_id, question_id=QUESTION_ID, rubric_version=RUBRIC_VERSION,
                        target_attributes=OBSERVATIONS, expected_answer="Synthetic scorer-only answer",
                        answerability=e.Answerability.ANSWERABLE, source_required=True,
                        expected_source_ids=("synthetic-scorer-only-source",),
                        provenance=e.AnnotationProvenance(annotator_id="fixture-annotator", reviewed_at=START,
                                                          revision="synthetic-review-1"))


def scope_for(*images: e.ImageRecord, phase: e.RunPhase = e.RunPhase.FORMAL) -> e.EvaluationScope:
    grants = {image.image_ref: e.UsageGrant(
        source=image.source_ref if type(image) is e.PilotImage else image.source_url, usage_permission=PERMISSION
    ) for image in images}
    return e.EvaluationScope(scope_ref=f"synthetic-scope-{phase.value}", phase=phase, image_grants=grants)


def make_manifest(*, vqa: bool = True, phase: e.RunPhase = e.RunPhase.FORMAL,
                  image_ids: tuple[str, ...] = (IMAGE_ID,)) -> e.ApprovedManifest:
    images = tuple(image_row(i, with_question=vqa) for i in image_ids)
    raw = ndjson(*images)
    if vqa:
        questions = ndjson(*(question_row(i) for i in image_ids))
        annotations = ndjson(*(annotation_row(i) for i in image_ids))
    else:
        questions = annotations = None
    return e.validate_manifest(raw, expected_sha256=f.digest(raw), scope=scope_for(*images, phase=phase),
                               questions_raw=questions, questions_sha256=None if questions is None else f.digest(questions),
                               annotations_raw=annotations,
                               annotations_sha256=None if annotations is None else f.digest(annotations))


def make_run(
    experiment: str, artifacts: f.SyntheticArtifacts, *, manifest: e.ApprovedManifest | None = None,
    phase: e.RunPhase = e.RunPhase.FORMAL, status: e.RunExecutionStatus = e.RunExecutionStatus.COMPLETED,
) -> tuple[e.RunProvenance, e.TrustedRun]:
    classifier = experiment.startswith(("E12", "E13"))
    if manifest is None:
        manifest = make_manifest(vqa=not classifier, phase=phase)
    profile = f.make_profile("encoder" if classifier else "assistant")
    run = e.RunProvenance(
        eval_ref=c.EvalRef(experiment_id=experiment,
                          run_id=f"{experiment}_{phase.value}_20990101T120000Z_123e4567-e89b-42d3-a456-426614174000"),
        phase=phase, execution_status=status, git_sha="a" * 40, protocol_revision="synthetic-protocol-1",
        lock_ref=f"synthetic-approved-lock-{experiment[:3]}" if phase is e.RunPhase.FORMAL else None,
        config_sha256=f.digest(f"synthetic run config for {experiment}".encode("utf-8")),
        environment_ref="synthetic-environment", mapping_sha256=artifacts.mapping.mapping_sha256,
        manifest_sha256=manifest.manifest_sha256,
        profiles=(e.RunProfileRef(profile_id=profile.profile_id, model=profile.model),),
        artifact_hashes={"synthetic-capture": f.digest(b"synthetic capture bytes")},
        started_at=START, finished_at=None if status is e.RunExecutionStatus.RUNNING else FINISH,
        prompt_bank_sha256=(artifacts.name if experiment.endswith("-A") else artifacts.morphology).prompt_bank_sha256
        if classifier else c.MISSING,
        kb_sha256=artifacts.knowledge.kb_sha256 if experiment == "E14-B" else c.MISSING,
        rubric_version=RUBRIC_VERSION if not classifier else c.MISSING,
        paired_group_id="synthetic-E14-pair" if experiment.startswith("E14") else c.MISSING,
    )
    return run, e.TrustedRun(provenance=run, manifest=manifest, approved_lock_ref=run.lock_ref,
                             knowledge=artifacts.knowledge if experiment == "E14-B" else None)


def prediction(
    run: e.RunProvenance, artifacts: f.SyntheticArtifacts, *, image_id: str = IMAGE_ID,
    status: e.PredictionStatus = e.PredictionStatus.OK,
) -> e.ClassificationPrediction:
    profile = f.make_profile()
    bank = artifacts.name if run.eval_ref.experiment_id.endswith("-A") else artifacts.morphology
    image = make_image(image_id)
    request = replace(f.make_zero_shot_request(profile, bank, eval_ref=run.eval_ref),
                      image=image, image_digest=image.image_digest)
    result = m.MockZeroShotClassifier(fail=status is not e.PredictionStatus.OK).classify(
        request, profile=profile, prompts=bank,
    )
    c.validate_zero_shot_result(result, request=request, profile=profile, prompts=bank, offline=True)
    return e.ClassificationPrediction(eval_ref=run.eval_ref, image_id=image_id, image_digest=image.image_digest,
                                      prediction_status=status,
                                      model_result=None if status is e.PredictionStatus.ABSTAIN else result,
                                      reason_code=c.MISSING if status is e.PredictionStatus.OK else result.error.code)


def serialize_attributes(attributes: tuple[c.AttributeObservation, ...]) -> bytes:
    """Compact, deterministic fixture bytes. Production serialization awaits a lock."""
    return b"[" + b",".join(c.encode_record(row) for row in attributes) + b"]"


@dataclass(frozen=True, slots=True)
class StageFixture:
    record: e.E14ModelInvocation
    capture: e.InvocationCapture
    result: c.AssistantResult | None


def make_stage(
    run: e.RunProvenance, artifacts: f.SyntheticArtifacts, *,
    operation: c.Operation = c.Operation.ATTRIBUTES_AND_VQA,
    retrieval_status: c.RetrievalStatus = c.RetrievalStatus.NOT_APPLICABLE,
    producer: StageFixture | None = None,
    publish_attributes: bool = False,
    outcome: str = "ok",
    invocation_id: str | None = None,
) -> StageFixture:
    profile = f.make_profile("assistant")
    retrieval = None
    if run.eval_ref.experiment_id == "E14-B" and operation is not c.Operation.ATTRIBUTES:
        knowledge = artifacts.knowledge
        hit = retrieval_status in (c.RetrievalStatus.HIT, c.RetrievalStatus.PARTIAL_HIT)
        lookup_request = replace(f.make_retrieval_request(knowledge), explicit_class_ids=(7,) if hit else (),
                                 lookup_applicable=retrieval_status is not c.RetrievalStatus.NOT_APPLICABLE)
        retrieval = c.validate_retrieval_result(
            f.make_retrieval_result(knowledge, retrieval_status), request=lookup_request, knowledge=knowledge,
            token_counter=m.MockContextTokenCounter({f.CONTEXT: 8}), render_context=f.render_context,
            policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED,
        )
    request = f.make_assistant_request(profile, operation=operation, eval_ref=run.eval_ref,
                                       retrieval=None if retrieval is None else retrieval.result)
    consumed = None
    if producer is not None:
        request = replace(request, intermediate_attributes=producer.record.validated_stage_output)
        consumed = e.AttributeInputRef(producer=producer.record.ref,
                                       attributes_sha256=producer.record.produced_attributes_sha256)
    result = None
    code = None
    published = None
    output = None
    if outcome == "ok":
        result = m.MockGenerativeAssistant().execute(request, profile=profile)
        c.validate_assistant_result(result, request=request, profile=profile, policy_mode=request.policy_mode,
                                    retrieval=retrieval, offline=True)
        raw = c.encode_record(result.generated).decode("utf-8")
        parsed, schema, terminal = e.ParseStatus.OK, e.SchemaValidationStatus.VALID, c.BranchOutcome.OK
        output = result.generated.attributes if operation is c.Operation.ATTRIBUTES else result.generated
        if publish_attributes:
            published = serialize_attributes(result.generated.attributes)
    elif outcome == "invalid_json":
        raw, parsed, schema = "", e.ParseStatus.INVALID_JSON, e.SchemaValidationStatus.NOT_ATTEMPTED
        terminal, code = c.BranchOutcome.INVALID_OUTPUT, c.ErrorCode.INVALID_OUTPUT
    elif outcome == "invalid_schema":
        raw, parsed, schema = "{}", e.ParseStatus.OK, e.SchemaValidationStatus.INVALID
        terminal, code = c.BranchOutcome.INVALID_OUTPUT, c.ErrorCode.INVALID_OUTPUT
    else:
        fault = m.MockGenerativeAssistant(fail=True).execute(request, profile=profile)
        raw, parsed, schema = None, e.ParseStatus.NOT_ATTEMPTED, e.SchemaValidationStatus.NOT_ATTEMPTED
        terminal, code = fault.status, fault.error.code
    record = e.E14ModelInvocation(
        eval_ref=run.eval_ref, invocation_id=invocation_id or f"fixture-{operation.value}", stage_type=operation,
        image_id=IMAGE_ID, question_id=None if operation is c.Operation.ATTRIBUTES else QUESTION_ID,
        model=profile.model, profile_id=profile.profile_id, raw_output=raw, parse_status=parsed,
        schema_validation_status=schema, terminal_outcome=terminal, error_code=code, latency_ms=0,
        context_provenance=request.context_provenance, validated_stage_output=output,
        produced_attributes_sha256=None if published is None else f.digest(published), consumed_attributes=consumed,
    )
    capture = e.InvocationCapture(
        request=request, profile=profile, policy_mode=request.policy_mode, retrieval=retrieval, raw_output=raw,
        terminal_outcome=terminal, error_code=code, published_attributes=published, shared_inputs_sha256=SHARED_INPUT,
    )
    return StageFixture(record, capture, result)


def trace_for(
    run: e.RunProvenance, *, answer: StageFixture | None = None, extraction: StageFixture | None = None,
    blocked: bool = False, controller_response: str | None = None,
) -> e.VqaOutputTrace:
    if blocked:
        reason = c.ErrorCode.POLICY_BLOCKED
    elif answer is not None and answer.result is not None:
        reason = c.MISSING
    elif answer is not None or extraction is not None:
        reason = (answer or extraction).record.error_code
    else:
        reason = c.ErrorCode.MODEL_UNAVAILABLE
    return e.VqaOutputTrace(
        eval_ref=run.eval_ref, image_id=IMAGE_ID, question_id=QUESTION_ID,
        response_id=f"fixture-opaque-{f.digest(run.eval_ref.run_id.encode())[:12]}",
        raw_model_output=None if answer is None else answer.record.raw_output,
        validated_model_output=None if answer is None else answer.result,
        controller_generated_response=controller_response, policy_blocked=blocked,
        retrieval_status=c.RetrievalStatus.NOT_APPLICABLE if answer is None else answer.record.context_provenance.retrieval_status,
        invocation_refs=tuple(stage.record.ref for stage in (extraction, answer) if stage is not None), reason_code=reason,
    )


def review_rubric() -> e.ReviewRubric:
    return e.ReviewRubric(rubric_version=RUBRIC_VERSION,
                          field_checks={"correct": lambda value: type(value) is bool},
                          reviewer_ids=("fixture-reviewer-1", "fixture-reviewer-2"), adjudicator_id="fixture-adjudicator")


def review_records(trace: e.VqaOutputTrace, rubric: e.ReviewRubric):
    index = e.ResponseIndex(response_id=trace.response_id, eval_ref=trace.eval_ref,
                            image_id=trace.image_id, question_id=trace.question_id)
    packet = e.make_blinded_packet(trace, rubric=rubric, question=question_row(), review_image_ref=REVIEW_REF)
    labels = tuple(e.ReviewerLabel(reviewer_label_id=f"synthetic-label-{i}", response_id=trace.response_id,
                                   reviewer_id=reviewer, rubric_version=rubric.rubric_version, labels={"correct": i == 0},
                                   submitted_at=FINISH)
                   for i, reviewer in enumerate(rubric.reviewer_ids))
    final = e.AdjudicatedLabel(response_id=trace.response_id, rubric_version=rubric.rubric_version,
                               reviewer_label_refs=tuple(label.reviewer_label_id for label in labels),
                               final_labels={"correct": True}, resolved_by=rubric.adjudicator_id,
                               resolution_note="Synthetic adjudication.")
    return index, packet, labels, final


def benchmark(run: e.RunProvenance) -> e.Benchmark:
    return e.Benchmark(eval_ref=run.eval_ref, profile_id=run.profiles[0].profile_id, hardware_ref="synthetic-cpu",
                       measurement="warm-inference", scenario="synthetic-batch-1", warmup_count=1,
                       measured_count=2, unit="ms", values=(11,), failure_count=1)


def metric(run: e.RunProvenance) -> e.Metric:
    return e.Metric(eval_ref=run.eval_ref, metric_id="fixture-top1", scope="whole-cohort", value=0.75,
                    unit="ratio", n_evaluated=4, numerator=3, denominator=4,
                    ci=e.ConfidenceInterval(lower=0.5, upper=1, level=0.95))
