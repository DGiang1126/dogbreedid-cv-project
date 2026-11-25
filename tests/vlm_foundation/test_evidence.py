"""Contract 06 shape, trust boundary and join tests. All inputs are synthetic."""
from dataclasses import FrozenInstanceError, fields, replace
import json

import pytest

from src.vlm import contracts as c, evidence as e

from . import evidence_fixtures as ef, fixtures as f
from . import mocks


@pytest.fixture(scope="module")
def artifacts():
    return f.make_artifacts()


@pytest.fixture
def all_records(artifacts):
    encoder, _ = ef.make_run("E12-A", artifacts)
    e14, _ = ef.make_run("E14-A", artifacts)
    answer = ef.make_stage(e14, artifacts)
    trace = ef.trace_for(e14, answer=answer)
    index, packet, labels, final = ef.review_records(trace, ef.review_rubric())
    return {
        row.record_type: row for row in (
            ef.pilot_row(), ef.image_row(), ef.question_row(), ef.annotation_row(),
            ef.prediction(encoder, artifacts), trace, answer.record,
            index, packet, labels[0], final, ef.metric(encoder), ef.benchmark(encoder), encoder,
        )
    }


RECORD_TYPES = (
    "pilot_image", "vlm_eval_image", "question", "annotation", "classification_prediction", "vqa_output_trace",
    "e14_model_invocation", "response_index", "blinded_response", "reviewer_label", "adjudicated_label", "metric",
    "benchmark", "run_provenance",
)


@pytest.mark.parametrize("record_type", RECORD_TYPES)
def test_all_evidence_records_round_trip_with_required_envelope(all_records, record_type):
    source = all_records[record_type]
    raw = c.encode_record(source)
    assert e.decode_evidence_record(raw) == source
    e.validate_evidence_record(source)
    payload = json.loads(raw)
    assert payload["record_type"] == record_type and payload["schema_version"] == "1.0"
    payload.pop("schema_version")
    with pytest.raises(c.ContractError, match="missing schema_version"):
        e.decode_evidence_record(f.json_bytes(payload))


@pytest.mark.parametrize("raw", [
    b'\xef\xbb\xbf{"record_type":"question"}',
    b'{"record_type":"question","question":"\xff"}',
    b'{"record_type":"question","record_type":"annotation"}',
    b'{"record_type":"reviewer_label","labels":{"nested":{"a":1,"a":2}}}',
    b'{"record_type":"metric","value":NaN}',
    b'{"record_type":"metric","value":Infinity}',
    b'{"record_type":"metric","value":1e999}',
    b'{"record_type":"unregistered"}',
    b'{"record_type":true}',
    b'{}',
    b'[]',
])
def test_evidence_rejects_malformed_wire(raw):
    with pytest.raises(c.ContractError):
        e.decode_evidence_record(raw)


@pytest.mark.parametrize("patch", [
    {"schema_version": "1.1"}, {"superfluous": 0}, {"category": "unknown"}, {"question": None},
    {"image_id": False}, {"question_id": 4},
])
def test_strict_record_rejects_unknown_missing_and_wrong_types(patch):
    question = json.loads(c.encode_record(ef.question_row()))
    question.update(patch)
    with pytest.raises(c.ContractError):
        e.decode_evidence_record(f.json_bytes(question))


def test_required_nullable_and_optional_fields_are_distinct():
    row = replace(ef.annotation_row(), expected_answer=None, expected_source_ids=c.MISSING)
    wire = json.loads(c.encode_record(row))
    assert wire["expected_answer"] is None and "expected_source_ids" not in wire
    assert e.decode_evidence_record(f.json_bytes(wire)) == row
    with pytest.raises(c.ContractError, match="missing expected_answer"):
        e.decode_evidence_record(f.json_bytes({k: v for k, v in wire.items() if k != "expected_answer"}))
    with pytest.raises(c.ContractError):
        e.decode_evidence_record(f.json_bytes(wire | {"expected_source_ids": None}))


def test_evidence_is_deeply_immutable(all_records):
    values = [{"detail": ["", True, 1.25, None]}]
    label = replace(all_records["reviewer_label"], labels={"nested": values})
    values[0]["detail"].append("late")
    values.append({})
    assert label.labels["nested"][0]["detail"] == ("", True, 1.25, None)
    with pytest.raises(TypeError):
        label.labels["nested"][0]["detail"] = ()
    with pytest.raises(FrozenInstanceError):
        label.response_id = "other"
    assert e.decode_evidence_record(c.encode_record(label)) == label
    with pytest.raises(c.ContractError):
        replace(label, labels={"nested": [float("nan")]})


@pytest.mark.parametrize("group,class_id", [
    (e.ImageGroup.IN_SCOPE_DOG, None),
    (e.ImageGroup.OUT_OF_SCOPE_DOG, 0),
    (e.ImageGroup.NON_DOG, 29),
])
def test_image_group_controls_class_id_nullability(group, class_id):
    with pytest.raises(c.ContractError, match="class_id"):
        replace(ef.image_row(), group=group, class_id=class_id)


@pytest.mark.parametrize("case", ["image", "image-sample", "digest", "pair", "question-sample", "question-missing",
                                  "question-extra", "annotation-missing", "audit", "scope", "usage", "hash"])
def test_manifest_rejects_duplicate_missing_or_unapproved_evidence(case):
    first = ef.image_row()
    second = ef.image_row("fixture-image-2")
    one, two = ef.question_row(), ef.question_row(second.image_id)
    annotations = ef.ndjson(ef.annotation_row(), ef.annotation_row(second.image_id))
    images = [first, second]
    questions = [one, two]
    if case == "image":
        images[1] = replace(second, image_id=first.image_id)
    elif case == "image-sample":
        images[1] = replace(second, sample_id=first.sample_id)
    elif case == "digest":
        images[1] = replace(second, image_digest=first.image_digest)
    elif case == "pair":
        questions += [one]
    elif case == "question-sample":
        questions[1] = replace(two, sample_id=one.sample_id)
    elif case == "question-missing":
        questions.pop()
    elif case == "question-extra":
        questions += [replace(one, question_id="extra", sample_id="extra")]
    elif case == "annotation-missing":
        annotations = ef.ndjson(ef.annotation_row())
    elif case == "audit":
        images[1] = replace(second, audit_status=e.AuditStatus.PENDING)
    elif case == "usage":
        images[1] = replace(second, usage_permission="unapproved permission")
    scope = ef.scope_for(first) if case == "scope" else ef.scope_for(first, second)
    raw, qraw = ef.ndjson(*images), ef.ndjson(*questions)
    expected = f.digest(raw + b" ") if case == "hash" else f.digest(raw)
    with pytest.raises(c.ContractError):
        e.validate_manifest(raw, expected_sha256=expected, scope=scope, questions_raw=qraw,
                            questions_sha256=f.digest(qraw), annotations_raw=annotations,
                            annotations_sha256=f.digest(annotations))


@pytest.mark.parametrize("overlap", [None, "image_id", "sample_id", "image_digest", "perceptual_hash"])
def test_pilot_and_formal_disjointness_uses_only_manifest_evidence(overlap):
    formal = ef.make_manifest()
    row = ef.pilot_row()
    if overlap is not None:
        row = replace(row, **{overlap: getattr(formal.images[ef.IMAGE_ID], overlap)})
    raw = ef.ndjson(row)
    pilot = e.validate_manifest(raw, expected_sha256=f.digest(raw), scope=ef.scope_for(row, phase=e.RunPhase.PILOT))
    if overlap is None:
        e.validate_disjoint_manifests(pilot, formal)
    else:
        with pytest.raises(c.ContractError, match="pilot"):
            e.validate_disjoint_manifests(pilot, formal)


def test_manifest_projector_is_allowlisted_and_excludes_scorer_context():
    manifest = ef.make_manifest()
    slot = ef.IMAGE_ID, ef.QUESTION_ID
    assert manifest.annotations[slot].expected_answer is not None
    projected = e.project_runtime_input(manifest, image_id=ef.IMAGE_ID, image=ef.make_image(),
                                        operation=c.Operation.VQA, question_id=ef.QUESTION_ID)
    assert {field.name for field in fields(projected)} == {"image", "operation", "question"}
    assert projected.question == f.QUESTION and projected.image.image_digest == manifest.images[ef.IMAGE_ID].image_digest
    with pytest.raises(c.ContractError):
        e.project_runtime_input(manifest, image_id=ef.IMAGE_ID, image=ef.make_image("foreign"),
                                operation=c.Operation.VQA, question_id=ef.QUESTION_ID)
    with pytest.raises(c.ContractError):
        e.project_runtime_input(manifest, image_id=ef.IMAGE_ID, image=ef.make_image(),
                                operation=c.Operation.VQA, question_id="unknown")


@pytest.mark.parametrize("status", list(e.RunExecutionStatus))
def test_classification_cohort_preserves_fail_and_abstain_and_requires_full_completed_run(artifacts, status):
    manifest = ef.make_manifest(vqa=False, image_ids=(ef.IMAGE_ID, "fixture-failure", "fixture-abstention"))
    run, trusted = ef.make_run("E12-A", artifacts, manifest=manifest, status=status)
    ok = ef.prediction(run, artifacts)
    fail = ef.prediction(run, artifacts, image_id="fixture-failure", status=e.PredictionStatus.FAIL)
    abstain = ef.prediction(run, artifacts, image_id="fixture-abstention", status=e.PredictionStatus.ABSTAIN)
    for row in (ok, fail, abstain):
        e.validate_evidence_run_binding(row, run=run, trusted=trusted)
    e.validate_classification_cohort((ok, fail, abstain), run=run, trusted=trusted)
    assert "scores" not in json.loads(c.encode_record(fail))["model_result"]
    assert abstain.model_result is None
    if status is e.RunExecutionStatus.COMPLETED:
        with pytest.raises(c.ContractError, match="cover"):
            e.validate_classification_cohort((ok,), run=run, trusted=trusted)
    else:
        e.validate_classification_cohort((ok,), run=run, trusted=trusted)
    with pytest.raises(c.ContractError, match="duplicate"):
        e.validate_classification_cohort((ok, ok), run=run, trusted=trusted)
    with pytest.raises(c.ContractError):
        replace(ok, prediction_status=e.PredictionStatus.FAIL)


@pytest.mark.parametrize("field", ["image", "image_digest", "mapping_sha256", "prompt_bank_sha256", "profile_id", "model", "run"])
def test_classification_prediction_binds_cohort_model_and_pins(artifacts, field):
    run, trusted = ef.make_run("E13-B", artifacts)
    a = ef.prediction(run, artifacts)
    if field == "image":
        a = replace(a, image_id="foreign")
    elif field == "image_digest":
        digest = f.digest(b"wrong image")
        a = replace(a, image_digest=digest, model_result=replace(a.model_result, image_digest=digest))
    elif field == "run":
        other, _ = ef.make_run("E13-A", artifacts)
        a = ef.prediction(other, artifacts)
    else:
        alt = {"mapping_sha256": f.digest(b"wrong mapping"), "prompt_bank_sha256": f.digest(b"wrong prompt"),
               "profile_id": "wrong profile", "model": f.make_profile("other").model}
        a = replace(a, model_result=replace(a.model_result, **{field: alt[field]}))
    with pytest.raises(c.ContractError):
        e.validate_evidence_run_binding(a, run=run, trusted=trusted)


@pytest.mark.parametrize("approval", [None, "unapproved-reference"])
def test_formal_run_string_alone_does_not_authorize_lock(artifacts, approval):
    run, trusted = ef.make_run("E12-A", artifacts)
    with pytest.raises(c.ContractError, match="unknown or unapproved") as error:
        e.validate_run_provenance(run, trusted=replace(trusted, approved_lock_ref=approval))
    assert error.value.code is c.ErrorCode.POLICY_BLOCKED


@pytest.mark.parametrize("changes", [
    {"protocol_revision": "unknown"}, {"environment_ref": "unknown"}, {"git_sha": "b" * 40},
    {"artifact_hashes": {"synthetic-capture": "1" * 64}}, {"manifest_sha256": "1" * 64},
    {"config_sha256": "1" * 64}, {"mapping_sha256": "1" * 64},
])
def test_run_uses_trusted_provenance_pins(artifacts, changes):
    run, trusted = ef.make_run("E12-A", artifacts)
    e.validate_evidence_run_binding(run, run=run, trusted=trusted)
    with pytest.raises(c.ContractError, match="trusted provenance"):
        e.validate_run_provenance(replace(run, **changes), trusted=trusted)


@pytest.mark.parametrize("changes", [
    {"phase": "pilot"}, {"execution_status": "ACCEPTED"}, {"finished_at": None}, {"lock_ref": None},
    {"finished_at": "2098-01-01T00:00:00Z"}, {"started_at": "2099-02-30T00:00:00Z"},
    {"started_at": "2099-01-01T12:00:00+07:00"},
])
def test_run_shape_rejects_lifecycle_phase_and_invalid_timestamp(artifacts, changes):
    run, _ = ef.make_run("E12-A", artifacts)
    with pytest.raises(c.ContractError):
        e.decode_evidence_record(f.json_bytes(json.loads(c.encode_record(run)) | changes))



def validate_e14(trace, run, trusted, *stages, **changes):
    args = dict(runs=(run,), trusted_runs={run.eval_ref: trusted},
                captures={stage.record.ref: stage.capture for stage in stages},
                serialize_attributes=ef.serialize_attributes)
    e.validate_e14_evidence((trace,), tuple(stage.record for stage in stages), **(args | changes))


@pytest.mark.parametrize("scenario", [
    "single-pass", "multi-pass", "invalid_json", "invalid_schema", "inference_failure",
    "block-before", "block-after", "extraction-failure", "fallback",
])
def test_e14_trace_covers_success_failures_policy_and_controller_layers(artifacts, scenario):
    run, trusted = ef.make_run("E14-A", artifacts)
    extraction = answer = None
    blocked = scenario.startswith("block-")
    if scenario == "multi-pass":
        extraction = ef.make_stage(run, artifacts, operation=c.Operation.ATTRIBUTES, publish_attributes=True)
        answer = ef.make_stage(run, artifacts, operation=c.Operation.VQA, producer=extraction)
    elif scenario in ("block-after", "extraction-failure"):
        extraction = ef.make_stage(run, artifacts, operation=c.Operation.ATTRIBUTES,
                                   outcome="inference_failure" if scenario == "extraction-failure" else "ok")
    elif not blocked:
        outcome = "inference_failure" if scenario == "fallback" else scenario
        answer = ef.make_stage(run, artifacts, outcome="ok" if scenario == "single-pass" else outcome)
    trace = ef.trace_for(run, answer=answer, extraction=extraction, blocked=blocked,
                         controller_response="Synthetic fallback" if scenario == "fallback" else None)
    trace = e.decode_evidence_record(c.encode_record(trace))
    validate_e14(trace, run, trusted, *(stage for stage in (extraction, answer) if stage is not None))
    if blocked or scenario == "extraction-failure":
        assert trace.raw_model_output is None and trace.validated_model_output is None
        assert trace.invocation_refs == (() if extraction is None else (extraction.record.ref,))
    if scenario == "invalid_json":
        assert trace.raw_model_output == "" and trace.validated_model_output is None
    if scenario == "fallback":
        packet = e.make_blinded_packet(trace, rubric=ef.review_rubric(), question=ef.question_row(),
                                      review_image_ref=ef.REVIEW_REF)
        assert packet.content_for_review is None and trace.controller_generated_response == "Synthetic fallback"


@pytest.mark.parametrize("status", list(c.RetrievalStatus))
def test_e14_b_invocation_uses_approved_kb_pin_even_without_hit(artifacts, status):
    run, trusted = ef.make_run("E14-B", artifacts)
    stage = ef.make_stage(run, artifacts, retrieval_status=status)
    trace = ef.trace_for(run, answer=stage)
    validate_e14(trace, run, trusted, stage)
    assert stage.record.context_provenance.kb_sha256 == artifacts.knowledge.kb_sha256
    assert trace.retrieval_status is status
    for changes in ({"kb_sha256": f.digest(b"wrong")}, {"kb_version": "other fixture version"}):
        invalid = replace(stage.record, context_provenance=replace(stage.record.context_provenance, **changes))
        with pytest.raises(c.ContractError, match="approved KB"):
            e.validate_evidence_run_binding(invalid, run=run, trusted=trusted)
    if status is c.RetrievalStatus.NOT_APPLICABLE:
        no_kb = f.make_assistant_request(f.make_profile("assistant")).context_provenance
        with pytest.raises(c.ContractError, match="approved KB"):
            e.validate_evidence_run_binding(replace(stage.record, context_provenance=no_kb), run=run, trusted=trusted)


def test_e14_a_run_binding_rejects_even_empty_pinned_context(artifacts):
    run, trusted = ef.make_run("E14-A", artifacts)
    a = ef.make_stage(run, artifacts)
    b_run, _ = ef.make_run("E14-B", artifacts)
    b = ef.make_stage(b_run, artifacts)
    with pytest.raises(c.ContractError, match="cannot use KB"):
        e.validate_evidence_run_binding(replace(a.record, context_provenance=b.record.context_provenance),
                                        run=run, trusted=trusted)


@pytest.mark.parametrize("scenario", [
    "duplicate-call", "fabricated-call", "unknown-reference", "duplicate-trace", "missing-slot", "raw-invocation",
    "raw-trace", "validated-invocation", "validated-trace", "unrelated-question", "blocked-with-vqa", "bad-parse",
    "bad-schema",
])
def test_e14_rejects_unverifiable_calls_and_mismatched_traces(artifacts, scenario):
    run, trusted = ef.make_run("E14-A", artifacts)
    original = ef.make_stage(run, artifacts)
    invocation = original.record
    trace = ef.trace_for(run, answer=original)
    traces, invocations = [trace], [invocation]
    capture = original.capture
    captures = {invocation.ref: capture}
    if scenario == "duplicate-call":
        invocations.append(invocation)
    elif scenario == "fabricated-call":
        captures = {}
    elif scenario == "unknown-reference":
        traces[0] = replace(trace, invocation_refs=(replace(invocation.ref, invocation_id="not-called"),))
    elif scenario == "duplicate-trace":
        traces.append(replace(trace, response_id="extra-opaque"))
    elif scenario == "missing-slot":
        traces.clear()
    elif scenario == "raw-invocation":
        invocations[0] = replace(invocation, raw_output="{}")
    elif scenario == "raw-trace":
        traces[0] = replace(trace, raw_model_output="{}")
    elif scenario == "validated-invocation":
        output = replace(invocation.validated_stage_output, answer_vi="Synthetic different answer")
        invocations[0] = replace(invocation, validated_stage_output=output)
    elif scenario == "validated-trace":
        output = replace(original.result.generated, answer_vi="Synthetic different answer")
        traces[0] = replace(trace, validated_model_output=replace(original.result, generated=output))
    elif scenario == "unrelated-question":
        captures[invocation.ref] = replace(capture, request=replace(capture.request, question="Other synthetic question"))
    elif scenario == "blocked-with-vqa":
        traces[0] = replace(ef.trace_for(run, blocked=True), invocation_refs=(invocation.ref,))
    elif scenario == "bad-parse":
        stage = ef.make_stage(run, artifacts, outcome="invalid_json")
        invocations[0] = replace(stage.record, raw_output="{}")
        captures[invocation.ref] = replace(stage.capture, raw_output="{}")
    elif scenario == "bad-schema":
        stage = ef.make_stage(run, artifacts, outcome="invalid_schema")
        invocations[0] = replace(stage.record, raw_output=invocation.raw_output)
        captures[invocation.ref] = replace(stage.capture, raw_output=invocation.raw_output)
    with pytest.raises(c.ContractError):
        e.validate_e14_evidence(traces, invocations, runs=(run,), trusted_runs={run.eval_ref: trusted}, captures=captures)


def test_unfinished_e14_run_can_keep_partial_evidence(artifacts):
    run, trusted = ef.make_run("E14-A", artifacts, status=e.RunExecutionStatus.FAILED)
    e.validate_e14_evidence((), (), runs=(run,), trusted_runs={run.eval_ref: trusted}, captures={})


def test_policy_block_neither_invents_model_call_nor_promotes_raw_extraction(artifacts):
    run, trusted = ef.make_run("E14-A", artifacts)
    stage = ef.make_stage(run, artifacts, operation=c.Operation.ATTRIBUTES)
    trace = ef.trace_for(run, extraction=stage, blocked=True)
    assert stage.record.raw_output is not None
    with pytest.raises(c.ContractError, match="null VQA output"):
        replace(trace, raw_model_output=stage.record.raw_output)
    with pytest.raises(c.ContractError, match="captured calls"):
        validate_e14(trace, run, trusted, stage, captures={})
    validate_e14(trace, run, trusted, stage)


@pytest.fixture
def pair_fixture(artifacts):
    manifest = ef.make_manifest()
    a, trusted_a = ef.make_run("E14-A", artifacts, manifest=manifest)
    b, trusted_b = ef.make_run("E14-B", artifacts, manifest=manifest)
    extraction = ef.make_stage(a, artifacts, operation=c.Operation.ATTRIBUTES, publish_attributes=True)
    aa = ef.make_stage(a, artifacts, operation=c.Operation.VQA, producer=extraction)
    ab = ef.make_stage(b, artifacts, operation=c.Operation.VQA, producer=extraction, retrieval_status=c.RetrievalStatus.MISS)
    traces = (ef.trace_for(a, answer=aa, extraction=extraction), ef.trace_for(b, answer=ab, extraction=extraction))
    pair = e.E14PairLock(paired_group_id=a.paired_group_id, a=a.eval_ref, b=b.eval_ref, shared_extraction=True,
                        answer_stage=c.Operation.VQA, shared_inputs={(ef.IMAGE_ID, ef.QUESTION_ID): ef.SHARED_INPUT})
    return (a, b), {a.eval_ref: trusted_a, b.eval_ref: trusted_b}, (extraction, aa, ab), traces, pair


def test_paired_shared_extraction_uses_one_invocation_for_both_slots(pair_fixture):
    runs, trusted, stages, traces, pair = pair_fixture
    invocations = tuple(stage.record for stage in stages)
    captures = {stage.record.ref: stage.capture for stage in stages}
    e.validate_e14_evidence(traces, invocations, runs=runs, trusted_runs=trusted, captures=captures,
                           serialize_attributes=ef.serialize_attributes, pair=pair)
    assert traces[0].invocation_refs[0] == traces[1].invocation_refs[0] == invocations[0].ref
    assert len(captures) == 3
    assert traces[0].retrieval_status is c.RetrievalStatus.NOT_APPLICABLE
    assert traces[1].retrieval_status is c.RetrievalStatus.MISS


@pytest.mark.parametrize("scenario,pattern", [
    ("unapproved-sharing", "paired sharing"), ("wrong-group", "trusted paired group"),
    ("different-shared-inputs", "captured shared VQA inputs"), ("wrong-consumed-hash", "snapshot hash"),
    ("wrong-published-hash", "SHA-256 mismatch"), ("bad-serialization", "do not serialize"),
    ("profile-mismatch", "paired profile"), ("different-pipeline", "single-pass lock"),
    ("not-same-producer", "reuse the same extraction"),
])
def test_paired_stage_binding_rejects_wrong_approval_and_snapshot(pair_fixture, scenario, pattern):
    runs, trusted, stages, traces, pair = pair_fixture
    extraction, aa, ab = stages
    invocations = [stage.record for stage in stages]
    captures = {stage.record.ref: stage.capture for stage in stages}
    if scenario == "unapproved-sharing":
        pair = None
    elif scenario == "wrong-group":
        pair = replace(pair, paired_group_id="self-declared synthetic group")
    elif scenario == "different-shared-inputs":
        captures[ab.record.ref] = replace(ab.capture, shared_inputs_sha256=f.digest(b"other shared input"))
    elif scenario == "wrong-consumed-hash":
        invocations[2] = replace(ab.record, consumed_attributes=replace(ab.record.consumed_attributes,
                                                                        attributes_sha256=f.digest(b"wrong")))
    elif scenario == "wrong-published-hash":
        invocations[0] = replace(extraction.record, produced_attributes_sha256=f.digest(b"wrong"))
    elif scenario == "bad-serialization":
        forged = b"[]"
        invocations[0] = replace(extraction.record, produced_attributes_sha256=f.digest(forged))
        captures[extraction.record.ref] = replace(extraction.capture, published_attributes=forged)
    elif scenario == "profile-mismatch":
        profile = replace(ab.capture.profile, config_sha256=f.digest(b"other profile config"))
        captures[ab.record.ref] = replace(ab.capture, profile=profile)
    elif scenario == "different-pipeline":
        pair = replace(pair, answer_stage=c.Operation.ATTRIBUTES_AND_VQA, shared_extraction=False)
        traces = (traces[0], ef.trace_for(runs[1], blocked=True))
        invocations.pop()
        captures.pop(ab.record.ref)
    elif scenario == "not-same-producer":
        other = replace(extraction.record, invocation_id="different-actual-extraction")
        captures[other.ref] = extraction.capture
        invocations.extend((other,))
        invocations[2] = replace(ab.record, consumed_attributes=replace(ab.record.consumed_attributes, producer=other.ref))
        traces = (traces[0], replace(traces[1], invocation_refs=(other.ref, ab.record.ref)))
    with pytest.raises(c.ContractError, match=pattern):
        e.validate_e14_evidence(traces, invocations, runs=runs, trusted_runs=trusted, captures=captures,
                               serialize_attributes=ef.serialize_attributes, pair=pair)


def test_human_projection_preserves_model_answer_and_private_index(artifacts):
    run, trusted = ef.make_run("E14-A", artifacts)
    stage = ef.make_stage(run, artifacts)
    trace = ef.trace_for(run, answer=stage, controller_response="Synthetic distinct controller text")
    validate_e14(trace, run, trusted, stage)
    rubric = ef.review_rubric()
    index, packet, labels, final = ef.review_records(trace, rubric)
    e.validate_evidence_run_binding(index, run=run, trusted=trusted)
    e.validate_review_joins((trace,), (index,), (packet,), labels, (final,), rubric=rubric,
                            manifest=trusted.manifest, review_image_refs={ef.IMAGE_ID: ef.REVIEW_REF})
    assert packet.content_for_review == stage.result.generated
    assert labels[0].labels["correct"] is True and labels[1].labels["correct"] is False
    assert e.new_response_id() != e.new_response_id()
    data = json.loads(c.encode_record(packet))
    assert set(data) == {"schema_version", "record_type", "response_id", "review_image_ref",
                         "question", "content_for_review", "rubric_version"}
    with pytest.raises(c.ContractError, match="unknown fields"):
        e.decode_evidence_record(f.json_bytes(data | {"eval_ref": json.loads(c.encode_record(run.eval_ref))}))


@pytest.mark.parametrize("scenario", [
    "private-index", "fallback", "duplicate-label-id", "same-reviewer", "wrong-label-type", "wrong-rubric",
    "wrong-original", "wrong-adjudicator", "wrong-final-field", "out-of-scope-review-image",
])
def test_review_joins_reject_identity_rubric_and_projection_mismatches(artifacts, scenario):
    run, trusted = ef.make_run("E14-A", artifacts)
    trace = ef.trace_for(run, answer=ef.make_stage(run, artifacts))
    rubric = ef.review_rubric()
    index, packet, labels, final = ef.review_records(trace, rubric)
    refs = {ef.IMAGE_ID: ef.REVIEW_REF}
    a, b = labels
    if scenario == "private-index":
        index = replace(index, question_id="unrelated")
    elif scenario == "fallback":
        packet = replace(packet, content_for_review=replace(packet.content_for_review, answer_vi="Wrong controller answer"))
    elif scenario == "duplicate-label-id":
        b = replace(b, reviewer_label_id=a.reviewer_label_id)
    elif scenario == "same-reviewer":
        b = replace(b, reviewer_id=a.reviewer_id)
    elif scenario == "wrong-label-type":
        b = replace(b, labels={"correct": 1})
    elif scenario == "wrong-rubric":
        b = replace(b, rubric_version="other rubric")
    elif scenario == "wrong-original":
        final = replace(final, reviewer_label_refs=(a.reviewer_label_id, "unknown label"))
    elif scenario == "wrong-adjudicator":
        final = replace(final, resolved_by="unknown adjudicator")
    elif scenario == "wrong-final-field":
        final = replace(final, final_labels={"extra": False})
    elif scenario == "out-of-scope-review-image":
        refs = {}
    with pytest.raises(c.ContractError):
        e.validate_review_joins((trace,), (index,), (packet,), (a, b), (final,), rubric=rubric,
                                manifest=trusted.manifest, review_image_refs=refs)


def test_metric_na_ratio_ci_and_non_ratio_scope(artifacts):
    run, trusted = ef.make_run("E12-A", artifacts)
    row = ef.metric(run)
    definitions = {(row.metric_id, row.scope): e.MetricDefinition(
        unit="ratio", min_value=0, max_value=1, requires_ratio=True, requires_ci=True, ci_level=0.95,
        max_evaluated=4,
    )}
    e.validate_evidence_run_binding(row, run=run, trusted=trusted)
    e.validate_metric(row, definitions=definitions)
    e.validate_metric(replace(row, value=None, n_evaluated=0, numerator=0, denominator=0,
                              ci=c.MISSING, na_reason="fixture denominator is zero"), definitions=definitions)
    with pytest.raises(c.ContractError, match="na_reason"):
        replace(row, value=None)
    with pytest.raises(c.ContractError, match="zero denominator"):
        replace(row, denominator=0)
    with pytest.raises(c.ContractError, match="interval"):
        replace(row.ci, upper=0)
    for changed in (replace(row, numerator=5), replace(row, n_evaluated=5), replace(row, ci=replace(row.ci, level=0.90)),
                    replace(row, scope="other"), replace(row, unit="ms"), replace(row, ci=c.MISSING)):
        with pytest.raises(c.ContractError):
            e.validate_metric(changed, definitions=definitions)
    unbounded = replace(row, metric_id="synthetic-latency", value=25, unit="ms",
                        numerator=c.MISSING, denominator=c.MISSING, ci=c.MISSING)
    e.validate_metric(unbounded, definitions={(unbounded.metric_id, unbounded.scope): e.MetricDefinition(unit="ms")})


def test_benchmark_matches_counts_and_approved_profile_hardware_scope(artifacts):
    run, trusted = ef.make_run("E12-A", artifacts)
    row = ef.benchmark(run)
    scope = e.BenchmarkScope(profile_id=row.profile_id, hardware_ref=row.hardware_ref, measurement=row.measurement,
                             scenario=row.scenario, unit=row.unit)
    e.validate_evidence_run_binding(row, run=run, trusted=trusted)
    e.validate_benchmark(row, scopes=(scope,))
    with pytest.raises(c.ContractError, match="count mismatch"):
        replace(row, failure_count=0)
    with pytest.raises(c.ContractError, match="outside run"):
        e.validate_evidence_run_binding(replace(row, profile_id="unknown profile"), run=run, trusted=trusted)
    for changes in ({"hardware_ref": "foreign"}, {"scenario": "foreign"}, {"unit": "seconds"}):
        with pytest.raises(c.ContractError, match="outside scope"):
            e.validate_benchmark(replace(row, **changes), scopes=(scope,))



def test_run_artifacts_are_bound_to_actual_bytes(artifacts):
    run, trusted = ef.make_run("E12-A", artifacts)
    e.validate_run_provenance(run, trusted=trusted)
    e.verify_run_artifacts(run, {"synthetic-capture": b"synthetic capture bytes"})
    with pytest.raises(c.ContractError, match="SHA-256 mismatch"):
        e.verify_run_artifacts(run, {"synthetic-capture": b"changed fixture bytes"})
    with pytest.raises(c.ContractError, match="run inventory"):
        e.verify_run_artifacts(run, {})


def test_trace_run_binding_checks_e14_b_pin_even_without_join(artifacts):
    run, trusted = ef.make_run("E14-B", artifacts)
    trace = ef.trace_for(run, answer=ef.make_stage(run, artifacts))
    result = trace.validated_model_output
    changed = replace(result.context_provenance, kb_sha256=f.digest(b"another KB"))
    row = replace(trace, validated_model_output=replace(result, context_provenance=changed))
    with pytest.raises(c.ContractError, match="approved KB"):
        e.validate_evidence_run_binding(row, run=run, trusted=trusted)


def test_trace_cannot_change_the_captured_failure_code(artifacts):
    run, trusted = ef.make_run("E14-A", artifacts)
    answer = ef.make_stage(run, artifacts, outcome="inference_failure")
    code = c.ErrorCode.INTERNAL_ERROR
    trace = replace(ef.trace_for(run, answer=answer), reason_code=code)
    stage = replace(answer, record=replace(answer.record, error_code=code))
    with pytest.raises(c.ContractError, match="error differs from captured call"):
        validate_e14(trace, run, trusted, stage)


def test_completed_trace_cannot_hide_dispatched_extraction_as_pre_call_block(artifacts):
    run, trusted = ef.make_run("E14-A", artifacts)
    stage = ef.make_stage(run, artifacts, operation=c.Operation.ATTRIBUTES)
    trace = ef.trace_for(run, blocked=True)
    with pytest.raises(c.ContractError, match="captured invocation without a trace"):
        validate_e14(trace, run, trusted, stage)


def test_pair_can_share_extraction_without_passing_attributes_to_the_vqa_call(artifacts, pair_fixture):
    runs, trusted, stages, _, pair = pair_fixture
    extraction = stages[0]
    aa, ab = (ef.make_stage(run, artifacts, operation=c.Operation.VQA) for run in runs)
    traces = tuple(ef.trace_for(run, answer=stage, extraction=extraction) for run, stage in zip(runs, (aa, ab)))
    assert aa.record.consumed_attributes is ab.record.consumed_attributes is None
    rows = (extraction, aa, ab)
    e.validate_e14_evidence(traces, tuple(stage.record for stage in rows), runs=runs, trusted_runs=trusted,
                           captures={stage.record.ref: stage.capture for stage in rows},
                           serialize_attributes=ef.serialize_attributes, pair=pair)



def test_e15_trace_wire_keeps_runtime_result_without_offline_eval_ref(artifacts):
    run, trusted = ef.make_run("E15", artifacts)
    profile = f.make_profile("assistant")
    request = f.make_assistant_request(profile, operation=c.Operation.VQA)
    result = mocks.MockGenerativeAssistant().execute(request, profile=profile)
    trace = replace(
        ef.trace_for(run), reason_code=c.MISSING, validated_model_output=result,
        raw_model_output=c.encode_record(result.generated).decode("utf-8"),
    )
    wire = e.decode_evidence_record(c.encode_record(trace))
    e.validate_evidence_run_binding(wire, run=run, trusted=trusted)
    assert wire.eval_ref == run.eval_ref and wire.validated_model_output.eval_ref is c.MISSING
    assert wire.invocation_refs == ()



def _pin_e14_inputs(images, *, annotations=None, phase=e.RunPhase.FORMAL):
    raw = ef.ndjson(*images)
    questions = ef.ndjson(*(ef.question_row(image.image_id) for image in images if image.question_ids))
    annotation_bytes = None if annotations is None else ef.ndjson(*annotations)
    return e.validate_manifest(
        raw, expected_sha256=f.digest(raw), scope=ef.scope_for(*images, phase=phase),
        questions_raw=questions, questions_sha256=f.digest(questions),
        annotations_raw=annotation_bytes,
        annotations_sha256=None if annotation_bytes is None else f.digest(annotation_bytes),
    )


@pytest.mark.parametrize("experiment", ["E14-A", "E14-B"])
def test_formal_e14_accepts_complete_pinned_inputs_and_nullable_answer(artifacts, experiment):
    annotation = replace(ef.annotation_row(), expected_answer=None, answerability=e.Answerability.ABSTENTION_EXPECTED,
                         source_required=False, expected_source_ids=c.MISSING)
    manifest = _pin_e14_inputs((ef.image_row(),), annotations=(annotation,))
    run, trusted = ef.make_run(experiment, artifacts, manifest=manifest)
    e.validate_run_provenance(run, trusted=trusted)
    stage = ef.make_stage(run, artifacts)
    validate_e14(ef.trace_for(run, answer=stage), run, trusted, stage)


@pytest.mark.parametrize("experiment", ["E14-A", "E14-B"])
@pytest.mark.parametrize("status", list(e.RunExecutionStatus))
def test_formal_e14_rejects_missing_annotations_for_every_run_status(artifacts, experiment, status):
    manifest = _pin_e14_inputs((ef.image_row(),))
    assert manifest.questions_sha256 is not None
    assert manifest.annotations_sha256 is None and not manifest.annotations
    run, trusted = ef.make_run(experiment, artifacts, manifest=manifest, status=status)
    stage = ef.make_stage(run, artifacts)
    for check in (lambda: e.validate_run_provenance(run, trusted=trusted),
                  lambda: validate_e14(ef.trace_for(run, answer=stage), run, trusted, stage)):
        with pytest.raises(c.ContractError, match="annotations artifact hash") as failure:
            check()
        assert failure.value.code is c.ErrorCode.INVALID_SNAPSHOT


@pytest.mark.parametrize("case, message", [
    ("missing-question-hash", "questions artifact hash"),
    ("missing-annotation-hash", "annotations artifact hash"),
    ("missing-annotation-slot", "annotation slots must match"),
    ("extra-annotation-slot", "annotation slots must match"),
    ("question-slot-mismatch", "question slots must match the image manifest"),
    ("wrong-image-type", "VLM evaluation image records"),
])
def test_formal_e14_rechecks_completeness_at_trusted_run_binding(artifacts, case, message):
    manifest = ef.make_manifest(image_ids=(ef.IMAGE_ID, "fixture-second-image"))
    assert len(manifest.questions) == len(manifest.annotations) == 2

    # Build inconsistent in-memory manifests to exercise the separate run boundary.
    if case == "missing-question-hash":
        manifest = replace(manifest, questions_sha256=None)
    elif case == "missing-annotation-hash":
        manifest = replace(manifest, annotations_sha256=None)
    elif case == "missing-annotation-slot":
        first = (ef.IMAGE_ID, ef.QUESTION_ID)
        manifest = replace(manifest, annotations={first: manifest.annotations[first]})
    elif case == "extra-annotation-slot":
        extra = ef.annotation_row("fixture-extra-image")
        manifest = replace(manifest, annotations=dict(manifest.annotations) | {(extra.image_id, extra.question_id): extra})
    elif case == "question-slot-mismatch":
        first = (ef.IMAGE_ID, ef.QUESTION_ID)
        manifest = replace(manifest, questions={first: manifest.questions[first]})
    else:
        pilot = ef.pilot_row()
        manifest = replace(manifest, images={pilot.image_id: pilot})
    run, trusted = ef.make_run("E14-A", artifacts, manifest=manifest)
    with pytest.raises(c.ContractError, match=message) as failure:
        e.validate_run_provenance(run, trusted=trusted)
    assert failure.value.code is c.ErrorCode.INVALID_SNAPSHOT


def test_formal_e14_rejects_annotations_pinned_to_a_different_rubric(artifacts):
    annotation = replace(ef.annotation_row(), rubric_version="other-reviewed-synthetic-rubric")
    manifest = _pin_e14_inputs((ef.image_row(),), annotations=(annotation,))
    run, trusted = ef.make_run("E14-B", artifacts, manifest=manifest)
    with pytest.raises(c.ContractError, match="rubric differs from run rubric") as failure:
        e.validate_run_provenance(run, trusted=trusted)
    assert failure.value.code is c.ErrorCode.INVALID_SNAPSHOT


def test_formal_e14_rejects_empty_cohort_of_question_slots(artifacts):
    manifest = _pin_e14_inputs((ef.image_row(with_question=False),), annotations=())
    assert manifest.images and not manifest.questions
    assert manifest.questions_sha256 is not None and manifest.annotations_sha256 is not None
    run, trusted = ef.make_run("E14-A", artifacts, manifest=manifest)
    with pytest.raises(c.ContractError, match="at least one question slot") as failure:
        e.validate_e14_evidence((), (), runs=(run,), trusted_runs={run.eval_ref: trusted}, captures={})
    assert failure.value.code is c.ErrorCode.INVALID_SNAPSHOT


def test_formal_e14_rejects_an_image_without_questions_in_a_nonempty_cohort(artifacts):
    manifest = _pin_e14_inputs((ef.image_row(), ef.image_row("fixture-image-without-question", with_question=False)),
                               annotations=(ef.annotation_row(),))
    assert len(manifest.images) == 2 and len(manifest.questions) == len(manifest.annotations) == 1
    run, trusted = ef.make_run("E14-A", artifacts, manifest=manifest)
    with pytest.raises(c.ContractError, match="question slots for every image") as failure:
        e.validate_run_provenance(run, trusted=trusted)
    assert failure.value.code is c.ErrorCode.INVALID_SNAPSHOT


@pytest.mark.parametrize("status", [e.RunExecutionStatus.FAILED, e.RunExecutionStatus.ABORTED])
def test_formal_e14_accepts_partial_traces_when_inputs_are_complete(artifacts, status):
    manifest = ef.make_manifest(image_ids=(ef.IMAGE_ID, "fixture-image-without-trace"))
    run, trusted = ef.make_run("E14-A", artifacts, manifest=manifest, status=status)
    stage = ef.make_stage(run, artifacts)
    trace = ef.trace_for(run, answer=stage)
    assert set(manifest.questions) - {(trace.image_id, trace.question_id)}
    validate_e14(trace, run, trusted, stage)


@pytest.mark.parametrize("experiment", ["E12-A", "E13-B", "E15"])
def test_formal_non_e14_runs_do_not_require_vqa_inputs(artifacts, experiment):
    manifest = ef.make_manifest(vqa=False)
    assert not manifest.questions and manifest.annotations_sha256 is None
    run, trusted = ef.make_run(experiment, artifacts, manifest=manifest)
    e.validate_run_provenance(run, trusted=trusted)
    if experiment != "E15":
        e.validate_classification_cohort((ef.prediction(run, artifacts),), run=run, trusted=trusted)


@pytest.mark.parametrize("phase", [e.RunPhase.PILOT, e.RunPhase.EXPLORATORY])
def test_nonformal_e14_can_use_questions_without_annotation_artifacts(artifacts, phase):
    manifest = _pin_e14_inputs((ef.image_row(),), phase=phase)
    run, trusted = ef.make_run("E14-A", artifacts, manifest=manifest, phase=phase)
    stage = ef.make_stage(run, artifacts)
    validate_e14(ef.trace_for(run, answer=stage), run, trusted, stage)
