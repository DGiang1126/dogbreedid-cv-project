"""Phase 3.2 contract invariants on synthetic in-memory data."""
from dataclasses import FrozenInstanceError, replace
import inspect
import json
from types import SimpleNamespace

import pytest

from src.vlm import contracts as c

from . import fixtures as f
from . import mocks as m


@pytest.fixture(scope="module")
def artifacts():
    return f.make_artifacts()


def mutate_blob(raw, change):
    payload = json.loads(raw)
    change(payload)
    return f.json_bytes(payload)


def knowledge_from(raw, artifacts, reviews=f.REVIEWED_SOURCES):
    return c.validate_knowledge_snapshot(
        raw, mapping=artifacts.mapping, expected_sha256=f.digest(raw),
        reviewed_sources=reviews, allowed_attribute_ids={f.KB_ATTRIBUTE_ID},
    )


def prompts_from(raw, artifacts):
    return c.validate_prompt_snapshot(
        raw, mapping=artifacts.mapping, expected_sha256=f.digest(raw), knowledge=artifacts.knowledge,
    )


@pytest.mark.parametrize("raw", [
    b'{"nest":{"key":1,"key":2}}',
    b'{"key":NaN}', b'{"key":Infinity}', b'{"key":-Infinity}', b'{"key":1e999}',
    b'{"key":"\xff"}', b'\xef\xbb\xbf{}', b'{"key":',
])
def test_strict_json_rejects_invalid_or_nonfinite_input(raw):
    with pytest.raises(c.ContractError):
        c.decode_json(raw)


@pytest.mark.parametrize("bad", [True, False, 7.0, "7", -1, 30, None])
def test_class_id_requires_integer_in_range(bad):
    with pytest.raises(c.ContractError, match="class_id"):
        c.validate_class_id(bad)
    assert c.validate_class_id(0) == 0 and c.validate_class_id(29) == 29


@pytest.mark.parametrize("bad", ["a" * 63, "a" * 65, "A" * 64, "z" * 64, None])
def test_hash_format_is_lowercase_hex(bad):
    with pytest.raises(c.ContractError, match="SHA-256"):
        c.validate_sha256(bad)


def test_hash_verification_uses_exact_bytes_and_rgb_keeps_upload_digest():
    uploaded = f.SOURCE_BYTES
    image = f.make_image()
    assert image.image_digest == f.digest(uploaded)
    assert f.digest(image.pixels) != image.image_digest
    c.verify_sha256(uploaded, image.image_digest)
    with pytest.raises(c.ContractError) as exc:
        c.verify_sha256(uploaded + b"\n", image.image_digest)
    assert exc.value.code is c.ErrorCode.HASH_MISMATCH
    with pytest.raises(c.ContractError):
        replace(image, pixels=image.pixels[:-1])
    with pytest.raises(c.ContractError):
        replace(image, width=True)


@pytest.mark.parametrize("change", [
    lambda p: p.pop("schema_version"),
    lambda p: p.update(schema_version="2.0"),
    lambda p: p.update(schema_version=None),
    lambda p: p.update(unknown=True),
    lambda p: p.pop("status"),
    lambda p: p.update(latency_ms=-1),
    lambda p: p.update(latency_ms=True),
    lambda p: p.update(latency_ms="0"),
    lambda p: p.update(eval_ref=None),
    lambda p: p["truncation"].update(unknown=True),
])
def test_wire_requires_fields_and_rejects_unknown_fields_and_coercion(artifacts, change):
    result = f.make_retrieval_result(artifacts.knowledge)
    raw = c.encode_record(result)
    assert c.decode_record(raw, c.RetrievalResult) == result
    with pytest.raises(c.ContractError):
        c.decode_record(mutate_blob(raw, change), c.RetrievalResult)


def test_unsupported_version_and_required_nullable_fields(artifacts):
    raw = c.encode_record(f.make_retrieval_result(artifacts.knowledge))
    with pytest.raises(c.ContractError) as exc:
        c.decode_record(mutate_blob(raw, lambda p: p.update(schema_version="1.1")), c.RetrievalResult)
    assert exc.value.code is c.ErrorCode.UNSUPPORTED_SCHEMA
    base = {"attribute_id": "ear_shape", "visibility": "unknown", "value": None}
    assert c.decode_record(f.json_bytes(base), c.AttributeObservation).value is None
    del base["value"]
    with pytest.raises(c.ContractError, match="missing value"):
        c.decode_record(f.json_bytes(base), c.AttributeObservation)
    cnn = c.decode_record(b'{"status":"ACCEPT"}', c.AllowedCNNContext)
    assert cnn.class_id is c.MISSING and json.loads(c.encode_record(cnn)) == {"status": "ACCEPT"}
    with pytest.raises(c.ContractError):
        c.decode_record(b'{"status":"ACCEPT","class_id":null}', c.AllowedCNNContext)


@pytest.mark.parametrize("run_id", [
    "E12-B_pilot_20990101T120000Z_123e4567-e89b-42d3-a456-426614174000",
    "E12-A_demo_20990101T120000Z_123e4567-e89b-42d3-a456-426614174000",
    "E12-A_pilot_20990230T120000Z_123e4567-e89b-42d3-a456-426614174000",
    "E12-A_pilot_20990101T120000_123e4567-e89b-42d3-a456-426614174000",
    "E12-A_pilot_20990101T120000Z_123e4567-e89b-12d3-a456-426614174000",
])
def test_eval_ref_requires_matching_id_phase_actual_utc_date_and_uuid4(run_id):
    with pytest.raises(c.ContractError):
        c.EvalRef(experiment_id="E12-A", run_id=run_id)
    valid = f.make_eval_ref()
    assert c.decode_record(c.encode_record(valid), c.EvalRef) == valid


def test_deep_immutability_copies_nested_collections(artifacts):
    original = ["mini poodle"]
    aliases = {"en": original, "vi": []}
    breed = replace(artifacts.knowledge.data.breeds[7], aliases=aliases)
    original.append("changed")
    aliases["vi"].append("thay đổi")
    assert breed.aliases["en"] == ("mini poodle",) and breed.aliases["vi"] == ()
    with pytest.raises(TypeError):
        breed.aliases["vi"] = ("changed",)
    with pytest.raises(FrozenInstanceError):
        breed.canonical_name = "changed"
    maps = dict(artifacts.mapping.names)
    canonical = replace(artifacts.mapping, names=maps)
    maps[0] = "changed"
    assert canonical.names[0] == f.CANONICAL_NAMES[0]
    limits = {"fixture": 2}
    profile = replace(f.make_profile(), limits=limits)
    limits["fixture"] = 9
    assert profile.limits["fixture"] == 2
    image = f.make_image()
    assert type(image.pixels) is bytes
    values = ["black", "white"]
    coat = c.AttributeObservation(attribute_id="coat_color", visibility=c.Visibility.OBSERVED, value=values)
    values.clear()
    assert coat.value == ("black", "white")


def test_canonical_mapping_requires_exact_keys_and_unique_names(artifacts):
    raw = mutate_blob(artifacts.mapping_raw, lambda p: p.update({"00": p.pop("0")}))
    with pytest.raises(c.ContractError):
        c.decode_canonical_mapping(raw, expected_sha256=f.digest(raw))
    duplicate = mutate_blob(artifacts.mapping_raw, lambda p: p.update({"0": p["1"]}))
    with pytest.raises(c.ContractError, match="unique"):
        c.decode_canonical_mapping(duplicate, expected_sha256=f.digest(duplicate))
    assert tuple(artifacts.mapping.names.values()) == f.CANONICAL_NAMES


@pytest.mark.parametrize("change", [
    lambda p: p["breeds"].pop(),
    lambda p: p["breeds"][0].update(class_id=1),
    lambda p: p["breeds"].reverse(),
    lambda p: p["breeds"][7].update(canonical_name="Mini Poodle"),
    lambda p: p["breeds"][7]["attributes"][0].update(source_ids=["missing"]),
    lambda p: p["breeds"][7]["attributes"][0].update(attribute_id="unknown"),
    lambda p: p["breeds"][7]["aliases"].update(vi=["Thú", "Thu\u0301"]),
    lambda p: p["breeds"][7].update(expected_answer="leak"),
    lambda p: p["sources"][7].update(review_status="pending"),
    lambda p: p["sources"][7].pop("review_ref"),
    lambda p: p.update(mapping_sha256=f.digest(b"wrong mapping")),
])
def test_kb_snapshot_rejects_bad_canonical_data_sources_and_scorer_fields(artifacts, change):
    raw = mutate_blob(artifacts.knowledge_raw, change)
    with pytest.raises(c.ContractError):
        knowledge_from(raw, artifacts)


def test_kb_binds_hash_and_independent_review_evidence_and_keeps_collisions(artifacts):
    assert knowledge_from(artifacts.knowledge_raw, artifacts) == artifacts.knowledge
    assert artifacts.knowledge.kb_sha256 == f.digest(artifacts.knowledge_raw)
    with pytest.raises(c.ContractError) as exc:
        knowledge_from(artifacts.knowledge_raw, artifacts, {})
    assert exc.value.code is c.ErrorCode.UNREVIEWED_SOURCE
    with pytest.raises(c.ContractError) as exc:
        c.validate_knowledge_snapshot(
            artifacts.knowledge_raw + b"\n", mapping=artifacts.mapping,
            expected_sha256=artifacts.knowledge.kb_sha256,
            reviewed_sources=f.REVIEWED_SOURCES, allowed_attribute_ids={f.KB_ATTRIBUTE_ID},
        )
    assert exc.value.code is c.ErrorCode.HASH_MISMATCH
    breeds = artifacts.knowledge.data.breeds
    assert breeds[6].aliases["en"] == breeds[7].aliases["en"] == ("shared fixture alias",)


@pytest.mark.parametrize("bank,change", [
    ("name_raw", lambda p: p.update(kb_version=None)),
    ("name_raw", lambda p: p["entries"][0].update(attribute_ids=[])),
    ("name_raw", lambda p: p["entries"][0].update(prompt_text=" \t")),
    ("name_raw", lambda p: p["entries"][0].update(canonical_name="eskimo dog")),
    ("name_raw", lambda p: p["entries"][0].update(class_id=1)),
    ("morphology_raw", lambda p: p.pop("kb_sha256")),
    ("morphology_raw", lambda p: p.update(kb_sha256=f.digest(b"wrong KB"))),
    ("morphology_raw", lambda p: p.update(kb_version="wrong version")),
    ("morphology_raw", lambda p: p["entries"][7].update(source_ids=["fixture-source-8"])),
    ("morphology_raw", lambda p: p["entries"][7].update(attribute_ids=["unknown"])),
    ("morphology_raw", lambda p: p["entries"][7].update(prompt_text="no canonical breed")),
])
def test_prompt_shape_and_provenance_are_enforced(artifacts, bank, change):
    with pytest.raises(c.ContractError):
        prompts_from(mutate_blob(getattr(artifacts, bank), change), artifacts)


def test_both_prompt_banks_bind_actual_bytes(artifacts):
    assert prompts_from(artifacts.name_raw, artifacts) == artifacts.name
    assert prompts_from(artifacts.morphology_raw, artifacts) == artifacts.morphology
    with pytest.raises(c.ContractError) as exc:
        c.validate_prompt_snapshot(
            artifacts.name_raw + b" ", mapping=artifacts.mapping,
            expected_sha256=artifacts.name.prompt_bank_sha256, knowledge=None,
        )
    assert exc.value.code is c.ErrorCode.HASH_MISMATCH


@pytest.mark.parametrize("change", [
    lambda p: p["audits"][0].update(prompt_bank_sha256=f.digest(b"wrong bank")),
    lambda p: p["audits"][0].update(profile_id="different profile"),
    lambda p: p["audits"][0].update(tokenizer_revision="different revision"),
    lambda p: p["audits"][0]["entries"][0].update(truncated=True),
    lambda p: p["audits"][0]["entries"][0].update(token_count=33),
    lambda p: p["audits"][0]["entries"][0].update(token_count=0),
    lambda p: p["audits"][0]["entries"][0].update(class_id=1),
    lambda p: p["audits"].append(p["audits"][0]),
])
def test_token_audit_rejects_bad_bindings_and_truncation(artifacts, change):
    profile = f.make_profile()
    audit = f.make_token_audit(artifacts.name, profile)
    c.validate_prompt_token_audit(audit, bank=artifacts.name, profile=profile)
    malformed = mutate_blob(c.encode_record(audit), change)
    with pytest.raises(c.ContractError):
        c.validate_prompt_token_audit(c.decode_record(malformed, c.PromptTokenAudit),
                                      bank=artifacts.name, profile=profile)


@pytest.mark.parametrize("audit_limit, token_count, truncated, error_code", [
    (32, 32, False, None),
    (128, 80, False, c.ErrorCode.INVALID_SNAPSHOT),
    (31, 6, False, c.ErrorCode.INVALID_SNAPSHOT),
    (32, 33, False, c.ErrorCode.TOKEN_TRUNCATION),
    (32, 6, True, c.ErrorCode.TOKEN_TRUNCATION),
])
def test_prompt_audit_enforces_trusted_profile_limit(artifacts, audit_limit, token_count, truncated, error_code):
    profile = f.make_profile()
    audit = f.make_token_audit(artifacts.name, profile)
    item = audit.audits[0]
    entries = (replace(item.entries[0], token_count=token_count, truncated=truncated), *item.entries[1:])
    audit = replace(audit, audits=(replace(item, token_limit=audit_limit, entries=entries),))
    if error_code is None:
        c.validate_prompt_token_audit(audit, bank=artifacts.name, profile=profile)
    else:
        with pytest.raises(c.ContractError) as failure:
            c.validate_prompt_token_audit(audit, bank=artifacts.name, profile=profile)
        assert failure.value.code is error_code


@pytest.mark.parametrize("limits", [{}, {"prompt_token_limit": 0}, {"prompt_token_limit": 32.0}])
def test_prompt_audit_rejects_missing_nonpositive_or_float_trusted_limit(artifacts, limits):
    profile = f.make_profile()
    audit = f.make_token_audit(artifacts.name, profile)
    with pytest.raises(c.ContractError, match="positive integer") as failure:
        c.validate_prompt_token_audit(audit, bank=artifacts.name, profile=replace(profile, limits=limits))
    assert failure.value.code is c.ErrorCode.INVALID_SNAPSHOT


@pytest.mark.parametrize("value", [True, "32"])
def test_profile_rejects_boolean_or_string_token_limit(value):
    with pytest.raises(c.ContractError) as failure:
        replace(f.make_profile(), limits={"prompt_token_limit": value})
    assert failure.value.code is c.ErrorCode.INVALID_INPUT


@pytest.mark.parametrize("status", list(c.RetrievalStatus))
def test_retrieval_mock_all_statuses_remain_consistent(artifacts, status):
    knowledge = artifacts.knowledge
    request = f.make_retrieval_request(knowledge)
    if status not in (c.RetrievalStatus.HIT, c.RetrievalStatus.PARTIAL_HIT):
        request = replace(request, explicit_class_ids=(), lookup_applicable=status is not c.RetrievalStatus.NOT_APPLICABLE)
    result = f.make_retrieval_result(knowledge, status)
    counter = m.MockContextTokenCounter({f.CONTEXT: 8})
    scripted = m.MockKnowledgeRetriever(result)(request, knowledge=knowledge, count_context_tokens=counter)
    wire = c.decode_record(c.encode_record(scripted), c.RetrievalResult)
    validated = c.validate_retrieval_result(wire, request=request, knowledge=knowledge, token_counter=counter,
                                            render_context=f.render_context,
                                            policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED)
    assert validated.result == wire == result
    assert (wire.error is not c.MISSING) == (status is c.RetrievalStatus.RETRIEVAL_ERROR)


@pytest.mark.parametrize("change", [
    lambda p: p.update(status="hit"),
    lambda p: p.update(resolved_class_ids=[8]),
    lambda p: p.update(context=""),
    lambda p: p.update(entries=[]),
    lambda p: p.update(source_ids=["wrong"]),
    lambda p: p["per_breed"][1].update(candidate_class_ids=[7, 6, 6]),
    lambda p: p["per_breed"][1].update(class_id=7),
    lambda p: p["truncation"].update(used_tokens=21),
    lambda p: p["truncation"].update(applied=True),
    lambda p: p.update(error={"code":"RETRIEVAL_ERROR","message_vi":"Lỗi","retryable":True}),
])
def test_retrieval_wire_requires_consistency(artifacts, change):
    result = f.make_retrieval_result(artifacts.knowledge, c.RetrievalStatus.PARTIAL_HIT)
    with pytest.raises(c.ContractError):
        c.decode_record(mutate_blob(c.encode_record(result), change), c.RetrievalResult)


@pytest.mark.parametrize("bad", [True, False, -1, 1.0, "8"])
def test_token_counter_rejects_non_integer_or_negative_values(bad):
    with pytest.raises(c.ContractError, match="nonnegative integer"):
        c.count_context_tokens(lambda _: bad, f.CONTEXT)


def test_retrieval_binds_policy_kb_explicit_ids_and_budget(artifacts):
    knowledge = artifacts.knowledge
    request = f.make_retrieval_request(knowledge)
    result = f.make_retrieval_result(knowledge)
    opts = dict(request=request, knowledge=knowledge, token_counter=m.MockContextTokenCounter({f.CONTEXT: 8}),
                render_context=f.render_context, policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED)
    c.validate_retrieval_result(result, **opts)
    for override in ({"policy_mode": None},
                     {"policy_mode": c.PolicyMode.OBSERVATION_ONLY},
                     {"token_counter": m.MockContextTokenCounter({f.CONTEXT: 9})},
                     {"request": replace(request, explicit_class_ids=(8,))},
                     {"request": replace(request, context_budget=replace(request.context_budget, max_tokens=19))}):
        with pytest.raises(c.ContractError):
            c.validate_retrieval_result(result, **(opts | override))
    with pytest.raises(c.ContractError):
        c.validate_retrieval_result(replace(result, kb_sha256=f.digest(b"wrong KB")), **opts)
    fake_attribute = replace(result.entries[0].attributes[0], value="edited synthetic value")
    fake_entry = replace(result.entries[0], attributes=(fake_attribute,))
    with pytest.raises(c.ContractError, match="pinned KB"):
        c.validate_retrieval_result(replace(result, entries=(fake_entry,)), **opts)


@pytest.mark.parametrize("context", [
    f.CONTEXT.replace("synthetic value", "forged fixture claim"),
    f.CONTEXT.replace("fixture-source-7", "foreign-source"),
    f.CONTEXT + " ",
])
def test_retrieval_rejects_text_outside_trusted_render_even_with_matching_count(artifacts, context):
    knowledge = artifacts.knowledge
    scripted = f.make_retrieval_result(knowledge)
    forged = replace(scripted, context=context)
    wire = c.decode_record(c.encode_record(forged), c.RetrievalResult)
    counter = m.MockContextTokenCounter({context: scripted.truncation.used_tokens})
    assert c.count_context_tokens(counter, context) == scripted.truncation.used_tokens
    assert (wire.entries, wire.source_ids, wire.kb_sha256) == (scripted.entries, scripted.source_ids, scripted.kb_sha256)
    with pytest.raises(c.ContractError, match="trusted rendered entries") as failure:
        c.validate_retrieval_result(wire, request=f.make_retrieval_request(knowledge), knowledge=knowledge,
                                    token_counter=counter, render_context=f.render_context,
                                    policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED)
    assert failure.value.code is c.ErrorCode.INVALID_OUTPUT


@pytest.mark.parametrize("status", [c.RetrievalStatus.MISS, c.RetrievalStatus.AMBIGUOUS,
                                    c.RetrievalStatus.NOT_APPLICABLE, c.RetrievalStatus.RETRIEVAL_ERROR])
def test_retrieval_without_hits_binds_empty_context_and_sources(artifacts, status):
    knowledge = artifacts.knowledge
    scripted = f.make_retrieval_result(knowledge, status)
    request = replace(f.make_retrieval_request(knowledge), explicit_class_ids=(),
                      lookup_applicable=status is not c.RetrievalStatus.NOT_APPLICABLE)

    def unexpected_render(entries):
        pytest.fail(f"renderer invoked without entries: {entries}")

    validated = c.validate_retrieval_result(
        scripted, request=request, knowledge=knowledge, token_counter=m.MockContextTokenCounter({}),
        render_context=unexpected_render, policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED,
    )
    assert validated.result.context == "" and validated.result.source_ids == ()
    with pytest.raises(c.ContractError):
        replace(scripted, context="no trusted entry for this text")
    with pytest.raises(c.ContractError):
        replace(scripted, source_ids=("fixture-source-7",))


def test_retrieval_budget_miss_retains_resolved_id(artifacts):
    knowledge = artifacts.knowledge
    request = replace(f.make_retrieval_request(knowledge),
                      context_budget=c.ContextBudget(max_tokens=0, tokenizer_revision="fixture-tokenizer"))
    miss = replace(f.make_retrieval_result(knowledge, c.RetrievalStatus.MISS),
                   per_breed=(c.BreedLookup(mention=None, class_id=7, status=c.BreedLookupStatus.MISS,
                                           candidate_class_ids=(), reason="budget_exhausted"),),
                   resolved_class_ids=(7,),
                   truncation=c.TruncationInfo(applied=True, budget_tokens=0, used_tokens=0,
                                               omitted_attribute_count=1))
    c.validate_retrieval_result(miss, request=request, knowledge=knowledge,
                                token_counter=m.MockContextTokenCounter({}), render_context=f.render_context,
                                policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED)
    with pytest.raises(c.ContractError, match="resolved miss"):
        replace(miss.per_breed[0], reason=c.MISSING)


@pytest.mark.parametrize("unresolved_status", [c.BreedLookupStatus.MISS, c.BreedLookupStatus.AMBIGUOUS])
def test_cnn_fallback_cannot_accompany_unresolved_mentions(artifacts, unresolved_status):
    knowledge = artifacts.knowledge
    cnn = c.AllowedCNNContext(status=c.CNNStatus.ACCEPT, class_id=7,
                              canonical_name=knowledge.data.breeds[7].canonical_name)
    request = replace(f.make_retrieval_request(knowledge), cnn_context=cnn, explicit_class_ids=())
    result = f.make_retrieval_result(knowledge, c.RetrievalStatus.PARTIAL_HIT)
    fallback = replace(result.per_breed[0], mention=None)
    unresolved = c.BreedLookup(mention="unresolved fixture mention", class_id=None, status=unresolved_status,
                               candidate_class_ids=(6, 8) if unresolved_status is c.BreedLookupStatus.AMBIGUOUS else ())
    options = dict(request=request, knowledge=knowledge, token_counter=m.MockContextTokenCounter({f.CONTEXT: 8}),
                   render_context=f.render_context, policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED)
    c.validate_retrieval_result(replace(result, status=c.RetrievalStatus.HIT, per_breed=(fallback,)), **options)
    mixed = replace(result, per_breed=(fallback, unresolved))
    with pytest.raises(c.ContractError, match="CNN fallback") as exc:
        c.validate_retrieval_result(mixed, **options)
    assert exc.value.code is c.ErrorCode.POLICY_BLOCKED
    c.validate_retrieval_result(mixed, **(options | {"request": replace(request, explicit_class_ids=(7,))}))


@pytest.fixture
def classifier(artifacts):
    profile = f.make_profile()
    request = f.make_zero_shot_request(profile, artifacts.name)
    result = m.MockZeroShotClassifier().classify(request, profile=profile, prompts=artifacts.name)
    return profile, request, result


def test_classifier_ranking_breaks_top1_and_top5_boundary_ties(artifacts, classifier):
    profile, request, result = classifier
    assert c.rank_scores(f.TIED_SCORES)[:6] == (1, 2, 0, 3, 4, 5)
    assert result.top1_class_id == 1 and result.top5_class_ids == (1, 2, 0, 3, 4)
    assert result.calibrated is False
    assert c.decode_record(c.encode_record(result), c.ZeroShotResult) == result
    c.validate_zero_shot_result(result, request=request, profile=profile, prompts=artifacts.name)


@pytest.mark.parametrize("change", [
    lambda p: p.update(scores=p["scores"][:-1]),
    lambda p: p["scores"].append(0),
    lambda p: p["scores"].__setitem__(0, float("nan")),
    lambda p: p["scores"].__setitem__(0, float("inf")),
    lambda p: p["scores"].__setitem__(0, True),
    lambda p: p["scores"].__setitem__(0, "0"),
    lambda p: p.update(top1_class_id=2),
    lambda p: p.update(top5_class_ids=[1, 2, 0, 3, 5]),
    lambda p: p.update(top5_class_ids=[1, 2, 0, 3, 3]),
    lambda p: p.update(top5_class_ids=[1, 2, 0, 3, 4.0]),
    lambda p: p.update(calibrated=True),
    lambda p: p.update(calibrated=0),
    lambda p: p.update(score_type="probability"),
    lambda p: p.pop("score_type"),
    lambda p: p["model"].update(model_revision="main"),
])
def test_classifier_rejects_bad_wire_ranking_and_score_semantics(classifier, change):
    _, _, result = classifier
    with pytest.raises(c.ContractError):
        c.decode_record(mutate_blob(c.encode_record(result), change), c.ZeroShotResult)


@pytest.mark.parametrize("field,value", [
    ("image_digest", f.digest(b"different image")),
    ("profile_id", "other profile"),
    ("mapping_sha256", f.digest(b"different mapping")),
    ("prompt_bank_sha256", f.digest(b"different bank")),
    ("model", f.make_profile("other model").model),
])
def test_classifier_rejects_result_without_matching_trusted_bindings(artifacts, classifier, field, value):
    profile, request, result = classifier
    bad_result = replace(result, **{field: value})
    with pytest.raises(c.ContractError):
        c.validate_zero_shot_result(bad_result, request=request, profile=profile, prompts=artifacts.name)


def test_classifier_checks_request_snapshot_and_offline_experiment(artifacts, classifier):
    profile, request, result = classifier
    with pytest.raises(c.ContractError, match="validated RGB"):
        replace(request, image_digest=f.digest(b"wrong"))
    with pytest.raises(c.ContractError, match="trusted snapshot"):
        c.validate_zero_shot_request(request, profile=profile, prompts=artifacts.morphology)
    with pytest.raises(c.ContractError, match="trusted profile"):
        c.validate_zero_shot_request(request, profile=f.make_profile("other"), prompts=artifacts.name)
    with pytest.raises(c.ContractError, match="adapter"):
        replace(request, eval_ref=f.make_eval_ref("E14-A"))
    with pytest.raises(c.ContractError, match="offline"):
        c.validate_zero_shot_request(request, profile=profile, prompts=artifacts.name, offline=True)
    offline = replace(request, eval_ref=f.make_eval_ref("E12-A"))
    offline_result = m.MockZeroShotClassifier().classify(offline, profile=profile, prompts=artifacts.name)
    c.validate_zero_shot_result(offline_result, request=offline, profile=profile, prompts=artifacts.name, offline=True)
    with pytest.raises(c.ContractError, match="offline"):
        c.validate_zero_shot_request(offline, profile=profile, prompts=artifacts.name)
    with pytest.raises(c.ContractError, match="prompt bank type"):
        c.validate_zero_shot_request(replace(offline, eval_ref=f.make_eval_ref("E12-B")),
                                     profile=profile, prompts=artifacts.name, offline=True)
    with pytest.raises(c.ContractError, match="eval_ref mismatch"):
        c.validate_zero_shot_result(result, request=offline, profile=profile, prompts=artifacts.name, offline=True)


def test_classifier_binding_requires_validated_result_not_duck_typed_metadata(artifacts, classifier):
    profile, request, result = classifier
    impostor = SimpleNamespace(
        eval_ref=result.eval_ref, image_digest=result.image_digest, profile_id=result.profile_id,
        model=result.model, mapping_sha256=result.mapping_sha256,
        prompt_bank_sha256=result.prompt_bank_sha256, scores=[float("inf")],
    )
    with pytest.raises(c.ContractError, match="ZeroShotResult"):
        c.validate_zero_shot_result(impostor, request=request, profile=profile, prompts=artifacts.name)


def test_classifier_error_omits_every_scoring_field(artifacts, classifier):
    profile, request, result = classifier
    failed = m.MockZeroShotClassifier(fail=True).classify(request, profile=profile, prompts=artifacts.name)
    c.validate_zero_shot_result(failed, request=request, profile=profile, prompts=artifacts.name)
    payload = json.loads(c.encode_record(failed))
    assert payload["status"] == "unavailable" and payload["error"]["code"] == "MODEL_UNAVAILABLE"
    assert not {"scores", "score_type", "top1_class_id", "top5_class_ids"} & payload.keys()
    with pytest.raises(c.ContractError):
        replace(failed, scores=f.TIED_SCORES)
    with pytest.raises(c.ContractError):
        replace(failed, error=c.MISSING)
    with pytest.raises(c.ContractError):
        replace(result, error=failed.error)


@pytest.mark.parametrize("attribute,visibility,value", [
    ("ear_position", "observed", "erect"),
    ("ear_shape", "observed", "triangular"),
    ("coat_color", "observed", ["black", "white"]),
    ("coat_length", "observed", "long"),
    ("coat_texture", "observed", "wiry"),
    ("muzzle_length", "observed", "medium"),
    ("tail_shape", "observed", "curled"),
    ("body_build", "observed", "stocky"),
    ("coat_color", "unknown", None),
    ("ear_shape", "not_visible", None),
    ("ear_position", "not_applicable", None),
])
def test_attributes_accept_exact_vocab_and_visibility(attribute, visibility, value):
    raw = f.json_bytes(dict(attribute_id=attribute, visibility=visibility, value=value))
    observation = c.decode_record(raw, c.AttributeObservation)
    assert observation.visibility.value == visibility
    assert json.loads(c.encode_record(observation))["value"] == value


@pytest.mark.parametrize("attribute,visibility,value", [
    ("ear_position", "observed", None),
    ("ear_position", "observed", ["erect"]),
    ("ear_position", "observed", "Erect"),
    ("ear_position", "unknown", "erect"),
    ("ear_shape", "not_visible", "triangular"),
    ("ear_shape", "not_applicable", "triangular"),
    ("ear_shape", "unseen", None),
    ("coat_color", "observed", "black"),
    ("coat_color", "observed", []),
    ("coat_color", "observed", ["black", "black"]),
    ("coat_color", "observed", ["white", "black"]),
    ("coat_color", "observed", ["black", "violet"]),
    ("breed", "observed", "Eskimo Dog"),
])
def test_invalid_attribute_shapes_and_vocab_are_rejected(attribute, visibility, value):
    raw = f.json_bytes(dict(attribute_id=attribute, visibility=visibility, value=value))
    with pytest.raises(c.ContractError):
        c.decode_record(raw, c.AttributeObservation)


@pytest.mark.parametrize("operation", list(c.Operation))
def test_assistant_mock_operations_and_model_only_json(operation):
    profile = f.make_profile("assistant")
    request = f.make_assistant_request(profile, operation=operation)
    adapter = m.MockGenerativeAssistant()
    method = adapter.extract_attributes if operation is c.Operation.ATTRIBUTES else adapter.answer_question
    result = method(request, profile=profile)
    c.validate_assistant_result(result, request=request, profile=profile,
                                policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED)
    assert c.decode_record(c.encode_record(result), c.AssistantResult) == result
    assert c.decode_generated_content(c.encode_record(result.generated), request=request) == result.generated
    assert (result.generated.answer_vi is None) == (operation is c.Operation.ATTRIBUTES)
    assert (not result.generated.attributes) == (operation is c.Operation.VQA)
    with pytest.raises(c.ContractError, match="question"):
        replace(request, question="extra" if request.question is None else None)


@pytest.mark.parametrize("change", [
    lambda p: p.update(disposition="not_applicable", answer_vi=None),
    lambda p: p.update(disposition="answered", answer_vi=None),
    lambda p: p.update(disposition="abstained", abstention_reason=None),
    lambda p: p.update(abstention_reason="extra"),
    lambda p: p.update(source_ids=["not-retrieved"]),
    lambda p: p["attributes"].append(p["attributes"][0]),
    lambda p: p.update(profile_id="forged"),
    lambda p: p.update(model={"model_id":"forged"}),
    lambda p: p.update(status="ok"),
    lambda p: p.update(answer_vi="\ud800"),
])
def test_assistant_rejects_invalid_or_forged_raw_generation(change):
    profile = f.make_profile("assistant")
    request = f.make_assistant_request(profile)
    result = m.MockGenerativeAssistant().execute(request, profile=profile)
    payload = json.loads(c.encode_record(result.generated))
    change(payload)
    raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    with pytest.raises(c.ContractError):
        c.decode_generated_content(raw, request=request)


def test_assistant_abstention_and_error_are_distinct():
    profile = f.make_profile("assistant")
    request = f.make_assistant_request(profile)
    ok = m.MockGenerativeAssistant().execute(request, profile=profile)
    abstained = replace(ok.generated, disposition=c.Disposition.ABSTAINED, abstention_reason="Không đủ bằng chứng")
    assert c.decode_generated_content(c.encode_record(abstained), request=request).disposition is c.Disposition.ABSTAINED
    result = m.MockGenerativeAssistant(fail=True).execute(request, profile=profile)
    c.validate_assistant_result(result, request=request, profile=profile, policy_mode=request.policy_mode)
    assert "generated" not in json.loads(c.encode_record(result))
    with pytest.raises(c.ContractError):
        replace(result, generated=abstained)


def test_assistant_binds_context_sources_and_offline_scope(artifacts):
    profile = f.make_profile("assistant")
    lookup = f.make_retrieval_result(artifacts.knowledge)
    request = f.make_assistant_request(profile, retrieval=lookup, eval_ref=f.make_eval_ref("E14-B"))
    bound = c.validate_retrieval_result(
        lookup, request=f.make_retrieval_request(artifacts.knowledge), knowledge=artifacts.knowledge,
        token_counter=m.MockContextTokenCounter({f.CONTEXT: 8}), render_context=f.render_context,
        policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED,
    )
    opts = dict(profile=profile, policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED, retrieval=bound, offline=True)
    c.validate_assistant_request(request, **opts)
    result = m.MockGenerativeAssistant().execute(request, profile=profile)
    c.validate_assistant_result(result, request=request, **opts)
    assert result.generated.source_ids == lookup.source_ids
    for override in ({"retrieval": None}, {"offline": False}, {"policy_mode": c.PolicyMode.OBSERVATION_ONLY}):
        with pytest.raises(c.ContractError):
            c.validate_assistant_request(request, **(opts | override))
    with pytest.raises(c.ContractError, match="sources"):
        c.validate_assistant_request(replace(request, allowed_source_ids=("fabricated",)), **opts)
    with pytest.raises(c.ContractError, match="context SHA-256"):
        replace(request, kb_context=request.kb_context + " ")
    with pytest.raises(c.ContractError, match="together"):
        replace(request.context_provenance, kb_sha256=None)
    with pytest.raises(c.ContractError, match="E14-A"):
        replace(request, eval_ref=f.make_eval_ref("E14-A"))
    with pytest.raises(c.ContractError) as exc:
        replace(request, policy_mode=c.PolicyMode.OBSERVATION_ONLY)
    assert exc.value.code is c.ErrorCode.POLICY_BLOCKED
    with pytest.raises(c.ContractError):
        c.validate_assistant_result(replace(result, model=f.make_profile("other").model),
                                    request=request, **opts)
    with pytest.raises(c.ContractError, match="context binding"):
        c.validate_assistant_result(replace(result, context_provenance=f.make_assistant_request(profile).context_provenance),
                                    request=request, **opts)


def test_e14_a_accepts_no_kb_and_rejects_even_an_empty_pinned_context(artifacts):
    profile = f.make_profile("assistant")
    eval_ref = f.make_eval_ref("E14-A")
    request = f.make_assistant_request(profile, operation=c.Operation.VQA, eval_ref=eval_ref)
    result = m.MockGenerativeAssistant().execute(request, profile=profile)
    c.validate_assistant_result(result, request=request, profile=profile,
                                policy_mode=request.policy_mode, offline=True)
    assert request.context_provenance.kb_version is None and request.context_provenance.kb_sha256 is None
    assert request.kb_context == "" and request.allowed_source_ids == ()
    with pytest.raises(c.ContractError, match="E14-A"):
        f.make_assistant_request(profile, operation=c.Operation.VQA, eval_ref=eval_ref,
                                 retrieval=f.make_retrieval_result(artifacts.knowledge, c.RetrievalStatus.NOT_APPLICABLE))


@pytest.mark.parametrize("operation", [c.Operation.VQA, c.Operation.ATTRIBUTES_AND_VQA])
def test_e14_b_vqa_rejects_missing_kb_pin_but_runtime_does_not(operation):
    profile = f.make_profile("assistant")
    request = f.make_assistant_request(profile, operation=operation)
    c.validate_assistant_request(request, profile=profile, policy_mode=request.policy_mode)
    with pytest.raises(c.ContractError, match="E14-B VQA requires a KB version and hash") as failure:
        replace(request, eval_ref=f.make_eval_ref("E14-B"))
    assert failure.value.code is c.ErrorCode.INVALID_SNAPSHOT


@pytest.mark.parametrize("operation", [c.Operation.VQA, c.Operation.ATTRIBUTES_AND_VQA])
@pytest.mark.parametrize("status", list(c.RetrievalStatus))
def test_e14_b_vqa_retains_pinned_lookup_through_all_retrieval_statuses(artifacts, operation, status):
    profile = f.make_profile("assistant")
    knowledge = artifacts.knowledge
    hit = status in (c.RetrievalStatus.HIT, c.RetrievalStatus.PARTIAL_HIT)
    lookup_request = replace(f.make_retrieval_request(knowledge), explicit_class_ids=(7,) if hit else (),
                             lookup_applicable=status is not c.RetrievalStatus.NOT_APPLICABLE)
    lookup = f.make_retrieval_result(knowledge, status)
    validated = c.validate_retrieval_result(
        lookup, request=lookup_request, knowledge=knowledge, token_counter=m.MockContextTokenCounter({f.CONTEXT: 8}),
        render_context=f.render_context, policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED,
    )
    request = f.make_assistant_request(profile, operation=operation, retrieval=validated.result,
                                       eval_ref=f.make_eval_ref("E14-B"))
    result = m.MockGenerativeAssistant(fail=status is c.RetrievalStatus.RETRIEVAL_ERROR).execute(request, profile=profile)
    c.validate_assistant_result(result, request=request, profile=profile, policy_mode=request.policy_mode,
                                retrieval=validated, offline=True)
    assert result.context_provenance == request.context_provenance
    assert result.context_provenance.retrieval_status is status
    assert result.context_provenance.kb_sha256 == knowledge.kb_sha256
    assert result.context_provenance.kb_version == knowledge.data.kb_version
    assert bool(request.kb_context) == bool(request.allowed_source_ids) == hit


def test_e14_b_rejects_unvalidated_lookup_and_mismatched_pins(artifacts):
    profile = f.make_profile("assistant")
    knowledge = artifacts.knowledge
    lookup = f.make_retrieval_result(knowledge)
    result_options = dict(request=f.make_retrieval_request(knowledge), knowledge=knowledge, render_context=f.render_context,
                          token_counter=m.MockContextTokenCounter({f.CONTEXT: 8}),
                          policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED)
    validated = c.validate_retrieval_result(lookup, **result_options)
    wrong_hash = f.digest(b"synthetic different pinned KB")
    with pytest.raises(c.ContractError) as failure:
        c.validate_retrieval_result(replace(lookup, kb_sha256=wrong_hash), **result_options)
    assert failure.value.code is c.ErrorCode.HASH_MISMATCH
    request = f.make_assistant_request(profile, operation=c.Operation.VQA, retrieval=lookup,
                                       eval_ref=f.make_eval_ref("E14-B"))
    assistant_options = dict(profile=profile, policy_mode=request.policy_mode, offline=True)
    with pytest.raises(c.ContractError, match="ValidatedRetrieval"):
        c.validate_assistant_request(request, retrieval=lookup, **assistant_options)
    with pytest.raises(TypeError, match="validate_retrieval_result"):
        c.ValidatedRetrieval()
    wrong_provenance = replace(request.context_provenance, kb_sha256=wrong_hash)
    with pytest.raises(c.ContractError, match="KB pins disagree") as failure:
        c.validate_assistant_request(replace(request, context_provenance=wrong_provenance),
                                     retrieval=validated, **assistant_options)
    assert failure.value.code is c.ErrorCode.HASH_MISMATCH
    result = m.MockGenerativeAssistant().execute(request, profile=profile)
    with pytest.raises(c.ContractError, match="context binding"):
        c.validate_assistant_result(replace(result, context_provenance=wrong_provenance), request=request,
                                    retrieval=validated, **assistant_options)


@pytest.mark.parametrize("experiment", ["E14-A", "E14-B"])
def test_e14_shared_attributes_stage_can_run_before_kb_intervention(experiment):
    profile = f.make_profile("assistant")
    request = f.make_assistant_request(profile, operation=c.Operation.ATTRIBUTES, eval_ref=f.make_eval_ref(experiment))
    result = m.MockGenerativeAssistant().extract_attributes(request, profile=profile)
    c.validate_assistant_result(result, request=request, profile=profile,
                                policy_mode=request.policy_mode, offline=True)
    assert result.context_provenance.kb_sha256 is None and request.kb_context == ""
    assert result.generated.disposition is c.Disposition.NOT_APPLICABLE


@pytest.mark.parametrize("mismatch", ["question", "CNN context", "policy", "tokenizer revision"])
def test_retrieval_binding_uses_trusted_request_and_profile(artifacts, mismatch):
    knowledge = artifacts.knowledge
    profile = f.make_profile("assistant")
    policy = c.PolicyMode.BREED_CONTEXT_ALLOWED
    lookup_request = f.make_retrieval_request(knowledge)
    if mismatch == "policy":
        lookup_request = replace(lookup_request, explicit_class_ids=(), lookup_applicable=False)
        lookup = f.make_retrieval_result(knowledge, c.RetrievalStatus.NOT_APPLICABLE)
    else:
        lookup = f.make_retrieval_result(knowledge)
    assistant = f.make_assistant_request(profile, retrieval=lookup)
    good = c.validate_retrieval_result(
        lookup, request=lookup_request, knowledge=knowledge, token_counter=m.MockContextTokenCounter({f.CONTEXT: 8}),
        render_context=f.render_context, policy_mode=policy,
    )
    opts = dict(profile=profile, policy_mode=policy, mapping=artifacts.mapping, retrieval=good)
    c.validate_assistant_request(assistant, **opts)
    result = m.MockGenerativeAssistant().execute(assistant, profile=profile)
    c.validate_assistant_result(result, request=assistant, **opts)

    if mismatch == "question":
        assistant = replace(assistant, question="Câu hỏi synthetic khác")
    elif mismatch == "CNN context":
        cnn = c.AllowedCNNContext(status=c.CNNStatus.ACCEPT, class_id=7, canonical_name=f.CANONICAL_NAMES[7])
        assistant = replace(assistant, cnn_context=cnn)
    elif mismatch == "policy":
        opts["retrieval"] = c.validate_retrieval_result(
            lookup, request=lookup_request, knowledge=knowledge, token_counter=m.MockContextTokenCounter({}),
            render_context=f.render_context, policy_mode=c.PolicyMode.OBSERVATION_ONLY,
        )
    else:
        opts["profile"] = replace(profile, tokenizer_revision=f.digest(b"synthetic other tokenizer")[:40])
    with pytest.raises(c.ContractError, match=mismatch) as failure:
        c.validate_assistant_request(assistant, **opts)
    assert failure.value.code is (c.ErrorCode.POLICY_BLOCKED if mismatch == "policy" else c.ErrorCode.INVALID_SNAPSHOT)
    with pytest.raises(c.ContractError, match=mismatch):
        c.validate_assistant_result(result, request=assistant, **opts)


def test_retrieval_binding_matches_nonempty_cnn_context(artifacts):
    knowledge = artifacts.knowledge
    profile = f.make_profile("assistant")
    cnn = c.AllowedCNNContext(status=c.CNNStatus.ACCEPT, class_id=7, canonical_name=f.CANONICAL_NAMES[7])
    lookup = f.make_retrieval_result(knowledge)
    request = replace(f.make_retrieval_request(knowledge), cnn_context=cnn)
    bound = c.validate_retrieval_result(
        lookup, request=request, knowledge=knowledge, token_counter=m.MockContextTokenCounter({f.CONTEXT: 8}),
        render_context=f.render_context, policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED,
    )
    assistant = replace(f.make_assistant_request(profile, retrieval=lookup), cnn_context=cnn)
    c.validate_assistant_request(assistant, profile=profile, policy_mode=assistant.policy_mode,
                                 mapping=artifacts.mapping, retrieval=bound)


def test_untrusted_cnn_context_cannot_authorize_breed_projection(artifacts):
    profile = f.make_profile("assistant")
    context = c.AllowedCNNContext(status=c.CNNStatus.ACCEPT, class_id=7, canonical_name=f.CANONICAL_NAMES[7])
    assistant = replace(f.make_assistant_request(profile), cnn_context=context)
    retrieval = replace(f.make_retrieval_request(artifacts.knowledge), cnn_context=context)
    c.validate_assistant_request(assistant, profile=profile, mapping=artifacts.mapping,
                                 policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED)
    c.validate_retrieval_request(retrieval, knowledge=artifacts.knowledge,
                                 policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED)
    with pytest.raises(c.ContractError) as exc:
        c.validate_retrieval_request(retrieval, knowledge=artifacts.knowledge)
    assert exc.value.code is c.ErrorCode.POLICY_BLOCKED
    with pytest.raises(c.ContractError) as exc:
        c.validate_assistant_request(assistant, profile=profile, mapping=artifacts.mapping,
                                     policy_mode=c.PolicyMode.OBSERVATION_ONLY)
    assert exc.value.code is c.ErrorCode.POLICY_BLOCKED
    with pytest.raises(c.ContractError):
        c.validate_assistant_request(assistant, profile=profile, policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED)
    wrong = replace(context, canonical_name="Other")
    with pytest.raises(c.ContractError, match="canonical mapping"):
        c.validate_retrieval_request(replace(retrieval, cnn_context=wrong), knowledge=artifacts.knowledge,
                                     policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED)
    with pytest.raises(c.ContractError):
        replace(context, status=c.CNNStatus.E7_REJECT)
    with pytest.raises(c.ContractError):
        replace(context, canonical_name=c.MISSING)


@pytest.mark.parametrize("state,status,class_id", [
    (c.CNNState.DECIDED, c.CNNStatus.ACCEPT, 7),
    (c.CNNState.DECIDED, c.CNNStatus.E7_REJECT, 7),
    (c.CNNState.DECIDED, c.CNNStatus.OOD_REJECT, None),
    (c.CNNState.NOT_RUN, None, None),
    (c.CNNState.ERROR, None, None),
])
def test_cnn_snapshot_valid_wire_shapes(state, status, class_id):
    snapshot = c.CNNDecisionSnapshot(state=state, status=status, class_id=class_id)
    assert c.decode_record(c.encode_record(snapshot), c.CNNDecisionSnapshot) == snapshot


@pytest.mark.parametrize("state,status,class_id", [
    ("decided", None, None),
    ("decided", "ACCEPT", None),
    ("decided", "E7_REJECT", None),
    ("decided", "OOD_REJECT", 7),
    ("not_run", "E7_REJECT", 7),
    ("error", None, 7),
    ("decided", "PLANNED", 7),
])
def test_cnn_snapshot_rejects_invalid_state_and_class_nullability(state, status, class_id):
    with pytest.raises(c.ContractError):
        c.decode_record(f.json_bytes(dict(state=state, status=status, class_id=class_id)), c.CNNDecisionSnapshot)


@pytest.mark.parametrize("status", ["stale", "ACCEPT", "BLOCKED", "PASS"])
def test_branch_state_rejects_other_status_namespaces(status):
    with pytest.raises(c.ContractError):
        c.decode_record(f.json_bytes(dict(status=status, request_id=None)), c.BranchState)


def test_integration_envelopes_validate_branch_payload_and_runtime_metadata(artifacts, classifier):
    profile, request, result = classifier
    expected = f.make_branch_request(request)
    job = f.make_worker_job(request)
    reply = f.make_worker_reply(request, result)
    c.validate_worker_identity(expected, job)
    c.validate_worker_identity(job, c.decode_record(c.encode_record(reply), c.WorkerReply))
    with pytest.raises(c.ContractError, match="wire"):
        c.encode_record(job)
    with pytest.raises(c.ContractError, match="branch"):
        replace(job, branch=c.Branch.ASSISTANT)
    with pytest.raises(c.ContractError, match="branch"):
        replace(reply, branch=c.Branch.ASSISTANT)
    with pytest.raises(c.ContractError, match="digest"):
        replace(job, image_digest=f.digest(b"wrong digest"))
    with pytest.raises(c.ContractError, match="profile"):
        replace(reply, profile_id="wrong")
    with pytest.raises(c.ContractError, match="eval_ref"):
        f.make_worker_job(replace(request, eval_ref=f.make_eval_ref("E12-A")))
    with pytest.raises(c.ContractError, match="eval_ref"):
        f.make_worker_reply(request, replace(result, eval_ref=f.make_eval_ref("E12-A")))
    leaked = mutate_blob(c.encode_record(reply), lambda p: p["result"].update(expected_answer="scorer-only"))
    with pytest.raises(c.ContractError):
        c.decode_record(leaked, c.WorkerReply)
    invalid_adapter_status = mutate_blob(c.encode_record(reply), lambda p: p["result"].update(status="busy"))
    with pytest.raises(c.ContractError):
        c.decode_record(invalid_adapter_status, c.WorkerReply)


@pytest.mark.parametrize("field,value", [
    ("request_id", "other request"),
    ("session_id", "other session"),
    ("image_digest", f.digest(b"other source")),
    ("revision", 4),
    ("cnn_mode", c.CNNMode.FIT),
    ("profile_id", "other profile"),
])
def test_identity_check_rejects_mismatch_even_with_valid_envelope(classifier, field, value):
    _, request, result = classifier
    expected = f.make_branch_request(request)
    reply = f.make_worker_reply(request, result)
    if field in {"image_digest", "profile_id"}:
        result = replace(result, **{field: value})
    reply = replace(reply, result=result, **{field: value})
    with pytest.raises(c.ContractError, match=field):
        c.validate_worker_identity(expected, reply)


def test_identity_rejects_wrong_branch_with_valid_assistant_reply(classifier):
    profile, request, _ = classifier
    assistant_request = f.make_assistant_request(profile)
    answer = m.MockGenerativeAssistant().execute(assistant_request, profile=profile)
    reply = f.make_worker_reply(assistant_request, answer)
    with pytest.raises(c.ContractError, match="branch"):
        c.validate_worker_identity(f.make_branch_request(request), reply)


def test_session_copy_and_branch_state_result_rules(classifier):
    _, request, result = classifier
    session = f.make_session_state(request)
    states = dict(session.branch_states)
    ok = c.BranchState(status=c.RuntimeStatus.OK, request_id="fixture", model_result=result)
    states[c.Branch.COMPARISON] = ok
    snapshot = replace(session, branch_states=states)
    states.clear()
    assert snapshot.branch_states[c.Branch.COMPARISON].model_result is result
    with pytest.raises(TypeError):
        snapshot.branch_states[c.Branch.ASSISTANT] = ok
    with pytest.raises(c.ContractError, match="both branches"):
        replace(session, branch_states={c.Branch.COMPARISON: ok})
    with pytest.raises(c.ContractError, match="branch"):
        replace(session, branch_states=dict(session.branch_states) | {c.Branch.ASSISTANT: ok})
    with pytest.raises(c.ContractError, match="only ok"):
        replace(ok, status=c.RuntimeStatus.PENDING)


def test_mock_public_signatures_match_protocols():
    interfaces = (
        (c.ZeroShotClassifier.classify, m.MockZeroShotClassifier.classify),
        (c.GenerativeAssistant.extract_attributes, m.MockGenerativeAssistant.extract_attributes),
        (c.GenerativeAssistant.answer_question, m.MockGenerativeAssistant.answer_question),
        (c.GenerativeAssistant.execute, m.MockGenerativeAssistant.execute),
        (c.KnowledgeRetriever.__call__, m.MockKnowledgeRetriever.__call__),
        (c.ContextTokenCounter.__call__, m.MockContextTokenCounter.__call__),
    )
    for protocol, mock in interfaces:
        params = lambda method: [(p.name, p.kind, p.default) for p in inspect.signature(method).parameters.values()]
        assert params(protocol) == params(mock)
