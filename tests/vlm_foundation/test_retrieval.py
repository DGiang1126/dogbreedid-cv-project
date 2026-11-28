from dataclasses import replace
import unicodedata

import pytest

from src.vlm import contracts as c
from vlm_foundation import fixtures as f, mocks


POLICY = c.PolicyMode.BREED_CONTEXT_ALLOWED


@pytest.fixture
def kb():
    return f.make_artifacts().knowledge


def run_lookup(kb, question, *, tokens=lambda text: len(text.split()), **kwargs):
    from src.vlm.retrieval import render_context, retrieve_knowledge

    fields = {"explicit_class_ids": (), **kwargs.pop("request_fields", {})}
    request = replace(f.make_retrieval_request(kb), question=question, **fields)
    result = retrieve_knowledge(
        request, knowledge=kb, count_context_tokens=tokens, policy_mode=POLICY, **kwargs,
    )
    return c.validate_retrieval_result(
        result, request=request, knowledge=kb, token_counter=tokens,
        render_context=render_context, policy_mode=POLICY,
    ).result


def test_unicode_longest_word_boundaries_and_explicit_order(kb):
    question = unicodedata.normalize("NFD", "GIỐNG   MINH HỌA 7, Border Collie, Collie; Border Collies")
    result = run_lookup(
        kb, question, tokens=lambda text: len(text.splitlines()),
        request_fields={"explicit_class_ids": (24, 24)},
    )
    assert result.status is c.RetrievalStatus.HIT
    assert result.resolved_class_ids == (24, 7, 23, 21)
    assert [item.mention for item in result.per_breed] == [
        None, "giống minh họa 7", "border collie", "collie",
    ]
    assert result.source_ids == tuple(f"fixture-source-{i}" for i in (24, 7, 23, 21))


def test_collision_keeps_every_candidate_and_never_uses_cnn(kb):
    cnn = c.AllowedCNNContext(status=c.CNNStatus.ACCEPT, class_id=24, canonical_name="Golden Retriever")
    result = run_lookup(kb, "shared fixture alias", image_only=True, request_fields={"cnn_context": cnn})
    assert result.status is c.RetrievalStatus.AMBIGUOUS
    assert result.per_breed[0].candidate_class_ids == c.CLASS_IDS
    assert result.resolved_class_ids == ()


def test_unknown_comparison_target_survives_alongside_hit(kb):
    result = run_lookup(kb, "So sánh Samoyed và Border Collie", unresolved_mentions=("Samoyed",))
    assert result.status is c.RetrievalStatus.PARTIAL_HIT
    assert [(item.mention, item.class_id) for item in result.per_breed] == [
        ("samoyed", None), ("border collie", 23),
    ]
    assert result.per_breed[0].reason == "no_alias_match"


def test_known_subphrase_cannot_replace_a_longer_out_of_scope_mention(kb):
    result = run_lookup(
        kb, "American Cocker Spaniel", unresolved_mentions=("American Cocker Spaniel",),
    )
    assert result.status is c.RetrievalStatus.MISS
    assert result.per_breed[0].mention == "american cocker spaniel"


def test_alias_index_uses_the_current_snapshot(kb):
    question = "chó tai dựng"
    assert run_lookup(kb, question).status is c.RetrievalStatus.MISS

    breed = replace(kb.data.breeds[0], aliases={"vi": (question,), "en": ()})
    updated = replace(kb, data=replace(kb.data, breeds=(breed, *kb.data.breeds[1:])),
                      kb_sha256=f.digest(b"updated alias fixture"))

    assert run_lookup(updated, question).resolved_class_ids == (0,)
    assert run_lookup(kb, question).status is c.RetrievalStatus.MISS


def test_cnn_fallback_needs_confirmed_image_only_scope(kb):
    cnn = c.AllowedCNNContext(status=c.CNNStatus.ACCEPT, class_id=24, canonical_name="Golden Retriever")
    args = {"request_fields": {"cnn_context": cnn}}
    assert run_lookup(kb, "Còn Samoyed thì sao?", **args).status is c.RetrievalStatus.MISS
    assert run_lookup(kb, "Giống trong ảnh có đặc điểm gì?", image_only=True, **args).resolved_class_ids == (24,)


def test_budget_counts_sources_and_keeps_resolved_misses(kb):
    result = run_lookup(kb, "Collie và Border Collie",
                        request_fields={"context_budget": c.ContextBudget(max_tokens=6, tokenizer_revision="test")})
    assert result.status is c.RetrievalStatus.PARTIAL_HIT
    assert result.resolved_class_ids == (21, 23)
    assert result.per_breed[1].reason == "budget_exhausted"
    assert result.source_ids == ("fixture-source-21",)
    assert result.context == "Collie | synthetic_coat: synthetic value [fixture-source-21]"
    assert result.truncation.used_tokens == 6
    assert result.truncation.omitted_attribute_count == 1

    too_small_for_source = run_lookup(
        kb, "Collie", request_fields={"context_budget": c.ContextBudget(max_tokens=5, tokenizer_revision="test")},
    )
    assert too_small_for_source.status is c.RetrievalStatus.MISS


def test_zero_budget_yields_empty_miss(kb):
    result = run_lookup(kb, "Collie", tokens=lambda _: 0,
                        request_fields={"context_budget": c.ContextBudget(max_tokens=0, tokenizer_revision="test")})
    assert result.status is c.RetrievalStatus.MISS
    assert result.resolved_class_ids == (21,)
    assert result.context == ""


def test_non_applicable_lookup_does_not_invoke_tokenizer(kb):
    def forbidden(_):
        pytest.fail("not_applicable invoked the tokenizer")

    result = run_lookup(kb, "Collie", tokens=forbidden, request_fields={"lookup_applicable": False})
    assert result.status is c.RetrievalStatus.NOT_APPLICABLE


@pytest.mark.parametrize("policy", [None, c.PolicyMode.OBSERVATION_ONLY])
def test_applicable_lookup_cannot_run_without_trusted_policy(kb, policy):
    from src.vlm.retrieval import retrieve_knowledge

    request = f.make_retrieval_request(kb)
    with pytest.raises(c.ContractError) as blocked:
        retrieve_knowledge(request, knowledge=kb, count_context_tokens=len, policy_mode=policy)
    assert blocked.value.code is c.ErrorCode.POLICY_BLOCKED


@pytest.mark.parametrize("applicable", [True, False])
def test_snapshot_mismatch_reports_error_without_leaking_context(kb, applicable):
    from src.vlm.retrieval import retrieve_knowledge

    request = replace(f.make_retrieval_request(kb), kb_sha256="a" * 64, lookup_applicable=applicable)
    result = retrieve_knowledge(request, knowledge=kb, count_context_tokens=len, policy_mode=POLICY)
    assert result.status is c.RetrievalStatus.RETRIEVAL_ERROR
    assert result.error.code is c.ErrorCode.HASH_MISMATCH
    assert result.context == ""


def test_unavailable_tokenizer_returns_safe_error(kb):
    def unavailable(_):
        raise RuntimeError("private path")

    result = run_lookup(kb, "Collie", tokens=unavailable)
    assert result.status is c.RetrievalStatus.RETRIEVAL_ERROR
    assert result.error.code is c.ErrorCode.RETRIEVAL_ERROR
    assert "private path" not in result.error.message_vi


def test_retrieval_context_is_accepted_by_the_assistant_contract(kb):
    from src.vlm.retrieval import render_context, retrieve_knowledge

    def count_tokens(text):
        return len(text.split())

    lookup = replace(f.make_retrieval_request(kb), question="Đặc điểm của Miniature Poodle?")
    result = retrieve_knowledge(lookup, knowledge=kb, count_context_tokens=count_tokens, policy_mode=POLICY)
    bound = c.validate_retrieval_result(
        result, request=lookup, knowledge=kb, token_counter=count_tokens, render_context=render_context,
        policy_mode=POLICY,
    )
    profile = f.make_profile("assistant")
    request = replace(
        f.make_assistant_request(
            profile, operation=c.Operation.VQA, retrieval=bound.result, eval_ref=f.make_eval_ref("E14-B"),
        ),
        question=lookup.question,
    )
    response = mocks.MockGenerativeAssistant().execute(request, profile=profile)
    c.validate_assistant_result(response, request=request, profile=profile, policy_mode=POLICY,
                                retrieval=bound, offline=True)
    assert request.allowed_source_ids == ("fixture-source-7",)
    assert request.kb_context == result.context


@pytest.fixture
def group_aware_fixture():
    """Exercise the 95-attribute shape with synthetic text and independent review refs."""
    import json

    from src.vlm.knowledge_base import KB_MORPHOLOGY_V1, load_knowledge_base

    artifacts = f.make_artifacts()
    payload = json.loads(artifacts.knowledge_raw)
    default = ("coat_texture", "ear_position", "body_build")
    layout = {
        3: ("body_build", "tail_shape", "coat_color"),
        4: ("tail_shape", "body_build", "coat_color"),
        6: ("coat_texture", "ear_position", "breed_size"),
        7: ("coat_texture", "ear_position", "breed_size"),
        8: ("coat_texture", "ear_position", "breed_size"),
    }
    for breed in payload["breeds"]:
        class_id = breed["class_id"]
        attribute_ids = layout.get(class_id, default)
        if class_id < 5:
            attribute_ids += ("head_shape",)
        if class_id in (6, 7, 8):
            breed["aliases"]["en"] = ["poodle"]
        breed["attributes"] = [
            {"attribute_id": attribute_id, "value": f"fixture-only {attribute_id}",
             "source_ids": [f"fixture-source-{class_id}"]}
            for attribute_id in attribute_ids
        ]
    raw = f.json_bytes(payload)
    snapshot = load_knowledge_base(
        raw, mapping=artifacts.mapping, expected_sha256=f.digest(raw),
        reviewed_sources=f.REVIEWED_SOURCES, allowed_attribute_ids=KB_MORPHOLOGY_V1,
    )
    assert sum(len(breed.attributes) for breed in snapshot.data.breeds) == 95
    return snapshot


def test_group_aware_single_breed_prioritizes_visual_discrimination_at_small_budget(group_aware_fixture):
    result = run_lookup(
        group_aware_fixture, "Appenzeller", tokens=lambda text: len(text.split()),
        request_fields={"context_budget": c.ContextBudget(max_tokens=6, tokenizer_revision="fixture-words")},
    )
    assert result.status is c.RetrievalStatus.HIT
    assert tuple(item.attribute_id for item in result.entries[0].attributes) == ("tail_shape",)
    assert result.source_ids == ("fixture-source-4",)
    assert result.context == "Appenzeller | tail_shape: fixture-only tail_shape [fixture-source-4]"
    assert result.truncation.used_tokens == 6
    assert result.truncation.omitted_attribute_count == 3


@pytest.mark.parametrize(("budget", "status", "entries", "omitted"), [
    (2, c.RetrievalStatus.PARTIAL_HIT, ((4, ("tail_shape", "body_build")),), 6),
    (5, c.RetrievalStatus.HIT, (
        (4, ("tail_shape", "body_build", "coat_color", "head_shape")), (3, ("body_build",)),
    ), 3),
])
def test_group_aware_comparison_preserves_breed_order_and_source_union(
    group_aware_fixture, budget, status, entries, omitted,
):
    kwargs = {
        "tokens": lambda text: len(text.splitlines()),
        "request_fields": {"context_budget": c.ContextBudget(max_tokens=budget, tokenizer_revision="fixture-lines")},
    }
    question = "So sánh Appenzeller và Greater Swiss Mountain Dog"
    first = run_lookup(group_aware_fixture, question, **kwargs)
    second = run_lookup(group_aware_fixture, question, **kwargs)
    assert replace(first, latency_ms=0) == replace(second, latency_ms=0)
    assert first.status is status
    assert first.resolved_class_ids == (4, 3)
    assert tuple((entry.class_id, tuple(a.attribute_id for a in entry.attributes))
                 for entry in first.entries) == entries
    assert first.source_ids == tuple(f"fixture-source-{class_id}" for class_id, _ in entries)
    assert first.truncation.used_tokens == budget
    assert first.truncation.omitted_attribute_count == omitted


def test_group_aware_scale_claims_follow_shared_visual_claims(group_aware_fixture):
    result = run_lookup(
        group_aware_fixture, "Toy Poodle", tokens=lambda text: len(text.splitlines()),
        request_fields={"context_budget": c.ContextBudget(max_tokens=2, tokenizer_revision="fixture-lines")},
    )
    assert result.status is c.RetrievalStatus.HIT
    assert tuple(item.attribute_id for item in result.entries[0].attributes) == ("coat_texture", "ear_position")
    assert result.truncation.omitted_attribute_count == 1


@pytest.mark.parametrize(("question", "unresolved", "status", "resolved", "candidates"), [
    ("Poodle", (), c.RetrievalStatus.AMBIGUOUS, (), (6, 7, 8)),
    ("American Cocker Spaniel", ("American Cocker Spaniel",), c.RetrievalStatus.MISS, (), ()),
    ("Samoyed và Toy Poodle", ("Samoyed",), c.RetrievalStatus.PARTIAL_HIT, (6,), ()),
])
def test_group_aware_caller_targets_take_precedence_over_cnn_fallback(
    group_aware_fixture, question, unresolved, status, resolved, candidates,
):
    cnn = c.AllowedCNNContext(status=c.CNNStatus.ACCEPT, class_id=24, canonical_name="Golden Retriever")
    result = run_lookup(
        group_aware_fixture, question, image_only=True, unresolved_mentions=unresolved,
        request_fields={"cnn_context": cnn},
    )
    assert result.status is status
    assert result.resolved_class_ids == resolved
    assert result.per_breed[0].candidate_class_ids == candidates
    assert result.source_ids == tuple(f"fixture-source-{class_id}" for class_id in resolved)
