from dataclasses import replace
import unicodedata

import pytest

from src.vlm import contracts as c
from vlm_foundation import fixtures as f


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
