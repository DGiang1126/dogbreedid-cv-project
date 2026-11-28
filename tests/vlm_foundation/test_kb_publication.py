from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import unicodedata

import pytest

from src.vlm import contracts as c
from src.vlm.knowledge_base import (
    KB_MORPHOLOGY_V1, KB_MORPHOLOGY_VERSION, load_knowledge_base, load_prompt_bank,
)
from src.vlm.retrieval import render_context, retrieve_knowledge
from vlm_foundation import fixtures as f


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "outputs/research/vlm/task08_04"
# Release-verification inputs captured before copying the release into the worktree.
EXPECTED_MANIFEST_SHA256 = "7adcb27532c50dfe6e1896517071133a02b1a9593acbd1e168c4d0c066c5449b"
APPROVAL_ID = "task08.04-kb-v1-20261128"
REVIEWED_CANDIDATE = {
    "kb": "543243596553cd2ef903891f41417d5e51ab7904330917e22c1d3aa05c05e344",
    "morphology_bank": "ace2fb4b428648f98fd7628a3a773e9fae68dca13d6dfdebccda2f8d3b0fc6ca",
    "mapping": "d86b3a3a232babea289070299727d20d56b8f1607593ce1aaf8b174465690c3c",
    "name_bank": "c1bdfde5f1e809a3dcb55f0c0eaeceb8c612c7f57c6761c9def12a2e57174c30",
    "tokenization_audit": "9d893f5750e98b56fe7fef3a1cf23f1544d465103f71fff5e9b5549f4143f15e",
    "source_evidence": "5d9c5b6cacc453762785cda2a6656994698b9015a48383c55537456226c3e003",
}
DESIGN_SHA256 = "1f357e59e3bef38f2e60217ff54e32504c40d068820e369e25e8e84b3b98bbaf"
REVISION_REVIEW_SHA256 = "2891ca3ef5861004a701fa1204d4529658a62ef3c05593440e9ab9369d8ddedf"
DESIGN_ONLY = {"fci-211", "imagenet-synsets", "stanford-dogs-paper", "stanford-wordnet"}
POLICY = c.PolicyMode.BREED_CONTEXT_ALLOWED


def verified_approval(manifest_raw, approval_raw, expected_manifest_hash):
    """Fixture at the caller boundary. The expected manifest pin comes from the caller."""
    c.verify_sha256(manifest_raw, expected_manifest_hash)
    manifest = json.loads(manifest_raw)
    c.verify_sha256(approval_raw, manifest["approval_record"]["sha256"])
    approval = json.loads(approval_raw)

    assert manifest["approval_id"] == approval["approval_id"] == APPROVAL_ID
    assert approval["decision_status"] == "APPROVED"
    assert (approval["decided_by"], approval["authority"], approval["decision_date"]) == (
        "Đỗ Đông Giang", "Trưởng nhóm DogBreedID", "2026-11-28",
    )
    assert approval["leader_decision"]["formal_evaluation_authorized"] is False
    assert approval["reviewed_candidate_hashes"] == manifest["reviewed_candidate_hashes"] == REVIEWED_CANDIDATE
    assert approval["design_sha256"] == manifest["design_sha256"] == DESIGN_SHA256
    assert approval["revision_report_sha256"] == manifest["revision_report_sha256"] == REVISION_REVIEW_SHA256
    assert approval["source_evidence"] == {"revision": "draft_4", "sha256": REVIEWED_CANDIDATE["source_evidence"]}
    assert approval["kb_vocabulary_version"] == manifest["kb_vocabulary_version"] == KB_MORPHOLOGY_VERSION
    return manifest, approval


@pytest.fixture(scope="module")
def release_artifacts():
    manifest_raw = (EVIDENCE / "kb_publication_manifest.json").read_bytes()
    approval_raw = (EVIDENCE / "kb_source_approval.json").read_bytes()
    manifest, approval = verified_approval(manifest_raw, approval_raw, EXPECTED_MANIFEST_SHA256)
    raw = {}
    for key in ("kb", "morphology_bank", "name_bank", "canonical_mapping", "source_evidence", "tokenization_audit"):
        record = manifest[key]
        raw[key] = (ROOT / record["path"]).read_bytes()
        c.verify_sha256(raw[key], record["sha256"])
    assert manifest["canonical_mapping"]["sha256"] == REVIEWED_CANDIDATE["mapping"]
    assert manifest["name_bank"]["sha256"] == REVIEWED_CANDIDATE["name_bank"]
    assert manifest["source_evidence"]["sha256"] == REVIEWED_CANDIDATE["source_evidence"]
    assert manifest["publication_status"] == "UNPUBLISHED"
    return manifest, approval, raw


def check_scope(approval, raw):
    kb = json.loads(raw["kb"])
    bank = json.loads(raw["morphology_bank"])
    evidence = json.loads(raw["source_evidence"])["draft_3"]
    original = {(b["class_id"], a["attribute_id"]): a
                for b in evidence["classes"] for a in b["claims"]}
    claims = {(s["class_id"], s["attribute_id"]): s for s in approval["claims"]}
    reviewed_claims = {source["source_id"]: [] for source in approval["runtime_sources"]}
    assert len(claims) == len(approval["claims"]) == 95
    assert Counter(s["decision"] for s in claims.values()) == {"APPROVED": 20, "APPROVED_WITH_SCOPE": 75}
    assert len(bank["entries"]) == len(kb["breeds"]) == 30

    checked = set()
    for breed, prompt in zip(kb["breeds"], bank["entries"], strict=True):
        cid = breed["class_id"]
        assert (prompt["class_id"], prompt["canonical_name"]) == (cid, breed["canonical_name"])
        bindings = {}
        for attr in breed["attributes"]:
            key = cid, attr["attribute_id"]
            checked.add(key)
            scope = claims[key]
            baseline = original[key]
            payload = {"class_id": cid, "attribute_id": key[1], "value": attr["value"], "source_ids": attr["source_ids"]}
            assert payload == {k: scope[k] for k in payload}
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            assert f.digest(encoded) == scope["scope_sha256"]
            assert scope["canonical_name"] == breed["canonical_name"] and scope["usage_limitations"]
            for name in ("variability_note", "uncertainty_note"):
                assert scope.get(name) == attr.get(name) and (name in scope) == (name in attr)
            assert scope["runtime_evidence"] == baseline["sources"]
            assert [ref["source_id"] for ref in scope["runtime_evidence"]] == attr["source_ids"]
            phrase = baseline["phrase_en"] if baseline["used_in_prompt"] else None
            assert scope["morphology_phrase"] == phrase
            if phrase is not None:
                bindings[key[1]] = phrase, attr["source_ids"]
            for sid in attr["source_ids"]:
                reviewed_claims[sid].append({
                    "class_id": cid, "attribute_id": key[1], "scope_sha256": scope["scope_sha256"],
                })
        assert bindings.keys() == set(prompt["attribute_ids"])
        ordered = [bindings[aid] for aid in prompt["attribute_ids"]]
        assert prompt["source_ids"] == list(dict.fromkeys(sid for _, sources in ordered for sid in sources))
        suffix = " and ".join(phrase for phrase, _ in ordered)
        assert prompt["prompt_text"] == f"A photo of the dog breed {breed['canonical_name']}, with {suffix}."
    assert checked == claims.keys() == original.keys()

    refs = {}
    for source in approval["runtime_sources"]:
        sid = source["source_id"]
        assert source["decision"] == "APPROVED_FOR_LISTED_SCOPE"
        assert source["review_ref"] == f"task08.04/source/{APPROVAL_ID}/{sid}"
        assert source["document"]["version"] and len(source["document"]["reviewed_raw_sha256"]) == 64
        assert source["claim_scope"] == reviewed_claims[sid] and source["claim_scope"]
        refs[sid] = source["review_ref"]
    assert len(refs) == 32
    assert {s["source_id"] for s in approval["design_only_sources"]} == DESIGN_ONLY
    assert all("review_ref" not in s and s["decision"] == "OUTSIDE_RUNTIME_APPROVAL"
               for s in approval["design_only_sources"])
    return refs


@pytest.fixture(scope="module")
def trusted_snapshot(release_artifacts):
    manifest, approval, raw = release_artifacts
    refs = check_scope(approval, raw)
    assert set(manifest["runtime_approved_source_ids"]) == refs.keys()
    assert set(manifest["design_only_source_ids"]) == DESIGN_ONLY
    mapping = c.decode_canonical_mapping(raw["canonical_mapping"], expected_sha256=REVIEWED_CANDIDATE["mapping"])
    kb = load_knowledge_base(
        raw["kb"], mapping=mapping, expected_sha256=manifest["kb"]["sha256"],
        reviewed_sources=refs, allowed_attribute_ids=KB_MORPHOLOGY_V1,
    )
    return kb, mapping, refs


def test_loads_actual_approved_kb_and_prompt_banks(release_artifacts, trusted_snapshot):
    manifest, _, raw = release_artifacts
    kb, mapping, refs = trusted_snapshot
    assert kb.data.kb_version == manifest["kb"]["version"] == "breed-kb-v1"
    assert len(kb.data.breeds) == 30
    assert sum(len(b.attributes) for b in kb.data.breeds) == 95
    for s in kb.data.sources:
        if s.source_id in DESIGN_ONLY:
            assert s.review_status is c.SourceReviewStatus.PENDING and s.review_ref is c.MISSING
        else:
            assert s.review_status is c.SourceReviewStatus.APPROVED and s.review_ref == refs[s.source_id]
    assert {s.source_id for s in kb.data.sources} == refs.keys() | DESIGN_ONLY
    morphology = load_prompt_bank(
        raw["morphology_bank"], mapping=mapping, expected_sha256=manifest["morphology_bank"]["sha256"], knowledge=kb,
    )
    name = load_prompt_bank(raw["name_bank"], mapping=mapping, expected_sha256=REVIEWED_CANDIDATE["name_bank"])
    assert morphology.data.bank_version == "morphology-v1" and name.data.bank_version == "name-1"
    assert len(morphology.data.entries) == len(name.data.entries) == 30
    assert sum(len(e.attribute_ids) for e in morphology.data.entries) == 56


@pytest.mark.parametrize("invalid", ["no-map", "different-ref", "missing-ref", "design-only", "tampered"])
def test_rejects_invalid_kb_trust_or_content(invalid, release_artifacts, trusted_snapshot):
    manifest, _, raw = release_artifacts
    _, mapping, refs = trusted_snapshot
    kb_raw, expected = raw["kb"], manifest["kb"]["sha256"]
    error = c.ErrorCode.UNREVIEWED_SOURCE
    sid = next(iter(refs))
    if invalid == "no-map":
        refs = {}
    elif invalid == "different-ref":
        refs = {**refs, sid: "wrong-caller-review-ref"}
    elif invalid == "tampered":
        kb_raw += b"\n"
        error = c.ErrorCode.HASH_MISMATCH
    else:
        mutant = json.loads(kb_raw)
        if invalid == "missing-ref":
            next(s for s in mutant["sources"] if s["source_id"] == sid).pop("review_ref")
        else:
            mutant["breeds"][0]["attributes"][0]["source_ids"] = ["fci-211"]
        # Isolate the Foundation source check after the byte-hash gate in these two fixtures.
        kb_raw = f.json_bytes(mutant)
        expected = f.digest(kb_raw)
    with pytest.raises(c.ContractError) as rejected:
        load_knowledge_base(
            kb_raw, mapping=mapping, expected_sha256=expected,
            reviewed_sources=refs, allowed_attribute_ids=KB_MORPHOLOGY_V1,
        )
    assert rejected.value.code is error


@pytest.mark.parametrize("field", ["kb_version", "kb_sha256"])
def test_rejects_morphology_bound_to_wrong_kb(field, release_artifacts, trusted_snapshot):
    _, _, raw = release_artifacts
    kb, mapping, _ = trusted_snapshot
    bank = json.loads(raw["morphology_bank"])
    bank[field] = "different-kb" if field == "kb_version" else "0" * 64
    wrong_bank = f.json_bytes(bank)
    with pytest.raises(c.ContractError) as rejected:
        load_prompt_bank(wrong_bank, mapping=mapping, expected_sha256=f.digest(wrong_bank), knowledge=kb)
    assert rejected.value.code is c.ErrorCode.HASH_MISMATCH


@pytest.mark.parametrize("invalid", ["manifest-bytes", "approval-bytes", "approval-scope"])
def test_caller_rejects_changes_outside_trusted_pins(invalid):
    manifest = (EVIDENCE / "kb_publication_manifest.json").read_bytes()
    approval = (EVIDENCE / "kb_source_approval.json").read_bytes()
    if invalid == "manifest-bytes":
        manifest += b"\n"
    elif invalid == "approval-bytes":
        approval += b"\n"
    else:
        modified = json.loads(approval)
        modified["claims"][0]["usage_limitations"] = "Unapproved new scope"
        approval = f.json_bytes(modified)
    with pytest.raises(c.ContractError) as rejected:
        verified_approval(manifest, approval, EXPECTED_MANIFEST_SHA256)
    assert rejected.value.code is c.ErrorCode.HASH_MISMATCH


def lookup(kb, question, budget=20, *, explicit_ids=(), cnn=None, **kwargs):
    request = c.RetrievalRequest(
        question=question, explicit_class_ids=explicit_ids, cnn_context=cnn,
        kb_version=kb.data.kb_version, kb_sha256=kb.kb_sha256, lookup_applicable=True,
        context_budget=c.ContextBudget(max_tokens=budget, tokenizer_revision="fixture-line-counter"),
    )

    def count_context_lines(text):
        return len(text.splitlines())

    result = retrieve_knowledge(
        request, knowledge=kb, count_context_tokens=count_context_lines, policy_mode=POLICY, **kwargs,
    )
    return c.validate_retrieval_result(
        result, request=request, knowledge=kb, token_counter=count_context_lines,
        render_context=render_context, policy_mode=POLICY,
    ).result


def test_retrieval_uses_actual_claim_scope_and_keeps_order(release_artifacts, trusted_snapshot):
    _, approval, _ = release_artifacts
    kb, _, _ = trusted_snapshot
    question = unicodedata.normalize("NFD", "CHÓ   HUSKY SIBERIA, Border Collie và Collie")
    first = lookup(kb, question)
    again = lookup(kb, question)
    assert replace(first, latency_ms=0) == replace(again, latency_ms=0)
    assert first.status is c.RetrievalStatus.HIT and first.resolved_class_ids == (2, 23, 21)
    assert [m.mention for m in first.per_breed] == ["chó husky siberia", "border collie", "collie"]
    scopes = {(s["class_id"], s["attribute_id"]): s for s in approval["claims"]}
    for entry in first.entries:
        for attr in entry.attributes:
            scope = scopes[(entry.class_id, attr.attribute_id)]
            assert attr.value == scope["value"] and attr.source_ids == tuple(scope["source_ids"])
    assert first.source_ids == tuple(dict.fromkeys(
        sid for entry in first.entries for attr in entry.attributes for sid in attr.source_ids
    ))

    one = lookup(kb, "Siberian Husky")
    assert one.status is c.RetrievalStatus.HIT and one.resolved_class_ids == (2,)
    assert one.source_ids == ("fci-270",)
    small = lookup(kb, "Collie và Border Collie", budget=2)
    assert small.status is c.RetrievalStatus.PARTIAL_HIT and small.resolved_class_ids == (21, 23)
    assert [e.class_id for e in small.entries] == [21] and len(small.entries[0].attributes) == 2
    assert small.per_breed[1].reason == "budget_exhausted"
    assert small.truncation.used_tokens == small.truncation.budget_tokens == 2
    assert small.truncation.applied


def test_retrieval_preserves_ambiguity_unknown_and_cnn_fallback_policy(trusted_snapshot):
    kb, _, _ = trusted_snapshot
    cnn = c.AllowedCNNContext(status=c.CNNStatus.ACCEPT, class_id=24, canonical_name="Golden Retriever")
    collision = lookup(kb, "Husky", cnn=cnn, image_only=True)
    assert collision.status is c.RetrievalStatus.AMBIGUOUS and collision.entries == ()
    assert collision.per_breed[0].candidate_class_ids == (0, 2)
    unknown = lookup(kb, "Samoyed", cnn=cnn, image_only=True, unresolved_mentions=("Samoyed",))
    assert unknown.status is c.RetrievalStatus.MISS and unknown.per_breed[0].reason == "no_alias_match"
    comparison = lookup(kb, "Samoyed và Border Collie", unresolved_mentions=("Samoyed",))
    assert comparison.status is c.RetrievalStatus.PARTIAL_HIT and comparison.resolved_class_ids == (23,)
    assert lookup(kb, "Giống trong ảnh?", cnn=cnn).status is c.RetrievalStatus.MISS
    assert lookup(kb, "Giống trong ảnh?", cnn=cnn, image_only=True).resolved_class_ids == (24,)
    assert lookup(kb, "Giống trong ảnh?", explicit_ids=(22,), cnn=cnn, image_only=True).resolved_class_ids == (22,)
