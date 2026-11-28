from collections import Counter
import json
from pathlib import Path

import pytest

from src.vlm import contracts as c
from src.vlm.knowledge_base import KB_MORPHOLOGY_V1, KB_MORPHOLOGY_VERSION, load_knowledge_base, load_prompt_bank
from vlm_foundation import fixtures as f


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/knowledge/vlm"
MAPPING_SHA256 = "d86b3a3a232babea289070299727d20d56b8f1607593ce1aaf8b174465690c3c"
NAME_SHA256 = "c1bdfde5f1e809a3dcb55f0c0eaeceb8c612c7f57c6761c9def12a2e57174c30"
VISUAL_IDS = {
    "ear_position", "ear_shape", "coat_color", "coat_length", "coat_texture",
    "muzzle_length", "tail_shape", "body_build",
}
KB_ONLY_IDS = {"breed_size", "head_shape", "facial_hair", "coat_fullness"}


def test_repository_name_bank_matches_the_pinned_mapping():
    bank_raw = (DATA / "name_prompt_bank.json").read_bytes()
    mapping_raw = (ROOT / "data/metadata/class_mapping.json").read_bytes()
    mapping = c.decode_canonical_mapping(mapping_raw, expected_sha256=MAPPING_SHA256)
    bank = load_prompt_bank(bank_raw, mapping=mapping, expected_sha256=NAME_SHA256)
    assert [p.canonical_name for p in bank.data.entries] == list(f.CANONICAL_NAMES)
    assert all(p.prompt_text == f"A photo of the dog breed {p.canonical_name}." for p in bank.data.entries)


def test_loaders_require_pins_and_trusted_source_review():
    artifacts = f.make_artifacts()
    kwargs = {
        "mapping": artifacts.mapping,
        "expected_sha256": artifacts.knowledge.kb_sha256,
        "allowed_attribute_ids": {f.KB_ATTRIBUTE_ID},
    }
    with pytest.raises(c.ContractError) as rejected:
        load_knowledge_base(artifacts.knowledge_raw, reviewed_sources={}, **kwargs)
    assert rejected.value.code is c.ErrorCode.UNREVIEWED_SOURCE
    kb = load_knowledge_base(artifacts.knowledge_raw, reviewed_sources=f.REVIEWED_SOURCES, **kwargs)
    bank = load_prompt_bank(
        artifacts.morphology_raw, mapping=artifacts.mapping, knowledge=kb,
        expected_sha256=artifacts.morphology.prompt_bank_sha256,
    )
    assert bank == artifacts.morphology
    with pytest.raises(c.ContractError) as bad_hash:
        load_prompt_bank(
            artifacts.morphology_raw + b"\n", mapping=artifacts.mapping, knowledge=kb,
            expected_sha256=artifacts.morphology.prompt_bank_sha256,
        )
    assert bad_hash.value.code is c.ErrorCode.HASH_MISMATCH


def test_prompt_audit_retains_actual_counts_and_rejects_overflow_on_validation():
    from src.vlm.knowledge_base import audit_prompt_bank

    artifacts = f.make_artifacts()
    profile = f.make_profile()
    audit = audit_prompt_bank(
        artifacts.name, profile=profile,
        count_untruncated_tokens=lambda prompt: (
            f.PROMPT_TOKEN_LIMIT + 1 if prompt.endswith("Shih-Tzu") else 9
        ),
    )
    assert len(audit.entries) == 30
    assert audit.entries[-1].truncated
    assert audit.entries[-1].token_count == f.PROMPT_TOKEN_LIMIT + 1
    with pytest.raises(c.ContractError) as rejected:
        c.validate_prompt_token_audit(c.PromptTokenAudit(audits=(audit,)), bank=artifacts.name, profile=profile)
    assert rejected.value.code is c.ErrorCode.TOKEN_TRUNCATION


@pytest.fixture(scope="module")
def draft_four():
    """Reconstruct the pinned Draft 4 to preserve coverage of the pending snapshot."""
    review = ROOT / "outputs/research/vlm/task08_04/kb_source_review.json"
    evidence = json.loads(review.read_bytes())["draft_4"]
    kb = json.loads((DATA / "breed_kb.json").read_bytes())
    kb["kb_version"] = evidence["kb_version"]
    for source in kb["sources"]:
        source["review_status"] = "pending"
        source.pop("review_ref", None)
    kb_raw = (json.dumps(kb, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    assert f.digest(kb_raw) == evidence["kb_sha256"]

    bank_file = (DATA / "morphology_prompt_bank.json").read_bytes()
    bank = json.loads(bank_file)
    bank.update(bank_version=evidence["morphology_bank_version"],
                kb_version=evidence["kb_version"], kb_sha256=evidence["kb_sha256"])
    bank_raw = (json.dumps(bank, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    assert f.digest(bank_raw) == evidence["morphology_bank_sha256"]

    audit_file = (DATA / "prompt_tokenization_audit.json").read_bytes()
    audit_raw = audit_file.replace(f.digest(bank_file).encode("ascii"), f.digest(bank_raw).encode("ascii"))
    assert f.digest(audit_raw) == evidence["tokenization_audit_sha256"]
    return kb_raw, bank_raw, audit_raw


def test_repository_morphology_draft_cannot_be_loaded_as_reviewed(draft_four):
    kb_raw, bank_raw, _ = draft_four
    bank = c.decode_record(bank_raw, c.PromptBank)
    mapping = c.decode_canonical_mapping(
        (ROOT / "data/metadata/class_mapping.json").read_bytes(), expected_sha256=MAPPING_SHA256,
    )
    with pytest.raises(c.ContractError) as rejected:
        load_knowledge_base(
            kb_raw, mapping=mapping, expected_sha256=bank.kb_sha256, reviewed_sources={},
            allowed_attribute_ids=KB_MORPHOLOGY_V1,
        )
    assert rejected.value.code is c.ErrorCode.UNREVIEWED_SOURCE


def test_draft_four_review_covers_runtime_claims_and_prompt_phrases(draft_four):
    kb_raw, bank_raw, audit_raw = draft_four
    kb = c.decode_record(kb_raw, c.BreedKnowledgeBase)
    bank = c.decode_record(bank_raw, c.PromptBank)
    review_path = ROOT / "outputs/research/vlm/task08_04/kb_source_review.json"
    review = json.loads(review_path.read_bytes())
    history = review["draft_3"]
    evidence = review["draft_4"]
    revisions = {(item["class_id"], item["attribute_id"]): item for item in evidence["revisions"]}

    assert len(revisions) == len(evidence["revisions"]) == 18
    assert Counter(item["finding"] for item in revisions.values()) == {"SA-01": 17, "SA-02": 1}
    assert sum(item["morphology_binding"] is not None for item in revisions.values()) == 12
    assert kb.kb_version == bank.kb_version == evidence["kb_version"] == "breed-kb-draft-4"
    assert bank.bank_version == evidence["morphology_bank_version"] == "morphology-draft-4"
    assert kb.mapping_sha256 == bank.mapping_sha256 == MAPPING_SHA256
    assert f.digest(kb_raw) == bank.kb_sha256 == evidence["kb_sha256"]
    assert f.digest(bank_raw) == evidence["morphology_bank_sha256"]
    assert f.digest(audit_raw) == evidence["tokenization_audit_sha256"]
    assert len(history["classes"]) == len(bank.entries) == len(kb.breeds) == 30
    assert sum(len(breed.attributes) for breed in kb.breeds) == 95
    assert {attribute.attribute_id for breed in kb.breeds for attribute in breed.attributes} == KB_MORPHOLOGY_V1
    assert sum(len(entry.attribute_ids) for entry in bank.entries) == 56
    assert evidence["vocabulary"]["version"] == KB_MORPHOLOGY_VERSION
    assert evidence["source_approval"] == history["design"]["source_approval"] == "pending"

    sources = {item["source_id"]: item for item in history["sources"]}
    runtime_sources = {sid for breed in kb.breeds for attr in breed.attributes for sid in attr.source_ids}
    assert len(runtime_sources) == 32
    assert sources.keys() - runtime_sources == {
        "fci-211", "imagenet-synsets", "stanford-dogs-paper", "stanford-wordnet",
    }
    assert len(sources) == len(kb.sources) == 36
    for source in kb.sources:
        recorded = sources[source.source_id]
        assert (source.title, source.url) == (recorded["title"], recorded["url"])
        assert source.review_status is c.SourceReviewStatus.PENDING
        assert recorded["review_status"] == "pending" and recorded["document_version"]
        assert source.review_ref is c.MISSING and "review_ref" not in recorded

    verified_revisions = set()
    for breed, entry, recorded in zip(kb.breeds, bank.entries, history["classes"], strict=True):
        assert breed.class_id == entry.class_id == recorded["class_id"]
        assert breed.canonical_name == entry.canonical_name == recorded["canonical_name"]
        assert entry.canonical_name == f.CANONICAL_NAMES[entry.class_id]
        selected = [claim for claim in recorded["claims"] if claim["used_in_prompt"]]
        assert 1 <= len(selected) <= 3
        assert all(claim["observability"] in {"V", "C"} for claim in selected)
        assert entry.attribute_ids == tuple(claim["attribute_id"] for claim in selected)
        assert entry.source_ids == tuple(dict.fromkeys(
            ref["source_id"] for claim in selected for ref in claim["sources"]
        ))
        phrases = " and ".join(claim["phrase_en"] for claim in selected)
        assert entry.prompt_text == recorded["morphology_prompt"]
        assert entry.prompt_text == f"A photo of the dog breed {entry.canonical_name}, with {phrases}."

        ordered = sorted(recorded["claims"], key=lambda claim: (
            claim["observability"] == "S", claim["role"] == "common",
        ))
        for order, (attribute, claim) in enumerate(zip(breed.attributes, ordered, strict=True), 1):
            assert order == claim["retrieval_order"]
            assert attribute.attribute_id == claim["attribute_id"]
            expected_parts = [claim["claim_vi"], claim["observation_conditions"]]
            if "runtime_scope_note" in claim:
                expected_parts.append(claim["runtime_scope_note"])
            previous_value = " ".join(expected_parts)
            key = (breed.class_id, attribute.attribute_id)
            revision = revisions.get(key)
            if revision is None:
                assert attribute.value == previous_value
            else:
                verified_revisions.add(key)
                assert (revision["before_value"], revision["after_value"]) == (previous_value, attribute.value)
                assert attribute.value != previous_value
                assert tuple(revision["source_ids"]) == attribute.source_ids
                binding = revision["morphology_binding"]
                if claim["used_in_prompt"]:
                    assert binding == {
                        "attribute_id": attribute.attribute_id,
                        "phrase": claim["phrase_en"],
                        "source_ids": list(attribute.source_ids),
                    }
                else:
                    assert binding is None
            assert all(ref["locator"] for ref in claim["sources"])
            assert attribute.source_ids == tuple(ref["source_id"] for ref in claim["sources"])
            assert all(sources[source_id]["usage"] == "claim_provenance" for source_id in attribute.source_ids)

    assert verified_revisions == revisions.keys()

    # Reconstruct both historical artifacts to check every field, including notes, aliases and entry order.
    prior_kb = json.loads(kb_raw)
    prior_kb["kb_version"] = history["kb_version"]
    for breed in prior_kb["breeds"]:
        for attribute in breed["attributes"]:
            revision = revisions.get((breed["class_id"], attribute["attribute_id"]))
            if revision is not None:
                attribute["value"] = revision["before_value"]
    restored_kb = (json.dumps(prior_kb, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    assert f.digest(restored_kb) == history["kb_sha256"] == evidence["reviewed_draft_3"]["kb_sha256"]

    prior_bank = json.loads(bank_raw)
    prior_bank.update(
        bank_version=history["morphology_bank_version"], kb_version=history["kb_version"],
        kb_sha256=history["kb_sha256"],
    )
    restored_bank = (json.dumps(prior_bank, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    assert f.digest(restored_bank) == history["morphology_bank_sha256"] == evidence["reviewed_draft_3"]["morphology_bank_sha256"]


def test_versioned_kb_allowlist_keeps_visual_vocabulary_separate():
    assert KB_MORPHOLOGY_VERSION == "kb-morphology-v1"
    assert set(c.ATTRIBUTE_VOCABULARY_V1) == VISUAL_IDS
    assert KB_MORPHOLOGY_V1 == VISUAL_IDS | KB_ONLY_IDS


@pytest.mark.parametrize("attribute_id", sorted(KB_ONLY_IDS))
def test_kb_only_ids_are_rejected_by_structured_visual_observation(attribute_id):
    with pytest.raises(c.ContractError, match="unknown attribute_id"):
        c.AttributeObservation(attribute_id=attribute_id, visibility=c.Visibility.UNKNOWN, value=None)


def test_new_vocabulary_accepts_synthetic_approved_artifact_and_rejects_old_allowlist():
    artifacts = f.make_artifacts()
    payload = json.loads(artifacts.knowledge_raw)
    for breed in payload["breeds"]:
        breed["attributes"][0]["attribute_id"] = "coat_texture"
    for attribute_id in sorted(KB_ONLY_IDS):
        payload["breeds"][0]["attributes"].append({
            "attribute_id": attribute_id, "value": "fixture-only morphological knowledge",
            "source_ids": ["fixture-source-0"],
        })
    raw = f.json_bytes(payload)
    kwargs = {
        "mapping": artifacts.mapping, "expected_sha256": f.digest(raw),
        "reviewed_sources": f.REVIEWED_SOURCES,
    }
    approved = load_knowledge_base(raw, allowed_attribute_ids=KB_MORPHOLOGY_V1, **kwargs)
    assert {attr.attribute_id for attr in approved.data.breeds[0].attributes} == KB_ONLY_IDS | {"coat_texture"}
    with pytest.raises(c.ContractError) as rejected:
        load_knowledge_base(raw, allowed_attribute_ids=c.ATTRIBUTE_VOCABULARY_V1, **kwargs)
    assert rejected.value.code is c.ErrorCode.INVALID_SNAPSHOT


def test_kb_loader_rejects_wrong_hash_or_independent_review_ref():
    artifacts = f.make_artifacts()
    kwargs = {
        "mapping": artifacts.mapping, "expected_sha256": artifacts.knowledge.kb_sha256,
        "reviewed_sources": f.REVIEWED_SOURCES, "allowed_attribute_ids": {f.KB_ATTRIBUTE_ID},
    }
    with pytest.raises(c.ContractError) as wrong_hash:
        load_knowledge_base(artifacts.knowledge_raw + b"\n", **kwargs)
    assert wrong_hash.value.code is c.ErrorCode.HASH_MISMATCH

    bad_reviews = {**f.REVIEWED_SOURCES, "fixture-source-0": "other-synthetic-review"}
    with pytest.raises(c.ContractError) as wrong_review:
        load_knowledge_base(artifacts.knowledge_raw, **{**kwargs, "reviewed_sources": bad_reviews})
    assert wrong_review.value.code is c.ErrorCode.UNREVIEWED_SOURCE


def test_morphology_loader_rejects_wrong_kb_version_with_valid_hash():
    artifacts = f.make_artifacts()
    bank = json.loads(artifacts.morphology_raw)
    bank["kb_version"] = "different-synthetic-kb"
    raw = f.json_bytes(bank)
    with pytest.raises(c.ContractError) as wrong_version:
        load_prompt_bank(
            raw, mapping=artifacts.mapping, expected_sha256=f.digest(raw), knowledge=artifacts.knowledge,
        )
    assert wrong_version.value.code is c.ErrorCode.HASH_MISMATCH


def test_tokenizer_preview_targets_both_current_bank_hashes():
    from scripts.audit_vlm_prompt_tokenization import PREVIEW_MODELS, TOKEN_LIMIT

    audit = c.decode_record((DATA / "prompt_tokenization_audit.json").read_bytes(), c.PromptTokenAudit)
    bank_hashes = {NAME_SHA256, f.digest((DATA / "morphology_prompt_bank.json").read_bytes())}
    expected = {
        (bank_hash, profile_id, revision)
        for bank_hash in bank_hashes for profile_id, _, revision in PREVIEW_MODELS
    }
    assert len(audit.audits) == len(expected) == 4
    assert {
        (entry.prompt_bank_sha256, entry.profile_id, entry.tokenizer_revision)
        for entry in audit.audits
    } == expected
    assert all(entry.token_limit == TOKEN_LIMIT for entry in audit.audits)
    assert all(
        not item.truncated and 0 < item.token_count <= entry.token_limit
        for entry in audit.audits for item in entry.entries
    )
