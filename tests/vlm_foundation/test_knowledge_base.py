import json
from pathlib import Path

import pytest

from src.vlm import contracts as c
from vlm_foundation import fixtures as f


def test_repository_name_bank_matches_the_pinned_mapping():
    from src.vlm.knowledge_base import load_prompt_bank

    root = Path(__file__).resolve().parents[2]
    bank_raw = (root / "data/knowledge/vlm/name_prompt_bank.json").read_bytes()
    mapping_raw = (root / "data/metadata/class_mapping.json").read_bytes()
    mapping = c.decode_canonical_mapping(
        mapping_raw, expected_sha256=json.loads(bank_raw)["mapping_sha256"],
    )
    bank = load_prompt_bank(bank_raw, mapping=mapping, expected_sha256=f.digest(bank_raw))
    assert [p.canonical_name for p in bank.data.entries] == list(f.CANONICAL_NAMES)
    assert all(p.prompt_text == f"A photo of the dog breed {p.canonical_name}." for p in bank.data.entries)


def test_loaders_require_pins_and_trusted_source_review():
    from src.vlm.knowledge_base import load_knowledge_base, load_prompt_bank

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


def test_repository_morphology_draft_cannot_be_loaded_as_reviewed():
    from src.vlm.knowledge_base import load_knowledge_base

    root = Path(__file__).resolve().parents[2]
    kb_raw = (root / "data/knowledge/vlm/breed_kb.json").read_bytes()
    bank_raw = (root / "data/knowledge/vlm/morphology_prompt_bank.json").read_bytes()
    mapping_raw = (root / "data/metadata/class_mapping.json").read_bytes()
    bank = c.decode_record(bank_raw, c.PromptBank)
    mapping = c.decode_canonical_mapping(mapping_raw, expected_sha256=bank.mapping_sha256)
    assert bank.kb_sha256 == f.digest(kb_raw)
    draft = c.decode_record(kb_raw, c.BreedKnowledgeBase)
    assert bank.kb_version == draft.kb_version
    for breed, entry in zip(draft.breeds, bank.entries):
        assert breed.canonical_name == entry.canonical_name == mapping.names[breed.class_id]
        assert set(entry.attribute_ids) == {attribute.attribute_id for attribute in breed.attributes}
        assert set(entry.source_ids) == {source for attribute in breed.attributes for source in attribute.source_ids}
    with pytest.raises(c.ContractError) as rejected:
        load_knowledge_base(
            kb_raw, mapping=mapping, expected_sha256=f.digest(kb_raw), reviewed_sources={},
            allowed_attribute_ids=c.ATTRIBUTE_VOCABULARY_V1,
        )
    assert rejected.value.code is c.ErrorCode.UNREVIEWED_SOURCE
