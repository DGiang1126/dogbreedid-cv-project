"""Load pinned knowledge artifacts and collect native tokenizer counts."""
from collections.abc import Callable, Collection, Mapping

from . import contracts as c


KB_MORPHOLOGY_VERSION = "kb-morphology-v1"
KB_MORPHOLOGY_V1 = frozenset({
    "ear_position",
    "ear_shape",
    "coat_color",
    "coat_length",
    "coat_texture",
    "muzzle_length",
    "tail_shape",
    "body_build",
    "breed_size",
    "head_shape",
    "facial_hair",
    "coat_fullness",
})


def load_knowledge_base(
    raw: bytes,
    *,
    mapping: c.CanonicalMapping,
    expected_sha256: c.Sha256,
    reviewed_sources: Mapping[str, str],
    allowed_attribute_ids: Collection[str],
) -> c.KnowledgeSnapshot:
    """Load bytes against the independently reviewed sources and vocabulary."""
    snapshot = c.validate_knowledge_snapshot(
        raw, mapping=mapping, expected_sha256=expected_sha256,
        reviewed_sources=reviewed_sources, allowed_attribute_ids=allowed_attribute_ids,
    )
    if any(not breed.attributes for breed in snapshot.data.breeds):
        raise c.ContractError("every breed needs reviewed attributes", c.ErrorCode.INVALID_SNAPSHOT)
    return snapshot


def load_prompt_bank(
    raw: bytes,
    *,
    mapping: c.CanonicalMapping,
    expected_sha256: c.Sha256,
    knowledge: c.KnowledgeSnapshot | None = None,
) -> c.PromptSnapshot:
    """Bind the exact mapping and the reviewed KB when needed."""
    snapshot = c.validate_prompt_snapshot(
        raw, mapping=mapping, expected_sha256=expected_sha256, knowledge=knowledge,
    )
    if any(entry.canonical_name not in entry.prompt_text for entry in snapshot.data.entries):
        raise c.ContractError("each prompt must retain its exact breed name", c.ErrorCode.INVALID_SNAPSHOT)
    return snapshot


def audit_prompt_bank(
    bank: c.PromptSnapshot,
    *,
    profile: c.ApprovedProfile,
    count_untruncated_tokens: Callable[[str], int],
) -> c.TokenAudit:
    """Count unpadded prompts with the caller's pinned native preprocessing.

    The counter must use the adapter's text normalization and special tokens,
    with padding and truncation disabled. Overflow is reported in the audit.
    """
    token_limit = profile.limits.get("prompt_token_limit")
    if type(token_limit) is not int or token_limit <= 0:
        raise c.ContractError("trusted profile needs a positive prompt_token_limit", c.ErrorCode.INVALID_SNAPSHOT)
    entries = []
    for entry in bank.data.entries:
        token_count = count_untruncated_tokens(entry.prompt_text)
        entries.append(c.TokenAuditEntry(
            class_id=entry.class_id, token_count=token_count,
            truncated=token_count > token_limit,
        ))
    return c.TokenAudit(
        prompt_bank_sha256=bank.prompt_bank_sha256, profile_id=profile.profile_id,
        tokenizer_revision=profile.tokenizer_revision, token_limit=token_limit,
        entries=tuple(entries),
    )
