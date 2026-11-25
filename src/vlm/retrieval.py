"""Deterministic local breed lookup for the assistant."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import re
from time import perf_counter
import unicodedata

from . import contracts as c


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFC", text).casefold()
    return " ".join(unicodedata.normalize("NFC", folded).split())


def render_context(entries: tuple[c.RetrievedEntry, ...]) -> str:
    """Render one line per attribute with its KB sources beside the claim."""
    return "\n".join(
        f"{entry.canonical_name} | {attribute.attribute_id}: {attribute.value} [{', '.join(attribute.source_ids)}]"
        for entry in entries for attribute in entry.attributes
    )


def _resolve_targets(
    request: c.RetrievalRequest, knowledge: c.KnowledgeSnapshot,
    unresolved_mentions: tuple[str, ...], image_only: bool,
) -> list[c.BreedLookup]:
    aliases: dict[str, set[int]] = defaultdict(set)
    for breed in knowledge.data.breeds:
        for alias in (breed.canonical_name, *breed.aliases["vi"], *breed.aliases["en"]):
            aliases[_normalize(alias)].add(breed.class_id)
    for mention in unresolved_mentions:
        normalized = _normalize(mention)
        if normalized:
            aliases.setdefault(normalized, set())

    ordered = sorted(aliases, key=lambda alias: (-len(alias), alias))
    phrases = "|".join(re.escape(alias) for alias in ordered)
    matches = re.finditer(rf"(?<!\w)(?:{phrases})(?!\w)", _normalize(request.question))
    resolved = set(request.explicit_class_ids)
    lookups = [
        c.BreedLookup(mention=None, class_id=class_id, status=c.BreedLookupStatus.HIT, candidate_class_ids=())
        for class_id in dict.fromkeys(request.explicit_class_ids)
    ]
    seen_mentions: set[str] = set()
    for match in matches:
        mention = match.group()
        if mention in seen_mentions:
            continue
        seen_mentions.add(mention)
        candidates = tuple(sorted(aliases[mention]))
        if len(candidates) > 1:
            lookups.append(c.BreedLookup(
                mention=mention, class_id=None, status=c.BreedLookupStatus.AMBIGUOUS,
                candidate_class_ids=candidates,
            ))
        elif not candidates:
            lookups.append(c.BreedLookup(
                mention=mention, class_id=None, status=c.BreedLookupStatus.MISS,
                candidate_class_ids=(), reason="no_alias_match",
            ))
        elif candidates[0] not in resolved:
            resolved.add(candidates[0])
            lookups.append(c.BreedLookup(
                mention=mention, class_id=candidates[0], status=c.BreedLookupStatus.HIT,
                candidate_class_ids=(),
            ))
    if lookups:
        return lookups

    cnn = request.cnn_context
    if image_only and not unresolved_mentions and cnn is not None and cnn.class_id is not c.MISSING:
        return [c.BreedLookup(
            mention=None, class_id=cnn.class_id, status=c.BreedLookupStatus.HIT, candidate_class_ids=(),
        )]
    return [c.BreedLookup(
        mention=None, class_id=None, status=c.BreedLookupStatus.MISS, candidate_class_ids=(),
        reason="no_alias_match",
    )]


def _fit_context(
    lookups: list[c.BreedLookup], knowledge: c.KnowledgeSnapshot, budget: int,
    counter: c.ContextTokenCounter,
) -> tuple[tuple[c.RetrievedEntry, ...], str, int, int]:
    entries: list[c.RetrievedEntry] = []
    context = ""
    used_tokens = omitted = 0
    for index, lookup in enumerate(lookups):
        if lookup.class_id is None:
            continue
        breed = knowledge.data.breeds[lookup.class_id]
        retained: tuple[c.MorphAttribute, ...] = ()
        for attribute in breed.attributes:
            candidate = c.RetrievedEntry(
                class_id=breed.class_id, canonical_name=breed.canonical_name,
                attributes=(*retained, attribute),
            )
            candidate_context = render_context((*entries, candidate))
            if budget:
                try:
                    token_count = c.count_context_tokens(counter, candidate_context)
                except Exception as exc:
                    raise c.ContractError("KB tokenizer failed", c.ErrorCode.RETRIEVAL_ERROR) from exc
                if token_count <= budget:
                    retained = candidate.attributes
                    context, used_tokens = candidate_context, token_count
                    continue
            omitted += 1
        if retained:
            entries.append(c.RetrievedEntry(
                class_id=breed.class_id, canonical_name=breed.canonical_name, attributes=retained,
            ))
        else:
            lookups[index] = replace(lookup, status=c.BreedLookupStatus.MISS, reason="budget_exhausted")
    return tuple(entries), context, used_tokens, omitted


def retrieve_knowledge(
    request: c.RetrievalRequest,
    *,
    knowledge: c.KnowledgeSnapshot,
    count_context_tokens: c.ContextTokenCounter,
    policy_mode: c.PolicyMode | None = None,
    unresolved_mentions: tuple[str, ...] = (),
    image_only: bool = False,
) -> c.RetrievalResult:
    """Resolve authorized targets and fit complete attributes into the KB slot.

    The caller binds policy and the profile tokenizer. An applicable request
    requires BREED_CONTEXT_ALLOWED. The caller may identify unresolved phrases
    or mark an image-only question eligible for CNN fallback.
    """
    start = perf_counter()
    if not isinstance(request, c.RetrievalRequest):
        raise c.ContractError("expected a retrieval request")
    budget = request.context_budget.max_tokens
    lookups: list[c.BreedLookup] = []
    entries: tuple[c.RetrievedEntry, ...] = ()
    context = ""
    used = omitted = 0
    error = c.MISSING

    try:
        c.validate_retrieval_request(request, knowledge=knowledge, policy_mode=policy_mode)
        if request.lookup_applicable:
            lookups = _resolve_targets(request, knowledge, unresolved_mentions, image_only)
            entries, context, used, omitted = _fit_context(lookups, knowledge, budget, count_context_tokens)
    except c.ContractError as exc:
        if exc.code in {c.ErrorCode.INVALID_INPUT, c.ErrorCode.POLICY_BLOCKED}:
            raise
        lookups = []
        status = c.RetrievalStatus.RETRIEVAL_ERROR
        error = c.ErrorInfo(code=exc.code, message_vi="Không thể truy xuất tri thức giống chó.", retryable=False)
    else:
        if not request.lookup_applicable:
            status = c.RetrievalStatus.NOT_APPLICABLE
        elif entries:
            status = c.RetrievalStatus.HIT if len(entries) == len(lookups) else c.RetrievalStatus.PARTIAL_HIT
        elif any(lookup.status is c.BreedLookupStatus.AMBIGUOUS for lookup in lookups):
            status = c.RetrievalStatus.AMBIGUOUS
        else:
            status = c.RetrievalStatus.MISS

    return c.RetrievalResult(
        status=status, per_breed=tuple(lookups),
        resolved_class_ids=tuple(item.class_id for item in lookups if item.class_id is not None),
        entries=entries, context=context,
        source_ids=tuple(dict.fromkeys(
            source for entry in entries for attribute in entry.attributes for source in attribute.source_ids
        )),
        kb_version=request.kb_version, kb_sha256=request.kb_sha256,
        truncation=c.TruncationInfo(
            applied=omitted > 0, budget_tokens=budget, used_tokens=used, omitted_attribute_count=omitted,
        ),
        latency_ms=(perf_counter() - start) * 1_000, error=error,
    )
