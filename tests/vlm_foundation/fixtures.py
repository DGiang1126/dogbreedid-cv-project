"""Synthetic in-memory inputs.

Only the 30 exact canonical names mirror repository metadata. The knowledge,
reviews, prompts, model revisions, scores and profile approvals are test data.
Nothing here is research evidence or a production allowlist.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

from src.vlm import contracts as c


CANONICAL_NAMES = (
    "Eskimo Dog", "Malamute", "Siberian Husky", "Greater Swiss Mountain Dog",
    "Appenzeller", "EntleBucher", "Toy Poodle", "Miniature Poodle", "Standard Poodle",
    "Miniature Schnauzer", "Standard Schnauzer", "Giant Schnauzer", "English Setter",
    "Irish Setter", "Gordon Setter", "English Springer", "Welsh Springer Spaniel",
    "Cocker Spaniel", "Norfolk Terrier", "Norwich Terrier", "Border Terrier",
    "Collie", "Shetland Sheepdog", "Border Collie", "Golden Retriever",
    "Labrador Retriever", "Chesapeake Bay Retriever", "Japanese Spaniel",
    "Pekinese", "Shih-Tzu",
)
SOURCE_BYTES = b"synthetic upload fixture: source bytes differ from decoded RGB"
KB_ATTRIBUTE_ID = "synthetic_coat"
REVIEWED_SOURCES = MappingProxyType({
    f"fixture-source-{i}": f"fixture-review-{i}" for i in c.CLASS_IDS
})
PROMPT_TOKEN_LIMIT = 32
QUESTION = "Câu hỏi minh họa về giống số 7"
CONTEXT = "Miniature Poodle: synthetic value [fixture-source-7]"


def render_context(entries: tuple[c.RetrievedEntry, ...]) -> str:
    """Render synthetic entries independently of the scripted result text."""
    return "\n".join(
        f"{entry.canonical_name}: {attribute.value} [{', '.join(attribute.source_ids)}]"
        for entry in entries
        for attribute in entry.attributes
    )
TIED_SCORES = (9, 10, 10, 8, 7, 7) + (0,) * 24


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class SyntheticArtifacts:
    mapping_raw: bytes
    mapping: c.CanonicalMapping
    knowledge_raw: bytes
    knowledge: c.KnowledgeSnapshot
    name_raw: bytes
    name: c.PromptSnapshot
    morphology_raw: bytes
    morphology: c.PromptSnapshot


def make_artifacts() -> SyntheticArtifacts:
    mapping_raw = json_bytes({str(i): name for i, name in enumerate(CANONICAL_NAMES)})
    mapping = c.decode_canonical_mapping(mapping_raw, expected_sha256=digest(mapping_raw))
    knowledge_raw = json_bytes({
        "schema_version": "1.0",
        "kb_version": "synthetic-kb-v1",
        "mapping_sha256": mapping.mapping_sha256,
        "breeds": [
            {
                "class_id": i,
                "canonical_name": name,
                "aliases": {"vi": [f"giống minh họa {i}"], "en": ["shared fixture alias"]},
                "attributes": [
                    {"attribute_id": KB_ATTRIBUTE_ID, "value": "synthetic value", "source_ids": [f"fixture-source-{i}"]},
                ],
            }
            for i, name in enumerate(CANONICAL_NAMES)
        ],
        "sources": [
            {
                "source_id": source_id,
                "title": "Synthetic source for shape tests",
                "url": "https://example.invalid/vlm-fixture",
                "review_status": "approved",
                "review_ref": review_ref,
            }
            for source_id, review_ref in REVIEWED_SOURCES.items()
        ],
    })
    knowledge = c.validate_knowledge_snapshot(
        knowledge_raw, mapping=mapping, expected_sha256=digest(knowledge_raw),
        reviewed_sources=REVIEWED_SOURCES, allowed_attribute_ids={KB_ATTRIBUTE_ID},
    )
    name_bank = {
        "schema_version": "1.0",
        "bank_version": "synthetic-name-v1",
        "prompt_type": "name",
        "mapping_sha256": mapping.mapping_sha256,
        "entries": [
            {"class_id": i, "canonical_name": name, "prompt_text": f"Synthetic name prompt for {name}"}
            for i, name in enumerate(CANONICAL_NAMES)
        ],
    }
    name_raw = json_bytes(name_bank)
    morphology_raw = json_bytes({
        **name_bank,
        "bank_version": "synthetic-morphology-v1",
        "prompt_type": "morphology",
        "kb_version": knowledge.data.kb_version,
        "kb_sha256": knowledge.kb_sha256,
        "entries": [
            {
                **entry,
                "prompt_text": f"{entry['canonical_name']} with a synthetic attribute",
                "attribute_ids": [KB_ATTRIBUTE_ID],
                "source_ids": [f"fixture-source-{entry['class_id']}"],
            }
            for entry in name_bank["entries"]
        ],
    })
    name = c.validate_prompt_snapshot(name_raw, mapping=mapping, expected_sha256=digest(name_raw), knowledge=None)
    morphology = c.validate_prompt_snapshot(
        morphology_raw, mapping=mapping, expected_sha256=digest(morphology_raw), knowledge=knowledge,
    )
    return SyntheticArtifacts(mapping_raw, mapping, knowledge_raw, knowledge, name_raw, name, morphology_raw, morphology)


def make_image() -> c.ValidatedRgb:
    return c.ValidatedRgb(pixels=bytes(range(12)), width=2, height=2, image_digest=digest(SOURCE_BYTES))


def make_profile(kind: str = "encoder") -> c.ApprovedProfile:
    return c.ApprovedProfile(
        profile_id=f"fixture-{kind}-v1",
        model=c.ModelRef(model_id=f"fixture/{kind}", model_revision=digest(b"synthetic model")[:40],
                         processor_revision=digest(b"synthetic processor")[:40]),
        config_sha256=digest(json_bytes({"synthetic_profile": kind})),
        device="cpu", dtype="fixture-dtype", limits={"prompt_token_limit": PROMPT_TOKEN_LIMIT},
        approval_ref=f"synthetic-approval-{kind}", tokenizer_revision=digest(b"synthetic tokenizer")[:40],
    )


def make_eval_ref(experiment_id: str = "E12-A") -> c.EvalRef:
    return c.EvalRef(experiment_id=experiment_id,
                     run_id=f"{experiment_id}_pilot_20990101T120000Z_123e4567-e89b-42d3-a456-426614174000")


def make_token_audit(bank: c.PromptSnapshot, profile: c.ApprovedProfile) -> c.PromptTokenAudit:
    return c.PromptTokenAudit(audits=(
        c.TokenAudit(prompt_bank_sha256=bank.prompt_bank_sha256, profile_id=profile.profile_id,
                     tokenizer_revision=profile.tokenizer_revision, token_limit=PROMPT_TOKEN_LIMIT,
                     entries=tuple(c.TokenAuditEntry(class_id=i, token_count=6, truncated=False) for i in c.CLASS_IDS)),
    ))


def make_retrieval_request(knowledge: c.KnowledgeSnapshot) -> c.RetrievalRequest:
    return c.RetrievalRequest(
        question=QUESTION, explicit_class_ids=(7,),
        cnn_context=None, kb_version=knowledge.data.kb_version, kb_sha256=knowledge.kb_sha256,
        context_budget=c.ContextBudget(max_tokens=20, tokenizer_revision=make_profile("assistant").tokenizer_revision),
        lookup_applicable=True,
    )


def make_retrieval_result(knowledge: c.KnowledgeSnapshot,
                          status: c.RetrievalStatus = c.RetrievalStatus.HIT) -> c.RetrievalResult:
    hit = c.BreedLookup(mention="fixture breed", class_id=7, status=c.BreedLookupStatus.HIT, candidate_class_ids=())
    ambiguous = c.BreedLookup(mention="shared fixture alias", class_id=None,
                              status=c.BreedLookupStatus.AMBIGUOUS, candidate_class_ids=(6, 7, 8))
    miss = c.BreedLookup(mention="unknown fixture", class_id=None,
                         status=c.BreedLookupStatus.MISS, candidate_class_ids=(), reason="no_alias_match")
    lookups = {
        c.RetrievalStatus.HIT: (hit,),
        c.RetrievalStatus.PARTIAL_HIT: (hit, ambiguous),
        c.RetrievalStatus.MISS: (miss,),
        c.RetrievalStatus.AMBIGUOUS: (ambiguous,),
        c.RetrievalStatus.NOT_APPLICABLE: (),
        c.RetrievalStatus.RETRIEVAL_ERROR: (),
    }[status]
    has_hit = status in (c.RetrievalStatus.HIT, c.RetrievalStatus.PARTIAL_HIT)
    breed = knowledge.data.breeds[7]
    return c.RetrievalResult(
        status=status, per_breed=lookups, resolved_class_ids=(7,) if has_hit else (),
        entries=(c.RetrievedEntry(class_id=7, canonical_name=breed.canonical_name,
                                  attributes=breed.attributes),) if has_hit else (),
        context=CONTEXT if has_hit else "", source_ids=("fixture-source-7",) if has_hit else (),
        kb_version=knowledge.data.kb_version, kb_sha256=knowledge.kb_sha256, latency_ms=0,
        truncation=c.TruncationInfo(applied=False, budget_tokens=20, used_tokens=8 if has_hit else 0,
                                    omitted_attribute_count=0),
        error=c.ErrorInfo(code=c.ErrorCode.RETRIEVAL_ERROR, message_vi="Lỗi minh họa", retryable=True)
        if status is c.RetrievalStatus.RETRIEVAL_ERROR else c.MISSING,
    )


def make_zero_shot_request(
    profile: c.ApprovedProfile, bank: c.PromptSnapshot, *, eval_ref: c.EvalRef | None = None,
) -> c.ZeroShotRequest:
    image = make_image()
    return c.ZeroShotRequest(
        image=image, image_digest=image.image_digest, profile_id=profile.profile_id,
        mapping_sha256=bank.data.mapping_sha256, prompt_bank_sha256=bank.prompt_bank_sha256,
        eval_ref=eval_ref if eval_ref is not None else c.MISSING,
    )


def make_assistant_request(
    profile: c.ApprovedProfile, *, operation: c.Operation = c.Operation.ATTRIBUTES_AND_VQA,
    retrieval: c.RetrievalResult | None = None, eval_ref: c.EvalRef | None = None,
) -> c.AssistantRequest:
    image = make_image()
    context = retrieval.context if retrieval is not None else ""
    return c.AssistantRequest(
        operation=operation, image=image, image_digest=image.image_digest,
        question=None if operation is c.Operation.ATTRIBUTES else QUESTION,
        cnn_context=None, intermediate_attributes=None, kb_context=context,
        allowed_source_ids=retrieval.source_ids if retrieval is not None else (),
        context_provenance=c.ContextProvenance(
            retrieval_status=retrieval.status if retrieval is not None else c.RetrievalStatus.NOT_APPLICABLE,
            kb_version=retrieval.kb_version if retrieval is not None else None,
            kb_sha256=retrieval.kb_sha256 if retrieval is not None else None,
            context_sha256=digest(context.encode("utf-8")),
        ),
        profile_id=profile.profile_id, policy_mode=c.PolicyMode.BREED_CONTEXT_ALLOWED,
        eval_ref=eval_ref if eval_ref is not None else c.MISSING,
    )


def request_identity(request: c.ZeroShotRequest | c.AssistantRequest) -> dict:
    return {
        "request_id": "fixture-request",
        "session_id": "fixture-session",
        "image_digest": request.image_digest,
        "revision": 3,
        "cnn_mode": c.CNNMode.STANDARD,
        "branch": c.Branch.COMPARISON if isinstance(request, c.ZeroShotRequest) else c.Branch.ASSISTANT,
        "profile_id": request.profile_id,
    }


def make_branch_request(request: c.ZeroShotRequest | c.AssistantRequest) -> c.BranchRequest:
    return c.BranchRequest(
        **request_identity(request), remaining_budget_ms=0,
        cnn_decision=c.CNNDecisionSnapshot(state=c.CNNState.DECIDED, status=c.CNNStatus.ACCEPT, class_id=7),
    )


def make_worker_job(request: c.ZeroShotRequest | c.AssistantRequest) -> c.WorkerJob:
    return c.WorkerJob(**request_identity(request), remaining_budget_ms=0, payload=request)


def make_worker_reply(request: c.ZeroShotRequest | c.AssistantRequest,
                      result: c.ZeroShotResult | c.AssistantResult) -> c.WorkerReply:
    return c.WorkerReply(**request_identity(request), result=result)


def make_session_state(request: c.ZeroShotRequest | c.AssistantRequest) -> c.SessionState:
    branch_request = make_branch_request(request)
    return c.SessionState(
        session_id=branch_request.session_id, revision=branch_request.revision,
        image_digest=branch_request.image_digest, cnn_mode=branch_request.cnn_mode,
        cnn_decision=branch_request.cnn_decision,
        latest_request_ids={branch: None for branch in c.Branch},
        selected_profile_ids={branch: None for branch in c.Branch},
        branch_states={branch: c.BranchState(status=c.RuntimeStatus.DISABLED, request_id=None)
                       for branch in c.Branch},
    )
