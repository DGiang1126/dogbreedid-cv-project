"""Scripted synthetic adapters. No model, tokenizer, lookup or I/O."""
from __future__ import annotations

from collections.abc import Mapping

from src.vlm import contracts as c

from .fixtures import TIED_SCORES


def _unavailable() -> c.ErrorInfo:
    return c.ErrorInfo(code=c.ErrorCode.MODEL_UNAVAILABLE, message_vi="Model minh họa không khả dụng", retryable=True)


class MockZeroShotClassifier(c.ZeroShotClassifier):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def classify(self, request: c.ZeroShotRequest, *, profile: c.ApprovedProfile,
                 prompts: c.PromptSnapshot) -> c.ZeroShotResult:
        metadata = dict(
            profile_id=profile.profile_id, model=profile.model, image_digest=request.image_digest,
            mapping_sha256=prompts.data.mapping_sha256, prompt_bank_sha256=prompts.prompt_bank_sha256,
            eval_ref=request.eval_ref, calibrated=False, latency_ms=0,
        )
        if self.fail:
            return c.ZeroShotResult(status=c.BranchOutcome.UNAVAILABLE, error=_unavailable(), **metadata)
        ranking = c.rank_scores(TIED_SCORES)
        return c.ZeroShotResult(
            status=c.BranchOutcome.OK, score_type=c.ScoreType.LOGIT, scores=TIED_SCORES,
            top1_class_id=ranking[0], top5_class_ids=ranking[:5], **metadata,
        )


class MockGenerativeAssistant(c.GenerativeAssistant):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def extract_attributes(self, request: c.AssistantRequest, *, profile: c.ApprovedProfile) -> c.AssistantResult:
        if request.operation is not c.Operation.ATTRIBUTES:
            raise c.ContractError("attributes-only operation expected")
        return self.execute(request, profile=profile)

    def answer_question(self, request: c.AssistantRequest, *, profile: c.ApprovedProfile) -> c.AssistantResult:
        if request.operation is c.Operation.ATTRIBUTES:
            raise c.ContractError("VQA operation expected")
        return self.execute(request, profile=profile)

    def execute(self, request: c.AssistantRequest, *, profile: c.ApprovedProfile) -> c.AssistantResult:
        metadata = dict(
            profile_id=profile.profile_id, model=profile.model, image_digest=request.image_digest,
            context_provenance=request.context_provenance, eval_ref=request.eval_ref, latency_ms=0,
        )
        if self.fail:
            return c.AssistantResult(status=c.BranchOutcome.UNAVAILABLE, error=_unavailable(), **metadata)
        attributes_only = request.operation is c.Operation.ATTRIBUTES
        attributes = (c.AttributeObservation(attribute_id="ear_shape", visibility=c.Visibility.UNKNOWN, value=None),)
        content = c.GeneratedContent(
            attributes=() if request.operation is c.Operation.VQA else attributes,
            answer_vi=None if attributes_only else "Câu trả lời minh họa.",
            disposition=c.Disposition.NOT_APPLICABLE if attributes_only else c.Disposition.ANSWERED,
            uncertainty=c.Uncertainty.UNSPECIFIED, abstention_reason=None,
            source_ids=() if attributes_only else request.allowed_source_ids[:1],
        )
        return c.AssistantResult(status=c.BranchOutcome.OK, generated=content, **metadata)


class MockKnowledgeRetriever(c.KnowledgeRetriever):
    """Return the caller's scripted success, miss, ambiguity or failure."""

    def __init__(self, result: c.RetrievalResult) -> None:
        self.result = result

    def __call__(self, request: c.RetrievalRequest, *, knowledge: c.KnowledgeSnapshot,
                 count_context_tokens: c.ContextTokenCounter) -> c.RetrievalResult:
        return self.result


class MockContextTokenCounter(c.ContextTokenCounter):
    """Look up explicitly configured counts, without tokenizing anything."""

    def __init__(self, counts: Mapping[str, int]) -> None:
        self._counts = dict(counts)

    def __call__(self, context: str) -> int:
        return self._counts[context]
