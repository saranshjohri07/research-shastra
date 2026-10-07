from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db.models import Paper
from ..retrieval.retriever import RetrievedEvidence, retrieve_evidence
from ..schemas.research import (
    EvidenceResponse,
    ResearchQueryRequest,
    ResearchQueryResponse,
)
from .answer_service import generate_grounded_answer
from .citation_service import build_citations
from .context_builder import build_evidence_context


INSUFFICIENT_EVIDENCE_ANSWER = (
    "The paper does not provide enough information to answer this question."
)
SEMANTIC_ABSTENTION_REASON = (
    "The supplied paper does not contain information needed to answer this question."
)


def _evidence_responses(
    evidence: list[RetrievedEvidence],
) -> list[EvidenceResponse]:
    return [EvidenceResponse(**item.__dict__) for item in evidence]


def _normalize_generated_answer(
    generated_answer: str | dict[str, str | bool | None],
) -> tuple[bool, str | None, str | None]:
    if isinstance(generated_answer, dict):
        answerable = generated_answer.get("answerable")
        answer = generated_answer.get("answer")
        reason = generated_answer.get("reason") or generated_answer.get("abstention_reason")

        if answerable is False:
            return False, None, reason or SEMANTIC_ABSTENTION_REASON

        if answerable is True:
            if isinstance(answer, str) and answer.strip():
                return True, answer.strip(), None
            if answer is None:
                return False, None, reason or SEMANTIC_ABSTENTION_REASON
            raise ValueError("The answer provider returned an invalid structured answer")

    if isinstance(generated_answer, str):
        answer_text = generated_answer.strip()
        if not answer_text:
            raise RuntimeError("The answer provider returned an empty response")
        lowered = answer_text.lower()
        if any(
            marker in lowered
            for marker in (
                "does not contain",
                "not enough information",
                "cannot determine",
                "not provided",
                "does not provide enough information",
            )
        ):
            return False, None, SEMANTIC_ABSTENTION_REASON
        return True, answer_text, None

    raise TypeError("Unsupported answer payload returned by the generation service")


def answer_research_query(
    db: Session,
    request: ResearchQueryRequest,
) -> ResearchQueryResponse:
    if db.get(Paper, request.paper_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paper not found",
        )

    evidence = retrieve_evidence(
        db=db,
        paper_id=request.paper_id,
        question=request.question,
        top_k=request.top_k,
    )
    if not evidence:
        return ResearchQueryResponse(
            question=request.question,
            answerable=False,
            answer=None,
            abstention_reason="no_evidence",
            citations=[],
            evidence=[],
        )

    threshold = get_settings().retrieval_min_similarity
    if max(item.similarity for item in evidence) < threshold:
        return ResearchQueryResponse(
            question=request.question,
            answerable=False,
            answer=None,
            abstention_reason="low_similarity",
            citations=[],
            evidence=_evidence_responses(evidence),
        )

    context = build_evidence_context(evidence)
    generated_answer = generate_grounded_answer(request.question, context)
    answerable, answer, abstention_reason = _normalize_generated_answer(generated_answer)

    if not answerable:
        return ResearchQueryResponse(
            question=request.question,
            answerable=False,
            answer=None,
            abstention_reason=abstention_reason,
            citations=[],
            evidence=_evidence_responses(evidence),
        )

    return ResearchQueryResponse(
        question=request.question,
        answerable=True,
        answer=answer,
        abstention_reason=None,
        citations=build_citations(evidence),
        evidence=_evidence_responses(evidence),
    )