from typing import Any, Literal, TypedDict

from ..retrieval.retriever import RetrievedEvidence
from ..schemas.research import CitationResponse


class TraceEventState(TypedDict):
    thread_id: str
    stage: str
    status: Literal["started", "completed", "failed", "abstained", "flagged"]
    timestamp: str
    message: str
    metadata: dict[str, Any]


class ResearchState(TypedDict, total=False):
    question: str
    thread_id: str
    paper_ids: list[str]
    top_k: int
    query_type: str
    sub_questions: list[str]
    retrieval_plan: dict[str, Any]
    retrieved_chunks: list[RetrievedEvidence]
    selected_evidence: list[RetrievedEvidence]
    confidence: float
    user_decision: Literal["continue", "flag"] | None
    draft_answer: str | None
    answerable: bool
    abstention_reason: str | None
    citations: list[CitationResponse]
    validation: dict[str, Any]
    trace: list[TraceEventState]
    status: Literal[
        "pending",
        "running",
        "completed",
        "abstained",
        "failed",
        "flagged",
        "awaiting_review",
    ]
    error: str | None
