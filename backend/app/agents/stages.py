import re
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.orm import Session

from ..config import get_settings
from ..db.models import Paper
from ..retrieval.retriever import RetrievedEvidence, retrieve_evidence
from ..schemas.research import CitationResponse
from ..services.answer_service import generate_grounded_answer
from ..services.citation_service import build_citations
from ..services.context_builder import build_evidence_context
from ..services.research_service import _normalize_generated_answer
from .state import ResearchState, TraceEventState


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_trace(
    state: ResearchState,
    stage: str,
    status: TraceEventState["status"],
    message: str,
    metadata: dict | None = None,
) -> list[TraceEventState]:
    event: TraceEventState = {
        "thread_id": state["thread_id"],
        "stage": stage,
        "status": status,
        "timestamp": utc_timestamp(),
        "message": message,
        "metadata": metadata or {},
    }
    return [*state.get("trace", []), event]


def parse_planner_output(raw: str, question: str) -> dict:
    """Parse optional structured planner output; malformed output uses safe fallback."""
    try:
        payload = json_loads(raw)
    except (ValueError, TypeError):
        return planner_fallback(question)
    if not isinstance(payload, dict):
        return planner_fallback(question)
    query_type = payload.get("query_type")
    sub_questions = payload.get("sub_questions")
    if query_type not in ("factual", "comparison", "explanatory"):
        return planner_fallback(question)
    if not isinstance(sub_questions, list) or not sub_questions:
        return planner_fallback(question)
    cleaned = [str(item).strip() for item in sub_questions if str(item).strip()]
    if not cleaned:
        return planner_fallback(question)
    return {"query_type": query_type, "sub_questions": cleaned}


def json_loads(value: str):
    import json

    return json.loads(value)


def planner_fallback(question: str) -> dict:
    lowered = question.lower()
    if any(word in lowered for word in ("compare", "difference", "versus", "between")):
        query_type = "comparison"
    elif any(word in lowered for word in ("how", "why", "mechanism", "method")):
        query_type = "explanatory"
    else:
        query_type = "factual"
    return {"query_type": query_type, "sub_questions": [question]}


def planner(state: ResearchState) -> ResearchState:
    trace = add_trace(state, "planner", "started", "Classifying and planning the question")
    question = state["question"].strip()
    plan = planner_fallback(question)
    query_type = plan["query_type"]
    sub_questions = plan["sub_questions"]

    return {
        **state,
        "query_type": query_type,
        "sub_questions": sub_questions,
        "retrieval_plan": {
            "strategy": "paper_scoped_vector",
            "top_k": state.get("top_k", 5),
            "paper_ids": state["paper_ids"],
        },
        "status": "running",
        "trace": add_trace(
            {**state, "trace": trace},
            "planner",
            "completed",
            "Question plan generated",
            {"query_type": query_type, "sub_questions": sub_questions},
        ),
    }


def retriever(db: Session, state: ResearchState) -> ResearchState:
    started = add_trace(state, "retriever", "started", "Retrieving paper-scoped evidence")
    try:
        evidence: list[RetrievedEvidence] = []
        for paper_id in state["paper_ids"]:
            if db.get(Paper, paper_id) is None:
                raise ValueError(f"Paper not found: {paper_id}")
            evidence.extend(
                retrieve_evidence(
                    db=db,
                    paper_id=paper_id,
                    question=state["question"],
                    top_k=state.get("top_k", 5),
                )
            )
        evidence.sort(key=lambda item: item.similarity, reverse=True)
        evidence = evidence[: state.get("top_k", 5)]
        return {
            **state,
            "retrieved_chunks": evidence,
            "trace": add_trace(
                {**state, "trace": started},
                "retriever",
                "completed",
                "Evidence retrieval completed",
                {"retrieved_count": len(evidence)},
            ),
        }
    except Exception as exc:
        return {
            **state,
            "retrieved_chunks": [],
            "status": "failed",
            "error": str(exc),
            "trace": add_trace(
                {**state, "trace": started},
                "retriever",
                "failed",
                "Evidence retrieval failed",
                {"error": str(exc)},
            ),
        }


def evidence_gatherer(state: ResearchState) -> ResearchState:
    started = add_trace(state, "evidence_gatherer", "started", "Selecting diverse evidence")
    retrieved = state.get("retrieved_chunks", [])
    if not retrieved:
        if state.get("status") == "failed":
            return {
                **state,
                "selected_evidence": [],
                "trace": add_trace(
                    {**state, "trace": started},
                    "evidence_gatherer",
                    "failed",
                    "Evidence selection skipped after retrieval failure",
                    {"error": state.get("error")},
                ),
            }
        return {
            **state,
            "selected_evidence": [],
            "confidence": 0.0,
            "answerable": False,
            "abstention_reason": "no_evidence",
            "status": "abstained",
            "trace": add_trace(
                {**state, "trace": started},
                "evidence_gatherer",
                "abstained",
                "No evidence was retrieved",
                {"selected_count": 0},
            ),
        }

    selected: list[RetrievedEvidence] = []
    seen_chunks: set[str] = set()
    seen_sections: set[str] = set()
    for item in sorted(retrieved, key=lambda evidence: evidence.similarity, reverse=True):
        section = (item.section or "").lower()
        if item.chunk_id in seen_chunks:
            continue
        if len(selected) < 3 or section not in seen_sections:
            selected.append(item)
            seen_chunks.add(item.chunk_id)
            seen_sections.add(section)
        if len(selected) >= state.get("top_k", 5):
            break
    for rank, item in enumerate(selected, start=1):
        item.evidence_id = f"E{rank}"

    threshold = get_settings().retrieval_min_similarity
    confidence = max((item.similarity for item in selected), default=0.0)
    if confidence < threshold:
        return {
            **state,
            "selected_evidence": selected,
            "confidence": confidence,
            "answerable": False,
            "abstention_reason": "low_similarity",
            "status": "abstained",
            "trace": add_trace(
                {**state, "trace": started},
                "evidence_gatherer",
                "abstained",
                "Retrieved evidence was below the similarity threshold",
                {
                    "selected_count": len(selected),
                    "confidence": confidence,
                    "threshold": threshold,
                },
            ),
        }
    return {
        **state,
        "selected_evidence": selected,
        "confidence": confidence,
        "trace": add_trace(
            {**state, "trace": started},
            "evidence_gatherer",
            "completed",
            "Evidence selection completed",
            {
                "selected_count": len(selected),
                "confidence": confidence,
                "threshold": threshold,
            },
        ),
    }


def human_checkpoint(state: ResearchState) -> ResearchState:
    decision = state.get("user_decision")
    if decision is None:
        trace = add_trace(
            state,
            "human_checkpoint",
            "started",
            "Human review decision is required",
            {"decision": None},
        )
        return {
            **state,
            "user_decision": None,
            "status": "awaiting_review",
            "error": None,
            "trace": trace,
        }
    if decision not in ("continue", "flag"):
        trace = add_trace(
            state,
            "human_checkpoint",
            "failed",
            "Invalid human review decision",
            {"decision": decision},
        )
        return {
            **state,
            "status": "failed",
            "error": "Invalid human review decision",
            "trace": trace,
        }
    trace = add_trace(
        state,
        "human_checkpoint",
        "completed" if decision == "continue" else "flagged",
        "Continue decision received" if decision == "continue" else "Run flagged for human review",
        {"decision": decision},
    )
    return {
        **state,
        "user_decision": decision,
        "status": "running" if decision == "continue" else "flagged",
        "trace": trace,
    }


def reasoner(state: ResearchState) -> ResearchState:
    started = add_trace(state, "reasoner", "started", "Generating grounded answer")
    evidence = state.get("selected_evidence", [])
    if not evidence:
        return {
            **state,
            "draft_answer": None,
            "answerable": False,
            "abstention_reason": "no_evidence",
            "status": "abstained",
            "trace": add_trace(
                {**state, "trace": started},
                "reasoner",
                "abstained",
                "Grounded generation skipped because evidence is empty",
                {},
            ),
        }
    try:
        generated = generate_grounded_answer(
            state["question"],
            build_evidence_context(evidence),
        )
        answerable, answer, reason = _normalize_generated_answer(generated)
        if answerable:
            return {
                **state,
                "draft_answer": answer,
                "answerable": True,
                "abstention_reason": None,
                "trace": add_trace(
                    {**state, "trace": started},
                    "reasoner",
                    "completed",
                    "Grounded answer generated",
                    {"answerable": True},
                ),
            }
        return {
            **state,
            "draft_answer": None,
            "answerable": False,
            "abstention_reason": reason,
            "status": "abstained",
            "trace": add_trace(
                {**state, "trace": started},
                "reasoner",
                "abstained",
                "The supplied evidence was insufficient",
                {"answerable": False},
            ),
        }
    except Exception as exc:
        return {
            **state,
            "draft_answer": None,
            "answerable": False,
            "status": "failed",
            "error": str(exc),
            "trace": add_trace(
                {**state, "trace": started},
                "reasoner",
                "failed",
                "Grounded generation failed",
                {"error": str(exc)},
            ),
        }


def citation_agent(state: ResearchState) -> ResearchState:
    evidence = state.get("selected_evidence", [])
    citations = build_citations(evidence)
    trace = add_trace(
        state,
        "citation_agent",
        "completed",
        "Citation mappings generated",
        {"citation_count": len(citations)},
    )
    return {**state, "citations": citations, "trace": trace}


def validator(state: ResearchState) -> ResearchState:
    citations: list[CitationResponse] = state.get("citations", [])
    evidence = state.get("selected_evidence", [])
    evidence_ids = {item.evidence_id for item in evidence}
    draft = state.get("draft_answer") or ""
    markers = set(re.findall(r"\[E\d+\]", draft))
    marker_ids = {marker.strip("[]") for marker in markers}
    unknown_markers = sorted(marker_ids - evidence_ids)

    valid_citations = [
        citation
        for citation in citations
        if citation.evidence_id in evidence_ids
    ]
    if marker_ids:
        valid_citations = [
            citation
            for citation in valid_citations
            if citation.evidence_id in marker_ids
        ]

    marker_validation = bool(marker_ids) and not unknown_markers
    valid = marker_validation and bool(valid_citations)
    validation_reason = None
    if state.get("answerable", False) and not marker_ids:
        valid = False
        marker_validation = False
        validation_reason = "answer_has_no_evidence_markers"
    elif unknown_markers:
        validation_reason = "unknown_evidence_markers"
    status = "completed" if valid else "failed"
    if not state.get("answerable", False):
        status = "abstained"
    trace = add_trace(
        state,
        "validator",
        "completed" if valid else "failed",
        "Citation validation completed" if valid else "Invalid citation mapping detected",
        {
            "valid": valid,
            "unknown_markers": unknown_markers,
            "validated_citation_count": len(valid_citations),
            "marker_validation": marker_validation,
            "validation_reason": validation_reason,
        },
    )
    return {
        **state,
        "citations": valid_citations if valid else [],
        "validation": {
            "valid": valid,
            "unknown_markers": unknown_markers,
            "marker_validation": marker_validation,
            "validation_reason": validation_reason,
        },
        "status": status,
        "error": None if valid else (
            "Answer contains no evidence markers"
            if validation_reason == "answer_has_no_evidence_markers"
            else "Invalid citation mapping"
        ),
        "trace": trace,
    }
