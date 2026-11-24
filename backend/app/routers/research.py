from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from ..db.models import (
    HumanCheckpointDecision,
    Paper,
    ResearchRun,
    ResearchRunPaper,
)
from ..db.session import get_db
from ..schemas.research import (
    CitationResponse,
    EvidenceResponse,
    ResearchQueryRequest,
    ResearchQueryResponse,
)
from ..services.research_service import answer_research_query
from ..services.research_recovery import persist_graph_result
from ..agents.workflow import build_research_graph


router = APIRouter(prefix="/research", tags=["research"])


@router.post("/query", response_model=ResearchQueryResponse)
def query_research(
    request: ResearchQueryRequest,
    db: Session = Depends(get_db),
) -> ResearchQueryResponse:
    return answer_research_query(db, request)


class GraphResearchRequest(BaseModel):
    paper_ids: list[UUID] = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)

    @model_validator(mode="before")
    @classmethod
    def reject_client_thread_id(cls, values: Any) -> Any:
        if isinstance(values, dict) and "thread_id" in values:
            raise ValueError("thread_id is assigned from the persisted research run")
        if isinstance(values, dict) and "user_decision" in values:
            raise ValueError("user_decision must be submitted through human review")
        return values


class GraphEvidenceResponse(EvidenceResponse):
    model_config = ConfigDict(from_attributes=True)


class GraphResearchResponse(BaseModel):
    research_run_id: str
    question: str
    thread_id: str
    paper_ids: list[str]
    top_k: int
    query_type: str | None = None
    sub_questions: list[str] = Field(default_factory=list)
    retrieval_plan: dict[str, Any] | None = None
    retrieved_chunks: list[GraphEvidenceResponse] = Field(default_factory=list)
    selected_evidence: list[GraphEvidenceResponse] = Field(default_factory=list)
    confidence: float | None = None
    user_decision: Literal["continue", "flag"] | None = None
    draft_answer: str | None = None
    answerable: bool | None = None
    abstention_reason: str | None = None
    citations: list[CitationResponse] = Field(default_factory=list)
    validation: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    interrupts: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal[
        "pending",
        "running",
        "completed",
        "abstained",
        "failed",
        "flagged",
        "awaiting_review",
    ]
    error: str | None = None


class ResumeResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approved", "rejected"]
    feedback: str | None = None


@router.post("/graph/query", response_model=GraphResearchResponse)
def query_research_graph(
    request: GraphResearchRequest,
    http_request: Request,
    db: Session = Depends(get_db),
) -> GraphResearchResponse:
    paper_ids = list(dict.fromkeys(str(paper_id) for paper_id in request.paper_ids))
    missing_paper_ids = [
        paper_id for paper_id in paper_ids if db.get(Paper, paper_id) is None
    ]
    if missing_paper_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "One or more papers were not found", "paper_ids": missing_paper_ids},
        )

    resources = getattr(http_request.app.state, "checkpointer_resources", None)
    if resources is None:
        raise RuntimeError("LangGraph checkpointer is not initialized")

    run_id = str(uuid4())
    run = ResearchRun(
        id=run_id,
        question=request.question,
        status="running",
        top_k=request.top_k,
        thread_id=run_id,
    )
    db.add(run)
    db.flush()
    for position, paper_id in enumerate(paper_ids):
        db.add(
            ResearchRunPaper(
                research_run_id=run_id,
                paper_id=paper_id,
                position=position,
            )
        )
    db.commit()

    thread_id = get_persisted_thread_id(db, run_id)
    initial_state = {
        "question": request.question,
        "thread_id": thread_id,
        "paper_ids": paper_ids,
        "top_k": request.top_k,
        "trace": [],
        "status": "pending",
        "error": None,
    }
    try:
        result = invoke_research_run(db, run, initial_state, resources.saver)
    except Exception as error:
        db.rollback()
        run = db.get(ResearchRun, run_id)
        if run is None:
            raise RuntimeError("Persisted research run disappeared during execution") from error
        run.status = "failed"
        run.error = str(error)
        run.completed_at = datetime.now(timezone.utc)
        db.commit()
        raise

    run.status = {
        "completed": "completed",
        "abstained": "abstained",
        "failed": "failed",
        "flagged": "cancelled",
    }.get(result.get("status"), "running")
    run.answer = result.get("draft_answer")
    run.answerable = result.get("answerable")
    run.error = result.get("error")
    run.abstention_reason = result.get("abstention_reason")
    if run.status in {"completed", "abstained", "failed", "cancelled"}:
        run.completed_at = datetime.now(timezone.utc)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    pending_interrupts = result.pop("__interrupt__", [])
    if pending_interrupts:
        result["interrupts"] = [item.value for item in pending_interrupts]
        result["status"] = "awaiting_review"
    return GraphResearchResponse(**result, research_run_id=run_id)


@router.post(
    "/runs/{research_run_id}/resume",
    response_model=GraphResearchResponse,
)
def resume_research_run(
    research_run_id: UUID,
    request: ResumeResearchRequest,
    http_request: Request,
    db: Session = Depends(get_db),
) -> GraphResearchResponse:
    run = db.get(ResearchRun, str(research_run_id))
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Research run not found",
        )
    if run.status != "running" or run.resume_token is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Research run resume has already been claimed or is not resumable",
        )

    thread_id = run.thread_id
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Research run has no valid persisted thread_id",
        )
    try:
        UUID(thread_id)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Research run has no valid persisted thread_id",
        ) from error

    resources = getattr(http_request.app.state, "checkpointer_resources", None)
    if resources is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LangGraph checkpointer is not initialized",
        )

    config = {"configurable": {"thread_id": thread_id}}
    graph = build_research_graph(db, checkpointer=resources.saver)
    try:
        snapshot = graph.get_state(config)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to restore the saved research checkpoint",
        ) from error

    if snapshot.values.get("thread_id") != thread_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Saved checkpoint identity does not match the research run",
        )

    if not any(
        isinstance(getattr(interrupt, "value", None), dict)
        and interrupt.value.get("type") == "human_review"
        for interrupt in snapshot.interrupts
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Research run is not awaiting human review",
        )

    checkpoint_config = getattr(snapshot, "config", None) or {}
    configurable = checkpoint_config.get("configurable", {})
    checkpoint_ns = configurable.get("checkpoint_ns")
    checkpoint_id = configurable.get("checkpoint_id")
    resume_token = str(uuid4())
    claim = db.execute(
        update(ResearchRun)
        .where(
            ResearchRun.id == str(research_run_id),
            ResearchRun.status == "running",
            ResearchRun.resume_token.is_(None),
        )
        .values(resume_token=resume_token)
        .execution_options(synchronize_session=False)
    )
    if claim.rowcount != 1:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Research run resume has already been claimed or is not resumable",
        )
    try:
        db.add(
            HumanCheckpointDecision(
                research_run_id=run.id,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id,
                decision=request.decision,
                comment=request.feedback,
            )
        )
        db.commit()
    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to persist the research run resume decision",
        ) from error

    locked_run = db.scalar(
        select(ResearchRun)
        .where(ResearchRun.id == run.id)
        .with_for_update()
    )
    if (
        locked_run is None
        or locked_run.resume_token != resume_token
    ):
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Research run resume is being handled by recovery",
        )
    if locked_run.status != "running":
        db.commit()
        return get_research_run(research_run_id, http_request, db)

    try:
        locked_snapshot = graph.get_state(config)
    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The persisted decision remains available, but its checkpoint "
                "could not be revalidated for resume"
            ),
        ) from error
    if locked_snapshot.values.get("thread_id") != thread_id:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Saved checkpoint identity does not match the research run",
        )
    locked_interrupts = [
        item
        for item in locked_snapshot.interrupts
        if isinstance(getattr(item, "value", None), dict)
        and item.value.get("type") == "human_review"
    ]
    locked_configurable = (
        getattr(locked_snapshot, "config", None) or {}
    ).get("configurable", {})
    if (
        not locked_interrupts
        or locked_configurable.get("checkpoint_id") != checkpoint_id
        or locked_configurable.get("checkpoint_ns") != checkpoint_ns
    ):
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "The persisted decision is awaiting checkpoint recovery; "
                "the graph was not resumed a second time"
            ),
        )
    run = locked_run

    try:
        result = dict(
            graph.invoke(
                Command(
                    resume={
                        "decision": request.decision,
                        "feedback": request.feedback,
                    }
                ),
                config=config,
            )
        )
    except Exception as error:
        db.rollback()
        pending_run = db.scalar(
            select(ResearchRun)
            .where(
                ResearchRun.id == str(research_run_id),
                ResearchRun.status == "running",
                ResearchRun.resume_token == resume_token,
            )
            .with_for_update()
        )
        if pending_run is not None:
            pending_run.error = f"Research run recovery pending: {error}"
            try:
                db.commit()
            except Exception:
                db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unable to resume the saved research checkpoint; "
                "the persisted decision remains available for recovery"
            ),
        ) from error

    persist_graph_result(run, result)
    try:
        db.commit()
    except Exception as error:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The graph resumed but SQL result persistence failed; "
                "checkpoint state may have advanced and the resume claim "
                "remains consumed"
            ),
        ) from error

    pending_interrupts = result.pop("__interrupt__", [])
    if pending_interrupts:
        result["interrupts"] = [item.value for item in pending_interrupts]
        result["status"] = "awaiting_review"
    return GraphResearchResponse(**result, research_run_id=run.id)


@router.get(
    "/runs/{research_run_id}",
    response_model=GraphResearchResponse,
)
def get_research_run(
    research_run_id: UUID,
    http_request: Request,
    db: Session = Depends(get_db),
) -> GraphResearchResponse:
    run = db.get(ResearchRun, str(research_run_id))
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Research run not found",
        )
    thread_id = run.thread_id
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Research run has no valid persisted thread_id",
        )

    resources = getattr(http_request.app.state, "checkpointer_resources", None)
    if resources is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LangGraph checkpointer is not initialized",
        )

    config = {"configurable": {"thread_id": thread_id}}
    graph = build_research_graph(db, checkpointer=resources.saver)
    try:
        snapshot = graph.get_state(config)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to restore the saved research checkpoint",
        ) from error
    if snapshot.values.get("thread_id") != thread_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Saved checkpoint identity does not match the research run",
        )

    result = dict(snapshot.values)
    if snapshot.interrupts:
        result["interrupts"] = [item.value for item in snapshot.interrupts]
        result["status"] = (
            "awaiting_review" if run.resume_token is None else run.status
        )
    else:
        result["status"] = (
            "flagged" if run.status == "cancelled" else run.status
        )
    result["error"] = run.error or result.get("error")
    return GraphResearchResponse(**result, research_run_id=run.id)


def invoke_research_run(
    db: Session,
    run: ResearchRun,
    initial_state: dict,
    checkpointer: BaseCheckpointSaver,
) -> dict:
    thread_id = get_persisted_thread_id(db, run.id)
    config = {"configurable": {"thread_id": thread_id}}
    graph = build_research_graph(db, checkpointer=checkpointer)
    state = {**initial_state, "thread_id": thread_id}
    return dict(graph.invoke(state, config=config))


def get_persisted_thread_id(db: Session, research_run_id: str) -> str:
    persisted_run = db.get(ResearchRun, research_run_id)
    if persisted_run is None:
        raise ValueError("Persisted research run no longer exists")
    thread_id = persisted_run.thread_id
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("Research run has no valid persisted thread_id")
    try:
        UUID(thread_id)
    except ValueError as error:
        raise ValueError("Research run has no valid persisted thread_id") from error
    return thread_id