from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db.session import get_db
from ..schemas.research import ResearchQueryRequest, ResearchQueryResponse
from ..services.research_service import answer_research_query
from ..agents.workflow import build_research_graph


router = APIRouter(prefix="/research", tags=["research"])


@router.post("/query", response_model=ResearchQueryResponse)
def query_research(
    request: ResearchQueryRequest,
    db: Session = Depends(get_db),
) -> ResearchQueryResponse:
    return answer_research_query(db, request)


class GraphResearchRequest(BaseModel):
    paper_ids: list[str] = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)
    thread_id: str | None = None
    user_decision: Literal["continue", "flag"] | None = None


@router.post("/graph/query")
def query_research_graph(
    request: GraphResearchRequest,
    db: Session = Depends(get_db),
) -> dict:
    thread_id = request.thread_id or str(uuid4())
    initial_state = {
        "question": request.question,
        "thread_id": thread_id,
        "paper_ids": request.paper_ids,
        "top_k": request.top_k,
        "user_decision": request.user_decision,
        "trace": [],
        "status": "pending",
        "error": None,
    }
    graph = build_research_graph(db)
    result = graph.invoke(initial_state)
    return dict(result)