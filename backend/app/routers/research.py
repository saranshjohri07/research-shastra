from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db.session import get_db
from ..schemas.research import ResearchQueryRequest, ResearchQueryResponse
from ..services.research_service import answer_research_query


router = APIRouter(prefix="/research", tags=["research"])


@router.post("/query", response_model=ResearchQueryResponse)
def query_research(
    request: ResearchQueryRequest,
    db: Session = Depends(get_db),
) -> ResearchQueryResponse:
    return answer_research_query(db, request)