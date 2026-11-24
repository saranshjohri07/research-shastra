from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db.session import get_db
from ..schemas.conversations import (
    AddMessageResponse,
    AssistantMessageDetail,
    ConversationCreateRequest,
    ConversationDetailResponse,
    ConversationSummaryResponse,
    MessageCreateRequest,
    MessageResponse,
)
from ..services.conversation_service import (
    add_message,
    create_conversation,
    get_conversation,
    list_conversations,
)


router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post(
    "",
    response_model=ConversationSummaryResponse,
    status_code=201,
    summary="Create a conversation for a paper",
)
def create_conversation_endpoint(
    request: ConversationCreateRequest,
    db: Session = Depends(get_db),
) -> ConversationSummaryResponse:
    return create_conversation(db, request)


@router.get(
    "",
    response_model=list[ConversationSummaryResponse],
    summary="List conversations ordered by most recent activity",
)
def list_conversations_endpoint(
    db: Session = Depends(get_db),
) -> list[ConversationSummaryResponse]:
    return list_conversations(db)


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
    response_model_exclude_none=False,
    summary="Get a conversation with ordered messages and full assistant snapshots",
    description=(
        "Returns the conversation and all its messages in chronological order. "
        "Assistant messages include the full persisted research result — "
        "answerability, answer, abstention reason, citations and evidence — "
        "without re-running retrieval."
    ),
)
def get_conversation_endpoint(
    conversation_id: str,
    db: Session = Depends(get_db),
) -> ConversationDetailResponse:
    return get_conversation(db, conversation_id)


@router.post(
    "/{conversation_id}/messages",
    response_model=AddMessageResponse,
    status_code=201,
    summary="Post a question to a conversation",
)
def add_message_endpoint(
    conversation_id: str,
    request: MessageCreateRequest,
    db: Session = Depends(get_db),
) -> AddMessageResponse:
    return add_message(db, conversation_id, request)
