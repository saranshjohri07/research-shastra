"""
Conversation service — create/list/retrieve conversations and persist messages.

Design notes
------------
- Conversation.paper_id uses SET NULL on paper deletion so history is preserved,
  but this service blocks new research messages when the paper is gone.
- Message.response_snapshot stores a JSON-serialised ResearchQueryResponse so
  reloading a conversation never re-runs retrieval.
- message_index is 0-based and incremented per message within a conversation
  (user + assistant each get their own row) ensuring stable chronological order.
  A UniqueConstraint(conversation_id, message_index) in the DB enforces this.
- Conversation.updated_at is bumped explicitly in add_message; the model column
  carries no onupdate hook so the service is the sole writer of that field.
"""

import json
from datetime import UTC, datetime

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import (
    Conversation,
    ConversationPaper,
    Message,
    Paper,
    ResearchRun,
    ResearchRunPaper,
)
from ..schemas.conversations import (
    AddMessageResponse,
    AssistantMessageDetail,
    AssistantMessageResponse,
    ConversationCreateRequest,
    ConversationDetailResponse,
    ConversationSummaryResponse,
    MessageCreateRequest,
    MessageResponse,
)
from ..schemas.research import (
    CitationResponse,
    EvidenceResponse,
    ResearchQueryRequest,
    ResearchQueryResponse,
)
from .research_service import answer_research_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_TITLE = "New Conversation"


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def _make_title(paper: Paper | None, requested: str | None) -> str:
    if requested:
        stripped = requested.strip()
        return stripped if stripped else _DEFAULT_TITLE
    if paper:
        return paper.title[:500]
    return _DEFAULT_TITLE


def _load_snapshot(msg: Message) -> ResearchQueryResponse:
    """
    Deserialise the JSON snapshot stored on an assistant Message.

    Raises RuntimeError if the message has no snapshot (i.e. it is a user
    message or the snapshot was not written correctly).
    """
    if msg.response_snapshot is None:
        raise RuntimeError(
            f"Message {msg.id!r} (role={msg.role!r}) has no response_snapshot; "
            "cannot reconstruct research result."
        )
    if isinstance(msg.response_snapshot, dict):
        raw = msg.response_snapshot
    elif isinstance(msg.response_snapshot, str):
        try:
            raw = json.loads(msg.response_snapshot)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Message {msg.id!r} has a corrupt response_snapshot: {exc}"
            ) from exc
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Message {msg.id!r} has a corrupt response_snapshot: {exc}"
                ) from exc
    else:
        raise RuntimeError(
            f"Message {msg.id!r} has a corrupt response_snapshot"
        )
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"Message {msg.id!r} has a corrupt response_snapshot"
        )

    return ResearchQueryResponse(
        question=raw["question"],
        answerable=raw["answerable"],
        answer=raw.get("answer"),
        abstention_reason=raw.get("abstention_reason"),
        citations=[CitationResponse(**c) for c in raw.get("citations", [])],
        evidence=[EvidenceResponse(**e) for e in raw.get("evidence", [])],
    )


def _assistant_detail_from_snapshot(
    msg: Message,
    snapshot: ResearchQueryResponse,
) -> AssistantMessageDetail:
    """Build a full AssistantMessageDetail from a DB row + its parsed snapshot."""
    return AssistantMessageDetail(
        id=msg.id,
        conversation_id=msg.conversation_id,
        role=msg.role,
        content=msg.content,
        message_index=msg.message_index,
        created_at=msg.created_at,
        answerable=snapshot.answerable,
        answer=snapshot.answer,
        abstention_reason=snapshot.abstention_reason,
        citations=snapshot.citations,
        evidence=snapshot.evidence,
    )


def _enrich_messages(
    messages_orm: list[Message],
) -> list[MessageResponse]:
    """
    Return each message as its richest schema type:
    - assistant messages → AssistantMessageDetail (includes snapshot fields)
    - user messages      → MessageResponse
    """
    result: list[MessageResponse] = []
    for msg in messages_orm:
        if msg.role == "assistant":
            snapshot = _load_snapshot(msg)
            result.append(_assistant_detail_from_snapshot(msg, snapshot))
        else:
            result.append(MessageResponse.model_validate(msg))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_conversation(
    db: Session,
    request: ConversationCreateRequest,
) -> ConversationSummaryResponse:
    """Create a new conversation for a valid paper."""
    paper = db.get(Paper, request.paper_id)
    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paper not found",
        )

    now = _utcnow()
    conv = Conversation(
        paper_id=paper.id,
        title=_make_title(paper, request.title),
        created_at=now,
        updated_at=now,
    )
    db.add(conv)
    db.flush()
    db.add(
        ConversationPaper(
            conversation_id=conv.id,
            paper_id=paper.id,
            position=0,
            created_at=now,
        )
    )
    db.commit()
    db.refresh(conv)

    return ConversationSummaryResponse(
        id=conv.id,
        paper_id=conv.paper_id,
        title=conv.title,
        message_count=0,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


def list_conversations(db: Session) -> list[ConversationSummaryResponse]:
    """List all conversations ordered by most recent activity first."""
    rows = db.execute(
        select(
            Conversation,
            func.count(Message.id).label("message_count"),
        )
        .outerjoin(Message, Message.conversation_id == Conversation.id)
        .group_by(Conversation.id)
        .order_by(Conversation.updated_at.desc())
    ).all()

    return [
        ConversationSummaryResponse(
            id=conv.id,
            paper_id=conv.paper_id,
            title=conv.title,
            message_count=count,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
        )
        for conv, count in rows
    ]


def get_conversation(
    db: Session,
    conversation_id: str,
) -> ConversationDetailResponse:
    """
    Return a conversation with its messages in chronological order.

    Assistant messages are enriched from their stored response_snapshot so the
    caller receives full answerability, answer, abstention_reason, citations and
    evidence without re-running retrieval.
    """
    conv = db.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    messages_orm = db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.message_index)
    ).all()

    return ConversationDetailResponse(
        id=conv.id,
        paper_id=conv.paper_id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=_enrich_messages(list(messages_orm)),
    )


def add_message(
    db: Session,
    conversation_id: str,
    request: MessageCreateRequest,
) -> AddMessageResponse:
    """
    Submit a user question to a conversation, run the RAG pipeline, and
    persist both the user message and the assistant response.
    """
    conv = db.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    # Block new questions when the paper has been deleted.
    if conv.paper_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The paper associated with this conversation has been deleted; no new questions can be asked.",
        )
    paper = db.get(Paper, conv.paper_id)
    if paper is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The paper associated with this conversation is no longer available.",
        )

    # Determine the next message_index for this conversation.
    current_max = db.scalar(
        select(func.max(Message.message_index)).where(
            Message.conversation_id == conversation_id
        )
    )
    next_index = 0 if current_max is None else current_max + 1

    # Run the RAG pipeline.  Errors propagate — no silent swallowing.
    research_response: ResearchQueryResponse = answer_research_query(
        db,
        ResearchQueryRequest(
            paper_id=conv.paper_id,
            question=request.question,
            top_k=request.top_k,
        ),
    )

    # Persist user message and assistant message in one transaction.
    now = _utcnow()
    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=request.question,
        message_index=next_index,
        response_snapshot=None,
        created_at=now,
    )
    db.add(user_msg)
    db.flush()

    research_run = ResearchRun(
        question=request.question,
        answer=research_response.answer,
        answerable=research_response.answerable,
        abstention_reason=research_response.abstention_reason,
        status="abstained" if not research_response.answerable else "completed",
        conversation_id=conversation_id,
        message_id=user_msg.id,
        top_k=request.top_k,
        created_at=now,
        completed_at=now,
    )
    db.add(research_run)
    db.flush()
    db.add(
        ResearchRunPaper(
            research_run_id=research_run.id,
            paper_id=conv.paper_id,
            position=0,
            created_at=now,
        )
    )

    assistant_content = research_response.answer or research_response.abstention_reason or ""
    assistant_msg = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=assistant_content,
        message_index=next_index + 1,
        response_snapshot=research_response.model_dump(mode="json"),
        research_run_id=research_run.id,
        created_at=now,
    )
    db.add(assistant_msg)

    # Bump conversation.updated_at so the sidebar reflects latest activity.
    # The model column has no onupdate hook; this service is the sole writer.
    conv.updated_at = now
    db.add(conv)

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(user_msg)
    db.refresh(assistant_msg)

    snapshot = _load_snapshot(assistant_msg)

    return AddMessageResponse(
        conversation_id=conversation_id,
        user_message=MessageResponse.model_validate(user_msg),
        assistant_message=_assistant_detail_from_snapshot(assistant_msg, snapshot),
    )
