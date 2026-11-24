from datetime import datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .research import CitationResponse, EvidenceResponse


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ConversationCreateRequest(BaseModel):
    paper_id: str = Field(min_length=1, max_length=36)
    title: str | None = Field(default=None, max_length=500)

    @field_validator("paper_id")
    @classmethod
    def strip_paper_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("paper_id cannot be blank")
        return value


class MessageCreateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question cannot be blank")
        return value


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class MessageResponse(BaseModel):
    """A single conversation turn — base schema used for user messages."""

    id: str
    conversation_id: str
    role: str
    content: str
    message_index: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AssistantMessageDetail(MessageResponse):
    """
    Assistant message with the full persisted research snapshot.

    All snapshot fields (answerability, answer, abstention_reason, citations,
    evidence) are read from the stored response_snapshot JSON; no retrieval is
    re-run when this schema is constructed.
    """

    answerable: bool
    answer: str | None
    abstention_reason: str | None
    citations: list[CitationResponse]
    evidence: list[EvidenceResponse]


# Backwards-compatible alias used by existing add_message tests.
AssistantMessageResponse = AssistantMessageDetail


class ConversationSummaryResponse(BaseModel):
    """Lightweight conversation record for the chat-history sidebar."""

    id: str
    paper_id: str | None
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationDetailResponse(BaseModel):
    """
    Full conversation with ordered messages.

    Each assistant message is serialised as AssistantMessageDetail and carries
    the full persisted research snapshot (answerability, answer,
    abstention_reason, citations, evidence) without re-running retrieval.
    User messages carry only the base MessageResponse fields.

    The list is typed as list[MessageResponse] because AssistantMessageDetail
    is a strict subclass; FastAPI serialises each item according to its actual
    runtime type.
    """

    id: str
    paper_id: str | None
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse]

    model_config = ConfigDict(from_attributes=True)


class AddMessageResponse(BaseModel):
    """Response returned after posting a new question to a conversation."""

    conversation_id: str
    user_message: MessageResponse
    assistant_message: AssistantMessageDetail
