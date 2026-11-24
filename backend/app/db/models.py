from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .types import UuidCompat as Uuid

from .base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def json_type():
    return JSON().with_variant(JSONB, "postgresql")


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    papers: Mapped[list["Paper"]] = relationship(
        back_populates="collection",
        cascade="all, delete-orphan",
    )


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    collection_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("collections.id", ondelete="SET NULL"),
        nullable=True,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    storage_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    mime_type: Mapped[str] = mapped_column(String(100), default="application/pdf", nullable=False)
    index_status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    index_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    embedding_dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    collection: Mapped["Collection | None"] = relationship(
        back_populates="papers"
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="paper",
        cascade="all, delete-orphan",
    )
    conversations: Mapped[list["ConversationPaper"]] = relationship(
        back_populates="paper",
        cascade="all",
        passive_deletes=True,
    )
    research_run_papers: Mapped[list["ResearchRunPaper"]] = relationship(
        back_populates="paper",
        passive_deletes=True,
    )
    indexing_runs: Mapped[list["IndexingRun"]] = relationship(
        back_populates="paper",
        cascade="all, delete-orphan",
    )
    __table_args__ = (
        CheckConstraint("page_count IS NULL OR page_count >= 0", name="ck_papers_page_count"),
        CheckConstraint(
            "index_status IN ('pending','indexing','indexed','failed')",
            name="ck_papers_index_status",
        ),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    paper_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    section: Mapped[str | None] = mapped_column(String(500), nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    embedding_dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)
    vector_status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    vector_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    paper: Mapped["Paper"] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("paper_id", "chunk_index"),
        CheckConstraint("chunk_index >= 0", name="ck_chunks_chunk_index"),
        CheckConstraint(
            "page_start IS NULL OR page_start >= 1",
            name="ck_chunks_page_start",
        ),
        CheckConstraint(
            "page_end IS NULL OR page_start IS NULL OR page_end >= page_start",
            name="ck_chunks_page_end",
        ),
        CheckConstraint(
            "vector_status IN ('pending','indexed','failed','stale')",
            name="ck_chunks_vector_status",
        ),
    )


class ResearchRun(Base):
    __tablename__ = "research_runs"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    answerable: Mapped[bool | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), default="pending", nullable=False
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    abstention_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    top_k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    thread_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    checkpoint_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    resume_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    conversation_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
    )
    message_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    trace_events: Mapped[list["TraceEvent"]] = relationship(
        back_populates="research_run",
        cascade="all, delete-orphan",
    )
    citations: Mapped[list["Citation"]] = relationship(
        back_populates="research_run",
        cascade="all, delete-orphan",
    )
    feedback: Mapped[list["Feedback"]] = relationship(
        back_populates="research_run",
        cascade="all, delete-orphan",
    )
    evidence_records: Mapped[list["EvidenceRecord"]] = relationship(
        back_populates="research_run",
        cascade="all, delete-orphan",
    )
    paper_scopes: Mapped[list["ResearchRunPaper"]] = relationship(
        back_populates="research_run",
        cascade="all, delete-orphan",
    )
    claim_links: Mapped[list["ClaimEvidenceLink"]] = relationship(
        back_populates="research_run",
        cascade="all, delete-orphan",
    )
    checkpoint_decisions: Mapped[list["HumanCheckpointDecision"]] = relationship(
        back_populates="research_run",
        cascade="all, delete-orphan",
    )
    conversation: Mapped["Conversation | None"] = relationship(
        "Conversation",
        foreign_keys=[conversation_id],
        back_populates="research_runs",
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','running','completed','abstained','failed','cancelled')",
            name="ck_research_runs_status",
        ),
        Index("ix_research_runs_conversation_id", "conversation_id"),
        Index("ix_research_runs_message_id", "message_id"),
        Index("ix_research_runs_thread_id", "thread_id"),
        Index("ix_research_runs_created_at", "created_at"),
    )


class TraceEvent(Base):
    __tablename__ = "trace_events"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    research_run_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    stage: Mapped[str] = mapped_column(String(100), nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    thread_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    sequence: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[dict | None] = mapped_column(json_type(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    research_run: Mapped["ResearchRun"] = relationship(
        back_populates="trace_events"
    )
    __table_args__ = (
        UniqueConstraint("research_run_id", "sequence", name="uq_trace_events_run_sequence"),
        Index("ix_trace_events_research_run_id", "research_run_id"),
    )


class Citation(Base):
    __tablename__ = "citations"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    research_run_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("chunks.id", ondelete="RESTRICT"),
        nullable=False,
    )
    evidence_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("evidence_records.id", ondelete="CASCADE"),
        nullable=True,
    )
    citation_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_snapshot: Mapped[dict | None] = mapped_column(json_type(), nullable=True)
    citation_index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    research_run: Mapped["ResearchRun"] = relationship(
        back_populates="citations"
    )
    evidence_record: Mapped["EvidenceRecord | None"] = relationship(
        "EvidenceRecord", foreign_keys=[evidence_id]
    )
    __table_args__ = (
        UniqueConstraint("research_run_id", "citation_index", name="uq_citations_run_index"),
    )


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    research_run_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    research_run: Mapped["ResearchRun"] = relationship(
        back_populates="feedback"
    )
    __table_args__ = (
        CheckConstraint("rating IS NULL OR rating BETWEEN 1 AND 5", name="ck_feedback_rating"),
    )


class Note(Base):
    __tablename__ = "notes"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    paper_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    conversation_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=True,
    )
    research_run_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "paper_id IS NOT NULL OR conversation_id IS NOT NULL OR research_run_id IS NOT NULL",
            name="ck_notes_context",
        ),
    )


class Conversation(Base):
    """A persistent research conversation with one or more selected papers."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    paper_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("papers.id", ondelete="SET NULL"),
        nullable=True,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    paper: Mapped["Paper | None"] = relationship("Paper")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.message_index",
    )
    selected_papers: Mapped[list["ConversationPaper"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
    )
    research_runs: Mapped[list["ResearchRun"]] = relationship(
        "ResearchRun",
        foreign_keys="[ResearchRun.conversation_id]",
        back_populates="conversation",
    )

    __table_args__ = (
        Index("ix_conversations_paper_id", "paper_id"),
        Index("ix_conversations_updated_at", "updated_at"),
    )


class Message(Base):
    """A single turn in a Conversation — either a user question or assistant answer."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    conversation_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    # 'user' or 'assistant'
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Deterministic ordering within a conversation (0-based, incremented per pair)
    message_index: Mapped[int] = mapped_column(Integer, nullable=False)
    # JSON snapshot of the full ResearchQueryResponse for assistant messages;
    # null for user messages.
    response_snapshot: Mapped[dict | str | None] = mapped_column(json_type(), nullable=True)
    research_run_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("research_runs.id", ondelete="SET NULL", use_alter=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
    research_run: Mapped["ResearchRun | None"] = relationship(
        "ResearchRun",
        foreign_keys=[research_run_id],
    )

    __table_args__ = (
        Index("ix_messages_conversation_id", "conversation_id"),
        UniqueConstraint("conversation_id", "message_index", name="uq_messages_conversation_index"),
    )


class ConversationPaper(Base):
    __tablename__ = "conversation_papers"

    conversation_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    paper_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("papers.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="selected_papers"
    )
    paper: Mapped["Paper"] = relationship("Paper", back_populates="conversations")

    __table_args__ = (
        Index("ix_conversation_papers_paper_id", "paper_id"),
    )


class ResearchRunPaper(Base):
    __tablename__ = "research_run_papers"

    research_run_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    paper_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("papers.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    research_run: Mapped["ResearchRun"] = relationship(
        "ResearchRun", back_populates="paper_scopes"
    )
    paper: Mapped["Paper"] = relationship(
        "Paper", back_populates="research_run_papers"
    )

    __table_args__ = (
        Index("ix_research_run_papers_paper_id", "paper_id"),
    )


class EvidenceRecord(Base):
    __tablename__ = "evidence_records"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    research_run_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("chunks.id", ondelete="RESTRICT"),
        nullable=False,
    )
    paper_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("papers.id", ondelete="RESTRICT"),
        nullable=False,
    )
    evidence_id: Mapped[str] = mapped_column(String(20), nullable=False)
    evidence_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    similarity: Mapped[float] = mapped_column(Float, nullable=False)
    retrieval_method: Mapped[str] = mapped_column(String(50), default="vector", nullable=False)
    metadata_snapshot: Mapped[dict] = mapped_column(json_type(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    research_run: Mapped["ResearchRun"] = relationship(
        "ResearchRun", back_populates="evidence_records"
    )

    __table_args__ = (
        UniqueConstraint("research_run_id", "evidence_id", name="uq_evidence_run_evidence_id"),
        UniqueConstraint("research_run_id", "evidence_rank", name="uq_evidence_run_rank"),
        UniqueConstraint("research_run_id", "chunk_id", name="uq_evidence_run_chunk"),
        Index("ix_evidence_records_chunk_id", "chunk_id"),
        Index("ix_evidence_records_paper_id", "paper_id"),
    )


class ClaimEvidenceLink(Base):
    __tablename__ = "claim_evidence_links"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    research_run_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    evidence_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("evidence_records.id", ondelete="CASCADE"),
        nullable=False,
    )
    citation_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("citations.id", ondelete="CASCADE"),
        nullable=True,
    )
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    claim_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    claim_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mapping_status: Mapped[str] = mapped_column(String(20), default="unverified", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    research_run: Mapped["ResearchRun"] = relationship(
        "ResearchRun", back_populates="claim_links"
    )

    __table_args__ = (
        CheckConstraint(
            "mapping_status IN ('validated','unverified','rejected')",
            name="ck_claim_evidence_mapping_status",
        ),
    )


class HumanCheckpointDecision(Base):
    __tablename__ = "human_checkpoint_decisions"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    research_run_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    trace_event_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("trace_events.id", ondelete="SET NULL"),
        nullable=True,
    )
    thread_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    checkpoint_ns: Mapped[str | None] = mapped_column(Text, nullable=True)
    checkpoint_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    decision: Mapped[str] = mapped_column(String(30), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer: Mapped[str | None] = mapped_column(String(255), nullable=True)
    payload: Mapped[dict | None] = mapped_column(json_type(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    research_run: Mapped["ResearchRun"] = relationship(
        "ResearchRun", back_populates="checkpoint_decisions"
    )

    __table_args__ = (
        CheckConstraint(
            "decision IN ('approved','rejected','edited','abstained','resume_requested')",
            name="ck_human_checkpoint_decision",
        ),
        Index("ix_human_checkpoint_run_created", "research_run_id", "created_at"),
    )


class IndexingRun(Base):
    __tablename__ = "indexing_runs"

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    paper_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False),
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
    )
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    chunker_version: Mapped[str] = mapped_column(String(50), default="chunker-v1", nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    paper: Mapped["Paper"] = relationship("Paper", back_populates="indexing_runs")

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','running','completed','failed')",
            name="ck_indexing_runs_status",
        ),
        Index("ix_indexing_runs_paper_started", "paper_id", "started_at"),
    )
