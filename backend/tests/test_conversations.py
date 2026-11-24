"""
Focused tests for the persistent conversations feature.

Coverage
--------
- Conversation creation and listing
- Retrieving a conversation with ordered messages
- Persisting a user question and assistant response
- Restoring the original evidence/citation snapshot without calling retrieval
- GET /conversations/{id} returns AssistantMessageDetail with full snapshot fields
- Independent conversation histories (no cross-leakage)
- Unknown conversation and paper IDs → 404
- Invalid question and top_k input → 422 / ValidationError
- Failed generation and database rollback
- UniqueConstraint on (conversation_id, message_index) enforced at DB level
- Conversation cascade-deletes its messages
- Blank / whitespace-only title falls back to paper title / default
- Corrupt response_snapshot raises RuntimeError (not bare AssertionError)
- Existing /research/query endpoint remains unchanged
"""

import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.db.base import Base
from backend.app.db.models import Conversation, Message, Paper
from backend.app.schemas.conversations import (
    AssistantMessageDetail,
    ConversationCreateRequest,
    MessageCreateRequest,
    MessageResponse,
)
from backend.app.schemas.research import (
    CitationResponse,
    EvidenceResponse,
    ResearchQueryRequest,
    ResearchQueryResponse,
)
from backend.app.services import conversation_service, research_service
from backend.app.services.conversation_service import (
    _load_snapshot,
    add_message,
    create_conversation,
    get_conversation,
    list_conversations,
)
from backend.app.services.research_service import answer_research_query


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db():
    """Isolated in-memory SQLite database, schema created fresh per test."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture()
def paper(db: Session) -> Paper:
    p = Paper(
        id="paper-1",
        title="Attention Is All You Need",
        filename="attention.pdf",
        file_path="attention.pdf",
        page_count=15,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@pytest.fixture()
def paper_b(db: Session) -> Paper:
    p = Paper(
        id="paper-2",
        title="BERT: Pre-training Deep Bidirectional Transformers",
        filename="bert.pdf",
        file_path="bert.pdf",
        page_count=16,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


def _fake_research_response(question: str = "What is attention?") -> ResearchQueryResponse:
    return ResearchQueryResponse(
        question=question,
        answerable=True,
        answer="Attention is a mechanism that allows the model to focus on relevant parts of the input.",
        abstention_reason=None,
        citations=[],
        evidence=[],
    )


def _fake_abstention_response(question: str = "Who wrote this?") -> ResearchQueryResponse:
    return ResearchQueryResponse(
        question=question,
        answerable=False,
        answer=None,
        abstention_reason="The supplied paper does not contain information needed to answer this question.",
        citations=[],
        evidence=[],
    )


def _patch_rag(monkeypatch, response: ResearchQueryResponse | None = None):
    """Convenience: monkeypatch answer_research_query to return a fake response."""
    resp = response or _fake_research_response()
    monkeypatch.setattr(
        conversation_service,
        "answer_research_query",
        lambda db, req: resp,
    )


# ---------------------------------------------------------------------------
# 1. Conversation creation
# ---------------------------------------------------------------------------


def test_create_conversation_returns_summary(db, paper):
    result = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="My chat")
    )
    assert result.id
    assert result.paper_id == paper.id
    assert result.title == "My chat"
    assert result.message_count == 0
    assert result.created_at is not None
    assert result.updated_at is not None


def test_create_conversation_uses_paper_title_as_default(db, paper):
    result = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id)
    )
    assert result.title == paper.title


def test_create_conversation_unknown_paper_raises_404(db):
    with pytest.raises(HTTPException) as exc:
        create_conversation(
            db, ConversationCreateRequest(paper_id="nonexistent")
        )
    assert exc.value.status_code == 404


def test_create_conversation_persists_to_db(db, paper):
    result = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="Persisted")
    )
    row = db.get(Conversation, result.id)
    assert row is not None
    assert row.paper_id == paper.id
    assert row.title == "Persisted"


def test_create_conversation_whitespace_title_falls_back_to_default(db, paper):
    """A title that is all whitespace must not be stored as blank — fall back
    to the default title string, not the blank value."""
    result = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="   ")
    )
    assert result.title != ""
    assert result.title.strip() != ""
    # Service strips whitespace and falls back to _DEFAULT_TITLE constant.
    assert result.title == "New Conversation"


# ---------------------------------------------------------------------------
# 2. Listing conversations
# ---------------------------------------------------------------------------


def test_list_conversations_empty(db):
    assert list_conversations(db) == []


def test_list_conversations_returns_all(db, paper, paper_b):
    create_conversation(db, ConversationCreateRequest(paper_id=paper.id, title="C1"))
    create_conversation(db, ConversationCreateRequest(paper_id=paper_b.id, title="C2"))
    results = list_conversations(db)
    assert len(results) == 2


def test_list_conversations_ordered_by_updated_at_desc(db, paper):
    """The conversation with the more recent updated_at must appear first."""
    c1 = create_conversation(db, ConversationCreateRequest(paper_id=paper.id, title="Older"))
    c2 = create_conversation(db, ConversationCreateRequest(paper_id=paper.id, title="Newer"))

    # Wind back updated_at on c1 using timezone-aware UTC.
    row1 = db.get(Conversation, c1.id)
    row1.updated_at = datetime.now(UTC) - timedelta(hours=1)
    db.commit()

    results = list_conversations(db)
    assert results[0].id == c2.id
    assert results[1].id == c1.id


def test_list_conversations_includes_message_count(db, paper, monkeypatch):
    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="C")
    )
    _patch_rag(monkeypatch)
    add_message(db, conv.id, MessageCreateRequest(question="Q1"))
    results = list_conversations(db)
    assert results[0].message_count == 2  # user + assistant


# ---------------------------------------------------------------------------
# 3. Retrieving a conversation with ordered messages
# ---------------------------------------------------------------------------


def test_get_conversation_returns_messages_in_order(db, paper, monkeypatch):
    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="Order test")
    )
    _patch_rag(monkeypatch)
    add_message(db, conv.id, MessageCreateRequest(question="First question"))
    add_message(db, conv.id, MessageCreateRequest(question="Second question"))

    detail = get_conversation(db, conv.id)

    assert len(detail.messages) == 4  # 2 user + 2 assistant
    indices = [m.message_index for m in detail.messages]
    assert indices == sorted(indices)
    assert detail.messages[0].role == "user"
    assert detail.messages[1].role == "assistant"
    assert detail.messages[2].role == "user"
    assert detail.messages[3].role == "assistant"


def test_get_conversation_unknown_id_raises_404(db):
    with pytest.raises(HTTPException) as exc:
        get_conversation(db, "does-not-exist")
    assert exc.value.status_code == 404


def test_get_conversation_does_not_include_other_conversations_messages(
    db, paper, monkeypatch
):
    _patch_rag(monkeypatch)
    c1 = create_conversation(db, ConversationCreateRequest(paper_id=paper.id, title="C1"))
    c2 = create_conversation(db, ConversationCreateRequest(paper_id=paper.id, title="C2"))
    add_message(db, c1.id, MessageCreateRequest(question="Question for C1"))
    add_message(db, c2.id, MessageCreateRequest(question="Question for C2"))

    detail_c1 = get_conversation(db, c1.id)
    assert all(m.conversation_id == c1.id for m in detail_c1.messages)
    assert len(detail_c1.messages) == 2


# ---------------------------------------------------------------------------
# 4. Persisting user question and assistant response
# ---------------------------------------------------------------------------


def test_add_message_persists_user_and_assistant(db, paper, monkeypatch):
    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="T")
    )
    _patch_rag(monkeypatch)
    result = add_message(db, conv.id, MessageCreateRequest(question="What is attention?"))

    assert result.user_message.role == "user"
    assert result.user_message.content == "What is attention?"
    assert result.assistant_message.role == "assistant"
    assert result.assistant_message.answerable is True
    assert result.assistant_message.answer is not None

    msgs = db.scalars(
        select(Message).where(Message.conversation_id == conv.id)
    ).all()
    assert len(msgs) == 2


def test_add_message_increments_message_index_across_turns(db, paper, monkeypatch):
    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="T")
    )
    _patch_rag(monkeypatch)
    add_message(db, conv.id, MessageCreateRequest(question="Q1"))
    add_message(db, conv.id, MessageCreateRequest(question="Q2"))

    msgs = db.scalars(
        select(Message)
        .where(Message.conversation_id == conv.id)
        .order_by(Message.message_index)
    ).all()
    assert [m.message_index for m in msgs] == [0, 1, 2, 3]


def test_add_message_bumps_conversation_updated_at(db, paper, monkeypatch):
    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="T")
    )
    row = db.get(Conversation, conv.id)
    # Wind back updated_at using timezone-aware UTC so we can detect the bump.
    row.updated_at = datetime.now(UTC) - timedelta(hours=1)
    db.commit()
    old_ts = row.updated_at

    _patch_rag(monkeypatch)
    add_message(db, conv.id, MessageCreateRequest(question="Does it bump?"))

    db.refresh(row)
    assert row.updated_at > old_ts


def test_add_message_unknown_conversation_raises_404(db):
    with pytest.raises(HTTPException) as exc:
        add_message(db, "nope", MessageCreateRequest(question="Q"))
    assert exc.value.status_code == 404


def test_add_message_deleted_paper_raises_422(db, paper, monkeypatch):
    """Conversation.paper_id becomes NULL when paper is deleted; reject new messages."""
    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="T")
    )
    row = db.get(Conversation, conv.id)
    row.paper_id = None
    db.commit()

    with pytest.raises(HTTPException) as exc:
        add_message(db, conv.id, MessageCreateRequest(question="Q"))
    assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# 5. Snapshot restoration — no retrieval re-run
# ---------------------------------------------------------------------------


def test_snapshot_restores_answer_without_retrieval(db, paper, monkeypatch):
    """Reloading a conversation must reconstruct the assistant response from
    the stored JSON snapshot, not by calling retrieval again."""
    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="Snapshot")
    )
    original_response = _fake_research_response()
    _patch_rag(monkeypatch, original_response)
    add_message(db, conv.id, MessageCreateRequest(question="What is attention?"))

    retrieval_called = {"count": 0}
    monkeypatch.setattr(
        conversation_service,
        "answer_research_query",
        lambda db, req: retrieval_called.update(count=retrieval_called["count"] + 1) or original_response,
    )

    detail = get_conversation(db, conv.id)
    assert retrieval_called["count"] == 0  # no retrieval on reload

    assistant_msgs = [m for m in detail.messages if m.role == "assistant"]
    assert len(assistant_msgs) == 1


def test_snapshot_stores_full_evidence_and_citations(db, paper, monkeypatch):
    """The JSON snapshot must preserve evidence and citation metadata exactly."""
    evidence_item = EvidenceResponse(
        evidence_id="E1",
        chunk_id="chunk-x",
        paper_id=paper.id,
        chunk_index=0,
        page_start=1,
        page_end=2,
        section="Introduction",
        title=paper.title,
        filename=paper.filename,
        similarity=0.9,
        text="Self-attention is...",
    )
    citation_item = CitationResponse(
        evidence_id="E1",
        chunk_id="chunk-x",
        paper_id=paper.id,
        title=paper.title,
        filename=paper.filename,
        page_start=1,
        page_end=2,
        section="Introduction",
    )
    rich_response = ResearchQueryResponse(
        question="What is self-attention?",
        answerable=True,
        answer="Self-attention is a mechanism...",
        abstention_reason=None,
        citations=[citation_item],
        evidence=[evidence_item],
    )

    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="Rich")
    )
    _patch_rag(monkeypatch, rich_response)
    result = add_message(
        db, conv.id, MessageCreateRequest(question="What is self-attention?")
    )

    assert len(result.assistant_message.evidence) == 1
    assert result.assistant_message.evidence[0].chunk_id == "chunk-x"
    assert len(result.assistant_message.citations) == 1
    assert result.assistant_message.citations[0].section == "Introduction"

    assistant_row = db.scalars(
        select(Message).where(
            Message.conversation_id == conv.id,
            Message.role == "assistant",
        )
    ).one()
    raw = (
        assistant_row.response_snapshot
        if isinstance(assistant_row.response_snapshot, dict)
        else json.loads(assistant_row.response_snapshot)
    )
    assert raw["answerable"] is True
    assert len(raw["evidence"]) == 1
    assert raw["evidence"][0]["chunk_id"] == "chunk-x"
    assert len(raw["citations"]) == 1


def test_snapshot_preserves_abstention(db, paper, monkeypatch):
    """Abstention responses must also be faithfully stored and retrieved."""
    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="Abstain")
    )
    _patch_rag(monkeypatch, _fake_abstention_response())
    result = add_message(db, conv.id, MessageCreateRequest(question="Who invented gravity?"))

    assert result.assistant_message.answerable is False
    assert result.assistant_message.answer is None
    assert result.assistant_message.abstention_reason is not None

    assistant_row = db.scalars(
        select(Message).where(
            Message.conversation_id == conv.id,
            Message.role == "assistant",
        )
    ).one()
    raw = (
        assistant_row.response_snapshot
        if isinstance(assistant_row.response_snapshot, dict)
        else json.loads(assistant_row.response_snapshot)
    )
    assert raw["answerable"] is False
    assert raw["abstention_reason"] is not None


# ---------------------------------------------------------------------------
# 6. Independent conversation histories
# ---------------------------------------------------------------------------


def test_independent_conversation_histories(db, paper, monkeypatch):
    """Adding messages to one conversation must not affect another."""
    _patch_rag(monkeypatch)
    c1 = create_conversation(db, ConversationCreateRequest(paper_id=paper.id, title="A"))
    c2 = create_conversation(db, ConversationCreateRequest(paper_id=paper.id, title="B"))

    add_message(db, c1.id, MessageCreateRequest(question="Only in C1"))

    d1 = get_conversation(db, c1.id)
    d2 = get_conversation(db, c2.id)

    assert len(d1.messages) == 2
    assert len(d2.messages) == 0


# ---------------------------------------------------------------------------
# 7. Input validation
# ---------------------------------------------------------------------------


def test_message_create_request_rejects_blank_question():
    with pytest.raises(ValidationError):
        MessageCreateRequest(question="   ")


def test_message_create_request_rejects_question_too_long():
    with pytest.raises(ValidationError):
        MessageCreateRequest(question="x" * 2001)


def test_message_create_request_rejects_top_k_zero():
    with pytest.raises(ValidationError):
        MessageCreateRequest(question="Valid?", top_k=0)


def test_message_create_request_rejects_top_k_too_large():
    with pytest.raises(ValidationError):
        MessageCreateRequest(question="Valid?", top_k=11)


def test_message_create_request_accepts_boundary_top_k():
    low = MessageCreateRequest(question="Q", top_k=1)
    high = MessageCreateRequest(question="Q", top_k=10)
    assert low.top_k == 1
    assert high.top_k == 10


def test_conversation_create_request_rejects_blank_paper_id():
    with pytest.raises(ValidationError):
        ConversationCreateRequest(paper_id="   ")


# ---------------------------------------------------------------------------
# 8. Failed generation and rollback
# ---------------------------------------------------------------------------


def test_failed_generation_does_not_persist_messages(db, paper, monkeypatch):
    """If answer_research_query raises, no messages must be persisted."""
    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="Rollback test")
    )
    monkeypatch.setattr(
        conversation_service,
        "answer_research_query",
        lambda db, req: (_ for _ in ()).throw(RuntimeError("Groq API unreachable")),
    )

    with pytest.raises(RuntimeError, match="Groq API unreachable"):
        add_message(db, conv.id, MessageCreateRequest(question="Will this persist?"))

    import sqlalchemy
    count = db.scalar(
        select(Message).where(Message.conversation_id == conv.id).with_only_columns(
            sqlalchemy.func.count()
        )
    )
    assert count == 0


def test_db_commit_failure_triggers_rollback(db, paper, monkeypatch):
    """If the DB commit itself fails, rollback must be called and the error
    must propagate — no silent success."""
    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="Commit fail")
    )
    _patch_rag(monkeypatch)

    original_commit = Session.commit
    call_count = {"n": 0}

    def failing_commit(self):
        call_count["n"] += 1
        raise RuntimeError("Simulated commit failure")

    monkeypatch.setattr(Session, "commit", failing_commit)

    with pytest.raises(RuntimeError, match="Simulated commit failure"):
        add_message(db, conv.id, MessageCreateRequest(question="Commit fails here"))

    monkeypatch.setattr(Session, "commit", original_commit)

    import sqlalchemy
    count = db.scalar(
        select(sqlalchemy.func.count(Message.id)).where(
            Message.conversation_id == conv.id
        )
    )
    assert count == 0


# ---------------------------------------------------------------------------
# 9. Existing /research/query endpoint is unchanged
# ---------------------------------------------------------------------------


def test_existing_research_query_still_works(db, paper, monkeypatch):
    from backend.app.retrieval.retriever import RetrievedEvidence

    evidence = RetrievedEvidence(
        evidence_id="E1",
        chunk_id="chunk-1",
        paper_id=paper.id,
        text="The transformer model uses attention.",
        chunk_index=0,
        page_start=1,
        page_end=1,
        section="Abstract",
        title=paper.title,
        filename=paper.filename,
        similarity=0.92,
    )
    monkeypatch.setattr(
        research_service, "retrieve_evidence", lambda **kwargs: [evidence]
    )
    monkeypatch.setattr(
        research_service,
        "generate_grounded_answer",
        lambda question, context: {
            "answerable": True,
            "answer": "The transformer uses self-attention.",
            "reason": None,
        },
    )

    response = answer_research_query(
        db,
        ResearchQueryRequest(paper_id=paper.id, question="What does the transformer use?"),
    )

    assert response.answerable is True
    assert response.answer == "The transformer uses self-attention."
    assert response.citations[0].chunk_id == "chunk-1"
    assert response.evidence[0].similarity == pytest.approx(0.92)


def test_existing_research_query_unknown_paper_still_404(db):
    with pytest.raises(HTTPException) as exc:
        answer_research_query(
            db,
            ResearchQueryRequest(paper_id="ghost", question="Anything?"),
        )
    assert exc.value.status_code == 404


# ===========================================================================
# NEW HARDENING TESTS (Task 6)
# ===========================================================================

# ---------------------------------------------------------------------------
# A. UniqueConstraint — DB enforces (conversation_id, message_index) uniqueness
# ---------------------------------------------------------------------------


def test_duplicate_message_index_in_same_conversation_raises_integrity_error(
    db, paper
):
    """Inserting two messages with the same (conversation_id, message_index)
    must raise an IntegrityError — the DB-level constraint must be active."""
    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="UC test")
    )
    msg1 = Message(
        conversation_id=conv.id,
        role="user",
        content="First",
        message_index=0,
        response_snapshot=None,
    )
    msg2 = Message(
        conversation_id=conv.id,
        role="user",
        content="Duplicate index",
        message_index=0,  # same index — must violate constraint
    )
    db.add(msg1)
    db.flush()
    db.add(msg2)
    with pytest.raises(IntegrityError):
        db.flush()


def test_same_message_index_in_different_conversations_is_allowed(db, paper):
    """Index 0 in conversation A and index 0 in conversation B are independent."""
    c1 = create_conversation(db, ConversationCreateRequest(paper_id=paper.id, title="A"))
    c2 = create_conversation(db, ConversationCreateRequest(paper_id=paper.id, title="B"))
    db.add(Message(
        conversation_id=c1.id,
        role="user",
        content="Q in C1",
        message_index=0,
        response_snapshot=None,
    ))
    db.add(Message(
        conversation_id=c2.id,
        role="user",
        content="Q in C2",
        message_index=0,
        response_snapshot=None,
    ))
    db.commit()  # must not raise

    assert db.scalar(
        select(Message).where(Message.conversation_id == c1.id).with_only_columns(
            __import__("sqlalchemy").func.count()
        )
    ) == 1
    assert db.scalar(
        select(Message).where(Message.conversation_id == c2.id).with_only_columns(
            __import__("sqlalchemy").func.count()
        )
    ) == 1


# ---------------------------------------------------------------------------
# B. Historical snapshot loading via GET /conversations/{id}
# ---------------------------------------------------------------------------


def test_get_conversation_returns_assistant_message_detail(db, paper, monkeypatch):
    """After adding a message, GET /conversations/{id} must return an
    AssistantMessageDetail for the assistant turn, not a bare MessageResponse."""
    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="Detail test")
    )
    _patch_rag(monkeypatch)
    add_message(db, conv.id, MessageCreateRequest(question="What is attention?"))

    detail = get_conversation(db, conv.id)

    assistant_msgs = [m for m in detail.messages if m.role == "assistant"]
    assert len(assistant_msgs) == 1
    am = assistant_msgs[0]
    # Must be the richer subtype
    assert isinstance(am, AssistantMessageDetail)
    assert am.answerable is True
    assert am.answer is not None
    assert am.citations == []
    assert am.evidence == []


def test_get_conversation_assistant_message_has_full_evidence(db, paper, monkeypatch):
    """Evidence and citations stored in the snapshot must be available when
    reloading the conversation via get_conversation, with no RAG call."""
    evidence_item = EvidenceResponse(
        evidence_id="E1",
        chunk_id="chunk-hist",
        paper_id=paper.id,
        chunk_index=1,
        page_start=3,
        page_end=4,
        section="Methods",
        title=paper.title,
        filename=paper.filename,
        similarity=0.85,
        text="The model applies multi-head attention.",
    )
    citation_item = CitationResponse(
        evidence_id="E1",
        chunk_id="chunk-hist",
        paper_id=paper.id,
        title=paper.title,
        filename=paper.filename,
        page_start=3,
        page_end=4,
        section="Methods",
    )
    rich_response = ResearchQueryResponse(
        question="How does the model attend?",
        answerable=True,
        answer="It uses multi-head attention.",
        abstention_reason=None,
        citations=[citation_item],
        evidence=[evidence_item],
    )

    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="Hist evidence")
    )
    _patch_rag(monkeypatch, rich_response)
    add_message(db, conv.id, MessageCreateRequest(question="How does the model attend?"))

    # Replace the RAG patch with a sentinel that must NOT be called.
    monkeypatch.setattr(
        conversation_service,
        "answer_research_query",
        lambda db, req: pytest.fail("RAG pipeline must not run on reload"),
    )

    detail = get_conversation(db, conv.id)
    am = next(m for m in detail.messages if m.role == "assistant")

    assert isinstance(am, AssistantMessageDetail)
    assert len(am.evidence) == 1
    assert am.evidence[0].chunk_id == "chunk-hist"
    assert am.evidence[0].section == "Methods"
    assert len(am.citations) == 1
    assert am.citations[0].page_start == 3


def test_get_conversation_abstention_reloads_correctly(db, paper, monkeypatch):
    """An abstention stored in the snapshot must reload with answerable=False
    and abstention_reason populated, without re-running the pipeline."""
    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="Abstain reload")
    )
    _patch_rag(monkeypatch, _fake_abstention_response("Who invented gravity?"))
    add_message(db, conv.id, MessageCreateRequest(question="Who invented gravity?"))

    monkeypatch.setattr(
        conversation_service,
        "answer_research_query",
        lambda db, req: pytest.fail("RAG pipeline must not run on reload"),
    )

    detail = get_conversation(db, conv.id)
    am = next(m for m in detail.messages if m.role == "assistant")

    assert isinstance(am, AssistantMessageDetail)
    assert am.answerable is False
    assert am.answer is None
    assert am.abstention_reason is not None


def test_user_messages_are_base_message_response(db, paper, monkeypatch):
    """User messages in the detail response must be plain MessageResponse,
    not AssistantMessageDetail."""
    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="User type test")
    )
    _patch_rag(monkeypatch)
    add_message(db, conv.id, MessageCreateRequest(question="Am I a plain message?"))

    detail = get_conversation(db, conv.id)
    user_msg = next(m for m in detail.messages if m.role == "user")

    # Must not carry snapshot fields — must be the plain base type
    assert not isinstance(user_msg, AssistantMessageDetail)
    assert user_msg.content == "Am I a plain message?"


# ---------------------------------------------------------------------------
# C. Corrupt / missing snapshot raises RuntimeError, not bare AssertionError
# ---------------------------------------------------------------------------


def test_load_snapshot_raises_runtime_error_for_user_message(db, paper):
    """_load_snapshot called on a user message (no snapshot) must raise
    RuntimeError with a descriptive message, not AssertionError."""
    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="Err test")
    )
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content="Question",
        message_index=0,
        response_snapshot=None,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    with pytest.raises(RuntimeError, match="no response_snapshot"):
        _load_snapshot(user_msg)


def test_load_snapshot_raises_runtime_error_for_corrupt_json(db, paper):
    """_load_snapshot with a malformed JSON blob must raise RuntimeError."""
    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="Corrupt")
    )
    bad_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content="Answer",
        message_index=0,
        response_snapshot="{ this is not json !!!",
    )
    db.add(bad_msg)
    db.commit()
    db.refresh(bad_msg)

    with pytest.raises(RuntimeError, match="corrupt response_snapshot"):
        _load_snapshot(bad_msg)


# ---------------------------------------------------------------------------
# D. Conversation cascade-delete removes its messages
# ---------------------------------------------------------------------------


def test_deleting_conversation_cascades_to_messages(db, paper, monkeypatch):
    """When a Conversation row is deleted, all its Message rows must be
    deleted automatically via the CASCADE foreign key."""
    conv = create_conversation(
        db, ConversationCreateRequest(paper_id=paper.id, title="Cascade test")
    )
    _patch_rag(monkeypatch)
    add_message(db, conv.id, MessageCreateRequest(question="Will I survive?"))

    conv_id = conv.id
    conv_row = db.get(Conversation, conv_id)
    assert conv_row is not None

    # Confirm messages exist before deletion
    pre_count = db.scalar(
        select(__import__("sqlalchemy").func.count(Message.id)).where(
            Message.conversation_id == conv_id
        )
    )
    assert pre_count == 2  # user + assistant

    db.delete(conv_row)
    db.commit()

    # Conversation must be gone
    assert db.get(Conversation, conv_id) is None

    # All messages must be gone via CASCADE
    post_count = db.scalar(
        select(__import__("sqlalchemy").func.count(Message.id)).where(
            Message.conversation_id == conv_id
        )
    )
    assert post_count == 0
