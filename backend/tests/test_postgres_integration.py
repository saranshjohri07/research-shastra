from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.db.models import (
    Chunk,
    Citation,
    EvidenceRecord,
    Conversation,
    ConversationPaper,
    Message,
    Paper,
    ResearchRun,
    ResearchRunPaper,
)
from backend.app.db.session import engine


pytestmark = pytest.mark.skipif(
    engine.dialect.name != "postgresql",
    reason="PostgreSQL integration tests require DATABASE_URL=postgresql...",
)


@pytest.fixture
def db():
    connection = engine.connect()
    transaction = connection.begin()
    with Session(bind=connection) as session:
        yield session
    if transaction.is_active:
        transaction.rollback()
    connection.close()


def make_paper(db: Session, title: str) -> Paper:
    paper = Paper(
        id=str(uuid4()),
        title=title,
        filename=f"{title}.pdf",
        file_path=f"{title}.pdf",
        page_count=1,
    )
    db.add(paper)
    db.flush()
    return paper


def test_conversation_and_multiple_papers(db: Session):
    paper_a = make_paper(db, "Audit A")
    paper_b = make_paper(db, "Audit B")
    conversation = Conversation(
        id=str(uuid4()),
        paper_id=paper_a.id,
        title="Audit conversation",
    )
    db.add(conversation)
    db.flush()
    db.add_all(
        [
            ConversationPaper(
                conversation_id=conversation.id,
                paper_id=paper_a.id,
                position=0,
            ),
            ConversationPaper(
                conversation_id=conversation.id,
                paper_id=paper_b.id,
                position=1,
            ),
        ]
    )
    db.flush()

    conversation_id = conversation.id
    db.expunge(conversation)
    loaded = db.get(Conversation, conversation_id)
    assert loaded is not None
    assert len(db.scalars(
        select(ConversationPaper.paper_id).where(
            ConversationPaper.conversation_id == conversation_id
        )
    ).all()) == 2


def test_research_run_message_insertion_order_and_scope(db: Session):
    paper = make_paper(db, "Scope")
    conversation = Conversation(
        id=str(uuid4()),
        paper_id=paper.id,
        title="Scope conversation",
    )
    db.add(conversation)
    db.flush()

    user_message = Message(
        id=str(uuid4()),
        conversation_id=conversation.id,
        role="user",
        content="Question",
        message_index=0,
    )
    db.add(user_message)
    db.flush()

    research_run = ResearchRun(
        id=str(uuid4()),
        conversation_id=conversation.id,
        message_id=user_message.id,
        question="Question",
        status="completed",
    )
    db.add(research_run)
    db.flush()

    db.add(
        ResearchRunPaper(
            research_run_id=research_run.id,
            paper_id=paper.id,
            position=0,
        )
    )
    assistant_message = Message(
        id=str(uuid4()),
        conversation_id=conversation.id,
        role="assistant",
        content="Answer",
        message_index=1,
        research_run_id=research_run.id,
        response_snapshot={"answerable": True, "answer": "Answer"},
    )
    db.add(assistant_message)
    db.flush()

    assert db.get(ResearchRun, research_run.id).message_id == user_message.id
    assert db.get(Message, assistant_message.id).research_run_id == research_run.id
    assert db.scalar(
        select(ResearchRunPaper.paper_id).where(
            ResearchRunPaper.research_run_id == research_run.id
        )
    ) == paper.id


def test_invalid_originating_message_fk_and_delete_set_null(db: Session):
    conversation = Conversation(id=str(uuid4()), title="FK conversation")
    db.add(conversation)

    invalid_run = ResearchRun(
        id=str(uuid4()),
        conversation_id=conversation.id,
        message_id=str(uuid4()),
        question="Invalid",
        status="pending",
    )
    db.add(invalid_run)
    with pytest.raises(IntegrityError):
        db.flush()
    db.rollback()

    paper = make_paper(db, "FK paper")
    conversation = Conversation(id=str(uuid4()), paper_id=paper.id, title="FK valid")
    db.add(conversation)
    db.flush()
    user_message = Message(
        id=str(uuid4()),
        conversation_id=conversation.id,
        role="user",
        content="Question",
        message_index=0,
    )
    db.add(user_message)
    db.flush()
    research_run = ResearchRun(
        id=str(uuid4()),
        conversation_id=conversation.id,
        message_id=user_message.id,
        question="Question",
        status="completed",
    )
    db.add(research_run)
    db.flush()

    db.delete(user_message)
    db.flush()
    db.expire_all()
    run_row = db.get(ResearchRun, research_run.id)
    assert run_row is not None
    assert run_row.message_id is None


def test_duplicate_association_and_restrictive_paper_delete(db: Session):
    paper = make_paper(db, "Unique")
    conversation = Conversation(
        id=str(uuid4()),
        paper_id=paper.id,
        title="Unique",
    )
    db.add(conversation)
    db.flush()
    conversation_id = conversation.id
    paper_id = paper.id
    db.add(
        ConversationPaper(
            conversation_id=conversation_id,
            paper_id=paper_id,
            position=0,
        )
    )
    db.flush()

    db.add(
        ConversationPaper(
            conversation_id=conversation.id,
            paper_id=paper.id,
            position=1,
        )
    )
    with pytest.raises(IntegrityError):
        db.flush()
    db.rollback()

    paper = make_paper(db, "Unique restricted")
    conversation = Conversation(
        id=str(uuid4()),
        paper_id=paper.id,
        title="Unique restricted",
    )
    db.add(conversation)
    db.flush()
    conversation_id = conversation.id
    paper_id = paper.id
    run = ResearchRun(
        id=str(uuid4()),
        question="Restricted",
        status="completed",
        conversation_id=conversation_id,
    )
    db.add(run)
    db.flush()
    db.add(
        ResearchRunPaper(
            research_run_id=run.id,
            paper_id=paper_id,
            position=0,
        )
    )
    db.flush()

    paper = db.get(Paper, paper_id)
    db.delete(paper)
    with pytest.raises(IntegrityError):
        db.flush()
    db.rollback()


def test_evidence_citation_and_transaction_relationships(db: Session):
    paper = make_paper(db, "Evidence")
    chunk = Chunk(
        id=str(uuid4()),
        paper_id=paper.id,
        chunk_index=0,
        text="Evidence text",
        page_start=1,
        page_end=1,
        vector_status="indexed",
        vector_id=None,
    )
    db.add(chunk)
    db.flush()
    run = ResearchRun(
        id=str(uuid4()),
        question="Evidence question",
        status="completed",
    )
    db.add(run)
    db.flush()
    evidence = EvidenceRecord(
        id=str(uuid4()),
        research_run_id=run.id,
        chunk_id=chunk.id,
        paper_id=paper.id,
        evidence_id="E1",
        evidence_rank=1,
        similarity=0.9,
        retrieval_method="vector",
        metadata_snapshot={"section": "Audit"},
    )
    db.add(evidence)
    db.flush()
    citation = Citation(
        id=str(uuid4()),
        research_run_id=run.id,
        chunk_id=chunk.id,
        evidence_id=evidence.id,
        citation_index=1,
    )
    db.add(citation)
    db.flush()

    assert db.get(EvidenceRecord, evidence.id).chunk_id == chunk.id
    assert db.get(Citation, citation.id).evidence_id == evidence.id
