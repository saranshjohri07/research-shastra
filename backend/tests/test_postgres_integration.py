from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from langgraph.types import Command

from backend.app.agents import stages
from backend.app.agents.workflow import build_research_graph
from backend.app.checkpointer import build_checkpointer, close_checkpointer
from backend.app.config import get_settings
from backend.app.db.models import (
    Chunk,
    Citation,
    EvidenceRecord,
    Conversation,
    ConversationPaper,
    HumanCheckpointDecision,
    Message,
    Paper,
    ResearchRun,
    ResearchRunPaper,
)
from backend.app.db.session import engine, get_db
from backend.app.routers import research as research_router
from backend.app.retrieval.retriever import RetrievedEvidence


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


def test_postgres_graph_checkpoint_is_available_to_new_graph(
    db: Session,
    monkeypatch,
    caplog,
):
    paper = make_paper(db, "Checkpoint persistence")
    run_id = str(uuid4())
    run = ResearchRun(
        id=run_id,
        question="Checkpoint question",
        status="running",
        thread_id=run_id,
    )
    db.add(run)
    db.flush()
    db.commit()

    database_url = str(get_settings().database_url)
    database_url = database_url.replace(
        "postgresql+psycopg://",
        "postgresql://",
        1,
    )
    resources = build_checkpointer(database_url)
    restored_resources = None
    initial_resources_closed = False
    config = {"configurable": {"thread_id": run.thread_id}}
    monkeypatch.setattr(
        stages,
        "retrieve_evidence",
        lambda **kwargs: [
            RetrievedEvidence(
                evidence_id="E1",
                chunk_id=str(uuid4()),
                paper_id=paper.id,
                text="Checkpoint evidence text",
                chunk_index=0,
                page_start=1,
                page_end=1,
                section="Methods",
                title=paper.title,
                filename=paper.filename,
                similarity=0.95,
            )
        ],
    )
    monkeypatch.setattr(
        stages,
        "generate_grounded_answer",
        lambda question, context: {
            "answerable": True,
            "answer": "The evidence supports this answer [E1].",
            "reason": None,
        },
    )

    try:
        state = {
            "question": run.question,
            "thread_id": run.thread_id,
            "paper_ids": [paper.id],
            "top_k": 5,
            "user_decision": None,
            "trace": [],
            "status": "pending",
            "error": None,
        }
        result = build_research_graph(db, resources.saver).invoke(
            state,
            config=config,
        )
        close_checkpointer(resources)
        initial_resources_closed = True
        restored_resources = build_checkpointer(database_url)
        restored = build_research_graph(
            db,
            restored_resources.saver,
        ).get_state(config)

        assert "__interrupt__" in result
        assert result["__interrupt__"][0].value["type"] == "human_review"
        assert restored.values["thread_id"] == run.thread_id
        assert restored.values["status"] == "running"
        assert restored.interrupts[0].value["type"] == "human_review"
        restored_evidence = restored.values["selected_evidence"][0]
        original_evidence = result["selected_evidence"][0]
        assert type(restored_evidence) is RetrievedEvidence
        assert restored_evidence == original_evidence
        assert {
            field: getattr(restored_evidence, field)
            for field in (
                "evidence_id",
                "chunk_id",
                "paper_id",
                "text",
                "chunk_index",
                "page_start",
                "page_end",
                "section",
                "title",
                "filename",
                "similarity",
            )
        } == {
            "evidence_id": "E1",
            "chunk_id": result["selected_evidence"][0].chunk_id,
            "paper_id": paper.id,
            "text": "Checkpoint evidence text",
            "chunk_index": 0,
            "page_start": 1,
            "page_end": 1,
            "section": "Methods",
            "title": paper.title,
            "filename": paper.filename,
            "similarity": 0.95,
        }
        assert type(restored_evidence.evidence_id) is str
        assert type(restored_evidence.chunk_id) is str
        assert type(restored_evidence.paper_id) is str
        assert type(restored_evidence.text) is str
        assert type(restored_evidence.chunk_index) is int
        assert type(restored_evidence.page_start) is int
        assert type(restored_evidence.page_end) is int
        assert type(restored_evidence.section) is str
        assert type(restored_evidence.title) is str
        assert type(restored_evidence.filename) is str
        assert type(restored_evidence.similarity) is float
        assert restored_resources.saver.get_tuple(config) is not None
        resumed_result = build_research_graph(
            db,
            restored_resources.saver,
        ).invoke(
            Command(
                resume={
                    "decision": "approved",
                    "feedback": "Evidence reviewed",
                }
            ),
            config=config,
        )
        assert resumed_result["status"] == "completed"
        assert resumed_result["draft_answer"] == (
            "The evidence supports this answer [E1]."
        )
        assert "reasoner" in [
            event["stage"] for event in resumed_result["trace"]
        ]
        assert run.thread_id == config["configurable"]["thread_id"]
        assert not any(
            "Deserializing unregistered type" in record.getMessage()
            for record in caplog.records
        )
    finally:
        try:
            if restored_resources is not None:
                restored_resources.saver.delete_thread(run.thread_id)
            elif not initial_resources_closed:
                resources.saver.delete_thread(run.thread_id)
        finally:
            if restored_resources is not None:
                close_checkpointer(restored_resources)
            if not initial_resources_closed:
                close_checkpointer(resources)


@pytest.mark.parametrize(
    ("decision", "expected_status"),
    [("approved", "completed"), ("rejected", "cancelled")],
)
def test_postgres_resume_persists_human_checkpoint_decision(
    db: Session,
    monkeypatch,
    decision: str,
    expected_status: str,
):
    run_id = str(uuid4())
    thread_id = str(uuid4())
    run = ResearchRun(
        id=run_id,
        question="Persist the human decision",
        status="running",
        thread_id=thread_id,
    )
    db.add(run)
    db.flush()

    checkpoint_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "review",
            "checkpoint_id": "checkpoint-persisted",
        }
    }

    class FakeGraph:
        def get_state(self, config):
            assert config == {"configurable": {"thread_id": thread_id}}
            return SimpleNamespace(
                values={"thread_id": thread_id},
                interrupts=[
                    SimpleNamespace(value={"type": "human_review"})
                ],
                config=checkpoint_config,
            )

        def invoke(self, command, config):
            assert command.resume == {
                "decision": decision,
                "feedback": "Persist this feedback",
            }
            assert config == {"configurable": {"thread_id": thread_id}}
            graph_status = "completed" if decision == "approved" else "flagged"
            return {
                "question": run.question,
                "thread_id": thread_id,
                "paper_ids": [],
                "top_k": 5,
                "status": graph_status,
                "trace": [],
            }

    monkeypatch.setattr(
        research_router,
        "build_research_graph",
        lambda request_db, checkpointer: FakeGraph(),
    )
    app = FastAPI()
    app.state.checkpointer_resources = SimpleNamespace(saver=object())
    app.include_router(research_router.router)
    app.dependency_overrides[get_db] = lambda: db
    response = TestClient(app).post(
        f"/research/runs/{run_id}/resume",
        json={
            "decision": decision,
            "feedback": "Persist this feedback",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == (
        "completed" if decision == "approved" else "flagged"
    )
    assert run.status == expected_status
    persisted_decisions = db.scalars(
        select(HumanCheckpointDecision).where(
            HumanCheckpointDecision.research_run_id == run_id
        )
    ).all()
    assert len(persisted_decisions) == 1
    persisted = persisted_decisions[0]
    assert persisted.research_run_id == run_id
    assert persisted.thread_id == thread_id
    assert persisted.checkpoint_ns == "review"
    assert persisted.checkpoint_id == "checkpoint-persisted"
    assert persisted.decision == decision
    assert persisted.comment == "Persist this feedback"
