from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.agents import stages
from backend.app.agents import workflow
from backend.app.agents.workflow import build_research_graph
from backend.app.checkpointer import build_checkpoint_serializer
from backend.app.db.base import Base
from backend.app.db import models  # noqa: F401
from backend.app.db.models import (
    HumanCheckpointDecision,
    Paper,
    ResearchRun,
)
from backend.app.retrieval.retriever import RetrievedEvidence
from backend.app.services import research_recovery


@pytest.fixture
def recovery_sessions(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )
    monkeypatch.setattr(research_recovery, "SessionLocal", session_factory)
    yield session_factory
    engine.dispose()


def _create_interrupted_run(session_factory):
    run_id = str(uuid4())
    thread_id = run_id
    paper_id = str(uuid4())
    with session_factory() as db:
        db.add(
            Paper(
                id=paper_id,
                title="Recovery fixture",
                filename="recovery.pdf",
                file_path="recovery.pdf",
                page_count=1,
            )
        )
        run = ResearchRun(
            id=run_id,
            question="What supports the answer?",
            status="running",
            thread_id=thread_id,
        )
        db.add(run)
        db.commit()
        return run_id, thread_id, paper_id


def _evidence(paper_id: str) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id="E1",
        chunk_id=str(uuid4()),
        paper_id=paper_id,
        text="Deterministic recovery evidence.",
        chunk_index=0,
        page_start=1,
        page_end=1,
        section="Methods",
        title="Recovery fixture",
        filename="recovery.pdf",
        similarity=0.99,
    )


def _pause_graph(
    session_factory,
    monkeypatch,
    checkpointer: MemorySaver,
    run_id: str,
    thread_id: str,
    paper_id: str,
):
    monkeypatch.setattr(
        stages,
        "retrieve_evidence",
        lambda **kwargs: [_evidence(kwargs["paper_id"])],
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
    config = {"configurable": {"thread_id": thread_id}}
    with session_factory() as db:
        result = build_research_graph(db, checkpointer).invoke(
            {
                "question": "What supports the answer?",
                "thread_id": thread_id,
                "paper_ids": [paper_id],
                "top_k": 5,
                "trace": [],
                "status": "pending",
                "error": None,
            },
            config=config,
        )
        assert result["__interrupt__"][0].value["type"] == "human_review"
        snapshot = build_research_graph(db, checkpointer).get_state(config)
        assert snapshot.interrupts[0].value["type"] == "human_review"
        return snapshot


def _persist_decision(
    session_factory,
    run_id: str,
    thread_id: str,
    snapshot,
    decision: str,
    checkpoint_id: str | None = None,
):
    configurable = snapshot.config["configurable"]
    with session_factory() as db:
        db.add(
            HumanCheckpointDecision(
                research_run_id=run_id,
                thread_id=thread_id,
                checkpoint_ns=configurable.get("checkpoint_ns", ""),
                checkpoint_id=checkpoint_id or configurable["checkpoint_id"],
                decision=decision,
                comment="Persisted review feedback",
            )
        )
        run = db.get(ResearchRun, run_id)
        run.resume_token = "claimed"
        db.commit()


@pytest.mark.parametrize(
    ("decision", "expected_status", "expected_user_decision"),
    [
        ("approved", "completed", "continue"),
        ("rejected", "cancelled", "flag"),
    ],
)
def test_recovery_replays_persisted_decision_once(
    recovery_sessions,
    monkeypatch,
    decision,
    expected_status,
    expected_user_decision,
):
    run_id, thread_id, paper_id = _create_interrupted_run(
        recovery_sessions
    )
    checkpointer = MemorySaver(serde=build_checkpoint_serializer())
    snapshot = _pause_graph(
        recovery_sessions,
        monkeypatch,
        checkpointer,
        run_id,
        thread_id,
        paper_id,
    )
    _persist_decision(
        recovery_sessions, run_id, thread_id, snapshot, decision
    )
    reasoner_calls = []
    original_reasoner = workflow.reasoner

    def count_reasoner(state):
        reasoner_calls.append(True)
        return original_reasoner(state)

    monkeypatch.setattr(workflow, "reasoner", count_reasoner)

    research_recovery.recover_pending_research_runs(checkpointer)
    research_recovery.recover_pending_research_runs(checkpointer)

    with recovery_sessions() as db:
        run = db.get(ResearchRun, run_id)
        decisions = db.scalars(
            select(HumanCheckpointDecision).where(
                HumanCheckpointDecision.research_run_id == run_id
            )
        ).all()
    assert run.status == expected_status
    assert run.thread_id == thread_id
    assert run.error is None
    assert len(decisions) == 1
    assert decisions[0].thread_id == thread_id
    assert decisions[0].decision == decision
    assert decisions[0].comment == "Persisted review feedback"
    assert reasoner_calls == ([True] if decision == "approved" else [])


def test_pending_review_without_decision_remains_paused(
    recovery_sessions,
    monkeypatch,
):
    run_id, thread_id, paper_id = _create_interrupted_run(recovery_sessions)
    checkpointer = MemorySaver(serde=build_checkpoint_serializer())
    config = {"configurable": {"thread_id": thread_id}}
    _pause_graph(
        recovery_sessions,
        monkeypatch,
        checkpointer,
        run_id,
        thread_id,
        paper_id,
    )

    research_recovery.recover_pending_research_runs(checkpointer)
    with recovery_sessions() as db:
        snapshot = build_research_graph(db, checkpointer).get_state(config)
    with recovery_sessions() as db:
        run = db.get(ResearchRun, run_id)
        decisions = db.scalars(
            select(HumanCheckpointDecision).where(
                HumanCheckpointDecision.research_run_id == run_id
            )
        ).all()
    assert run.status == "running"
    assert run.resume_token is None
    assert run.error is None
    assert decisions == []
    assert snapshot.interrupts[0].value["type"] == "human_review"


def test_missing_decision_checkpoint_stays_recoverable_error(
    recovery_sessions,
    monkeypatch,
):
    run_id, thread_id, paper_id = _create_interrupted_run(
        recovery_sessions
    )
    checkpointer = MemorySaver(serde=build_checkpoint_serializer())
    snapshot = _pause_graph(
        recovery_sessions,
        monkeypatch,
        checkpointer,
        run_id,
        thread_id,
        paper_id,
    )
    _persist_decision(
        recovery_sessions,
        run_id,
        thread_id,
        snapshot,
        "approved",
        checkpoint_id="missing-checkpoint",
    )

    research_recovery.recover_pending_research_runs(checkpointer)

    with recovery_sessions() as db:
        run = db.get(ResearchRun, run_id)
    assert run.status == "running"
    assert run.completed_at is None
    assert run.resume_token == "claimed"
    assert "checkpoint is missing" in run.error


@pytest.mark.parametrize(
    ("decision", "expected_status", "expected_next"),
    [
        ("approved", "completed", ("reasoner",)),
        ("rejected", "cancelled", ()),
    ],
)
def test_recovery_continues_after_decision_checkpoint_was_advanced(
    recovery_sessions,
    monkeypatch,
    decision,
    expected_status,
    expected_next,
):
    run_id, thread_id, paper_id = _create_interrupted_run(
        recovery_sessions
    )
    checkpointer = MemorySaver(serde=build_checkpoint_serializer())
    paused_snapshot = _pause_graph(
        recovery_sessions,
        monkeypatch,
        checkpointer,
        run_id,
        thread_id,
        paper_id,
    )
    _persist_decision(
        recovery_sessions, run_id, thread_id, paused_snapshot, decision
    )
    config = {"configurable": {"thread_id": thread_id}}
    with recovery_sessions() as db:
        graph = build_research_graph(db, checkpointer)
        stream = graph.stream(
            Command(
                resume={
                    "decision": decision,
                    "feedback": "Persisted review feedback",
                }
            ),
            config=config,
            stream_mode="updates",
        )
        for update in stream:
            if "human_checkpoint" in update:
                break
        stream.close()
        advanced_snapshot = graph.get_state(config)
    assert advanced_snapshot.values["user_decision"] == (
        "continue" if decision == "approved" else "flag"
    )
    assert advanced_snapshot.next == expected_next
    assert not advanced_snapshot.interrupts

    research_recovery.recover_pending_research_runs(checkpointer)

    with recovery_sessions() as db:
        run = db.get(ResearchRun, run_id)
    assert run.status == expected_status
    assert run.thread_id == thread_id
    assert run.error is None


@pytest.mark.parametrize(
    ("decision", "expected_status"),
    [("approved", "completed"), ("rejected", "cancelled")],
)
def test_recovery_reconciles_graph_completion_before_sql_status_commit(
    recovery_sessions,
    monkeypatch,
    decision,
    expected_status,
):
    run_id, thread_id, paper_id = _create_interrupted_run(recovery_sessions)
    checkpointer = MemorySaver(serde=build_checkpoint_serializer())
    snapshot = _pause_graph(
        recovery_sessions,
        monkeypatch,
        checkpointer,
        run_id,
        thread_id,
        paper_id,
    )
    _persist_decision(
        recovery_sessions, run_id, thread_id, snapshot, decision
    )
    config = {"configurable": {"thread_id": thread_id}}
    with recovery_sessions() as db:
        result = build_research_graph(db, checkpointer).invoke(
            Command(
                resume={
                    "decision": decision,
                    "feedback": "Persisted review feedback",
                }
            ),
            config=config,
        )
        assert result["status"] in ("completed", "flagged")
    with recovery_sessions() as db:
        assert db.get(ResearchRun, run_id).status == "running"

    research_recovery.recover_pending_research_runs(checkpointer)

    with recovery_sessions() as db:
        run = db.get(ResearchRun, run_id)
    assert run.status == expected_status
    assert run.thread_id == thread_id
    assert run.error is None
    assert run.completed_at is not None
