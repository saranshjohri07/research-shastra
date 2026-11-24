from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError
from starlette.requests import Request
from fastapi import FastAPI
from fastapi.testclient import TestClient

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from backend.app.checkpointer import build_checkpoint_serializer
from backend.app.db.models import (
    HumanCheckpointDecision,
    Paper,
    ResearchRun,
    ResearchRunPaper,
)
from backend.app.db.session import get_db
from backend.app.routers import research as research_router
from backend.app.routers.research import (
    GraphResearchRequest,
    GraphResearchResponse,
    invoke_research_run,
    query_research_graph,
)
from backend.app.agents import stages
from backend.app.agents.stages import (
    citation_agent,
    evidence_gatherer,
    human_checkpoint,
    planner,
    validator,
)
from backend.app.agents.workflow import build_research_graph
from backend.app.retrieval.retriever import RetrievedEvidence


class RouteDb:
    def __init__(self, paper_ids=(), fail_commit_number=None):
        self.paper_ids = set(paper_ids)
        self.rows = []
        self.commits = 0
        self.fail_commit_number = fail_commit_number
        self.committed_runs = {}
        self.committed_decisions = set()

    def add(self, row):
        self.rows.append(row)

    def flush(self):
        pass

    def get(self, model, item_id):
        if model is Paper:
            return SimpleNamespace(id=item_id) if item_id in self.paper_ids else None
        return next(
            (row for row in self.rows if isinstance(row, model) and row.id == item_id),
            None,
        )

    def commit(self):
        self.commits += 1
        if self.commits == self.fail_commit_number:
            raise RuntimeError("result commit failed")
        for row in self.rows:
            if isinstance(row, ResearchRun):
                self.committed_runs[row.id] = {
                    "status": row.status,
                    "answer": row.answer,
                    "answerable": row.answerable,
                    "error": row.error,
                    "abstention_reason": row.abstention_reason,
                    "completed_at": row.completed_at,
                    "resume_token": row.resume_token,
                }
            elif isinstance(row, HumanCheckpointDecision):
                self.committed_decisions.add(id(row))

    def execute(self, statement):
        parameters = statement.compile().params
        run = next(
            (
                row
                for row in self.rows
                if isinstance(row, ResearchRun) and row.id in parameters.values()
            ),
            None,
        )
        token = next(
            (
                value
                for key, value in parameters.items()
                if "resume_token" in key and value is not None
            ),
            None,
        )
        if (
            run is not None
            and run.status == "running"
            and run.resume_token is None
            and token is not None
        ):
            run.resume_token = token
            return SimpleNamespace(rowcount=1)
        return SimpleNamespace(rowcount=0)

    def scalar(self, statement):
        parameters = statement.compile().params
        return next(
            (
                row
                for row in self.rows
                if isinstance(row, ResearchRun)
                and row.id in parameters.values()
            ),
            None,
        )

    def rollback(self):
        for row in self.rows:
            if isinstance(row, ResearchRun):
                if row.id in self.committed_runs:
                    for field, value in self.committed_runs[row.id].items():
                        setattr(row, field, value)
                else:
                    row.resume_token = None
        self.rows = [
            row
            for row in self.rows
            if not isinstance(row, HumanCheckpointDecision)
            or id(row) in self.committed_decisions
        ]


def graph_route_client(db, saver=None):
    app = FastAPI()
    app.state.checkpointer_resources = SimpleNamespace(
        saver=saver if saver is not None else MemorySaver(
            serde=build_checkpoint_serializer()
        )
    )
    app.include_router(research_router.router)
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def completed_graph(state, config):
    return {
        **state,
        "status": "completed",
        "draft_answer": "Grounded answer.",
        "answerable": True,
        "citations": [],
        "retrieved_chunks": [],
        "selected_evidence": [],
        "trace": [],
    }


class FakeDb:
    def get(self, model, paper_id):
        if paper_id == "missing":
            return None
        return SimpleNamespace(id=paper_id)


def evidence(evidence_id="E1", similarity=0.9, chunk_id="chunk-1"):
    return RetrievedEvidence(
        evidence_id=evidence_id,
        chunk_id=chunk_id,
        paper_id="paper-1",
        text="Supporting passage.",
        chunk_index=0,
        page_start=1,
        page_end=1,
        section="Methods",
        title="Paper",
        filename="paper.pdf",
        similarity=similarity,
    )


def base_state(**updates):
    state = {
        "question": "How does attention work?",
        "thread_id": "thread-test",
        "paper_ids": ["paper-1"],
        "top_k": 5,
        "trace": [],
        "status": "pending",
        "error": None,
    }
    state.update(updates)
    return state


def invoke_graph(state, decision=None, feedback=None, db=None):
    checkpointer = MemorySaver(serde=build_checkpoint_serializer())
    graph = build_research_graph(db or FakeDb(), checkpointer)
    config = {"configurable": {"thread_id": state["thread_id"]}}
    result = graph.invoke(state, config=config)
    if result.get("__interrupt__") and decision is not None:
        result = graph.invoke(
            Command(
                resume={
                    "decision": decision,
                    "feedback": feedback,
                }
            ),
            config=config,
        )
    return result, graph, config


def test_planner_output_and_malformed_fallback():
    planned = planner(base_state())
    assert planned["query_type"] == "explanatory"
    assert planned["sub_questions"] == ["How does attention work?"]
    fallback = stages.parse_planner_output("{bad", "Compare A and B")
    assert fallback["query_type"] == "comparison"
    assert fallback["sub_questions"] == ["Compare A and B"]


def test_graph_stage_order_with_continue(monkeypatch):
    monkeypatch.setattr(
        stages,
        "retrieve_evidence",
        lambda **kwargs: [evidence()],
    )
    monkeypatch.setattr(
        stages,
        "generate_grounded_answer",
        lambda question, context: {
            "answerable": True,
            "answer": "Attention assigns weights [E1].",
            "reason": None,
        },
    )

    result, _, _ = invoke_graph(base_state(), decision="approved")

    assert [event["stage"] for event in result["trace"]] == [
        "planner",
        "planner",
        "retriever",
        "retriever",
        "evidence_gatherer",
        "evidence_gatherer",
        "human_checkpoint",
        "human_checkpoint",
        "reasoner",
        "reasoner",
        "citation_agent",
        "validator",
    ]
    assert result["status"] == "completed"
    assert result["answerable"] is True
    assert result["validation"]["valid"] is True


def test_zero_evidence_abstains(monkeypatch):
    monkeypatch.setattr(
        stages,
        "retrieve_evidence",
        lambda **kwargs: [],
    )
    result, _, _ = invoke_graph(base_state())
    assert result["status"] == "abstained"
    assert result["answerable"] is False
    assert result["abstention_reason"] == "no_evidence"
    assert "reasoner" not in [event["stage"] for event in result["trace"]]


def test_flag_path_ends_at_human_checkpoint(monkeypatch):
    monkeypatch.setattr(
        stages,
        "retrieve_evidence",
        lambda **kwargs: [evidence()],
    )
    result, _, _ = invoke_graph(base_state(), decision="rejected")
    assert result["status"] == "flagged"
    assert "reasoner" not in [event["stage"] for event in result["trace"]]


def test_low_similarity_abstains(monkeypatch):
    monkeypatch.setattr(
        stages,
        "retrieve_evidence",
        lambda **kwargs: [evidence(similarity=0.05)],
    )
    result, _, _ = invoke_graph(base_state())
    assert result["status"] == "abstained"
    assert result["abstention_reason"] == "low_similarity"


def test_retriever_stage_failure_is_traced(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("retrieval unavailable")

    monkeypatch.setattr(stages, "retrieve_evidence", fail)
    result, _, _ = invoke_graph(base_state())
    assert result["status"] == "failed"
    assert result["error"] == "retrieval unavailable"
    assert any(
        event["stage"] == "retriever" and event["status"] == "failed"
        for event in result["trace"]
    )


def test_unknown_paper_is_a_stage_failure(monkeypatch):
    result, _, _ = invoke_graph(base_state(paper_ids=["missing"]))
    assert result["status"] == "failed"
    assert "Paper not found" in result["error"]


def test_citation_validation_rejects_unknown_marker():
    state = base_state(
        draft_answer="Unsupported claim [E99].",
        answerable=True,
        selected_evidence=[evidence()],
    )
    state = citation_agent(state)
    state = validator(state)
    assert state["validation"]["valid"] is False
    assert state["validation"]["unknown_markers"] == ["E99"]
    assert state["citations"] == []
    assert state["status"] == "failed"


def test_human_checkpoint_interrupts_and_exposes_review_payload(monkeypatch):
    monkeypatch.setattr(
        stages,
        "retrieve_evidence",
        lambda **kwargs: [evidence()],
    )
    result, graph, config = invoke_graph(base_state())
    snapshot = graph.get_state(config)
    interrupt = result["__interrupt__"][0]
    assert interrupt.value["type"] == "human_review"
    assert interrupt.value["question"] == "How does attention work?"
    assert interrupt.value["evidence"][0]["chunk_id"] == "chunk-1"
    assert snapshot.next == ("human_checkpoint",)
    assert snapshot.interrupts[0].value == interrupt.value
    assert "reasoner" not in [event["stage"] for event in snapshot.values["trace"]]
    assert "citation_agent" not in [
        event["stage"] for event in snapshot.values["trace"]
    ]
    assert "validator" not in [event["stage"] for event in snapshot.values["trace"]]


def test_initial_state_decision_cannot_bypass_human_checkpoint(monkeypatch):
    monkeypatch.setattr(
        stages,
        "retrieve_evidence",
        lambda **kwargs: [evidence()],
    )
    result, _, _ = invoke_graph(base_state(user_decision="continue"))
    assert "__interrupt__" in result


def test_initial_api_human_decision_is_rejected():
    with pytest.raises(ValidationError, match="through human review"):
        GraphResearchRequest(
            paper_ids=[str(uuid4())],
            question="Question",
            user_decision="continue",
        )


def test_client_thread_id_is_rejected():
    with pytest.raises(ValidationError, match="thread_id is assigned"):
        GraphResearchRequest(
            paper_ids=[str(uuid4())],
            question="Question",
            thread_id="client-thread",
        )


def test_invalid_resume_decision_fails_without_reasoner(monkeypatch):
    monkeypatch.setattr(stages, "retrieve_evidence", lambda **kwargs: [evidence()])
    result, _, _ = invoke_graph(base_state(), decision="approved-invalid")
    assert result["status"] == "failed"
    assert result["error"] == "Invalid human review decision"
    assert "reasoner" not in [event["stage"] for event in result["trace"]]


def test_approval_and_rejection_resume_checkpoint(monkeypatch):
    monkeypatch.setattr(stages, "retrieve_evidence", lambda **kwargs: [evidence()])
    monkeypatch.setattr(
        stages,
        "generate_grounded_answer",
        lambda question, context: {
            "answerable": True,
            "answer": "Supported answer [E1].",
            "reason": None,
        },
    )
    approved, approved_graph, approved_config = invoke_graph(
        base_state(), decision="approved", feedback="Evidence is relevant"
    )
    assert approved["status"] == "completed"
    assert "reasoner" in [event["stage"] for event in approved["trace"]]
    approved_snapshot = approved_graph.get_state(approved_config)
    assert approved_snapshot.values["user_decision"] == "continue"
    assert approved_snapshot.values["user_feedback"] == "Evidence is relevant"

    rejected, rejected_graph, rejected_config = invoke_graph(
        base_state(), decision="rejected", feedback="Insufficient support"
    )
    assert rejected["status"] == "flagged"
    assert "reasoner" not in [event["stage"] for event in rejected["trace"]]
    rejected_snapshot = rejected_graph.get_state(rejected_config)
    assert rejected_snapshot.values["user_decision"] == "flag"
    assert rejected_snapshot.values["user_feedback"] == "Insufficient support"


def test_graph_compiles_with_checkpointer_and_persists_thread_state(monkeypatch):
    monkeypatch.setattr(stages, "retrieve_evidence", lambda **kwargs: [])
    checkpointer = MemorySaver(serde=build_checkpoint_serializer())
    state = base_state()
    config = {"configurable": {"thread_id": state["thread_id"]}}

    result = build_research_graph(FakeDb(), checkpointer).invoke(
        state,
        config=config,
    )

    assert result["status"] == "abstained"
    graph = build_research_graph(FakeDb(), checkpointer)
    saved_state = graph.get_state(config)
    assert saved_state.values["thread_id"] == state["thread_id"]
    assert saved_state.values["status"] == "abstained"


def test_research_run_invocations_reuse_immutable_persisted_threads(monkeypatch):
    class FakeSession:
        def __init__(self):
            self.commits = 0
            self.runs = {}

        def get(self, model, item_id):
            return self.runs.get(item_id)

        def commit(self):
            self.commits += 1

    invocations = []

    class FakeGraph:
        def invoke(self, state, config):
            invocations.append(config["configurable"]["thread_id"])
            return {**state, "status": "awaiting_review"}

    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda db, checkpointer: FakeGraph(),
    )
    db = FakeSession()
    first_id = str(uuid4())
    second_id = str(uuid4())
    first_thread_id = str(uuid4())
    second_thread_id = str(uuid4())
    first_run = ResearchRun(
        id=first_id,
        question="First",
        thread_id=first_thread_id,
    )
    second_run = ResearchRun(
        id=second_id,
        question="Second",
        thread_id=second_thread_id,
    )

    stale_first_run = ResearchRun(
        id=first_id,
        question="First",
        thread_id=str(uuid4()),
    )
    db.runs = {first_id: first_run, second_id: second_run}

    for run in (stale_first_run, first_run, second_run):
        invoke_research_run(
            db,
            run,
            base_state(thread_id=run.id),
            checkpointer=object(),
        )

    assert first_run.thread_id == first_thread_id
    assert second_run.thread_id == second_thread_id
    assert stale_first_run.thread_id != first_thread_id
    assert invocations == [
        first_thread_id,
        first_thread_id,
        second_thread_id,
    ]
    assert db.commits == 0


@pytest.mark.parametrize("thread_id", [None, "", "not-a-uuid"])
def test_research_run_invocation_rejects_invalid_persisted_thread_without_mutation(
    monkeypatch,
    thread_id,
):
    class FakeSession:
        def get(self, model, item_id):
            return run

        def commit(self):
            pytest.fail("Invalid persisted identity must not be committed")

    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda *args, **kwargs: pytest.fail("Graph must not be constructed"),
    )
    run = ResearchRun(
        id=str(uuid4()),
        question="Question",
        thread_id=thread_id,
    )

    with pytest.raises(ValueError, match="valid persisted thread_id"):
        invoke_research_run(
            FakeSession(),
            run,
            base_state(),
            checkpointer=object(),
        )

    assert run.thread_id == thread_id


def test_graph_endpoint_persists_run_scope_and_returns_declared_response(
    monkeypatch,
):
    paper_ids = [str(uuid4()), str(uuid4())]
    db = RouteDb(paper_ids=paper_ids)
    saver = MemorySaver()
    captured = {}

    class FakeGraph:
        def invoke(self, state, config):
            captured["state"] = state
            captured["config"] = config
            result = completed_graph(state, config)
            result["retrieved_chunks"] = [evidence()]
            result["selected_evidence"] = [evidence()]
            return result

    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda db, checkpointer: (
            captured.update(checkpointer=checkpointer) or FakeGraph()
        ),
    )
    client = graph_route_client(db, saver)
    response = client.post(
        "/research/graph/query",
        json={
            "paper_ids": [paper_ids[0], paper_ids[1], paper_ids[0]],
            "question": "Question",
        },
    )

    assert response.status_code == 200
    response_data = response.json()
    run = next(row for row in db.rows if isinstance(row, ResearchRun))
    scopes = [row for row in db.rows if isinstance(row, ResearchRunPaper)]
    assert response_data["research_run_id"] == run.id
    assert response_data["selected_evidence"][0]["chunk_id"] == "chunk-1"
    assert response_data["retrieved_chunks"][0]["text"] == "Supporting passage."
    assert run.thread_id == run.id
    assert run.status == "completed"
    assert [(row.research_run_id, row.paper_id, row.position) for row in scopes] == [
        (run.id, paper_ids[0], 0),
        (run.id, paper_ids[1], 1),
    ]
    assert captured["checkpointer"] is saver
    assert captured["config"] == {
        "configurable": {"thread_id": run.id}
    }
    assert captured["state"]["paper_ids"] == paper_ids
    assert set(response_data) == set(GraphResearchResponse.model_fields)
    assert response_data["research_run_id"] == run.id
    assert db.commits == 2


def test_graph_endpoint_returns_pending_interrupt_without_changing_sql_run_status(
    monkeypatch,
):
    paper_id = str(uuid4())
    db = RouteDb(paper_ids=[paper_id])
    monkeypatch.setattr(
        stages,
        "retrieve_evidence",
        lambda **kwargs: [evidence()],
    )
    response = graph_route_client(db).post(
        "/research/graph/query",
        json={"paper_ids": [paper_id], "question": "Question"},
    )

    assert response.status_code == 200
    result = response.json()
    run = next(row for row in db.rows if isinstance(row, ResearchRun))
    assert result["status"] == "awaiting_review"
    assert result["interrupts"][0]["type"] == "human_review"
    assert result["interrupts"][0]["question"] == "Question"
    assert result["interrupts"][0]["evidence"][0]["chunk_id"] == "chunk-1"
    assert run.status == "running"


def test_graph_endpoint_rejects_malformed_uuid_without_creating_run(monkeypatch):
    db = RouteDb()
    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda *args, **kwargs: pytest.fail("Graph must not run"),
    )
    response = graph_route_client(db).post(
        "/research/graph/query",
        json={"paper_ids": ["not-a-uuid"], "question": "Question"},
    )
    assert response.status_code == 422
    assert "UUID" in response.text
    assert not any(isinstance(row, ResearchRun) for row in db.rows)


def test_graph_endpoint_rejects_nonexistent_paper_without_creating_run(monkeypatch):
    db = RouteDb()
    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda *args, **kwargs: pytest.fail("Graph must not run"),
    )
    response = graph_route_client(db).post(
        "/research/graph/query",
        json={"paper_ids": [str(uuid4())], "question": "Question"},
    )
    assert response.status_code == 404
    assert "not found" in response.text
    assert not any(isinstance(row, ResearchRun) for row in db.rows)


def test_graph_endpoint_rejects_empty_paper_scope_without_creating_run(monkeypatch):
    db = RouteDb()
    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda *args, **kwargs: pytest.fail("Graph must not run"),
    )
    response = graph_route_client(db).post(
        "/research/graph/query",
        json={"paper_ids": [], "question": "Question"},
    )
    assert response.status_code == 422
    assert not any(isinstance(row, ResearchRun) for row in db.rows)


def test_graph_endpoint_rejects_client_thread_id_at_http_boundary():
    response = graph_route_client(RouteDb()).post(
        "/research/graph/query",
        json={
            "paper_ids": [str(uuid4())],
            "question": "Question",
            "thread_id": "client-thread",
        },
    )
    assert response.status_code == 422
    assert "thread_id is assigned" in response.text


@pytest.mark.parametrize(
    ("decision", "expected_graph_status", "expected_run_status"),
    [
        ("approved", "completed", "completed"),
        ("rejected", "flagged", "cancelled"),
    ],
)
def test_resume_route_uses_persisted_thread_and_reuses_existing_run(
    monkeypatch,
    decision,
    expected_graph_status,
    expected_run_status,
):
    run_id = str(uuid4())
    stored_thread_id = str(uuid4())
    run = ResearchRun(
        id=run_id,
        question="Question",
        status="running",
        top_k=5,
        thread_id=stored_thread_id,
    )
    db = RouteDb()
    db.rows.append(run)
    saver = MemorySaver(serde=build_checkpoint_serializer())
    captured = {}

    class FakeGraph:
        def get_state(self, config):
            captured["state_config"] = config
            return SimpleNamespace(
                values={"thread_id": stored_thread_id},
                interrupts=[SimpleNamespace(value={"type": "human_review"})],
                config={
                    "configurable": {
                        "thread_id": stored_thread_id,
                        "checkpoint_ns": "review",
                        "checkpoint_id": "checkpoint-123",
                    }
                },
            )

        def invoke(self, command, config):
            captured["command"] = command
            captured["invoke_config"] = config
            result = completed_graph(
                base_state(
                    thread_id=stored_thread_id,
                    question=run.question,
                    paper_ids=[],
                ),
                config,
            )
            if decision == "rejected":
                result.update(status="flagged", draft_answer=None, answerable=False)
            return result

    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda request_db, checkpointer: (
            captured.update(db=request_db, checkpointer=checkpointer)
            or FakeGraph()
        ),
    )
    response = graph_route_client(db, saver).post(
        f"/research/runs/{run_id}/resume",
        json={"decision": decision, "feedback": "Evidence is relevant"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["research_run_id"] == run_id
    assert result["thread_id"] == stored_thread_id
    assert result["status"] == expected_graph_status
    assert captured["db"] is db
    assert captured["checkpointer"] is saver
    assert captured["state_config"] == {
        "configurable": {"thread_id": stored_thread_id}
    }
    assert captured["invoke_config"] == captured["state_config"]
    assert isinstance(captured["command"], Command)
    assert captured["command"].resume == {
        "decision": decision,
        "feedback": "Evidence is relevant",
    }
    assert run.thread_id == stored_thread_id
    assert run.status == expected_run_status
    assert not any(isinstance(row, ResearchRunPaper) for row in db.rows)
    assert len([row for row in db.rows if isinstance(row, ResearchRun)]) == 1
    decisions = [
        row for row in db.rows if isinstance(row, HumanCheckpointDecision)
    ]
    assert len(decisions) == 1
    assert decisions[0].research_run_id == run_id
    assert decisions[0].thread_id == stored_thread_id
    assert decisions[0].checkpoint_ns == "review"
    assert decisions[0].checkpoint_id == "checkpoint-123"
    assert decisions[0].decision == decision
    assert decisions[0].comment == "Evidence is relevant"


def test_resume_route_rejects_duplicate_after_success(monkeypatch):
    run_id = str(uuid4())
    thread_id = str(uuid4())
    run = ResearchRun(
        id=run_id,
        question="Question",
        status="running",
        thread_id=thread_id,
    )
    db = RouteDb()
    db.rows.append(run)
    graph_calls = []

    class FakeGraph:
        def get_state(self, config):
            return SimpleNamespace(
                values={"thread_id": thread_id},
                interrupts=[SimpleNamespace(value={"type": "human_review"})],
            )

        def invoke(self, command, config):
            graph_calls.append(command)
            return completed_graph(
                base_state(thread_id=thread_id, paper_ids=[]),
                config,
            )

    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda *args, **kwargs: FakeGraph(),
    )
    client = graph_route_client(db)
    path = f"/research/runs/{run_id}/resume"
    first = client.post(path, json={"decision": "approved"})
    duplicate = client.post(path, json={"decision": "approved"})

    assert first.status_code == 200
    assert duplicate.status_code == 409
    assert "already been claimed" in duplicate.text
    assert len(graph_calls) == 1
    assert run.status == "completed"
    assert run.resume_token is not None
    assert len(
        [row for row in db.rows if isinstance(row, HumanCheckpointDecision)]
    ) == 1


def test_resume_decision_insert_failure_rolls_back_resume_claim(monkeypatch):
    run_id = str(uuid4())
    thread_id = str(uuid4())
    run = ResearchRun(
        id=run_id,
        question="Question",
        status="running",
        thread_id=thread_id,
    )
    db = RouteDb(fail_commit_number=1)
    db.rows.append(run)

    class FakeGraph:
        def get_state(self, config):
            return SimpleNamespace(
                values={"thread_id": thread_id},
                interrupts=[
                    SimpleNamespace(value={"type": "human_review"})
                ],
            )

        def invoke(self, *args, **kwargs):
            pytest.fail("Graph resume must wait for claim and decision commit")

    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda *args, **kwargs: FakeGraph(),
    )
    response = graph_route_client(db).post(
        f"/research/runs/{run_id}/resume",
        json={"decision": "approved"},
    )

    assert response.status_code == 500
    assert "persist the research run resume decision" in response.text
    assert run.resume_token is None
    assert not any(
        isinstance(row, HumanCheckpointDecision) for row in db.rows
    )


@pytest.mark.parametrize("run_status", ["completed", "cancelled", "failed"])
def test_resume_route_rejects_non_running_run_before_checkpoint_access(
    monkeypatch,
    run_status,
):
    run_id = str(uuid4())
    run = ResearchRun(
        id=run_id,
        question="Question",
        status=run_status,
        thread_id=str(uuid4()),
    )
    db = RouteDb()
    db.rows.append(run)
    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda *args, **kwargs: pytest.fail("Non-running run must not load a graph"),
    )

    response = graph_route_client(db).post(
        f"/research/runs/{run_id}/resume",
        json={"decision": "approved"},
    )

    assert response.status_code == 409
    assert db.commits == 0


def test_resume_route_rejects_without_pending_human_review(monkeypatch):
    run_id = str(uuid4())
    run = ResearchRun(
        id=run_id,
        question="Question",
        status="running",
        thread_id=str(uuid4()),
    )
    db = RouteDb()
    db.rows.append(run)

    class FakeGraph:
        def get_state(self, config):
            return SimpleNamespace(
                values={"thread_id": run.thread_id},
                interrupts=(),
            )

        def invoke(self, *args, **kwargs):
            pytest.fail("A run without a pending interrupt must not resume")

    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda *args, **kwargs: FakeGraph(),
    )
    response = graph_route_client(db).post(
        f"/research/runs/{run_id}/resume",
        json={"decision": "approved"},
    )

    assert response.status_code == 409
    assert run.status == "running"
    assert db.commits == 0
    assert not any(
        isinstance(row, HumanCheckpointDecision) for row in db.rows
    )


def test_resume_route_rejects_mismatched_checkpoint_identity(monkeypatch):
    run_id = str(uuid4())
    run = ResearchRun(
        id=run_id,
        question="Question",
        status="running",
        thread_id=str(uuid4()),
    )
    db = RouteDb()
    db.rows.append(run)

    class MismatchedGraph:
        def get_state(self, config):
            return SimpleNamespace(
                values={"thread_id": str(uuid4())},
                interrupts=[SimpleNamespace(value={"type": "human_review"})],
            )

    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda *args, **kwargs: MismatchedGraph(),
    )
    response = graph_route_client(db).post(
        f"/research/runs/{run_id}/resume",
        json={"decision": "approved"},
    )

    assert response.status_code == 409
    assert "identity does not match" in response.text
    assert run.resume_token is None
    assert db.commits == 0
    assert not any(
        isinstance(row, HumanCheckpointDecision) for row in db.rows
    )


@pytest.mark.parametrize("body", [
    {"decision": "continue"},
    {"decision": "approved", "thread_id": str(uuid4())},
])
def test_resume_route_rejects_invalid_decision_and_client_identity(body):
    run_id = str(uuid4())
    run = ResearchRun(
        id=run_id,
        question="Question",
        status="running",
        thread_id=str(uuid4()),
    )
    db = RouteDb()
    db.rows.append(run)
    response = graph_route_client(db).post(
        f"/research/runs/{run_id}/resume",
        json=body,
    )
    assert response.status_code == 422
    assert run.status == "running"
    assert db.commits == 0
    assert not any(
        isinstance(row, HumanCheckpointDecision) for row in db.rows
    )


@pytest.mark.parametrize("thread_id", [None, "", "not-a-uuid"])
def test_resume_route_rejects_invalid_stored_thread_without_mutation(
    monkeypatch,
    thread_id,
):
    run_id = str(uuid4())
    run = ResearchRun(
        id=run_id,
        question="Question",
        status="running",
        thread_id=thread_id,
    )
    db = RouteDb()
    db.rows.append(run)
    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda *args, **kwargs: pytest.fail("Invalid identity must not load a graph"),
    )
    response = graph_route_client(db).post(
        f"/research/runs/{run_id}/resume",
        json={"decision": "approved"},
    )

    assert response.status_code == 500
    assert "persisted thread_id" in response.text
    assert run.thread_id == thread_id
    assert run.status == "running"
    assert db.commits == 0


def test_resume_route_reports_checkpoint_restore_failure(monkeypatch):
    run_id = str(uuid4())
    run = ResearchRun(
        id=run_id,
        question="Question",
        status="running",
        thread_id=str(uuid4()),
    )
    db = RouteDb()
    db.rows.append(run)

    class FailedRestoreGraph:
        def get_state(self, config):
            raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda *args, **kwargs: FailedRestoreGraph(),
    )
    response = graph_route_client(db).post(
        f"/research/runs/{run_id}/resume",
        json={"decision": "approved"},
    )

    assert response.status_code == 500
    assert "Unable to restore" in response.text
    assert run.status == "running"
    assert db.commits == 0


def test_resume_route_does_not_restart_after_graph_exception(monkeypatch):
    run_id = str(uuid4())
    thread_id = str(uuid4())
    run = ResearchRun(
        id=run_id,
        question="Question",
        status="running",
        thread_id=thread_id,
    )
    db = RouteDb()
    db.rows.append(run)
    captured = {}

    class FailedResumeGraph:
        def get_state(self, config):
            captured["state_config"] = config
            return SimpleNamespace(
                values={"thread_id": thread_id},
                interrupts=[SimpleNamespace(value={"type": "human_review"})],
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": "",
                        "checkpoint_id": "checkpoint-456",
                    }
                },
            )

        def invoke(self, command, config):
            captured["command"] = command
            captured["invoke_config"] = config
            raise RuntimeError("resume failed")

    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda *args, **kwargs: FailedResumeGraph(),
    )
    response = graph_route_client(db).post(
        f"/research/runs/{run_id}/resume",
        json={"decision": "rejected"},
    )

    assert response.status_code == 500
    assert "Unable to resume" in response.text
    assert isinstance(captured["command"], Command)
    assert captured["state_config"] == captured["invoke_config"]
    assert captured["invoke_config"]["configurable"]["thread_id"] == thread_id
    assert run.thread_id == thread_id
    assert run.status == "running"
    assert run.resume_token is not None
    assert "recovery pending" in run.error
    assert db.commits == 2
    decisions = [
        row for row in db.rows if isinstance(row, HumanCheckpointDecision)
    ]
    assert len(decisions) == 1
    assert decisions[0].decision == "rejected"


def test_resume_result_commit_failure_does_not_keep_run_completed(monkeypatch):
    run_id = str(uuid4())
    thread_id = str(uuid4())
    run = ResearchRun(
        id=run_id,
        question="Question",
        status="running",
        thread_id=thread_id,
    )
    db = RouteDb(fail_commit_number=2)
    db.rows.append(run)
    db.committed_runs[run_id] = {
        "status": "running",
        "answer": None,
        "answerable": None,
        "error": None,
        "abstention_reason": None,
        "completed_at": None,
        "resume_token": None,
    }

    class FakeGraph:
        def get_state(self, config):
            return SimpleNamespace(
                values={"thread_id": thread_id},
                interrupts=[SimpleNamespace(value={"type": "human_review"})]
            )

        def invoke(self, command, config):
            return completed_graph(
                base_state(thread_id=thread_id, paper_ids=[]),
                config,
            )

    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda *args, **kwargs: FakeGraph(),
    )
    response = graph_route_client(db).post(
        f"/research/runs/{run_id}/resume",
        json={"decision": "approved"},
    )

    assert response.status_code == 500
    assert "checkpoint state may have advanced" in response.text
    assert run.status == "running"
    assert run.completed_at is None
    assert run.resume_token is not None


def test_resume_route_returns_404_for_unknown_run():
    response = graph_route_client(RouteDb()).post(
        f"/research/runs/{uuid4()}/resume",
        json={"decision": "approved"},
    )
    assert response.status_code == 404


def test_get_research_run_restores_result_from_persisted_thread(monkeypatch):
    run_id = str(uuid4())
    thread_id = str(uuid4())
    run = ResearchRun(
        id=run_id,
        question="Question",
        status="completed",
        thread_id=thread_id,
        answer="Grounded answer.",
        answerable=True,
    )
    db = RouteDb()
    db.rows.append(run)
    captured = {}

    class FakeGraph:
        def get_state(self, config):
            captured["config"] = config
            return SimpleNamespace(
                values={
                    **base_state(thread_id=thread_id),
                    "status": "completed",
                    "draft_answer": "Grounded answer.",
                    "answerable": True,
                    "citations": [],
                    "selected_evidence": [],
                },
                interrupts=[],
            )

    monkeypatch.setattr(
        research_router,
        "build_research_graph",
        lambda request_db, checkpointer: FakeGraph(),
    )
    response = graph_route_client(db).get(f"/research/runs/{run_id}")

    assert response.status_code == 200
    assert response.json()["research_run_id"] == run_id
    assert response.json()["thread_id"] == thread_id
    assert response.json()["status"] == "completed"
    assert response.json()["draft_answer"] == "Grounded answer."
    assert captured["config"] == {"configurable": {"thread_id": thread_id}}


def test_get_research_run_rejects_mismatched_checkpoint_identity(monkeypatch):
    run_id = str(uuid4())
    thread_id = str(uuid4())
    run = ResearchRun(
        id=run_id,
        question="Question",
        status="running",
        thread_id=thread_id,
    )
    db = RouteDb()
    db.rows.append(run)

    class FakeGraph:
        def get_state(self, config):
            return SimpleNamespace(
                values=base_state(thread_id=str(uuid4())),
                interrupts=[],
            )

    monkeypatch.setattr(
        research_router,
        "build_research_graph",
        lambda request_db, checkpointer: FakeGraph(),
    )
    response = graph_route_client(db).get(f"/research/runs/{run_id}")

    assert response.status_code == 409
    assert "identity does not match" in response.text


def test_graph_response_schema_is_declared_in_openapi():
    schema = graph_route_client(RouteDb()).get("/openapi.json").json()
    operation = schema["paths"]["/research/graph/query"]["post"]
    response_ref = operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"]
    response_schema = schema["components"]["schemas"][response_ref.rsplit("/", 1)[-1]]
    assert "research_run_id" in response_schema["properties"]
    assert "research_run_id" in response_schema["required"]


def test_graph_exception_marks_persisted_run_failed(monkeypatch):
    paper_id = str(uuid4())
    db = RouteDb(paper_ids=[paper_id])

    class FailedGraph:
        def invoke(self, state, config):
            raise RuntimeError("graph failed")

    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda db, checkpointer: FailedGraph(),
    )

    with pytest.raises(RuntimeError, match="graph failed"):
        query_research_graph(
            GraphResearchRequest(paper_ids=[paper_id], question="Question"),
            Request(
                {
                    "type": "http",
                    "app": SimpleNamespace(
                        state=SimpleNamespace(
                            checkpointer_resources=SimpleNamespace(
                                saver=MemorySaver(
                                    serde=build_checkpoint_serializer()
                                )
                            )
                        )
                    ),
                }
            ),
            db,
        )

    run = next(row for row in db.rows if isinstance(row, ResearchRun))
    assert run.status == "failed"
    assert run.error == "graph failed"
    assert run.completed_at is not None


def test_failed_final_result_commit_rolls_back_completion(monkeypatch):
    paper_id = str(uuid4())
    db = RouteDb(paper_ids=[paper_id], fail_commit_number=2)

    class FakeGraph:
        def invoke(self, state, config):
            return completed_graph(state, config)

    monkeypatch.setattr(
        "backend.app.routers.research.build_research_graph",
        lambda db, checkpointer: FakeGraph(),
    )

    with pytest.raises(RuntimeError, match="result commit failed"):
        query_research_graph(
            GraphResearchRequest(paper_ids=[paper_id], question="Question"),
            Request(
                {
                    "type": "http",
                    "app": SimpleNamespace(
                        state=SimpleNamespace(
                            checkpointer_resources=SimpleNamespace(
                                saver=MemorySaver(
                                    serde=build_checkpoint_serializer()
                                )
                            )
                        )
                    ),
                }
            ),
            db,
        )

    run = next(row for row in db.rows if isinstance(row, ResearchRun))
    assert db.committed_runs[run.id]["status"] == "running"
    assert run.status == "running"
    assert run.completed_at is None


def test_explicit_continue_reaches_reasoner(monkeypatch):
    monkeypatch.setattr(
        stages,
        "retrieve_evidence",
        lambda **kwargs: [evidence()],
    )
    monkeypatch.setattr(
        stages,
        "generate_grounded_answer",
        lambda question, context: {
            "answerable": True,
            "answer": "Answer [E1].",
            "reason": None,
        },
    )
    result, _, _ = invoke_graph(base_state(), decision="approved")
    assert result["status"] == "completed"
    assert "reasoner" in [event["stage"] for event in result["trace"]]


def test_explicit_flag_never_calls_reasoner(monkeypatch):
    monkeypatch.setattr(
        stages,
        "retrieve_evidence",
        lambda **kwargs: [evidence()],
    )
    monkeypatch.setattr(
        stages,
        "generate_grounded_answer",
        lambda *args: pytest.fail("Reasoner must not run on Flag"),
    )
    result, _, _ = invoke_graph(base_state(), decision="rejected")
    assert result["status"] == "flagged"
    assert "reasoner" not in [event["stage"] for event in result["trace"]]


def test_reasoner_exception_is_distinct_and_traced(monkeypatch):
    monkeypatch.setattr(
        stages,
        "retrieve_evidence",
        lambda **kwargs: [evidence()],
    )
    monkeypatch.setattr(
        stages,
        "generate_grounded_answer",
        lambda *args: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )
    result, _, _ = invoke_graph(base_state(), decision="approved")
    assert result["status"] == "failed"
    assert result["error"] == "model unavailable"
    assert "citation_agent" not in [event["stage"] for event in result["trace"]]
    assert "validator" not in [event["stage"] for event in result["trace"]]


def test_answer_without_evidence_markers_fails_validation():
    state = base_state(
        draft_answer="A factual answer without markers.",
        answerable=True,
        selected_evidence=[evidence()],
    )
    state = citation_agent(state)
    state = validator(state)
    assert state["status"] == "failed"
    assert state["citations"] == []
    assert state["validation"]["validation_reason"] == "answer_has_no_evidence_markers"


def test_valid_marker_maps_to_original_chunk():
    state = base_state(
        draft_answer="Supporting result [E1].",
        answerable=True,
        selected_evidence=[evidence(evidence_id="E1", chunk_id="original-chunk")],
    )
    state = citation_agent(state)
    state = validator(state)
    assert state["status"] == "completed"
    assert state["validation"]["valid"] is True
    assert state["citations"][0].chunk_id == "original-chunk"
