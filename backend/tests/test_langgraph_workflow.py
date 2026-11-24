from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.app.routers.research import GraphResearchRequest
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
        "user_decision": "continue",
        "trace": [],
        "status": "pending",
        "error": None,
    }
    state.update(updates)
    return state


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

    graph = build_research_graph(FakeDb())
    result = graph.invoke(base_state())

    assert [event["stage"] for event in result["trace"]] == [
        "planner",
        "planner",
        "retriever",
        "retriever",
        "evidence_gatherer",
        "evidence_gatherer",
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
    result = build_research_graph(FakeDb()).invoke(base_state())
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
    result = build_research_graph(FakeDb()).invoke(
        base_state(user_decision="flag")
    )
    assert result["status"] == "flagged"
    assert "reasoner" not in [event["stage"] for event in result["trace"]]


def test_low_similarity_abstains(monkeypatch):
    monkeypatch.setattr(
        stages,
        "retrieve_evidence",
        lambda **kwargs: [evidence(similarity=0.05)],
    )
    result = build_research_graph(FakeDb()).invoke(base_state())
    assert result["status"] == "abstained"
    assert result["abstention_reason"] == "low_similarity"


def test_retriever_stage_failure_is_traced(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("retrieval unavailable")

    monkeypatch.setattr(stages, "retrieve_evidence", fail)
    result = build_research_graph(FakeDb()).invoke(base_state())
    assert result["status"] == "failed"
    assert result["error"] == "retrieval unavailable"
    assert any(
        event["stage"] == "retriever" and event["status"] == "failed"
        for event in result["trace"]
    )


def test_unknown_paper_is_a_stage_failure(monkeypatch):
    result = build_research_graph(FakeDb()).invoke(
        base_state(paper_ids=["missing"])
    )
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


def test_missing_human_decision_awaits_review(monkeypatch):
    monkeypatch.setattr(
        stages,
        "retrieve_evidence",
        lambda **kwargs: [evidence()],
    )
    result = build_research_graph(FakeDb()).invoke(
        base_state(user_decision=None)
    )
    assert result["status"] == "awaiting_review"
    assert "reasoner" not in [event["stage"] for event in result["trace"]]
    assert "citation_agent" not in [event["stage"] for event in result["trace"]]
    assert "validator" not in [event["stage"] for event in result["trace"]]


def test_invalid_direct_human_decision_fails(monkeypatch):
    monkeypatch.setattr(
        stages,
        "retrieve_evidence",
        lambda **kwargs: [evidence()],
    )
    result = build_research_graph(FakeDb()).invoke(
        base_state(user_decision="approve")
    )
    assert result["status"] == "failed"
    assert result["error"] == "Invalid human review decision"
    assert "reasoner" not in [event["stage"] for event in result["trace"]]


def test_invalid_api_human_decision_is_rejected():
    with pytest.raises(ValidationError):
        GraphResearchRequest(
            paper_ids=["paper-1"],
            question="Question",
            user_decision="approve",
        )


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
    result = build_research_graph(FakeDb()).invoke(
        base_state(user_decision="continue")
    )
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
    result = build_research_graph(FakeDb()).invoke(
        base_state(user_decision="flag")
    )
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
    result = build_research_graph(FakeDb()).invoke(
        base_state(user_decision="continue")
    )
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
