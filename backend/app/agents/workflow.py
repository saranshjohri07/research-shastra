from typing import Callable

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from .state import ResearchState
from .stages import (
    citation_agent,
    evidence_gatherer,
    human_checkpoint,
    planner,
    reasoner,
    retriever,
    validator,
)


def route_after_evidence(state: ResearchState) -> str:
    if state.get("status") == "failed":
        return END
    if state.get("status") == "abstained":
        return END
    if not state.get("selected_evidence"):
        return END
    return "human_checkpoint"


def route_after_checkpoint(state: ResearchState) -> str:
    if state.get("user_decision") == "flag":
        return END
    if state.get("status") == "failed":
        return END
    if state.get("status") == "running" and state.get("user_decision") == "continue":
        return "reasoner"
    return END


def route_after_reasoner(state: ResearchState) -> str:
    if state.get("status") in ("failed", "abstained"):
        return END
    return "citation_agent"


def build_research_graph(db: Session):
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner)
    graph.add_node(
        "retriever",
        lambda state: retriever(db, state),
    )
    graph.add_node("evidence_gatherer", evidence_gatherer)
    graph.add_node("human_checkpoint", human_checkpoint)
    graph.add_node("reasoner", reasoner)
    graph.add_node("citation_agent", citation_agent)
    graph.add_node("validator", validator)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "evidence_gatherer")
    graph.add_conditional_edges(
        "evidence_gatherer",
        route_after_evidence,
        {
            "human_checkpoint": "human_checkpoint",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "human_checkpoint",
        route_after_checkpoint,
        {
            "reasoner": "reasoner",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "reasoner",
        route_after_reasoner,
        {
            "citation_agent": "citation_agent",
            END: END,
        },
    )
    graph.add_edge("citation_agent", "validator")
    graph.add_edge("validator", END)
    return graph.compile()
