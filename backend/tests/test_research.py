import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.db.base import Base
from backend.app.db.models import Chunk, Paper
from backend.app.retrieval import retriever
from backend.app.retrieval.retriever import RetrievedEvidence
from backend.app.schemas.research import ResearchQueryRequest
from backend.app.services import research_service
from backend.app.services.citation_service import build_citations
from backend.app.services.context_builder import build_evidence_context
from backend.app.services.research_service import answer_research_query


class FakeCollection:
    def __init__(self, chunk_ids: list[str], distances: list[float]):
        self.chunk_ids = chunk_ids
        self.distances = distances
        self.query_arguments = None

    def query(self, **kwargs):
        self.query_arguments = kwargs
        return {"ids": [self.chunk_ids], "distances": [self.distances]}


@pytest.fixture
def database():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add_all(
            [
                Paper(
                    id="paper-a",
                    title="Paper A",
                    filename="paper-a.pdf",
                    file_path="paper-a.pdf",
                    page_count=4,
                ),
                Paper(
                    id="paper-b",
                    title="Paper B",
                    filename="paper-b.pdf",
                    file_path="paper-b.pdf",
                    page_count=4,
                ),
            ]
        )
        db.flush()
        db.add_all(
            [
                Chunk(
                    id="chunk-a",
                    paper_id="paper-a",
                    chunk_index=2,
                    text="QASPER contains research questions.",
                    page_start=2,
                    page_end=3,
                    section="Dataset",
                ),
                Chunk(
                    id="chunk-b",
                    paper_id="paper-b",
                    chunk_index=0,
                    text="This belongs to another paper.",
                    page_start=1,
                    page_end=1,
                    section="Introduction",
                ),
            ]
        )
        db.commit()
        yield db


@pytest.fixture
def evidence() -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id="E1",
        chunk_id="chunk-a",
        paper_id="paper-a",
        text="QASPER contains research questions.",
        chunk_index=2,
        page_start=2,
        page_end=3,
        section="Dataset",
        title="Paper A",
        filename="paper-a.pdf",
        similarity=0.8,
    )


def test_retrieval_is_scoped_and_uses_sqlite_provenance(database, monkeypatch):
    collection = FakeCollection(["chunk-a", "chunk-b"], [0.2, 0.3])
    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr(retriever, "get_chroma_collection", lambda: collection)

    evidence = retriever.retrieve_evidence(
        database,
        paper_id="paper-a",
        question="What is QASPER?",
        top_k=5,
    )

    assert collection.query_arguments["where"] == {"paper_id": "paper-a"}
    assert [item.chunk_id for item in evidence] == ["chunk-a"]
    assert evidence[0].page_start == 2
    assert evidence[0].page_end == 3
    assert evidence[0].section == "Dataset"
    assert evidence[0].chunk_index == 2
    assert evidence[0].title == "Paper A"
    assert evidence[0].similarity == pytest.approx(0.8)


def test_context_builder_formats_stable_evidence_block(evidence):
    context = build_evidence_context([evidence])

    assert "[E1]" in context
    assert "Paper: Paper A" in context
    assert "Pages: 2-3" in context
    assert "Section: Dataset" in context
    assert "Chunk: 2" in context
    assert evidence.text in context


def test_citations_are_built_from_retrieved_metadata(evidence):
    citations = build_citations([evidence])

    assert len(citations) == 1
    assert citations[0].chunk_id == evidence.chunk_id
    assert citations[0].page_start == evidence.page_start
    assert citations[0].page_end == evidence.page_end
    assert citations[0].section == evidence.section
    assert citations[0].filename == evidence.filename


def test_empty_retrieval_abstains_without_generating(database, monkeypatch):
    monkeypatch.setattr(research_service, "retrieve_evidence", lambda **kwargs: [])
    monkeypatch.setattr(
        research_service,
        "generate_grounded_answer",
        lambda *args: pytest.fail("generation must not run without evidence"),
    )

    response = answer_research_query(
        database,
        ResearchQueryRequest(paper_id="paper-a", question="Unknown question"),
    )

    assert response.answerable is False
    assert response.answer is None
    assert response.abstention_reason == "no_evidence"
    assert response.citations == []
    assert response.evidence == []


def test_low_similarity_abstains_and_keeps_evidence(database, monkeypatch, evidence):
    evidence.similarity = 0.1
    monkeypatch.setattr(
        research_service,
        "retrieve_evidence",
        lambda **kwargs: [evidence],
    )
    monkeypatch.setattr(
        research_service,
        "generate_grounded_answer",
        lambda *args: pytest.fail("generation must not run below threshold"),
    )

    response = answer_research_query(
        database,
        ResearchQueryRequest(paper_id="paper-a", question="Unknown question"),
    )

    assert response.answerable is False
    assert response.answer is None
    assert response.abstention_reason == "low_similarity"
    assert response.citations == []
    assert response.evidence[0].chunk_id == evidence.chunk_id


def test_semantic_abstention_includes_reason_and_null_answer(database, monkeypatch, evidence):
    monkeypatch.setattr(
        research_service,
        "retrieve_evidence",
        lambda **kwargs: [evidence],
    )
    monkeypatch.setattr(
        research_service,
        "generate_grounded_answer",
        lambda question, context: {
            "answerable": False,
            "answer": None,
            "reason": "The supplied paper does not contain information needed to answer this question.",
        },
    )

    response = answer_research_query(
        database,
        ResearchQueryRequest(paper_id="paper-a", question="What is the population of Japan in 2026?"),
    )

    assert response.answerable is False
    assert response.answer is None
    assert response.abstention_reason == "The supplied paper does not contain information needed to answer this question."
    assert response.citations == []
    assert response.evidence[0].chunk_id == evidence.chunk_id


def test_generation_exception_is_not_silently_swallowed(database, monkeypatch, evidence):
    monkeypatch.setattr(
        research_service,
        "retrieve_evidence",
        lambda **kwargs: [evidence],
    )
    monkeypatch.setattr(
        research_service,
        "generate_grounded_answer",
        lambda question, context: (_ for _ in ()).throw(RuntimeError("Groq failed")),
    )

    with pytest.raises(RuntimeError, match="Groq failed"):
        answer_research_query(
            database,
            ResearchQueryRequest(paper_id="paper-a", question="What is QASPER?"),
        )


def test_generation_result_does_not_control_citation_metadata(
    database,
    monkeypatch,
    evidence,
):
    monkeypatch.setattr(
        research_service,
        "retrieve_evidence",
        lambda **kwargs: [evidence],
    )
    monkeypatch.setattr(
        research_service,
        "generate_grounded_answer",
        lambda question, context: "Answer supported by [E1].",
    )

    response = answer_research_query(
        database,
        ResearchQueryRequest(paper_id="paper-a", question="What is QASPER?"),
    )

    assert response.answerable is True
    assert response.answer == "Answer supported by [E1]."
    assert response.citations[0].chunk_id == "chunk-a"
    assert response.citations[0].section == "Dataset"


def test_query_request_validation():
    with pytest.raises(ValidationError):
        ResearchQueryRequest(paper_id="paper-a", question="   ")
    with pytest.raises(ValidationError):
        ResearchQueryRequest(paper_id="paper-a", question="Question", top_k=11)
    with pytest.raises(ValidationError):
        ResearchQueryRequest(paper_id="paper-a", question="Question", top_k=0)
    with pytest.raises(ValidationError):
        ResearchQueryRequest(paper_id="paper-a", question="x" * 2001)


def test_unknown_paper_returns_not_found(database):
    with pytest.raises(HTTPException) as error:
        answer_research_query(
            database,
            ResearchQueryRequest(paper_id="missing", question="Question"),
        )

    assert error.value.status_code == 404