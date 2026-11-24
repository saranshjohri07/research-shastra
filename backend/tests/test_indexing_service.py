from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.app.db.base import Base
from backend.app.db.models import Chunk, Citation, EvidenceRecord, Paper, ResearchRun
from backend.app.ingestion.document import DocumentPage, ExtractedDocument
from backend.app.services import indexing_service


def test_reindexing_replaces_existing_chunks_and_vectors(monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    indexed_vectors: dict[str, str] = {}

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0] for text in texts]

    def fake_replace_paper_vectors(paper, chunks, embeddings):
        indexed_vectors.clear()
        indexed_vectors.update(
            (chunk.id, chunk.text) for chunk in chunks
        )
        assert len(chunks) == len(embeddings)

    monkeypatch.setattr(indexing_service, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(
        indexing_service,
        "replace_paper_vectors",
        fake_replace_paper_vectors,
    )

    document = ExtractedDocument(
        filename="sample.pdf",
        pages=[DocumentPage(page_number=1, text="ABSTRACT\nEvidence from page one.")],
    )

    with Session(engine) as db:
        paper = Paper(
            id="paper-1",
            title="Sample",
            filename="sample.pdf",
            file_path="sample.pdf",
            page_count=1,
        )
        db.add(paper)
        db.commit()

        first_count = indexing_service.index_paper(db, paper, document)
        first_chunks = db.scalars(
            select(Chunk).where(Chunk.paper_id == paper.id)
        ).all()

        second_count = indexing_service.index_paper(db, paper, document)
        second_chunks = db.scalars(
            select(Chunk).where(Chunk.paper_id == paper.id)
        ).all()

        assert first_count == second_count == 1
        assert len(first_chunks) == len(second_chunks) == 1
        assert len(indexed_vectors) == 1
        assert second_chunks[0].id in indexed_vectors
        assert second_chunks[0].text == indexed_vectors[second_chunks[0].id]
        assert second_chunks[0].page_start == 1
        assert second_chunks[0].section == "ABSTRACT"


def test_reindexing_removes_unreferenced_stale_chunks(monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    monkeypatch.setattr(
        indexing_service,
        "embed_texts",
        lambda texts: [[float(len(text)), 0.0] for text in texts],
    )
    monkeypatch.setattr(
        indexing_service,
        "replace_paper_vectors",
        lambda paper, chunks, embeddings: None,
    )

    from backend.app.retrieval import chroma_store
    monkeypatch.setattr(
        chroma_store,
        "delete_stale_paper_vectors",
        lambda *args, **kwargs: None,
    )

    document = ExtractedDocument(
        filename="sample.pdf",
        pages=[DocumentPage(page_number=1, text="ABSTRACT\nEvidence.")],
    )

    with Session(engine) as db:
        paper = Paper(
            id="paper-stale",
            title="Stale",
            filename="stale.pdf",
            file_path="stale.pdf",
            page_count=1,
        )
        db.add(paper)
        db.flush()
        stale = Chunk(
            id="stale-chunk",
            paper_id=paper.id,
            chunk_index=9,
            text="Removed chunk",
            page_start=1,
            page_end=1,
            vector_status="indexed",
            vector_id="stale-chunk",
        )
        db.add(stale)
        db.commit()

        indexing_service.index_paper(db, paper, document)
        remaining = db.scalars(
            select(Chunk).where(Chunk.paper_id == paper.id)
        ).all()
        assert len(remaining) == 1
        assert remaining[0].chunk_index == 0


def test_reindexing_failure_can_retry(monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    attempts = {"count": 0}

    monkeypatch.setattr(
        indexing_service,
        "embed_texts",
        lambda texts: [[1.0, 0.0] for text in texts],
    )

    def flaky_replace(paper, chunks, embeddings):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("Chroma unavailable")

    monkeypatch.setattr(
        indexing_service,
        "replace_paper_vectors",
        flaky_replace,
    )

    document = ExtractedDocument(
        filename="retry.pdf",
        pages=[DocumentPage(page_number=1, text="ABSTRACT\nRetry evidence.")],
    )

    with Session(engine) as db:
        paper = Paper(
            id="paper-retry",
            title="Retry",
            filename="retry.pdf",
            file_path="retry.pdf",
            page_count=1,
        )
        db.add(paper)
        db.commit()

        try:
            indexing_service.index_paper(db, paper, document)
        except RuntimeError:
            pass

        assert paper.index_status == "failed"
        retry_count = indexing_service.index_paper(db, paper, document)
        assert retry_count == 1
        assert paper.index_status == "indexed"


def test_reindexing_preserves_stale_chunk_referenced_by_evidence(monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    monkeypatch.setattr(
        indexing_service,
        "embed_texts",
        lambda texts: [[1.0, 0.0] for text in texts],
    )
    monkeypatch.setattr(
        indexing_service,
        "replace_paper_vectors",
        lambda paper, chunks, embeddings: None,
    )
    from backend.app.retrieval import chroma_store
    monkeypatch.setattr(
        chroma_store,
        "delete_stale_paper_vectors",
        lambda *args, **kwargs: None,
    )

    document = ExtractedDocument(
        filename="historical.pdf",
        pages=[DocumentPage(page_number=1, text="ABSTRACT\nNew evidence.")],
    )

    with Session(engine) as db:
        paper = Paper(
            id="paper-historical",
            title="Historical",
            filename="historical.pdf",
            file_path="historical.pdf",
            page_count=1,
        )
        db.add(paper)
        db.flush()
        old_chunk = Chunk(
            id="old-referenced-chunk",
            paper_id=paper.id,
            chunk_index=5,
            text="Old evidence",
            page_start=1,
            page_end=1,
            vector_status="indexed",
            vector_id="old-referenced-chunk",
        )
        db.add(old_chunk)
        db.flush()
        run = ResearchRun(
            id="historical-run",
            question="Question",
            status="completed",
        )
        db.add(run)
        db.flush()
        db.add(
            EvidenceRecord(
                id="historical-evidence",
                research_run_id=run.id,
                chunk_id=old_chunk.id,
                paper_id=paper.id,
                evidence_id="E1",
                evidence_rank=1,
                similarity=0.8,
                retrieval_method="vector",
                metadata_snapshot={"text": "Old evidence"},
            )
        )
        db.commit()

        indexing_service.index_paper(db, paper, document)
        preserved = db.get(Chunk, old_chunk.id)
        assert preserved is not None
        assert preserved.vector_status == "stale"