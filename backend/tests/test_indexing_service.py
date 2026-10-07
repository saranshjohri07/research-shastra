from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.app.db.base import Base
from backend.app.db.models import Chunk, Paper
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