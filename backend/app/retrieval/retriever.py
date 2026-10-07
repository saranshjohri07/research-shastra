from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Chunk, Paper
from .chroma_store import get_chroma_collection
from .embeddings import embed_texts


@dataclass
class RetrievedEvidence:
    evidence_id: str
    chunk_id: str
    paper_id: str
    text: str
    chunk_index: int
    page_start: int | None
    page_end: int | None
    section: str | None
    title: str
    filename: str
    similarity: float


def retrieve_evidence(
    db: Session,
    paper_id: str,
    question: str,
    top_k: int = 5,
) -> list[RetrievedEvidence]:
    question_embedding = embed_texts([question])[0]
    results = get_chroma_collection().query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        where={"paper_id": paper_id},
        include=["distances"],
    )
    chunk_ids = results["ids"][0]
    distances = results["distances"][0]
    if not chunk_ids:
        return []

    rows = db.execute(
        select(Chunk, Paper)
        .join(Paper, Paper.id == Chunk.paper_id)
        .where(Chunk.id.in_(chunk_ids), Chunk.paper_id == paper_id)
    ).all()
    rows_by_id = {chunk.id: (chunk, paper) for chunk, paper in rows}

    evidence: list[RetrievedEvidence] = []
    for rank, (chunk_id, distance) in enumerate(zip(chunk_ids, distances), start=1):
        row = rows_by_id.get(chunk_id)
        if row is None:
            continue

        chunk, paper = row
        evidence.append(
            RetrievedEvidence(
                evidence_id=f"E{rank}",
                chunk_id=chunk.id,
                paper_id=paper.id,
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section=chunk.section,
                title=paper.title,
                filename=paper.filename,
                similarity=max(-1.0, min(1.0, 1.0 - float(distance))),
            )
        )

    return evidence