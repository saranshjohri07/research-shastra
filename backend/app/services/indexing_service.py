from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Chunk, Paper
from ..ingestion.chunker import chunk_document
from ..ingestion.document import ExtractedDocument
from ..retrieval.chroma_store import replace_paper_vectors
from ..retrieval.embeddings import embed_texts


def index_paper(
    db: Session,
    paper: Paper,
    document: ExtractedDocument,
) -> int:
    document_chunks = chunk_document(document)
    embeddings = embed_texts([chunk.text for chunk in document_chunks])

    if len(document_chunks) != len(embeddings):
        raise ValueError("Embedding count does not match chunk count")

    existing_chunks = db.scalars(
        select(Chunk).where(Chunk.paper_id == paper.id)
    ).all()
    for existing_chunk in existing_chunks:
        db.delete(existing_chunk)
    db.flush()

    stored_chunks = [
        Chunk(
            id=str(uuid4()),
            paper_id=paper.id,
            chunk_index=document_chunk.chunk_index,
            text=document_chunk.text,
            section=document_chunk.section,
            page_start=document_chunk.page_start,
            page_end=document_chunk.page_end,
        )
        for document_chunk in document_chunks
    ]
    db.add_all(stored_chunks)

    try:
        db.flush()
        db.commit()
    except Exception:
        db.rollback()
        raise

    replace_paper_vectors(paper, stored_chunks, embeddings)
    return len(stored_chunks)