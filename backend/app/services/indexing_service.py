import hashlib
from datetime import timezone, datetime
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db.models import Chunk, Citation, EvidenceRecord, IndexingRun, Paper
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

    settings = get_settings()
    existing_chunks = db.scalars(
        select(Chunk).where(Chunk.paper_id == paper.id)
    ).all()
    existing_by_index = {chunk.chunk_index: chunk for chunk in existing_chunks}

    indexing_run = IndexingRun(
        paper_id=paper.id,
        embedding_model=settings.embedding_model,
        embedding_dimension=len(embeddings[0]) if embeddings else 0,
        chunker_version="chunker-v1",
        status="running",
        started_at=datetime.now(timezone.utc),
    )
    db.add(indexing_run)
    paper.index_status = "indexing"
    paper.embedding_model = settings.embedding_model
    paper.embedding_dimension = len(embeddings[0]) if embeddings else 0
    paper.index_error = None

    stored_chunks: list[Chunk] = []
    seen_indexes: set[int] = set()
    for embedding_index, document_chunk in enumerate(document_chunks):
        chunk_index = document_chunk.chunk_index
        seen_indexes.add(chunk_index)
        chunk = existing_by_index.get(chunk_index)
        if chunk is None:
            chunk = Chunk(
                id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{paper.id}:chunk:{chunk_index}",
                    )
                ),
                paper_id=paper.id,
                chunk_index=chunk_index,
            )
            db.add(chunk)

        chunk.text = document_chunk.text
        chunk.section = document_chunk.section
        chunk.page_start = document_chunk.page_start
        chunk.page_end = document_chunk.page_end
        chunk.text_hash = hashlib.sha256(document_chunk.text.encode("utf-8")).hexdigest()
        chunk.embedding_model = settings.embedding_model
        chunk.embedding_dimension = len(embeddings[embedding_index])
        chunk.vector_status = "pending"
        chunk.vector_id = chunk.id
        stored_chunks.append(chunk)

    for stale_chunk in existing_chunks:
        if stale_chunk.chunk_index not in seen_indexes:
            stale_chunk.vector_status = "stale"

    try:
        db.flush()
        db.commit()
    except Exception:
        db.rollback()
        raise

    try:
        replace_paper_vectors(paper, stored_chunks, embeddings)
        db.add_all(stored_chunks)
        for chunk in stored_chunks:
            chunk.vector_status = "indexed"
            chunk.indexed_at = datetime.now(timezone.utc)
        paper.index_status = "indexed"
        indexing_run.status = "completed"
        indexing_run.finished_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as exc:
        db.rollback()
        paper_row = db.get(Paper, paper.id)
        run_row = db.get(IndexingRun, indexing_run.id)
        if paper_row is not None:
            paper_row.index_status = "failed"
            paper_row.index_error = str(exc)
        if run_row is not None:
            run_row.status = "failed"
            run_row.error = str(exc)
            run_row.finished_at = datetime.now(timezone.utc)
        db.commit()
        raise

    stale_ids = [
        chunk.id
        for chunk in existing_chunks
        if chunk.chunk_index not in seen_indexes
    ]
    if stale_ids:
        from ..retrieval.chroma_store import delete_stale_paper_vectors

        delete_stale_paper_vectors(paper.id, keep_ids={chunk.id for chunk in stored_chunks})
        # Once Chroma reconciliation succeeds, remove only stale chunks that have
        # no immutable evidence/citation references. Historical evidence snapshots
        # protect run traceability even if their source chunk can be deleted.
        for stale_chunk in existing_chunks:
            if stale_chunk.chunk_index not in seen_indexes:
                referenced = db.scalar(
                    select(EvidenceRecord.id)
                    .where(EvidenceRecord.chunk_id == stale_chunk.id)
                    .limit(1)
                )
                cited = referenced is not None or db.scalar(
                    select(Citation.id)
                    .where(Citation.chunk_id == stale_chunk.id)
                    .limit(1)
                ) is not None
                if cited:
                    stale_chunk.vector_status = "stale"
                else:
                    db.delete(stale_chunk)
        db.commit()
    return len(stored_chunks)