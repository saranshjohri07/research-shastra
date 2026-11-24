from functools import lru_cache

import chromadb

from ..config import get_settings
from ..db.models import Chunk, Paper


COLLECTION_NAME = "researchshastra_chunks"


@lru_cache(maxsize=1)
def get_chroma_collection():
    client = chromadb.PersistentClient(path=get_settings().chroma_path)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def replace_paper_vectors(
    paper: Paper,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have exactly one embedding")

    collection = get_chroma_collection()
    keep_ids = {chunk.id for chunk in chunks}
    existing = collection.get(
        where={"paper_id": paper.id},
        include=[],
    )
    stale_ids = [
        vector_id
        for vector_id in existing.get("ids", [])
        if vector_id not in keep_ids
    ]
    if stale_ids:
        collection.delete(ids=stale_ids)

    if not chunks:
        return

    metadatas = [
        {
            "paper_id": paper.id,
            "chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "page_start": chunk.page_start if chunk.page_start is not None else 0,
            "page_end": chunk.page_end if chunk.page_end is not None else 0,
            "section": chunk.section or "",
            "title": paper.title,
            "filename": paper.filename,
        }
        for chunk in chunks
    ]
    collection.upsert(
        ids=[chunk.id for chunk in chunks],
        embeddings=embeddings,
        documents=[chunk.text for chunk in chunks],
        metadatas=metadatas,
    )


def delete_stale_paper_vectors(paper_id: str, keep_ids: set[str]) -> None:
    collection = get_chroma_collection()
    existing = collection.get(where={"paper_id": paper_id}, include=[])
    stale_ids = [
        vector_id
        for vector_id in existing.get("ids", [])
        if vector_id not in keep_ids
    ]
    if stale_ids:
        collection.delete(ids=stale_ids)