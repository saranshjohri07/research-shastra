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
    collection.delete(where={"paper_id": paper.id})

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
    collection.add(
        ids=[chunk.id for chunk in chunks],
        embeddings=embeddings,
        documents=[chunk.text for chunk in chunks],
        metadatas=metadatas,
    )