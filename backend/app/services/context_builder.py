from ..retrieval.retriever import RetrievedEvidence


def build_evidence_context(evidence: list[RetrievedEvidence]) -> str:
    blocks: list[str] = []
    for item in evidence:
        if item.page_start == item.page_end:
            pages = str(item.page_start or "Unknown")
        else:
            pages = f"{item.page_start or 'Unknown'}-{item.page_end or 'Unknown'}"

        blocks.append(
            f"[{item.evidence_id}]\n"
            f"Paper: {item.title}\n"
            f"Filename: {item.filename}\n"
            f"Pages: {pages}\n"
            f"Section: {item.section or 'Not specified'}\n"
            f"Chunk: {item.chunk_index}\n\n"
            f"{item.text}"
        )

    return "\n\n".join(blocks)