from ..retrieval.retriever import RetrievedEvidence
from ..schemas.research import CitationResponse


def build_citations(
    evidence: list[RetrievedEvidence],
) -> list[CitationResponse]:
    return [
        CitationResponse(
            evidence_id=item.evidence_id,
            chunk_id=item.chunk_id,
            paper_id=item.paper_id,
            title=item.title,
            filename=item.filename,
            page_start=item.page_start,
            page_end=item.page_end,
            section=item.section,
        )
        for item in evidence
    ]