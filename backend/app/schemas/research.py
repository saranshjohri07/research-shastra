from pydantic import BaseModel, Field, field_validator


class ResearchQueryRequest(BaseModel):
    paper_id: str = Field(min_length=1, max_length=36)
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("paper_id", "question")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be blank")
        return value


class EvidenceResponse(BaseModel):
    evidence_id: str
    chunk_id: str
    paper_id: str
    chunk_index: int
    page_start: int | None
    page_end: int | None
    section: str | None
    title: str
    filename: str
    similarity: float
    text: str


class CitationResponse(BaseModel):
    evidence_id: str
    chunk_id: str
    paper_id: str
    title: str
    filename: str
    page_start: int | None
    page_end: int | None
    section: str | None


class ResearchQueryResponse(BaseModel):
    question: str
    answerable: bool
    answer: str | None
    abstention_reason: str | None
    citations: list[CitationResponse]
    evidence: list[EvidenceResponse]