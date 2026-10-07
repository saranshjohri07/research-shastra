from dataclasses import dataclass


@dataclass
class DocumentPage:
    page_number: int
    text: str


@dataclass
class ExtractedDocument:
    filename: str
    pages: list[DocumentPage]