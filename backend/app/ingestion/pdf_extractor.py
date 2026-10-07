from pathlib import Path

from pypdf import PdfReader

from .document import DocumentPage, ExtractedDocument


def extract_pdf(pdf_path: Path) -> ExtractedDocument:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    reader = PdfReader(str(pdf_path))

    pages: list[DocumentPage] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append(
            DocumentPage(
                page_number=page_number,
                text=text.strip(),
            )
        )

    return ExtractedDocument(
    filename=pdf_path.name,
    pages=pages,
)