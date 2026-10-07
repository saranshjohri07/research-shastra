import re
from dataclasses import dataclass

from .document import ExtractedDocument


@dataclass
class DocumentChunk:
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    section: str | None = None


SECTION_PATTERN = re.compile(
    r"^(?:"
    r"(?:\d+(?:\.\d+)*)?\s*"
    r"(?:abstract|introduction|background|related work|"
    r"literature review|method|methods|methodology|"
    r"approach|experiments?|results?|discussion|"
    r"conclusion|limitations?|references|appendix)"
    r")$",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_section_heading(text: str) -> bool:
    text = text.strip()

    if not text or len(text) > 100:
        return False

    if SECTION_PATTERN.match(text):
        return True

    if (
        text.isupper()
        and 2 <= len(text.split()) <= 8
        and not text.endswith(".")
        and len(re.findall(r"\d+(?:\.\d+)?%?", text)) <= 1
    ):
        return True

    return False


def split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        piece = text[start:end].strip()

        if piece:
            pieces.append(piece)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return pieces


def chunk_document(
    document: ExtractedDocument,
    max_chars: int = 1500,
    overlap: int = 200,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []

    current_text = ""
    current_section: str | None = None
    current_page_start: int | None = None
    current_page_end: int | None = None

    for page in document.pages:
        text = clean_text(page.text)

        if not text:
            continue

        lines = [line.strip() for line in text.splitlines() if line.strip()]

        for line in lines:
            if is_section_heading(line):
                if current_text:
                    for piece in split_text(current_text, max_chars, overlap):
                        chunks.append(
                            DocumentChunk(
                                chunk_index=len(chunks),
                                text=piece,
                                page_start=current_page_start or page.page_number,
                                page_end=current_page_end or page.page_number,
                                section=current_section,
                            )
                        )

                    current_text = ""

                current_section = line
                current_page_start = page.page_number
                current_page_end = page.page_number
                continue

            if not current_text:
                current_page_start = page.page_number

            candidate = f"{current_text}\n{line}".strip()

            if len(candidate) <= max_chars:
                current_text = candidate
                current_page_end = page.page_number
            else:
                if current_text:
                    chunks.append(
                        DocumentChunk(
                            chunk_index=len(chunks),
                            text=current_text,
                            page_start=current_page_start or page.page_number,
                            page_end=current_page_end or page.page_number,
                            section=current_section,
                        )
                    )

                overlap_text = current_text[-overlap:] if current_text else ""
                current_text = f"{overlap_text}\n{line}".strip()
                current_page_start = page.page_number
                current_page_end = page.page_number

    if current_text:
        chunks.append(
            DocumentChunk(
                chunk_index=len(chunks),
                text=current_text,
                page_start=current_page_start or 1,
                page_end=current_page_end or 1,
                section=current_section,
            )
        )

    return chunks
