from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Paper
from ..db.session import get_db
from ..ingestion.pdf_extractor import extract_pdf
from ..schemas.papers import PaperResponse
from ..services.indexing_service import index_paper
from ..services.paper_service import create_paper_record
from ..services.paper_storage import save_pdf


router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("/", response_model=list[PaperResponse])
def list_papers(db: Session = Depends(get_db)) -> list[Paper]:
    return list(
        db.scalars(select(Paper).order_by(Paper.created_at.desc())).all()
    )


@router.post("/upload")
async def upload_paper(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    stored = await save_pdf(file)

    pdf_path = Path(stored["stored_path"])

    # We temporarily use a placeholder ID during extraction.
    # The final Paper ID is created immediately afterward.
    document = extract_pdf(pdf_path)

    paper = create_paper_record(
        db=db,
        filename=stored["original_filename"],
        file_path=stored["stored_path"],
        page_count=len(document.pages),
    )
    chunk_count = index_paper(db=db, paper=paper, document=document)

    return {
        "id": paper.id,
        "title": paper.title,
        "filename": paper.filename,
        "page_count": paper.page_count,
        "chunk_count": chunk_count,
        "indexing_status": "complete",
        "file_path": paper.file_path,
    }