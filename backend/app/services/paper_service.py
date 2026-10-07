from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from ..db.models import Paper


def create_paper_record(
    db: Session,
    filename: str,
    file_path: str,
    page_count: int,
    title: str | None = None,
) -> Paper:
    paper = Paper(
        id=str(uuid4()),
        title=title or Path(filename).stem,
        filename=filename,
        file_path=file_path,
        page_count=page_count,
    )

    db.add(paper)
    db.commit()
    db.refresh(paper)

    return paper
