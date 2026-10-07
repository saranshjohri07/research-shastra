from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from ..config import get_settings


settings = get_settings()


async def save_pdf(upload: UploadFile) -> dict[str, str | int]:
    if upload.content_type != "application/pdf":
        raise ValueError("Only PDF files are supported")

    original_filename = upload.filename or "document.pdf"
    storage_dir = Path(settings.upload_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    stored_filename = f"{uuid4()}.pdf"
    stored_path = storage_dir / stored_filename

    content = await upload.read()

    if not content:
        raise ValueError("Uploaded PDF is empty")

    stored_path.write_bytes(content)

    return {
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "stored_path": str(stored_path),
        "size_bytes": len(content),
    }
