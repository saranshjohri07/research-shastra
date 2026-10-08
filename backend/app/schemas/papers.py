from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PaperResponse(BaseModel):
    id: str
    title: str
    filename: str
    page_count: int | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
