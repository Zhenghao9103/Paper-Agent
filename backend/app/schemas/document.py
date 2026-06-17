from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentRead(BaseModel):
    id: int
    title: str
    authors: str | None
    year: int | None
    file_type: str
    file_path: str
    status: str
    source_type: str
    abstract: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
