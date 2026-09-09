from datetime import datetime

from pydantic import BaseModel, ConfigDict

from .parsing import DocumentChunkDetailRead


class DocumentChunkRead(BaseModel):
    id: int
    document_id: int
    page_id: int | None
    page_number: int
    chunk_index: int
    content: str
    created_at: datetime
    detail: DocumentChunkDetailRead | None = None

    model_config = ConfigDict(from_attributes=True)
