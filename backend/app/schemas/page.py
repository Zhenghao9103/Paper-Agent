from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentPageRead(BaseModel):
    id: int
    document_id: int
    page_number: int
    text: str
    image_path: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
