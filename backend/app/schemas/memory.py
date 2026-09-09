from datetime import datetime

from pydantic import BaseModel, ConfigDict


class MemoryRead(BaseModel):
    id: int
    memory_type: str
    content: str
    source_type: str
    source_id: int | None
    importance_score: float
    status: str
    is_pinned: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
