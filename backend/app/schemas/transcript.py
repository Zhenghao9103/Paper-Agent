from datetime import datetime

from pydantic import BaseModel, ConfigDict


class TranscriptEventRead(BaseModel):
    id: int
    session_id: int | None
    event_type: str
    payload: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
