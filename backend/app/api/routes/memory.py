from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...db.session import get_db
from ...models.chat import TranscriptEvent
from ...models.memory import Memory
from ...schemas.memory import MemoryRead
from ...schemas.transcript import TranscriptEventRead

router = APIRouter(tags=["memory"])


@router.get("/memories", response_model=list[MemoryRead])
def list_memories(db: Session = Depends(get_db)) -> list[Memory]:
    return list(db.scalars(select(Memory).order_by(Memory.created_at.desc())).all())


@router.get("/transcripts/session/{session_id}", response_model=list[TranscriptEventRead])
def list_session_transcripts(
    session_id: int,
    db: Session = Depends(get_db),
) -> list[TranscriptEvent]:
    statement = (
        select(TranscriptEvent)
        .where(TranscriptEvent.session_id == session_id)
        .order_by(TranscriptEvent.created_at)
    )
    return list(db.scalars(statement).all())
