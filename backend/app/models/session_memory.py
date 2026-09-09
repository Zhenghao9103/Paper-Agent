from datetime import datetime

from sqlalchemy import DateTime, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..db.base import Base


class SessionMemory(Base):
    __tablename__ = "session_memories"

    session_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    goals_and_constraints: Mapped[str] = mapped_column(Text, nullable=False)
    confirmed_findings: Mapped[str] = mapped_column(Text, nullable=False)
    current_decisions: Mapped[str] = mapped_column(Text, nullable=False)
    open_questions: Mapped[str] = mapped_column(Text, nullable=False)
    next_actions: Mapped[str] = mapped_column(Text, nullable=False)
    source_checkpoint_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
