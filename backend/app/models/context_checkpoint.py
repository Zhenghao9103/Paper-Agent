from datetime import datetime

from sqlalchemy import DateTime, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..db.base import Base


class ContextCheckpoint(Base):
    __tablename__ = "context_checkpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    first_kept_message_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )
    tokens_before: Mapped[int] = mapped_column(Integer, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    details: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
