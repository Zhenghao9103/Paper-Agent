from datetime import datetime

from sqlalchemy import JSON, String, Boolean, DateTime, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from ..db.base import Base


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    memory_type: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(Text, nullable=False, default="chat")
    source_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="active", index=True)
    is_pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class MemorySyncOperation(Base):
    __tablename__ = "memory_sync_operations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    source_session_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_checkpoint_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default="pending", server_default="pending", index=True
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        server_default=func.current_timestamp(),
    )


class ProceduralSkill(Base):
    __tablename__ = "procedural_skills"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[int] = mapped_column(default=1)
    active: Mapped[bool] = mapped_column(default=True)
    condition: Mapped[str] = mapped_column(Text)
    steps: Mapped[list[str]] = mapped_column(JSON)
    constraints: Mapped[list[str]] = mapped_column(JSON)
    source_event_id: Mapped[int]
    source_session_id: Mapped[int]


class ProceduralFeedback(Base):
    __tablename__ = "procedural_feedback"

    message_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    status: Mapped[str] = mapped_column(String, default="pending", index=True)
    trace_id: Mapped[int | None]
    strategy: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    attempts: Mapped[int] = mapped_column(default=0)
    reason: Mapped[str | None]
    skill_id: Mapped[str | None]
    version: Mapped[int | None]
    action: Mapped[str | None]


class ProceduralSync(Base):
    __tablename__ = "procedural_sync"

    id: Mapped[int] = mapped_column(primary_key=True)
    skill_id: Mapped[str]
    version: Mapped[int]
    attempts: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(default="pending", index=True)
    last_error: Mapped[str | None]
