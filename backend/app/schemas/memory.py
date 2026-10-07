from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MemoryType = Literal["user_preference", "research_interest", "research_context", "open_loop"]
MEMORY_TYPES = ("user_preference", "research_interest", "research_context", "open_loop")


class SessionMemorySnapshot(BaseModel):
    """An immutable projection: findings and evidence never cross this boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    goals_and_constraints: str = ""
    current_decisions: str = ""
    open_questions: str = ""
    next_actions: str = ""


def snapshot_session_memory(value: Any) -> SessionMemorySnapshot:
    fields = SessionMemorySnapshot.model_fields
    if isinstance(value, dict):
        return SessionMemorySnapshot(**{key: value.get(key, "") for key in fields})
    return SessionMemorySnapshot(**{key: getattr(value, key, "") for key in fields})


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    memory_type: MemoryType
    content: str = Field(min_length=1, max_length=1000)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class MemoryExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidates: list[MemoryCandidate] = Field(max_length=8)


class MemorySuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    candidate_index: int = Field(ge=0)
    action: Literal["ADD", "UPDATE", "MERGE", "SUPERSEDE", "IGNORE"]
    target_memory_id: int | None = Field(default=None, gt=0)
    merged_content: str | None = Field(default=None, min_length=1, max_length=1000)
    reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_action(self) -> "MemorySuggestion":
        if self.action in {"UPDATE", "MERGE", "SUPERSEDE"} and self.target_memory_id is None:
            raise ValueError("target_required")
        if self.action == "ADD" and self.target_memory_id is not None:
            raise ValueError("add_cannot_target")
        if (self.action == "MERGE") != (self.merged_content is not None):
            raise ValueError("merged_content_only_for_merge")
        return self


class MemoryRead(BaseModel):
    id: int
    memory_type: str
    content: str
    source_type: str
    source_id: int | None
    status: str
    is_pinned: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
