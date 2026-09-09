import time
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

RuntimeMode = Literal["production", "evaluation"]
ExecutionStatus = Literal["success", "degraded", "failed"]
FailureCategory = Literal["validation", "transient", "permanent", "semantic"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FailureInfo(_StrictModel):
    category: FailureCategory
    code: str = Field(min_length=1, max_length=128)
    retryable: bool


class ExecutionResult(_StrictModel):
    status: ExecutionStatus
    value: Any = None
    failure: FailureInfo | None = None
    attempts: int = Field(default=1, ge=0)
    fallback_used: bool = False
    fallback_from: str | None = None
    fallback_to: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "ExecutionResult":
        if self.status != "failed" and self.attempts == 0:
            raise ValueError("only failed result can have zero attempts")
        if self.status == "success" and self.failure is not None:
            raise ValueError("success result cannot include failure")
        if self.status in {"degraded", "failed"} and self.failure is None:
            raise ValueError("non-success result requires failure")
        if self.fallback_used and not (self.fallback_from and self.fallback_to):
            raise ValueError("fallback lineage is required")
        if not self.fallback_used and (self.fallback_from or self.fallback_to):
            raise ValueError("fallback lineage requires fallback_used")
        return self

    @classmethod
    def success(cls, value: Any, *, attempts: int = 1) -> "ExecutionResult":
        return cls(status="success", value=value, attempts=attempts)

    @classmethod
    def degraded(
        cls,
        value: Any,
        *,
        failure: FailureInfo,
        attempts: int = 1,
        fallback_from: str | None = None,
        fallback_to: str | None = None,
    ) -> "ExecutionResult":
        fallback_used = fallback_from is not None or fallback_to is not None
        return cls(
            status="degraded",
            value=value,
            failure=failure,
            attempts=attempts,
            fallback_used=fallback_used,
            fallback_from=fallback_from,
            fallback_to=fallback_to,
        )

    @classmethod
    def failed(cls, failure: FailureInfo, *, attempts: int = 1) -> "ExecutionResult":
        return cls(status="failed", failure=failure, attempts=attempts)


class ExecutionBudget(_StrictModel):
    total_timeout_seconds: float = Field(default=180.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    max_agent_rounds: int = Field(default=3, gt=0, le=3)
    max_tool_calls: int = Field(default=5, gt=0, le=5)
    max_fallbacks: int = Field(default=1, ge=0)


class BudgetUsage(_StrictModel):
    retries: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    agent_rounds: int = Field(default=0, ge=0)
    fallbacks: int = Field(default=0, ge=0)


class RequestContext(_StrictModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    mode: RuntimeMode = "production"
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    started_at_monotonic: float = Field(default_factory=time.monotonic, ge=0)
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    _usage_lock: Any = PrivateAttr(default_factory=Lock)

    def remaining_seconds(self, now: float) -> float:
        elapsed = max(0.0, now - self.started_at_monotonic)
        return max(0.0, self.budget.total_timeout_seconds - elapsed)

    def reserve_retry(self) -> bool:
        return self._reserve("retries", self.budget.max_retries)

    def reserve_tool_call(self) -> bool:
        return self._reserve("tool_calls", self.budget.max_tool_calls)

    def reserve_agent_round(self) -> bool:
        return self._reserve("agent_rounds", self.budget.max_agent_rounds)

    def reserve_fallback(self) -> bool:
        return self._reserve("fallbacks", self.budget.max_fallbacks)

    def _reserve(self, field: str, limit: int) -> bool:
        with self._usage_lock:
            used = getattr(self.usage, field)
            if used >= limit:
                return False
            setattr(self.usage, field, used + 1)
            return True
