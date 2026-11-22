"""Application-side MCP contracts; no dependency on the runtime package."""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SANDBOX_EXECUTION_EVENT = "sandbox_execution"

class ComputeCode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(default="", max_length=65536)
    inputs: dict[str, Any] = Field(default_factory=dict)
    retrieval_query: str | None = Field(default=None, min_length=1, max_length=1000)
    reuse_inputs: bool = False

    @field_validator("inputs")
    @classmethod
    def bounded_json(cls, value):
        stack = [(value, 0)]
        while stack:
            item, depth = stack.pop()
            if depth > 16:
                raise ValueError("input_depth")
            if type(item) is dict:
                if any(type(k) is not str for k in item):
                    raise ValueError("input_not_json")
                stack.extend((v, depth + 1) for v in item.values())
            elif type(item) is list:
                stack.extend((v, depth + 1) for v in item)
            elif item is not None and type(item) not in (str, int, float, bool):
                raise ValueError("input_not_json")
        if len(json.dumps(value, allow_nan=False).encode()) > 262144:
            raise ValueError("input_limit")
        return value


class ComputeArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    filename: str = Field(min_length=1, max_length=255)
    mime_type: Literal["image/png", "text/csv", "application/json"]
    size_bytes: int = Field(ge=0, le=10485760)


class SandboxExecution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    execution_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    status: Literal["success", "execution_error", "timed_out", "rejected", "infra_error"]
    stdout: str = Field(default="", max_length=65536)
    stderr: str = Field(default="", max_length=65536)
    exit_code: int | None = None
    duration_ms: float = Field(ge=0, allow_inf_nan=False)
    artifacts: list[ComputeArtifact] = Field(default_factory=list, max_length=5)
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    error_category: str | None = Field(default=None, max_length=100)
    cleanup_status: Literal["not_required", "succeeded", "failed"] = "not_required"
    cleanup_error_category: str | None = Field(default=None, max_length=100)
    cleanup_duration_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)


def execution_payloads(events: list[dict]) -> list[SandboxExecution]:
    return [
        SandboxExecution.model_validate(event["execution"])
        for event in events
        if event.get("type") == "sandbox_result" and event.get("execution")
    ]
