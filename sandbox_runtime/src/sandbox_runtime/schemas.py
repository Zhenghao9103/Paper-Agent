from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ExecutePythonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(min_length=1)
    inputs: dict[str, Any] = Field(default_factory=dict)


class ArtifactInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_id: str
    filename: str
    mime_type: str
    size_bytes: int = Field(ge=0)


class ExecutePythonResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    execution_id: str
    status: Literal["success", "execution_error", "timed_out", "rejected", "infra_error"]
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: float = Field(ge=0)
    artifacts: list[ArtifactInfo] = Field(default_factory=list)
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    error_category: str | None = None
    cleanup_status: Literal["not_required", "succeeded", "failed"] = "not_required"
    cleanup_error_category: str | None = None
    cleanup_duration_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
