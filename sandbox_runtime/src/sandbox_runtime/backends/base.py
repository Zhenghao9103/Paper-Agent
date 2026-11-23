from dataclasses import dataclass
from threading import Event
from typing import Protocol

from ..policy import SandboxPolicy
from ..schemas import ExecutePythonRequest


class BackendFailure(RuntimeError):
    """A safe, fixed infrastructure error category."""


class ExecutionCancelled(RuntimeError):
    pass


@dataclass
class BackendOutput:
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    oom_killed: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    artifact_archive: bytes | None = None
    cleanup_status: str = "not_required"
    cleanup_error_category: str | None = None
    cleanup_duration_ms: float = 0.0


class SandboxBackend(Protocol):
    def run(
        self,
        request: ExecutePythonRequest,
        *,
        execution_id: str,
        policy: SandboxPolicy,
        deadline: float,
        cancelled: Event | None = None,
    ) -> BackendOutput: ...
