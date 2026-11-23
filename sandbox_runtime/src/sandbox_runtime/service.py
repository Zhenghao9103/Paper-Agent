import logging
import threading
import time
from pathlib import Path
from uuid import uuid4

from .artifacts import ArtifactStore, default_artifact_root
from .backends.base import BackendFailure, ExecutionCancelled, SandboxBackend
from .policy import PolicyViolation, SandboxPolicy
from .schemas import ExecutePythonRequest, ExecutePythonResult

logger = logging.getLogger(__name__)


class SandboxService:
    def __init__(
        self,
        backend: SandboxBackend,
        *,
        policy: SandboxPolicy | None = None,
        artifact_root: Path | None = None,
    ):
        self.backend = backend
        self.policy = policy or SandboxPolicy()
        self.artifact_root = artifact_root
        self.artifacts = ArtifactStore(artifact_root or default_artifact_root(), self.policy)
        configure_owner = getattr(backend, "configure_owner", None)
        if configure_owner:
            configure_owner(self.artifacts.root)
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._closed = False
        self._cancelled = threading.Event()

    def close(self) -> None:
        with self._state_lock:
            self._closed = True
            self._cancelled.set()
        close = getattr(self.backend, "close", None)
        if close:
            close()

    def cancel(self, cancelled: threading.Event) -> None:
        cancelled.set()
        cancel = getattr(self.backend, "cancel", None)
        if cancel:
            cancel(cancelled)

    def execute(
        self, request: ExecutePythonRequest, *, cancelled: threading.Event | None = None
    ) -> ExecutePythonResult:
        started = time.monotonic()
        result = ExecutePythonResult(execution_id=uuid4().hex, status="infra_error", duration_ms=0)
        if not self._lock.acquire(blocking=False):
            result.status, result.error_category = "rejected", "runtime_busy"
            return result
        outcome = None
        try:
            with self._state_lock:
                if self._closed:
                    raise BackendFailure("runtime_closed")
                self._cancelled = cancelled if cancelled is not None else threading.Event()
            if self._cancelled.is_set():
                raise ExecutionCancelled()
            self.policy.validate(request)
            output = self.backend.run(
                request,
                execution_id=result.execution_id,
                policy=self.policy,
                deadline=started + self.policy.timeout_seconds,
                cancelled=self._cancelled,
            )
            outcome = output
            if self._cancelled.is_set():
                raise ExecutionCancelled()
            result.stdout, result.stderr = output.stdout, output.stderr
            result.exit_code = output.exit_code
            result.stdout_truncated = output.stdout_truncated
            result.stderr_truncated = output.stderr_truncated
            if output.timed_out:
                result.status, result.error_category = "timed_out", "deadline_exceeded"
            elif output.oom_killed:
                result.status, result.error_category = "execution_error", "out_of_memory"
            elif output.stdout_truncated or output.stderr_truncated:
                result.status, result.error_category = "execution_error", "output_limit"
            elif output.exit_code == 0:
                result.status = "success"
                if output.artifact_archive:
                    result.artifacts = self.artifacts.collect(output.artifact_archive)
            else:
                result.status, result.error_category = "execution_error", "python_error"
        except PolicyViolation as exc:
            outcome = outcome or exc
            result.status, result.error_category = "rejected", str(exc)
        except TimeoutError as exc:
            outcome = outcome or exc
            result.status, result.error_category = "timed_out", "deadline_exceeded"
        except ExecutionCancelled as exc:
            outcome = outcome or exc
            result.status, result.error_category = "execution_error", "cancelled"
        except BackendFailure as exc:
            outcome = outcome or exc
            result.status, result.error_category = "infra_error", str(exc)
        except Exception as exc:
            outcome = outcome or exc
            result.status, result.error_category = "infra_error", "backend_unavailable"
        finally:
            result.cleanup_status = getattr(outcome, "cleanup_status", "not_required")
            result.cleanup_error_category = getattr(outcome, "cleanup_error_category", None)
            result.cleanup_duration_ms = getattr(outcome, "cleanup_duration_ms", 0.0)
            self._lock.release()
            result.duration_ms = (time.monotonic() - started) * 1000
            logger.info(
                "execution_id=%s status=%s duration_ms=%.1f category=%s cleanup_status=%s",
                result.execution_id,
                result.status,
                result.duration_ms,
                result.error_category,
                result.cleanup_status,
            )
        return result
