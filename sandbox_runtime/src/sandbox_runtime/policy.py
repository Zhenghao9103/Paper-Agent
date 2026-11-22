import json
import math
from dataclasses import dataclass

from .schemas import ExecutePythonRequest


class PolicyViolation(ValueError):
    pass


@dataclass(frozen=True)
class SandboxPolicy:
    image: str = "paper-agent-sandbox:0.1.0"
    timeout_seconds: float = 30.0
    memory_bytes: int = 1024 * 1024 * 1024
    cpus: float = 1.0
    pids: int = 64
    workspace_bytes: int = 64 * 1024 * 1024
    tmp_bytes: int = 32 * 1024 * 1024
    max_code_bytes: int = 64 * 1024
    max_input_bytes: int = 256 * 1024
    max_json_depth: int = 16
    max_log_bytes: int = 64 * 1024
    max_artifacts: int = 5
    max_artifact_bytes: int = 10 * 1024 * 1024
    max_total_artifact_bytes: int = 25 * 1024 * 1024
    max_archive_bytes: int = 32 * 1024 * 1024
    artifact_ttl_seconds: int = 86400
    artifact_store_bytes: int = 256 * 1024 * 1024

    def __post_init__(self):
        for name, value in vars(self).items():
            if name != "image" and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
            if name not in {"image", "cpus", "timeout_seconds"} and type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        if int(self.cpus * 1_000_000_000) < 1:
            raise ValueError("cpus must be representable as positive NanoCpus")

    def validate(self, request: ExecutePythonRequest) -> None:
        if not request.code.strip() or len(request.code.encode()) > self.max_code_bytes:
            raise PolicyViolation("code_limit")
        try:
            stack = [(request.inputs, 0)]
            while stack:
                value, depth = stack.pop()
                if depth > self.max_json_depth:
                    raise PolicyViolation("input_depth")
                if type(value) is dict:
                    if any(type(key) is not str for key in value):
                        raise PolicyViolation("input_not_json")
                    stack.extend((v, depth + 1) for v in value.values())
                elif type(value) is list:
                    stack.extend((v, depth + 1) for v in value)
                elif value is not None and type(value) not in (str, int, float, bool):
                    raise PolicyViolation("input_not_json")
            encoded = json.dumps(request.inputs, allow_nan=False).encode()
        except (ValueError, TypeError, RecursionError) as exc:
            if isinstance(exc, PolicyViolation):
                raise
            raise PolicyViolation("input_not_json") from exc
        if len(encoded) > self.max_input_bytes:
            raise PolicyViolation("input_limit")
