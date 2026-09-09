"""Lifecycle owner for the local llama.cpp Router server.

The root launcher starts the frozen Qwen3-1.7B Router GGUF through
``llama-server`` with the parameters validated in Router Phase 6. A compatible
service that already owns the port is reused; anything else holding the port
is an error. Only a process this object spawned is ever terminated.
"""

from __future__ import annotations

import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from ..core.config import Settings, get_settings

HEALTH_PATH = "/models"  # appended to base_url, which already ends with /v1
_HTTP_TIMEOUT_SECONDS = 2.0


class RouterServerError(RuntimeError):
    """Raised when the managed Router server cannot be brought up."""


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(0.25)
        return client.connect_ex((host, port)) == 0


class ManagedRouterServer:
    """Start, health-check, and stop the local Router server process."""

    def __init__(
        self,
        *,
        executable: Path,
        model_path: Path,
        host: str,
        port: int,
        log_path: Path,
        ready_timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 0.2,
        client_factory: Callable[[], httpx.Client] | None = None,
        spawn: Callable[[list[str]], subprocess.Popen] | None = None,
    ) -> None:
        self.executable = Path(executable)
        self.model_path = Path(model_path)
        self.host = host
        self.port = port
        self.log_path = Path(log_path)
        self.ready_timeout_seconds = ready_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._client_factory = client_factory or self._default_client_factory
        self._spawn = spawn or self._default_spawn
        self._process: subprocess.Popen | None = None
        self._log_handle: Any = None
        self._owns_process = False

    @classmethod
    def from_settings(
        cls,
        *,
        log_dir: Path,
        get_settings_fn: Callable[[], Settings] | None = None,
    ) -> ManagedRouterServer:
        settings = (get_settings_fn or get_settings)()
        if not settings.router_managed:
            raise RouterServerError("Router management is disabled (ROUTER_MANAGED=false).")
        if not settings.router_server_path or not settings.router_gguf_path:
            raise RouterServerError(
                "Managed Router requires ROUTER_SERVER_PATH and ROUTER_GGUF_PATH."
            )
        return cls(
            executable=Path(settings.router_server_path),
            model_path=Path(settings.router_gguf_path),
            host=settings.router_host,
            port=settings.router_port,
            log_path=Path(log_dir) / "router-server.log",
            ready_timeout_seconds=settings.router_ready_timeout_seconds,
        )

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def build_command(self) -> list[str]:
        """Fixed Phase 6 inference parameters for the Q4_K_M Router GGUF."""

        return [
            str(self.executable),
            "-m",
            str(self.model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--ctx-size",
            "4096",
            "--threads",
            "4",
            "--threads-batch",
            "4",
            "--parallel",
            "1",
            "--cache-prompt",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
        ]

    def start(self) -> str:
        """Ensure the Router endpoint answers and return its base URL."""

        client = self._client_factory()
        try:
            if self._probe(client):
                self._owns_process = False
                return self.base_url()
            if _port_in_use(self.host, self.port):
                raise RouterServerError(
                    f"router port {self.port} is occupied by an incompatible service"
                )
        finally:
            client.close()

        self._validate_artifacts()
        self._process = self._spawn(self.build_command())
        self._owns_process = True
        try:
            self.wait_ready()
        except Exception:
            self.stop()
            raise
        return self.base_url()

    def wait_ready(self) -> None:
        if self._process is None:
            raise RouterServerError("router server has not been started")
        client = self._client_factory()
        deadline = time.monotonic() + self.ready_timeout_seconds
        try:
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RouterServerError(
                        "router server exited before becoming ready "
                        f"(exit code {self._process.poll()})"
                    )
                if self._probe(client):
                    return
                time.sleep(self.poll_interval_seconds)
        finally:
            client.close()
        raise TimeoutError(
            f"router server readiness timeout after {self.ready_timeout_seconds}s"
        )

    def stop(self) -> None:
        process, self._process = self._process, None
        owned, self._owns_process = self._owns_process, False
        if owned and process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._log_handle is not None and not self._log_handle.closed:
            self._log_handle.close()

    def _probe(self, client: Any) -> bool:
        try:
            response = client.get(f"{self.base_url()}{HEALTH_PATH}")
            return getattr(response, "status_code", 0) == 200
        except Exception:
            return False

    def _validate_artifacts(self) -> None:
        if not self.executable.is_file():
            raise RouterServerError(
                f"llama-server executable not found: {self.executable}"
            )
        if not self.model_path.is_file():
            raise RouterServerError(f"router model not found: {self.model_path}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("w", encoding="utf-8", newline="\n")

    def _default_client_factory(self) -> httpx.Client:
        return httpx.Client(timeout=_HTTP_TIMEOUT_SECONDS, trust_env=False)

    def _default_spawn(self, command: list[str]) -> subprocess.Popen:
        return subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            command,
            stdout=self._log_handle or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
