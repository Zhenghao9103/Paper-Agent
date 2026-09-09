from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")


class FigureRuntimeError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class FigureModelManifest:
    model_id: str
    revision: str
    local_path: Path


def build_mineru_environment(mineru_root: Path | str) -> dict[str, str]:
    root = Path(mineru_root).resolve()
    return {
        "MODELSCOPE_CACHE": str(root / "modelscope"),
        "MINERU_TOOLS_CONFIG_JSON": str(root / "mineru.json"),
        "MINERU_MODEL_SOURCE": "local",
        "HF_HOME": str(root / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(root / "huggingface" / "hub"),
        "PADDLE_HOME": str(root / "paddle"),
    }


def load_figure_model_manifest(
    manifest_path: Path | str, mineru_root: Path | str
) -> FigureModelManifest:
    path = Path(manifest_path)
    root = Path(mineru_root).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_id = str(payload["model_id"]).strip()
        revision = str(payload["revision"]).strip()
        local_path = Path(payload["local_path"]).resolve()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FigureRuntimeError("mineru_vlm_unavailable") from exc
    if (
        not model_id
        or not _COMMIT_RE.fullmatch(revision)
        or not local_path.is_dir()
        or not local_path.is_relative_to(root)
    ):
        raise FigureRuntimeError("mineru_vlm_unavailable")
    return FigureModelManifest(model_id, revision, local_path)


class MinerUFigureServiceManager:
    def __init__(
        self,
        *,
        mineru_root: Path | str,
        manifest_path: Path | str,
        service_url: str = "http://127.0.0.1:8002",
        python_executable: str | None = None,
        process_factory: Callable[..., Any] = subprocess.Popen,
        health_check: Callable[[], bool] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        startup_timeout_seconds: float = 180.0,
    ) -> None:
        self.mineru_root = Path(mineru_root).resolve()
        self.manifest_path = Path(manifest_path)
        self.service_url = service_url.rstrip("/")
        self.python_executable = python_executable or sys.executable
        self.process_factory = process_factory
        self.health_check = health_check or self._default_health_check
        self.sleep = sleep
        self.monotonic = monotonic
        self.startup_timeout_seconds = startup_timeout_seconds
        self._process: Any | None = None
        self._token: str | None = None
        self._idle_timer: threading.Timer | None = None

    @property
    def owns_process(self) -> bool:
        return self._process is not None

    def _default_health_check(self) -> bool:
        try:
            response = httpx.get(f"{self.service_url}/health", timeout=1.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def ensure_started(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
        if self.health_check():
            return
        manifest = load_figure_model_manifest(self.manifest_path, self.mineru_root)
        runtime_dir = self.mineru_root / "runtime"
        logs_dir = self.mineru_root / "logs"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        environment = os.environ.copy()
        environment.update(build_mineru_environment(self.mineru_root))
        environment.update(
            {
                "MINERU_FIGURE_MANIFEST": str(self.manifest_path.resolve()),
                "MINERU_FIGURE_OWNERSHIP_TOKEN": token,
            }
        )
        command = [
            self.python_executable,
            "-m",
            "backend.app.mineru_figure_service",
            "--host",
            "127.0.0.1",
            "--port",
            self.service_url.rsplit(":", 1)[-1],
        ]
        process = self.process_factory(
            command,
            env=environment,
            cwd=str(Path(__file__).resolve().parents[3]),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._process = process
        self._token = token
        metadata = {
            "pid": int(process.pid),
            "token": token,
            "model_id": manifest.model_id,
            "revision": manifest.revision,
        }
        (runtime_dir / "figure-service.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        started = self.monotonic()
        while not self.health_check():
            if (
                process.poll() is not None
                or self.monotonic() - started >= self.startup_timeout_seconds
            ):
                self.close()
                raise FigureRuntimeError("figure_service_startup_timeout")
            self.sleep(0.2)

    def schedule_idle_close(self, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            self.close()
            return
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        self._idle_timer = threading.Timer(timeout_seconds, self.close)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def close(self) -> None:
        timer, self._idle_timer = self._idle_timer, None
        if timer is not None and timer is not threading.current_thread():
            timer.cancel()
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except (subprocess.TimeoutExpired, TimeoutError):
                    process.kill()
                    process.wait(timeout=5)
        finally:
            metadata_path = self.mineru_root / "runtime" / "figure-service.json"
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            if payload.get("token") == self._token:
                metadata_path.unlink(missing_ok=True)
            self._token = None

    def __enter__(self) -> MinerUFigureServiceManager:
        self.ensure_started()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
