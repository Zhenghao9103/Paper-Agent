from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    served_model: str
    quantization: str
    path: Path


def load_model_registry(
    path: Path, repo_root: Path, model_id: str | None = None
) -> list[ModelSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    router_root = (repo_root / "models" / "router").resolve()
    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for record in payload["models"]:
        current_id = str(record["id"])
        if current_id in seen:
            raise ValueError(f"duplicate model id: {current_id}")
        seen.add(current_id)
        if model_id is not None and current_id != model_id:
            continue
        model_path = (repo_root / record["path"]).resolve()
        if not model_path.is_relative_to(router_root):
            raise ValueError(f"model path is outside models/router: {current_id}")
        if not model_path.is_file():
            raise FileNotFoundError(f"model artifact not found: {current_id}")
        specs.append(
            ModelSpec(
                id=current_id,
                display_name=str(record["display_name"]),
                served_model=str(record["served_model"]),
                quantization=str(record["quantization"]),
                path=model_path,
            )
        )
    if not specs:
        raise ValueError(
            f"model not found in registry: {model_id}"
            if model_id
            else "model registry is empty"
        )
    return specs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LlamaServerManager:
    _FORBIDDEN_EXTRA_ARGS = {
        "-m",
        "--model",
        "--host",
        "--port",
        "--parallel",
        "-np",
    }

    def __init__(
        self,
        executable: Path,
        model: ModelSpec,
        output_dir: Path,
        port: int,
        client: Any | None = None,
        extra_args: tuple[str, ...] = (),
    ) -> None:
        self.executable = executable.resolve()
        self.model = model
        self.output_dir = output_dir
        self.port = port
        for argument in extra_args:
            option = argument.split("=", 1)[0]
            if option in self._FORBIDDEN_EXTRA_ARGS:
                raise ValueError(f"forbidden server argument: {option}")
        self.extra_args = tuple(extra_args)
        self._client = client or httpx.Client(timeout=2.0, trust_env=False)
        self._process: subprocess.Popen | None = None
        self._log_handle: Any | None = None
        self.launch_args: list[str] = []

    def build_command(self) -> list[str]:
        return [
            str(self.executable),
            "-m",
            str(self.model.path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--ctx-size",
            "4096",
            "--threads",
            "8",
            "--parallel",
            "1",
            *self.extra_args,
        ]

    def start(self) -> list[str]:
        if not self.executable.is_file():
            raise FileNotFoundError(
                f"llama-server executable not found: {self.executable}"
            )
        if not self.model.path.is_file():
            raise FileNotFoundError(f"model artifact not found: {self.model.id}")
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("llama-server is already running")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log_handle = (self.output_dir / "server.log").open(
            "w", encoding="utf-8", newline="\n"
        )
        self.launch_args = self.build_command()
        try:
            self._process = subprocess.Popen(
                self.launch_args,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            self._log_handle.close()
            raise
        return list(self.launch_args)

    def wait_ready(self, timeout_s: float) -> None:
        if self._process is None:
            raise RuntimeError("llama-server has not been started")
        deadline = time.monotonic() + timeout_s
        url = f"http://127.0.0.1:{self.port}/v1/models"
        while time.monotonic() < deadline:
            returncode = self._process.poll()
            if returncode is not None:
                raise RuntimeError("llama-server exited before readiness")
            try:
                response = self._client.get(url)
                response.raise_for_status()
            except Exception:
                time.sleep(0.2)
            else:
                return
        raise TimeoutError("llama-server readiness timeout")

    def stop(self) -> None:
        try:
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
        finally:
            if self._log_handle is not None and not self._log_handle.closed:
                self._log_handle.close()
