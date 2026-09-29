from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import psutil

from eval.router.router_perf.contract import ServerConfig


class RssPeakSampler:
    def __init__(
        self,
        pid: int,
        *,
        reader: Callable[[int], int] | None = None,
        interval_s: float = 0.05,
    ) -> None:
        self.pid = pid
        self._reader = reader or self._read_process_rss
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_bytes = 0

    @staticmethod
    def _read_process_rss(pid: int) -> int:
        return int(psutil.Process(pid).memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return self._peak_bytes / 1024**2

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("RSS sampler is already running")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._peak_bytes = max(self._peak_bytes, self._reader(self.pid))
            except (psutil.Error, OSError):
                return
            self._stop.wait(self._interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


class PerfServerManager:
    def __init__(
        self,
        *,
        executable: Path,
        model_path: Path,
        output_dir: Path,
        port: int,
        config: ServerConfig,
        process_factory: Callable[..., Any] | None = None,
        client: Any | None = None,
    ) -> None:
        self.executable = executable.resolve()
        self.model_path = model_path.resolve()
        self.output_dir = output_dir
        self.port = port
        self.config = config
        self._process_factory = process_factory or subprocess.Popen
        self._client = client or httpx.Client(timeout=2.0, trust_env=False)
        self._process: Any | None = None
        self._log_handle: Any | None = None
        self._rss_sampler: RssPeakSampler | None = None

    @property
    def rss_peak_mb(self) -> float | None:
        return self._rss_sampler.peak_mb if self._rss_sampler is not None else None

    def build_command(self) -> list[str]:
        cache_flag = "--cache-prompt" if self.config.cache_prompt else "--no-cache-prompt"
        return [
            str(self.executable),
            "-m",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.config.ctx_size),
            "--threads",
            str(self.config.threads),
            "--threads-batch",
            str(self.config.threads_batch),
            "--parallel",
            str(self.config.parallel),
            "--cache-type-k",
            self.config.cache_type_k,
            "--cache-type-v",
            self.config.cache_type_v,
            cache_flag,
        ]

    def start(self) -> list[str]:
        if not self.executable.is_file():
            raise FileNotFoundError(f"llama-server executable not found: {self.executable}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model artifact not found: {self.model_path}")
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("llama-server is already running")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log_handle = (self.output_dir / "server.log").open(
            "w", encoding="utf-8", newline="\n"
        )
        command = self.build_command()
        try:
            self._process = self._process_factory(
                command,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
            )
            self._rss_sampler = RssPeakSampler(self._process.pid)
            self._rss_sampler.start()
        except Exception:
            self._log_handle.close()
            raise
        return command

    def wait_ready(self, timeout_s: float) -> float:
        if self._process is None:
            raise RuntimeError("llama-server has not been started")
        started = time.perf_counter()
        deadline = time.monotonic() + timeout_s
        url = f"http://127.0.0.1:{self.port}/v1/models"
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError("llama-server exited before readiness")
            try:
                response = self._client.get(url)
                response.raise_for_status()
            except Exception:
                time.sleep(0.2)
            else:
                return (time.perf_counter() - started) * 1000
        raise TimeoutError("llama-server readiness timeout")

    def stop(self) -> None:
        try:
            if self._rss_sampler is not None:
                self._rss_sampler.stop()
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
