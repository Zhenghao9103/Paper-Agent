import subprocess
from pathlib import Path

import pytest
from backend.app.services import router_server as router_server_module
from backend.app.services.router_server import ManagedRouterServer, RouterServerError


class FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class FakeClient:
    def __init__(self, status_code: int = 200, fail: bool = False) -> None:
        self.status_code = status_code
        self.fail = fail
        self.get_calls = 0

    def get(self, url: str) -> FakeResponse:
        self.get_calls += 1
        if self.fail:
            raise RuntimeError("connection refused")
        return FakeResponse(self.status_code)

    def close(self) -> None:
        pass


class FakeProcess:
    def __init__(self, exit_after_polls: int | None = None) -> None:
        self.polls = 0
        self.exit_after_polls = exit_after_polls
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self) -> int | None:
        self.polls += 1
        if self.exit_after_polls is not None and self.polls > self.exit_after_polls:
            return 1
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        return 0


def make_server(
    tmp_path: Path, *, client=None, spawn=None, monkeypatch=None, **kwargs
) -> ManagedRouterServer:
    defaults = dict(
        executable=tmp_path / "llama-server.exe",
        model_path=tmp_path / "router.gguf",
        host="127.0.0.1",
        port=8089,
        log_path=tmp_path / "logs" / "router-server.log",
        ready_timeout_seconds=0.5,
        poll_interval_seconds=0.01,
    )
    defaults.update(kwargs)
    server = ManagedRouterServer(**defaults)
    server.executable.touch()
    server.model_path.touch()
    if client is not None:
        server._client_factory = lambda: client
    if spawn is not None:
        server._spawn = spawn
    if monkeypatch is not None:
        # A live llama-server on this host must not influence unit tests of
        # the spawn path; the occupancy test patches this itself.
        monkeypatch.setattr(
            router_server_module, "_port_in_use", lambda host, port: False
        )
    return server


def test_build_command_uses_frozen_phase06_parameters(tmp_path: Path) -> None:
    server = make_server(tmp_path)

    command = server.build_command()

    assert command[:2] == [str(tmp_path / "llama-server.exe"), "-m"]
    assert str(tmp_path / "router.gguf") in command
    for index, value in enumerate(command):
        if value == "--ctx-size":
            assert command[index + 1] == "4096"
        if value == "--threads":
            assert command[index + 1] == "4"
        if value == "--threads-batch":
            assert command[index + 1] == "4"
        if value == "--parallel":
            assert command[index + 1] == "1"
    assert "--cache-prompt" in command
    assert command[command.index("--cache-type-k") + 1] == "f16"
    assert command[command.index("--cache-type-v") + 1] == "f16"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8089"


def test_start_reuses_compatible_running_service_without_spawning(tmp_path: Path) -> None:
    client = FakeClient(status_code=200)
    spawned: list[list[str]] = []

    def spawn(command: list[str]) -> subprocess.Popen:
        spawned.append(command)
        return FakeProcess()  # pragma: no cover - must never run

    server = make_server(tmp_path, client=client, spawn=spawn)

    url = server.start()

    assert url == "http://127.0.0.1:8089/v1"
    assert spawned == []
    assert server._owns_process is False
    # stop() must not terminate a process we do not own.
    server.stop()


def test_start_reports_incompatible_port_occupant(tmp_path: Path, monkeypatch) -> None:
    client = FakeClient(fail=True)
    monkeypatch.setattr(router_server_module, "_port_in_use", lambda host, port: True)
    spawned: list[list[str]] = []

    def spawn(command: list[str]) -> subprocess.Popen:
        spawned.append(command)
        return FakeProcess()  # pragma: no cover - must never run

    server = make_server(tmp_path, client=client, spawn=spawn)

    with pytest.raises(RouterServerError, match="occupied"):
        server.start()

    assert spawned == []


def test_start_detects_process_that_exits_before_ready(tmp_path: Path, monkeypatch) -> None:
    client = FakeClient(fail=True)
    process = FakeProcess(exit_after_polls=1)
    spawned: list[list[str]] = []

    def spawn(command: list[str]) -> subprocess.Popen:
        spawned.append(command)
        return process

    server = make_server(tmp_path, client=client, spawn=spawn, monkeypatch=monkeypatch)

    with pytest.raises(RouterServerError, match="exited"):
        server.start()

    assert spawned and spawned[0][0] == str(tmp_path / "llama-server.exe")
    # The fake process already exited (poll() returns a code), so stop() must
    # not try to terminate it, and ownership must be released either way.
    assert server._owns_process is False
    assert server._process is None


def test_start_times_out_when_service_never_becomes_ready(tmp_path: Path, monkeypatch) -> None:
    client = FakeClient(fail=True)
    process = FakeProcess()
    server = make_server(
        tmp_path,
        client=client,
        spawn=lambda command: process,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(TimeoutError):
        server.start()

    server.stop()
    assert process.terminated or process.killed


def test_start_waits_until_probe_succeeds(tmp_path: Path, monkeypatch) -> None:
    class SlowClient(FakeClient):
        def get(self, url: str) -> FakeResponse:
            if self.get_calls < 3:
                self.get_calls += 1
                raise RuntimeError("not ready yet")
            return FakeResponse(200)

    client = SlowClient()
    server = make_server(
        tmp_path,
        client=client,
        spawn=lambda command: FakeProcess(),
        ready_timeout_seconds=2.0,
        monkeypatch=monkeypatch,
    )

    url = server.start()

    assert url == "http://127.0.0.1:8089/v1"
    assert server._owns_process is True
    server.stop()


def test_stop_only_terminates_self_owned_process(tmp_path: Path) -> None:
    process = FakeProcess()
    server = make_server(tmp_path, spawn=lambda command: process)
    server._process = process
    server._owns_process = False

    server.stop()

    assert not process.terminated
    assert not process.killed

    server._owns_process = True
    server._process = process
    server.stop()

    assert process.terminated or process.killed


def test_from_settings_reads_managed_configuration(monkeypatch, tmp_path: Path) -> None:
    from backend.app.core import config

    server = ManagedRouterServer.from_settings(
        log_dir=tmp_path,
        get_settings_fn=lambda: config.Settings(
            _env_file=None,
            router_managed=True,
            router_server_path=".tools/llama.cpp/llama-server.exe",
            router_gguf_path="models/router/router.gguf",
            router_host="127.0.0.1",
            router_port=8089,
            router_ready_timeout_seconds=45.0,
        ),
    )

    assert server.executable.name == "llama-server.exe"
    assert server.model_path.name == "router.gguf"
    assert server.port == 8089
    assert server.ready_timeout_seconds == 45.0
    assert server.log_path == tmp_path / "router-server.log"


def test_from_settings_rejects_disabled_management(monkeypatch, tmp_path: Path) -> None:
    from backend.app.core import config

    with pytest.raises(RouterServerError, match="disabled"):
        ManagedRouterServer.from_settings(
            log_dir=tmp_path,
            get_settings_fn=lambda: config.Settings(
                _env_file=None,
                router_managed=False,
                router_server_path=".tools/llama.cpp/llama-server.exe",
                router_gguf_path="models/router/router.gguf",
            ),
        )
