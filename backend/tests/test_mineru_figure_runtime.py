from __future__ import annotations

import json
from pathlib import Path

import pytest
from backend.app.services.mineru_figure_runtime import (
    FigureRuntimeError,
    MinerUFigureServiceManager,
    build_mineru_environment,
    load_figure_model_manifest,
)


class _FakeProcess:
    pid = 4321

    def __init__(self) -> None:
        self.terminated = False
        self.waited = False

    def poll(self):
        return None if not self.terminated else 0

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None) -> None:
        del timeout
        self.waited = True

    def kill(self) -> None:
        self.terminated = True


def _manifest(tmp_path: Path) -> Path:
    model = tmp_path / "modelscope" / "models" / "mineru-vlm"
    model.mkdir(parents=True)
    path = tmp_path / "figure-model.json"
    path.write_text(
        json.dumps(
            {
                "model_id": "OpenDataLab/MinerU2.5-Pro-2605-1.2B",
                "revision": "a" * 40,
                "local_path": str(model),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_environment_pins_all_model_caches_below_mineru_root(tmp_path: Path) -> None:
    env = build_mineru_environment(tmp_path)

    for key in (
        "MODELSCOPE_CACHE",
        "MINERU_TOOLS_CONFIG_JSON",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "PADDLE_HOME",
    ):
        assert Path(env[key]).resolve().is_relative_to(tmp_path.resolve())
    assert env["MINERU_MODEL_SOURCE"] == "local"


def test_manifest_requires_immutable_revision_and_local_path(tmp_path: Path) -> None:
    manifest = load_figure_model_manifest(_manifest(tmp_path), tmp_path)
    assert manifest.revision == "a" * 40
    assert manifest.local_path.is_dir()

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"model_id": "x", "revision": "master", "local_path": str(tmp_path)}),
        encoding="utf-8",
    )
    with pytest.raises(FigureRuntimeError) as exc:
        load_figure_model_manifest(bad, tmp_path)
    assert exc.value.code == "mineru_vlm_unavailable"


def test_missing_manifest_does_not_start_subprocess(tmp_path: Path) -> None:
    started = False

    def process_factory(*_args, **_kwargs):
        nonlocal started
        started = True

    manager = MinerUFigureServiceManager(
        mineru_root=tmp_path,
        manifest_path=tmp_path / "missing.json",
        process_factory=process_factory,
        health_check=lambda: False,
    )
    with pytest.raises(FigureRuntimeError) as exc:
        manager.ensure_started()
    assert exc.value.code == "mineru_vlm_unavailable"
    assert started is False


def test_manager_reuses_external_service_without_stopping_it(tmp_path: Path) -> None:
    manager = MinerUFigureServiceManager(
        mineru_root=tmp_path,
        manifest_path=tmp_path / "missing.json",
        health_check=lambda: True,
        process_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )

    manager.ensure_started()
    manager.close()

    assert manager.owns_process is False


def test_manager_starts_waits_and_stops_only_owned_process(tmp_path: Path) -> None:
    process = _FakeProcess()
    health_values = iter([False, False, True])
    calls: list[tuple[list[str], dict]] = []

    def process_factory(command, **kwargs):
        calls.append((command, kwargs))
        return process

    manager = MinerUFigureServiceManager(
        mineru_root=tmp_path,
        manifest_path=_manifest(tmp_path),
        python_executable="figure-python",
        process_factory=process_factory,
        health_check=lambda: next(health_values),
        sleep=lambda _seconds: None,
        startup_timeout_seconds=5,
    )

    manager.ensure_started()
    manager.close()

    assert manager.owns_process is False
    assert process.terminated is True
    assert process.waited is True
    command, kwargs = calls[0]
    assert command[:3] == ["figure-python", "-m", "backend.app.mineru_figure_service"]
    assert kwargs["env"]["MINERU_MODEL_SOURCE"] == "local"


def test_startup_timeout_terminates_owned_process(tmp_path: Path) -> None:
    process = _FakeProcess()
    manager = MinerUFigureServiceManager(
        mineru_root=tmp_path,
        manifest_path=_manifest(tmp_path),
        process_factory=lambda *_args, **_kwargs: process,
        health_check=lambda: False,
        sleep=lambda _seconds: None,
        monotonic=iter([0.0, 1.0, 3.0]).__next__,
        startup_timeout_seconds=2,
    )

    with pytest.raises(FigureRuntimeError) as exc:
        manager.ensure_started()
    assert exc.value.code == "figure_service_startup_timeout"
    assert process.terminated is True


def test_stale_pid_metadata_is_never_used_to_kill_process(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "figure-service.json").write_text(
        json.dumps({"pid": 99999, "token": "stale"}), encoding="utf-8"
    )
    manager = MinerUFigureServiceManager(
        mineru_root=tmp_path,
        manifest_path=tmp_path / "missing.json",
        health_check=lambda: True,
    )

    manager.ensure_started()
    manager.close()

    assert (runtime / "figure-service.json").exists()
