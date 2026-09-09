from __future__ import annotations

import json
from pathlib import Path

import pytest
from backend.app.cli.download_mineru_figure_model import (
    install_figure_model,
    modelscope_download_adapters,
)
from backend.app.mineru_figure_service import create_app
from fastapi.testclient import TestClient


class _Backend:
    model_name = "local-mineru-vlm"

    def __init__(self, output=None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error

    def ready(self) -> bool:
        return self.error is None

    def describe(self, _image: bytes, **metadata):
        assert metadata["title"] == "Paper"
        if self.error:
            raise self.error
        return self.output


def test_service_health_and_structured_description() -> None:
    backend = _Backend(
        {
            "figure_type": "flowchart",
            "components": ["Input", "Model"],
            "summary": "A processing flow.",
        }
    )
    client = TestClient(create_app(backend))

    assert client.get("/health").json() == {"status": "ready", "model": "local-mineru-vlm"}
    response = client.post(
        "/v1/figure-descriptions",
        files={"image": ("figure.png", b"png", "image/png")},
        data={"title": "Paper", "section": "Method", "caption": "Figure 1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["description"]["figure_type"] == "flowchart"
    assert payload["description"]["visible_text"] == []
    assert payload["latency_ms"] >= 0


def test_service_maps_invalid_model_output_to_stable_502() -> None:
    client = TestClient(create_app(_Backend("not-json")))
    response = client.post(
        "/v1/figure-descriptions",
        files={"image": ("figure.png", b"png", "image/png")},
        data={"title": "Paper"},
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "figure_description_invalid_json"


def test_installer_writes_immutable_manifest_under_root(tmp_path: Path) -> None:
    downloaded = tmp_path / "modelscope" / "models" / "vlm"
    downloaded.mkdir(parents=True)
    calls = []

    def download(model_id: str, revision: str, cache_dir: str) -> str:
        calls.append((model_id, revision, cache_dir))
        return str(downloaded)

    manifest = install_figure_model(
        mineru_root=tmp_path,
        model_id="OpenDataLab/MinerU2.5-Pro-2605-1.2B",
        requested_revision="main",
        resolve_revision=lambda *_args: "b" * 40,
        snapshot_download=download,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["revision"] == "b" * 40
    assert Path(payload["local_path"]).resolve() == downloaded.resolve()
    assert calls[0][1] == "b" * 40
    assert manifest.resolve().is_relative_to(tmp_path.resolve())


def test_installer_rejects_download_path_outside_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-model"
    outside.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="outside MINERU_ROOT"):
        install_figure_model(
            mineru_root=tmp_path,
            model_id="model",
            requested_revision="main",
            resolve_revision=lambda *_args: "c" * 40,
            snapshot_download=lambda *_args, **_kwargs: str(outside),
        )


def test_modelscope_adapters_digest_branch_snapshot_and_use_requested_cache(
    tmp_path: Path,
) -> None:
    downloaded = tmp_path / "modelscope" / "models" / "vlm"
    downloaded.mkdir(parents=True)
    (downloaded / "config.json").write_text('{"model":"vlm"}', encoding="utf-8")
    calls = []

    class Api:
        def get_model_files(self, model_id, revision):
            assert model_id == "OpenDataLab/model"
            assert revision == "master"
            return [{"Path": "config.json", "Size": 15}]

    def download(model_id, *, revision, cache_dir):
        calls.append((model_id, revision, cache_dir))
        return str(downloaded)

    resolve, snapshot = modelscope_download_adapters(api=Api(), download=download)

    assert resolve("OpenDataLab/model", "main") == "master"
    assert snapshot(
        "OpenDataLab/model", revision="master", cache_dir=str(tmp_path / "modelscope")
    ) == str(downloaded)
    assert calls == [
        ("OpenDataLab/model", "master", str(tmp_path / "modelscope"))
    ]

    manifest = install_figure_model(
        mineru_root=tmp_path,
        model_id="OpenDataLab/model",
        requested_revision="master",
        resolve_revision=resolve,
        snapshot_download=snapshot,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(payload["revision"]) == 64
    assert payload["revision"] != "master"
    assert payload["source_revision"] == "master"
