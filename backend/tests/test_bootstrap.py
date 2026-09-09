from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from scripts.bootstrap import (
    INSTALL_STAGES,
    Resource,
    ResourceInstallError,
    ResourceLockError,
    extract_verified_zip,
    generate_env,
    generate_mineru_config,
    install_archive,
    install_file,
    install_modelscope_snapshot,
    install_snapshot,
    load_resource_lock,
    sha256_tree,
)


def _write_lock(path: Path, resource: dict[str, object]) -> None:
    path.write_text(
        json.dumps({"schema_version": 1, "resources": [resource]}),
        encoding="utf-8",
    )


def test_lock_rejects_mutable_model_revision(tmp_path: Path) -> None:
    lock = tmp_path / "resources.lock.json"
    _write_lock(
        lock,
        {
            "id": "bge-m3",
            "kind": "huggingface_snapshot",
            "repo_id": "BAAI/bge-m3",
            "revision": "main",
            "destination": ".hf-cache/hub/models--BAAI--bge-m3",
            "tree_sha256": "a" * 64,
        },
    )
    with pytest.raises(ResourceLockError, match="immutable revision"):
        load_resource_lock(lock)


def test_lock_rejects_destination_outside_repository(tmp_path: Path) -> None:
    lock = tmp_path / "resources.lock.json"
    _write_lock(
        lock,
        {
            "id": "archive",
            "kind": "http_archive",
            "url": "https://example.invalid/archive.zip",
            "destination": "../escape",
            "sha256": "a" * 64,
        },
    )
    with pytest.raises(ResourceLockError, match="destination"):
        load_resource_lock(lock)


def test_release_lock_contains_exact_required_resources() -> None:
    lock = load_resource_lock(Path(__file__).resolve().parents[2] / "resources.lock.json")
    assert {resource.id for resource in lock.resources} == {
        "router-q4-gguf",
        "llama-cpp-windows-cpu",
        "bge-m3",
        "bge-reranker-base",
        "mineru-pdf-extract-kit",
        "mineru-figure-vlm",
        "tiktoken-cl100k-base",
        "tiktoken-o200k-base",
    }
    destinations = {resource.id: resource.destination.as_posix() for resource in lock.resources}
    assert destinations["tiktoken-cl100k-base"] == (
        ".cache/tiktoken/9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
    )
    snapshots = {
        resource.id: resource
        for resource in lock.resources
        if resource.kind in {"huggingface_snapshot", "modelscope_snapshot"}
    }
    assert all(resource.tree_sha256 for resource in snapshots.values())


def test_verified_file_is_reused_without_download(tmp_path: Path) -> None:
    target = tmp_path / ".cache" / "asset.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"verified")
    resource = Resource(
        id="asset",
        kind="http_file",
        destination=Path(".cache/asset.bin"),
        url="https://example.invalid/asset.bin",
        sha256=hashlib.sha256(b"verified").hexdigest(),
    )
    called = False

    def download(_url: str, _target: Path) -> None:
        nonlocal called
        called = True

    assert install_file(resource, tmp_path, download=download) == target
    assert called is False


def test_archive_rejects_member_outside_destination(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("../escape.exe", b"bad")
    with pytest.raises(ResourceInstallError, match="outside destination"):
        extract_verified_zip(archive, tmp_path / ".tools" / "llama.cpp")
    assert not (tmp_path / "escape.exe").exists()


def test_snapshot_installer_uses_locked_revision(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        snapshot = tmp_path / "downloaded"
        snapshot.mkdir()
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        return str(snapshot)

    resource = Resource(
        id="bge-m3",
        kind="huggingface_snapshot",
        repo_id="BAAI/bge-m3",
        revision="a" * 40,
        destination=Path(".hf-cache/hub/models--BAAI--bge-m3"),
    )
    installed = install_snapshot(resource, tmp_path, snapshot_download=download)
    assert installed == tmp_path / "downloaded"
    assert calls == [
        {
            "repo_id": "BAAI/bge-m3",
            "revision": "a" * 40,
            "cache_dir": str((tmp_path / ".hf-cache" / "hub").resolve()),
            "local_files_only": False,
        }
    ]


def test_verified_snapshot_is_reused_without_network(tmp_path: Path) -> None:
    revision = "a" * 40
    snapshot = tmp_path / ".hf-cache" / "hub" / "models--BAAI--bge-m3" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    resource = Resource(
        id="bge-m3",
        kind="huggingface_snapshot",
        repo_id="BAAI/bge-m3",
        revision=revision,
        destination=Path(".hf-cache/hub/models--BAAI--bge-m3"),
        tree_sha256=sha256_tree(snapshot),
    )

    def download(**_kwargs: object) -> str:
        raise AssertionError("verified snapshot must not access network")

    assert install_snapshot(resource, tmp_path, snapshot_download=download) == snapshot


def test_verified_modelscope_snapshot_is_reused_without_network(tmp_path: Path) -> None:
    revision = "b" * 40
    snapshot = (
        tmp_path
        / ".mineru"
        / "modelscope"
        / "models"
        / "Owner--Repo"
        / "snapshots"
        / revision
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    resource = Resource(
        id="mineru",
        kind="modelscope_snapshot",
        repo_id="Owner/Repo",
        revision=revision,
        destination=Path(".mineru/modelscope/models/Owner--Repo"),
        tree_sha256=sha256_tree(snapshot),
    )

    def download(**_kwargs: object) -> str:
        raise AssertionError("verified snapshot must not access network")

    assert install_modelscope_snapshot(resource, tmp_path, snapshot_download=download) == snapshot


def test_verified_llama_tree_is_reused_without_download(tmp_path: Path) -> None:
    destination = tmp_path / ".tools" / "llama.cpp"
    destination.mkdir(parents=True)
    (destination / "llama-server.exe").write_bytes(b"exe")
    (destination / "llama-server-impl.dll").write_bytes(b"dll")
    resource = Resource(
        id="llama",
        kind="http_archive",
        destination=Path(".tools/llama.cpp"),
        url="https://example.invalid/llama.zip",
        sha256="a" * 64,
        tree_sha256=sha256_tree(destination),
    )

    def download(_url: str, _target: Path) -> None:
        raise AssertionError("verified archive tree must not be downloaded")

    assert install_archive(resource, tmp_path, download=download) == destination


def _read_env(path: Path) -> dict[str, str]:
    return {
        key: value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.lstrip().startswith("#") and "=" in line
        for key, value in [line.split("=", 1)]
    }


def test_generate_env_enables_complete_local_stack(tmp_path: Path) -> None:
    generate_env(tmp_path)
    values = _read_env(tmp_path / ".env")
    assert values["ROUTER_MANAGED"] == "true"
    assert values["MINERU_FORMULA_ENABLED"] == "true"
    assert values["MINERU_FIGURE_ENABLED"] == "true"
    assert values["HF_HOME"] == ".hf-cache"
    assert values["TIKTOKEN_CACHE_DIR"] == ".cache/tiktoken"


def test_generate_env_preserves_provider_credentials(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# private values\nAGENT_API_KEY=private-agent\nJUDGE_API_KEY=private-judge\n",
        encoding="utf-8",
    )
    generate_env(tmp_path)
    values = _read_env(env)
    assert values["AGENT_API_KEY"] == "private-agent"
    assert values["JUDGE_API_KEY"] == "private-judge"
    assert "# private values" in env.read_text(encoding="utf-8")


def test_generate_mineru_config_uses_target_clone_only(tmp_path: Path) -> None:
    target = generate_mineru_config(tmp_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    pipeline = Path(payload["models-dir"]["pipeline"])
    assert pipeline.is_relative_to((tmp_path / ".mineru").resolve())
    assert "bucket_info" not in payload
    assert "api_key" not in target.read_text(encoding="utf-8").casefold()
    manifest = json.loads(
        (tmp_path / ".mineru" / "figure-model.json").read_text(encoding="utf-8")
    )
    assert Path(manifest["local_path"]).is_relative_to((tmp_path / ".mineru").resolve())


def test_generated_configuration_is_idempotent(tmp_path: Path) -> None:
    env = generate_env(tmp_path)
    mineru = generate_mineru_config(tmp_path)
    first = (env.read_bytes(), mineru.read_bytes())
    generate_env(tmp_path)
    generate_mineru_config(tmp_path)
    assert (env.read_bytes(), mineru.read_bytes()) == first


def test_stage_order_is_complete() -> None:
    assert INSTALL_STAGES == (
        "preflight",
        "venv",
        "dependencies",
        "git-lfs",
        "llama-cpp",
        "retrieval-models",
        "mineru-primary",
        "mineru-figure",
        "tiktoken",
        "configuration",
        "post-install-check",
    )
