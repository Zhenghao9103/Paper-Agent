import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from backend.app.core import paths
from backend.app.rag import rerank, vector_store

REAL_GET_BGE_M3_MODEL = vector_store._get_bge_m3_model
REAL_GET_BGE_RERANKER = rerank._get_bge_reranker


def _repo_dir(hub_root: Path, model_name: str) -> Path:
    return hub_root / ("models--" + model_name.replace("/", "--"))


def _create_snapshot(
    hub_root: Path,
    model_name: str,
    revision: str,
    *,
    weights: str | None = "model.safetensors",
    modified_ns: int | None = None,
) -> Path:
    snapshot = _repo_dir(hub_root, model_name) / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    if weights is not None:
        (snapshot / weights).write_bytes(b"weights")
    if modified_ns is not None:
        os.utime(snapshot, ns=(modified_ns, modified_ns))
    return snapshot.resolve()


def _write_main_ref(hub_root: Path, model_name: str, revision: str) -> None:
    main_ref = _repo_dir(hub_root, model_name) / "refs" / "main"
    main_ref.parent.mkdir(parents=True, exist_ok=True)
    main_ref.write_text(revision + "\n", encoding="utf-8")


def _use_hub_cache(monkeypatch: pytest.MonkeyPatch, hub_root: Path) -> None:
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub_root))


@pytest.mark.parametrize("weights", ["model.safetensors", "pytorch_model.bin"])
def test_require_hf_model_snapshot_accepts_supported_weight_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    weights: str,
) -> None:
    hub_root = tmp_path / "custom-hub"
    expected = _create_snapshot(hub_root, "BAAI/bge-m3", "revision-a", weights=weights)
    _use_hub_cache(monkeypatch, hub_root)

    resolved = paths.require_hf_model_snapshot("BAAI/bge-m3")

    assert resolved == str(expected)
    assert Path(resolved).is_absolute()


def test_require_hf_model_snapshot_prefers_main_ref_over_newer_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_name = "BAAI/bge-m3"
    hub_root = tmp_path / "custom-hub"
    expected = _create_snapshot(hub_root, model_name, "main-revision", modified_ns=1_000_000_000)
    _create_snapshot(hub_root, model_name, "stale-revision", modified_ns=3_000_000_000)
    _write_main_ref(hub_root, model_name, "main-revision")
    _use_hub_cache(monkeypatch, hub_root)

    assert paths.require_hf_model_snapshot(model_name) == str(expected)


def test_require_hf_model_snapshot_rejects_multiple_snapshots_without_main_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_name = "BAAI/bge-m3"
    hub_root = tmp_path / "custom-hub"
    _create_snapshot(hub_root, model_name, "revision-a")
    _create_snapshot(hub_root, model_name, "revision-b")
    _use_hub_cache(monkeypatch, hub_root)

    with pytest.raises(RuntimeError) as exc_info:
        paths.require_hf_model_snapshot(model_name)

    message = str(exc_info.value)
    assert "refs/main" in message
    assert "setup.ps1" in message


def test_require_hf_model_snapshot_does_not_fallback_from_incomplete_main_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_name = "BAAI/bge-m3"
    hub_root = tmp_path / "custom-hub"
    _create_snapshot(hub_root, model_name, "main-revision", weights=None)
    _create_snapshot(hub_root, model_name, "alternate-complete")
    _write_main_ref(hub_root, model_name, "main-revision")
    _use_hub_cache(monkeypatch, hub_root)

    with pytest.raises(RuntimeError, match="main-revision"):
        paths.require_hf_model_snapshot(model_name)


@pytest.mark.parametrize(
    "model_name",
    [
        "BAAI",
        "BAAI/bge/m3",
        "./bge-m3",
        "BAAI/..",
        "BAAI/.",
        "BAAI\\bge-m3/repo",
        "BAAI/repo\\escape",
        "BAAI/repo name",
    ],
)
def test_require_hf_model_snapshot_rejects_invalid_model_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
) -> None:
    _use_hub_cache(monkeypatch, tmp_path / "custom-hub")

    with pytest.raises(ValueError, match="owner/repo"):
        paths.require_hf_model_snapshot(model_name)


def test_require_hf_model_snapshot_rejects_main_ref_path_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_name = "BAAI/bge-m3"
    hub_root = tmp_path / "custom-hub"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "config.json").write_text("{}", encoding="utf-8")
    (outside / "model.safetensors").write_bytes(b"weights")
    _write_main_ref(hub_root, model_name, "../../../outside")
    _use_hub_cache(monkeypatch, hub_root)

    with pytest.raises(ValueError, match="revision"):
        paths.require_hf_model_snapshot(model_name)


def test_require_hf_model_snapshot_rejects_missing_or_incomplete_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_name = "BAAI/bge-reranker-base"
    hub_root = tmp_path / "custom-hub"
    _create_snapshot(hub_root, model_name, "incomplete", weights=None)
    _use_hub_cache(monkeypatch, hub_root)

    with pytest.raises(RuntimeError) as exc_info:
        paths.require_hf_model_snapshot(model_name)

    message = str(exc_info.value)
    assert model_name in message
    assert "setup.ps1" in message


def test_retrieval_model_loaders_receive_only_absolute_local_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub_root = tmp_path / "custom-hub"
    embedding_snapshot = _create_snapshot(
        hub_root, vector_store.BGE_M3_MODEL_NAME, "embedding"
    )
    reranker_snapshot = _create_snapshot(
        hub_root, rerank.BGE_RERANKER_MODEL_NAME, "reranker"
    )
    embedding_calls: list[str] = []
    reranker_calls: list[tuple[str, bool]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_path: str) -> None:
            embedding_calls.append(model_path)
            self.max_seq_length = 8192

    class FakeFlagReranker:
        def __init__(self, model_path: str, *, use_fp16: bool) -> None:
            reranker_calls.append((model_path, use_fp16))

    _use_hub_cache(monkeypatch, hub_root)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setitem(
        sys.modules,
        "FlagEmbedding",
        SimpleNamespace(FlagReranker=FakeFlagReranker),
    )
    REAL_GET_BGE_M3_MODEL.cache_clear()
    REAL_GET_BGE_RERANKER.cache_clear()
    try:
        embedding_model = REAL_GET_BGE_M3_MODEL()
        REAL_GET_BGE_RERANKER()
    finally:
        REAL_GET_BGE_M3_MODEL.cache_clear()
        REAL_GET_BGE_RERANKER.cache_clear()

    assert embedding_calls == [str(embedding_snapshot)]
    assert reranker_calls == [(str(reranker_snapshot), False)]
    assert Path(embedding_calls[0]).is_absolute()
    assert Path(reranker_calls[0][0]).is_absolute()
    assert embedding_model.max_seq_length == 512
