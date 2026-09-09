import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from backend.app.core import paths
from backend.app.rag import rerank, vector_store

REAL_GET_BGE_M3_MODEL = vector_store._get_bge_m3_model
REAL_GET_BGE_RERANKER = rerank._get_bge_reranker


def _create_snapshot(
    project_root: Path,
    model_name: str,
    revision: str,
    *,
    weights: str | None = "model.safetensors",
    modified_ns: int | None = None,
) -> Path:
    snapshot = (
        project_root
        / ".hf-cache"
        / "hub"
        / ("models--" + model_name.replace("/", "--"))
        / "snapshots"
        / revision
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    if weights is not None:
        (snapshot / weights).write_bytes(b"weights")
    if modified_ns is not None:
        os.utime(snapshot, ns=(modified_ns, modified_ns))
    return snapshot.resolve()


@pytest.mark.parametrize("weights", ["model.safetensors", "pytorch_model.bin"])
def test_require_hf_model_snapshot_accepts_supported_weight_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    weights: str,
) -> None:
    expected = _create_snapshot(tmp_path, "BAAI/bge-m3", "revision-a", weights=weights)
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)

    resolved = paths.require_hf_model_snapshot("BAAI/bge-m3")

    assert resolved == str(expected)
    assert Path(resolved).is_absolute()


def test_require_hf_model_snapshot_selects_latest_complete_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_name = "BAAI/bge-m3"
    _create_snapshot(tmp_path, model_name, "older", modified_ns=1_000_000_000)
    expected = _create_snapshot(tmp_path, model_name, "latest", modified_ns=2_000_000_000)
    _create_snapshot(
        tmp_path,
        model_name,
        "newer-but-incomplete",
        weights=None,
        modified_ns=3_000_000_000,
    )
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)

    assert paths.require_hf_model_snapshot(model_name) == str(expected)


def test_require_hf_model_snapshot_breaks_timestamp_ties_by_revision_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_name = "BAAI/bge-m3"
    _create_snapshot(tmp_path, model_name, "revision-a", modified_ns=1_000_000_000)
    expected = _create_snapshot(
        tmp_path,
        model_name,
        "revision-z",
        modified_ns=1_000_000_000,
    )
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)

    assert paths.require_hf_model_snapshot(model_name) == str(expected)


def test_require_hf_model_snapshot_rejects_missing_or_incomplete_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_name = "BAAI/bge-reranker-base"
    _create_snapshot(tmp_path, model_name, "incomplete", weights=None)
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)

    with pytest.raises(RuntimeError) as exc_info:
        paths.require_hf_model_snapshot(model_name)

    message = str(exc_info.value)
    assert model_name in message
    assert "setup.ps1" in message


def test_retrieval_model_loaders_receive_only_absolute_local_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_snapshot = _create_snapshot(tmp_path, vector_store.BGE_M3_MODEL_NAME, "embedding")
    reranker_snapshot = _create_snapshot(tmp_path, rerank.BGE_RERANKER_MODEL_NAME, "reranker")
    embedding_calls: list[str] = []
    reranker_calls: list[tuple[str, bool]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_path: str) -> None:
            embedding_calls.append(model_path)
            self.max_seq_length = 8192

    class FakeFlagReranker:
        def __init__(self, model_path: str, *, use_fp16: bool) -> None:
            reranker_calls.append((model_path, use_fp16))

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
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
