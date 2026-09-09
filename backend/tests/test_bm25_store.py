import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import backend.app.rag.tokenization as tokenization
import jieba
import pytest
from backend.app.db.base import Base
from backend.app.models.chunk import DocumentChunk
from backend.app.models.document import Document
from backend.app.rag.bm25_store import (
    clear_bm25_index,
    delete_document_index,
    ensure_bm25_schema,
    index_chunk,
    rebuild_bm25_index,
    search_bm25,
)
from backend.app.rag.tokenization import tokenize_mixed
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def _memory_engine():
    engine = create_engine("sqlite:///:memory:")
    ensure_bm25_schema(engine)
    return engine


def test_tokenize_mixed_does_not_mutate_jieba_global_tokenizer() -> None:
    phrase = "谱聚类"
    before = jieba.lcut(phrase)

    tokenize_mixed(phrase)

    assert jieba.lcut(phrase) == before


def test_tokenize_mixed_shares_one_private_cjk_tokenizer() -> None:
    primary = tokenization._cjk_tokenizer()
    worker_count = 16
    barrier = Barrier(worker_count)

    def get_tokenizer_identity() -> int:
        identity = id(tokenization._cjk_tokenizer())
        barrier.wait()
        return identity

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(get_tokenizer_identity) for _ in range(worker_count)]

    assert {future.result() for future in futures} == {id(primary)}


def test_tokenize_mixed_preserves_technical_tokens_and_adds_stems() -> None:
    tokens = tokenize_mixed(
        "谱聚类使用 SpecNet-2 models，在 CIFAR-10 上 running experiments。"
    )

    assert "谱聚类" in tokens
    assert "specnet-2" in tokens
    assert "cifar-10" in tokens
    assert "run" in tokens


def test_tokenize_mixed_normalizes_deduplicates_and_preserves_domain_terms() -> None:
    tokens = tokenize_mixed("最优传输 ＢＧＥ＿Ｍ３ BGE_M3 studies studies 3.14")

    assert "最优传输" in tokens
    assert "bge_m3" in tokens
    assert "studi" in tokens
    assert "3.14" in tokens
    assert len(tokens) == len(set(tokens))


def test_tokenize_mixed_can_preserve_index_term_frequency() -> None:
    tokens = tokenize_mixed("running running", deduplicate=False)

    assert tokens == ["running", "run", "running", "run"]


def test_tokenize_mixed_is_stable_under_concurrent_calls() -> None:
    phrases = (
        "models running experiments relational conditional",
        "clustering studies optimized generalized embeddings",
        "transport comparisons evaluated reproducibly",
        "spectral networks learning representations repeatedly",
    )
    expected = {phrase: tokenize_mixed(phrase) for phrase in phrases}
    worker_count = 16
    barrier = Barrier(worker_count)

    def tokenize_repeatedly(worker_index: int) -> list[list[str]]:
        phrase = phrases[worker_index % len(phrases)]
        barrier.wait()
        return [tokenize_mixed(phrase) for _ in range(200)]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(tokenize_repeatedly, index)
            for index in range(worker_count)
        ]

    for index, future in enumerate(futures):
        phrase = phrases[index % len(phrases)]
        assert future.result() == [expected[phrase]] * 200


def test_search_bm25_filters_document_and_weights_title() -> None:
    engine = _memory_engine()
    with Session(engine) as db:
        index_chunk(
            db,
            chunk_id=2,
            document_id=1,
            title="Optimal Transport",
            content="baseline",
        )
        index_chunk(
            db,
            chunk_id=1,
            document_id=1,
            title="Baseline",
            content="optimal transport",
        )
        index_chunk(
            db,
            chunk_id=3,
            document_id=2,
            title="Optimal Transport",
            content="optimal transport",
        )
        db.commit()

        hits = search_bm25(db, ["optimal", "transport"], document_id=1, limit=5)

    assert [hit.chunk_id for hit in hits] == [2, 1]
    assert hits[0].raw_score < hits[1].raw_score


def test_search_bm25_preserves_term_frequency_for_ranking() -> None:
    engine = _memory_engine()
    with Session(engine) as db:
        index_chunk(
            db,
            chunk_id=9,
            document_id=1,
            title="",
            content="central central central central fillers",
        )
        index_chunk(
            db,
            chunk_id=1,
            document_id=1,
            title="",
            content="central unrelated unrelated unrelated unrelated",
        )
        db.commit()

        hits = search_bm25(db, ["central"], document_id=1, limit=5)

    assert [hit.chunk_id for hit in hits] == [9, 1]
    assert hits[0].raw_score < hits[1].raw_score


def test_search_bm25_empty_terms_returns_no_hits() -> None:
    engine = _memory_engine()
    with Session(engine) as db:
        assert search_bm25(db, ["", "   "], document_id=None, limit=5) == []


def test_search_bm25_uses_stable_numeric_tie_order_and_one_based_ranks() -> None:
    engine = _memory_engine()
    with Session(engine) as db:
        index_chunk(db, chunk_id=8, document_id=1, title="", content="same term")
        index_chunk(db, chunk_id=3, document_id=1, title="", content="same term")
        db.commit()

        hits = search_bm25(db, ["same"], document_id=None, limit=5)

    assert [(hit.chunk_id, hit.rank) for hit in hits] == [(3, 1), (8, 2)]


def test_index_chunk_upsert_replaces_stale_tokens() -> None:
    engine = _memory_engine()
    with Session(engine) as db:
        index_chunk(db, chunk_id=1, document_id=1, title="", content="obsolete")
        index_chunk(db, chunk_id=1, document_id=1, title="", content="replacement")
        db.commit()

        assert search_bm25(db, ["obsolete"], document_id=None, limit=5) == []
        assert [
            hit.chunk_id
            for hit in search_bm25(db, ["replacement"], document_id=None, limit=5)
        ] == [1]


def test_delete_document_index_and_clear_return_expected_results() -> None:
    engine = _memory_engine()
    with Session(engine) as db:
        index_chunk(db, chunk_id=1, document_id=1, title="", content="shared")
        index_chunk(db, chunk_id=2, document_id=2, title="", content="shared")
        delete_document_index(db, 1)

        assert [
            hit.chunk_id
            for hit in search_bm25(db, ["shared"], document_id=None, limit=5)
        ] == [2]
        assert clear_bm25_index(db) == 1
        assert search_bm25(db, ["shared"], document_id=None, limit=5) == []


def test_rebuild_bm25_index_uses_document_titles_and_chunk_content() -> None:
    engine = _memory_engine()
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        document = Document(
            title="Optimal Transport",
            file_type="pdf",
            file_path="paper.pdf",
        )
        db.add(document)
        db.flush()
        db.add_all(
            [
                DocumentChunk(
                    document_id=document.id,
                    page_number=1,
                    chunk_index=0,
                    content="first baseline",
                ),
                DocumentChunk(
                    document_id=document.id,
                    page_number=2,
                    chunk_index=1,
                    content="second baseline",
                ),
            ]
        )
        db.flush()
        index_chunk(db, chunk_id=999, document_id=999, title="stale", content="stale")

        assert rebuild_bm25_index(db) == 2
        assert search_bm25(db, ["stale"], document_id=None, limit=5) == []
        hits = search_bm25(db, ["optimal"], document_id=document.id, limit=5)

    assert [hit.chunk_id for hit in hits] == [1, 2]


@pytest.mark.parametrize(
    ("document_id", "limit"),
    [(None, 0), (None, -1), (0, 5), (-1, 5)],
)
def test_search_bm25_rejects_invalid_limits_and_document_ids(
    document_id: int | None,
    limit: int,
) -> None:
    engine = _memory_engine()
    with Session(engine) as db, pytest.raises(ValueError):
        search_bm25(db, ["term"], document_id=document_id, limit=limit)


@pytest.mark.parametrize(
    ("working_directory", "config_path"),
    [("backend", "alembic.ini"), ("repository", "backend/alembic.ini")],
)
def test_alembic_upgrade_is_idempotent_on_isolated_sqlite_database(
    tmp_path: Path,
    working_directory: str,
    config_path: str,
) -> None:
    backend_dir = Path(__file__).resolve().parents[1]
    repository_dir = backend_dir.parent
    database_path = tmp_path / f"migration-{working_directory}.db"
    env = os.environ.copy()
    env["DATABASE_URL"] = f"sqlite:///{database_path.as_posix()}"
    command = [sys.executable, "-m", "alembic", "-c", config_path, "upgrade", "head"]
    cwd = backend_dir if working_directory == "backend" else repository_dir
    heads = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", config_path, "heads"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    first = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    second = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert heads.returncode == 0, heads.stderr
    assert "20260810_0001 (head)" in heads.stdout
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    with sqlite3.connect(database_path) as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("document_chunks_fts",),
        ).fetchone()
    assert table == ("document_chunks_fts",)
