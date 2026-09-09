from __future__ import annotations

import pytest
from backend.app.models.chunk import DocumentChunk
from backend.app.models.document import Document
from backend.app.rag import hybrid, rerank
from backend.app.rag.bm25_store import BM25Hit, index_chunk
from backend.app.rag.fusion import reciprocal_rank_fusion
from backend.app.schemas.retrieval import QueryPlan
from sqlalchemy.orm import Session


def _plan(*, document_id: int | None = None) -> QueryPlan:
    return QueryPlan(
        intent="simple_rag",
        confidence=0.9,
        standalone_query="graph construction",
        lexical_terms=["graph", "construction"],
        synonyms=["network"],
        semantic_queries=["How is the graph constructed?"],
        document_id=document_id,
    )


def _seed_chunks(db: Session) -> list[DocumentChunk]:
    document = Document(
        title="Paper",
        file_type="pdf",
        file_path="paper.pdf",
        status="indexed",
    )
    db.add(document)
    db.flush()
    chunks = [
        DocumentChunk(
            document_id=document.id,
            page_number=index,
            chunk_index=0,
            content=f"Graph construction evidence {index}",
        )
        for index in range(1, 5)
    ]
    db.add_all(chunks)
    db.flush()
    for chunk in chunks:
        index_chunk(
            db,
            chunk_id=chunk.id,
            document_id=document.id,
            title=document.title,
            content=chunk.content,
        )
    db.commit()
    return chunks


def test_reciprocal_rank_fusion_sorts_scores_and_deduplicates() -> None:
    fused = reciprocal_rank_fusion([[3, 1, 2], [2, 1, 4]], k=60)

    assert [item.key for item in fused] == [2, 1, 3, 4]
    assert [item.score for item in fused] == sorted(
        (item.score for item in fused), reverse=True
    )


def test_reciprocal_rank_fusion_duplicate_inputs_and_ties_are_deterministic() -> None:
    first = reciprocal_rank_fusion([["b", "b", "a"], ["c", "a", "c"]], k=1)
    second = reciprocal_rank_fusion([["b", "b", "a"], ["c", "a", "c"]], k=1)

    assert first == second
    assert [item.key for item in first] == ["a", "b", "c"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1]], k=0)


def test_reciprocal_rank_fusion_deduplicates_before_assigning_ranks() -> None:
    fused = reciprocal_rank_fusion([["a", "a", "b"]], k=1)

    assert [(item.key, item.score) for item in fused] == [
        ("a", pytest.approx(1 / 2)),
        ("b", pytest.approx(1 / 3)),
    ]


def test_hybrid_search_fuses_channels_and_reranks_all_candidates(db_session, monkeypatch) -> None:
    chunks = _seed_chunks(db_session)
    bm25_ids = [chunks[0].id, chunks[1].id]
    vector_ids = [chunks[2].id, chunks[1].id]
    rerank_seen: list[int] = []

    monkeypatch.setattr(
        hybrid,
        "search_bm25",
        lambda *args, **kwargs: [
            BM25Hit(chunk_id=bm25_ids[0], rank=1, raw_score=-1),
            BM25Hit(chunk_id=bm25_ids[1], rank=2, raw_score=-2),
        ],
    )
    monkeypatch.setattr(
        hybrid,
        "query_chunks",
        lambda *args, **kwargs: [
            {"content": "vector", "metadata": {"chunk_id": vector_ids[0]}, "score": 0.9},
            {"content": "vector", "metadata": {"chunk_id": vector_ids[1]}, "score": 0.8},
        ],
    )

    def fake_rerank(query, candidates, top_k):
        rerank_seen.extend(int(item["chunk_id"]) for item in candidates)
        return candidates[:top_k]

    monkeypatch.setattr(hybrid, "rerank_chunks", fake_rerank)

    result = hybrid.hybrid_search(db_session, _plan(), evidence_limit=3)

    assert len(result.candidates) == 3
    assert result.candidates[0].fusion_rank == 1
    assert len(rerank_seen) == 3
    assert result.diagnostics.degraded_channels == []
    assert set(result.diagnostics.timings_ms) == {"bm25", "vector", "fusion", "rerank"}
    assert result.diagnostics.channel_status["bm25"]["executed"] is True
    assert result.diagnostics.channel_status["vector"]["executed"] is True
    assert [row["chunk_id"] for row in result.diagnostics.channel_candidates["bm25"]] == bm25_ids
    assert [row["chunk_id"] for row in result.diagnostics.channel_candidates["vector"]] == vector_ids
    assert result.diagnostics.fusion_candidates
    assert result.diagnostics.rerank_candidates


def test_hybrid_search_reports_internal_channel_rrf_rank_not_raw_query_rank(
    db_session, monkeypatch
) -> None:
    chunks = _seed_chunks(db_session)
    first_id, second_id = chunks[0].id, chunks[1].id
    calls = 0

    def fake_bm25(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [
                BM25Hit(chunk_id=first_id, rank=1, raw_score=-1),
                BM25Hit(chunk_id=second_id, rank=2, raw_score=-2),
            ]
        return [BM25Hit(chunk_id=second_id, rank=1, raw_score=-1)]

    monkeypatch.setattr(hybrid, "search_bm25", fake_bm25)
    monkeypatch.setattr(hybrid, "query_chunks", lambda *args, **kwargs: [])
    monkeypatch.setattr(hybrid, "rerank_chunks", lambda query, candidates, top_k: candidates)

    result = hybrid.hybrid_search(db_session, _plan(), evidence_limit=2)
    by_id = {candidate.chunk_id: candidate for candidate in result.candidates}

    assert by_id[first_id].bm25_rank == 2
    assert by_id[second_id].bm25_rank == 1


def test_hybrid_search_reports_internal_vector_rrf_rank_not_raw_query_rank(
    db_session, monkeypatch
) -> None:
    chunks = _seed_chunks(db_session)
    first_id, second_id = chunks[0].id, chunks[1].id

    monkeypatch.setattr(hybrid, "search_bm25", lambda *args, **kwargs: [])

    def fake_vectors(query, *args, **kwargs):
        if query == "q1":
            return [
                {"metadata": {"chunk_id": first_id}, "content": "first", "score": 0.9},
                {"metadata": {"chunk_id": second_id}, "content": "second", "score": 0.8},
            ]
        return [{"metadata": {"chunk_id": second_id}, "content": "second", "score": 0.9}]

    monkeypatch.setattr(hybrid, "query_chunks", fake_vectors)
    monkeypatch.setattr(hybrid, "rerank_chunks", lambda query, candidates, top_k: candidates)

    plan = _plan().model_copy(update={"semantic_queries": ["q1", "q2"]})
    result = hybrid.hybrid_search(db_session, plan, evidence_limit=2)
    by_id = {candidate.chunk_id: candidate for candidate in result.candidates}

    assert by_id[first_id].vector_rank == 2
    assert by_id[second_id].vector_rank == 1


@pytest.mark.parametrize(
    ("failed_channel", "expected_degraded"),
    [("bm25", "bm25"), ("vector", "vector")],
)
def test_hybrid_search_survives_single_channel_failure(
    db_session, monkeypatch, failed_channel: str, expected_degraded: str
) -> None:
    chunks = _seed_chunks(db_session)
    if failed_channel == "bm25":
        def fail_bm25(*args, **kwargs):
            raise RuntimeError("bm25")

        monkeypatch.setattr(hybrid, "search_bm25", fail_bm25)
        monkeypatch.setattr(
            hybrid,
            "query_chunks",
            lambda *args, **kwargs: [
                {"metadata": {"chunk_id": chunks[0].id}, "content": "v", "score": 0.9}
            ],
        )
    else:
        monkeypatch.setattr(
            hybrid,
            "search_bm25",
            lambda *args, **kwargs: [BM25Hit(chunks[0].id, 1, -1)],
        )

        def fail_vector(*args, **kwargs):
            raise RuntimeError("vector")

        monkeypatch.setattr(hybrid, "query_chunks", fail_vector)
    monkeypatch.setattr(hybrid, "rerank_chunks", lambda query, candidates, top_k: candidates)

    result = hybrid.hybrid_search(db_session, _plan(), evidence_limit=2)

    assert expected_degraded in result.diagnostics.degraded_channels
    assert all(
        name in result.diagnostics.timings_ms
        for name in ("bm25", "vector", "fusion", "rerank")
    )


def test_hybrid_search_raises_when_both_channels_fail(db_session, monkeypatch) -> None:
    _seed_chunks(db_session)

    def fail_bm25(*args, **kwargs):
        raise RuntimeError("bm25")

    def fail_vector(*args, **kwargs):
        raise RuntimeError("vector")

    monkeypatch.setattr(hybrid, "search_bm25", fail_bm25)
    monkeypatch.setattr(hybrid, "query_chunks", fail_vector)

    with pytest.raises(hybrid.RetrievalUnavailable):
        hybrid.hybrid_search(db_session, _plan())


def test_hybrid_search_modes_and_ghost_vector_ids(db_session, monkeypatch) -> None:
    chunks = _seed_chunks(db_session)
    monkeypatch.setattr(
        hybrid,
        "search_bm25",
        lambda *args, **kwargs: [BM25Hit(chunks[0].id, 1, -1)],
    )
    monkeypatch.setattr(
        hybrid,
        "query_chunks",
        lambda *args, **kwargs: [
            {"metadata": {"chunk_id": "ghost"}, "content": "ghost", "score": 1.0},
            {"metadata": {"chunk_id": chunks[1].id}, "content": "vector", "score": 0.8},
        ],
    )
    monkeypatch.setattr(hybrid, "rerank_chunks", lambda query, candidates, top_k: candidates)

    bm25_result = hybrid.hybrid_search(db_session, _plan(), mode="bm25")
    vector_result = hybrid.hybrid_search(db_session, _plan(), mode="vector")

    assert [candidate.chunk_id for candidate in bm25_result.candidates] == [chunks[0].id]
    assert [candidate.chunk_id for candidate in vector_result.candidates] == [chunks[1].id]


def test_hybrid_search_marks_reranker_degraded_and_keeps_fusion_order(
    db_session, monkeypatch
) -> None:
    chunks = _seed_chunks(db_session)
    monkeypatch.setattr(
        hybrid,
        "search_bm25",
        lambda *args, **kwargs: [BM25Hit(chunks[0].id, 1, -1)],
    )
    monkeypatch.setattr(hybrid, "query_chunks", lambda *args, **kwargs: [])

    def fail_reranker(*args, **kwargs):
        raise RuntimeError("reranker")

    monkeypatch.setattr(hybrid, "rerank_chunks", fail_reranker)

    result = hybrid.hybrid_search(db_session, _plan())

    assert result.candidates[0].chunk_id == chunks[0].id
    assert "reranker" in result.diagnostics.degraded_channels


def test_rerank_chunks_propagates_bge_unavailable(monkeypatch) -> None:
    def unavailable():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(rerank, "_get_bge_reranker", unavailable)
    matches = [
        {"chunk_id": 2, "content": "high", "score": 0.9},
        {"chunk_id": 1, "content": "low", "score": 0.2},
    ]

    with pytest.raises(RuntimeError, match="model unavailable"):
        rerank.rerank_chunks("query", matches, top_k=2)


def test_rerank_chunks_propagates_score_count_mismatch(monkeypatch) -> None:
    class MismatchedReranker:
        def compute_score(self, pairs):
            return [0.8]

    monkeypatch.setattr(rerank, "_get_bge_reranker", lambda: MismatchedReranker())
    matches = [
        {"chunk_id": 1, "content": "first", "score": 0.9},
        {"chunk_id": 2, "content": "second", "score": 0.8},
    ]

    with pytest.raises(RuntimeError, match="returned 1 scores for 2 candidates"):
        rerank.rerank_chunks("query", matches, top_k=2)
