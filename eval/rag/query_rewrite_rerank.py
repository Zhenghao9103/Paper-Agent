"""Helpers for reranking frozen query-ablation Fusion candidates."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from backend.app.rag.rerank import rerank_chunks


def rerank_frozen_case(
    query: str,
    frozen_candidates: Sequence[Mapping[str, Any]],
    *,
    chunks_by_id: Mapping[int, Any],
    rerank_fn: Callable[..., list[dict[str, Any]]] = rerank_chunks,
) -> dict[str, Any]:
    """Apply the production reranker without changing the Fusion candidate set."""

    matches: list[dict[str, Any]] = []
    frozen_ids: list[int] = []
    for candidate in frozen_candidates:
        chunk_id = int(candidate["chunk_id"])
        if chunk_id not in chunks_by_id:
            raise ValueError(f"Fusion chunk is absent from SQLite: {chunk_id}")
        chunk = chunks_by_id[chunk_id]
        frozen_ids.append(chunk_id)
        matches.append(
            {
                "chunk_id": chunk_id,
                "document_id": int(chunk.document_id),
                "page_number": int(chunk.page_number),
                "content": str(chunk.content),
                "fusion_rank": int(candidate.get("rank") or len(matches) + 1),
                "fusion_score": float(candidate.get("score") or 0.0),
                "score": float(candidate.get("score") or 0.0),
                "bm25_rank": candidate.get("bm25_rank"),
                "vector_rank": candidate.get("vector_rank"),
            }
        )

    started = time.perf_counter()
    reranked = rerank_fn(query, matches, top_k=len(matches))
    latency_ms = (time.perf_counter() - started) * 1000
    reranked_ids = [int(item["chunk_id"]) for item in reranked]
    if len(reranked_ids) != len(set(reranked_ids)) or set(reranked_ids) != set(
        frozen_ids
    ):
        raise ValueError("reranker changed the frozen candidate set")

    ranked = [
        {
            "chunk_id": int(item["chunk_id"]),
            "document_id": int(item["document_id"]),
            "page_number": int(item["page_number"]),
            "rank": rank,
            "score": float(item["rerank_score"]),
            "rerank_logit": float(item["rerank_logit"]),
            "fusion_rank": int(item["fusion_rank"]),
            "fusion_score": float(item["fusion_score"]),
            "bm25_rank": item.get("bm25_rank"),
            "vector_rank": item.get("vector_rank"),
        }
        for rank, item in enumerate(reranked, start=1)
    ]
    return {
        "latency_ms": latency_ms,
        "error": None,
        "degraded": None,
        "rerank_applied": True,
        "ranked_chunks": ranked,
    }


def extract_raw_rerank_cases(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract the production Raw-query reranked arm from its frozen report."""

    output: dict[str, dict[str, Any]] = {}
    cases = report.get("retrieval", {}).get("hybrid", {}).get("cases", [])
    for case in cases:
        case_id = str(case.get("case_id") or "")
        ranked = case.get("ranked_chunks") or []
        if not case_id or not ranked:
            raise ValueError(f"Raw report lacks reranked candidates for {case_id or '<unknown>'}")
        diagnostics = case.get("diagnostics") or {}
        timings = diagnostics.get("timings_ms") or {}
        output[case_id] = {
            "case_id": case_id,
            "question": case.get("question"),
            "latency_ms": float(timings.get("rerank") or 0.0),
            "error": case.get("error"),
            "degraded": case.get("degraded"),
            "rerank_applied": True,
            "ranked_chunks": [dict(row) for row in ranked],
        }
    return output
