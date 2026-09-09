"""Hybrid BM25/vector retrieval with two-level RRF and optional BGE reranking."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..models.chunk import DocumentChunk
from ..models.document import Document
from ..schemas.retrieval import QueryPlan, RetrievalCandidate, RetrievalDiagnostics
from .bm25_store import BM25Hit, search_bm25
from .fusion import FusedRank, reciprocal_rank_fusion
from .rerank import rerank_chunks
from .vector_store import query_chunks

RetrievalMode = Literal["bm25", "vector", "hybrid"]


class RetrievalUnavailable(RuntimeError):
    """Raised when both requested first-stage retrieval channels fail."""


class HybridSearchResult(BaseModel):
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    diagnostics: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)


def hybrid_search(
    db: Session,
    plan: QueryPlan,
    evidence_limit: int | None = None,
    mode: RetrievalMode = "hybrid",
    rerank: bool = True,
) -> HybridSearchResult:
    """Retrieve SQLite-backed evidence using BM25, vectors, or both.

    The supplied session belongs to the caller.  This function only reads from it;
    it never commits, rolls back, closes, or otherwise changes session ownership.
    ``rerank=False`` keeps the fusion order, for single-channel baselines that
    must not share the cross-encoder post-processing with other modes.
    """

    if mode not in {"bm25", "vector", "hybrid"}:
        raise ValueError("mode must be one of: bm25, vector, hybrid")
    settings = get_settings()
    requested_limit = evidence_limit
    if requested_limit is None:
        requested_limit = settings.simple_evidence_limit
    if requested_limit <= 0:
        raise ValueError("evidence_limit must be greater than zero")

    diagnostics = RetrievalDiagnostics()
    degraded: list[str] = []

    bm25_rankings: list[list[int]] = []
    bm25_ranks: dict[int, int] = {}
    bm25_failed = False
    bm25_attempts = 0
    bm25_successes = 0
    if mode in {"bm25", "hybrid"}:
        started = perf_counter()
        bm25_query_failed = False
        terms_sets = _bm25_term_sets(plan)
        for terms in terms_sets:
            bm25_attempts += 1
            try:
                hits = search_bm25(
                    db,
                    terms,
                    document_id=plan.document_id,
                    limit=settings.bm25_candidates,
                )
                ids = _bm25_ids(hits)
                bm25_successes += 1
                bm25_rankings.append(ids)
                for rank, chunk_id in enumerate(ids, start=1):
                    bm25_ranks[chunk_id] = min(rank, bm25_ranks.get(chunk_id, rank))
            except Exception:
                bm25_query_failed = True
        bm25_failed = bm25_query_failed and not bm25_rankings
        if bm25_query_failed:
            degraded.append("bm25")
        diagnostics.timings_ms["bm25"] = _elapsed_ms(started)
    else:
        diagnostics.timings_ms["bm25"] = 0.0

    vector_rankings: list[list[int]] = []
    vector_ranks: dict[int, int] = {}
    vector_queries: dict[int, list[str]] = {}
    vector_failed = False
    vector_attempts = 0
    vector_successes = 0
    if mode in {"vector", "hybrid"}:
        started = perf_counter()
        vector_query_failed = False
        malformed_vector_hit = False
        for semantic_query in plan.semantic_queries:
            vector_attempts += 1
            try:
                hits = query_chunks(
                    semantic_query,
                    document_id=plan.document_id,
                    limit=settings.vector_candidates_per_query,
                )
                vector_successes += 1
                ids: list[int] = []
                for rank, hit in enumerate(hits, start=1):
                    chunk_id = _vector_chunk_id(hit)
                    if chunk_id is None:
                        malformed_vector_hit = True
                        continue
                    if chunk_id in ids:
                        continue
                    ids.append(chunk_id)
                    vector_ranks[chunk_id] = min(rank, vector_ranks.get(chunk_id, rank))
                    vector_queries.setdefault(chunk_id, []).append(semantic_query)
                vector_rankings.append(ids)
            except Exception:
                vector_query_failed = True
        vector_failed = vector_query_failed and not vector_rankings
        if vector_query_failed or malformed_vector_hit:
            degraded.append("vector")
        diagnostics.timings_ms["vector"] = _elapsed_ms(started)
    else:
        diagnostics.timings_ms["vector"] = 0.0

    requested_channels = int(mode in {"bm25", "hybrid"}) + int(
        mode in {"vector", "hybrid"}
    )
    failed_channels = int(bm25_failed) + int(vector_failed)
    if requested_channels and failed_channels == requested_channels:
        raise RetrievalUnavailable("all requested retrieval channels are unavailable")

    started = perf_counter()
    bm25_fused = reciprocal_rank_fusion(
        bm25_rankings,
        k=settings.rrf_k,
    )
    vector_fused = reciprocal_rank_fusion(
        vector_rankings,
        k=settings.rrf_k,
    )
    bm25_channel_ranks = {
        int(item.key): rank for rank, item in enumerate(bm25_fused, start=1)
    }
    vector_channel_ranks = {
        int(item.key): rank for rank, item in enumerate(vector_fused, start=1)
    }
    diagnostics.channel_status = {
        "bm25": {
            "requested": mode in {"bm25", "hybrid"},
            "executed": bm25_successes > 0,
            "query_count": bm25_attempts,
            "successful_queries": bm25_successes,
            "failed": bm25_failed,
        },
        "vector": {
            "requested": mode in {"vector", "hybrid"},
            "executed": vector_successes > 0,
            "query_count": vector_attempts,
            "successful_queries": vector_successes,
            "failed": vector_failed,
        },
    }
    diagnostics.channel_candidates = {
        "bm25": [
            {"chunk_id": int(item.key), "rank": rank, "rrf_score": float(item.score)}
            for rank, item in enumerate(bm25_fused, start=1)
        ],
        "vector": [
            {"chunk_id": int(item.key), "rank": rank, "rrf_score": float(item.score)}
            for rank, item in enumerate(vector_fused, start=1)
        ],
    }
    channel_rankings: list[list[int]] = []
    if mode in {"bm25", "hybrid"} and not bm25_failed:
        channel_rankings.append([int(item.key) for item in bm25_fused])
    if mode in {"vector", "hybrid"} and not vector_failed:
        channel_rankings.append([int(item.key) for item in vector_fused])
    fused = reciprocal_rank_fusion(channel_rankings, k=settings.rrf_k)
    diagnostics.timings_ms["fusion"] = _elapsed_ms(started)

    candidates = _load_candidates(
        db,
        fused,
        document_id=plan.document_id,
        bm25_ranks=bm25_channel_ranks,
        vector_ranks=vector_channel_ranks,
        vector_queries=vector_queries,
    )
    candidates = candidates[: settings.fusion_candidates]
    diagnostics.fusion_candidates = [
        {
            "chunk_id": candidate.chunk_id,
            "rank": candidate.fusion_rank,
            "score": candidate.fusion_score,
            "bm25_rank": candidate.bm25_rank,
            "vector_rank": candidate.vector_rank,
        }
        for candidate in candidates
    ]

    started = perf_counter()
    if not candidates:
        diagnostics.timings_ms["rerank"] = 0.0
        diagnostics.degraded_channels = _unique(degraded)
        return HybridSearchResult(candidates=[], diagnostics=diagnostics)

    fusion_candidates = list(candidates)
    if rerank:
        try:
            rerank_input = [
                _candidate_for_rerank(candidate) for candidate in fusion_candidates
            ]
            reranked = rerank_chunks(
                plan.standalone_query,
                rerank_input,
                top_k=min(settings.fusion_candidates, len(fusion_candidates)),
            )
            if any(bool(item.get("_reranker_degraded")) for item in reranked):
                degraded.append("reranker")
            candidates = _candidates_from_rerank(reranked, fusion_candidates)
        except Exception:
            degraded.append("reranker")
            candidates = fusion_candidates
    diagnostics.timings_ms["rerank"] = _elapsed_ms(started)
    diagnostics.degraded_channels = _unique(degraded)
    diagnostics.rerank_candidates = [
        {
            "chunk_id": candidate.chunk_id,
            "rank": rank,
            "score": (
                candidate.rerank_score
                if candidate.rerank_score is not None
                else candidate.fusion_score
            ),
        }
        for rank, candidate in enumerate(candidates, start=1)
    ]
    return HybridSearchResult(
        candidates=candidates[:requested_limit],
        diagnostics=diagnostics,
    )


def _bm25_term_sets(plan: QueryPlan) -> list[list[str]]:
    lexical = list(plan.lexical_terms)
    lexical_keys = {item.casefold() for item in lexical}
    expanded = lexical + [
        term for term in plan.synonyms if term.casefold() not in lexical_keys
    ]
    return [lexical, expanded]


def _bm25_ids(hits: Iterable[BM25Hit | Mapping[str, Any] | Any]) -> list[int]:
    ids: list[int] = []
    for hit in hits:
        raw_id = (
            hit.get("chunk_id")
            if isinstance(hit, Mapping)
            else getattr(hit, "chunk_id", None)
        )
        try:
            chunk_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if chunk_id not in ids:
            ids.append(chunk_id)
    return ids


def _vector_chunk_id(hit: Any) -> int | None:
    if not isinstance(hit, Mapping):
        return None
    metadata = hit.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    raw_id = metadata.get("chunk_id", hit.get("chunk_id"))
    try:
        chunk_id = int(raw_id)
    except (TypeError, ValueError):
        return None
    return chunk_id if chunk_id > 0 else None


def _load_candidates(
    db: Session,
    fused: list[FusedRank],
    *,
    document_id: int | None,
    bm25_ranks: Mapping[int, int],
    vector_ranks: Mapping[int, int],
    vector_queries: Mapping[int, list[str]],
) -> list[RetrievalCandidate]:
    if not fused:
        return []
    ids = [int(item.key) for item in fused]
    statement = (
        select(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.id.in_(ids))
    )
    if document_id is not None:
        statement = statement.where(DocumentChunk.document_id == document_id)
    rows = db.execute(statement).all()
    by_id = {int(chunk.id): (chunk, document) for chunk, document in rows}
    candidates: list[RetrievalCandidate] = []
    for fusion_rank, fused_rank in enumerate(fused, start=1):
        chunk_id = int(fused_rank.key)
        row = by_id.get(chunk_id)
        if row is None:
            continue
        chunk, document = row
        candidates.append(
            RetrievalCandidate(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                title=document.title,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                bm25_rank=bm25_ranks.get(chunk_id),
                vector_rank=vector_ranks.get(chunk_id),
                fusion_rank=fusion_rank,
                fusion_score=float(fused_rank.score),
                matched_queries=list(vector_queries.get(chunk_id, [])),
            )
        )
    return candidates


def _candidate_for_rerank(candidate: RetrievalCandidate) -> dict[str, Any]:
    payload = candidate.model_dump()
    payload["score"] = candidate.fusion_score
    payload["metadata"] = {
        "chunk_id": candidate.chunk_id,
        "document_id": candidate.document_id,
        "title": candidate.title,
        "page_number": candidate.page_number,
        "chunk_index": candidate.chunk_index,
    }
    return payload


def _candidates_from_rerank(
    reranked: Iterable[Mapping[str, Any]],
    original: list[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    by_id = {candidate.chunk_id: candidate for candidate in original}
    output: list[RetrievalCandidate] = []
    seen: set[int] = set()
    for item in reranked:
        raw_id = item.get("chunk_id")
        if raw_id is None and isinstance(item.get("metadata"), Mapping):
            raw_id = item["metadata"].get("chunk_id")
        try:
            chunk_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        candidate = by_id.get(chunk_id)
        if candidate is None or chunk_id in seen:
            continue
        seen.add(chunk_id)
        rerank_score = item.get("rerank_score")
        output.append(candidate.model_copy(update={"rerank_score": _as_float(rerank_score)}))
    for candidate in original:
        if candidate.chunk_id not in seen:
            output.append(candidate)
    return output


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _elapsed_ms(started: float) -> float:
    return max(0.0, (perf_counter() - started) * 1000.0)


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output
