from collections.abc import Sequence
from dataclasses import dataclass, field
from math import log2
from typing import Any

from backend.app.schemas.chat import Citation

FALLBACK_TRACE_MARKERS = {
    "retrieval_degraded",
    "web_search_failed",
    "long_term_memory_unavailable",
    "long_term_memory_vector_upsert_failed",
}


@dataclass(frozen=True)
class RAGEvalCase:
    question: str
    gold_pages: set[tuple[int, int]] = field(default_factory=set)
    gold_keywords: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class RAGRunResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None
    retrieval_mode: str = "hybrid"


@dataclass(frozen=True)
class RAGQualityMetrics:
    case_count: int
    recall_at_k: float
    mrr_at_k: float
    citation_hit_rate: float
    answer_contains_evidence_rate: float
    groundedness: float
    avg_latency_ms: float
    fallback_rate: float
    error_rate: float


@dataclass(frozen=True)
class RankedChunk:
    chunk_id: int
    document_id: int
    page_number: int
    rank: int
    score: float


@dataclass(frozen=True)
class RetrievalEvalCase:
    case_id: str
    relevance_by_chunk: dict[int, int] = field(default_factory=dict)
    chunk_document_ids: dict[int, int] = field(default_factory=dict)
    required_document_ids: set[int] = field(default_factory=set)
    expect_insufficient_evidence: bool = False


@dataclass(frozen=True)
class RetrievalRunResult:
    case_id: str
    ranked_chunks: list[RankedChunk] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None
    degraded: str | None = None


@dataclass(frozen=True)
class LatencyPercentiles:
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


@dataclass(frozen=True)
class RetrievalQualityMetrics:
    case_count: int
    answerable_case_count: int
    no_evidence_case_count: int
    recall_at_k: dict[int, float]
    hit_at_k: dict[int, float]
    mrr_at_k: dict[int, float]
    map_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    all_documents_recall_at_k: dict[int, float]
    no_evidence_returned_context_rate_at_k: dict[int, float]
    latency_ms: LatencyPercentiles
    error_rate: float
    degraded_rate: float = 0.0
    unmapped_case_count: int = 0


def evaluate_retrieval_results(
    cases: Sequence[RetrievalEvalCase],
    results: Sequence[RetrievalRunResult],
    *,
    top_ks: Sequence[int] = (3, 5, 10, 20),
) -> RetrievalQualityMetrics:
    ks = tuple(sorted(set(int(k) for k in top_ks)))
    if not ks or ks[0] <= 0:
        raise ValueError("top_ks must contain positive integers")
    result_by_id = {result.case_id: result for result in results}
    answerable = [case for case in cases if not case.expect_insufficient_evidence]
    # cases whose gold evidence could not be bound to any chunk are excluded
    # from ranking metrics instead of being counted as total misses
    scored_cases = [case for case in answerable if case.relevance_by_chunk]
    unmapped_cases = len(answerable) - len(scored_cases)
    no_evidence = [case for case in cases if case.expect_insufficient_evidence]
    recall: dict[int, float] = {}
    hit_rate: dict[int, float] = {}
    mrr: dict[int, float] = {}
    mean_ap: dict[int, float] = {}
    ndcg: dict[int, float] = {}
    all_documents: dict[int, float] = {}
    returned_context: dict[int, float] = {}

    for k in ks:
        recall_rows: list[float] = []
        hit_rows: list[float] = []
        mrr_rows: list[float] = []
        ap_rows: list[float] = []
        ndcg_rows: list[float] = []
        document_rows: list[float] = []
        for case in scored_cases:
            ranked = result_by_id.get(
                case.case_id, RetrievalRunResult(case.case_id)
            ).ranked_chunks[:k]
            relevant_ids = set(case.relevance_by_chunk)
            hit_ids = [item.chunk_id for item in ranked if item.chunk_id in relevant_ids]
            recall_rows.append(len(set(hit_ids)) / len(relevant_ids) if relevant_ids else 0.0)
            hit_rows.append(float(bool(hit_ids)))
            first = next(
                (
                    index
                    for index, item in enumerate(ranked, 1)
                    if item.chunk_id in relevant_ids
                ),
                None,
            )
            mrr_rows.append(1 / first if first else 0.0)
            hit_count = 0
            precision_sum = 0.0
            for rank, item in enumerate(ranked, 1):
                if item.chunk_id in relevant_ids:
                    hit_count += 1
                    precision_sum += hit_count / rank
            ap_rows.append(
                precision_sum / min(len(relevant_ids), k) if relevant_ids else 0.0
            )
            gains = [case.relevance_by_chunk.get(item.chunk_id, 0) for item in ranked]
            dcg = sum((2**grade - 1) / log2(rank + 1) for rank, grade in enumerate(gains, 1))
            ideal_grades = sorted(case.relevance_by_chunk.values(), reverse=True)[:k]
            ideal = sum(
                (2**grade - 1) / log2(rank + 1)
                for rank, grade in enumerate(ideal_grades, 1)
            )
            ndcg_rows.append(dcg / ideal if ideal else 0.0)
            if case.required_document_ids:
                hit_documents = {item.document_id for item in ranked}
                document_rows.append(float(case.required_document_ids <= hit_documents))
        recall[k] = _mean(recall_rows)
        hit_rate[k] = _mean(hit_rows)
        mrr[k] = _mean(mrr_rows)
        mean_ap[k] = _mean(ap_rows)
        ndcg[k] = _mean(ndcg_rows)
        all_documents[k] = _mean(document_rows)
        returned_context[k] = _mean(
            [
                float(
                    bool(
                        result_by_id.get(
                            case.case_id, RetrievalRunResult(case.case_id)
                        ).ranked_chunks[:k]
                    )
                )
                for case in no_evidence
            ]
        )

    latencies = [result.latency_ms for result in results if result.error is None]
    errors = sum(bool(result.error) for result in results)
    degraded_count = sum(bool(result.degraded) for result in results)
    return RetrievalQualityMetrics(
        case_count=len(cases),
        answerable_case_count=len(answerable),
        no_evidence_case_count=len(no_evidence),
        recall_at_k=recall,
        hit_at_k=hit_rate,
        mrr_at_k=mrr,
        map_at_k=mean_ap,
        ndcg_at_k=ndcg,
        all_documents_recall_at_k=all_documents,
        no_evidence_returned_context_rate_at_k=returned_context,
        latency_ms=LatencyPercentiles(
            p50=_percentile(latencies, 0.50),
            p95=_percentile(latencies, 0.95),
            p99=_percentile(latencies, 0.99),
        ),
        error_rate=errors / len(cases) if cases else 0.0,
        degraded_rate=degraded_count / len(cases) if cases else 0.0,
        unmapped_case_count=unmapped_cases,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def evaluate_rag_results(
    cases: Sequence[RAGEvalCase],
    results: Sequence[RAGRunResult],
    *,
    k: int = 5,
) -> RAGQualityMetrics:
    case_count = min(len(cases), len(results))
    if case_count == 0:
        return RAGQualityMetrics(
            case_count=0,
            recall_at_k=0.0,
            mrr_at_k=0.0,
            citation_hit_rate=0.0,
            answer_contains_evidence_rate=0.0,
            groundedness=0.0,
            avg_latency_ms=0.0,
            fallback_rate=0.0,
            error_rate=0.0,
        )

    paired = list(zip(cases[:case_count], results[:case_count], strict=False))
    ranks = [_first_relevant_rank(case, result.citations[:k]) for case, result in paired]
    recall_hits = sum(1 for rank in ranks if rank is not None)
    reciprocal_rank_sum = sum(1 / rank for rank in ranks if rank is not None)
    citation_hits = sum(
        1 for case, result in paired if _has_relevant_citation(case, result.citations)
    )
    answer_hits = sum(1 for case, result in paired if _answer_contains_gold_keyword(case, result))
    grounded_hits = sum(1 for case, result in paired if _answer_is_grounded(case, result))
    fallback_hits = sum(1 for _, result in paired if _used_fallback(result.trace))
    error_hits = sum(1 for _, result in paired if result.error)
    latency_sum = sum(result.latency_ms for _, result in paired)

    return RAGQualityMetrics(
        case_count=case_count,
        recall_at_k=round(recall_hits / case_count, 2),
        mrr_at_k=round(reciprocal_rank_sum / case_count, 2),
        citation_hit_rate=round(citation_hits / case_count, 2),
        answer_contains_evidence_rate=round(answer_hits / case_count, 2),
        groundedness=round(grounded_hits / case_count, 2),
        avg_latency_ms=round(latency_sum / case_count, 2),
        fallback_rate=round(fallback_hits / case_count, 2),
        error_rate=round(error_hits / case_count, 2),
    )


def load_eval_payload(payload: dict[str, Any]) -> tuple[list[RAGEvalCase], list[RAGRunResult]]:
    cases = [
        RAGEvalCase(
            question=str(item.get("question", "")),
            gold_pages={tuple(page) for page in item.get("gold_pages", [])},
            gold_keywords={str(keyword) for keyword in item.get("gold_keywords", [])},
        )
        for item in payload.get("cases", [])
    ]
    results = [
        RAGRunResult(
            answer=str(item.get("answer", "")),
            citations=[Citation(**citation) for citation in item.get("citations", [])],
            trace=[str(node) for node in item.get("trace", [])],
            latency_ms=float(item.get("latency_ms", 0.0)),
            error=item.get("error"),
            retrieval_mode=str(item.get("retrieval_mode", "hybrid")),
        )
        for item in payload.get("results", [])
    ]
    return cases, results


def metrics_to_dict(metrics: RAGQualityMetrics) -> dict[str, float | int]:
    return {
        "case_count": metrics.case_count,
        "recall_at_k": metrics.recall_at_k,
        "mrr_at_k": metrics.mrr_at_k,
        "citation_hit_rate": metrics.citation_hit_rate,
        "answer_contains_evidence_rate": metrics.answer_contains_evidence_rate,
        "groundedness": metrics.groundedness,
        "avg_latency_ms": metrics.avg_latency_ms,
        "fallback_rate": metrics.fallback_rate,
        "error_rate": metrics.error_rate,
    }


def _first_relevant_rank(case: RAGEvalCase, citations: Sequence[Citation]) -> int | None:
    for index, citation in enumerate(citations, start=1):
        if _citation_matches_gold_page(case, citation):
            return index
    return None


def _has_relevant_citation(case: RAGEvalCase, citations: Sequence[Citation]) -> bool:
    return any(_citation_matches_gold_page(case, citation) for citation in citations)


def _citation_matches_gold_page(case: RAGEvalCase, citation: Citation) -> bool:
    if not case.gold_pages:
        return False
    return (citation.document_id, citation.page_number) in case.gold_pages


def _answer_contains_gold_keyword(case: RAGEvalCase, result: RAGRunResult) -> bool:
    if not case.gold_keywords:
        return False
    answer = result.answer.lower()
    return any(keyword.lower() in answer for keyword in case.gold_keywords)


def _answer_is_grounded(case: RAGEvalCase, result: RAGRunResult) -> bool:
    if not case.gold_keywords:
        return False
    answer = result.answer.lower()
    evidence = " ".join(citation.content for citation in result.citations).lower()
    return any(
        keyword.lower() in answer and keyword.lower() in evidence for keyword in case.gold_keywords
    )


def _used_fallback(trace: Sequence[str]) -> bool:
    return any(marker in FALLBACK_TRACE_MARKERS for marker in trace)
