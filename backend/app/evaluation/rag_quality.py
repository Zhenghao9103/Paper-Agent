from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any

from ..schemas.chat import Citation


FALLBACK_TRACE_MARKERS = {
    "sqlite_fallback_retrieve",
    "sqlite_overview_retrieve",
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
    citation_hits = sum(1 for case, result in paired if _has_relevant_citation(case, result.citations))
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
    return any(keyword.lower() in answer and keyword.lower() in evidence for keyword in case.gold_keywords)


def _used_fallback(trace: Sequence[str]) -> bool:
    return any(marker in FALLBACK_TRACE_MARKERS for marker in trace)
