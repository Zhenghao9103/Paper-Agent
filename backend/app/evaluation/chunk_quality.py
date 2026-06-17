from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math
import re
from typing import Any


ChunkLike = Any
RetrievedLike = Any


@dataclass(frozen=True)
class ChunkDistributionMetrics:
    chunk_count: int
    avg_tokens: float
    p50_tokens: int
    p90_tokens: int
    too_short_rate: float
    too_long_rate: float


@dataclass(frozen=True)
class BoundaryQualityMetrics:
    sentence_boundary_rate: float
    page_crossing_rate: float


@dataclass(frozen=True)
class RetrievalEvalCase:
    query: str
    gold_chunk_ids: set[str | int] = field(default_factory=set)
    gold_pages: set[tuple[int, int]] = field(default_factory=set)


@dataclass(frozen=True)
class RetrievalQualityMetrics:
    case_count: int
    recall_at_k: float
    mrr_at_k: float


def evaluate_chunk_distribution(
    chunks: Sequence[ChunkLike],
    *,
    min_tokens: int = 120,
    max_tokens: int = 900,
) -> ChunkDistributionMetrics:
    token_counts = [_count_tokens(_get_value(chunk, "content", "")) for chunk in chunks]
    chunk_count = len(token_counts)
    if chunk_count == 0:
        return ChunkDistributionMetrics(
            chunk_count=0,
            avg_tokens=0.0,
            p50_tokens=0,
            p90_tokens=0,
            too_short_rate=0.0,
            too_long_rate=0.0,
        )

    too_short = sum(1 for count in token_counts if count < min_tokens)
    too_long = sum(1 for count in token_counts if count > max_tokens)
    return ChunkDistributionMetrics(
        chunk_count=chunk_count,
        avg_tokens=round(sum(token_counts) / chunk_count, 2),
        p50_tokens=_percentile(token_counts, 50),
        p90_tokens=_percentile(token_counts, 90),
        too_short_rate=round(too_short / chunk_count, 2),
        too_long_rate=round(too_long / chunk_count, 2),
    )


def evaluate_boundary_quality(chunks: Sequence[ChunkLike]) -> BoundaryQualityMetrics:
    chunk_count = len(chunks)
    if chunk_count == 0:
        return BoundaryQualityMetrics(sentence_boundary_rate=0.0, page_crossing_rate=0.0)

    sentence_boundaries = sum(
        1 for chunk in chunks if _ends_at_sentence_boundary(_get_value(chunk, "content", ""))
    )
    page_crossings = sum(1 for chunk in chunks if _crosses_page_boundary(chunk))
    return BoundaryQualityMetrics(
        sentence_boundary_rate=round(sentence_boundaries / chunk_count, 2),
        page_crossing_rate=round(page_crossings / chunk_count, 2),
    )


def evaluate_retrieval_quality(
    cases: Sequence[RetrievalEvalCase],
    *,
    retriever: Callable[[str, int], Sequence[RetrievedLike]],
    k: int = 5,
) -> RetrievalQualityMetrics:
    case_count = len(cases)
    if case_count == 0:
        return RetrievalQualityMetrics(case_count=0, recall_at_k=0.0, mrr_at_k=0.0)

    hits = 0
    reciprocal_rank_sum = 0.0
    for case in cases:
        rank = _first_relevant_rank(case, retriever(case.query, k)[:k])
        if rank is not None:
            hits += 1
            reciprocal_rank_sum += 1 / rank

    return RetrievalQualityMetrics(
        case_count=case_count,
        recall_at_k=round(hits / case_count, 2),
        mrr_at_k=round(reciprocal_rank_sum / case_count, 2),
    )


def _count_tokens(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text or ""))


def _percentile(values: Sequence[int], percentile: int) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
    return ordered[index]


def _ends_at_sentence_boundary(text: str) -> bool:
    stripped = (text or "").strip()
    return bool(stripped) and stripped[-1] in {".", "!", "?", ";", ":", "。", "！", "？", "；", "："}


def _crosses_page_boundary(chunk: ChunkLike) -> bool:
    start_page = _get_value(chunk, "start_page", None)
    end_page = _get_value(chunk, "end_page", None)
    if start_page is None or end_page is None:
        return False
    return start_page != end_page


def _first_relevant_rank(case: RetrievalEvalCase, results: Sequence[RetrievedLike]) -> int | None:
    for index, result in enumerate(results, start=1):
        if _matches_case(case, result):
            return index
    return None


def _matches_case(case: RetrievalEvalCase, result: RetrievedLike) -> bool:
    metadata = _get_metadata(result)
    chunk_id = _get_value(result, "chunk_id", _get_value(result, "id", metadata.get("chunk_id")))
    gold_chunk_ids = {str(gold_chunk_id) for gold_chunk_id in case.gold_chunk_ids}
    if chunk_id is not None and str(chunk_id) in gold_chunk_ids:
        return True

    document_id = metadata.get("document_id", _get_value(result, "document_id", None))
    page_number = metadata.get("page_number", _get_value(result, "page_number", None))
    if document_id is None or page_number is None:
        return False
    return (int(document_id), int(page_number)) in case.gold_pages


def _get_metadata(item: Any) -> dict[str, Any]:
    metadata = _get_value(item, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _get_value(item: Any, key: str, default: Any) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)
