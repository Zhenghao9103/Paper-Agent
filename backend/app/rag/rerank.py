from copy import deepcopy
from functools import lru_cache
from typing import Any


BGE_RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def _get_bge_reranker():
    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise RuntimeError(
            "BGE-Reranker requires FlagEmbedding. Install backend requirements before reranking."
        ) from exc
    return FlagReranker(BGE_RERANKER_MODEL_NAME, use_fp16=False)


def rerank_chunks(query: str, matches: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    if not matches:
        return []

    pairs = [[query, str(match.get("content", ""))] for match in matches]
    scores = _get_bge_reranker().compute_score(pairs)
    if isinstance(scores, int | float):
        scores = [float(scores)]
    else:
        scores = [float(score) for score in scores]

    reranked: list[dict[str, Any]] = []
    for match, rerank_score in zip(matches, scores, strict=False):
        updated = deepcopy(match)
        updated["original_score"] = float(match.get("score", 0.0))
        updated["rerank_score"] = rerank_score
        updated["score"] = rerank_score
        reranked.append(updated)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:top_k]
