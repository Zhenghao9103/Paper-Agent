import math
from copy import deepcopy
from functools import lru_cache
from typing import Any

from ..core.paths import require_hf_model_snapshot

BGE_RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
BGE_CPU_BATCH_SIZE = 4


@lru_cache(maxsize=1)
def _get_bge_reranker():
    try:
        from FlagEmbedding import FlagReranker
    except ImportError as exc:
        raise RuntimeError(
            "BGE-Reranker requires FlagEmbedding. Install backend requirements before reranking."
        ) from exc
    return FlagReranker(
        require_hf_model_snapshot(BGE_RERANKER_MODEL_NAME), use_fp16=False
    )


def normalize_rerank_score(raw_score: float) -> float:
    """Squash an unbounded cross-encoder logit into [0, 1].

    BGE rerankers emit raw logits (roughly -10..+10), while first-stage recall emits
    cosine similarities in [0, 1]. Without this the same relevance threshold would
    mean two completely different things depending on whether reranking ran.
    """
    value = float(raw_score)
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _compute_cpu_bge_scores(
    reranker: Any,
    pairs: list[list[str]],
    *,
    batch_size: int = BGE_CPU_BATCH_SIZE,
) -> list[float]:
    """Score CPU pairs once in length-sorted micro-batches."""
    if not pairs:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    import torch

    tokenizer = reranker.tokenizer
    max_length = int(reranker.max_length)
    query_max_length = reranker.query_max_length or max_length * 3 // 4
    queries = tokenizer(
        [pair[0] for pair in pairs],
        return_tensors=None,
        add_special_tokens=False,
        max_length=query_max_length,
        truncation=True,
    )["input_ids"]
    passages = tokenizer(
        [pair[1] for pair in pairs],
        return_tensors=None,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
    )["input_ids"]
    prepared = [
        tokenizer.prepare_for_model(
            query_tokens,
            passage_tokens,
            truncation="only_second",
            max_length=max_length,
            padding=False,
        )
        for query_tokens, passage_tokens in zip(queries, passages, strict=True)
    ]
    order = sorted(
        range(len(prepared)),
        key=lambda index: len(prepared[index]["input_ids"]),
        reverse=True,
    )
    sorted_inputs = [prepared[index] for index in order]

    device = reranker.target_devices[0]
    model = reranker.model.to(device)
    model.eval()
    sorted_scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(sorted_inputs), batch_size):
            inputs = tokenizer.pad(
                sorted_inputs[start : start + batch_size],
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs, return_dict=True).logits.view(-1).float()
            sorted_scores.extend(float(score) for score in logits.cpu().tolist())

    scores = [0.0] * len(pairs)
    for sorted_index, original_index in enumerate(order):
        scores[original_index] = sorted_scores[sorted_index]
    return scores


def _compute_bge_scores(reranker: Any, pairs: list[list[str]]) -> Any:
    target_devices = getattr(reranker, "target_devices", None)
    has_cpu_internals = (
        target_devices
        and str(target_devices[0]).lower().startswith("cpu")
        and hasattr(reranker, "tokenizer")
        and hasattr(reranker, "model")
    )
    if has_cpu_internals:
        return _compute_cpu_bge_scores(reranker, pairs)
    return reranker.compute_score(pairs)


def rerank_chunks(
    query: str,
    matches: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if not matches:
        return []
    if top_k <= 0:
        return []

    pairs = [[query, str(match.get("content", ""))] for match in matches]
    try:
        scores = _compute_bge_scores(_get_bge_reranker(), pairs)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("BGE reranker failed while computing scores") from exc
    if isinstance(scores, int | float):
        scores = [float(scores)]
    else:
        try:
            scores = [float(score) for score in scores]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("BGE reranker returned malformed scores") from exc
    if len(scores) != len(matches):
        raise RuntimeError(
            f"BGE reranker returned {len(scores)} scores for {len(matches)} candidates"
        )

    reranked: list[dict[str, Any]] = []
    for match, rerank_score in zip(matches, scores, strict=False):
        updated = deepcopy(match)
        updated["original_score"] = float(match.get("score", 0.0))
        updated["rerank_logit"] = rerank_score
        updated["rerank_score"] = normalize_rerank_score(rerank_score)
        updated["score"] = updated["rerank_score"]
        reranked.append(updated)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:top_k]
