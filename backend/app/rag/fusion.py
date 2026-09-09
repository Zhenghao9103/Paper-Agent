"""Deterministic reciprocal-rank fusion primitives."""

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import TypeVar

KeyT = TypeVar("KeyT", bound=Hashable)


@dataclass(frozen=True, slots=True)
class FusedRank:
    """A key and its accumulated reciprocal-rank score."""

    key: Hashable
    score: float


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[KeyT]],
    *,
    k: int = 60,
) -> list[FusedRank]:
    """Fuse ranked lists with reciprocal-rank scores.

    Each input ranking contributes at most once for a key; duplicate entries are
    ignored after the first occurrence.  Ties retain first-seen ordering and use
    ``str(key)`` as a final deterministic tie breaker.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero")

    scores: dict[Hashable, float] = {}
    first_seen: dict[Hashable, int] = {}
    seen_count = 0
    for ranking in rankings:
        for rank, key in enumerate(dict.fromkeys(ranking), start=1):
            if key not in first_seen:
                first_seen[key] = seen_count
                seen_count += 1
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    fused = [FusedRank(key=key, score=score) for key, score in scores.items()]
    fused.sort(key=lambda item: (-item.score, first_seen[item.key], str(item.key)))
    return fused
