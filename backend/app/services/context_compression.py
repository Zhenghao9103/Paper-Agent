from ..schemas.chat import Citation


def compress_evidence(
    citations: list[Citation],
    *,
    max_items: int = 5,
    max_chars_per_item: int = 260,
) -> tuple[list[Citation], dict[str, int]]:
    original_chars = sum(len(citation.content) for citation in citations)
    ranked = sorted(citations, key=lambda citation: citation.score, reverse=True)
    kept = [
        citation.model_copy(update={"content": _truncate(citation.content, max_chars_per_item)})
        for citation in ranked[:max_items]
    ]
    compressed_chars = sum(len(citation.content) for citation in kept)
    return kept, {
        "original_items": len(citations),
        "kept_items": len(kept),
        "original_chars": original_chars,
        "compressed_chars": compressed_chars,
        "dropped_items": max(0, len(citations) - len(kept)),
    }


def _truncate(text: str, max_chars: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."
