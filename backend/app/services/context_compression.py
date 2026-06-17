from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.chat import ChatMessage, ChatSession
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


def update_session_summary_if_needed(
    db: Session,
    session: ChatSession,
    *,
    message_threshold: int = 6,
) -> bool:
    messages = list(
        db.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session.id)
            .order_by(ChatMessage.created_at, ChatMessage.id)
        )
    )
    if len(messages) < message_threshold:
        return False

    user_messages = [message.content for message in messages if message.role == "user"]
    if not user_messages:
        return False

    session.session_summary = _build_session_summary(user_messages)
    db.add(session)
    return True


def _build_session_summary(user_messages: list[str]) -> str:
    recent = user_messages[-3:]
    topics = "；".join(_truncate(message, 80) for message in recent)
    return f"本轮会话主要围绕这些问题展开：{topics}"


def _truncate(text: str, max_chars: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."
