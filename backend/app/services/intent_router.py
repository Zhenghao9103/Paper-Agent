"""Intent routing for the Paper Agent production chain.

The Router only classifies intent. It never plans queries, never answers, and
never emits confidence; any protocol, provider, or timeout failure falls back
conservatively to ``agentic_rag`` so a broken Router cannot lose evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from . import model_clients
from .router_contract import (
    FALLBACK_INTENT,
    ROUTER_SYSTEM_PROMPT,
    IntentName,
    RouterDecision,
    RoutingOutcome,
    build_router_user_content,
    safe_router_error_category,
)

__all__ = [
    "FALLBACK_INTENT",
    "ROUTER_SYSTEM_PROMPT",
    "IntentName",
    "RoutingOutcome",
    "RouterDecision",
    "build_router_messages",
    "route_question",
    "router_json",
]


def router_json(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Indirection kept patchable for tests and alternate OpenAI-compatible clients."""

    return model_clients.router_json(messages)


def build_router_messages(
    question: str,
    short_term_memory: Any = None,
) -> list[dict[str, str]]:
    """Build the frozen prompt with the last four session messages as context."""

    return [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_router_user_content(
                _session_messages(short_term_memory), question
            ),
        },
    ]


def route_question(
    question: str,
    document_id: int | None = None,
    short_term_memory: Any = None,
) -> RoutingOutcome:
    """Classify one question; every failure degrades to ``agentic_rag``.

    ``document_id`` is accepted for call-site compatibility but is deliberately
    ignored: the Router does not own the request's data scope.
    """

    del document_id
    try:
        payload = router_json(build_router_messages(question, short_term_memory))
        decision = RouterDecision.model_validate(payload)
        return RoutingOutcome(intent=decision.intent)
    except Exception as exc:
        return RoutingOutcome(
            intent=FALLBACK_INTENT,
            degraded=True,
            error=safe_router_error_category(exc),
        )


def _session_messages(memory: Any) -> list[dict[str, str]]:
    """Render bounded session context entries in role/content message form."""

    if memory is None:
        return []
    rendered: list[dict[str, str]] = []
    if isinstance(memory, Mapping):
        session_memory = memory.get("session_memory")
        if isinstance(session_memory, Mapping):
            lines = [
                f"{key}: {str(value).strip()}"
                for key, value in session_memory.items()
                if str(value).strip()
            ]
            if lines:
                rendered.append({"role": "user", "content": "\n".join(lines)})
        elif isinstance(session_memory, str) and session_memory.strip():
            rendered.append({"role": "user", "content": session_memory})
        summary = str(memory.get("summary", memory.get("session_summary", "")) or "").strip()
        if summary:
            rendered.append({"role": "user", "content": f"会话摘要：{summary}"})
        recent = memory.get(
            "messages", memory.get("recent", memory.get("recent_messages"))
        )
        if isinstance(recent, Sequence) and not isinstance(recent, (str, bytes)):
            for item in recent:
                if isinstance(item, Mapping):
                    rendered.append(
                        {
                            "role": str(item.get("role", "user")) or "user",
                            "content": str(item.get("content", "")),
                        }
                    )
                elif str(item).strip():
                    rendered.append({"role": "user", "content": str(item)})
        elif recent and str(recent).strip():
            rendered.append({"role": "user", "content": str(recent)})
    elif isinstance(memory, Sequence) and not isinstance(memory, (str, bytes)):
        for item in memory:
            if isinstance(item, Mapping):
                rendered.append(
                    {
                        "role": str(item.get("role", "user")) or "user",
                        "content": str(item.get("content", "")),
                    }
                )
            elif str(item).strip():
                rendered.append({"role": "user", "content": str(item)})
    else:
        rendered.append({"role": "user", "content": str(memory)})
    return rendered
