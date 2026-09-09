import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.chat import TranscriptEvent
from ..models.memory import Memory
from ..rag.vector_store import (
    collection_space,
    distance_to_similarity,
    embed_text,
    get_user_memories_collection,
)
from ..schemas.chat import Citation, WebSource

PREFERENCE_MARKERS = (
    "记住",
    "以后",
    "偏好",
    "长期",
    "always",
    "remember",
    "prefer",
)
DECISION_MARKERS = (
    "决定",
    "采用",
    "选择",
    "不要",
    "必须",
    "禁止",
    "must",
    "do not",
)
MEMORY_CANDIDATE_LIMIT = 20
MIN_MEMORY_SEMANTIC_SCORE = 0.20
MEMORY_RECENCY_HALF_LIFE_DAYS = 90.0
SEMANTIC_WEIGHT = 0.65
RECENCY_WEIGHT = 0.20
IMPORTANCE_WEIGHT = 0.15
ACTIVE_MEMORY_STATUS = "active"
RETRIEVABLE_MEMORY_TYPES = ("research_topic",)


def _contains_marker(text: str, marker: str) -> bool:
    if not marker.isascii():
        return marker in text
    return re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(marker)}(?![A-Za-z0-9_])",
        text,
    ) is not None


def calculate_memory_importance(text: str, *, is_pinned: bool = False) -> float:
    if is_pinned:
        return 1.0
    lowered = text.casefold()
    if any(_contains_marker(lowered, marker) for marker in PREFERENCE_MARKERS):
        return 0.9
    if any(_contains_marker(lowered, marker) for marker in DECISION_MARKERS):
        return 0.75
    return 0.5


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def calculate_recency_score(
    timestamp: datetime,
    *,
    now: datetime | None = None,
) -> float:
    reference = now or datetime.utcnow()
    age_days = max(0.0, (reference - timestamp).total_seconds() / 86400.0)
    return _clamp_score(2 ** (-age_days / MEMORY_RECENCY_HALF_LIFE_DAYS))


def calculate_final_memory_score(
    semantic_score: float,
    recency_score: float,
    importance_score: float,
) -> float:
    return round(
        SEMANTIC_WEIGHT * _clamp_score(semantic_score)
        + RECENCY_WEIGHT * _clamp_score(recency_score)
        + IMPORTANCE_WEIGHT * _clamp_score(importance_score),
        4,
    )


@dataclass(frozen=True)
class MemoryHit:
    memory_id: int
    content: str
    memory_type: str
    source_id: int | None
    score: float


def upsert_memory_vector(memory: Memory) -> None:
    collection = get_user_memories_collection()
    metadata = {
        "memory_id": memory.id,
        "memory_type": memory.memory_type,
        "source_type": memory.source_type,
        "source_id": memory.source_id,
        "importance_score": memory.importance_score,
        "status": memory.status,
    }
    collection.upsert(
        ids=[str(memory.id)],
        documents=[memory.content],
        metadatas=[{key: value for key, value in metadata.items() if value is not None}],
        embeddings=[embed_text(memory.content, input_type="document")],
    )


def query_relevant_memories(
    db: Session,
    query: str,
    limit: int = 3,
    *,
    now: datetime | None = None,
) -> list[MemoryHit]:
    collection = get_user_memories_collection()
    result = collection.query(
        query_embeddings=[embed_text(query, input_type="query")],
        n_results=max(MEMORY_CANDIDATE_LIMIT, limit),
        where={
            "$and": [
                {"status": {"$eq": ACTIVE_MEMORY_STATUS}},
                {"source_type": {"$eq": "chat"}},
                {"memory_type": {"$in": list(RETRIEVABLE_MEMORY_TYPES)}},
            ]
        },
        include=["documents", "metadatas", "distances"],
    )

    space = collection_space(collection)
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    semantic_scores: dict[int, float] = {}
    for metadata, distance in zip(metadatas, distances, strict=False):
        metadata = metadata or {}
        memory_id = metadata.get("memory_id")
        if memory_id is None:
            continue
        try:
            normalized_id = int(memory_id)
        except (TypeError, ValueError):
            continue
        semantic_score = distance_to_similarity(distance, space)
        if semantic_score < MIN_MEMORY_SEMANTIC_SCORE:
            continue
        semantic_scores[normalized_id] = max(
            semantic_score,
            semantic_scores.get(normalized_id, 0.0),
        )

    if not semantic_scores:
        return []

    memories = list(
        db.scalars(
            select(Memory).where(
                Memory.id.in_(semantic_scores),
                Memory.status == ACTIVE_MEMORY_STATUS,
                Memory.source_type == "chat",
                Memory.memory_type.in_(RETRIEVABLE_MEMORY_TYPES),
            )
        )
    )
    scored: list[tuple[MemoryHit, datetime]] = []
    for memory in memories:
        timestamp = memory.updated_at or memory.created_at or now or datetime.utcnow()
        importance_score = 1.0 if memory.is_pinned else memory.importance_score
        final_score = calculate_final_memory_score(
            semantic_scores[memory.id],
            calculate_recency_score(timestamp, now=now),
            importance_score,
        )
        scored.append(
            (
                MemoryHit(
                    memory_id=memory.id,
                    content=memory.content,
                    memory_type=memory.memory_type,
                    source_id=memory.source_id,
                    score=final_score,
                ),
                timestamp,
            )
        )

    scored.sort(
        key=lambda item: (item[0].score, item[1], item[0].memory_id),
        reverse=True,
    )
    return [hit for hit, _ in scored[:limit]]


def add_transcript_event(
    db: Session,
    *,
    session_id: int | None,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    db.add(
        TranscriptEvent(
            session_id=session_id,
            event_type=event_type,
            payload=_json_dumps(payload),
        )
    )


def build_trace_events(
    *,
    trace: list[str],
    citations: list[Citation],
    web_sources: list[WebSource],
    memory_hits: list[MemoryHit],
    errors: list[dict[str, Any]] | None = None,
    context_compression: dict[str, int] | None = None,
    context_checkpoint: dict[str, int | str | None] | None = None,
    extra_events: list[dict[str, Any]] | None = None,
    answer: str,
    turn_id: str | None = None,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {"type": "node", "node": node, "message": f"Agent node completed: {node}"}
        for node in trace
    ]
    events.extend(
        {
            "type": "memory_hit",
            "memory_id": hit.memory_id,
            "memory_type": hit.memory_type,
            "content": hit.content,
            "score": hit.score,
        }
        for hit in memory_hits
    )
    events.extend(
        {
            "type": "retrieval_hit",
            "document_id": citation.document_id,
            "title": citation.title,
            "page_number": citation.page_number,
            "chunk_index": citation.chunk_index,
            "score": citation.score,
        }
        for citation in citations
    )
    events.extend(
        {
            "type": "error",
            "node": error.get("node"),
            "message": error.get("message", ""),
            "recoverable": bool(error.get("recoverable", True)),
        }
        for error in (errors or [])
    )
    if context_compression is not None:
        events.append({"type": "context_compression", **context_compression})
    if context_checkpoint is not None:
        events.append({"type": "context_checkpoint", **context_checkpoint})
    if extra_events:
        # Research events are already sanitized at the orchestration boundary.
        # Copy each mapping so callers cannot mutate the persisted transcript.
        events.extend(dict(event) for event in extra_events)
    events.extend(
        {
            "type": "web_source",
            "title": source.title,
            "published": source.published,
            "entry_url": source.entry_url,
        }
        for source in web_sources
    )
    events.append({"type": "final_answer", "answer": answer})
    if turn_id is not None:
        for sequence, event in enumerate(events, start=1):
            event["turn_id"] = turn_id
            event["sequence"] = sequence
            event["event_id"] = f"{turn_id}:{sequence}"
    return events


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)
