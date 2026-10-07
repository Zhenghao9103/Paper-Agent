"""Validated memory mutations and a durable, replayable SQLite -> Chroma outbox.

Call write_memories only after committing the source checkpoint/messages. Staging
and synchronization own their transactions; vectors never precede durable intent.
The existing single-process server serializes writers with one reentrant lock.
"""

import json
import logging
import math
import re
import threading
import unicodedata
from datetime import datetime
from typing import Any

from pydantic import ValidationError
from sqlalchemy import inspect, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..models.memory import Memory, MemorySyncOperation
from ..rag.vector_store import (
    collection_space,
    distance_to_similarity,
    embed_text,
    get_user_memories_collection,
)
from ..schemas.memory import MemoryCandidate, MemorySuggestion
from .memory import add_transcript_event, upsert_memory_vector
from .memory_extractor import extract_memories, recommend_memory

_WRITE_LOCK = threading.RLock()
_EVIDENCE_MARKER = re.compile(r"\[doc:\d+\|page:|\[(?:E|chunk:)[0-9]+\]")
logger = logging.getLogger(__name__)


def validate_memory_schema(engine: Engine) -> None:
    inspector = inspect(engine)
    if not inspector.has_table("memories"):
        return
    columns = {column["name"] for column in inspector.get_columns("memories")}
    if "importance_score" in columns or not inspector.has_table("memory_sync_operations"):
        raise RuntimeError(
            "Memory schema requires migration. From the project root run: "
            "python -m alembic -c backend/alembic.ini upgrade head"
        )


def _normalized(content: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", content).casefold().split())


def _event(action: str, status: str, reason: str, **fields: Any) -> dict[str, Any]:
    return {"type": "memory_write", "action": action, "status": status,
            "reason": reason, **fields}


def _snapshot(memory: Memory) -> dict[str, Any]:
    return {"memory_id": memory.id, "memory_type": memory.memory_type,
            "content": memory.content, "status": memory.status,
            "updated_at": memory.updated_at.isoformat()}


def _related_memories(
    db: Session, candidate: MemoryCandidate, existing: list[Memory]
) -> list[dict[str, Any]]:
    settings = get_settings()
    collection = get_user_memories_collection()
    result = collection.query(
        query_embeddings=[embed_text(candidate.content, input_type="query")],
        n_results=settings.memory_related_limit,
        where={"$and": [{"status": {"$eq": "active"}},
                        {"memory_type": {"$eq": candidate.memory_type}},
                        {"source_type": {"$eq": "chat"}}]},
        include=["documents", "metadatas", "distances"],
    )
    by_id = {memory.id: memory for memory in existing}
    related = []
    seen = set()
    validated_count = 0
    space = collection_space(collection)
    for metadata, document, distance in zip(
        result.get("metadatas", [[]])[0], result.get("documents", [[]])[0],
        result.get("distances", [[]])[0], strict=True,
    ):
        memory_id = (metadata or {}).get("memory_id")
        if type(memory_id) is not int or memory_id not in by_id:
            continue
        memory = by_id[memory_id]
        if document != memory.content or not math.isfinite(float(distance)):
            raise ValueError("memory_index_inconsistent")
        validated_count += 1
        if (memory_id not in seen and distance_to_similarity(distance, space)
                >= settings.memory_related_threshold):
            related.append(_snapshot(memory))
            seen.add(memory_id)
    if not validated_count:
        raise ValueError("memory_index_incomplete")
    return related


def _final_row(memory: Memory, content: str, status: str, *, changed: bool = False) -> dict:
    return {"memory_id": memory.id, "content": content, "status": status,
            "refresh_updated_at": changed, "replace_metadata": False}


def apply_candidate(
    db: Session,
    candidate: MemoryCandidate,
    *,
    candidate_index: int,
    source_session_id: int,
    source_checkpoint_id: int | None = None,
) -> dict[str, Any]:
    """Stage one validated candidate durably, without touching Chroma."""
    with _WRITE_LOCK:
        try:
            candidate = MemoryCandidate.model_validate(
                candidate.model_dump() if isinstance(candidate, MemoryCandidate) else candidate)
            if candidate.confidence < get_settings().memory_min_confidence:
                return _event("IGNORE", "ignored", "low_confidence")
            if _EVIDENCE_MARKER.search(candidate.content):
                return _event("IGNORE", "ignored", "evidence_marker")
            existing = list(db.scalars(select(Memory).where(
                Memory.memory_type == candidate.memory_type,
                Memory.source_type == "chat",
                Memory.status.in_(("active", "pending_sync")),
            )))
            normalized = _normalized(candidate.content)
            if any(_normalized(memory.content) == normalized for memory in existing):
                return _event("IGNORE", "ignored", "exact_duplicate")
            # Pending content is deliberately not authoritative for semantic updates.
            if any(memory.status == "pending_sync" for memory in existing):
                return _event("IGNORE", "ignored", "same_type_sync_pending")
            related = _related_memories(db, candidate, existing) if existing else []
            proposal = (recommend_memory(
                candidate_index=candidate_index, candidate=candidate,
                existing_memories=related,
            ) if related else MemorySuggestion(
                candidate_index=candidate_index, action="ADD", reason="no_related_memory"))
            advice = MemorySuggestion.model_validate(
                proposal.model_dump() if isinstance(proposal, MemorySuggestion) else proposal)
            if advice.candidate_index != candidate_index:
                return _event("IGNORE", "ignored", "invalid_candidate_index")
            supplied = {item["memory_id"]: item for item in related}
            if advice.target_memory_id is not None and advice.target_memory_id not in supplied:
                return _event("IGNORE", "ignored", "invalid_target")
            # Refresh after model egress: recommendations never authorize stale writes.
            db.expire_all()
            for memory_id, expected in supplied.items():
                current = db.get(Memory, memory_id)
                if current is None or _snapshot(current) != expected:
                    return _event("IGNORE", "ignored", "stale_target")
            if advice.action == "IGNORE":
                return _event("IGNORE", "ignored", "semantic_duplicate_or_uncertain")
            target = db.get(Memory, advice.target_memory_id) if advice.target_memory_id else None
            content = advice.merged_content if advice.action == "MERGE" else candidate.content
            assert content is not None
            if _EVIDENCE_MARKER.search(content):
                return _event("IGNORE", "ignored", "evidence_marker")
            if target and _normalized(target.content) == _normalized(content):
                return _event("IGNORE", "ignored", "unchanged_content")
            if any(memory.id != advice.target_memory_id
                   and _normalized(memory.content) == _normalized(content)
                   for memory in existing):
                return _event("IGNORE", "ignored", "exact_duplicate")
            rows = []
            if target:
                expected_time = target.updated_at
                modified = db.execute(update(Memory).where(
                    Memory.id == target.id, Memory.status == "active",
                    Memory.updated_at == expected_time,
                ).values(status="pending_sync", updated_at=Memory.updated_at))
                if modified.rowcount != 1:
                    db.rollback()
                    return _event("IGNORE", "ignored", "stale_target")
                rows.append(_final_row(
                    target, target.content if advice.action == "SUPERSEDE" else content,
                    "superseded" if advice.action == "SUPERSEDE" else "active",
                    changed=advice.action in {"UPDATE", "MERGE"},
                ))
            if advice.action in {"ADD", "SUPERSEDE"}:
                new_memory = Memory(
                    memory_type=candidate.memory_type, content=content,
                    source_type="chat", source_id=source_session_id, status="pending_sync",
                    is_pinned=target.is_pinned if target else False,
                )
                db.add(new_memory)
                db.flush()
                rows.append(_final_row(new_memory, content, "active"))
            operation = MemorySyncOperation(
                action=advice.action, payload=json.dumps(rows, ensure_ascii=False),
                source_session_id=source_session_id, source_checkpoint_id=source_checkpoint_id,
            )
            db.add(operation)
            db.flush()
            operation_id = operation.id
            db.commit()
            return _event(advice.action, "pending", "sync_staged", operation_id=operation_id)
        except Exception as exc:
            db.rollback()
            reason = "invalid_schema" if isinstance(exc, ValidationError) else "memory_write_failed"
            logger.warning("Memory candidate ignored: %s", type(exc).__name__)
            return _event("IGNORE", "ignored", reason)


def process_pending_writes(
    db: Session, *, operation_ids: list[int] | None = None,
    exclude_ids: set[int] | None = None, limit: int | None = None,
) -> list[dict[str, Any]]:
    """Replay bounded durable operations; active SQL rows require all vector ACKs."""
    events = []
    with _WRITE_LOCK:
        statement = select(MemorySyncOperation.id).where(MemorySyncOperation.status == "pending")
        if operation_ids is not None:
            statement = statement.where(MemorySyncOperation.id.in_(operation_ids))
        if exclude_ids:
            statement = statement.where(MemorySyncOperation.id.not_in(exclude_ids))
        ids = list(db.scalars(statement.order_by(MemorySyncOperation.id).limit(
            min(limit or get_settings().memory_sync_batch_size,
                get_settings().memory_sync_batch_size))))
        for operation_id in ids:
            action = "REINDEX"
            try:
                operation = db.get(MemorySyncOperation, operation_id)
                if operation is None or operation.status != "pending":
                    continue
                action = operation.action
                rows = json.loads(operation.payload)
                if not rows or not isinstance(rows, list):
                    raise ValueError("invalid_sync_payload")
                memories = []
                for row in rows:
                    memory = db.get(Memory, row["memory_id"])
                    if memory is None or memory.status != "pending_sync":
                        raise ValueError("invalid_sync_target")
                    memories.append(memory)
                for row, memory in zip(rows, memories, strict=True):
                    vector_memory = Memory(
                        id=memory.id, memory_type=memory.memory_type, content=row["content"],
                        source_type=memory.source_type, source_id=memory.source_id,
                        status=row["status"],
                    )
                    upsert_memory_vector(vector_memory, replace_metadata=row["replace_metadata"])
                for row, memory in zip(rows, memories, strict=True):
                    timestamp = (datetime.utcnow() if row["refresh_updated_at"]
                                 and memory.content != row["content"] else memory.updated_at)
                    db.execute(update(Memory).where(Memory.id == memory.id).values(
                        content=row["content"], status=row["status"], updated_at=timestamp,
                    ))
                operation.status = "done"
                operation.attempts += 1
                operation.last_error = None
                db.commit()
                events.append(_event(
                    action, "applied", "sync_completed", operation_id=operation_id))
            except Exception as exc:
                db.rollback()
                safe_error = type(exc).__name__
                try:
                    operation = db.get(MemorySyncOperation, operation_id)
                    if operation is not None:
                        operation.attempts += 1
                        operation.last_error = safe_error
                        db.commit()
                except Exception:
                    db.rollback()
                logger.warning("Memory sync %s remains pending: %s", operation_id, safe_error)
                events.append(_event(
                    action, "failed", "sync_pending_retry", operation_id=operation_id))
    return events


def write_memories(
    db: Session, *, source_session_id: int, source_checkpoint_id: int | None = None,
    previous_session_memory: Any = None, current_session_memory: Any = None,
    explicit_intent: str | None = None,
) -> list[dict[str, Any]]:
    """Shared checkpoint/explicit path. Extraction failures never undo their source."""
    # Recovered prior work must become authoritative before judging a new instruction.
    events = process_pending_writes(db)
    attempted = {event["operation_id"] for event in events}
    budget = get_settings().memory_sync_batch_size
    try:
        candidates = extract_memories(
            previous_session_memory=previous_session_memory,
            current_session_memory=current_session_memory, explicit_intent=explicit_intent,
        )
        for index, candidate in enumerate(candidates):
            event = apply_candidate(
                db, candidate, candidate_index=index, source_session_id=source_session_id,
                source_checkpoint_id=source_checkpoint_id,
            )
            events.append(event)
            operation_id = event.get("operation_id")
            if operation_id is not None and len(attempted) < budget:
                attempted.add(operation_id)
                events.extend(process_pending_writes(db, operation_ids=[operation_id], limit=1))
    except Exception as exc:
        db.rollback()
        logger.warning("Memory extraction skipped: %s", type(exc).__name__)
        events.append(_event("IGNORE", "failed", "extraction_failed"))
    if len(attempted) < budget:
        events.extend(process_pending_writes(
            db, exclude_ids=attempted, limit=budget-len(attempted)))
    for event in events:
        add_transcript_event(
            db, session_id=source_session_id, event_type="memory_write", payload=event)
    db.commit()
    return events
