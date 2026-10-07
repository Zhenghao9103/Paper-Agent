import json
from typing import NoReturn

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..models.chat import ChatMessage, ChatSession, TranscriptEvent
from ..models.trace import AgentTrace
from ..schemas.chat import Citation, WebSource
from .context_checkpoint import (
    CheckpointOutcome,
    ContextCompactionUnavailable,
    compact_session_context,
    is_hard_overflow,
)
from .context_compression import compress_evidence
from .llm import ContextWindowExceededError
from .memory import (
    MemoryHit,
    add_transcript_event,
    build_trace_events,
)
from .memory_extractor import has_explicit_memory_intent
from .memory_harness import process_pending_writes, write_memories
from .progress import ProgressCallback, emit_progress
from .research import run_research

CONTEXT_ERROR_PAYLOAD = {
    "trace": "context_overflow_recovery_failed",
    "message": "上下文压缩未能恢复模型窗口。",
}


def _checkpoint_trace_names(outcome: CheckpointOutcome) -> list[str]:
    if outcome.status == "not_needed":
        return ["context_compaction_not_needed"]
    terminal = {
        "created": "context_compaction_completed",
        "failed": "context_compaction_failed",
    }[outcome.status]
    return ["context_compaction_started", terminal]


def _checkpoint_payload(
    preflight: CheckpointOutcome,
    forced: CheckpointOutcome | None,
    postflight: CheckpointOutcome,
) -> dict[str, int | str | None]:
    for outcome in (postflight, forced, preflight):
        if outcome is not None and outcome.status == "created":
            return outcome.as_payload()
    return postflight.as_payload()


def _record_context_error_and_raise(
    db: Session,
    *,
    session_id: int,
    detail: str,
) -> NoReturn:
    db.rollback()
    add_transcript_event(
        db,
        session_id=session_id,
        event_type="context_error",
        payload=CONTEXT_ERROR_PAYLOAD,
    )
    db.commit()
    raise ContextCompactionUnavailable(detail)


def get_or_create_session(db: Session, session_id: int | None, title: str) -> ChatSession:
    if session_id is not None:
        session = db.get(ChatSession, session_id)
        if session is not None:
            return session

    session = ChatSession(title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def answer_question(
    db: Session,
    *,
    question: str,
    document_id: int | None,
    session_id: int | None,
    forced_intent: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[int, str, list[Citation], list[WebSource], list[str], list[MemoryHit], list[dict]]:
    settings = get_settings()
    session = get_or_create_session(db, session_id, question[:80] or "Research Chat")
    user_turn_count = int(
        db.scalar(
            select(func.count(ChatMessage.id)).where(
                ChatMessage.session_id == session.id,
                ChatMessage.role == "user",
            )
        )
        or 0
    )
    turn_id = f"session-{session.id}-turn-{user_turn_count + 1}"
    memory_write_events: list[dict] = []

    def write_checkpoint(outcome: CheckpointOutcome) -> None:
        if outcome.status == "created":
            # Commit the source before the memory path owns any transactions.
            db.commit()
            memory_write_events.extend(write_memories(
                db, source_session_id=session.id, source_checkpoint_id=outcome.checkpoint_id,
                previous_session_memory=outcome.previous_session_memory,
                current_session_memory=outcome.current_session_memory,
            ))
    emit_progress(progress_callback, "context")
    preflight = compact_session_context(
        db,
        session.id,
        settings=settings,
    )
    write_checkpoint(preflight)
    if preflight.status == "failed" and is_hard_overflow(preflight, settings):
        _record_context_error_and_raise(
            db,
            session_id=session.id,
            detail="上下文已超过模型窗口，且 LLM 压缩失败。",
        )
    if has_explicit_memory_intent(question):
        db.commit()
        memory_write_events.extend(write_memories(
            db, source_session_id=session.id, explicit_intent=question,
        ))
    elif preflight.status != "created":
        # Retry durable work even when this turn does not trigger a new checkpoint.
        memory_write_events.extend(process_pending_writes(db))

    def invoke_research() -> dict:
        result = run_research(
            db,
            question=question,
            document_id=document_id,
            session_id=session.id,
            forced_intent=forced_intent,
            progress_callback=progress_callback,
        )
        return result.model_dump()

    forced: CheckpointOutcome | None = None
    overflow_retried = False
    try:
        result = invoke_research()
    except ContextWindowExceededError:
        forced = compact_session_context(
            db,
            session.id,
            settings=settings,
            force=True,
        )
        if forced.status != "created":
            _record_context_error_and_raise(
                db,
                session_id=session.id,
                detail="上下文溢出恢复失败。",
            )
        write_checkpoint(forced)
        overflow_retried = True
        try:
            result = invoke_research()
        except ContextWindowExceededError:
            _record_context_error_and_raise(
                db,
                session_id=session.id,
                detail="压缩后仍超过模型上下文窗口。",
            )

    answer = result["answer"]
    citations = [
        item if isinstance(item, Citation) else Citation.model_validate(item)
        for item in result.get("citations", [])
    ]
    web_sources = [
        item if isinstance(item, WebSource) else WebSource.model_validate(item)
        for item in result.get("web_sources", [])
    ]
    trace = list(result.get("trace", []))
    extra_trace_events = list(result.get("trace_events", []))
    memory_hits = [
        item if isinstance(item, MemoryHit) else MemoryHit(**item)
        for item in result.get("memory_hits", [])
    ]
    errors = result.get("errors", [])
    trace.extend(_checkpoint_trace_names(preflight))
    if forced is not None:
        trace.extend(_checkpoint_trace_names(forced))
    if overflow_retried:
        trace.append("context_overflow_retry")

    citations, context_compression = compress_evidence(citations)
    if citations:
        trace.append("compress_retrieval_context")

    db.add(ChatMessage(session_id=session.id, role="user", content=question))
    db.add(
        ChatMessage(
            session_id=session.id,
            role="assistant",
            content=answer,
            citations=json.dumps(
                {
                    "local": [citation.model_dump() for citation in citations],
                    "web": [source.model_dump() for source in web_sources],
                }
            ),
        )
    )
    db.flush()
    postflight = compact_session_context(
        db,
        session.id,
        settings=settings,
    )
    trace.extend(_checkpoint_trace_names(postflight))
    context_checkpoint = _checkpoint_payload(
        preflight,
        forced,
        postflight,
    )
    write_checkpoint(postflight)
    extra_trace_events.extend(memory_write_events)

    db.add(
        AgentTrace(
            session_id=session.id,
            user_query=question,
            nodes=json.dumps(trace),
            final_answer=answer,
        )
    )

    trace_events = build_trace_events(
        trace=trace,
        citations=citations,
        web_sources=web_sources,
        memory_hits=memory_hits,
        errors=errors,
        context_compression=context_compression,
        context_checkpoint=context_checkpoint,
        extra_events=extra_trace_events,
        answer=answer,
        turn_id=turn_id,
    )
    for event in trace_events:
        add_transcript_event(
            db,
            session_id=session.id,
            event_type="trace_event",
            payload=event,
        )
    db.add(
        TranscriptEvent(
            session_id=session.id,
            event_type="qa",
            payload=json.dumps(
                {
                    "question": question,
                    "answer": answer,
                    "trace": trace,
                    "trace_events": trace_events,
                    "context_compression": context_compression,
                    "context_checkpoint": context_checkpoint,
                },
                ensure_ascii=False,
            ),
        )
    )
    db.commit()
    return session.id, answer, citations, web_sources, trace, memory_hits, trace_events
