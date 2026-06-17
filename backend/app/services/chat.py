import json

from sqlalchemy.orm import Session

from ..agents.research import build_research_graph
from ..models.chat import ChatMessage, ChatSession, TranscriptEvent
from ..models.memory import Memory
from ..models.trace import AgentTrace
from ..schemas.chat import Citation, WebSource
from .context_compression import compress_evidence, update_session_summary_if_needed
from .memory import (
    MemoryHit,
    add_transcript_event,
    build_trace_events,
    upsert_memory_vector,
)


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
) -> tuple[int, str, list[Citation], list[WebSource], list[str], list[MemoryHit], list[dict]]:
    session = get_or_create_session(db, session_id, question[:80] or "Research Chat")
    result = build_research_graph().invoke(
        {"question": question, "document_id": document_id, "session_id": session.id, "db": db}
    )
    answer = result["answer"]
    citations = result.get("citations", [])
    web_sources = result.get("web_sources", [])
    trace = result.get("trace", [])
    memory_hits = result.get("memory_hits", [])
    errors = result.get("errors", [])
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
    db.add(
        AgentTrace(
            session_id=session.id,
            user_query=question,
            nodes=json.dumps(trace),
            final_answer=answer,
        )
    )
    if citations and len(question) > 12:
        memory = Memory(
            memory_type="research_topic",
            content=f"User asked about: {question[:240]}",
            source_type="chat",
            source_id=session.id,
            confidence=0.6,
        )
        db.add(memory)
        db.flush()
        try:
            upsert_memory_vector(memory)
            trace.append("update_long_term_vector_memory")
        except Exception:
            trace.append("long_term_memory_vector_upsert_failed")

    if update_session_summary_if_needed(db, session):
        trace.append("compress_short_term_memory")

    trace_events = build_trace_events(
        trace=trace,
        citations=citations,
        web_sources=web_sources,
        memory_hits=memory_hits,
        errors=errors,
        context_compression=context_compression,
        answer=answer,
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
                },
                ensure_ascii=False,
            ),
        )
    )
    db.commit()
    return session.id, answer, citations, web_sources, trace, memory_hits, trace_events
