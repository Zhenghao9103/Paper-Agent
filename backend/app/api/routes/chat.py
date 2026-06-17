import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from ...db.session import get_db
from ...schemas.chat import ChatRequest, ChatResponse
from ...services.chat import answer_question

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=ChatResponse)
def ask_question(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    session_id, answer, citations, web_sources, trace, memory_hits, trace_events = answer_question(
        db,
        question=request.question,
        document_id=request.document_id,
        session_id=request.session_id,
    )
    return ChatResponse(
        session_id=session_id,
        answer=answer,
        citations=citations,
        web_sources=web_sources,
        trace=trace,
        memory_hits=memory_hits,
        trace_events=trace_events,
    )


@router.post("/stream")
def stream_question(request: ChatRequest, db: Session = Depends(get_db)) -> EventSourceResponse:
    session_id, answer, citations, web_sources, trace, memory_hits, trace_events = answer_question(
        db,
        question=request.question,
        document_id=request.document_id,
        session_id=request.session_id,
    )

    async def event_generator():
        yield {
            "event": "session",
            "data": json.dumps({"type": "session", "session_id": session_id}),
        }
        for event in trace_events:
            event_name = _sse_event_name(event)
            yield {"event": event_name, "data": json.dumps(event, ensure_ascii=False)}

    return EventSourceResponse(event_generator())


def _sse_event_name(event: dict) -> str:
    event_type = event.get("type")
    if event_type == "retrieval_hit":
        return "citation"
    if event_type == "memory_hit":
        return "memory"
    if event_type == "final_answer":
        return "final"
    return "trace"
