import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from ...db.session import SessionLocal, get_db
from ...schemas.chat import ChatRequest, ChatResponse
from ...services.chat import answer_question
from ...services.context_checkpoint import ContextCompactionUnavailable

router = APIRouter(prefix="/chat", tags=["chat"])


def _run_question(request: ChatRequest, db: Session):
    try:
        return answer_question(
            db,
            question=request.question,
            document_id=request.document_id,
            session_id=request.session_id,
        )
    except ContextCompactionUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/ask", response_model=ChatResponse)
def ask_question(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    session_id, answer, citations, web_sources, trace, memory_hits, trace_events = (
        _run_question(request, db)
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


def _sse(event: str, payload: dict[str, Any]) -> dict[str, str]:
    return {
        "event": event,
        "data": json.dumps(payload, ensure_ascii=False),
    }


def _safe_error(exc: Exception) -> dict[str, str]:
    if isinstance(exc, ContextCompactionUnavailable):
        return {
            "type": "error",
            "code": "context_unavailable",
            "message": "上下文处理失败，请缩短问题后重试。",
        }
    return {
        "type": "error",
        "code": "request_failed",
        "message": "模型调用失败，请稍后重试。",
    }


async def _stream_events(
    request: ChatRequest,
    *,
    session_factory: Callable[[], Session] = SessionLocal,
    heartbeat_seconds: float = 5.0,
) -> AsyncIterator[dict[str, str]]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    started = loop.time()

    def publish(kind: str, payload: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

    def worker() -> None:
        db: Session | None = None
        try:
            db = session_factory()
            result = answer_question(
                db,
                question=request.question,
                document_id=request.document_id,
                session_id=request.session_id,
                progress_callback=lambda event: publish("status", event),
            )
            publish("result", result)
        except Exception as exc:
            if db is not None:
                try:
                    db.rollback()
                except Exception:
                    pass
            publish("failure", exc)
        finally:
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass

    yield _sse(
        "status",
        {
            "type": "status",
            "code": "request_received",
            "message": "已收到问题，正在开始处理",
        },
    )
    worker_task = asyncio.create_task(asyncio.to_thread(worker))
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(
                    queue.get(),
                    timeout=heartbeat_seconds,
                )
            except TimeoutError:
                elapsed = max(0, int(loop.time() - started))
                yield _sse(
                    "heartbeat",
                    {
                        "type": "heartbeat",
                        "code": "heartbeat",
                        "message": f"仍在处理中，已用时 {elapsed} 秒",
                        "elapsed_seconds": elapsed,
                    },
                )
                continue

            if kind == "status":
                yield _sse("status", {"type": "status", **payload})
                continue
            if kind == "failure":
                yield _sse("error", _safe_error(payload))
                break

            session_id, answer, citations, *_ = payload
            citation_chunk_ids = sorted(
                {
                    int(citation.chunk_id)
                    for citation in citations
                    if citation.chunk_id is not None
                }
            )
            yield _sse(
                "session",
                {"type": "session", "session_id": session_id},
            )
            yield _sse(
                "final",
                {
                    "type": "final_answer",
                    "session_id": session_id,
                    "answer": answer,
                    "citation_chunk_ids": citation_chunk_ids,
                },
            )
            break
    finally:
        await asyncio.shield(worker_task)


@router.post("/stream")
def stream_question(request: ChatRequest) -> EventSourceResponse:
    return EventSourceResponse(_stream_events(request), ping=None)


def _sse_event_name(event: dict) -> str:
    event_type = event.get("type")
    if event_type == "retrieval_hit":
        return "citation"
    if event_type == "memory_hit":
        return "memory"
    if event_type == "final_answer":
        return "final"
    return "trace"
