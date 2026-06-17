from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from ..models.chat import TranscriptEvent
from ..models.memory import Memory
from ..rag.vector_store import embed_text, get_user_memories_collection
from ..schemas.chat import Citation, WebSource


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
        "confidence": memory.confidence,
    }
    collection.upsert(
        ids=[str(memory.id)],
        documents=[memory.content],
        metadatas=[{key: value for key, value in metadata.items() if value is not None}],
        embeddings=[embed_text(memory.content, input_type="document")],
    )


def query_relevant_memories(query: str, limit: int = 3) -> list[MemoryHit]:
    collection = get_user_memories_collection()
    result = collection.query(
        query_embeddings=[embed_text(query, input_type="query")],
        n_results=limit,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    hits: list[MemoryHit] = []
    for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
        metadata = metadata or {}
        memory_id = metadata.get("memory_id")
        if memory_id is None:
            continue
        hits.append(
            MemoryHit(
                memory_id=int(memory_id),
                content=str(document),
                memory_type=str(metadata.get("memory_type", "memory")),
                source_id=_optional_int(metadata.get("source_id")),
                score=round(max(0.0, 1.0 - float(distance)), 4),
            )
        )
    return hits


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
    answer: str,
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
    return events


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)
