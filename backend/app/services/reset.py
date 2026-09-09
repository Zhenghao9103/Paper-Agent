from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ..models.analysis import PaperAnalysis
from ..models.chat import ChatMessage, ChatSession, TranscriptEvent
from ..models.chunk import DocumentChunk
from ..models.context_checkpoint import ContextCheckpoint
from ..models.document import Document
from ..models.memory import Memory
from ..models.page import DocumentPage
from ..models.session_memory import SessionMemory
from ..models.trace import AgentTrace
from ..rag.bm25_store import clear_bm25_index
from ..rag.vector_store import reset_vector_collections

RESET_MODEL_ORDER = [
    PaperAnalysis,
    AgentTrace,
    TranscriptEvent,
    Memory,
    SessionMemory,
    ContextCheckpoint,
    ChatMessage,
    DocumentChunk,
    DocumentPage,
    Document,
    ChatSession,
]


def clear_all_runtime_data(db: Session) -> dict[str, int]:
    deleted: dict[str, int] = {"document_chunks_fts": clear_bm25_index(db)}
    for model in RESET_MODEL_ORDER:
        table_name = model.__tablename__
        deleted[table_name] = int(db.scalar(select(func.count()).select_from(model)) or 0)
        db.execute(delete(model))
    db.commit()
    reset_vector_collections()
    return deleted
