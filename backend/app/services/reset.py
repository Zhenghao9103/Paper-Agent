from sqlalchemy import delete, select, func
from sqlalchemy.orm import Session

from ..models.analysis import PaperAnalysis
from ..models.chat import ChatMessage, ChatSession, TranscriptEvent
from ..models.chunk import DocumentChunk
from ..models.document import Document
from ..models.memory import Memory
from ..models.page import DocumentPage
from ..models.trace import AgentTrace
from ..rag.vector_store import reset_vector_collections


RESET_MODEL_ORDER = [
    PaperAnalysis,
    AgentTrace,
    TranscriptEvent,
    ChatMessage,
    Memory,
    DocumentChunk,
    DocumentPage,
    Document,
    ChatSession,
]


def clear_all_runtime_data(db: Session) -> dict[str, int]:
    deleted: dict[str, int] = {}
    for model in RESET_MODEL_ORDER:
        table_name = model.__tablename__
        deleted[table_name] = int(db.scalar(select(func.count()).select_from(model)) or 0)
        db.execute(delete(model))
    db.commit()
    reset_vector_collections()
    return deleted
