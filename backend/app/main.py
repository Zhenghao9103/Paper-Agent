from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.router import api_router
from .core.config import get_settings
from .core.paths import ensure_runtime_dirs
from .db.base import Base
from .db.session import SessionLocal, engine
from .models import (
    AgentTrace,
    ChatMessage,
    ChatSession,
    ContextCheckpoint,
    Document,
    DocumentBlock,
    DocumentChunk,
    DocumentChunkDetail,
    DocumentCrossReference,
    DocumentElement,
    DocumentPage,
    DocumentParseRun,
    Memory,
    PaperAnalysis,
    SessionMemory,
    TranscriptEvent,
)
from .rag.bm25_store import ensure_bm25_schema
from .services.document_jobs import enqueue_pending_and_legacy
from .web import router as web_router

STATIC_DIR = Path(__file__).resolve().parent / "static"

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    ensure_runtime_dirs()
    Base.metadata.create_all(bind=engine)
    ensure_bm25_schema(engine)
    session_factory = getattr(app.state, "test_session_factory", None) or SessionLocal
    enqueue_pending_and_legacy(session_factory=session_factory)
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(web_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

__all__ = [
    "app",
    "AgentTrace",
    "ChatMessage",
    "ChatSession",
    "ContextCheckpoint",
    "Document",
    "DocumentChunk",
    "DocumentChunkDetail",
    "DocumentCrossReference",
    "DocumentBlock",
    "DocumentElement",
    "DocumentPage",
    "DocumentParseRun",
    "Memory",
    "PaperAnalysis",
    "SessionMemory",
    "TranscriptEvent",
]
