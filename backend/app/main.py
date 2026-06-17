from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.router import api_router
from .core.config import get_settings
from .core.paths import ensure_runtime_dirs
from .db.base import Base
from .db.session import engine
from .models import (
    AgentTrace,
    ChatMessage,
    ChatSession,
    Document,
    DocumentChunk,
    DocumentPage,
    Memory,
    PaperAnalysis,
    TranscriptEvent,
)
from .web import router as web_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    ensure_runtime_dirs()
    Base.metadata.create_all(bind=engine)
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

__all__ = [
    "app",
    "AgentTrace",
    "ChatMessage",
    "ChatSession",
    "Document",
    "DocumentChunk",
    "DocumentPage",
    "Memory",
    "PaperAnalysis",
    "TranscriptEvent",
]
