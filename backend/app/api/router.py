from fastapi import APIRouter

from .routes import arxiv, chat, documents, health, memory

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(arxiv.router)
api_router.include_router(memory.router)
