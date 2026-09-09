from collections.abc import Generator
from pathlib import Path

import pytest
from backend.app.core import paths
from backend.app.db.base import Base
from backend.app.db.session import get_db
from backend.app.main import app
from backend.app.rag import rerank, vector_store
from backend.app.rag.bm25_store import ensure_bm25_schema
from backend.app.services import answering, document_store, ingestion, research
from backend.app.services.intent_router import RoutingOutcome
from backend.app.services.query_planner import QueryPlanningOutcome, fallback_plan
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


class FakeBgeM3Model:
    def encode(self, text, **kwargs):
        # Mirrors SentenceTransformer: a single vector for a string, one per item
        # for a batch.
        if isinstance(text, str):
            return [1.0] * vector_store.EMBEDDING_DIMENSIONS
        return [[1.0] * vector_store.EMBEDDING_DIMENSIONS for _ in text]


class FakeBgeReranker:
    def compute_score(self, pairs) -> list[float]:
        return [1.0 - index * 0.01 for index, _ in enumerate(pairs)]


@pytest.fixture(autouse=True)
def fake_bge_m3_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vector_store, "_get_bge_m3_model", lambda: FakeBgeM3Model())
    monkeypatch.setattr(rerank, "_get_bge_reranker", lambda: FakeBgeReranker())
    monkeypatch.setattr(answering, "answer_json", lambda messages: None)

    def offline_route(question: str, document_id=None, short_term_memory=None):
        del short_term_memory
        lowered = question.casefold()
        intent = "direct" if any(
            marker in lowered for marker in ("how many papers", "有几篇", "list papers")
        ) else "simple_rag"
        return RoutingOutcome(intent=intent)

    def offline_plan(question: str, document_id=None, short_term_memory=None):
        del short_term_memory
        return QueryPlanningOutcome(plan=fallback_plan(question, document_id))

    monkeypatch.setattr(research, "route_question", offline_route)
    monkeypatch.setattr(research, "plan_queries", offline_plan)


@pytest.fixture()
def test_storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    storage_root = tmp_path / "storage"
    chroma_root = tmp_path / "chroma"
    monkeypatch.setattr(paths, "storage_root", lambda: storage_root)
    monkeypatch.setattr(paths, "uploads_dir", lambda: storage_root / "uploads")
    monkeypatch.setattr(paths, "pages_dir", lambda: storage_root / "pages")
    monkeypatch.setattr(paths, "documents_dir", lambda: storage_root / "documents")
    monkeypatch.setattr(paths, "staging_dir", lambda: storage_root / ".staging")
    monkeypatch.setattr(paths, "exports_dir", lambda: storage_root / "exports")
    monkeypatch.setattr(paths, "chroma_dir", lambda: chroma_root)
    monkeypatch.setattr(vector_store, "chroma_dir", lambda: chroma_root)
    monkeypatch.setattr(document_store, "documents_dir", lambda: storage_root / "documents")
    monkeypatch.setattr(
        document_store,
        "document_dir",
        lambda document_id: storage_root / "documents" / str(document_id),
    )
    monkeypatch.setattr(ingestion, "staging_dir", lambda: storage_root / ".staging")
    return storage_root


@pytest.fixture()
def client(tmp_path: Path, test_storage_root: Path) -> Generator[TestClient, None, None]:
    database_url = f"sqlite:///{tmp_path / 'test.db'}"
    test_engine = create_engine(database_url, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    ensure_bm25_schema(test_engine)
    app.state.test_session_factory = TestingSessionLocal

    def override_get_db() -> Generator[Session, None, None]:
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    app.state.test_session_factory = None


@pytest.fixture()
def db_session(tmp_path: Path) -> Generator[Session, None, None]:
    database_url = f"sqlite:///{tmp_path / 'fixture.db'}"
    test_engine = create_engine(database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    ensure_bm25_schema(test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        test_engine.dispose()
