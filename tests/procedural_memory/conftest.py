import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.db.base import Base
from backend.app import models  # noqa: F401 - register all ORM tables


@pytest.fixture
def semantic_index(tmp_path, monkeypatch):
    import chromadb
    from backend.app.services import memory, memory_harness

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.create_collection("user-memories", metadata={"hnsw:space": "cosine"})
    for module in (memory, memory_harness):
        monkeypatch.setattr(module, "get_user_memories_collection", lambda: collection)
        monkeypatch.setattr(module, "embed_text", lambda *_a, **_kw: [1.0, 0.0])
    return collection


@pytest.fixture
def db_session(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "fixture.db"))
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as session:
            yield session
    finally:
        engine.dispose()
