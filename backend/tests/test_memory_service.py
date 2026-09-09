from datetime import datetime, timedelta

import pytest
from backend.app.db.base import Base
from backend.app.models.memory import Memory
from backend.app.services import memory as memory_service
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ordinary spectral clustering question", 0.50),
        ("请记住我以后偏好中文回答", 0.90),
        ("我决定采用这个模型", 0.75),
        ("REMEMBER that I prefer concise answers", 0.90),
        ("We must use this method", 0.75),
        ("What causes spectral clustering instability?", 0.50),
    ],
)
def test_calculate_memory_importance_uses_deterministic_rules(
    text: str,
    expected: float,
) -> None:
    assert memory_service.calculate_memory_importance(text) == expected


def test_calculate_memory_importance_pinned_is_maximum() -> None:
    assert (
        memory_service.calculate_memory_importance(
            "ordinary topic",
            is_pinned=True,
        )
        == 1.0
    )


def test_memory_model_replaces_confidence_with_policy_fields() -> None:
    assert "confidence" not in Memory.__table__.columns
    assert "importance_score" in Memory.__table__.columns
    assert "status" in Memory.__table__.columns


class FakeMemoryCollection:
    def __init__(self, result: dict | None = None) -> None:
        self.upserts: list[dict] = []
        self.queries: list[dict] = []
        self.result = result or {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    def upsert(self, **kwargs) -> None:
        self.upserts.append(kwargs)

    def query(self, **kwargs) -> dict:
        self.queries.append(kwargs)
        return self.result


@pytest.fixture()
def memory_db() -> Session:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


def test_upsert_memory_vector_stores_bge_m3_document_embedding(monkeypatch) -> None:
    collection = FakeMemoryCollection()
    embed_calls: list[dict] = []

    def fake_embed_text(text: str, input_type: str = "document") -> list[float]:
        embed_calls.append({"text": text, "input_type": input_type})
        return [0.1] * 1024

    monkeypatch.setattr(memory_service, "get_user_memories_collection", lambda: collection)
    monkeypatch.setattr(memory_service, "embed_text", fake_embed_text)

    memory = Memory(
        id=7,
        memory_type="research_topic",
        content="User asked about spectral clustering stability",
        source_type="chat",
        source_id=3,
        importance_score=0.75,
        status="active",
    )

    memory_service.upsert_memory_vector(memory)

    assert embed_calls == [
        {
            "text": "User asked about spectral clustering stability",
            "input_type": "document",
        }
    ]
    assert collection.upserts == [
        {
            "ids": ["7"],
            "documents": ["User asked about spectral clustering stability"],
            "metadatas": [
                {
                    "memory_id": 7,
                    "memory_type": "research_topic",
                    "source_type": "chat",
                    "source_id": 3,
                    "importance_score": 0.75,
                    "status": "active",
                }
            ],
            "embeddings": [[0.1] * 1024],
        }
    ]


def test_query_relevant_memories_filters_metadata_and_returns_fused_score(
    memory_db: Session,
    monkeypatch,
) -> None:
    now = datetime(2026, 8, 10, 12, 0, 0)
    memory_db.add(
        Memory(
            id=7,
            memory_type="research_topic",
            content="SQL authoritative spectral clustering memory",
            source_type="chat",
            source_id=3,
            importance_score=0.5,
            status="active",
            created_at=now,
            updated_at=now,
        )
    )
    memory_db.flush()
    collection = FakeMemoryCollection(
        {
            "ids": [["7"]],
            "documents": [["stale Chroma document"]],
            "metadatas": [[{"memory_id": 7}]],
            "distances": [[0.2]],
        }
    )
    embed_calls: list[dict] = []

    def fake_embed_text(text: str, input_type: str = "document") -> list[float]:
        embed_calls.append({"text": text, "input_type": input_type})
        return [0.2] * 1024

    monkeypatch.setattr(memory_service, "get_user_memories_collection", lambda: collection)
    monkeypatch.setattr(memory_service, "embed_text", fake_embed_text)

    hits = memory_service.query_relevant_memories(
        memory_db,
        "How stable is spectral clustering?",
        limit=2,
        now=now,
    )

    assert embed_calls == [
        {
            "text": "How stable is spectral clustering?",
            "input_type": "query",
        }
    ]
    assert collection.queries[0]["n_results"] == 20
    assert collection.queries[0]["where"] == {
        "$and": [
            {"status": {"$eq": "active"}},
            {"source_type": {"$eq": "chat"}},
            {"memory_type": {"$in": ["research_topic"]}},
        ]
    }
    assert "source_id" not in repr(collection.queries[0]["where"])
    assert hits[0].memory_id == 7
    assert hits[0].content == "SQL authoritative spectral clustering memory"
    assert hits[0].score == 0.795


def test_memory_scoring_uses_half_life_and_clamps_future_dates() -> None:
    now = datetime(2026, 8, 10, 12, 0, 0)

    assert memory_service.calculate_recency_score(now, now=now) == 1.0
    assert memory_service.calculate_recency_score(
        now - timedelta(days=90),
        now=now,
    ) == pytest.approx(0.5)
    assert memory_service.calculate_recency_score(
        now + timedelta(days=1),
        now=now,
    ) == 1.0
    assert memory_service.calculate_final_memory_score(0.8, 0.5, 0.9) == 0.755


def test_query_relevant_memories_applies_threshold_sql_validation_and_ranking(
    memory_db: Session,
    monkeypatch,
) -> None:
    now = datetime(2026, 8, 10, 12, 0, 0)
    memories = [
        Memory(
            id=1,
            memory_type="research_topic",
            content="recent memory from session one",
            source_type="chat",
            source_id=1,
            importance_score=0.5,
            status="active",
            created_at=now,
            updated_at=now,
        ),
        Memory(
            id=2,
            memory_type="research_topic",
            content="important memory from another session",
            source_type="chat",
            source_id=99,
            importance_score=0.9,
            status="active",
            created_at=now - timedelta(days=90),
            updated_at=now - timedelta(days=90),
        ),
        Memory(
            id=3,
            memory_type="research_topic",
            content="inactive memory",
            source_type="chat",
            source_id=2,
            importance_score=1.0,
            status="inactive",
            created_at=now,
            updated_at=now,
        ),
        Memory(
            id=4,
            memory_type="research_topic",
            content="low semantic memory",
            source_type="chat",
            source_id=2,
            importance_score=1.0,
            status="active",
            created_at=now,
            updated_at=now,
        ),
    ]
    memory_db.add_all(memories)
    memory_db.flush()
    collection = FakeMemoryCollection(
        {
            "ids": [["1", "2", "3", "4", "999"]],
            "documents": [["one", "two", "three", "four", "ghost"]],
            "metadatas": [[
                {"memory_id": 1},
                {"memory_id": 2},
                {"memory_id": 3},
                {"memory_id": 4},
                {"memory_id": 999},
            ]],
            "distances": [[0.3, 0.2, 0.01, 0.81, 0.01]],
        }
    )
    monkeypatch.setattr(memory_service, "get_user_memories_collection", lambda: collection)
    monkeypatch.setattr(memory_service, "embed_text", lambda *args, **kwargs: [0.2] * 1024)

    hits = memory_service.query_relevant_memories(
        memory_db,
        "spectral stability",
        limit=5,
        now=now,
    )

    assert [hit.memory_id for hit in hits] == [2, 1]
    assert [hit.source_id for hit in hits] == [99, 1]
    assert [hit.score for hit in hits] == [0.755, 0.73]


def test_query_relevant_memories_uses_timestamp_then_id_as_tiebreakers(
    memory_db: Session,
    monkeypatch,
) -> None:
    now = datetime(2026, 8, 10, 12, 0, 0)
    for memory_id, updated_at in [
        (10, now - timedelta(days=1)),
        (11, now),
        (12, now),
    ]:
        memory_db.add(
            Memory(
                id=memory_id,
                memory_type="research_topic",
                content=f"memory {memory_id}",
                source_type="chat",
                source_id=memory_id,
                importance_score=0.5,
                status="active",
                created_at=updated_at,
                updated_at=updated_at,
            )
        )
    memory_db.flush()
    collection = FakeMemoryCollection(
        {
            "ids": [["10", "11", "12"]],
            "documents": [["ten", "eleven", "twelve"]],
            "metadatas": [[
                {"memory_id": 10},
                {"memory_id": 11},
                {"memory_id": 12},
            ]],
            "distances": [[0.2, 0.2, 0.2]],
        }
    )
    monkeypatch.setattr(memory_service, "get_user_memories_collection", lambda: collection)
    monkeypatch.setattr(memory_service, "embed_text", lambda *args, **kwargs: [0.2] * 1024)

    hits = memory_service.query_relevant_memories(
        memory_db,
        "tie",
        limit=3,
        now=now,
    )

    assert [hit.memory_id for hit in hits] == [12, 11, 10]
