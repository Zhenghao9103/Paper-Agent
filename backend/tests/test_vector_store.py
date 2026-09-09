from pathlib import Path

from backend.app.rag import vector_store
from backend.app.rag.vector_store import get_or_create_collection


def test_get_or_create_collection_uses_persistent_path(tmp_path: Path) -> None:
    collection = get_or_create_collection(tmp_path, "paper_chunks")

    assert collection.name == "paper_chunks"
    assert tmp_path.exists()


def test_collections_are_created_with_cosine_space(tmp_path: Path) -> None:
    """Chroma defaults to squared-L2, which makes `1 - distance` go negative."""
    collection = get_or_create_collection(tmp_path, "paper_chunks")

    assert vector_store.collection_space(collection) == "cosine"


def test_distance_to_similarity_handles_each_distance_space() -> None:
    # Identical normalised vectors.
    assert vector_store.distance_to_similarity(0.0, "cosine") == 1.0
    assert vector_store.distance_to_similarity(0.0, "l2") == 1.0

    # Orthogonal normalised vectors: cosine distance 1.0, squared-L2 distance 2.0.
    assert vector_store.distance_to_similarity(1.0, "cosine") == 0.0
    assert vector_store.distance_to_similarity(2.0, "l2") == 0.0

    # A closely related pair must not be reported as irrelevant under l2.
    assert vector_store.distance_to_similarity(0.5, "l2") == 0.75

    # Opposed vectors clamp to 0 rather than going negative.
    assert vector_store.distance_to_similarity(4.0, "l2") == 0.0
    assert vector_store.distance_to_similarity(2.0, "cosine") == 0.0


def test_collection_space_defaults_to_cosine_when_unavailable() -> None:
    class CollectionWithoutConfiguration:
        pass

    assert vector_store.collection_space(CollectionWithoutConfiguration()) == "cosine"


def test_embed_texts_encodes_the_whole_batch_in_one_call(monkeypatch) -> None:
    model = FakeBgeM3Model()
    monkeypatch.setattr(vector_store, "_get_bge_m3_model", lambda: model)

    embeddings = vector_store.embed_texts(["first chunk", "second chunk", "third chunk"])

    assert len(embeddings) == 3
    assert all(len(embedding) == 1024 for embedding in embeddings)
    assert len(model.calls) == 1
    assert model.calls[0]["text"] == ["first chunk", "second chunk", "third chunk"]
    assert model.calls[0]["batch_size"] == vector_store.EMBEDDING_BATCH_SIZE
    assert "prompt_name" not in model.calls[0]


def test_embed_texts_returns_empty_for_no_input(monkeypatch) -> None:
    model = FakeBgeM3Model()
    monkeypatch.setattr(vector_store, "_get_bge_m3_model", lambda: model)

    assert vector_store.embed_texts([]) == []
    assert model.calls == []


def test_upsert_chunks_indexes_every_chunk_in_one_upsert(monkeypatch) -> None:
    upserts: list[dict] = []

    class FakeCollection:
        def upsert(self, **kwargs):
            upserts.append(kwargs)

    monkeypatch.setattr(vector_store, "get_paper_chunks_collection", lambda: FakeCollection())
    monkeypatch.setattr(
        vector_store,
        "embed_texts",
        lambda texts, input_type="document", batch_size=None: [[0.25] * 4 for _ in texts],
    )

    vector_store.upsert_chunks(
        [
            {"chunk_id": 1, "content": "first", "metadata": {"document_id": 1, "image_path": None}},
            {"chunk_id": 2, "content": "second", "metadata": {"document_id": 1}},
        ]
    )

    assert len(upserts) == 1
    assert upserts[0]["ids"] == ["1", "2"]
    assert upserts[0]["documents"] == ["first", "second"]
    # None-valued metadata is dropped because Chroma rejects null values.
    assert upserts[0]["metadatas"] == [{"document_id": 1}, {"document_id": 1}]


def test_upsert_chunks_removes_stale_document_vectors_before_reparse(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeCollection:
        def delete(self, **kwargs):
            calls.append({"method": "delete", **kwargs})

        def upsert(self, **kwargs):
            calls.append({"method": "upsert", **kwargs})

    monkeypatch.setattr(vector_store, "get_paper_chunks_collection", lambda: FakeCollection())
    monkeypatch.setattr(
        vector_store,
        "embed_texts",
        lambda texts, input_type="document", batch_size=None: [[0.25] * 4 for _ in texts],
    )

    vector_store.upsert_chunks(
        [
            {"chunk_id": 1, "content": "replacement", "metadata": {"document_id": 7}},
        ]
    )

    assert calls[0] == {"method": "delete", "where": {"document_id": 7}}
    assert calls[1]["method"] == "upsert"


class FakeBgeM3Model:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def encode(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        if isinstance(text, str):
            return [1.0] * vector_store.EMBEDDING_DIMENSIONS
        return [[1.0] * vector_store.EMBEDDING_DIMENSIONS for _ in text]


def test_embed_text_encodes_queries_without_prompt(monkeypatch) -> None:
    # BGE-M3 is symmetric and the snapshot ships no prompt templates, so the
    # query side must not request prompt_name (it would raise on every call).
    model = FakeBgeM3Model()
    monkeypatch.setattr(vector_store, "_get_bge_m3_model", lambda: model)

    embedding = vector_store.embed_text("谱聚类如何构图？", input_type="query")

    assert len(embedding) == 1024
    assert embedding == [1.0] * 1024
    assert model.calls == [
        {
            "text": "谱聚类如何构图？",
            "normalize_embeddings": True,
        }
    ]


def test_embed_text_uses_bge_m3_without_prompt_for_documents(monkeypatch) -> None:
    model = FakeBgeM3Model()
    monkeypatch.setattr(vector_store, "_get_bge_m3_model", lambda: model)

    vector_store.embed_text("The method builds a graph.", input_type="document")

    assert model.calls == [
        {
            "text": "The method builds a graph.",
            "normalize_embeddings": True,
        }
    ]


def test_upsert_and_query_use_bge_m3_embeddings(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeCollection:
        def upsert(self, **kwargs):
            calls.append({"method": "upsert", **kwargs})

        def query(self, **kwargs):
            calls.append({"method": "query", **kwargs})
            return {
                "documents": [["matched chunk"]],
                "metadatas": [[{"chunk_id": "c1", "document_id": 1}]],
                "distances": [[0.25]],
            }

    def fake_embed_text(text: str, input_type: str = "document") -> list[float]:
        calls.append({"method": "embed_text", "text": text, "input_type": input_type})
        return [0.5] * vector_store.EMBEDDING_DIMENSIONS

    monkeypatch.setattr(vector_store, "get_paper_chunks_collection", lambda: FakeCollection())
    monkeypatch.setattr(vector_store, "embed_text", fake_embed_text)

    vector_store.upsert_chunk(
        chunk_id="c1",
        content="The document chunk.",
        metadata={"chunk_id": "c1", "document_id": 1, "unused": None},
    )
    matches = vector_store.query_chunks("What does the paper say?", document_id=1, limit=3)

    assert calls[0] == {
        "method": "embed_text",
        "text": "The document chunk.",
        "input_type": "document",
    }
    assert calls[1]["method"] == "upsert"
    assert calls[1]["embeddings"] == [[0.5] * 1024]
    assert calls[1]["metadatas"] == [{"chunk_id": "c1", "document_id": 1}]
    assert calls[2] == {
        "method": "embed_text",
        "text": "What does the paper say?",
        "input_type": "query",
    }
    assert calls[3]["method"] == "query"
    assert calls[3]["where"] == {"document_id": 1}
    assert matches == [
        {
            "content": "matched chunk",
            "metadata": {"chunk_id": "c1", "document_id": 1},
            "score": 0.75,
        }
    ]
