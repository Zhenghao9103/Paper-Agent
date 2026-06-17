from pathlib import Path
from functools import lru_cache
from typing import Literal

import chromadb
from chromadb.api.models.Collection import Collection

from ..core.paths import chroma_dir

PAPER_CHUNKS_COLLECTION = "paper_chunks"
USER_MEMORIES_COLLECTION = "user_memories"
BGE_M3_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSIONS = 1024
EmbeddingInputType = Literal["document", "query"]


def get_chroma_client(path: Path | None = None) -> chromadb.PersistentClient:
    persist_path = path or chroma_dir()
    persist_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_path))


def get_or_create_collection(path: Path | None, name: str) -> Collection:
    return get_chroma_client(path).get_or_create_collection(name=name)


def get_paper_chunks_collection() -> Collection:
    return get_or_create_collection(None, PAPER_CHUNKS_COLLECTION)


def get_user_memories_collection() -> Collection:
    return get_or_create_collection(None, USER_MEMORIES_COLLECTION)


def reset_vector_collections(path: Path | None = None) -> None:
    client = get_chroma_client(path)
    for name in (PAPER_CHUNKS_COLLECTION, USER_MEMORIES_COLLECTION):
        try:
            client.delete_collection(name)
        except Exception:
            pass
        client.get_or_create_collection(name=name)


@lru_cache(maxsize=1)
def _get_bge_m3_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "BGE-M3 embeddings require sentence-transformers. "
            "Install backend requirements before indexing documents."
        ) from exc
    return SentenceTransformer(BGE_M3_MODEL_NAME)


def embed_text(text: str, input_type: EmbeddingInputType = "document") -> list[float]:
    model = _get_bge_m3_model()
    encode_kwargs = {"normalize_embeddings": True}
    if input_type == "query":
        encode_kwargs["prompt_name"] = "query"

    embedding = model.encode(text, **encode_kwargs)
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    return [float(value) for value in embedding]


def upsert_chunk(
    *,
    chunk_id: str,
    content: str,
    metadata: dict[str, str | int | float | None],
) -> None:
    collection = get_paper_chunks_collection()
    collection.upsert(
        ids=[chunk_id],
        documents=[content],
        metadatas=[{key: value for key, value in metadata.items() if value is not None}],
        embeddings=[embed_text(content, input_type="document")],
    )


def query_chunks(query: str, document_id: int | None = None, limit: int = 5) -> list[dict]:
    collection = get_paper_chunks_collection()
    where = {"document_id": document_id} if document_id is not None else None
    result = collection.query(
        query_embeddings=[embed_text(query, input_type="query")],
        n_results=limit,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    matches: list[dict] = []
    for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
        matches.append(
            {
                "content": document,
                "metadata": metadata,
                "score": max(0.0, 1.0 - float(distance)),
            }
        )
    return matches
