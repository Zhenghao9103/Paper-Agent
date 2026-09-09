import logging
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import chromadb
from chromadb.api.models.Collection import Collection

from ..core.paths import chroma_dir, require_hf_model_snapshot

PAPER_CHUNKS_COLLECTION = "paper_chunks"
USER_MEMORIES_COLLECTION = "user_memories"
BGE_M3_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSIONS = 1024
EMBEDDING_BATCH_SIZE = 16
EmbeddingInputType = Literal["document", "query"]

# Chroma defaults to squared-L2 distance. We always create collections with cosine
# space so that `1 - distance` is a real cosine similarity in [0, 1]. Collections
# created before this change still report `l2`, so distances are converted using the
# space the collection actually reports rather than an assumed one.
COSINE_CONFIGURATION = {"hnsw": {"space": "cosine"}}
DEFAULT_SPACE = "cosine"
logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _persistent_client(persist_path: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=persist_path)


def get_chroma_client(path: Path | None = None) -> chromadb.PersistentClient:
    persist_path = path or chroma_dir()
    persist_path.mkdir(parents=True, exist_ok=True)
    return _persistent_client(str(persist_path))


def get_or_create_collection(path: Path | None, name: str) -> Collection:
    client = get_chroma_client(path)
    try:
        return client.get_collection(name=name)
    except Exception:
        pass

    try:
        return client.create_collection(name=name, configuration=COSINE_CONFIGURATION)
    except Exception:
        # Either another caller created it first, or this Chroma build rejects the
        # configuration argument. Fall back to whatever space the server defaults to;
        # `collection_space` keeps score conversion correct either way.
        return client.get_or_create_collection(name=name)


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
        get_or_create_collection(path, name)


def collection_space(collection: Collection) -> str:
    """Report the distance space a collection was created with."""
    try:
        space = collection.configuration["hnsw"]["space"]
    except Exception:
        return DEFAULT_SPACE
    return str(space) if space else DEFAULT_SPACE


def distance_to_similarity(distance: float, space: str = DEFAULT_SPACE) -> float:
    """Convert a Chroma distance into a cosine similarity in [0, 1].

    Embeddings are L2-normalised, so for a cosine of `c`:
      cosine / ip -> distance = 1 - c
      l2          -> distance = ||a - b||^2 = 2 - 2c
    """
    value = float(distance)
    similarity = 1.0 - value / 2.0 if space == "l2" else 1.0 - value
    return max(0.0, min(1.0, similarity))


@lru_cache(maxsize=1)
def _get_bge_m3_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "BGE-M3 embeddings require sentence-transformers. "
            "Install backend requirements before indexing documents."
        ) from exc
    model = SentenceTransformer(require_hf_model_snapshot(BGE_M3_MODEL_NAME))
    # median chunk is ~600 chars but a few formula/table chunks exceed 30k
    # chars; the 8192-token default pads whole batches to the longest member
    # and makes CPU encoding intractable. 512 tokens covers >99% of chunks.
    model.max_seq_length = 512
    return model


def embed_text(text: str, input_type: EmbeddingInputType = "document") -> list[float]:
    # BGE-M3 is a symmetric retriever and the downloaded snapshot defines no
    # prompt templates, so query and document sides must encode identically;
    # requesting prompt_name="query" would raise on every call.
    model = _get_bge_m3_model()
    embedding = model.encode(text, normalize_embeddings=True)
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    return [float(value) for value in embedding]


def embed_texts(
    texts: Sequence[str],
    input_type: EmbeddingInputType = "document",
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    """Encode many texts in one batched forward pass."""
    if not texts:
        return []

    model = _get_bge_m3_model()
    encode_kwargs: dict[str, Any] = {
        "normalize_embeddings": True,
        "batch_size": batch_size,
    }
    embeddings = model.encode(list(texts), **encode_kwargs)
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    return [[float(value) for value in embedding] for embedding in embeddings]


def _clean_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if value is not None}


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
        metadatas=[_clean_metadata(metadata)],
        embeddings=[embed_text(content, input_type="document")],
    )


def upsert_chunks(
    items: Sequence[Mapping[str, Any]],
    *,
    purge_existing: bool = True,
) -> None:
    """Embed and index many chunks in one batched call.

    Roughly an order of magnitude faster than calling `upsert_chunk` per chunk,
    because BGE-M3 runs one batched forward pass instead of N sequential ones.
    ``purge_existing=False`` skips the where-filtered delete pass and must only
    be used on a fresh collection: on this chromadb build the where-delete
    interleaved with inserts leaves an hnsw segment that fails to load in the
    next process (reproduced with a minimal fixture).
    """
    if not items:
        return

    contents = [str(item["content"]) for item in items]
    collection = get_paper_chunks_collection()
    if purge_existing:
        document_ids: set[int] = set()
        for item in items:
            document_id = item.get("metadata", {}).get("document_id")
            if document_id is None:
                continue
            try:
                document_ids.add(int(document_id))
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid vector document id %r", document_id)
        for document_id in document_ids:
            try:
                delete_document_chunks(document_id)
            except Exception:
                # Vector cleanup is best effort and must never affect SQLite/FTS state.
                logger.warning(
                    "Unable to remove stale vectors for document %s",
                    document_id,
                    exc_info=True,
                )
    collection.upsert(
        ids=[str(item["chunk_id"]) for item in items],
        documents=contents,
        metadatas=[_clean_metadata(item.get("metadata", {})) for item in items],
        embeddings=embed_texts(
            contents,
            input_type="document",
            batch_size=EMBEDDING_BATCH_SIZE,
        ),
    )


def delete_document_chunks(document_id: int) -> None:
    get_paper_chunks_collection().delete(where={"document_id": document_id})


def query_chunks(query: str, document_id: int | None = None, limit: int = 5) -> list[dict]:
    collection = get_paper_chunks_collection()
    where = {"document_id": document_id} if document_id is not None else None
    result = collection.query(
        query_embeddings=[embed_text(query, input_type="query")],
        n_results=limit,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    space = collection_space(collection)
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    matches: list[dict] = []
    for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
        matches.append(
            {
                "content": document,
                "metadata": metadata,
                "score": distance_to_similarity(distance, space),
            }
        )
    return matches
