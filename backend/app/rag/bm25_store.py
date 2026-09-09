from dataclasses import dataclass

from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session

from ..models.chunk import DocumentChunk
from ..models.document import Document
from .tokenization import tokenize_mixed, tokens_to_fts_text

CREATE_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    title_tokens,
    body_tokens,
    tokenize='unicode61 remove_diacritics 2'
)
"""


@dataclass(frozen=True)
class BM25Hit:
    chunk_id: int
    rank: int
    raw_score: float


def ensure_bm25_schema(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(text(CREATE_FTS_SQL))


def _match_expression(terms: list[str]) -> str:
    safe = [token for token in tokenize_mixed(" ".join(terms)) if token]
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in safe)


def index_chunk(
    db: Session,
    *,
    chunk_id: int,
    document_id: int,
    title: str,
    content: str,
) -> None:
    db.execute(
        text("DELETE FROM document_chunks_fts WHERE chunk_id = :chunk_id"),
        {"chunk_id": chunk_id},
    )
    db.execute(
        text(
            """INSERT INTO document_chunks_fts
               (chunk_id, document_id, title_tokens, body_tokens)
               VALUES (:chunk_id, :document_id, :title_tokens, :body_tokens)"""
        ),
        {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "title_tokens": tokens_to_fts_text(
                tokenize_mixed(title, deduplicate=False)
            ),
            "body_tokens": tokens_to_fts_text(
                tokenize_mixed(content, deduplicate=False)
            ),
        },
    )


def search_bm25(
    db: Session,
    terms: list[str],
    *,
    document_id: int | None,
    limit: int,
) -> list[BM25Hit]:
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if document_id is not None and document_id <= 0:
        raise ValueError("document_id must be greater than zero")

    expression = _match_expression(terms)
    if not expression:
        return []

    rows = db.execute(
        text(
            """SELECT chunk_id,
                      bm25(document_chunks_fts, 0.0, 0.0, 5.0, 1.0) AS score
               FROM document_chunks_fts
               WHERE document_chunks_fts MATCH :query
                 AND (:document_id IS NULL OR document_id = :document_id)
               ORDER BY score ASC, CAST(chunk_id AS INTEGER) ASC
               LIMIT :limit"""
        ),
        {"query": expression, "document_id": document_id, "limit": limit},
    ).all()
    return [
        BM25Hit(chunk_id=int(row.chunk_id), rank=rank, raw_score=float(row.score))
        for rank, row in enumerate(rows, start=1)
    ]


def delete_document_index(db: Session, document_id: int) -> None:
    db.execute(
        text("DELETE FROM document_chunks_fts WHERE document_id = :document_id"),
        {"document_id": document_id},
    )


def clear_bm25_index(db: Session) -> int:
    count = int(db.execute(text("SELECT count(*) FROM document_chunks_fts")).scalar_one())
    db.execute(text("DELETE FROM document_chunks_fts"))
    return count


def rebuild_bm25_index(db: Session) -> int:
    clear_bm25_index(db)
    rows = db.execute(
        select(DocumentChunk, Document.title).join(
            Document,
            Document.id == DocumentChunk.document_id,
        )
    ).all()
    for chunk, title in rows:
        index_chunk(
            db,
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            title=title,
            content=chunk.content,
        )
    return len(rows)
