"""Isolated PDF corpus setup for RAG retrieval evaluation."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

IngestionFunction = Callable[[Session, Any], bool]


def resolve_corpus(dataset: Any, uploads: Path) -> dict[str, Path]:
    """Resolve every corpus prefix to exactly one PDF in ``uploads``.

    Longest prefixes claim their files first, so deliberately overlapping
    prefixes (for example ``DSC`` vs ``DSC-LSP``) resolve deterministically
    instead of failing with an ambiguous match.
    """

    uploads = Path(uploads).resolve()
    if not uploads.is_dir():
        raise FileNotFoundError(f"Uploads directory does not exist: {uploads}")

    pdfs = [
        path.resolve()
        for path in uploads.iterdir()
        if path.is_file() and path.suffix.casefold() == ".pdf"
    ]
    resolved: dict[str, Path] = {}
    claimed: set[Path] = set()
    ordered = sorted(
        dataset.corpus, key=lambda document: (-len(document.file_prefix), document.key)
    )
    for document in ordered:
        prefix = document.file_prefix.casefold()
        matches = sorted(
            (
                path
                for path in pdfs
                if path not in claimed and path.name.casefold().startswith(prefix)
            ),
            key=lambda path: path.name.casefold(),
        )
        if not matches:
            raise FileNotFoundError(
                f"Corpus key {document.key!r} has no PDF with prefix "
                f"{document.file_prefix!r} in {uploads}"
            )
        if len(matches) > 1:
            names = ", ".join(path.name for path in matches)
            raise ValueError(
                f"Corpus key {document.key!r} with prefix {document.file_prefix!r} "
                f"matches multiple PDFs in {uploads}: {names}"
            )
        resolved[document.key] = matches[0]
        claimed.add(matches[0])
    return resolved


def create_eval_database(workdir: Path) -> tuple[Engine, Session]:
    """Create the isolated SQLite database, ORM tables, and BM25 FTS table."""

    from ..db.base import Base
    from ..models import Document  # noqa: F401 - register all ORM tables
    from ..rag.bm25_store import ensure_bm25_schema

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{(workdir / 'agent-eval.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    ensure_bm25_schema(engine)
    db = sessionmaker(bind=engine, autocommit=False, autoflush=False)()
    return engine, db


def prepare_corpus(
    db: Session,
    corpus: Mapping[str, Path],
    *,
    reuse_index: bool,
    ingestion_fn: IngestionFunction | None = None,
) -> dict[str, int]:
    """Ingest the corpus or map it onto documents already present in the work DB."""

    from ..models.document import Document

    if reuse_index:
        rows = list(db.scalars(select(Document).order_by(Document.id)))
        by_path: dict[str, list[Document]] = {}
        for row in rows:
            normalized = _normalized_path(Path(row.file_path))
            by_path.setdefault(normalized, []).append(row)

        mapping: dict[str, int] = {}
        for key, path in corpus.items():
            matches = by_path.get(_normalized_path(path), [])
            if not matches:
                raise RuntimeError(
                    f"--reuse-index was passed but corpus key {key!r} is missing "
                    "from the isolated working database"
                )
            if len(matches) > 1:
                raise RuntimeError(
                    f"--reuse-index found multiple Documents for corpus key {key!r}"
                )
            mapping[key] = matches[0].id
        return mapping

    if db.scalar(select(Document.id).limit(1)) is not None:
        raise RuntimeError(
            "Fresh ingestion cannot use a non-empty evaluation database; use a new "
            "--workdir or pass --reuse-index."
        )

    if ingestion_fn is None:
        from ..services.ingestion import parse_and_store_document

        ingestion_fn = parse_and_store_document

    mapping = {}
    for key, path in corpus.items():
        document = Document(
            title=_title_from_path(path),
            file_type="pdf",
            file_path=str(path.resolve()),
            status="uploaded",
            source_type="uploaded",
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        if not ingestion_fn(db, document):
            raise RuntimeError(
                f"Failed to ingest corpus key {key!r} from {path.name!r}"
            )
        mapping[key] = document.id
    return mapping


@contextmanager
def _isolated_runtime(workdir: Path, chroma_dir: Path | None = None):
    """Temporarily redirect every import-bound runtime path used by ingestion.

    ``chroma_dir`` decouples the vector store from the SQLite workdir: SQLite,
    structured storage, and staging stay inside ``workdir`` while HNSW
    persists wherever the caller points (e.g. a pure-ASCII directory).
    """

    from ..core import paths
    from ..rag import vector_store
    from ..services import document_store, ingestion

    storage = workdir / "storage"
    uploads = storage / "uploads"
    pages = storage / "pages"
    documents = storage / "documents"
    staging = storage / ".staging"
    exports = storage / "exports"
    chroma = Path(chroma_dir).resolve() if chroma_dir else workdir / "chroma"
    for directory in (uploads, pages, documents, staging, exports, chroma):
        directory.mkdir(parents=True, exist_ok=True)

    def isolated_document_dir(document_id: int) -> Path:
        if document_id <= 0:
            raise ValueError("document_id must be positive")
        return documents / str(document_id)

    patches = [
        (paths, "storage_root", lambda: storage),
        (paths, "uploads_dir", lambda: uploads),
        (paths, "pages_dir", lambda: pages),
        (paths, "documents_dir", lambda: documents),
        (paths, "document_dir", isolated_document_dir),
        (paths, "staging_dir", lambda: staging),
        (paths, "exports_dir", lambda: exports),
        (paths, "chroma_dir", lambda: chroma),
        (ingestion, "staging_dir", lambda: staging),
        (document_store, "documents_dir", lambda: documents),
        (document_store, "document_dir", isolated_document_dir),
        (vector_store, "chroma_dir", lambda: chroma),
    ]
    originals = [(module, name, getattr(module, name)) for module, name, _ in patches]
    vector_store._persistent_client.cache_clear()
    try:
        for module, name, replacement in patches:
            setattr(module, name, replacement)
        yield
    finally:
        vector_store._persistent_client.cache_clear()
        for module, name, original in originals:
            setattr(module, name, original)


def _normalized_path(path: Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _title_from_path(path: Path) -> str:
    return path.stem.rsplit("-", 1)[0].replace("-", " ").replace("_", " ")
