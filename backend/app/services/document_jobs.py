"""Single-worker priority queue for document parsing and indexing."""

from __future__ import annotations

import itertools
import logging
from concurrent.futures import Future
from queue import PriorityQueue
from threading import Lock, Thread

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from ..db.session import SessionLocal
from ..models.document import Document
from ..models.parsing import DocumentParseRun
from .document_store import DocumentStore
from .ingestion import parse_and_store_document

logger = logging.getLogger(__name__)
NEW_UPLOAD_PRIORITY = 0
LEGACY_REPARSE_PRIORITY = 10


class DocumentJobQueue:
    def __init__(self) -> None:
        self._items: PriorityQueue = PriorityQueue()
        self._sequence = itertools.count()
        self._queued: set[int] = set()
        self._lock = Lock()
        Thread(target=self._run, daemon=True, name="paper-ingestion").start()

    def enqueue(
        self,
        document_id: int,
        *,
        session_factory: sessionmaker,
        priority: int = NEW_UPLOAD_PRIORITY,
    ) -> Future[bool] | None:
        with self._lock:
            if document_id in self._queued:
                return None
            self._queued.add(document_id)
        future: Future[bool] = Future()
        self._items.put((priority, next(self._sequence), document_id, session_factory, future))
        return future

    def _run(self) -> None:
        while True:
            _priority, _sequence, document_id, session_factory, future = self._items.get()
            try:
                future.set_result(self._process(document_id, session_factory))
            except Exception as exc:  # pragma: no cover
                logger.exception("Background ingestion failed for document %s", document_id)
                future.set_exception(exc)
            finally:
                with self._lock:
                    self._queued.discard(document_id)
                self._items.task_done()

    @staticmethod
    def _process(document_id: int, session_factory: sessionmaker) -> bool:
        try:
            with session_factory() as db:
                document = db.get(Document, document_id)
                if document is None:
                    return False
                document.status = "parsing"
                db.add(document)
                db.commit()
                return parse_and_store_document(db, document)
        except Exception:
            logger.exception("Background ingestion failed for document %s", document_id)
            with session_factory() as db:
                document = db.get(Document, document_id)
                if document is not None:
                    document.status = "failed"
                    db.add(document)
                    db.commit()
            return False


document_jobs = DocumentJobQueue()


def enqueue_document(
    document_id: int,
    *,
    session_factory: sessionmaker = SessionLocal,
    priority: int = NEW_UPLOAD_PRIORITY,
) -> Future[bool] | None:
    return document_jobs.enqueue(document_id, session_factory=session_factory, priority=priority)


def enqueue_pending_and_legacy(
    *,
    session_factory: sessionmaker = SessionLocal,
) -> list[int]:
    with session_factory() as db:
        documents = list(db.scalars(select(Document).order_by(Document.created_at.desc())))
        pending: list[int] = []
        for document in documents:
            version = db.scalar(
                select(DocumentParseRun.parser_version)
                .where(DocumentParseRun.document_id == document.id)
                .order_by(DocumentParseRun.id.desc())
            )
            if document.file_type == "pdf" and (
                document.status in {"uploaded", "parsing", "parsed"}
                or version != DocumentStore.parser_version
            ):
                pending.append(document.id)
    for document_id in pending:
        enqueue_document(
            document_id,
            session_factory=session_factory,
            priority=LEGACY_REPARSE_PRIORITY,
        )
    return pending
