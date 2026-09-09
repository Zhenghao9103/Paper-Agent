import itertools
from concurrent.futures import Future
from queue import PriorityQueue
from threading import Lock

from backend.app.models.document import Document
from backend.app.models.parsing import DocumentParseRun
from backend.app.services.document_jobs import (
    LEGACY_REPARSE_PRIORITY,
    NEW_UPLOAD_PRIORITY,
    DocumentJobQueue,
    enqueue_pending_and_legacy,
)
from sqlalchemy.orm import sessionmaker


def test_document_queue_uses_one_worker_and_new_upload_priority() -> None:
    queue = DocumentJobQueue.__new__(DocumentJobQueue)
    queue._items = PriorityQueue()
    queue._sequence = itertools.count()
    queue._queued = set()
    queue._lock = Lock()
    fake_factory = object()

    legacy = Future()
    fresh = Future()
    queue._items.put((LEGACY_REPARSE_PRIORITY, 0, 1, fake_factory, legacy))
    queue._items.put((NEW_UPLOAD_PRIORITY, 1, 2, fake_factory, fresh))

    assert queue._items.get()[2] == 2
    assert queue._items.get()[2] == 1


def test_legacy_scan_enqueues_without_changing_visible_status(
    db_session,
    monkeypatch,
) -> None:
    document = Document(
        title="legacy",
        file_type="pdf",
        file_path="legacy.pdf",
        status="indexed",
    )
    db_session.add(document)
    db_session.commit()
    document_id = document.id
    queued: list[int] = []
    factory = sessionmaker(autoflush=False, bind=db_session.get_bind())
    monkeypatch.setattr(
        "backend.app.services.document_jobs.enqueue_document",
        lambda queued_id, **_kwargs: queued.append(queued_id),
    )

    assert enqueue_pending_and_legacy(session_factory=factory) == [document_id]
    db_session.refresh(document)
    assert document.status == "indexed"
    assert queued == [document_id]


def test_current_parser_version_is_not_requeued(db_session, monkeypatch) -> None:
    document = Document(
        title="current",
        file_type="pdf",
        file_path="current.pdf",
        status="indexed",
    )
    db_session.add(document)
    db_session.flush()
    db_session.add(
        DocumentParseRun(
            document_id=document.id,
            parser_version="mineru-primary-v1",
            status="success",
        )
    )
    db_session.commit()
    queued: list[int] = []
    factory = sessionmaker(autoflush=False, bind=db_session.get_bind())
    monkeypatch.setattr(
        "backend.app.services.document_jobs.enqueue_document",
        lambda queued_id, **_kwargs: queued.append(queued_id),
    )

    assert enqueue_pending_and_legacy(session_factory=factory) == []
    assert queued == []
