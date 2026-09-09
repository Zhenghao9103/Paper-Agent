import time
from pathlib import Path

import fitz
from backend.app.db.session import get_db
from backend.app.ingestion.figure_descriptions import (
    FigureDescription,
    FigureDescriptionResult,
)
from backend.app.models.chat import ChatMessage, ChatSession, TranscriptEvent
from backend.app.models.chunk import DocumentChunk
from backend.app.models.context_checkpoint import ContextCheckpoint
from backend.app.models.document import Document
from backend.app.models.memory import Memory
from backend.app.models.page import DocumentPage
from backend.app.models.parsing import DocumentElement
from backend.app.models.session_memory import SessionMemory
from backend.app.rag.bm25_store import search_bm25
from backend.app.rag.vector_store import get_paper_chunks_collection
from backend.app.services.analysis import fallback_analysis
from backend.app.services.ingestion import parse_and_store_document
from fastapi.testclient import TestClient
from sqlalchemy import select, text


def make_pdf_bytes(text: str = "Graph neural network paper") -> bytes:
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    data = pdf.tobytes()
    pdf.close()
    return data


def set_document_status(db, document: Document, status: str) -> bool:
    document.status = status
    db.add(document)
    db.commit()
    return status == "indexed"


def test_list_documents_starts_empty(client: TestClient) -> None:
    response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json() == []


def test_upload_document_creates_record(client: TestClient) -> None:
    files = {"file": ("sample.pdf", make_pdf_bytes(), "application/pdf")}

    response = client.post("/api/documents/upload", files=files)

    assert response.status_code == 201
    payload = response.json()
    assert payload["title"] == "sample"
    assert payload["file_type"] == "pdf"
    assert payload["status"] == "indexed"
    assert payload["source_type"] == "uploaded"


def test_async_upload_returns_parsing_before_background_result(
    client: TestClient,
    monkeypatch,
) -> None:
    observed_statuses: list[str] = []

    def fake_parse(db, document) -> bool:
        observed_statuses.append(document.status)
        return set_document_status(db, document, "indexed")

    monkeypatch.setattr(
        "backend.app.services.document_jobs.parse_and_store_document",
        fake_parse,
    )
    response = client.post(
        "/api/documents/upload-async",
        files={"file": ("async.pdf", make_pdf_bytes(), "application/pdf")},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "parsing"
    document_id = response.json()["id"]
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if client.get(f"/api/documents/{document_id}").json()["status"] == "indexed":
            break
        time.sleep(0.01)
    assert client.get(f"/api/documents/{document_id}").json()["status"] == "indexed"
    assert observed_statuses == ["parsing"]


def test_upload_document_indexes_bm25_chunks(client: TestClient) -> None:
    files = {"file": ("bm25.pdf", make_pdf_bytes("BM25 integration evidence"), "application/pdf")}

    response = client.post("/api/documents/upload", files=files)

    assert response.status_code == 201
    document_id = response.json()["id"]
    db = client.app.state.test_session_factory()
    try:
        count = db.scalar(
            text("SELECT count(*) FROM document_chunks_fts WHERE document_id = :document_id"),
            {"document_id": document_id},
        )
    finally:
        db.close()
    assert count == 1


def test_upload_document_rejects_txt(client: TestClient) -> None:
    files = {"file": ("notes.txt", b"plain text", "text/plain")}

    response = client.post("/api/documents/upload", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported"


def test_upload_document_rejects_pptx(client: TestClient) -> None:
    files = {
        "file": (
            "slides.pptx",
            b"not accepted even when uploaded as a presentation",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
    }

    response = client.post("/api/documents/upload", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported"


def test_get_document_by_id(client: TestClient) -> None:
    files = {"file": ("detail.pdf", make_pdf_bytes("Detail page text"), "application/pdf")}
    created = client.post("/api/documents/upload", files=files).json()

    response = client.get(f"/api/documents/{created['id']}")

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]
    assert response.json()["title"] == "detail"


def test_list_document_pages_after_upload(client: TestClient) -> None:
    files = {"file": ("pages.pdf", make_pdf_bytes("Important method section"), "application/pdf")}
    created = client.post("/api/documents/upload", files=files).json()

    response = client.get(f"/api/documents/{created['id']}/pages")

    assert response.status_code == 200
    pages = response.json()
    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "Important method section" in pages[0]["text"]
    assert pages[0]["image_path"].endswith("page-1.png")


def test_list_document_chunks_after_upload(client: TestClient) -> None:
    files = {
        "file": (
            "chunks.pdf",
            make_pdf_bytes("Chunking makes retrieval possible."),
            "application/pdf",
        )
    }
    created = client.post("/api/documents/upload", files=files).json()

    response = client.get(f"/api/documents/{created['id']}/chunks")

    assert response.status_code == 200
    chunks = response.json()
    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 1
    assert "Chunking makes retrieval possible" in chunks[0]["content"]
    assert chunks[0]["detail"]["chunk_type"] in {"text", "paragraph", "unknown"}
    assert isinstance(chunks[0]["detail"]["metadata"], dict)


def test_element_image_endpoint_serves_only_document_owned_asset(
    client: TestClient, test_storage_root: Path
) -> None:
    created = client.post(
        "/api/documents/upload",
        files={"file": ("asset.pdf", make_pdf_bytes("Asset text"), "application/pdf")},
    ).json()
    asset = test_storage_root / "documents" / str(created["id"]) / "figures" / "f.png"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(b"not-a-real-png")
    db = client.app.state.test_session_factory()
    try:
        row = DocumentElement(
            document_id=created["id"],
            element_uid="figure-api",
            element_type="figure",
            image_path=str(asset),
            raw_text="figure",
            structured_data_json='{"summary":"diagram"}',
            parse_status="success",
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        element_id = row.id
    finally:
        db.close()

    response = client.get(
        f"/api/documents/{created['id']}/elements/{element_id}/image"
    )
    assert response.status_code == 200
    assert response.content == b"not-a-real-png"

    db = client.app.state.test_session_factory()
    try:
        row = db.get(DocumentElement, element_id)
        row.image_path = str(test_storage_root.parent / "secret.png")
        db.commit()
    finally:
        db.close()
    assert client.get(
        f"/api/documents/{created['id']}/elements/{element_id}/image"
    ).status_code == 404


def test_figure_enrichment_is_lazy_idempotent_and_creates_one_chunk(
    client: TestClient,
    test_storage_root: Path,
    monkeypatch,
) -> None:
    created = client.post(
        "/api/documents/upload",
        files={"file": ("figure.pdf", make_pdf_bytes("Figure page"), "application/pdf")},
    ).json()
    asset = test_storage_root / "documents" / str(created["id"]) / "figures" / "f.png"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(b"image")
    db = client.app.state.test_session_factory()
    try:
        element = DocumentElement(
            document_id=created["id"],
            element_uid="figure-lazy",
            element_type="figure",
            page_number=1,
            image_path=str(asset),
            structured_data_json="{}",
            parse_status="pending",
        )
        db.add(element)
        db.commit()
        db.refresh(element)
        element_id = element.id
    finally:
        db.close()

    calls = {"start": 0, "describe": 0, "idle": 0}

    class FakeManager:
        def ensure_started(self) -> None:
            calls["start"] += 1

        def schedule_idle_close(self, _seconds: float) -> None:
            calls["idle"] += 1

    class FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def describe(self, *_args, **_kwargs) -> FigureDescriptionResult:
            calls["describe"] += 1
            return FigureDescriptionResult(
                description=FigureDescription(
                    figure_type="diagram",
                    components=["encoder", "decoder"],
                    summary="An encoder-decoder architecture.",
                ),
                model="mineru-test",
                latency_ms=17,
            )

    monkeypatch.setattr("backend.app.services.figure_enrichment._manager", FakeManager())
    monkeypatch.setattr("backend.app.services.figure_enrichment.MinerUFigureClient", FakeClient)
    monkeypatch.setattr(
        "backend.app.services.figure_enrichment.upsert_chunk",
        lambda **_kwargs: None,
    )

    first = client.post(f"/api/documents/{created['id']}/elements/{element_id}/enrich")
    second = client.post(f"/api/documents/{created['id']}/elements/{element_id}/enrich")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["structured_data"]["description_model"] == "mineru-test"
    assert calls == {"start": 1, "describe": 1, "idle": 1}
    db = client.app.state.test_session_factory()
    try:
        chunks = db.scalars(
            select(DocumentChunk).where(DocumentChunk.document_id == created["id"])
        ).all()
        assert sum("encoder-decoder" in chunk.content for chunk in chunks) == 1
    finally:
        db.close()


def test_structured_parse_routes_expose_blocks_elements_and_quality(
    client: TestClient,
) -> None:
    files = {"file": ("structured.pdf", make_pdf_bytes("Structured body text."), "application/pdf")}
    created = client.post("/api/documents/upload", files=files).json()
    document_id = created["id"]

    blocks = client.get(f"/api/documents/{document_id}/blocks")
    elements = client.get(f"/api/documents/{document_id}/elements")
    quality = client.get(f"/api/documents/{document_id}/quality")

    assert blocks.status_code == 200
    assert blocks.json()[0]["block_type"] in {"paragraph", "unknown"}
    assert elements.status_code == 200
    assert elements.json() == []
    assert quality.status_code == 200
    assert quality.json()["page_count"] == 1
    assert quality.json()["pipeline_stage"] == "indexed"
    assert "chunking" in quality.json()["timings_ms"]
    assert "vector_embedding" in quality.json()["timings_ms"]


def test_upload_keeps_parsed_text_when_vector_index_fails(
    client: TestClient,
    monkeypatch,
) -> None:
    def fail_vector_write(*args, **kwargs) -> None:
        raise RuntimeError("broken vector index")

    monkeypatch.setattr("backend.app.services.ingestion.upsert_chunks", fail_vector_write)
    files = {
        "file": (
            "vector-fail.pdf",
            make_pdf_bytes("Vector failure should not lose text."),
            "application/pdf",
        )
    }

    response = client.post("/api/documents/upload", files=files)

    assert response.status_code == 201
    created = response.json()
    assert created["status"] == "indexed"
    pages = client.get(f"/api/documents/{created['id']}/pages").json()
    chunks = client.get(f"/api/documents/{created['id']}/chunks").json()
    assert "Vector failure should not lose text" in pages[0]["text"]
    assert "Vector failure should not lose text" in chunks[0]["content"]


def test_upload_keeps_bm25_index_when_vector_index_fails(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "backend.app.services.ingestion.upsert_chunks",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    files = {
        "file": (
            "bm25-vector-fail.pdf",
            make_pdf_bytes("UniqueKeyword survives vector outage"),
            "application/pdf",
        )
    }

    response = client.post("/api/documents/upload", files=files)

    assert response.status_code == 201
    document_id = response.json()["id"]
    db = client.app.state.test_session_factory()
    try:
        assert search_bm25(db, ["UniqueKeyword"], document_id=document_id, limit=5)
    finally:
        db.close()


def test_refresh_failure_after_commit_does_not_mark_document_failed(
    db_session,
    monkeypatch,
    tmp_path,
    test_storage_root,
) -> None:
    pdf_path = tmp_path / "refresh-failure.pdf"
    pdf_path.write_bytes(make_pdf_bytes("Committed content"))
    document = Document(
        title="refresh failure",
        file_type="pdf",
        file_path=str(pdf_path),
        status="uploaded",
    )
    db_session.add(document)
    db_session.commit()

    def fail_refresh(*args, **kwargs):
        raise RuntimeError("refresh failure")

    monkeypatch.setattr(db_session, "refresh", fail_refresh)

    assert parse_and_store_document(db_session, document) is True
    assert db_session.scalar(select(Document.status).where(Document.id == document.id)) == "indexed"


def test_analyze_document_returns_chinese_sections(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr("backend.app.services.analysis.complete", lambda prompt: None)

    paper_text = (
        "Abstract. We propose a spectral clustering method with bootstrap ensembles "
        "to improve nonlinear manifold clustering. Method. The approach builds "
        "bounded plane segments, estimates local affinity, and aggregates multiple "
        "bootstrap partitions with optimal transport. Experiments. On synthetic "
        "manifolds and benchmark datasets, the method improves NMI and ARI against "
        "k-means, spectral clustering, and deep clustering baselines. Limitations. "
        "The method depends on bootstrap count and can be slower on large datasets."
    )
    files = {"file": ("analysis.pdf", make_pdf_bytes(paper_text), "application/pdf")}
    created = client.post("/api/documents/upload", files=files).json()

    response = client.post(f"/api/documents/{created['id']}/analyze")

    assert response.status_code == 200
    payload = response.json()
    assert payload["document_id"] == created["id"]
    assert payload["summary_zh"]
    assert payload["innovations"]
    assert "本文" in payload["summary_zh"]
    assert "方法" in payload["methodology"]
    assert "实验" in payload["experiments"]
    assert paper_text not in payload["summary_zh"]

    saved = client.get(f"/api/documents/{created['id']}/analysis")
    assert saved.status_code == 200
    assert saved.json()["id"] == payload["id"]


def test_analyze_reparses_document_when_pages_are_missing(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr("backend.app.services.analysis.complete", lambda prompt: None)
    monkeypatch.setattr(
        "backend.app.api.routes.documents.parse_and_store_document",
        lambda db, document: False,
    )
    files = {
        "file": (
            "missing-pages.pdf",
            make_pdf_bytes("Ignored during upload"),
            "application/pdf",
        )
    }
    created = client.post("/api/documents/upload", files=files).json()

    def repair_parse(db, document) -> bool:
        db.add(
            DocumentPage(
                document_id=document.id,
                page_number=1,
                text="Abstract. We propose a spectral clustering method.",
            )
        )
        document.status = "indexed"
        db.add(document)
        db.commit()
        return True

    monkeypatch.setattr("backend.app.api.routes.documents.parse_and_store_document", repair_parse)
    response = client.post(f"/api/documents/{created['id']}/analyze")

    assert response.status_code == 200
    payload = response.json()
    assert "谱聚类" in payload["summary_zh"]


def test_analyze_returns_400_when_no_text_can_be_extracted(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "backend.app.api.routes.documents.parse_and_store_document",
        lambda db, document: False,
    )
    files = {
        "file": (
            "empty.pdf",
            make_pdf_bytes("Upload text ignored by fake parser"),
            "application/pdf",
        )
    }
    created = client.post("/api/documents/upload", files=files).json()

    response = client.post(f"/api/documents/{created['id']}/analyze")

    assert response.status_code == 400
    assert "No parsed text" in response.json()["detail"]


def test_fallback_analysis_summarizes_instead_of_echoing_source() -> None:
    context = (
        "Abstract. We propose a spectral clustering method with bootstrap ensembles "
        "to improve nonlinear manifold clustering. Method. The approach builds "
        "bounded plane segments, estimates local affinity, and aggregates multiple "
        "bootstrap partitions with optimal transport. Experiments. On synthetic "
        "manifolds and benchmark datasets, the method improves NMI and ARI against "
        "k-means, spectral clustering, and deep clustering baselines. Limitations. "
        "The method depends on bootstrap count and can be slower on large datasets."
    )

    result = fallback_analysis("Spectral Bootstrap Clustering", context)

    assert "本文" in result["summary_zh"]
    assert "谱聚类" in result["summary_zh"]
    assert "bounded plane segments" not in result["summary_zh"]
    assert "bootstrap" in result["innovations"].lower()
    assert "NMI" in result["experiments"]
    assert "计算" in result["limitations"] or "速度" in result["limitations"]


def test_get_document_by_id_returns_404(client: TestClient) -> None:
    response = client.get("/api/documents/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_reparse_failed_document_returns_updated_document(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "backend.app.api.routes.documents.parse_and_store_document",
        lambda db, document: set_document_status(db, document, "failed"),
    )
    created = client.post(
        "/api/documents/upload",
        files={"file": ("retry.pdf", make_pdf_bytes("Retry text"), "application/pdf")},
    ).json()
    monkeypatch.setattr(
        "backend.app.api.routes.documents.parse_and_store_document",
        lambda db, document: set_document_status(db, document, "indexed"),
    )

    response = client.post(f"/api/documents/{created['id']}/reparse")

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]
    assert response.json()["status"] == "indexed"


def test_reparse_returns_failed_document_when_parsing_fails(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "backend.app.api.routes.documents.parse_and_store_document",
        lambda db, document: set_document_status(db, document, "failed"),
    )
    created = client.post(
        "/api/documents/upload",
        files={"file": ("still-broken.pdf", make_pdf_bytes(), "application/pdf")},
    ).json()

    response = client.post(f"/api/documents/{created['id']}/reparse")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"


def test_reparse_returns_404_for_unknown_document(client: TestClient) -> None:
    response = client.post("/api/documents/999/reparse")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_reparse_returns_409_when_source_pdf_is_missing(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "backend.app.api.routes.documents.parse_and_store_document",
        lambda db, document: set_document_status(db, document, "failed"),
    )
    created = client.post(
        "/api/documents/upload",
        files={"file": ("missing-source.pdf", make_pdf_bytes(), "application/pdf")},
    ).json()
    Path(created["file_path"]).unlink()

    response = client.post(f"/api/documents/{created['id']}/reparse")

    assert response.status_code == 409
    assert response.json()["detail"] == "Source PDF not found"


def test_reparse_returns_409_when_document_is_already_parsing(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "backend.app.api.routes.documents.parse_and_store_document",
        lambda db, document: set_document_status(db, document, "parsing"),
    )
    created = client.post(
        "/api/documents/upload",
        files={"file": ("parsing.pdf", make_pdf_bytes(), "application/pdf")},
    ).json()

    response = client.post(f"/api/documents/{created['id']}/reparse")

    assert response.status_code == 409
    assert response.json()["detail"] == "Document is already being parsed"


def test_clear_all_data_removes_database_rows_and_vector_collections(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr("backend.app.agents.tools.search_arxiv", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.count_text_tokens",
        lambda *args, **kwargs: 0,
    )
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.count_message_tokens",
        lambda *args, **kwargs: 1,
    )
    files = {"file": ("clear-all.pdf", make_pdf_bytes("RAG cleanup evidence."), "application/pdf")}
    created = client.post("/api/documents/upload", files=files).json()
    client.post(f"/api/documents/{created['id']}/analyze")
    chat = client.post(
        "/api/chat/ask",
        json={"question": "What evidence should cleanup keep?"},
    ).json()

    override = client.app.dependency_overrides[get_db]
    db = next(override())
    try:
        boundary = db.scalar(
            select(ChatMessage)
            .where(ChatMessage.session_id == chat["session_id"])
            .order_by(ChatMessage.id)
            .limit(1)
        )
        assert boundary is not None
        checkpoint = ContextCheckpoint(
            session_id=chat["session_id"],
            summary="Cleanup checkpoint",
            first_kept_message_id=boundary.id,
            tokens_before=128,
            model="test-model",
        )
        db.add(checkpoint)
        db.flush()
        db.add(
            SessionMemory(
                session_id=chat["session_id"],
                goals_and_constraints="cleanup",
                confirmed_findings="present",
                current_decisions="clear all",
                open_questions="none",
                next_actions="reset",
                source_checkpoint_id=checkpoint.id,
            )
        )
        db.commit()
    finally:
        db.close()

    assert client.get("/api/documents").json()
    assert client.get("/api/memories").json()
    assert client.get(f"/api/transcripts/session/{chat['session_id']}").json()
    assert get_paper_chunks_collection().count() > 0

    response = client.post("/api/documents/clear")

    assert response.status_code == 200
    payload = response.json()
    assert payload["deleted"]["documents"] == 1
    assert payload["deleted"]["document_chunks"] >= 1
    assert payload["deleted"]["document_chunks_fts"] >= 1
    assert payload["deleted"]["chat_sessions"] == 1
    assert payload["deleted"]["context_checkpoints"] == 1
    assert payload["deleted"]["session_memories"] == 1
    assert client.get("/api/documents").json() == []
    assert client.get("/api/memories").json() == []
    assert client.get(f"/api/transcripts/session/{chat['session_id']}").json() == []
    assert get_paper_chunks_collection().count() == 0

    override = client.app.dependency_overrides[get_db]
    db = next(override())
    try:
        assert db.scalar(select(Document).limit(1)) is None
        assert db.scalar(select(Memory).limit(1)) is None
        assert db.scalar(select(ChatSession).limit(1)) is None
        assert db.scalar(select(TranscriptEvent).limit(1)) is None
        assert db.scalar(select(ContextCheckpoint).limit(1)) is None
        assert db.scalar(select(SessionMemory).limit(1)) is None
    finally:
        db.close()
