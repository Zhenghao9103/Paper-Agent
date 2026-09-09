from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from backend.app.ingestion.domain import ParseStatus
from backend.app.ingestion.figure_descriptions import FigureDescriptionError
from backend.app.models import Document, DocumentChunk, DocumentChunkDetail, DocumentElement
from backend.app.services import ingestion
from backend.app.services.ingestion import configured_pdf_pipeline
from sqlalchemy import select


def _settings(root: Path, *, figures: bool = False) -> SimpleNamespace:
    pipeline = root / "modelscope" / "pipeline"
    (pipeline / "models" / "MFR" / "unimernet_hf_small_2503").mkdir(parents=True)
    (root / "mineru.json").write_text(
        json.dumps({"models-dir": {"pipeline": str(pipeline)}}), encoding="utf-8"
    )
    return SimpleNamespace(
        mineru_root=str(root),
        mineru_formula_enabled=True,
        mineru_figure_enabled=figures,
        mineru_figure_service_url="http://127.0.0.1:8002",
        mineru_figure_request_timeout_seconds=4.0,
        mineru_figure_startup_timeout_seconds=5.0,
        mineru_figure_python=str(root / "figure-env" / "Scripts" / "python.exe"),
        mineru_figure_manifest=str(root / "figure-model.json"),
    )


def test_configured_pipeline_uses_local_formula_model_and_disables_legacy_vision(
    tmp_path: Path,
) -> None:
    with configured_pdf_pipeline(_settings(tmp_path)) as pipeline:
        assert pipeline.equation_parser.recognizer.model_path.is_relative_to(tmp_path)
        assert pipeline.figure_parser.describe(object()) is None


def test_configured_pipeline_starts_and_closes_owned_figure_service(tmp_path: Path) -> None:
    calls: list[str] = []

    class Manager:
        def ensure_started(self) -> None:
            calls.append("start")

        def close(self) -> None:
            calls.append("close")

    class Client:
        def describe(self, *_args, **_kwargs):
            calls.append("describe")
            raise FigureDescriptionError("figure_description_timeout")

    with configured_pdf_pipeline(
        _settings(tmp_path, figures=True),
        manager_factory=lambda **_kwargs: Manager(),
        client_factory=lambda *_args, **_kwargs: Client(),
    ) as pipeline:
        try:
            pipeline.figure_parser.describe(
                SimpleNamespace(
                    image_path=tmp_path / "figure.png",
                    title="Paper",
                    section=None,
                    caption=None,
                )
            )
        except FigureDescriptionError as exc:
            assert exc.code == "figure_description_timeout"

    assert calls == ["start", "describe", "close"]


def test_formula_config_outside_mineru_root_is_rejected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    outside = tmp_path.parent / "outside-pipeline"
    (outside / "models" / "MFR" / "unimernet_hf_small_2503").mkdir(
        parents=True, exist_ok=True
    )
    (tmp_path / "mineru.json").write_text(
        json.dumps({"models-dir": {"pipeline": str(outside)}}), encoding="utf-8"
    )

    with configured_pdf_pipeline(settings) as pipeline:
        assert pipeline.equation_parser.recognizer is None


def test_parse_enriches_saved_figures_before_final_indexes(
    db_session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = Document(
        title="Batch figure enrichment",
        file_type="pdf",
        file_path=str(tmp_path / "paper.pdf"),
        status="uploaded",
    )
    db_session.add(document)
    db_session.commit()
    events: list[str] = []
    manifests: list[dict[str, int]] = []
    written_vectors: list[dict] = []

    @contextmanager
    def fake_pipeline(**_kwargs):
        yield object()

    parsed = SimpleNamespace(
        report=SimpleNamespace(status=ParseStatus.SUCCESS),
        metadata={"parser": "mineru-primary"},
    )
    result = SimpleNamespace(document=parsed, staging_dir=tmp_path, timings_ms={})

    def fake_save(_self, db, saved_document, _parsed, _staging) -> None:
        events.append("save")
        body = DocumentChunk(
            document_id=saved_document.id,
            page_number=1,
            chunk_index=0,
            content="Preserved body text.",
        )
        figure = DocumentElement(
            document_id=saved_document.id,
            element_uid="figure-1",
            element_type="figure",
            page_number=1,
            parse_status="pending",
        )
        figure_chunk = DocumentChunk(
            document_id=saved_document.id,
            page_number=1,
            chunk_index=1,
            content="Original figure text.",
        )
        db.add_all([body, figure, figure_chunk])
        db.flush()
        db.add(
            DocumentChunkDetail(
                chunk_id=figure_chunk.id,
                chunk_uid="figure-1-chunk",
                chunk_type="figure",
                element_id=figure.id,
            )
        )
        db.commit()

    def fake_enrich(db, saved_document) -> int:
        events.append("enrich")
        figure_chunk = db.scalar(
            select(DocumentChunk)
            .join(DocumentChunkDetail)
            .where(DocumentChunk.document_id == saved_document.id)
        )
        figure_chunk.content = "Enriched figure summary."
        db.commit()
        return 1

    def fake_index_chunk(_db, *, content: str, **_kwargs) -> None:
        events.append(f"bm25:{content}")

    def fake_upsert_chunks(payload) -> None:
        events.append("vector")
        written_vectors.extend(payload)
        assert [item["content"] for item in payload] == [
            "Preserved body text.",
            "Enriched figure summary.",
        ]

    monkeypatch.setattr(ingestion, "configured_pdf_pipeline", fake_pipeline)
    monkeypatch.setattr(ingestion, "run_primary_with_fallback", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(ingestion, "MinerUPrimaryParser", lambda: object())
    monkeypatch.setattr(ingestion.DocumentStore, "save", fake_save)
    monkeypatch.setattr(ingestion, "enrich_document_figures", fake_enrich, raising=False)
    monkeypatch.setattr(ingestion, "delete_document_index", lambda *_args: events.append("delete"))
    monkeypatch.setattr(ingestion, "index_chunk", fake_index_chunk)
    monkeypatch.setattr(ingestion, "upsert_chunks", fake_upsert_chunks)
    monkeypatch.setattr(
        ingestion,
        "document_vector_fingerprints",
        lambda _document_id: {
            str(item["chunk_id"]): str(item["metadata"]["content_sha256"])
            for item in written_vectors
        },
    )
    monkeypatch.setattr(
        ingestion,
        "_write_pipeline_manifest",
        lambda _document_id, _stage, _timings, counts, **_kwargs: manifests.append(
            dict(counts)
        ),
    )
    monkeypatch.setattr(ingestion, "staging_dir", lambda: tmp_path / ".tmp")

    assert ingestion.parse_and_store_document(db_session, document) is True
    assert events == [
        "save",
        "enrich",
        "delete",
        "bm25:Preserved body text.",
        "bm25:Enriched figure summary.",
        "vector",
    ]
    assert manifests[-1]["figure_enriched"] == 1
