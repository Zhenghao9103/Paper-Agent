import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from backend.app.db.base import Base
from backend.app.ingestion.domain import (
    ParseReport,
    ParseStatus,
    ParseWarning,
    SectionNode,
    StructuredChunk,
)
from backend.app.ingestion.figure_descriptions import (
    FigureDescription,
    FigureDescriptionResult,
)
from backend.app.models import (
    Document,
    DocumentBlock,
    DocumentChunk,
    DocumentChunkDetail,
    DocumentCrossReference,
    DocumentElement,
    DocumentParseRun,
)
from backend.app.schemas.parsing import (
    DocumentBlockRead,
    DocumentChunkDetailRead,
    DocumentCrossReferenceRead,
    DocumentElementRead,
    ParseRunRead,
    QualityReportRead,
    SectionNodeRead,
    StructuredChunkRead,
)
from backend.app.services import figure_enrichment
from backend.app.services.document_store import DocumentStore
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session


@pytest.fixture()
def parsing_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'parsing.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _document() -> Document:
    return Document(
        title="Persistence test",
        file_type="pdf",
        file_path="/tmp/persistence-test.pdf",
        status="uploaded",
        source_type="uploaded",
    )


def test_parsing_tables_are_registered_without_changing_legacy_columns(parsing_db):
    table_names = set(inspect(parsing_db.bind).get_table_names())
    assert {
        "document_parse_runs",
        "document_blocks",
        "document_elements",
        "document_chunk_details",
        "document_cross_references",
    } <= table_names

    assert {
        column["name"] for column in inspect(parsing_db.bind).get_columns("documents")
    } == {
        "id",
        "title",
        "authors",
        "year",
        "file_type",
        "file_path",
        "status",
        "source_type",
        "abstract",
        "created_at",
        "updated_at",
    }
    assert {
        column["name"]
        for column in inspect(parsing_db.bind).get_columns("document_chunks")
    } == {
        "id",
        "document_id",
        "page_id",
        "page_number",
        "chunk_index",
        "content",
        "created_at",
    }


def test_parsing_models_round_trip_text_json_and_foreign_keys(parsing_db):
    document = _document()
    parsing_db.add(document)
    parsing_db.flush()

    run = DocumentParseRun(
        document_id=document.id,
        parser_version="v1",
        status="success_with_warnings",
        page_count=2,
        block_count=4,
        element_count=1,
        text_chunk_count=1,
        figure_count=1,
        table_count=0,
        equation_count=0,
        text_coverage=0.75,
        warnings_json=json.dumps([{"code": "low_confidence", "page_number": 2}]),
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        duration_ms=125,
    )
    parsing_db.add(run)
    parsing_db.flush()

    block = DocumentBlock(
        document_id=document.id,
        parse_run_id=run.id,
        block_uid="block-1",
        page_number=1,
        source_index=0,
        bbox_json=json.dumps([1.0, 2.0, 10.0, 20.0]),
        text="Introduction",
        raw_json=json.dumps({"lines": [{"text": "Introduction"}]}),
        block_type="heading",
        confidence=0.98,
        reason_codes_json=json.dumps(["font_rank_heading"]),
        reading_order=0,
        section="Introduction",
        subsection=None,
        section_uid="section-block-1",
        parse_status="success",
    )
    element = DocumentElement(
        document_id=document.id,
        parse_run_id=run.id,
        element_uid="figure-1",
        page_number=1,
        reading_order=1,
        element_type="figure",
        label="Figure 1",
        caption="Overview",
        bbox_json=json.dumps([2.0, 3.0, 20.0, 30.0]),
        section="Introduction",
        subsection=None,
        section_uid="section-block-1",
        image_path="storage/documents/1/figure-1.png",
        raw_text="Overview",
        structured_data_json=json.dumps({"rows": []}),
        vision_description=None,
        parse_status="success",
    )
    chunk = DocumentChunk(
        document_id=document.id,
        page_number=1,
        chunk_index=0,
        content="Introduction text",
    )
    parsing_db.add_all([block, element, chunk])
    parsing_db.flush()
    detail = DocumentChunkDetail(
        chunk_id=chunk.id,
        chunk_uid="chunk-1",
        chunk_type="paragraph",
        page_start=1,
        page_end=1,
        section="Introduction",
        subsection=None,
        section_uid="section-block-1",
        bbox_json=json.dumps([1.0, 2.0, 20.0, 30.0]),
        block_uids_json=json.dumps(["block-1"]),
        token_count=3,
        contextual_prefix="Introduction",
        embedding_text="Introduction\nIntroduction text",
        element_id=element.id,
        metadata_json=json.dumps({"source": "parser"}),
    )
    cross_reference = DocumentCrossReference(
        document_id=document.id,
        source_chunk_id=chunk.id,
        reference_text="Figure 1",
        reference_type="figure",
        normalized_label="figure 1",
        target_element_id=element.id,
        target_heading_uid=None,
        resolution_status="resolved",
    )
    parsing_db.add_all([detail, cross_reference])
    parsing_db.commit()

    parsing_db.expire_all()
    loaded_block = parsing_db.get(DocumentBlock, block.id)
    loaded_element = parsing_db.get(DocumentElement, element.id)
    loaded_detail = parsing_db.get(DocumentChunkDetail, chunk.id)
    loaded_reference = parsing_db.get(DocumentCrossReference, cross_reference.id)
    assert loaded_block.raw_json == json.dumps({"lines": [{"text": "Introduction"}]})
    assert loaded_element.structured_data_json == json.dumps({"rows": []})
    assert loaded_detail.block_uids_json == json.dumps(["block-1"])
    assert loaded_block.section_uid == "section-block-1"
    assert loaded_element.section_uid == "section-block-1"
    assert loaded_detail.section_uid == "section-block-1"
    assert loaded_reference.target_element_id == loaded_element.id


def test_parse_run_assigns_a_utc_start_time_by_default(parsing_db):
    document = _document()
    parsing_db.add(document)
    parsing_db.flush()
    run = DocumentParseRun(document_id=document.id, parser_version="v1")
    parsing_db.add(run)
    parsing_db.flush()
    assert run.started_at is not None


def test_delete_current_rows_removes_orphan_chunk_details_before_id_reuse(
    parsing_db,
) -> None:
    document = _document()
    parsing_db.add(document)
    parsing_db.flush()
    parsing_db.execute(
        DocumentChunkDetail.__table__.insert().values(
            chunk_id=1,
            chunk_uid="orphan-from-legacy-delete",
            chunk_type="text",
        )
    )
    parsing_db.flush()

    DocumentStore._delete_current_rows(parsing_db, document.id)
    replacement = DocumentChunk(
        document_id=document.id,
        page_number=1,
        chunk_index=0,
        content="replacement",
    )
    parsing_db.add(replacement)
    parsing_db.flush()
    DocumentStore._delete_conflicting_chunk_detail(parsing_db, replacement.id)
    parsing_db.add(
        DocumentChunkDetail(
            chunk_id=replacement.id,
            chunk_uid="replacement",
            chunk_type="text",
        )
    )

    parsing_db.flush()
    assert replacement.id == 1


def test_uids_are_unique_per_document_and_cross_reference_target_is_nullable(parsing_db):
    document = _document()
    parsing_db.add(document)
    parsing_db.flush()
    parsing_db.add_all(
        [
            DocumentBlock(
                document_id=document.id,
                block_uid="same",
                page_number=1,
                source_index=0,
                text="one",
                block_type="paragraph",
                parse_status="success",
            ),
            DocumentElement(
                document_id=document.id,
                element_uid="same",
                page_number=1,
                element_type="figure",
                parse_status="success",
            ),
        ]
    )
    parsing_db.commit()

    parsing_db.add(
        DocumentBlock(
            document_id=document.id,
            block_uid="same",
            page_number=2,
            source_index=1,
            text="two",
            block_type="paragraph",
            parse_status="success",
        )
    )
    with pytest.raises(IntegrityError):
        parsing_db.commit()
    parsing_db.rollback()

    chunk = DocumentChunk(
        document_id=document.id,
        page_number=1,
        chunk_index=0,
        content="text",
    )
    parsing_db.add(chunk)
    parsing_db.flush()
    reference = DocumentCrossReference(
        document_id=document.id,
        source_chunk_id=chunk.id,
        reference_text="Section 2",
        reference_type="heading",
        normalized_label="section 2",
        target_element_id=None,
        target_heading_uid="section-block-2",
        resolution_status="unresolved",
    )
    parsing_db.add(reference)
    parsing_db.commit()
    assert reference.target_element_id is None


def test_enrich_document_figures_reuses_service_and_isolates_each_failure(
    parsing_db,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document()
    parsing_db.add(document)
    parsing_db.flush()
    body = DocumentChunk(
        document_id=document.id,
        page_number=1,
        chunk_index=0,
        content="Body text must survive figure enrichment.",
    )
    parsing_db.add(body)
    parsing_db.flush()

    figure_chunks: dict[str, DocumentChunk] = {}
    for index, name in enumerate(("first", "broken", "last"), start=1):
        image = tmp_path / f"{name}.png"
        image.write_bytes(b"image")
        element = DocumentElement(
            document_id=document.id,
            element_uid=f"figure-{name}",
            element_type="figure",
            page_number=index,
            image_path=str(image),
            structured_data_json=json.dumps({"source": name}),
            parse_status="pending",
        )
        chunk = DocumentChunk(
            document_id=document.id,
            page_number=index,
            chunk_index=index,
            content=f"Original {name} figure text.",
        )
        parsing_db.add_all([element, chunk])
        parsing_db.flush()
        parsing_db.add(
            DocumentChunkDetail(
                chunk_id=chunk.id,
                chunk_uid=f"figure-{name}-chunk",
                chunk_type="figure",
                element_id=element.id,
            )
        )
        figure_chunks[name] = chunk
    parsing_db.commit()

    calls: list[str] = []

    class FakeManager:
        def __init__(self, **_kwargs) -> None:
            calls.append("manager")

        def ensure_started(self) -> None:
            calls.append("start")

        def schedule_idle_close(self, _seconds: float) -> None:
            calls.append("idle")

    class FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            calls.append("client")

        def describe(self, image_path: Path, **_kwargs) -> FigureDescriptionResult:
            calls.append(f"describe:{image_path.stem}")
            if image_path.stem == "broken":
                raise RuntimeError("broken figure")
            return FigureDescriptionResult(
                description=FigureDescription(summary=f"Summary for {image_path.stem}."),
                model="fake-model",
                latency_ms=1,
            )

    settings = SimpleNamespace(
        mineru_figure_enabled=True,
        mineru_root=str(tmp_path),
        mineru_figure_manifest=str(tmp_path / "manifest.json"),
        mineru_figure_service_url="http://127.0.0.1:8002",
        mineru_figure_python=str(tmp_path / "python.exe"),
        mineru_figure_startup_timeout_seconds=2.0,
        mineru_figure_request_timeout_seconds=3.0,
        mineru_figure_idle_timeout_seconds=4.0,
    )
    monkeypatch.setattr(figure_enrichment, "get_settings", lambda: settings)
    monkeypatch.setattr(figure_enrichment, "MinerUFigureServiceManager", FakeManager)
    monkeypatch.setattr(figure_enrichment, "MinerUFigureClient", FakeClient)
    monkeypatch.setattr(
        figure_enrichment,
        "index_chunk",
        lambda **_kwargs: pytest.fail("batch enrichment must defer BM25 indexing"),
    )
    monkeypatch.setattr(
        figure_enrichment,
        "upsert_chunk",
        lambda **_kwargs: pytest.fail("batch enrichment must defer vector indexing"),
    )

    enrich_document_figures = getattr(
        figure_enrichment, "enrich_document_figures", None
    )
    assert enrich_document_figures is not None
    enriched = enrich_document_figures(parsing_db, document)

    assert enriched == 2
    assert calls == [
        "manager",
        "client",
        "start",
        "describe:first",
        "describe:broken",
        "describe:last",
        "idle",
    ]
    assert parsing_db.get(DocumentChunk, body.id).content == body.content
    assert "Summary for first." in parsing_db.get(
        DocumentChunk, figure_chunks["first"].id
    ).content
    assert "Summary for last." in parsing_db.get(
        DocumentChunk, figure_chunks["last"].id
    ).content
    assert parsing_db.get(
        DocumentChunk, figure_chunks["broken"].id
    ).content == "Original broken figure text."
    broken = parsing_db.scalar(
        select(DocumentElement).where(DocumentElement.element_uid == "figure-broken")
    )
    assert broken.parse_status == "warning"
    assert json.loads(broken.structured_data_json)["enrichment_error"] == "broken figure"


def test_single_figure_with_empty_summary_is_not_described_or_chunked_twice(
    parsing_db,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document()
    parsing_db.add(document)
    parsing_db.flush()
    image = tmp_path / "empty-summary-single.png"
    image.write_bytes(b"image")
    element = DocumentElement(
        document_id=document.id,
        element_uid="figure-empty-summary-single",
        element_type="figure",
        page_number=1,
        image_path=str(image),
        structured_data_json="{}",
        parse_status="pending",
    )
    parsing_db.add(element)
    parsing_db.commit()
    calls = {"describe": 0}

    class FakeManager:
        def ensure_started(self) -> None:
            pass

        def schedule_idle_close(self, _seconds: float) -> None:
            pass

    class FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def describe(self, *_args, **_kwargs) -> FigureDescriptionResult:
            calls["describe"] += 1
            return FigureDescriptionResult(
                description=FigureDescription(figure_type="diagram", summary=""),
                model="fake-model",
                latency_ms=1,
            )

    settings = SimpleNamespace(
        mineru_figure_service_url="http://127.0.0.1:8002",
        mineru_figure_request_timeout_seconds=3.0,
        mineru_figure_idle_timeout_seconds=4.0,
    )
    monkeypatch.setattr(figure_enrichment, "get_settings", lambda: settings)
    monkeypatch.setattr(figure_enrichment, "_manager", FakeManager())
    monkeypatch.setattr(figure_enrichment, "MinerUFigureClient", FakeClient)
    monkeypatch.setattr(figure_enrichment, "index_chunk", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(figure_enrichment, "upsert_chunk", lambda **_kwargs: None)

    figure_enrichment.enrich_figure(parsing_db, document, element)
    figure_enrichment.enrich_figure(parsing_db, document, element)

    chunks = list(
        parsing_db.scalars(
            select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
    )
    assert calls["describe"] == 1
    assert len(chunks) == 1


def test_batch_figure_with_empty_summary_is_not_described_or_chunked_twice(
    parsing_db,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document()
    parsing_db.add(document)
    parsing_db.flush()
    image = tmp_path / "empty-summary-batch.png"
    image.write_bytes(b"image")
    element = DocumentElement(
        document_id=document.id,
        element_uid="figure-empty-summary-batch",
        element_type="figure",
        page_number=1,
        image_path=str(image),
        structured_data_json="{}",
        parse_status="pending",
    )
    parsing_db.add(element)
    parsing_db.commit()
    calls = {"describe": 0}

    class FakeManager:
        def __init__(self, **_kwargs) -> None:
            pass

        def ensure_started(self) -> None:
            pass

        def schedule_idle_close(self, _seconds: float) -> None:
            pass

    class FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def describe(self, *_args, **_kwargs) -> FigureDescriptionResult:
            calls["describe"] += 1
            return FigureDescriptionResult(
                description=FigureDescription(figure_type="diagram", summary=""),
                model="fake-model",
                latency_ms=1,
            )

    settings = SimpleNamespace(
        mineru_figure_enabled=True,
        mineru_root=str(tmp_path),
        mineru_figure_manifest=str(tmp_path / "manifest.json"),
        mineru_figure_service_url="http://127.0.0.1:8002",
        mineru_figure_python=str(tmp_path / "python.exe"),
        mineru_figure_startup_timeout_seconds=2.0,
        mineru_figure_request_timeout_seconds=3.0,
        mineru_figure_idle_timeout_seconds=4.0,
    )
    monkeypatch.setattr(figure_enrichment, "get_settings", lambda: settings)
    monkeypatch.setattr(figure_enrichment, "MinerUFigureServiceManager", FakeManager)
    monkeypatch.setattr(figure_enrichment, "MinerUFigureClient", FakeClient)

    assert figure_enrichment.enrich_document_figures(parsing_db, document) == 1
    assert figure_enrichment.enrich_document_figures(parsing_db, document) == 0

    chunks = list(
        parsing_db.scalars(
            select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
    )
    assert calls["describe"] == 1
    assert len(chunks) == 1


def test_single_figure_start_failure_keeps_primary_error_when_idle_close_fails(
    parsing_db,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document()
    parsing_db.add(document)
    parsing_db.flush()
    image = tmp_path / "start-failure.png"
    image.write_bytes(b"image")
    element = DocumentElement(
        document_id=document.id,
        element_uid="figure-start-failure",
        element_type="figure",
        image_path=str(image),
        structured_data_json="{}",
        parse_status="pending",
    )
    parsing_db.add(element)
    parsing_db.commit()
    calls: list[str] = []

    class FakeManager:
        def ensure_started(self) -> None:
            calls.append("start")
            raise RuntimeError("start failed")

        def schedule_idle_close(self, _seconds: float) -> None:
            calls.append("idle")
            raise RuntimeError("idle failed")

    settings = SimpleNamespace(
        mineru_figure_service_url="http://127.0.0.1:8002",
        mineru_figure_request_timeout_seconds=3.0,
        mineru_figure_idle_timeout_seconds=4.0,
    )
    monkeypatch.setattr(figure_enrichment, "get_settings", lambda: settings)
    monkeypatch.setattr(figure_enrichment, "_manager", FakeManager())

    with pytest.raises(RuntimeError, match="start failed"):
        figure_enrichment.enrich_figure(parsing_db, document, element)

    assert calls == ["start", "idle"]
    assert element.parse_status == "warning"
    assert json.loads(element.structured_data_json)["enrichment_error"] == "start failed"


def test_single_figure_idle_close_failure_does_not_fail_successful_enrichment(
    parsing_db,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document()
    parsing_db.add(document)
    parsing_db.flush()
    image = tmp_path / "idle-failure.png"
    image.write_bytes(b"image")
    element = DocumentElement(
        document_id=document.id,
        element_uid="figure-idle-failure",
        element_type="figure",
        image_path=str(image),
        structured_data_json="{}",
        parse_status="pending",
    )
    parsing_db.add(element)
    parsing_db.commit()
    calls: list[str] = []

    class FakeManager:
        def ensure_started(self) -> None:
            calls.append("start")

        def schedule_idle_close(self, _seconds: float) -> None:
            calls.append("idle")
            raise RuntimeError("idle failed")

    class FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def describe(self, *_args, **_kwargs) -> FigureDescriptionResult:
            calls.append("describe")
            return FigureDescriptionResult(
                description=FigureDescription(summary="Figure summary."),
                model="fake-model",
                latency_ms=1,
            )

    settings = SimpleNamespace(
        mineru_figure_service_url="http://127.0.0.1:8002",
        mineru_figure_request_timeout_seconds=3.0,
        mineru_figure_idle_timeout_seconds=4.0,
    )
    monkeypatch.setattr(figure_enrichment, "get_settings", lambda: settings)
    monkeypatch.setattr(figure_enrichment, "_manager", FakeManager())
    monkeypatch.setattr(figure_enrichment, "MinerUFigureClient", FakeClient)
    monkeypatch.setattr(figure_enrichment, "index_chunk", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(figure_enrichment, "upsert_chunk", lambda **_kwargs: None)

    result = figure_enrichment.enrich_figure(parsing_db, document, element)

    assert result.parse_status == "success"
    assert calls == ["start", "describe", "idle"]


def test_read_schemas_decode_json_and_preserve_nullable_fields(parsing_db):
    run = DocumentParseRun(
        id=1,
        document_id=1,
        parser_version="v1",
        status="success",
        page_count=1,
        block_count=1,
        element_count=1,
        text_chunk_count=1,
        figure_count=0,
        table_count=0,
        equation_count=0,
        text_coverage=1.0,
        warnings_json='[{"code":"none"}]',
        started_at=datetime.utcnow(),
        duration_ms=None,
    )
    block = DocumentBlock(
        id=1,
        document_id=1,
        block_uid="b1",
        page_number=1,
        source_index=0,
        bbox_json="[0, 1, 2, 3]",
        text="body",
        raw_json='{"kind":"text"}',
        block_type="paragraph",
        confidence=0.9,
        reason_codes_json='["rule"]',
        reading_order=2,
        parse_status="success",
        section_uid="section-b1",
    )
    element = DocumentElement(
        id=1,
        document_id=1,
        element_uid="e1",
        page_number=1,
        element_type="table",
        bbox_json="[1, 2, 3, 4]",
        structured_data_json='{"columns":["a"]}',
        parse_status="partial",
        section_uid="section-e1",
    )
    detail = DocumentChunkDetail(
        chunk_id=1,
        chunk_uid="c1",
        chunk_type="table",
        page_start=1,
        page_end=2,
        bbox_json="[0, 0, 4, 4]",
        section_uid="section-b1",
        block_uids_json='["b1"]',
        token_count=12,
        metadata_json='{"source":"test"}',
    )
    cross_reference = DocumentCrossReference(
        id=1,
        document_id=1,
        source_chunk_id=1,
        reference_text="Table 1",
        reference_type="table",
        normalized_label="table 1",
        target_element_id=None,
        target_heading_uid=None,
        resolution_status="unresolved",
    )

    assert ParseRunRead.model_validate(run).warnings == [{"code": "none"}]
    assert DocumentBlockRead.model_validate(block).bbox == [0.0, 1.0, 2.0, 3.0]
    assert DocumentBlockRead.model_validate(block).reason_codes == ["rule"]
    assert DocumentElementRead.model_validate(element).structured_data == {
        "columns": ["a"]
    }
    assert DocumentBlockRead.model_validate(block).section_uid == "section-b1"
    assert DocumentElementRead.model_validate(element).section_uid == "section-e1"
    assert DocumentChunkDetailRead.model_validate(detail).section_uid == "section-b1"
    assert DocumentChunkDetailRead.model_validate(detail).block_uids == ["b1"]
    assert DocumentChunkDetailRead.model_validate(detail).metadata == {"source": "test"}
    assert DocumentCrossReferenceRead.model_validate(cross_reference).target_element_id is None

    section = SectionNodeRead(
        uid="section-1",
        title="Introduction",
        level=1,
        block_uids='["b1"]',
        children=[],
    )
    structured = StructuredChunkRead(
        uid="c1",
        text="body",
        page_numbers="[1, 2]",
        section_path='["Introduction"]',
        block_uids='["b1"]',
        element_uids='["e1"]',
        metadata='{"source":"test"}',
    )
    quality = QualityReportRead(
        status="success",
        page_count=1,
        block_count=1,
        element_count=1,
        chunk_count=1,
        text_coverage=1.0,
        warnings='[{"code":"none"}]',
    )
    assert section.block_uids == ["b1"]
    assert structured.page_numbers == [1, 2]
    assert structured.metadata == {"source": "test"}
    assert quality.warnings == [{"code": "none"}]


def test_read_schemas_accept_domain_sequences_and_mappings():
    section = SectionNode(uid="section-1", title="Introduction", level=1, block_uids=["b1"])
    section.children.append(SectionNode(uid="section-1-1", title="Methods", level=2))
    chunk = StructuredChunk(
        uid="chunk-1",
        text="body",
        page_numbers=(1, 2),
        section_path=("Introduction", "Methods"),
        block_uids=("b1",),
        element_uids=("e1",),
        metadata={"source": "domain"},
    )
    report = ParseReport(
        status=ParseStatus.SUCCESS_WITH_WARNINGS,
        page_count=2,
        block_count=3,
        element_count=1,
        chunk_count=1,
        warnings=[ParseWarning(code="notice", message="partial")],
    )

    section_read = SectionNodeRead.model_validate(section)
    chunk_read = StructuredChunkRead.model_validate(chunk)
    report_read = QualityReportRead.model_validate(report)
    assert section_read.block_uids == ["b1"]
    assert section_read.children[0].title == "Methods"
    assert chunk_read.page_numbers == [1, 2]
    assert chunk_read.section_path == ["Introduction", "Methods"]
    assert chunk_read.metadata == {"source": "domain"}
    assert report_read.status == "success_with_warnings"
    assert report_read.warnings[0].code == "notice"
