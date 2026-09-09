"""Application service for structured PDF parsing and hybrid indexing."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree
from threading import Lock
from time import perf_counter
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import Settings, get_settings
from ..core.paths import document_dir, staging_dir
from ..ingestion.equations import EquationParser
from ..ingestion.figure_descriptions import MinerUFigureClient
from ..ingestion.figures import FigureParser
from ..ingestion.mineru_formula import MinerUFormulaRecognizer
from ..ingestion.mineru_primary import MinerUPrimaryParser
from ..ingestion.pipeline import PDFIngestionPipeline, PipelineResult
from ..models.chunk import DocumentChunk
from ..models.document import Document
from ..models.page import DocumentPage
from ..models.parsing import DocumentChunkDetail, DocumentElement
from ..rag.bm25_store import delete_document_index, index_chunk
from ..rag.vector_store import delete_document_chunks, upsert_chunks
from .document_store import DocumentStore
from .mineru_figure_runtime import MinerUFigureServiceManager

logger = logging.getLogger(__name__)


def run_primary_with_fallback(
    primary: Any,
    fallback: PDFIngestionPipeline,
    pdf_path: Path,
    paper_id: str,
    title: str,
    output_dir: Path,
) -> PipelineResult:
    """Use MinerU as the sole source, falling back only on whole-document failure."""
    try:
        document = primary.parse(
            pdf_path,
            paper_id=paper_id,
            title=title,
            output_dir=output_dir,
        )
        timings = document.metadata.get("timings_ms", {}) if hasattr(document, "metadata") else {}
        return PipelineResult(
            document=document,
            staging_dir=output_dir,
            timings_ms=dict(timings) if isinstance(timings, dict) else {},
        )
    except Exception:
        logger.warning("MinerU primary parse failed; using legacy fallback", exc_info=True)
        rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        return fallback.run(
            pdf_path,
            paper_id=paper_id,
            title=title,
            output_dir=output_dir,
        )


def _formula_pipeline_root(settings: Settings) -> Path | None:
    root = Path(settings.mineru_root).resolve()
    config_path = root / "mineru.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        pipeline_root = Path(payload["models-dir"]["pipeline"]).resolve()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return pipeline_root if pipeline_root.is_relative_to(root) else None


@contextmanager
def configured_pdf_pipeline(
    settings: Settings | Any | None = None,
    *,
    manager_factory: Any = MinerUFigureServiceManager,
    client_factory: Any = MinerUFigureClient,
    enable_formula: bool = True,
    enable_figure: bool | None = None,
):
    """Build one document-scoped pipeline without implicit model downloads."""
    settings = settings or get_settings()
    recognizer = None
    if enable_formula and settings.mineru_formula_enabled:
        pipeline_root = _formula_pipeline_root(settings)
        if pipeline_root is not None:
            candidate = MinerUFormulaRecognizer(pipeline_root)
            if candidate.model_path.is_dir():
                recognizer = candidate

    manager = None
    def figure_describe(_request):
        return None
    if enable_figure if enable_figure is not None else settings.mineru_figure_enabled:
        manager = manager_factory(
            mineru_root=settings.mineru_root,
            manifest_path=settings.mineru_figure_manifest,
            service_url=settings.mineru_figure_service_url,
            python_executable=settings.mineru_figure_python,
            startup_timeout_seconds=settings.mineru_figure_startup_timeout_seconds,
        )
        client = client_factory(
            settings.mineru_figure_service_url,
            timeout_seconds=settings.mineru_figure_request_timeout_seconds,
        )
        request_lock = Lock()

        def figure_describe(request):
            with request_lock:
                manager.ensure_started()
                return client.describe(
                    request.image_path,
                    title=request.title,
                    section=request.section,
                    caption=request.caption,
                )

    pipeline = PDFIngestionPipeline(
        equation_parser=EquationParser(recognizer=recognizer),
        figure_parser=FigureParser(describe=figure_describe),
    )
    try:
        yield pipeline
    finally:
        if manager is not None:
            manager.close()


def parse_and_store_document(db: Session, document: Document) -> bool:
    """Parse one PDF, persist its structure, and refresh both retrieval indexes."""
    if document.file_type != "pdf":
        document.status = "failed"
        db.add(document)
        db.commit()
        return False

    staging = staging_dir() / f"{document.id}-{uuid4().hex}"
    total_started = perf_counter()
    timings: dict[str, int] = {}
    counts = {
        "formula_success": 0,
        "formula_failed": 0,
        "formula_skipped": 0,
        "figure_enriched": 0,
    }
    try:
        with configured_pdf_pipeline(enable_formula=False, enable_figure=False) as pipeline:
            result = run_primary_with_fallback(
                MinerUPrimaryParser(),
                pipeline,
                Path(document.file_path),
                str(document.id),
                document.title,
                staging,
            )
        timings.update(result.timings_ms)
        if result.document.report.status.value == "failed":
            document.status = "failed"
            db.add(document)
            db.commit()
            return False
        persist_started = perf_counter()
        DocumentStore().save(db, document, result.document, result.staging_dir)
        timings["persist"] = round((perf_counter() - persist_started) * 1000)
    except Exception:
        logger.warning("Document %s parsing failed", document.id, exc_info=True)
        db.rollback()
        try:
            document.status = "failed"
            db.add(document)
            db.commit()
        except Exception:
            db.rollback()
            logger.warning(
                "Document %s failure status could not be persisted", document.id,
                exc_info=True,
            )
        rmtree(staging, ignore_errors=True)
        return False

    rows = db.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.id)
    ).all()
    elements = db.scalars(
        select(DocumentElement).where(DocumentElement.document_id == document.id)
    ).all()
    element_chunk_ids = set(
        db.scalars(
            select(DocumentChunkDetail.chunk_id).where(
                DocumentChunkDetail.element_id.is_not(None)
            )
        ).all()
    )
    element_counts = {"table": 0, "equation": 0, "figure": 0}
    for element in elements:
        if element.element_type in element_counts:
            element_counts[element.element_type] += 1
    counts.update(
        {
            "chunk_count": len(rows),
            "text_chunk_count": sum(1 for row in rows if row.id not in element_chunk_ids),
            "table_count": element_counts["table"],
            "equation_count": element_counts["equation"],
            "figure_count": element_counts["figure"],
        }
    )
    page_images = {
        page.page_number: page.image_path
        for page in db.scalars(
            select(DocumentPage).where(DocumentPage.document_id == document.id)
        ).all()
    }

    bm25_started = perf_counter()
    try:
        delete_document_index(db, document.id)
        for row in rows:
            index_chunk(
                db,
                chunk_id=row.id,
                document_id=document.id,
                title=document.title,
                content=row.content,
            )
        document.status = "parsed"
        db.add(document)
        db.commit()
    except Exception:
        logger.warning("Document %s BM25 indexing failed", document.id, exc_info=True)
        db.rollback()
        document.status = "failed"
        db.add(document)
        db.commit()
        return False
    timings["bm25"] = round((perf_counter() - bm25_started) * 1000)
    _write_pipeline_manifest(document.id, "parsed", timings, counts)

    formula_started = perf_counter()
    if result.document.metadata.get("parser") == "mineru-primary":
        counts["formula_success"] = element_counts["equation"]
    else:
        _enrich_formula_elements(db, document, counts)
    timings["formula_enrichment"] = round(
        (perf_counter() - formula_started) * 1000
    )

    vector_started = perf_counter()
    try:
        db.refresh(document)
    except Exception:
        logger.warning("Document %s committed but refresh failed", document.id, exc_info=True)

    index_payload: list[dict[str, Any]] = [
        {
            "chunk_id": str(row.id),
            "content": row.content,
            "metadata": {
                "chunk_id": row.id,
                "document_id": document.id,
                "title": document.title,
                "page_number": row.page_number,
                "chunk_index": row.chunk_index,
                "file_type": document.file_type,
                "image_path": page_images.get(row.page_number),
            },
        }
        for row in rows
    ]
    try:
        if index_payload:
            upsert_chunks(index_payload)
        else:
            delete_document_chunks(document.id)
    except Exception:
        logger.warning("Vector indexing failed for document %s", document.id, exc_info=True)
    timings["vector_embedding"] = round((perf_counter() - vector_started) * 1000)
    timings["total"] = round((perf_counter() - total_started) * 1000)
    document.status = "indexed"
    db.add(document)
    db.commit()
    _write_pipeline_manifest(document.id, "indexed", timings, counts)
    return True


def _enrich_formula_elements(
    db: Session,
    document: Document,
    counts: dict[str, int],
) -> None:
    settings = get_settings()
    root = _formula_pipeline_root(settings)
    elements = list(
        db.scalars(
            select(DocumentElement).where(
                DocumentElement.document_id == document.id,
                DocumentElement.element_type == "equation",
            )
        )
    )
    if not settings.mineru_formula_enabled or root is None:
        counts["formula_skipped"] += len(elements)
        return
    recognizer = MinerUFormulaRecognizer(root)
    if not recognizer.model_path.is_dir():
        counts["formula_skipped"] += len(elements)
        return
    for element in elements:
        try:
            structured = json.loads(element.structured_data_json or "{}")
        except json.JSONDecodeError:
            structured = {}
        recognition_image = structured.get("recognition_image_path")
        image_path = Path(recognition_image or element.image_path) if (
            recognition_image or element.image_path
        ) else None
        if image_path is None or not image_path.is_file():
            counts["formula_skipped"] += 1
            continue
        try:
            recognition = recognizer.recognize(image_path)
        except Exception as exc:
            element.parse_status = "warning"
            element.structured_data_json = json.dumps(
                {"recognition": {"status": "failed", "error": str(exc)}},
                ensure_ascii=False,
            )
            db.add(element)
            counts["formula_failed"] += 1
            continue
        structured["recognition"] = recognition.to_dict()
        if recognition.latex:
            structured["latex"] = recognition.latex
            element.parse_status = "success"
            counts["formula_success"] += 1
        else:
            counts["formula_failed"] += 1
        element.structured_data_json = json.dumps(structured, ensure_ascii=False)
        detail = db.scalar(
            select(DocumentChunkDetail).where(DocumentChunkDetail.element_id == element.id)
        )
        if detail is not None:
            chunk = db.get(DocumentChunk, detail.chunk_id)
            if chunk is not None and recognition.latex:
                chunk.content = recognition.latex
                index_chunk(
                    db,
                    chunk_id=chunk.id,
                    document_id=document.id,
                    title=document.title,
                    content=chunk.content,
                )
        db.add(element)
    db.commit()


def _write_pipeline_manifest(
    document_id: int,
    stage: str,
    timings_ms: dict[str, int],
    counts: dict[str, int],
) -> None:
    path = document_dir(document_id) / "pipeline-status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"pipeline_stage": stage, "timings_ms": timings_ms, "counts": counts},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
