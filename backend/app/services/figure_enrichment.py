"""On-demand MinerU enrichment for persisted figure elements."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..ingestion.figure_descriptions import MinerUFigureClient, project_figure_description
from ..models.chunk import DocumentChunk
from ..models.document import Document
from ..models.page import DocumentPage
from ..models.parsing import DocumentChunkDetail, DocumentElement
from ..rag.bm25_store import index_chunk
from ..rag.vector_store import upsert_chunk
from .mineru_figure_runtime import MinerUFigureServiceManager

_lock = Lock()
_manager: MinerUFigureServiceManager | None = None


def enrich_figure(db: Session, document: Document, element: DocumentElement) -> DocumentElement:
    if element.element_type != "figure":
        raise ValueError("Only figure elements can be enriched")
    try:
        structured = json.loads(element.structured_data_json or "{}")
    except json.JSONDecodeError:
        structured = {}
    if structured.get("summary") and structured.get("description_model"):
        return element
    if not element.image_path:
        raise FileNotFoundError("Figure image not found")

    settings = get_settings()
    global _manager
    with _lock:
        if _manager is None:
            _manager = MinerUFigureServiceManager(
                mineru_root=settings.mineru_root,
                manifest_path=settings.mineru_figure_manifest,
                service_url=settings.mineru_figure_service_url,
                python_executable=settings.mineru_figure_python,
                startup_timeout_seconds=settings.mineru_figure_startup_timeout_seconds,
            )
        _manager.ensure_started()
        try:
            result = MinerUFigureClient(
                settings.mineru_figure_service_url,
                timeout_seconds=settings.mineru_figure_request_timeout_seconds,
            ).describe(
                Path(element.image_path),
                title=document.title,
                section=element.section,
                caption=element.caption,
            )
        except Exception as exc:
            structured["enrichment_error"] = str(exc)
            element.structured_data_json = json.dumps(structured, ensure_ascii=False)
            element.parse_status = "warning"
            db.add(element)
            db.commit()
            raise
        finally:
            _manager.schedule_idle_close(settings.mineru_figure_idle_timeout_seconds)

    structured = result.description.model_dump()
    structured.update(
        {
            "description_model": result.model,
            "description_latency_ms": result.latency_ms,
        }
    )
    element.structured_data_json = json.dumps(structured, ensure_ascii=False)
    element.vision_description = result.description.summary or None
    element.parse_status = "success"
    db.add(element)
    detail = db.scalar(
        select(DocumentChunkDetail).where(DocumentChunkDetail.element_id == element.id)
    )
    chunk = db.get(DocumentChunk, detail.chunk_id) if detail is not None else None
    if chunk is None:
        page_number = element.page_number or 0
        page_id = db.scalar(
            select(DocumentPage.id).where(
                DocumentPage.document_id == document.id,
                DocumentPage.page_number == page_number,
            )
        )
        next_index = (db.scalar(
            select(func.max(DocumentChunk.chunk_index)).where(
                DocumentChunk.document_id == document.id
            )
        ) or -1) + 1
        chunk = DocumentChunk(
            document_id=document.id,
            page_id=page_id,
            page_number=page_number,
            chunk_index=next_index,
            content="",
        )
        db.add(chunk)
        db.flush()
        db.add(
            DocumentChunkDetail(
                chunk_id=chunk.id,
                chunk_uid=f"{element.element_uid}-figure",
                chunk_type="figure",
                page_start=element.page_number,
                page_end=element.page_number,
                section=element.section,
                subsection=element.subsection,
                section_uid=element.section_uid,
                element_id=element.id,
                metadata_json=json.dumps({"chunk_type": "figure"}),
            )
        )
    chunk.content = project_figure_description(result.description)
    db.add(chunk)
    index_chunk(
        db,
        chunk_id=chunk.id,
        document_id=document.id,
        title=document.title,
        content=chunk.content,
    )
    db.commit()
    try:
        upsert_chunk(
            chunk_id=str(chunk.id),
            content=chunk.content,
            metadata={
                "chunk_id": chunk.id,
                "document_id": document.id,
                "title": document.title,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "file_type": document.file_type,
            },
        )
    except Exception:
        pass
    return element
