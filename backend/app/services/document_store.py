"""Transactional persistence for the structured PDF parse result."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..core.paths import document_dir, documents_dir
from ..ingestion.domain import ParsedDocument, ParseStatus
from ..ingestion.quality import ParseQualityChecker
from ..models import (
    Document,
    DocumentBlock,
    DocumentChunk,
    DocumentChunkDetail,
    DocumentCrossReference,
    DocumentElement,
    DocumentPage,
    DocumentParseRun,
)


class DocumentStore:
    """Persist one parse atomically while retaining compatible legacy rows."""

    parser_version = "mineru-primary-v1"

    def save(
        self,
        db: Session,
        document: Document,
        parsed_document: ParsedDocument,
        staging_path: Path,
    ) -> DocumentParseRun:
        staging = Path(staging_path).resolve()
        self._validate_staging(staging)
        report = ParseQualityChecker().check(parsed_document)
        if report.status is ParseStatus.FAILED:
            raise ValueError("Parsed document failed quality validation")
        if document.id is None:
            db.flush()
        if document.id is None:
            raise ValueError("document must have a database id")

        formal = document_dir(document.id).resolve()
        root = documents_dir().resolve()
        self._assert_child(formal, root)
        backup = root / f".{document.id}.backup-{uuid4().hex}"
        started = time.perf_counter()
        run: DocumentParseRun | None = None
        promoted = False
        had_formal = formal.exists()
        try:
            self._delete_current_rows(db, document.id)
            run = self._insert_parse_run(db, document.id, parsed_document)
            page_ids = self._insert_pages(
                db, document.id, parsed_document, staging, formal
            )
            self._insert_blocks(db, document.id, run.id, parsed_document)
            element_ids = self._insert_elements(
                db, document.id, run.id, parsed_document, staging, formal
            )
            chunk_ids = self._insert_chunks(
                db,
                document.id,
                parsed_document,
                page_ids,
                element_ids,
                staging,
                formal,
            )
            self._insert_cross_references(
                db,
                document.id,
                parsed_document,
                chunk_ids,
                element_ids,
            )
            document.status = "parsed"
            db.flush()

            if had_formal:
                formal.rename(backup)
            try:
                formal.parent.mkdir(parents=True, exist_ok=True)
                staging.rename(formal)
                promoted = True
            except Exception:
                if had_formal and backup.exists() and not formal.exists():
                    backup.rename(formal)
                raise

            run.completed_at = _utcnow()
            run.duration_ms = int((time.perf_counter() - started) * 1000)
            db.flush()
            db.commit()
        except Exception:
            db.rollback()
            if promoted and formal.exists():
                shutil.rmtree(formal)
            if backup.exists() and not formal.exists():
                backup.rename(formal)
            raise
        else:
            if backup.exists():
                shutil.rmtree(backup)
            if run is None:  # pragma: no cover - defensive for type checkers
                raise RuntimeError("parse run was not created")
            return run

    @staticmethod
    def _validate_staging(staging: Path) -> None:
        if not staging.exists():
            raise FileNotFoundError(staging)
        if not staging.is_dir():
            raise NotADirectoryError(staging)

    @staticmethod
    def _assert_child(path: Path, parent: Path) -> None:
        try:
            path.relative_to(parent)
        except ValueError as exc:
            raise ValueError("document path escapes documents root") from exc

    @staticmethod
    def _delete_current_rows(db: Session, document_id: int) -> None:
        chunk_ids = list(
            db.scalars(select(DocumentChunk.id).where(DocumentChunk.document_id == document_id))
        )
        if chunk_ids:
            db.execute(
                delete(DocumentCrossReference).where(
                    DocumentCrossReference.source_chunk_id.in_(chunk_ids)
                )
            )
            db.execute(
                delete(DocumentChunkDetail).where(DocumentChunkDetail.chunk_id.in_(chunk_ids))
            )
        db.execute(
            delete(DocumentCrossReference).where(
                DocumentCrossReference.document_id == document_id
            )
        )
        if chunk_ids:
            db.execute(delete(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids)))
        db.execute(delete(DocumentElement).where(DocumentElement.document_id == document_id))
        db.execute(delete(DocumentBlock).where(DocumentBlock.document_id == document_id))
        db.execute(delete(DocumentPage).where(DocumentPage.document_id == document_id))

    @staticmethod
    def _insert_parse_run(
        db: Session, document_id: int, parsed: ParsedDocument
    ) -> DocumentParseRun:
        report = parsed.report
        run = DocumentParseRun(
            document_id=document_id,
            parser_version=DocumentStore.parser_version,
            status=report.status.value,
            page_count=len(parsed.pages),
            block_count=len(parsed.blocks),
            element_count=len(parsed.elements),
            text_chunk_count=sum(1 for chunk in parsed.chunks if not chunk.element_uids),
            figure_count=sum(
                1 for element in parsed.elements if element.element_type.value == "figure"
            ),
            table_count=sum(
                1 for element in parsed.elements if element.element_type.value == "table"
            ),
            equation_count=sum(
                1 for element in parsed.elements if element.element_type.value == "equation"
            ),
            text_coverage=report.text_coverage,
            warnings_json=json.dumps(
                [warning.to_dict() for warning in report.warnings],
                ensure_ascii=False,
            ),
            started_at=_utcnow(),
        )
        db.add(run)
        db.flush()
        return run

    @staticmethod
    def _insert_pages(
        db: Session,
        document_id: int,
        parsed: ParsedDocument,
        staging: Path,
        formal: Path,
    ) -> dict[int, int]:
        page_ids: dict[int, int] = {}
        for page in parsed.pages:
            text = "\n".join(block.text for block in page.blocks if block.text.strip())
            candidate = staging / f"page-{page.page_number}.png"
            final_candidate = formal / f"page-{page.page_number}.png"
            row = DocumentPage(
                document_id=document_id,
                page_number=page.page_number,
                text=text,
                image_path=str(final_candidate) if candidate.exists() else None,
            )
            db.add(row)
            db.flush()
            page_ids[page.page_number] = row.id
        return page_ids

    @staticmethod
    def _insert_blocks(
        db: Session,
        document_id: int,
        run_id: int,
        parsed: ParsedDocument,
    ) -> None:
        for reading_order, block in enumerate(parsed.blocks):
            source_index = (
                block.source_block_indices[0]
                if block.source_block_indices
                else reading_order
            )
            db.add(
                DocumentBlock(
                    document_id=document_id,
                    parse_run_id=run_id,
                    block_uid=block.uid,
                    page_number=block.page_number,
                    source_index=source_index,
                    bbox_json=json.dumps(block.bbox.as_list()),
                    text=block.text,
                    raw_json=json.dumps(block.to_dict(), ensure_ascii=False),
                    block_type=block.block_type.value,
                    confidence=block.confidence,
                    reason_codes_json=json.dumps(list(block.reason_codes), ensure_ascii=False),
                    reading_order=reading_order,
                    section=block.section,
                    subsection=block.subsection,
                    section_uid=block.section_uid,
                    parse_status="success",
                )
            )
        db.flush()

    @staticmethod
    def _insert_elements(
        db: Session,
        document_id: int,
        run_id: int,
        parsed: ParsedDocument,
        staging: Path,
        formal: Path,
    ) -> dict[str, int]:
        element_ids: dict[str, int] = {}
        for reading_order, element in enumerate(parsed.elements):
            metadata = _rewrite_asset_metadata(
                element.to_dict().get("metadata", {}), staging, formal
            )
            image_path = _final_asset_path(element.image_path, staging, formal)
            row = DocumentElement(
                document_id=document_id,
                parse_run_id=run_id,
                element_uid=element.uid,
                page_number=element.page_number,
                reading_order=reading_order,
                element_type=element.element_type.value,
                label=element.label,
                caption=element.caption,
                bbox_json=json.dumps(element.bbox.as_list()),
                section=element.section,
                subsection=element.subsection,
                section_uid=element.section_uid,
                image_path=str(image_path) if image_path else None,
                raw_text=element.content,
                structured_data_json=json.dumps(metadata.get("structured_data"), ensure_ascii=False)
                if metadata.get("structured_data") is not None
                else None,
                vision_description=element.vision_description,
                parse_status=element.status.value,
            )
            db.add(row)
            db.flush()
            element_ids[element.uid] = row.id
        return element_ids

    @staticmethod
    def _insert_chunks(
        db: Session,
        document_id: int,
        parsed: ParsedDocument,
        page_ids: dict[int, int],
        element_ids: dict[str, int],
        staging: Path,
        formal: Path,
    ) -> dict[str, int]:
        chunk_ids: dict[str, int] = {}
        for index, chunk in enumerate(parsed.chunks):
            metadata = _rewrite_asset_metadata(
                chunk.to_dict().get("metadata", {}), staging, formal
            )
            page_numbers = list(chunk.page_numbers)
            page_start = min(page_numbers) if page_numbers else None
            page_end = max(page_numbers) if page_numbers else None
            row = DocumentChunk(
                document_id=document_id,
                page_id=page_ids.get(page_start) if page_start is not None else None,
                page_number=page_start or 0,
                chunk_index=index,
                content=chunk.text,
            )
            db.add(row)
            db.flush()
            DocumentStore._delete_conflicting_chunk_detail(db, row.id)
            element_id = next(
                (element_ids[uid] for uid in chunk.element_uids if uid in element_ids),
                None,
            )
            db.add(
                DocumentChunkDetail(
                    chunk_id=row.id,
                    chunk_uid=chunk.uid,
                    chunk_type=str(metadata.get("chunk_type", "text")),
                    page_start=page_start,
                    page_end=page_end,
                    section=metadata.get("section")
                    or (chunk.section_path[0] if chunk.section_path else None),
                    subsection=metadata.get("subsection")
                    or (chunk.section_path[-1] if len(chunk.section_path) > 1 else None),
                    section_uid=metadata.get("section_uid"),
                    bbox_json=json.dumps(metadata.get("bbox"), ensure_ascii=False)
                    if metadata.get("bbox") is not None
                    else None,
                    block_uids_json=json.dumps(list(chunk.block_uids), ensure_ascii=False),
                    token_count=_int_or_none(metadata.get("token_count")),
                    contextual_prefix=metadata.get("contextual_prefix"),
                    embedding_text=metadata.get("embedding_text"),
                    element_id=element_id,
                    metadata_json=json.dumps(metadata, ensure_ascii=False),
                )
            )
            chunk_ids[chunk.uid] = row.id
        db.flush()
        return chunk_ids

    @staticmethod
    def _delete_conflicting_chunk_detail(db: Session, chunk_id: int) -> None:
        if db.get(DocumentChunkDetail, chunk_id) is None:
            return
        db.execute(
            delete(DocumentChunkDetail).where(DocumentChunkDetail.chunk_id == chunk_id)
        )

    @staticmethod
    def _insert_cross_references(
        db: Session,
        document_id: int,
        parsed: ParsedDocument,
        chunk_ids: dict[str, int],
        element_ids: dict[str, int],
    ) -> None:
        references = getattr(parsed, "cross_references", ())
        if not references:
            return
        block_to_chunk: dict[str, int] = {}
        for chunk in parsed.chunks:
            chunk_id = chunk_ids.get(chunk.uid)
            if chunk_id is not None:
                for block_uid in chunk.block_uids:
                    block_to_chunk[block_uid] = chunk_id
        for reference in references:
            source_uid = getattr(reference, "source_uid", None)
            source_chunk_id = chunk_ids.get(source_uid) or block_to_chunk.get(source_uid)
            if source_chunk_id is None:
                continue
            target_uid = getattr(reference, "target_id", None)
            target_element_id = element_ids.get(target_uid) if target_uid else None
            target_heading_uid = (
                target_uid
                if target_uid and getattr(reference, "reference_type", "") == "section"
                else None
            )
            db.add(
                DocumentCrossReference(
                    document_id=document_id,
                    source_chunk_id=source_chunk_id,
                    reference_text=str(getattr(reference, "reference_text", "")),
                    reference_type=str(getattr(reference, "reference_type", "unknown")),
                    normalized_label=getattr(reference, "normalized_label", None),
                    target_element_id=target_element_id,
                    target_heading_uid=target_heading_uid,
                    resolution_status=str(
                        getattr(reference, "resolution_status", "unresolved")
                    ),
                )
            )
        db.flush()


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _final_asset_path(
    value: Path | None,
    staging: Path,
    formal: Path,
) -> Path | None:
    if value is None:
        return None
    candidate = Path(value)
    try:
        relative = candidate.resolve().relative_to(staging.resolve())
    except ValueError:
        return candidate
    return formal / relative


def _rewrite_asset_metadata(
    metadata: object,
    staging: Path,
    formal: Path,
) -> dict[str, object]:
    if not isinstance(metadata, dict):
        return {}
    rewritten = dict(metadata)
    for key, value in tuple(rewritten.items()):
        if isinstance(value, dict):
            rewritten[key] = _rewrite_asset_metadata(value, staging, formal)
            continue
        if (key.endswith("image_path") or key == "recognition_image_path") and value:
            final_path = _final_asset_path(Path(str(value)), staging, formal)
            rewritten[key] = str(final_path) if final_path else None
    return rewritten


def _utcnow():
    from datetime import UTC, datetime

    return datetime.now(UTC)
