from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..db.base import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class DocumentParseRun(Base):
    __tablename__ = "document_parse_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    parser_version: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="parsing")
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    block_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    element_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    text_chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    figure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    table_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    equation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    text_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    warnings_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)


class DocumentBlock(Base):
    __tablename__ = "document_blocks"
    __table_args__ = (UniqueConstraint("document_id", "block_uid"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    parse_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_parse_runs.id"), nullable=True, index=True
    )
    block_uid: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    source_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bbox_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    raw_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    block_type: Mapped[str] = mapped_column(Text, nullable=False, default="unknown")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason_codes_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    reading_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    section: Mapped[str | None] = mapped_column(Text, nullable=True)
    subsection: Mapped[str | None] = mapped_column(Text, nullable=True)
    section_uid: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    parse_status: Mapped[str] = mapped_column(Text, nullable=False, default="success")


class DocumentElement(Base):
    __tablename__ = "document_elements"
    __table_args__ = (UniqueConstraint("document_id", "element_uid"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    parse_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_parse_runs.id"), nullable=True, index=True
    )
    element_uid: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reading_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    element_type: Mapped[str] = mapped_column(Text, nullable=False, default="unknown")
    label: Mapped[str | None] = mapped_column(Text, nullable=True)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    bbox_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    section: Mapped[str | None] = mapped_column(Text, nullable=True)
    subsection: Mapped[str | None] = mapped_column(Text, nullable=True)
    section_uid: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    structured_data_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    vision_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    parse_status: Mapped[str] = mapped_column(Text, nullable=False, default="success")


class DocumentChunkDetail(Base):
    __tablename__ = "document_chunk_details"

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("document_chunks.id"), primary_key=True, index=True
    )
    chunk_uid: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_type: Mapped[str] = mapped_column(Text, nullable=False, default="paragraph")
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(Text, nullable=True)
    subsection: Mapped[str | None] = mapped_column(Text, nullable=True)
    section_uid: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    bbox_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    block_uids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    contextual_prefix: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    element_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_elements.id"), nullable=True, index=True
    )
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class DocumentCrossReference(Base):
    __tablename__ = "document_cross_references"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    source_chunk_id: Mapped[int] = mapped_column(
        ForeignKey("document_chunks.id"), index=True
    )
    reference_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    reference_type: Mapped[str] = mapped_column(Text, nullable=False, default="unknown")
    normalized_label: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_element_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_elements.id"), nullable=True, index=True
    )
    target_heading_uid: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolution_status: Mapped[str] = mapped_column(Text, nullable=False, default="unresolved")
