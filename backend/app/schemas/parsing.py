from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _decode_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _as_items(value: Any) -> list[Any]:
    decoded = _decode_json(value, [])
    if isinstance(decoded, Sequence) and not isinstance(decoded, (str, bytes, bytearray)):
        return list(decoded)
    return []


def _as_mapping(value: Any) -> dict[str, Any]:
    decoded = _decode_json(value, {})
    return dict(decoded) if isinstance(decoded, Mapping) else {}


class ParseRunRead(BaseModel):
    id: int
    document_id: int
    parser_version: str
    status: str
    page_count: int
    block_count: int
    element_count: int
    text_chunk_count: int
    figure_count: int
    table_count: int
    equation_count: int
    text_coverage: float | None
    warnings: list[Any] = Field(
        default_factory=list,
        validation_alias="warnings_json",
    )
    started_at: datetime
    completed_at: datetime | None
    duration_ms: int | None

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @field_validator("warnings", mode="before")
    @classmethod
    def decode_warnings(cls, value: Any) -> list[Any]:
        return _as_items(value)


class DocumentBlockRead(BaseModel):
    id: int
    document_id: int
    parse_run_id: int | None = None
    block_uid: str
    page_number: int
    source_index: int
    bbox: list[float] | None = Field(
        default=None,
        validation_alias="bbox_json",
    )
    text: str
    raw_json: dict[str, Any] | list[Any] | None = None
    block_type: str
    confidence: float | None
    reason_codes: list[str] = Field(
        default_factory=list,
        validation_alias="reason_codes_json",
    )
    reading_order: int
    section: str | None
    subsection: str | None
    section_uid: str | None
    parse_status: str

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @field_validator("bbox", mode="before")
    @classmethod
    def decode_bbox(cls, value: Any) -> list[float] | None:
        decoded = _decode_json(value, None)
        if decoded is None:
            return None
        return [float(item) for item in _as_items(decoded)]

    @field_validator("raw_json", mode="before")
    @classmethod
    def decode_raw_json(cls, value: Any) -> dict[str, Any] | list[Any] | None:
        decoded = _decode_json(value, None)
        if isinstance(decoded, Mapping):
            return dict(decoded)
        if isinstance(decoded, Sequence) and not isinstance(decoded, (str, bytes, bytearray)):
            return list(decoded)
        return None

    @field_validator("reason_codes", mode="before")
    @classmethod
    def decode_reason_codes(cls, value: Any) -> list[str]:
        return [str(item) for item in _as_items(value)]


class SectionNodeRead(BaseModel):
    uid: str
    title: str
    level: int
    parent_uid: str | None = None
    block_uids: list[str] = Field(default_factory=list)
    children: list[SectionNodeRead] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @field_validator("block_uids", mode="before")
    @classmethod
    def decode_block_uids(cls, value: Any) -> list[str]:
        return [str(item) for item in _as_items(value)]


class DocumentElementRead(BaseModel):
    id: int
    document_id: int
    parse_run_id: int | None = None
    element_uid: str
    page_number: int | None
    reading_order: int | None
    element_type: str
    label: str | None
    caption: str | None
    bbox: list[float] | None = Field(
        default=None,
        validation_alias="bbox_json",
    )
    section: str | None
    subsection: str | None
    section_uid: str | None
    image_path: str | None
    raw_text: str | None
    structured_data: dict[str, Any] | list[Any] | None = Field(
        default=None,
        validation_alias="structured_data_json",
    )
    vision_description: str | None
    parse_status: str
    warning_codes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @field_validator("bbox", mode="before")
    @classmethod
    def decode_bbox(cls, value: Any) -> list[float] | None:
        decoded = _decode_json(value, None)
        if decoded is None:
            return None
        return [float(item) for item in _as_items(decoded)]

    @field_validator("structured_data", mode="before")
    @classmethod
    def decode_structured_data(cls, value: Any) -> dict[str, Any] | list[Any] | None:
        decoded = _decode_json(value, None)
        if isinstance(decoded, Mapping):
            return dict(decoded)
        if isinstance(decoded, Sequence) and not isinstance(decoded, (str, bytes, bytearray)):
            return list(decoded)
        return None

    @field_validator("warning_codes", mode="before")
    @classmethod
    def decode_warning_codes(cls, value: Any) -> list[str]:
        return [str(item) for item in _as_items(value)]

    @field_validator("metadata", mode="before")
    @classmethod
    def decode_element_metadata(cls, value: Any) -> dict[str, Any]:
        return _as_mapping(value)


class DocumentChunkDetailRead(BaseModel):
    chunk_id: int
    chunk_uid: str
    chunk_type: str
    page_start: int | None
    page_end: int | None
    section: str | None
    subsection: str | None
    section_uid: str | None
    bbox: list[float] | None = Field(
        default=None,
        validation_alias="bbox_json",
    )
    block_uids: list[str] = Field(
        default_factory=list,
        validation_alias="block_uids_json",
    )
    token_count: int | None
    contextual_prefix: str | None
    embedding_text: str | None
    element_id: int | None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="metadata_json",
    )

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @field_validator("bbox", mode="before")
    @classmethod
    def decode_bbox(cls, value: Any) -> list[float] | None:
        decoded = _decode_json(value, None)
        if decoded is None:
            return None
        return [float(item) for item in _as_items(decoded)]

    @field_validator("block_uids", mode="before")
    @classmethod
    def decode_block_uids(cls, value: Any) -> list[str]:
        return [str(item) for item in _as_items(value)]

    @field_validator("metadata", mode="before")
    @classmethod
    def decode_metadata(cls, value: Any) -> dict[str, Any]:
        return _as_mapping(value)


class StructuredChunkRead(BaseModel):
    uid: str
    text: str
    page_numbers: list[int] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    block_uids: list[str] = Field(default_factory=list)
    element_uids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @field_validator("page_numbers", mode="before")
    @classmethod
    def decode_page_numbers(cls, value: Any) -> list[int]:
        return [int(item) for item in _as_items(value)]

    @field_validator("section_path", "block_uids", "element_uids", mode="before")
    @classmethod
    def decode_string_lists(cls, value: Any) -> list[str]:
        return [str(item) for item in _as_items(value)]

    @field_validator("metadata", mode="before")
    @classmethod
    def decode_metadata(cls, value: Any) -> dict[str, Any]:
        return _as_mapping(value)


class DocumentCrossReferenceRead(BaseModel):
    id: int
    document_id: int
    source_chunk_id: int
    reference_text: str
    reference_type: str
    normalized_label: str | None
    target_element_id: int | None
    target_heading_uid: str | None
    resolution_status: str

    model_config = ConfigDict(from_attributes=True)


class QualityReportRead(BaseModel):
    status: str
    page_count: int = 0
    block_count: int = 0
    element_count: int = 0
    chunk_count: int = 0
    text_chunk_count: int | None = None
    text_coverage: float | None = None
    warnings: list[Any] = Field(default_factory=list)
    errors: list[Any] = Field(default_factory=list)
    pipeline_stage: str | None = None
    timings_ms: dict[str, int] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)

    @field_validator("warnings", "errors", mode="before")
    @classmethod
    def decode_messages(cls, value: Any) -> list[Any]:
        return _as_items(value)
