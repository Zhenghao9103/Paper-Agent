from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import Enum, StrEnum
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any


class BlockType(StrEnum):
    TITLE = "title"
    AUTHOR = "author"
    ABSTRACT = "abstract"
    HEADING = "heading"
    SUBHEADING = "subheading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    FIGURE = "figure"
    FIGURE_CAPTION = "figure_caption"
    TABLE = "table"
    TABLE_CAPTION = "table_caption"
    EQUATION = "equation"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    REFERENCE = "reference"
    UNKNOWN = "unknown"


class ElementType(StrEnum):
    FIGURE = "figure"
    TABLE = "table"
    EQUATION = "equation"


class ParseStatus(StrEnum):
    PARSING = "parsing"
    SUCCESS = "success"
    SUCCESS_WITH_WARNINGS = "success_with_warnings"
    FAILED = "failed"


class ElementParseStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


def _to_json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BBox):
        return value.as_list()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, _Serializable):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_to_json_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        )
    return value


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_metadata(item) for item in value)
    return value


class _Serializable:
    def to_dict(self) -> dict[str, Any]:
        return {item.name: _to_json_value(getattr(self, item.name)) for item in fields(self)}


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        for name in ("x0", "y0", "x1", "y1"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if not all(isfinite(value) for value in (self.x0, self.y0, self.x1, self.y1)):
            raise ValueError("coordinates must be finite")
        if self.x0 > self.x1:
            raise ValueError("x0 must not exceed x1")
        if self.y0 > self.y1:
            raise ValueError("y0 must not exceed y1")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    def to_dict(self) -> list[float]:
        return self.as_list()

    def union(self, other: BBox) -> BBox:
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def contains_within(self, page_width: float, page_height: float) -> bool:
        return (
            self.x0 >= 0.0
            and self.y0 >= 0.0
            and self.x1 <= float(page_width)
            and self.y1 <= float(page_height)
        )


def _normalize_vector(value: Any, name: str) -> tuple[float, float]:
    message = f"{name} must contain exactly two finite numeric values"
    try:
        normalized = tuple(float(component) for component in value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    if len(normalized) != 2 or not all(isfinite(component) for component in normalized):
        raise ValueError(message)
    return normalized[0], normalized[1]


@dataclass(frozen=True)
class RawSpan(_Serializable):
    text: str
    bbox: BBox
    font: str
    size: float
    flags: int = 0
    color: int | None = None
    origin: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.origin is not None:
            object.__setattr__(self, "origin", _normalize_vector(self.origin, "origin"))


@dataclass(frozen=True)
class RawLine(_Serializable):
    text: str
    bbox: BBox
    spans: tuple[RawSpan, ...] = ()
    direction: tuple[float, float] = (1.0, 0.0)
    wmode: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "spans", tuple(self.spans))
        object.__setattr__(
            self, "direction", _normalize_vector(self.direction, "direction")
        )


@dataclass(frozen=True)
class RawBlock(_Serializable):
    page_number: int
    block_index: int
    bbox: BBox
    text: str
    lines: tuple[RawLine, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "lines", tuple(self.lines))


@dataclass(frozen=True)
class ImageRegion(_Serializable):
    source_index: int
    bbox: BBox
    width: int | None = None
    height: int | None = None
    xref: int | None = None
    ext: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True)
class PageLayout(_Serializable):
    page_number: int
    width: float
    height: float
    blocks: tuple[RawBlock, ...] = ()
    image_regions: tuple[ImageRegion, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "image_regions", tuple(self.image_regions))


@dataclass(frozen=True)
class ClassifiedBlock(_Serializable):
    uid: str
    block_type: BlockType
    page_number: int
    bbox: BBox
    text: str
    confidence: float = 1.0
    source_block_indices: tuple[int, ...] = ()
    reason_codes: tuple[str, ...] = ()
    section: str | None = None
    subsection: str | None = None
    section_uid: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_block_indices", tuple(self.source_block_indices))
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")


@dataclass(frozen=True)
class LayoutDiagnosis(_Serializable):
    page_number: int
    column_count: int = 1
    column_bounds: tuple[BBox, ...] = ()
    reading_order: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    confidence: float = 0.0
    reason_codes: tuple[str, ...] = ()
    central_gap: BBox | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "column_bounds", tuple(self.column_bounds))
        object.__setattr__(self, "reading_order", tuple(self.reading_order))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        if (
            not isinstance(self.column_count, int)
            or isinstance(self.column_count, bool)
            or self.column_count not in {1, 2}
        ):
            raise ValueError("column_count must be 1 or 2")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")
        if self.column_count == 2:
            if len(self.column_bounds) != 2:
                raise ValueError("two-column layout requires two column bounds")
            left, right = self.column_bounds
            if not isinstance(left, BBox) or not isinstance(right, BBox):
                raise ValueError("column bounds must contain BBox values")
            if left.x1 > right.x0:
                raise ValueError("column bounds must be ordered and non-overlapping")
            if self.central_gap is not None and (
                self.central_gap.x0 < left.x1 or self.central_gap.x1 > right.x0
            ):
                raise ValueError("central gap must lie between column bounds")

    @property
    def layout_type(self) -> str:
        return "two_column" if self.column_count == 2 else "single_column"

    @property
    def gap(self) -> BBox | None:
        return self.central_gap


@dataclass
class SectionNode(_Serializable):
    uid: str
    title: str
    level: int
    parent_uid: str | None = None
    block_uids: list[str] = field(default_factory=list)
    children: list[SectionNode] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedElement(_Serializable):
    uid: str
    element_type: ElementType
    status: ElementParseStatus
    page_number: int
    bbox: BBox
    content: str = ""
    caption: str | None = None
    source_block_uids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    section: str | None = None
    subsection: str | None = None
    section_uid: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_block_uids", tuple(self.source_block_uids))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def parse_status(self) -> ElementParseStatus:
        return self.status

    @property
    def image_path(self) -> Path | None:
        value = self.metadata.get("image_path")
        return Path(value) if value else None

    @property
    def vision_description(self) -> str | None:
        value = self.metadata.get("vision_description")
        return str(value) if value else None

    @property
    def label(self) -> str | None:
        """Canonical element label (for example ``Table 3``).

        Element-specific parsers keep their richer payload in ``metadata`` so
        that the common parsed-element contract remains stable.  These small
        accessors make that payload convenient to consume without requiring
        callers to know the metadata key used by each parser.
        """

        for key in ("table_id", "figure_id", "equation_id", "label"):
            value = self.metadata.get(key)
            if value:
                return str(value)
        return None

    @property
    def structured_data(self) -> Mapping[str, Any]:
        value = self.metadata.get("structured_data", {})
        decoded = _to_json_value(value) if isinstance(value, Mapping) else {}
        return decoded if isinstance(decoded, Mapping) else {}

    @property
    def equation_id(self) -> str | None:
        value = self.metadata.get("equation_id")
        return str(value) if value else None

    @property
    def latex(self) -> str | None:
        value = self.metadata.get("latex")
        return str(value) if value else None

    @property
    def raw_expression(self) -> str:
        return str(self.metadata.get("raw_expression") or "")

    @property
    def surrounding_text(self) -> str:
        return str(self.metadata.get("surrounding_text") or "")


@dataclass(frozen=True)
class StructuredChunk(_Serializable):
    uid: str
    text: str
    page_numbers: tuple[int, ...]
    section_path: tuple[str, ...] = ()
    block_uids: tuple[str, ...] = ()
    element_uids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "page_numbers", tuple(self.page_numbers))
        object.__setattr__(self, "section_path", tuple(self.section_path))
        object.__setattr__(self, "block_uids", tuple(self.block_uids))
        object.__setattr__(self, "element_uids", tuple(self.element_uids))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def contextual_prefix(self) -> str:
        return str(self.metadata.get("contextual_prefix") or "")

    @property
    def embedding_text(self) -> str:
        return str(self.metadata.get("embedding_text") or self.text)

    @property
    def token_count(self) -> int:
        return int(self.metadata.get("token_count") or len(self.text.split()))

    @property
    def chunk_type(self) -> str:
        return str(self.metadata.get("chunk_type") or "text")


@dataclass(frozen=True)
class ParseWarning(_Serializable):
    code: str
    message: str
    page_number: int | None = None
    block_uid: str | None = None
    element_uid: str | None = None
    stage: str | None = None


@dataclass
class ParseReport(_Serializable):
    status: ParseStatus = ParseStatus.PARSING
    page_count: int = 0
    block_count: int = 0
    element_count: int = 0
    chunk_count: int = 0
    text_coverage: float | None = None
    warnings: list[ParseWarning] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["warning_count"] = len(self.warnings)
        payload["error_count"] = len(self.errors)
        return payload


@dataclass
class ParsedDocument(_Serializable):
    uid: str
    source_path: Path | str
    title: str = ""
    pages: list[PageLayout] = field(default_factory=list)
    blocks: list[ClassifiedBlock] = field(default_factory=list)
    layout_diagnoses: list[LayoutDiagnosis] = field(default_factory=list)
    sections: list[SectionNode] = field(default_factory=list)
    elements: list[ParsedElement] = field(default_factory=list)
    chunks: list[StructuredChunk] = field(default_factory=list)
    cross_references: list[Any] = field(default_factory=list)
    report: ParseReport = field(default_factory=ParseReport)
    metadata: dict[str, Any] = field(default_factory=dict)
