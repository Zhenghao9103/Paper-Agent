from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import fitz

from .domain import BBox, ImageRegion, PageLayout, RawBlock, RawLine, RawSpan

_OMIT_METADATA = object()


def _bbox(values: Any) -> BBox:
    return BBox(*values)


def _span(payload: Mapping[str, Any]) -> RawSpan:
    origin = payload.get("origin")
    return RawSpan(
        text=str(payload.get("text", "")),
        bbox=_bbox(payload["bbox"]),
        font=str(payload.get("font", "")),
        size=float(payload.get("size", 0.0)),
        flags=int(payload.get("flags", 0)),
        color=payload.get("color"),
        origin=tuple(origin) if origin is not None else None,
    )


def _line(payload: Mapping[str, Any]) -> RawLine:
    spans = tuple(_span(span) for span in payload.get("spans", ()))
    return RawLine(
        text="".join(span.text for span in spans),
        bbox=_bbox(payload["bbox"]),
        spans=spans,
        direction=tuple(payload.get("dir", (1.0, 0.0))),
        wmode=int(payload.get("wmode", 0)),
    )


def _text_block(
    payload: Mapping[str, Any], page_number: int, source_index: int
) -> RawBlock:
    lines = tuple(_line(line) for line in payload.get("lines", ()))
    return RawBlock(
        page_number=page_number,
        block_index=int(payload.get("number", source_index)),
        bbox=_bbox(payload["bbox"]),
        text="\n".join(line.text for line in lines),
        lines=lines,
    )


def _sanitize_metadata(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _OMIT_METADATA
    if isinstance(value, Mapping):
        sanitized = {}
        for key, item in value.items():
            if isinstance(key, (bytes, bytearray, memoryview)):
                continue
            clean_item = _sanitize_metadata(item)
            if clean_item is not _OMIT_METADATA:
                sanitized[key] = clean_item
        return sanitized
    if isinstance(value, (set, frozenset)):
        sanitized_items = []
        for item in value:
            clean_item = _sanitize_metadata(item)
            if clean_item is not _OMIT_METADATA:
                sanitized_items.append(clean_item)
        return type(value)(sanitized_items)
    if isinstance(value, Sequence) and not isinstance(value, str):
        sanitized_items = []
        for item in value:
            clean_item = _sanitize_metadata(item)
            if clean_item is not _OMIT_METADATA:
                sanitized_items.append(clean_item)
        return tuple(sanitized_items) if isinstance(value, tuple) else sanitized_items
    return value


def _image_region(payload: Mapping[str, Any], source_index: int) -> ImageRegion:
    excluded = {"type", "number", "bbox", "width", "height", "xref", "ext", "image"}
    metadata = _sanitize_metadata(
        {key: value for key, value in payload.items() if key not in excluded}
    )
    return ImageRegion(
        source_index=int(payload.get("number", source_index)),
        bbox=_bbox(payload["bbox"]),
        width=payload.get("width"),
        height=payload.get("height"),
        xref=payload.get("xref"),
        ext=payload.get("ext"),
        metadata=metadata,
    )


class TextExtractor:
    def extract(self, path: Path, paper_id: str) -> list[PageLayout]:
        document = fitz.open(path)
        try:
            if document.page_count == 0:
                raise ValueError("PDF contains no pages")

            pages: list[PageLayout] = []
            for page_number, page in enumerate(document, start=1):
                payload = page.get_text("dict", sort=False)
                blocks: list[RawBlock] = []
                image_regions: list[ImageRegion] = []
                for source_index, block in enumerate(payload.get("blocks", ())):
                    if block.get("type") == 0:
                        blocks.append(_text_block(block, page_number, source_index))
                    elif block.get("type") == 1:
                        image_regions.append(_image_region(block, source_index))
                pages.append(
                    PageLayout(
                        page_number=page_number,
                        width=float(payload["width"]),
                        height=float(payload["height"]),
                        blocks=tuple(blocks),
                        image_regions=tuple(image_regions),
                    )
                )
            return pages
        finally:
            document.close()

    def render_page(
        self,
        path: Path,
        page_number: int,
        output_path: Path,
        scale: float = 1.5,
    ) -> None:
        if not isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and greater than 0")

        document = fitz.open(path)
        temporary_path: Path | None = None
        try:
            if not 1 <= page_number <= document.page_count:
                raise ValueError(
                    f"page_number must be between 1 and {document.page_count}"
                )
            page = document.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                dir=output_path.parent,
                prefix=f".{output_path.stem}-",
                suffix=".tmp.png",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            pixmap.save(temporary_path)
            if temporary_path.stat().st_size == 0:
                raise OSError("Rendered PNG is empty")
            temporary_path.replace(output_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            document.close()
