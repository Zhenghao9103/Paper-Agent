"""MinerU-first PDF parsing mapped onto the application's stable domain model."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import fitz
from bs4 import BeautifulSoup

from .chunking import StructureAwareChunker
from .domain import (
    BBox,
    BlockType,
    ClassifiedBlock,
    ElementParseStatus,
    ElementType,
    PageLayout,
    ParsedDocument,
    ParsedElement,
    ParseReport,
    RawBlock,
)
from .quality import ParseQualityChecker

_IGNORED_TYPES = {"page_header", "page_footer", "page_footnote", "page_number"}
_FIGURE_TYPES = {"image", "chart"}
_TABLE_LABEL = re.compile(
    r"\bTable\s+(?:[A-Z]?\d+(?:[.-]\d+)*|[IVXLCDM]+)(?=\D|$)", re.IGNORECASE
)


def _default_run_mineru(pdf_path: Path, output_dir: Path) -> None:
    from ..core.config import get_settings

    root = Path(get_settings().mineru_root).resolve()
    config_path = root / "mineru.json"
    if not config_path.is_file():
        raise RuntimeError(f"MinerU config is missing: {config_path}")
    # The application setting must govern MinerU's internal MFR stage too;
    # without this the subprocess always runs formula recognition (minutes
    # per math-heavy paper on CPU) even when formulas are disabled.
    environment = {
        **os.environ,
        "MINERU_FORMULA_ENABLED": str(
            bool(get_settings().mineru_formula_enabled)
        ).lower(),
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "backend.app.cli.run_mineru_pipeline",
            "--pdf",
            str(pdf_path.resolve()),
            "--output",
            str(output_dir.resolve()),
            "--config",
            str(config_path),
        ],
        check=True,
        env=environment,
    )


def _parts_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        kind = str(value.get("type") or "")
        content = value.get("content")
        if kind in {"equation_inline", "equation", "interline_equation"}:
            return f"${str(content or '').strip()}$"
        return _parts_text(content)
    if isinstance(value, Sequence):
        return "".join(_parts_text(item) for item in value)
    return ""


def _entry_text(entry: Mapping[str, Any]) -> str:
    content = entry.get("content")
    if not isinstance(content, Mapping):
        return ""
    preferred = (
        "title_content",
        "paragraph_content",
        "page_header_content",
        "page_footer_content",
        "page_footnote_content",
        "page_number_content",
    )
    for key in preferred:
        if key in content:
            return _parts_text(content[key]).strip()
    items = content.get("list_items")
    if isinstance(items, Sequence):
        values = [
            _parts_text(item.get("item_content"))
            for item in items
            if isinstance(item, Mapping)
        ]
        return "\n".join(value for value in values if value)
    return ""


def _caption(content: Mapping[str, Any], kind: str) -> str | None:
    keys = ("table_caption",) if kind == "table" else ("image_caption", "chart_caption")
    for key in keys:
        value = _parts_text(content.get(key))
        if value:
            return value
    return None


def _scaled_bbox(value: Any, width: float, height: float) -> BBox:
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return BBox(0, 0, width, height)
    # MinerU pipeline v2 uses a normalized 1000 x 1000 page coordinate space.
    return BBox(
        max(0.0, min(width, x0 * width / 1000.0)),
        max(0.0, min(height, y0 * height / 1000.0)),
        max(0.0, min(width, x1 * width / 1000.0)),
        max(0.0, min(height, y1 * height / 1000.0)),
    )


def _table_data(html: str) -> dict[str, Any] | None:
    soup = BeautifulSoup(html or "", "html.parser")
    table = soup.find("table")
    if table is None:
        return None

    rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
    grid: list[list[str]] = []
    pending: dict[int, tuple[str, int]] = {}
    for row in rows:
        values: dict[int, str] = {
            column: value for column, (value, _remaining) in pending.items()
        }
        pending = {
            column: (value, remaining - 1)
            for column, (value, remaining) in pending.items()
            if remaining > 1
        }
        column = 0
        cells = row.find_all(["th", "td"], recursive=False)
        for cell in cells:
            while column in values:
                column += 1
            value = cell.get_text(" ", strip=True)
            try:
                colspan = max(1, int(cell.get("colspan", 1)))
                rowspan = max(1, int(cell.get("rowspan", 1)))
            except (TypeError, ValueError):
                colspan = rowspan = 1
            for offset in range(colspan):
                target = column + offset
                values[target] = value
                if rowspan > 1:
                    pending[target] = (value, rowspan - 1)
            column += colspan
        if values:
            grid.append([values.get(index, "") for index in range(max(values) + 1)])

    if len(grid) < 2:
        return None
    width = max(len(row) for row in grid)
    if width < 2:
        return None
    grid = [row + [""] * (width - len(row)) for row in grid]

    columns: list[str] = []
    seen: dict[str, int] = {}
    for index, value in enumerate(grid[0], start=1):
        base = value.strip() or f"column_{index}"
        seen[base] = seen.get(base, 0) + 1
        columns.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    body = [row for row in grid[1:] if any(value.strip() for value in row)]
    if not body:
        return None
    return {
        "headers": columns,
        "columns": columns,
        "rows": [dict(zip(columns, row, strict=True)) for row in body],
        "raw_rows": body,
        "raw_text": "\n".join(" | ".join(row) for row in grid),
    }


class MinerUPrimaryParser:
    def __init__(self, run_mineru: Callable[[Path, Path], None] | None = None) -> None:
        self.run_mineru = run_mineru or _default_run_mineru

    def parse(
        self,
        pdf_path: Path,
        *,
        paper_id: str,
        title: str,
        output_dir: Path,
    ) -> ParsedDocument:
        total_started = perf_counter()
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mineru_started = perf_counter()
        self.run_mineru(pdf_path, output_dir)
        mineru_ms = round((perf_counter() - mineru_started) * 1000)
        candidates = sorted(output_dir.rglob("*_content_list_v2.json"))
        if not candidates:
            raise RuntimeError("MinerU did not produce content_list_v2.json")
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise RuntimeError("MinerU content_list_v2.json is empty or invalid")
        parse_dir = candidates[0].parent
        document = fitz.open(pdf_path)
        try:
            pages = [
                PageLayout(index + 1, float(page.rect.width), float(page.rect.height))
                for index, page in enumerate(document)
            ]
            if len(payload) != len(pages):
                raise RuntimeError(
                    f"MinerU page count mismatch: output={len(payload)}, pdf={len(pages)}"
                )
            for index, page in enumerate(document, start=1):
                page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(
                    output_dir / f"page-{index}.png"
                )
        finally:
            document.close()
        conversion_started = perf_counter()
        blocks: list[ClassifiedBlock] = []
        elements: list[ParsedElement] = []
        current_section: str | None = None
        table_ordinal = 0
        for page_index, entries in enumerate(payload):
            if page_index >= len(pages) or not isinstance(entries, list):
                continue
            page = pages[page_index]
            for index, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    continue
                kind = str(entry.get("type") or "")
                if kind in _IGNORED_TYPES:
                    continue
                bbox = _scaled_bbox(entry.get("bbox"), page.width, page.height)
                uid = f"mineru-p{page_index + 1}-{index}"
                if kind == "title":
                    text = _entry_text(entry)
                    if text:
                        current_section = text
                        blocks.append(
                            ClassifiedBlock(
                                uid=uid,
                                block_type=BlockType.HEADING,
                                page_number=page_index + 1,
                                bbox=bbox,
                                text=text,
                                source_block_indices=(index,),
                                reason_codes=("mineru",),
                                section=text,
                                section_uid=f"section-{len(blocks) + 1}",
                            )
                        )
                    continue
                if kind in {"paragraph", "list", "index"}:
                    text = _entry_text(entry)
                    if text:
                        blocks.append(
                            ClassifiedBlock(
                                uid=uid,
                                block_type=(
                                    BlockType.LIST
                                    if kind in {"list", "index"}
                                    else BlockType.PARAGRAPH
                                ),
                                page_number=page_index + 1,
                                bbox=bbox,
                                text=text,
                                source_block_indices=(index,),
                                reason_codes=("mineru",),
                                section=current_section,
                            )
                        )
                    continue
                if kind == "table":
                    content = (
                        entry.get("content")
                        if isinstance(entry.get("content"), Mapping)
                        else {}
                    )
                    structured = _table_data(str(content.get("html") or ""))
                    if structured is None:
                        continue
                    table_ordinal += 1
                    caption = _caption(content, kind)
                    label_match = _TABLE_LABEL.search(caption or "")
                    label = label_match.group(0) if label_match else f"Table {table_ordinal}"
                    structured["table_id"] = label
                    asset = self._asset(parse_dir, output_dir, content.get("image_source"), uid)
                    elements.append(
                        ParsedElement(
                            uid=uid,
                            element_type=ElementType.TABLE,
                            status=ElementParseStatus.SUCCESS,
                            page_number=page_index + 1,
                            bbox=bbox,
                            content=structured["raw_text"],
                            caption=caption,
                            metadata={
                                "table_id": label,
                                "structured_data": structured,
                                "image_path": str(asset) if asset else None,
                                "parser": "mineru",
                            },
                            section=current_section,
                        )
                    )
                    continue
                if kind in {"equation_interline", "interline_equation", "equation"}:
                    content = (
                        entry.get("content")
                        if isinstance(entry.get("content"), Mapping)
                        else {}
                    )
                    latex = str(content.get("math_content") or "").strip()
                    if not latex:
                        continue
                    asset = self._asset(parse_dir, output_dir, content.get("image_source"), uid)
                    elements.append(
                        ParsedElement(
                            uid=uid,
                            element_type=ElementType.EQUATION,
                            status=ElementParseStatus.SUCCESS,
                            page_number=page_index + 1,
                            bbox=bbox,
                            content=latex,
                            metadata={
                                "latex": latex,
                                "raw_expression": latex,
                                "structured_data": {"latex": latex, "raw_expression": latex},
                                "image_path": str(asset) if asset else None,
                                "parser": "mineru",
                            },
                            section=current_section,
                        )
                    )
                    continue
                if kind in _FIGURE_TYPES:
                    content = (
                        entry.get("content")
                        if isinstance(entry.get("content"), Mapping)
                        else {}
                    )
                    caption = _caption(content, kind)
                    if not caption:
                        continue
                    asset = self._asset(parse_dir, output_dir, content.get("image_source"), uid)
                    elements.append(
                        ParsedElement(
                            uid=uid,
                            element_type=ElementType.FIGURE,
                            status=ElementParseStatus.SUCCESS,
                            page_number=page_index + 1,
                            bbox=bbox,
                            content=caption,
                            caption=caption,
                            metadata={
                                "structured_data": {"caption": caption, "figure_type": kind},
                                "image_path": str(asset) if asset else None,
                                "parser": "mineru",
                            },
                            section=current_section,
                        )
                    )
        page_blocks: dict[int, list[RawBlock]] = {page.page_number: [] for page in pages}
        for block in blocks:
            page_blocks[block.page_number].append(
                RawBlock(
                    page_number=block.page_number,
                    block_index=block.source_block_indices[0] if block.source_block_indices else 0,
                    bbox=block.bbox,
                    text=block.text,
                )
            )
        pages = [
            PageLayout(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                blocks=tuple(page_blocks[page.page_number]),
            )
            for page in pages
        ]
        conversion_ms = round((perf_counter() - conversion_started) * 1000)
        chunk_started = perf_counter()
        chunks = StructureAwareChunker().chunk(blocks, title=title, elements=elements)
        chunking_ms = round((perf_counter() - chunk_started) * 1000)
        total_ms = round((perf_counter() - total_started) * 1000)
        timings = {
            "extract_render": mineru_ms,
            "layout_structure": conversion_ms,
            "table_parse": conversion_ms,
            "chunking": chunking_ms,
            "total": total_ms,
        }
        parsed = ParsedDocument(
            uid=paper_id,
            source_path=pdf_path,
            title=title,
            pages=pages,
            blocks=blocks,
            elements=elements,
            chunks=chunks,
            report=ParseReport(),
            metadata={
                "parser": "mineru-primary",
                "content_list": str(candidates[0]),
                "timings_ms": timings,
            },
        )
        ParseQualityChecker().check(parsed)
        if parsed.report.status.value == "failed":
            raise RuntimeError("MinerU output failed application quality validation")
        (output_dir / "manifest.json").write_text(
            json.dumps(parsed.report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return parsed

    @staticmethod
    def _asset(
        parse_dir: Path,
        output_dir: Path,
        source: Any,
        uid: str,
    ) -> Path | None:
        if not isinstance(source, Mapping) or not source.get("path"):
            return None
        path = (parse_dir / str(source["path"])).resolve()
        if not path.is_file() or not path.is_relative_to(parse_dir.resolve()):
            return None
        target = output_dir / "elements" / f"{uid}{path.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        return target
