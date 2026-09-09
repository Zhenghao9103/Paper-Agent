from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import fitz

from .domain import (
    BBox,
    ElementParseStatus,
    ElementType,
    PageLayout,
    ParsedElement,
)

_CAPTION_RE = re.compile(
    r"^\s*(?:table|tab\.?)\s*(?P<label>[A-Za-z]*\d+(?:\.\d+)*|[IVXLCDM]+)"
    r"(?=\s|[:.)-]|$)",
    re.IGNORECASE,
)


def _bbox(value: Any) -> BBox | None:
    if isinstance(value, BBox):
        return value
    try:
        return BBox(*value)
    except (TypeError, ValueError):
        return None


def _text_blocks(blocks: Sequence[Any]) -> list[tuple[str, BBox, int]]:
    result: list[tuple[str, BBox, int]] = []
    for index, block in enumerate(blocks):
        lines = getattr(block, "lines", ())
        if lines:
            for line in lines:
                line_box = _bbox(getattr(line, "bbox", None))
                line_text = str(getattr(line, "text", "")).strip()
                if line_box is not None and line_text:
                    result.append((line_text, line_box, index))
            continue
        raw_box = getattr(block, "bbox", None)
        if raw_box is None and isinstance(block, Mapping):
            raw_box = block.get("bbox")
        box = _bbox(raw_box)
        text = getattr(block, "text", None)
        if text is None and isinstance(block, Mapping):
            text = block.get("text", "")
        if box is not None and str(text or "").strip():
            result.append((str(text).strip(), box, index))
    return result


def _caption(
    blocks: Sequence[tuple[str, BBox, int]], candidate: BBox
) -> tuple[str, str, int] | None:
    matches: list[tuple[float, float, str, str, int]] = []
    for text, _box, index in blocks:
        match = _CAPTION_RE.match(" ".join(text.split()))
        if not match:
            continue
        horizontal = max(
            0.0,
            min(candidate.x1, _box.x1) - max(candidate.x0, _box.x0),
        ) / max(min(candidate.width, _box.width), 1.0)
        if horizontal < 0.15:
            continue
        if _box.y1 <= candidate.y0:
            gap = candidate.y0 - _box.y1
        elif _box.y0 >= candidate.y1:
            gap = _box.y0 - candidate.y1
        else:
            gap = 0.0
        if gap > 96.0:
            continue
        matches.append(
            (
                -gap,
                horizontal,
                text,
                f"Table {match.group('label')}",
                index,
            )
        )
    if not matches:
        return None
    _, _, text, label, index = max(matches, key=lambda item: (item[0], item[1], -item[4]))
    return text, label, index


def _line_bounds(
    drawings: Sequence[Mapping[str, Any]], candidate: BBox
) -> tuple[list[float], list[float]]:
    xs = {candidate.x0, candidate.x1}
    ys = {candidate.y0, candidate.y1}
    for drawing in drawings:
        box = _bbox(drawing.get("rect"))
        if box is None or box.x0 < candidate.x0 - 1 or box.x1 > candidate.x1 + 1:
            continue
        if box.y0 < candidate.y0 - 1 or box.y1 > candidate.y1 + 1:
            continue
        if box.width <= 1.5 and box.height >= candidate.height * 0.5:
            xs.add(round(box.x0, 3))
        elif box.height <= 1.5 and box.width >= candidate.width * 0.5:
            ys.add(round(box.y0, 3))
    return sorted(xs), sorted(ys)


def _markdown(headers: list[str], rows: list[list[str]]) -> str:
    def escape(value: str) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    width = max(1, len(headers))
    headers = (headers + [""] * width)[:width]
    lines = [
        "| " + " | ".join(escape(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(escape(value) for value in (row + [""] * width)[:width]) + " |"
        for row in rows
    )
    return "\n".join(lines)


class TableParser:
    def __init__(self, *, min_area_ratio: float = 0.01) -> None:
        if not 0 < min_area_ratio < 1:
            raise ValueError("min_area_ratio must be between 0 and 1")
        self.min_area_ratio = min_area_ratio

    @staticmethod
    def canonicalize_rows(
        headers: Sequence[str], rows: Sequence[Sequence[str]]
    ) -> tuple[list[str], list[dict[str, str]], list[str]]:
        columns: list[str] = []
        seen: dict[str, int] = {}
        for index, raw_header in enumerate(headers, start=1):
            base = str(raw_header or "").strip() or f"column_{index}"
            count = seen.get(base, 0) + 1
            seen[base] = count
            columns.append(base if count == 1 else f"{base}_{count}")
        normalized_rows: list[dict[str, str]] = []
        warnings: list[str] = []
        for raw_row in rows:
            values = [str(value or "").strip() for value in raw_row]
            if len(values) > len(columns) and "table_row_overflow" not in warnings:
                warnings.append("table_row_overflow")
            padded = (values + [""] * len(columns))[: len(columns)]
            normalized_rows.append(dict(zip(columns, padded, strict=True)))
        return columns, normalized_rows, warnings

    def parse(
        self,
        pdf_path: Path,
        page: PageLayout | Any,
        blocks: Sequence[Any],
        output_dir: Path,
    ) -> list[ParsedElement]:
        page_number = int(getattr(page, "page_number", 1))
        width = float(getattr(page, "width", 0.0))
        height = float(getattr(page, "height", 0.0))
        if width <= 0 or height <= 0:
            return []
        text_blocks = _text_blocks(blocks)
        document = fitz.open(pdf_path)
        try:
            pdf_page = document.load_page(page_number - 1)
            drawings = pdf_page.get_drawings()
            try:
                line_finder = pdf_page.find_tables(strategy="lines_strict")
                line_tables = getattr(line_finder, "tables", ())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                line_tables = ()
            if line_tables:
                clusters = [table.bbox for table in line_tables]
            else:
                clusters = pdf_page.cluster_drawings(drawings=drawings)
            results: list[ParsedElement] = []
            for ordinal, rect in enumerate(clusters, start=1):
                candidate = _bbox(rect)
                if candidate is None or candidate.area < width * height * self.min_area_ratio:
                    continue
                xs, ys = _line_bounds(drawings, candidate)
                if len(xs) < 3 or len(ys) < 2:
                    continue
                caption = _caption(text_blocks, candidate)
                label = caption[1] if caption else f"Table {ordinal}"
                data_blocks = [
                    (text, box)
                    for text, box, _index in text_blocks
                    if candidate.x0 <= box.x0 and box.x1 <= candidate.x1
                    and candidate.y0 <= box.y0 and box.y1 <= candidate.y1
                    and not _CAPTION_RE.match(" ".join(text.split()))
                ]
                headers, rows, reliable = self._cells(data_blocks, xs, ys)
                columns, row_objects, structure_warnings = self.canonicalize_rows(
                    headers, rows
                )
                reliable = reliable and not structure_warnings
                raw_text = "\n".join(text for text, _box in data_blocks)
                cell_bboxes = [
                    [
                        [xs[column], ys[row], xs[column + 1], ys[row + 1]]
                        for column in range(len(xs) - 1)
                    ]
                    for row in range(len(ys) - 1)
                ]
                structured = {
                    "headers": columns,
                    "columns": columns,
                    "rows": row_objects,
                    "raw_rows": rows,
                    "markdown_text": _markdown(columns, rows) if reliable else "",
                    "raw_text": raw_text,
                    "table_id": label,
                    "cell_bboxes": cell_bboxes,
                }
                image_path = output_dir / f"page{page_number}_{label.replace(' ', '_')}.png"
                self._crop(pdf_page, candidate, image_path, width, height)
                warnings = list(structure_warnings)
                if not reliable and "table_structure_partial" not in warnings:
                    warnings.append("table_structure_partial")
                status = ElementParseStatus.SUCCESS if reliable else ElementParseStatus.PARTIAL
                results.append(
                    ParsedElement(
                        uid=f"table-p{page_number}-{ordinal}",
                        element_type=ElementType.TABLE,
                        status=status,
                        page_number=page_number,
                        bbox=candidate,
                        content=structured["markdown_text"] or raw_text,
                        caption=caption[0] if caption else None,
                        metadata={
                            "label": label,
                            "table_id": label,
                            "structured_data": structured,
                            "image_path": str(image_path),
                            "table_image_path": str(image_path),
                            "warning_codes": warnings,
                        },
                    )
                )
            if results:
                return results
            return self._parse_text_tables(
                pdf_page, page_number, width, height, text_blocks, output_dir
            )
        finally:
            document.close()

    def _parse_text_tables(
        self,
        pdf_page: fitz.Page,
        page_number: int,
        width: float,
        height: float,
        text_blocks: Sequence[tuple[str, BBox, int]],
        output_dir: Path,
    ) -> list[ParsedElement]:
        anchored = self._parse_caption_anchored_pipe_tables(
            pdf_page, page_number, width, height, text_blocks, output_dir
        )
        if anchored:
            return anchored
        try:
            finder = pdf_page.find_tables(strategy="text")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return []
        results: list[ParsedElement] = []
        for ordinal, table in enumerate(getattr(finder, "tables", ()), start=1):
            candidate = _bbox(getattr(table, "bbox", None))
            if candidate is None or candidate.area < width * height * self.min_area_ratio:
                continue
            extracted = table.extract()
            rows = [
                [str(cell or "").strip() for cell in row]
                for row in extracted
                if any(str(cell or "").strip() for cell in row)
            ]
            if len(rows) < 2 or len(rows[0]) < 2:
                continue
            caption = _caption(text_blocks, candidate)
            label = caption[1] if caption else f"Table {ordinal}"
            headers, body = rows[0], rows[1:]
            reliable = all(
                len(row) == len(headers) and all(cell for cell in row)
                for row in rows
            )
            source_rows: list[list[tuple[str, BBox, int]]] = []
            for item in sorted(text_blocks, key=lambda value: (value[1].y0, value[1].x0)):
                if not (candidate.y0 <= item[1].y0 <= candidate.y1):
                    continue
                if source_rows and abs(item[1].y0 - source_rows[-1][0][1].y0) <= 12:
                    source_rows[-1].append(item)
                else:
                    source_rows.append([item])
            row_counts = [len(row) for row in source_rows if row]
            if row_counts and (
                len(set(row_counts)) > 1 or any(count != len(headers) for count in row_counts)
            ):
                reliable = False
            max_cell_length = max(
                (len(cell) for row in rows for cell in row),
                default=0,
            )
            high_confidence_text_table = (
                reliable
                and len(rows) >= 3
                and max_cell_length <= 40
            )
            if caption is None and not high_confidence_text_table:
                continue
            columns, row_objects, structure_warnings = self.canonicalize_rows(
                headers, body
            )
            reliable = reliable and not structure_warnings
            cell_bboxes = [list(cell) for cell in (getattr(table, "cells", ()) or ())]
            structured = {
                "headers": columns,
                "columns": columns,
                "rows": row_objects,
                "raw_rows": body,
                "markdown_text": _markdown(columns, body) if reliable else "",
                "raw_text": "\n".join(" ".join(row) for row in rows),
                "table_id": label,
                "cell_bboxes": cell_bboxes,
            }
            image_path = output_dir / f"page{page_number}_{label.replace(' ', '_')}.png"
            self._crop(pdf_page, candidate, image_path, width, height)
            results.append(
                ParsedElement(
                    uid=f"table-p{page_number}-text-{ordinal}",
                    element_type=ElementType.TABLE,
                    status=(
                        ElementParseStatus.SUCCESS
                        if reliable
                        else ElementParseStatus.PARTIAL
                    ),
                    page_number=page_number,
                    bbox=candidate,
                    content=structured["markdown_text"] or structured["raw_text"],
                    caption=caption[0] if caption else None,
                    metadata={
                        "label": label,
                        "table_id": label,
                        "structured_data": structured,
                        "image_path": str(image_path),
                        "table_image_path": str(image_path),
                        "warning_codes": (
                            structure_warnings
                            if reliable
                            else list(
                                dict.fromkeys(
                                    [*structure_warnings, "table_structure_partial"]
                                )
                            )
                        ),
                    },
                )
            )
        return results

    def _parse_caption_anchored_pipe_tables(
        self,
        pdf_page: fitz.Page,
        page_number: int,
        width: float,
        height: float,
        text_blocks: Sequence[tuple[str, BBox, int]],
        output_dir: Path,
    ) -> list[ParsedElement]:
        results: list[ParsedElement] = []
        captions = [
            (text, box, index, match)
            for text, box, index in text_blocks
            if (match := _CAPTION_RE.match(" ".join(text.split())))
        ]
        for ordinal, (caption_text, caption_box, _index, match) in enumerate(
            captions, start=1
        ):
            geometric = self._caption_geometry_table(
                pdf_page,
                page_number,
                width,
                height,
                text_blocks,
                output_dir,
                ordinal,
                caption_text,
                caption_box,
                match.group("label"),
            )
            if geometric is not None:
                results.append(geometric)
                continue
            same_column = [
                (text, box)
                for text, box, _source_index in text_blocks
                if box.y0 > caption_box.y1
                and box.y0 - caption_box.y1 <= 180
                and box.x0 < caption_box.x1 + width * 0.12
                and box.x1 > caption_box.x0 - width * 0.02
                and not _CAPTION_RE.match(" ".join(text.split()))
            ]
            same_column.sort(key=lambda item: (item[1].y0, item[1].x0))
            pipe_blocks: list[tuple[str, BBox]] = []
            expected_columns = 0
            for text, box in same_column:
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                if not lines or not any("|" in line for line in lines):
                    if pipe_blocks:
                        break
                    continue
                column_counts = [
                    len([cell for cell in line.split("|") if cell.strip()])
                    for line in lines
                    if "|" in line
                ]
                block_columns = max(column_counts, default=0)
                if block_columns < 2:
                    continue
                if expected_columns and block_columns != expected_columns:
                    break
                expected_columns = expected_columns or block_columns
                pipe_blocks.append((text, box))
            if len(pipe_blocks) < 2 or expected_columns < 2:
                continue
            logical_rows: list[list[str]] = []
            for text, _box in pipe_blocks:
                cells = [cell.strip() for cell in text.replace("\n", " | ").split("|")]
                cells = [cell for cell in cells if cell]
                for start in range(0, len(cells), expected_columns):
                    row = cells[start : start + expected_columns]
                    if len(row) == expected_columns:
                        logical_rows.append(row)
            if len(logical_rows) < 3:
                continue
            headers, body = logical_rows[0], logical_rows[1:]
            columns, row_objects, warnings = self.canonicalize_rows(headers, body)
            if warnings or any(column.startswith("column_") for column in columns):
                continue
            candidate = caption_box
            for _text, box in pipe_blocks:
                candidate = candidate.union(box)
            label = f"Table {match.group('label')}"
            structured = {
                "headers": columns,
                "columns": columns,
                "rows": row_objects,
                "raw_rows": body,
                "markdown_text": _markdown(columns, body),
                "raw_text": "\n".join(text for text, _box in pipe_blocks),
                "table_id": label,
                "cell_bboxes": [],
            }
            image_path = output_dir / f"page{page_number}_{label.replace(' ', '_')}.png"
            self._crop(pdf_page, candidate, image_path, width, height)
            results.append(
                ParsedElement(
                    uid=f"table-p{page_number}-anchored-{ordinal}",
                    element_type=ElementType.TABLE,
                    status=ElementParseStatus.SUCCESS,
                    page_number=page_number,
                    bbox=candidate,
                    content=structured["markdown_text"],
                    caption=caption_text,
                    metadata={
                        "label": label,
                        "table_id": label,
                        "structured_data": structured,
                        "image_path": str(image_path),
                        "table_image_path": str(image_path),
                        "warning_codes": [],
                    },
                )
            )
        return results

    def _caption_geometry_table(
        self,
        pdf_page: fitz.Page,
        page_number: int,
        width: float,
        height: float,
        text_blocks: Sequence[tuple[str, BBox, int]],
        output_dir: Path,
        ordinal: int,
        caption_text: str,
        caption_box: BBox,
        label_number: str,
    ) -> ParsedElement | None:
        entries = [
            (text, box)
            for text, box, _source_index in text_blocks
            if box.y0 > caption_box.y1
            and box.y0 - caption_box.y1 <= 180
            and not _CAPTION_RE.match(" ".join(text.split()))
        ]
        entries.sort(key=lambda item: (item[1].y0, item[1].x0))
        row_groups: list[list[tuple[str, BBox]]] = []
        for item in entries:
            if row_groups and abs(item[1].y0 - row_groups[-1][0][1].y0) <= 2.5:
                row_groups[-1].append(item)
            else:
                row_groups.append([item])
        header_index = next(
            (
                index
                for index, row in enumerate(row_groups)
                if 2 <= len(row) <= 10
                and all(0 < len(text) <= 40 for text, _box in row)
            ),
            None,
        )
        if header_index is None:
            return None
        header_items = sorted(row_groups[header_index], key=lambda item: item[1].x0)
        headers = [text for text, _box in header_items]
        right_limit = min(width, header_items[-1][1].x1 + width * 0.035)
        left_limit = max(0.0, header_items[0][1].x0 - width * 0.035)
        rows: list[list[str]] = []
        row_boxes: list[BBox] = []
        for group in row_groups[header_index + 1 :]:
            cells = sorted(
                [item for item in group if left_limit <= item[1].x0 and item[1].x1 <= right_limit],
                key=lambda item: item[1].x0,
            )
            if len(cells) != len(headers):
                if rows:
                    break
                continue
            if any(len(text) > 60 for text, _box in cells):
                break
            rows.append([text for text, _box in cells])
            row_box = cells[0][1]
            for _text, box in cells[1:]:
                row_box = row_box.union(box)
            row_boxes.append(row_box)
        if len(rows) < 2:
            return None
        columns, row_objects, warnings = self.canonicalize_rows(headers, rows)
        if warnings or any(column.startswith("column_") for column in columns):
            return None
        candidate = caption_box
        for _text, box in header_items:
            candidate = candidate.union(box)
        for box in row_boxes:
            candidate = candidate.union(box)
        label = f"Table {label_number}"
        structured = {
            "headers": columns,
            "columns": columns,
            "rows": row_objects,
            "raw_rows": rows,
            "markdown_text": _markdown(columns, rows),
            "raw_text": "\n".join(" | ".join(row) for row in [headers, *rows]),
            "table_id": label,
            "cell_bboxes": [],
        }
        image_path = output_dir / f"page{page_number}_{label.replace(' ', '_')}.png"
        self._crop(pdf_page, candidate, image_path, width, height)
        return ParsedElement(
            uid=f"table-p{page_number}-anchored-{ordinal}",
            element_type=ElementType.TABLE,
            status=ElementParseStatus.SUCCESS,
            page_number=page_number,
            bbox=candidate,
            content=structured["markdown_text"],
            caption=caption_text,
            metadata={
                "label": label,
                "table_id": label,
                "structured_data": structured,
                "image_path": str(image_path),
                "table_image_path": str(image_path),
                "warning_codes": [],
            },
        )

    @staticmethod
    def _cells(
        blocks: Sequence[tuple[str, BBox]], xs: Sequence[float], ys: Sequence[float]
    ) -> tuple[list[str], list[list[str]], bool]:
        columns = len(xs) - 1
        row_count = len(ys) - 1
        cells: list[list[str]] = [["" for _ in range(columns)] for _ in range(row_count)]
        for text, box in blocks:
            cx = (box.x0 + box.x1) / 2
            cy = (box.y0 + box.y1) / 2
            col = next((i for i in range(columns) if xs[i] <= cx <= xs[i + 1]), None)
            row = next((i for i in range(row_count) if ys[i] <= cy <= ys[i + 1]), None)
            if row is not None and col is not None:
                cells[row][col] = f"{cells[row][col]} {text}".strip()
        headers = cells[0] if cells else []
        rows = cells[1:] if len(cells) > 1 else []
        reliable = columns >= 2 and row_count >= 2 and all(value for row in cells for value in row)
        return headers, rows, reliable

    @staticmethod
    def _crop(page: fitz.Page, bbox: BBox, output: Path, width: float, height: float) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        rect = fitz.Rect(
            max(0, bbox.x0), max(0, bbox.y0), min(width, bbox.x1), min(height, bbox.y1)
        )
        temporary: Path | None = None
        try:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, alpha=False)
            with NamedTemporaryFile(
                dir=output.parent,
                prefix=f".{output.stem}-",
                suffix=".tmp.png",
                delete=False,
            ) as file:
                temporary = Path(file.name)
            pixmap.save(temporary)
            if temporary.stat().st_size <= 0:
                raise OSError("empty table crop")
            temporary.replace(output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
