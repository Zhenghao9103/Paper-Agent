from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
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

_LABEL_RE = re.compile(
    r"^\s*(?:(?:eq(?:uation)?\.?\s*)?\(([0-9]+)\)|"
    r"(?:eq(?:uation)?\.?\s+)([0-9]+))\s*$",
    re.IGNORECASE,
)
_MATH_RE = re.compile(r"(?:\\[A-Za-z]+|[=+\-*/^_]|[<>≤≥∑∫]|\d)")
_URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_PDF_INLINE_LABEL_RE = re.compile(r"(?:ð|\()?([0-9]{1,3})(?:Þ|\))\s*$")


def _bbox(value: Any) -> BBox | None:
    if isinstance(value, BBox):
        return value
    try:
        return BBox(*value)
    except (TypeError, ValueError):
        return None


def _value(block: Any, name: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _text(block: Any) -> str:
    return str(_value(block, "text", "") or "").strip()


def _label_number(match: re.Match[str]) -> str:
    return match.group(1) or match.group(2)


def _is_math_candidate(
    block: Any,
    *,
    page_width: float = 0.0,
    labels: Sequence[tuple[Any, BBox, str]] = (),
) -> bool:
    text = _text(block)
    if not text or len(text) > 180 or text.endswith((".", ",")):
        return False
    if _URL_RE.search(text):
        return False
    if len(text.split()) > 8:
        return False
    symbols = sum(text.count(char) for char in "=^_+-*/")
    if not bool(_MATH_RE.search(text)):
        return False
    if symbols >= 2:
        return True
    block_type = _value(_value(block, "block_type"), "value", _value(block, "block_type"))
    if block_type != "equation":
        return False
    box = _bbox(_value(block, "bbox"))
    centered = bool(
        box
        and page_width > 0
        and abs((box.x0 + box.x1) / 2 - page_width / 2) <= page_width * 0.15
    )
    labeled = bool(
        box
        and any(
            label_box.y0 >= box.y0 - max(box.height, label_box.height) * 1.5
            and abs(label_box.y0 - box.y0) <= max(box.height, label_box.height) * 1.5
            and label_box.x0 >= box.x1 - 5
            for _label_block, label_box, _number in labels
        )
    )
    return centered or labeled


@dataclass(frozen=True)
class _MergedBlock:
    text: str
    bbox: BBox
    section: str | None = None


class EquationParser:
    def __init__(self, recognizer: Any | None = None) -> None:
        self.recognizer = recognizer

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
        entries = [
            (block, _bbox(_value(block, "bbox")))
            for block in blocks
            if _bbox(_value(block, "bbox")) is not None
        ]
        labels = [
            (block, box, _label_number(match))
            for block, box in entries
            if box is not None and (match := _LABEL_RE.match(_text(block)))
        ]
        expressions = [
            (block, box)
            for block, box in entries
            if box is not None
            and _LABEL_RE.match(_text(block)) is None
            and _is_math_candidate(block, page_width=width, labels=labels)
        ]
        expressions = self._merge_expressions(expressions)
        document = fitz.open(pdf_path)
        try:
            pdf_page = document.load_page(page_number - 1)
            results: list[ParsedElement] = []
            used: set[int] = set()
            for ordinal, (expression, expression_box) in enumerate(expressions, start=1):
                if expression_box is None:
                    continue
                label_item = self._label_for(expression_box, labels, used)
                label_box = label_item[1] if label_item else None
                raw_text = _text(expression)
                inline_label = re.search(r"(?:^|\n)\s*\(?([0-9]+)\)?\s*$", raw_text)
                if inline_label is None:
                    inline_label = _PDF_INLINE_LABEL_RE.search(raw_text)
                if label_item is None and inline_label:
                    equation_id = f"Eq. {inline_label.group(1)}"
                    raw_expression = raw_text[: inline_label.start()].strip()
                else:
                    equation_id = f"Eq. {label_item[2]}" if label_item else None
                    raw_expression = raw_text
                bbox = expression_box.union(label_box) if label_box else expression_box
                if inline_label and label_box is None:
                    bbox = BBox(bbox.x0, bbox.y0, min(width, bbox.x1 + 60.0), bbox.y1)
                latex = (
                    raw_expression
                    if all(ord(char) < 128 for char in raw_expression) and "?" not in raw_expression
                    else None
                )
                surrounding = self._surrounding(entries, expression_box, expression)
                file_label = equation_id.replace(" ", "_") if equation_id else f"equation_{ordinal}"
                image_path = output_dir / f"page{page_number}_{file_label}.png"
                self._crop(pdf_page, bbox, image_path, width, height)
                recognition_path = output_dir / f"page{page_number}_{file_label}.mfr.png"
                self._crop(
                    pdf_page,
                    expression_box,
                    recognition_path,
                    width,
                    height,
                )
                recognition = None
                if self.recognizer is not None:
                    recognition = self.recognizer.recognize(recognition_path)
                    if recognition.latex:
                        latex = recognition.latex
                    else:
                        latex = None
                warnings = [] if latex else ["latex_unreliable"]
                if recognition is not None:
                    warnings.extend(recognition.warnings)
                status = ElementParseStatus.SUCCESS if latex else ElementParseStatus.PARTIAL
                structured_data = {
                    "equation_id": equation_id,
                    "latex": latex,
                    "raw_expression": raw_expression,
                    "surrounding_text": surrounding,
                    "recognition": recognition.to_dict() if recognition else None,
                    "recognition_image_path": str(recognition_path),
                }
                metadata = {
                    "equation_id": equation_id,
                    "latex": latex,
                    "raw_expression": raw_expression,
                    "surrounding_text": surrounding,
                    "structured_data": structured_data,
                    "recognition": recognition.to_dict() if recognition else None,
                    "image_path": str(image_path),
                    "recognition_image_path": str(recognition_path),
                    "warning_codes": warnings,
                }
                results.append(
                    ParsedElement(
                        uid=f"equation-p{page_number}-{ordinal}",
                        element_type=ElementType.EQUATION,
                        status=status,
                        page_number=page_number,
                        bbox=bbox,
                        content=latex or raw_expression,
                        source_block_uids=tuple(
                            str(_value(block, "uid"))
                            for block, block_box in entries
                            if _value(block, "uid")
                            and block_box is not None
                            and block_box.x0 >= expression_box.x0 - 1
                            and block_box.x1 <= expression_box.x1 + 1
                            and block_box.y0 >= expression_box.y0 - 1
                            and block_box.y1 <= expression_box.y1 + 1
                        ),
                        metadata=metadata,
                    )
                )
            return results
        finally:
            document.close()

    @staticmethod
    def _merge_expressions(
        expressions: Sequence[tuple[Any, BBox]]
    ) -> list[tuple[Any, BBox]]:
        merged: list[tuple[Any, BBox]] = []
        for block, box in sorted(expressions, key=lambda item: (item[1].y0, item[1].x0)):
            if merged:
                previous, previous_box = merged[-1]
                same_column = abs(previous_box.x0 - box.x0) <= 12
                close = 0 <= box.y0 - previous_box.y1 <= 12
                if same_column and close:
                    merged[-1] = (
                        _MergedBlock(
                            text=f"{_text(previous)}\n{_text(block)}",
                            bbox=previous_box.union(box),
                            section=_value(previous, "section"),
                        ),
                        previous_box.union(box),
                    )
                    continue
            merged.append((block, box))
        return merged

    @staticmethod
    def _label_for(
        expression_box: BBox,
        labels: Sequence[tuple[Any, BBox, str]],
        used: set[int],
    ) -> tuple[Any, BBox, str] | None:
        choices: list[tuple[float, int, tuple[Any, BBox, str]]] = []
        for index, item in enumerate(labels):
            if index in used:
                continue
            _block, box, _number = item
            if box.x0 < expression_box.x1 - 5:
                continue
            if abs(box.y0 - expression_box.y0) > max(expression_box.height, box.height) * 1.5:
                continue
            gap = max(0.0, box.x0 - expression_box.x1, expression_box.x0 - box.x1)
            if gap > 120:
                continue
            choices.append((gap, index, item))
        if not choices:
            return None
        _, index, item = min(choices, key=lambda value: (value[0], value[1]))
        used.add(index)
        return item

    @staticmethod
    def _surrounding(
        entries: Sequence[tuple[Any, BBox | None]], expression: BBox, expression_block: Any
    ) -> str:
        candidates: list[tuple[float, str]] = []
        for block, box in entries:
            if box is None or block is expression_block:
                continue
            text = _text(block)
            if not text or _LABEL_RE.match(text) or _is_math_candidate(block):
                continue
            expression_section = _value(expression_block, "section")
            block_section = _value(block, "section")
            if expression_section and block_section and expression_section != block_section:
                continue
            if box.y1 <= expression.y0:
                candidates.append((expression.y0 - box.y1, text))
            elif box.y0 >= expression.y1:
                candidates.append((box.y0 - expression.y1, text))
        before = [item for item in candidates if item[0] <= 140 and item[1]]
        return " ".join(text for _distance, text in sorted(before, key=lambda item: item[0])[:2])

    @staticmethod
    def _crop(page: fitz.Page, bbox: BBox, output: Path, width: float, height: float) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        rect = fitz.Rect(
            max(0, bbox.x0), max(0, bbox.y0), min(width, bbox.x1), min(height, bbox.y1)
        )
        if rect.width <= 0 or rect.height <= 0:
            raise ValueError("equation bbox is outside page")
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
            temporary.replace(output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
