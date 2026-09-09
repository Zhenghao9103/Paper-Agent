from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .domain import BBox, ClassifiedBlock, ParsedElement, StructuredChunk


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _token_count(text: str) -> int:
    """Count tokens without making ingestion depend on a network cache.

    tiktoken is the preferred counter when its local encoding data is
    available.  Some deployments intentionally run offline, so the first
    failed lookup is memoized and the deterministic whitespace fallback is
    used instead of retrying a network request for every block.
    """
    counter = getattr(_token_count, "_counter", None)
    if counter is None:
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")

            def counter(value: str) -> int:
                return len(encoding.encode(value))

        except Exception:
            def counter(value: str) -> int:
                return len(value.split())

        _token_count._counter = counter  # type: ignore[attr-defined]
    return int(counter(text))


def _sentences(text: str) -> list[str]:
    parts = [
        part.strip()
        for part in re.split(
            r"(?<=[.!?])(?=\s+)|(?<=[。！？])(?=[^\s])", text.strip()
        )
        if part.strip()
    ]
    merged: list[str] = []
    abbreviations = {
        "e.g.",
        "i.e.",
        "etc.",
        "et al.",
        "fig.",
        "eq.",
        "sec.",
        "no.",
        "ref.",
        "tab.",
        "cf.",
    }
    for part in parts:
        if merged and merged[-1].casefold() in abbreviations:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def _normalize_pdf_text(text: str) -> str:
    protected = re.sub(
        r"\b([A-Za-z])-(?:\r?\n)(nearest|means|fold|value|space|NN)\b",
        r"\1-\2",
        text,
        flags=re.IGNORECASE,
    )
    dehyphenated = re.sub(r"(?<=[A-Za-z])-(?:\r?\n)(?=[a-z])", "", protected)
    return re.sub(r"[ \t]*\r?\n[ \t]*", " ", dehyphenated).strip()


def _element_metadata(element: Any) -> dict[str, Any]:
    metadata = _value(element, "metadata", {})
    return dict(metadata) if hasattr(metadata, "items") else {}


def _figure_text(element: Any, structured: dict[str, Any]) -> str:
    lines: list[str] = []
    caption = str(_value(element, "caption", "") or structured.get("caption") or "").strip()
    if caption:
        lines.append(caption)
    lines.append(f"Figure type: {structured.get('figure_type') or 'other'}")
    for label, key in (("Components", "components"), ("Relationships", "relationships")):
        values = structured.get(key)
        if isinstance(values, (list, tuple)) and values:
            lines.append(f"{label}: {'; '.join(str(item) for item in values[:20])}")
    summary = str(structured.get("summary") or "").strip()
    if summary:
        lines.append(f"Summary: {summary}")
    visible = structured.get("visible_text")
    if isinstance(visible, (list, tuple)) and visible:
        lines.append(f"Visible text: {'; '.join(str(item) for item in visible[:20])}")
    return "\n".join(lines)


def _table_text(element: Any, structured: dict[str, Any]) -> str:
    lines: list[str] = []
    caption = str(_value(element, "caption", "") or "").strip()
    if caption:
        lines.append(caption)
    columns = structured.get("columns") or structured.get("headers") or []
    if isinstance(columns, (list, tuple)) and columns:
        normalized_columns = [str(column) for column in columns]
        lines.append(f"Columns: {'; '.join(normalized_columns)}")
        rows = structured.get("rows") or []
        for row in rows[:20] if isinstance(rows, (list, tuple)) else []:
            if isinstance(row, Mapping):
                lines.append(
                    " | ".join(
                        f"{column}: {str(row.get(column, '')).strip()}"
                        for column in normalized_columns
                    )
                )
    return "\n".join(lines)


def _equation_text(metadata: dict[str, Any]) -> str:
    lines: list[str] = []
    equation_id = str(metadata.get("equation_id") or "").strip()
    latex = str(metadata.get("latex") or "").strip()
    surrounding = str(metadata.get("surrounding_text") or "").strip()
    if equation_id:
        lines.append(f"Equation: {equation_id}")
    if latex:
        lines.append(f"LaTeX: {latex}")
    if surrounding:
        lines.append(f"Context: {surrounding}")
    return "\n".join(lines)


def _element_text(element: Any) -> tuple[str, dict[str, Any]]:
    metadata = _element_metadata(element)
    structured_value = metadata.get("structured_data")
    structured = dict(structured_value) if hasattr(structured_value, "items") else {}
    kind = str(getattr(_value(element, "element_type"), "value", _value(element, "element_type")))
    if kind == "figure" and structured:
        text = _figure_text(element, structured)
    elif kind == "table" and structured:
        text = _table_text(element, structured)
    elif kind == "equation":
        text = _equation_text(metadata)
    else:
        text = str(_value(element, "content", "") or "").strip()
    if not text:
        text = str(_value(element, "caption", "") or kind or "element").strip()
    return text, metadata


class Contextualizer:
    def __init__(
        self,
        generate: Callable[[str, str | None, str | None, str], str] | None = None,
    ) -> None:
        self.generate = generate

    def prefix(
        self,
        title: str,
        section: str | None,
        subsection: str | None,
        text: str,
    ) -> str:
        if self.generate is not None:
            try:
                value = self.generate(title, section, subsection, text).strip()
                if value and self._is_grounded(value, title, section, subsection, text):
                    return value[:320]
            except Exception:
                try:
                    value = self.generate(text).strip()  # type: ignore[call-arg]
                    if value and self._is_grounded(value, title, section, subsection, text):
                        return value[:320]
                except Exception:
                    pass
        location = subsection or section
        if location:
            return f"This passage comes from the {location} section of {title or 'the paper'}."
        return f"This passage comes from {title or 'the paper'}."

    @staticmethod
    def _is_grounded(
        prefix: str,
        title: str,
        section: str | None,
        subsection: str | None,
        text: str,
    ) -> bool:
        source = " ".join(value or "" for value in (title, section, subsection, text))
        source_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", source))
        prefix_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", prefix))
        if not prefix_numbers <= source_numbers:
            return False
        if len(re.findall(r"[.!?](?:\s|$)", prefix)) > 2:
            return False
        source_words = set(re.findall(r"[A-Za-z]{4,}", source.casefold()))
        prefix_words = set(re.findall(r"[A-Za-z]{4,}", prefix.casefold()))
        return bool(prefix_words & source_words)


class StructureAwareChunker:
    def __init__(
        self,
        *,
        target_tokens: int = 350,
        min_tokens: int = 100,
        max_tokens: int = 800,
        overlap_ratio: float = 0.12,
        contextualizer: Contextualizer | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if not 0 < min_tokens <= target_tokens <= max_tokens:
            raise ValueError("expected min_tokens <= target_tokens <= max_tokens")
        if not 0 <= overlap_ratio < 0.5:
            raise ValueError("overlap_ratio must be between 0 and 0.5")
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_ratio = overlap_ratio
        self.contextualizer = contextualizer or Contextualizer()
        self.token_counter = token_counter or _token_count

    def chunk(
        self,
        blocks: Sequence[ClassifiedBlock | Any],
        *,
        title: str = "",
        elements: Sequence[ParsedElement | Any] = (),
    ) -> list[StructuredChunk]:
        allowed_types = {"paragraph", "list", "abstract", "reference", "unknown"}
        excluded_source_uids = {
            str(uid)
            for element in elements
            if str(
                getattr(
                    _value(element, "element_type"),
                    "value",
                    _value(element, "element_type"),
                )
            )
            == "equation"
            and str(
                getattr(_value(element, "status"), "value", _value(element, "status", ""))
            )
            == "success"
            for uid in (_value(element, "source_block_uids", ()) or ())
        }
        ordered = [
            block
            for block in blocks
            if str(_value(block, "text", "")).strip()
            and str(_value(block, "uid", "")) not in excluded_source_uids
            and str(getattr(_value(block, "block_type"), "value", _value(block, "block_type")))
            in allowed_types
        ]
        grouped: list[tuple[tuple[str | None, str | None, str | None], list[Any]]] = []
        for block in ordered:
            key = (
                _value(block, "section_uid"),
                _value(block, "section"),
                _value(block, "subsection"),
            )
            block_kind = str(
                getattr(_value(block, "block_type"), "value", _value(block, "block_type"))
            )
            previous = grouped[-1][1][-1] if grouped else None
            previous_kind = str(
                getattr(
                    _value(previous, "block_type"),
                    "value",
                    _value(previous, "block_type"),
                )
            )
            references_are_separate = "reference" in {block_kind, previous_kind}
            if grouped and grouped[-1][0] == key and not references_are_separate:
                grouped[-1][1].append(block)
            else:
                grouped.append((key, [block]))
        chunks: list[StructuredChunk] = []
        for group_index, (key, group) in enumerate(grouped, start=1):
            chunks.extend(self._chunk_group(group, key, title, group_index))
        for element_index, element in enumerate(elements, start=1):
            text, element_metadata = _element_text(element)
            kind = str(
                getattr(_value(element, "element_type"), "value", _value(element, "element_type"))
            )
            structured = element_metadata.get("structured_data") or {}
            status = str(
                getattr(_value(element, "status"), "value", _value(element, "status", ""))
            )
            if kind == "table" and (
                status != "success"
                or not isinstance(structured, Mapping)
                or not any(
                    any(str(value).strip() for value in row.values())
                    for row in structured.get("rows", ())
                    if isinstance(row, Mapping)
                )
            ):
                continue
            if kind == "figure" and not (
                str(_value(element, "caption", "") or "").strip()
                or str(element_metadata.get("vision_description") or "").strip()
                or element_metadata.get("structured_data")
            ):
                continue
            section = _value(element, "section")
            subsection = _value(element, "subsection")
            page_number = _value(element, "page_number")
            source_block_uids = tuple(
                str(uid)
                for uid in (_value(element, "source_block_uids", ()) or ())
                if uid
            )
            element_uid = _value(element, "uid")
            element_bbox = _value(element, "bbox")
            page_bboxes = {}
            if isinstance(element_bbox, BBox) and page_number is not None:
                page_bboxes[str(int(page_number))] = element_bbox.as_list()
            prefix = self.contextualizer.prefix(title, section, subsection, text)
            metadata = {
                "chunk_type": str(_value(_value(element, "element_type"), "value", "element")),
                "page_numbers": [page_number],
                "section_uid": _value(element, "section_uid"),
                "contextual_prefix": prefix,
                "embedding_text": self._embedding_text(title, section, subsection, prefix, text),
                "token_count": self.token_counter(text),
                "element_uid": element_uid,
                "element_uids": (element_uid,) if element_uid else (),
                "source_block_uids": source_block_uids,
                "block_uids": source_block_uids,
                "bbox": element_bbox.as_list() if isinstance(element_bbox, BBox) else None,
                "page_bboxes": page_bboxes,
                "image_path": (
                    str(element_metadata.get("image_path"))
                    if element_metadata.get("image_path")
                    else None
                ),
                "structured_data": element_metadata.get("structured_data", {}),
                "parse_status": str(
                    getattr(_value(element, "status"), "value", _value(element, "status", ""))
                ),
                "warning_codes": element_metadata.get("warning_codes", ()),
            }
            chunks.append(
                StructuredChunk(
                    uid=f"chunk-element-{element_index}",
                    text=text,
                    page_numbers=(page_number,),
                    section_path=tuple(value for value in (section, subsection) if value),
                    block_uids=source_block_uids,
                    element_uids=(element_uid,) if element_uid else (),
                    metadata=metadata,
                )
            )
        return chunks

    def _chunk_group(
        self,
        blocks: Sequence[Any],
        key: tuple[str | None, str | None, str | None],
        title: str,
        group_index: int,
    ) -> list[StructuredChunk]:
        section_uid, section, subsection = key
        result: list[StructuredChunk] = []
        current: list[Any] = []
        current_tokens = 0
        overlap_only = False
        chunk_index = 0
        for block in blocks:
            text = _normalize_pdf_text(str(_value(block, "text", "")))
            count = self.token_counter(text)
            if count > self.max_tokens:
                if current:
                    result.append(self._make_chunk(current, key, title, group_index, chunk_index))
                    chunk_index += 1
                    current = []
                    current_tokens = 0
                words = text.split()
                sentences = _sentences(text)
                if len(sentences) > 1:
                    start_sentence = 0
                    while start_sentence < len(sentences):
                        end_sentence = start_sentence
                        sentence_text = ""
                        while end_sentence < len(sentences):
                            candidate = " ".join(
                                sentences[start_sentence : end_sentence + 1]
                            )
                            if (
                                self.token_counter(candidate) > self.max_tokens
                                and end_sentence > start_sentence
                            ):
                                break
                            sentence_text = candidate
                            end_sentence += 1
                        if end_sentence == start_sentence:
                            sentence_text = sentences[start_sentence]
                            end_sentence += 1
                        result.append(
                            self._make_chunk(
                                [(block, sentence_text)],
                                key,
                                title,
                                group_index,
                                chunk_index,
                            )
                        )
                        chunk_index += 1
                        if end_sentence >= len(sentences):
                            break
                        overlap_limit = max(
                            1, int(self.max_tokens * self.overlap_ratio)
                        )
                        overlap_count = 0
                        overlap_size = 0
                        for sentence in reversed(
                            sentences[start_sentence:end_sentence]
                        ):
                            sentence_count = self.token_counter(sentence)
                            if overlap_size and overlap_count + sentence_count > overlap_limit:
                                break
                            overlap_count += sentence_count
                            overlap_size += 1
                        start_sentence = max(
                            end_sentence - overlap_size, start_sentence + 1
                        )
                    continue
                start = 0
                while start < len(words):
                    end = start
                    while end < len(words):
                        candidate = " ".join(words[start : end + 1])
                        if self.token_counter(candidate) > self.max_tokens and end > start:
                            break
                        end += 1
                    if end == start:
                        end += 1
                    if end < len(words):
                        punctuation = max(
                            (
                                index
                                for index in range(start, end)
                                if words[index].endswith((".", "!", "?"))
                            ),
                            default=start,
                        )
                        if punctuation > start + self.min_tokens // 2:
                            end = punctuation + 1
                    piece = " ".join(words[start:end])
                    result.append(
                        self._make_chunk(
                            [(block, piece)], key, title, group_index, chunk_index
                        )
                    )
                    chunk_index += 1
                    if end >= len(words):
                        break
                    overlap_words = max(1, int((end - start) * self.overlap_ratio))
                    start = max(end - overlap_words, start + 1)
                continue
            if current and current_tokens + count > self.target_tokens:
                result.append(self._make_chunk(current, key, title, group_index, chunk_index))
                chunk_index += 1
                current = self._overlap_items(current)
                current_tokens = sum(self.token_counter(text) for _, text in current)
                overlap_only = bool(current)
            if overlap_only and current_tokens + count > self.max_tokens:
                current = []
                current_tokens = 0
                overlap_only = False
            current.append((block, text))
            current_tokens += count
            overlap_only = False
        if current:
            result.append(self._make_chunk(current, key, title, group_index, chunk_index))
        return result

    def _overlap_items(self, items: Sequence[Any]) -> list[tuple[Any, str]]:
        """Return trailing complete sentence text for same-section overlap."""
        if not items or self.overlap_ratio <= 0:
            return []
        block, text = items[-1] if isinstance(items[-1], tuple) else (
            items[-1],
            str(_value(items[-1], "text", "")).strip(),
        )
        sentences = _sentences(text)
        if not sentences:
            return []
        limit = max(1, int(self.target_tokens * self.overlap_ratio))
        selected: list[str] = []
        total = 0
        for sentence in reversed(sentences):
            count = self.token_counter(sentence)
            if selected and total + count > limit:
                break
            selected.insert(0, sentence)
            total += count
        return [(block, " ".join(selected))] if selected else []

    def _make_chunk(
        self,
        items: Sequence[Any],
        key: tuple[str | None, str | None, str | None],
        title: str,
        group_index: int,
        chunk_index: int,
    ) -> StructuredChunk:
        section_uid, section, subsection = key
        texts: list[str] = []
        block_uids: list[str] = []
        pages: list[int] = []
        boxes: list[BBox] = []
        page_boxes: dict[int, BBox] = {}
        for item in items:
            if isinstance(item, tuple):
                block, text = item
            else:
                block, text = item, _normalize_pdf_text(str(_value(item, "text", "")))
            texts.append(text)
            uid = _value(block, "uid") or _value(block, "block_uid")
            if uid:
                block_uids.append(str(uid))
            page = _value(block, "page_number")
            if page is not None:
                pages.append(int(page))
            box = _value(block, "bbox")
            if isinstance(box, BBox):
                boxes.append(box)
                if page is not None:
                    page_number = int(page)
                    page_boxes[page_number] = (
                        box
                        if page_number not in page_boxes
                        else page_boxes[page_number].union(box)
                    )
        text = "\n\n".join(texts)
        prefix = self.contextualizer.prefix(title, section, subsection, text)
        embedding = self._embedding_text(title, section, subsection, prefix, text)
        bbox = None
        for box in boxes:
            bbox = box if bbox is None else bbox.union(box)
        metadata = {
            "chunk_type": "text",
            "section_uid": section_uid,
            "section": section,
            "subsection": subsection,
            "block_uids": block_uids,
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "bbox": bbox.as_list() if bbox else None,
            "page_bboxes": {
                str(page): page_box.as_list()
                for page, page_box in sorted(page_boxes.items())
            },
            "token_count": self.token_counter(text),
            "contextual_prefix": prefix,
            "embedding_text": embedding,
        }
        return StructuredChunk(
            uid=f"chunk-text-{group_index}-{chunk_index}",
            text=text,
            page_numbers=tuple(sorted(set(pages))),
            section_path=tuple(value for value in (section, subsection) if value),
            block_uids=tuple(block_uids),
            metadata=metadata,
        )

    @staticmethod
    def _embedding_text(
        title: str,
        section: str | None,
        subsection: str | None,
        prefix: str,
        text: str,
    ) -> str:
        return "\n".join(value for value in (title, section, subsection, prefix, text) if value)
