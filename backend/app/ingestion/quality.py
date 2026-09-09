"""Deterministic, non-fatal quality checks for parsed PDF documents."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable

from .domain import (
    BlockType,
    ElementParseStatus,
    ElementType,
    ParsedDocument,
    ParseReport,
    ParseStatus,
    ParseWarning,
)

_SUSPICIOUS_RE = re.compile(r"(?:�|Ã.|Â.){2,}")
_TEXT_BLOCK_TYPES = {
    BlockType.PARAGRAPH,
    BlockType.LIST,
    BlockType.ABSTRACT,
    BlockType.REFERENCE,
}


class ParseQualityChecker:
    """Check parser invariants without making local element errors fatal.

    The checker deliberately works on the typed in-memory result.  It only marks a
    document failed when there is no usable Text Layer at all; malformed local
    geometry, partial tables, and missing captions become explicit warnings.
    """

    def check(self, document: ParsedDocument) -> ParseReport:
        report = document.report
        warnings: list[ParseWarning] = list(report.warnings)
        reported_page_count = report.page_count
        page_count = len(document.pages)
        block_count = len(document.blocks)
        element_count = len(document.elements)
        chunk_count = len(document.chunks)
        report.page_count = page_count
        report.block_count = block_count
        report.element_count = element_count
        report.chunk_count = chunk_count

        self._warn_page_count(reported_page_count, page_count, warnings)
        self._warn_bboxes(document, warnings)
        self._warn_text(document, warnings)
        self._warn_order(document, warnings)
        self._warn_chunks(document, warnings)
        self._warn_elements(document, warnings)
        self._warn_references(document, warnings)

        usable_text = any(
            block.text.strip()
            and block.block_type in _TEXT_BLOCK_TYPES
            for block in document.blocks
        ) or any(chunk.text.strip() for chunk in document.chunks if not chunk.element_uids)
        if not document.pages:
            warnings.append(
                ParseWarning(
                    code="no_pages",
                    message="The PDF contains no extracted pages.",
                    stage="quality",
                )
            )
        if not usable_text:
            warnings.append(
                ParseWarning(
                    code="no_usable_text",
                    message="The document has no usable PDF Text Layer body text.",
                    stage="quality",
                )
            )
            report.status = ParseStatus.FAILED
        elif report.errors:
            report.status = ParseStatus.FAILED
        elif warnings:
            report.status = ParseStatus.SUCCESS_WITH_WARNINGS
        else:
            report.status = ParseStatus.SUCCESS
        report.warnings = _dedupe_warnings(warnings)
        return report

    # ``validate`` is a descriptive alias used by orchestration callers.
    validate = check

    @staticmethod
    def _warn_page_count(
        reported_page_count: int,
        page_count: int,
        warnings: list[ParseWarning],
    ) -> None:
        if reported_page_count and reported_page_count != page_count:
            warnings.append(
                ParseWarning(
                    code="page_count_mismatch",
                    message="Parse report page count differs from extracted pages.",
                    stage="quality",
                )
            )

    @staticmethod
    def _warn_bboxes(document: ParsedDocument, warnings: list[ParseWarning]) -> None:
        pages = {page.page_number: page for page in document.pages}
        for block in document.blocks:
            page = pages.get(block.page_number)
            if page is not None and not block.bbox.contains_within(page.width, page.height):
                warnings.append(
                    ParseWarning(
                        code="bbox_out_of_bounds",
                        message="Block bbox lies outside its page bounds.",
                        page_number=block.page_number,
                        block_uid=block.uid,
                        stage="quality",
                    )
                )
        for element in document.elements:
            page = pages.get(element.page_number)
            if page is not None and not element.bbox.contains_within(page.width, page.height):
                warnings.append(
                    ParseWarning(
                        code="bbox_out_of_bounds",
                        message="Element bbox lies outside its page bounds.",
                        page_number=element.page_number,
                        element_uid=element.uid,
                        stage="quality",
                    )
                )

    @staticmethod
    def _warn_text(document: ParsedDocument, warnings: list[ParseWarning]) -> None:
        all_text = " ".join(block.text for block in document.blocks)
        if all_text and _SUSPICIOUS_RE.search(all_text):
            warnings.append(
                ParseWarning(
                    code="suspicious_characters",
                    message="Extracted text contains repeated replacement or mojibake characters.",
                    stage="quality",
                )
            )
        if document.pages:
            populated_pages = {
                block.page_number
                for block in document.blocks
                if block.text.strip()
            }
            nonempty = sum(
                bool(page.blocks) or page.page_number in populated_pages
                for page in document.pages
            )
            document.report.text_coverage = nonempty / len(document.pages)
            if document.report.text_coverage < 0.5:
                warnings.append(
                    ParseWarning(
                        code="low_text_coverage",
                        message="Fewer than half of pages contain extracted text blocks.",
                        stage="quality",
                    )
                )

    @staticmethod
    def _warn_order(document: ParsedDocument, warnings: list[ParseWarning]) -> None:
        by_page: dict[int, list[int]] = defaultdict(list)
        for block in document.blocks:
            source_index = block.source_block_indices[0] if block.source_block_indices else 0
            by_page[block.page_number].append(source_index)
        for page_number, indices in by_page.items():
            if len(indices) != len(set(indices)):
                warnings.append(
                    ParseWarning(
                        code="duplicate_source_index",
                        message="Multiple classified blocks share a source occurrence index.",
                        page_number=page_number,
                        stage="quality",
                    )
                )

    @staticmethod
    def _warn_chunks(document: ParsedDocument, warnings: list[ParseWarning]) -> None:
        seen_uids: set[str] = set()
        seen_hashes: dict[str, str] = {}
        block_map = {block.uid: block for block in document.blocks}
        for chunk in document.chunks:
            if chunk.uid in seen_uids:
                warnings.append(
                    ParseWarning(
                        code="duplicate_chunk",
                        message="Chunk UID is repeated.",
                        stage="quality",
                    )
                )
            seen_uids.add(chunk.uid)
            normalized = " ".join(chunk.text.split()).lower()
            if normalized:
                digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
                previous = seen_hashes.get(digest)
                if previous and previous != chunk.uid:
                    warnings.append(
                        ParseWarning(
                            code="duplicate_chunk_content",
                            message=f"Chunk content duplicates {previous}.",
                            stage="quality",
                        )
                    )
                seen_hashes[digest] = chunk.uid
            section_uids = {
                block_map[uid].section_uid
                for uid in chunk.block_uids
                if uid in block_map and block_map[uid].section_uid
            }
            if len(section_uids) > 1:
                warnings.append(
                    ParseWarning(
                        code="cross_section_chunk",
                        message="Chunk spans unrelated section UIDs.",
                        stage="quality",
                    )
                )

    @staticmethod
    def _warn_elements(document: ParsedDocument, warnings: list[ParseWarning]) -> None:
        for element in document.elements:
            if element.status is not ElementParseStatus.SUCCESS:
                warnings.append(
                    ParseWarning(
                        code="partial_element",
                        message=f"{element.element_type.value} parsing is {element.status.value}.",
                        page_number=element.page_number,
                        element_uid=element.uid,
                        stage="quality",
                    )
                )
            missing_caption = (
                element.element_type in {ElementType.FIGURE, ElementType.TABLE}
                and not element.caption
            )
            if missing_caption:
                warnings.append(
                    ParseWarning(
                        code="missing_caption",
                        message="Figure or table has no linked caption.",
                        page_number=element.page_number,
                        element_uid=element.uid,
                        stage="quality",
                    )
                )
            image_path = element.image_path
            if image_path and not image_path.exists():
                warnings.append(
                    ParseWarning(
                        code="missing_element_image",
                        message="Saved element image does not exist.",
                        page_number=element.page_number,
                        element_uid=element.uid,
                        stage="quality",
                    )
                )

    @staticmethod
    def _warn_references(document: ParsedDocument, warnings: list[ParseWarning]) -> None:
        for reference in document.cross_references:
            if getattr(reference, "resolution_status", "unresolved") == "resolved":
                continue
            warnings.append(
                ParseWarning(
                    code="unresolved_reference",
                    message=(
                        f"Could not resolve reference "
                        f"{getattr(reference, 'reference_text', '')}."
                    ),
                    page_number=getattr(reference, "page_number", None),
                    block_uid=getattr(reference, "source_uid", None),
                    stage="quality",
                )
            )


def _dedupe_warnings(warnings: Iterable[ParseWarning]) -> list[ParseWarning]:
    result: list[ParseWarning] = []
    seen: set[tuple[object, ...]] = set()
    for warning in warnings:
        key = (
            warning.code,
            warning.page_number,
            warning.block_uid,
            warning.element_uid,
            warning.message,
        )
        if key not in seen:
            seen.add(key)
            result.append(warning)
    return result
