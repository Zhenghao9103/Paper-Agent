from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .domain import BlockType, ClassifiedBlock, ParsedElement, SectionNode
from .layout import heading_number_depth


class DocumentStructureParser:
    """Build a lightweight section tree while preserving classified block identity."""

    def parse(
        self,
        ordered_blocks: Sequence[ClassifiedBlock],
    ) -> tuple[list[SectionNode], list[ClassifiedBlock]]:
        roots, enriched, _context_events = self._parse_blocks(ordered_blocks)
        return roots, enriched

    def parse_with_elements(
        self,
        ordered_blocks: Sequence[ClassifiedBlock],
        elements: Sequence[ParsedElement],
    ) -> tuple[list[SectionNode], list[ClassifiedBlock], list[ParsedElement]]:
        roots, enriched, context_events = self._parse_blocks(ordered_blocks)
        enriched_elements = [
            replace(element, **self._element_context(element, context_events))
            for element in elements
        ]
        return roots, enriched, enriched_elements

    def _parse_blocks(
        self, ordered_blocks: Sequence[ClassifiedBlock]
    ) -> tuple[
        list[SectionNode],
        list[ClassifiedBlock],
        list[tuple[int, float, str | None, str | None, str | None]],
    ]:
        roots: list[SectionNode] = []
        enriched: list[ClassifiedBlock] = []
        context_events: list[tuple[int, float, str | None, str | None, str | None]] = []
        stack: list[SectionNode] = []
        section_uids: set[str] = set()

        for block in ordered_blocks:
            reasons = list(block.reason_codes)
            heading = block.block_type in {BlockType.HEADING, BlockType.SUBHEADING}
            if heading:
                requested_level = heading_number_depth(block.text)
                if requested_level is None:
                    requested_level = (
                        2
                        if (
                            block.block_type is BlockType.SUBHEADING
                            or "font_rank_subheading" in block.reason_codes
                        )
                        else 1
                    )
                previous_level = stack[-1].level if stack else 0
                if requested_level > previous_level + 1:
                    reasons.append("hierarchy_jump")
                base_uid = f"section_{block.uid}"
                section_uid = base_uid
                collision_index = 2
                while section_uid in section_uids:
                    section_uid = f"{base_uid}_{collision_index}"
                    collision_index += 1
                section_uids.add(section_uid)
                node = SectionNode(
                    uid=section_uid,
                    title=block.text,
                    level=requested_level,
                )
                while stack and stack[-1].level >= requested_level:
                    stack.pop()
                if stack:
                    node.parent_uid = stack[-1].uid
                    stack[-1].children.append(node)
                else:
                    roots.append(node)
                stack.append(node)
                root = self._root(stack)
                section_title = root.title
                subsection_title = node.title if node.level > root.level else None
                node.block_uids.append(block.uid)
                enriched.append(
                    replace(
                        block,
                        reason_codes=tuple(dict.fromkeys(reasons)),
                        section=section_title,
                        subsection=subsection_title,
                        section_uid=root.uid,
                    )
                )
                context_events.append(
                    (
                        block.page_number,
                        block.bbox.y1,
                        section_title,
                        subsection_title,
                        root.uid,
                    )
                )
                continue

            if stack:
                root = self._root(stack)
                active = stack[-1]
                section_title = root.title
                subsection_title = active.title if active.level > root.level else None
                active.block_uids.append(block.uid)
                enriched.append(
                    replace(
                        block,
                        reason_codes=tuple(dict.fromkeys(reasons)),
                        section=section_title,
                        subsection=subsection_title,
                        section_uid=root.uid,
                    )
                )
                context_events.append(
                    (
                        block.page_number,
                        block.bbox.y1,
                        section_title,
                        subsection_title,
                        root.uid,
                    )
                )
            else:
                enriched.append(replace(block, reason_codes=tuple(dict.fromkeys(reasons))))
                context_events.append((block.page_number, block.bbox.y1, None, None, None))

        return roots, enriched, context_events

    @staticmethod
    def _element_context(
        element: ParsedElement,
        context_events: Sequence[tuple[int, float, str | None, str | None, str | None]],
    ) -> dict[str, str | None]:
        selected: tuple[int, float, str | None, str | None, str | None] | None = None
        for event in context_events:
            page_number, y1, *_ = event
            if page_number < element.page_number or (
                page_number == element.page_number and y1 <= element.bbox.y0
            ):
                selected = event
        if selected is None:
            return {"section": None, "subsection": None, "section_uid": None}
        return {
            "section": selected[2],
            "subsection": selected[3],
            "section_uid": selected[4],
        }

    @staticmethod
    def _root(stack: Sequence[SectionNode]) -> SectionNode:
        return stack[0]
