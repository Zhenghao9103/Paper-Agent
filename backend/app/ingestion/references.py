from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

_REFERENCE_RE = re.compile(
    r"(?P<text>\b(?:fig(?:ure)?|tab(?:le)?|eq(?:uation)?|sec(?:tion)?)\.?\s*"
    r"(?:\(?[0-9]+(?:\.[0-9]+)*\)?|[IVXLCDM]+(?:\.[0-9]+)*))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CrossReference:
    source_uid: str
    reference_type: str
    reference_text: str
    normalized_label: str
    target_id: str | None = None
    resolution_status: str = "unresolved"
    page_number: int | None = None
    section: str | None = None


def _block_value(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, Mapping):
        return block.get(key, default)
    return getattr(block, key, default)


def _canonical(match: re.Match[str]) -> tuple[str, str]:
    text = match.group("text")
    compact = re.sub(r"\s+", " ", text.strip().rstrip(".,;: "))
    prefix, value = re.match(r"([A-Za-z]+)\.?\s*\(?(.+?)\)?$", compact).groups()
    prefix = prefix.casefold()
    if prefix.startswith("fig"):
        return "figure", f"Figure {value}"
    if prefix.startswith("tab"):
        return "table", f"Table {value}"
    if prefix.startswith("eq"):
        return "equation", f"Eq. {value}"
    return "section", f"Section {value}"


class ReferenceParser:
    def parse(self, blocks: Sequence[Any]) -> list[CrossReference]:
        references: list[CrossReference] = []
        for block_index, block in enumerate(blocks):
            text = str(_block_value(block, "text", "") or "")
            source_uid = str(
                _block_value(block, "uid", None)
                or _block_value(block, "block_uid", None)
                or f"block-{block_index}"
            )
            for match in _REFERENCE_RE.finditer(text):
                reference_type, normalized = _canonical(match)
                references.append(
                    CrossReference(
                        source_uid=source_uid,
                        reference_type=reference_type,
                        reference_text=match.group("text"),
                        normalized_label=normalized,
                        page_number=_block_value(block, "page_number"),
                        section=_block_value(block, "section"),
                    )
                )
        return references


class CrossReferenceResolver:
    def resolve(
        self,
        references: Iterable[CrossReference],
        elements: Sequence[Any] = (),
        sections: Sequence[Any] = (),
    ) -> list[CrossReference]:
        targets: dict[tuple[str, str], list[str]] = {}

        def register(key: tuple[str, str], uid: str) -> None:
            values = targets.setdefault(key, [])
            if uid not in values:
                values.append(uid)

        for element in elements:
            uid = _block_value(element, "uid") or _block_value(element, "element_uid")
            metadata = _block_value(element, "metadata", {}) or {}
            label = _block_value(element, "label") or metadata.get("label")
            element_type = _block_value(element, "element_type")
            if element_type is not None:
                kind = getattr(element_type, "value", element_type)
            else:
                kind = metadata.get("element_type")
            if uid and label and kind:
                register((str(kind), str(label).casefold()), str(uid))
            for key, ref_type in (
                ("figure_id", "figure"),
                ("table_id", "table"),
                ("equation_id", "equation"),
            ):
                value = metadata.get(key)
                if uid and value:
                    register((ref_type, str(value).casefold()), str(uid))
        for section in sections:
            uid = getattr(section, "uid", None) or _block_value(section, "uid")
            title = getattr(section, "title", None) or _block_value(section, "title")
            if uid and title:
                register(("section", f"section {title}".casefold()), str(uid))
                register(("section", str(title).casefold()), str(uid))
                number = re.match(r"\s*([0-9]+(?:\.[0-9]+)*)\b", str(title))
                if number:
                    register(("section", f"section {number.group(1)}"), str(uid))
                roman = re.match(r"\s*([IVXLCDM]+(?:\.[0-9]+)*)\b", str(title), re.IGNORECASE)
                if roman:
                    register(("section", f"section {roman.group(1)}".casefold()), str(uid))
        resolved: list[CrossReference] = []
        for reference in references:
            candidates = targets.get(
                (reference.reference_type, reference.normalized_label.casefold()), []
            )
            target = candidates[0] if len(candidates) == 1 else None
            resolved.append(
                replace(
                    reference,
                    target_id=target,
                    resolution_status="resolved" if target else "unresolved",
                )
            )
        return resolved
