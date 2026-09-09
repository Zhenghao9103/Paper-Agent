from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import fitz

from ..services.llm import describe_figure
from .domain import (
    BBox,
    BlockType,
    ElementParseStatus,
    ElementType,
    ImageRegion,
    PageLayout,
    ParsedElement,
)
from .figure_descriptions import FigureDescriptionError, FigureDescriptionResult

_CAPTION_RE = re.compile(
    r"^\s*(?:figure|fig\.?)[ \t]*"
    r"(?P<label>[A-Za-z]*\d+(?:\.\d+)*|[IVXLCDM]+)"
    r"(?=\s|[:.)-]|$)",
    re.IGNORECASE,
)
_TABLE_RE = re.compile(
    r"^\s*(?:table|tab\.?)[ \t]*(?:[A-Za-z]*\d+(?:\.\d+)*|[IVXLCDM]+)"
    r"(?=\s|[:.)-]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VisionRequest:
    image_path: Path
    title: str
    section: str | None
    caption: str | None


@dataclass(frozen=True)
class _BlockView:
    text: str
    bbox: BBox
    uid: str
    block_type: str | None = None
    section: str | None = None
    subsection: str | None = None
    source_index: int = 0


@dataclass(frozen=True)
class _Candidate:
    bbox: BBox
    kind: str
    source_index: int


def _as_bbox(value: Any) -> BBox | None:
    if isinstance(value, BBox):
        return value
    try:
        return BBox(*value)
    except (TypeError, ValueError):
        return None


def _block_view(block: Any, page_number: int, ordinal: int) -> _BlockView | None:
    if isinstance(block, Mapping):
        text = str(block.get("text", ""))
        bbox = _as_bbox(block.get("bbox"))
        if bbox is None:
            return None
        block_type = block.get("block_type") or block.get("type")
        uid = str(block.get("uid") or f"page{page_number}_block{ordinal}")
        return _BlockView(
            text=text,
            bbox=bbox,
            uid=uid,
            block_type=str(block_type) if block_type is not None else None,
            section=block.get("section"),
            subsection=block.get("subsection"),
            source_index=int(block.get("block_index", block.get("source_index", ordinal))),
        )
    bbox = _as_bbox(getattr(block, "bbox", None))
    if bbox is None:
        return None
    block_type = getattr(block, "block_type", None)
    block_type_value = getattr(block_type, "value", block_type)
    return _BlockView(
        text=str(getattr(block, "text", "")),
        bbox=bbox,
        uid=str(getattr(block, "uid", f"page{page_number}_block{ordinal}")),
        block_type=str(block_type_value) if block_type_value is not None else None,
        section=getattr(block, "section", None),
        subsection=getattr(block, "subsection", None),
        source_index=int(
            getattr(block, "block_index", getattr(block, "source_index", ordinal))
        ),
    )


def _page_number(page: Any) -> int:
    value = getattr(page, "page_number", None)
    if value is None:
        value = getattr(page, "number", 0) + 1
    return int(value)


def _page_size(page: Any) -> tuple[float, float]:
    width = getattr(page, "width", None)
    height = getattr(page, "height", None)
    if width is not None and height is not None:
        return float(width), float(height)
    rect = getattr(page, "rect", None)
    if rect is None:
        raise ValueError("page must expose width and height")
    return float(rect.width), float(rect.height)


def _intersection(left: BBox, right: BBox) -> float:
    return max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0)) * max(
        0.0, min(left.y1, right.y1) - max(left.y0, right.y0)
    )


def _union(left: BBox, right: BBox) -> BBox:
    return left.union(right)


def _canonical_label(match: re.Match[str]) -> str:
    return f"Figure {match.group('label')}"


def _caption_label(text: str) -> str | None:
    match = _CAPTION_RE.match(" ".join(text.split()))
    return _canonical_label(match) if match else None


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    return slug or "unknown"


class FigureParser:
    """Detect figure regions from PDF graphics and persist atomic PNG crops.

    The parser is intentionally deterministic.  Vision is an optional injected
    callable and is only asked to describe an already saved crop.
    """

    def __init__(
        self,
        describe: Callable[[VisionRequest], str | None] | None = None,
        *,
        min_area_ratio: float = 0.01,
    ) -> None:
        if not 0 < min_area_ratio < 1:
            raise ValueError("min_area_ratio must be between 0 and 1")
        self.describe = describe or self._default_describe
        self.min_area_ratio = min_area_ratio

    @staticmethod
    def _default_describe(request: VisionRequest) -> str | None:
        return describe_figure(
            request.image_path,
            title=request.title,
            section=request.section,
            caption=request.caption,
        )

    def parse(
        self,
        pdf_path: Path,
        page: PageLayout | Any,
        blocks: Sequence[Any],
        output_dir: Path,
        *,
        title: str = "",
    ) -> list[ParsedElement]:
        page_number = _page_number(page)
        page_width, page_height = _page_size(page)
        if (
            not isfinite(page_width)
            or not isfinite(page_height)
            or page_width <= 0
            or page_height <= 0
        ):
            return []
        block_views = tuple(
            view
            for ordinal, block in enumerate(blocks)
            if (view := _block_view(block, page_number, ordinal)) is not None
        )
        document = fitz.open(pdf_path)
        try:
            if not 1 <= page_number <= document.page_count:
                raise ValueError(f"page_number must be between 1 and {document.page_count}")
            pdf_page = document.load_page(page_number - 1)
            candidates = self._candidates(
                pdf_page,
                page,
                block_views,
                page_width,
                page_height,
            )
            captions = self._captions(block_views)
            return self._build_elements(
                pdf_page,
                page_number,
                page_width,
                page_height,
                candidates,
                captions,
                output_dir,
                title,
            )
        finally:
            document.close()

    def _candidates(
        self,
        pdf_page: fitz.Page,
        page: Any,
        blocks: Sequence[_BlockView],
        page_width: float,
        page_height: float,
    ) -> list[_Candidate]:
        page_area = page_width * page_height
        candidates: list[_Candidate] = []
        image_regions = getattr(page, "image_regions", ())
        seen_raster_xrefs: set[int] = set()
        seen_margin_signatures: set[tuple[Any, ...]] = set()
        for ordinal, region in enumerate(image_regions):
            if not isinstance(region, ImageRegion):
                bbox = _as_bbox(getattr(region, "bbox", None))
                xref = getattr(region, "xref", None)
                metadata = getattr(region, "metadata", {})
                width = getattr(region, "width", None)
                height = getattr(region, "height", None)
                ext = getattr(region, "ext", None)
            else:
                bbox = region.bbox
                xref = region.xref
                metadata = region.metadata
                width = region.width
                height = region.height
                ext = region.ext
            if bbox is not None:
                if xref is not None and xref in seen_raster_xrefs:
                    continue
                if xref is not None:
                    seen_raster_xrefs.add(int(xref))
                if bbox.y0 <= page_height * 0.12 or bbox.y1 >= page_height * 0.88:
                    signature = (
                        width,
                        height,
                        ext,
                        metadata.get("size") if isinstance(metadata, Mapping) else None,
                        metadata.get("colorspace")
                        if isinstance(metadata, Mapping)
                        else None,
                        metadata.get("bpc") if isinstance(metadata, Mapping) else None,
                    )
                    if signature in seen_margin_signatures:
                        continue
                    seen_margin_signatures.add(signature)
                candidates.append(_Candidate(bbox, "raster", ordinal))

        drawings: Sequence[Mapping[str, Any]] = ()
        try:
            drawings = pdf_page.get_drawings()
            clusters = pdf_page.cluster_drawings(drawings=drawings)
        except (AttributeError, TypeError, RuntimeError, ValueError):
            clusters = []
            try:
                clusters = [drawing.get("rect") for drawing in pdf_page.get_drawings()]
            except (AttributeError, RuntimeError, ValueError):
                pass
        for ordinal, rect in enumerate(clusters):
            bbox = _as_bbox(rect)
            if bbox is not None and not self._is_grid_region(bbox, drawings):
                candidates.append(_Candidate(bbox, "vector", ordinal))

        filtered: list[_Candidate] = []
        for candidate in candidates:
            bbox = candidate.bbox
            if bbox.area < page_area * self.min_area_ratio:
                continue
            if bbox.width <= 2 or bbox.height <= 2 or bbox.area > page_area * 0.9:
                continue
            if self._is_table_region(bbox, blocks):
                continue
            if candidate.kind == "vector" and self._is_text_dominated(bbox, blocks):
                continue
            filtered.append(candidate)
        return self._merge_candidates(filtered, page_width, page_height)

    @staticmethod
    def _is_grid_region(
        bbox: BBox, drawings: Sequence[Mapping[str, Any]]
    ) -> bool:
        horizontal = 0
        vertical = 0
        for drawing in drawings:
            rect = _as_bbox(drawing.get("rect"))
            if rect is None:
                continue
            if (
                rect.x0 < bbox.x0 - 1
                or rect.x1 > bbox.x1 + 1
                or rect.y0 < bbox.y0 - 1
                or rect.y1 > bbox.y1 + 1
            ):
                continue
            if rect.width >= max(20.0, bbox.width * 0.5) and rect.height <= 1.5:
                horizontal += 1
            elif rect.height >= max(20.0, bbox.height * 0.5) and rect.width <= 1.5:
                vertical += 1
        return horizontal >= 2 and vertical >= 2

    @staticmethod
    def _is_table_region(bbox: BBox, blocks: Sequence[_BlockView]) -> bool:
        for block in blocks:
            kind = (block.block_type or "").casefold()
            if kind not in {BlockType.TABLE.value, BlockType.TABLE_CAPTION.value, "table"}:
                continue
            if _intersection(bbox, block.bbox) / max(bbox.area, 1.0) >= 0.45:
                return True
        return False

    @staticmethod
    def _is_text_dominated(bbox: BBox, blocks: Sequence[_BlockView]) -> bool:
        intersected = sum(_intersection(bbox, block.bbox) for block in blocks)
        return intersected / max(bbox.area, 1.0) >= 0.55

    @staticmethod
    def _merge_candidates(
        candidates: Sequence[_Candidate], page_width: float, page_height: float
    ) -> list[_Candidate]:
        merged: list[_Candidate] = []
        max_gap = max(page_width, page_height) * 0.025
        for candidate in sorted(
            candidates, key=lambda item: (item.bbox.y0, item.bbox.x0, item.kind)
        ):
            match_index: int | None = None
            for index, existing in enumerate(merged):
                intersection = _intersection(candidate.bbox, existing.bbox)
                smaller_area = min(candidate.bbox.area, existing.bbox.area)
                gap_x = max(
                    existing.bbox.x0 - candidate.bbox.x1,
                    candidate.bbox.x0 - existing.bbox.x1,
                    0.0,
                )
                gap_y = max(
                    existing.bbox.y0 - candidate.bbox.y1,
                    candidate.bbox.y0 - existing.bbox.y1,
                    0.0,
                )
                if intersection / max(smaller_area, 1.0) >= 0.55 or (
                    gap_x <= max_gap and gap_y <= max_gap and intersection > 0
                ):
                    match_index = index
                    break
            if match_index is None:
                merged.append(candidate)
            else:
                existing = merged[match_index]
                merged[match_index] = _Candidate(
                    _union(existing.bbox, candidate.bbox),
                    "raster" if existing.kind == "raster" else existing.kind,
                    existing.source_index,
                )
        return merged

    @staticmethod
    def _captions(blocks: Sequence[_BlockView]) -> list[_BlockView]:
        return [
            block
            for block in blocks
            if _caption_label(block.text) is not None and not _TABLE_RE.match(block.text)
        ]

    def _build_elements(
        self,
        pdf_page: fitz.Page,
        page_number: int,
        page_width: float,
        page_height: float,
        candidates: Sequence[_Candidate],
        captions: Sequence[_BlockView],
        output_dir: Path,
        title: str,
    ) -> list[ParsedElement]:
        used_captions: set[str] = set()
        elements: list[ParsedElement] = []
        for ordinal, candidate in enumerate(candidates, start=1):
            caption = self._link_caption(
                candidate.bbox, captions, used_captions, page_width, page_height
            )
            label = _caption_label(caption.text) if caption else None
            stem = (
                f"page{page_number}_fig{_safe_slug(label.split()[-1])}"
                if label
                else f"page{page_number}_figure{ordinal}"
            )
            output_path = output_dir / f"{stem}.png"
            warnings: list[str] = []
            structured_data: dict[str, Any] = {}
            description_model: str | None = None
            description_latency_ms: int | None = None
            try:
                self._crop(pdf_page, candidate.bbox, output_path, page_width, page_height)
            except (OSError, RuntimeError, ValueError):
                warnings.append("figure_crop_failed")
                image_path: str | None = None
                vision = None
            else:
                image_path = str(output_path)
                vision = None
                try:
                    vision = self.describe(
                        VisionRequest(
                            image_path=output_path,
                            title=title,
                            section=caption.section if caption else None,
                            caption=caption.text if caption else None,
                        )
                    )
                    if isinstance(vision, FigureDescriptionResult):
                        structured_data = vision.description.model_dump()
                        description_model = vision.model
                        description_latency_ms = vision.latency_ms
                        warnings.extend(vision.warnings)
                        vision = vision.description.summary or None
                    elif vision:
                        vision = vision.strip() or None
                except FigureDescriptionError as exc:
                    warnings.append(exc.code)
                    vision = None
                except Exception:
                    warnings.append("vision_description_failed")
                    vision = None
            if caption is None:
                warnings.append("caption_unresolved")
            status = ElementParseStatus.SUCCESS if not warnings else ElementParseStatus.PARTIAL
            metadata = {
                "figure_id": label,
                "image_path": image_path,
                "candidate_kind": candidate.kind,
                "bbox": candidate.bbox.as_list(),
                "vision_description": vision,
                "structured_data": structured_data,
                "description_model": description_model,
                "description_latency_ms": description_latency_ms,
                "warning_codes": warnings,
                "source_index": candidate.source_index,
            }
            elements.append(
                ParsedElement(
                    uid=f"figure-p{page_number}-{_safe_slug(label or str(ordinal))}",
                    element_type=ElementType.FIGURE,
                    status=status,
                    page_number=page_number,
                    bbox=candidate.bbox,
                    content=vision or (caption.text if caption else ""),
                    caption=caption.text if caption else None,
                    source_block_uids=(caption.uid,) if caption else (),
                    metadata=metadata,
                    section=caption.section if caption else None,
                    subsection=caption.subsection if caption else None,
                )
            )
        return elements

    @staticmethod
    def _link_caption(
        bbox: BBox,
        captions: Sequence[_BlockView],
        used: set[str],
        page_width: float,
        page_height: float,
    ) -> _BlockView | None:
        candidates: list[tuple[float, _BlockView]] = []
        max_gap = min(page_height * 0.12, 96.0)
        for caption in captions:
            if caption.uid in used:
                continue
            horizontal = _intersection(
                BBox(bbox.x0, 0, bbox.x1, 1),
                BBox(caption.bbox.x0, 0, caption.bbox.x1, 1),
            ) / max(min(bbox.width, caption.bbox.width), 1.0)
            if horizontal < 0.15:
                continue
            if caption.bbox.y0 >= bbox.y1:
                gap = caption.bbox.y0 - bbox.y1
                direction_bonus = 2.0
            elif bbox.y0 >= caption.bbox.y1:
                gap = bbox.y0 - caption.bbox.y1
                direction_bonus = 1.7
            else:
                gap = 0.0
                direction_bonus = 0.1
            if gap > max_gap:
                continue
            score = direction_bonus + min(horizontal, 1.0) - gap / max(page_height, 1.0)
            candidates.append((score, caption))
        if not candidates:
            return None
        _, selected = max(
            candidates,
            key=lambda item: (item[0], -item[1].bbox.y0, -item[1].source_index),
        )
        used.add(selected.uid)
        return selected

    @staticmethod
    def _crop(
        pdf_page: fitz.Page,
        bbox: BBox,
        output_path: Path,
        page_width: float,
        page_height: float,
    ) -> None:
        rect = fitz.Rect(
            max(0.0, bbox.x0),
            max(0.0, bbox.y0),
            min(page_width, bbox.x1),
            min(page_height, bbox.y1),
        )
        if rect.width <= 0 or rect.height <= 0:
            raise ValueError("figure bbox is outside page")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            pixmap = pdf_page.get_pixmap(
                matrix=fitz.Matrix(2, 2), clip=rect, alpha=False
            )
            with NamedTemporaryFile(
                dir=output_path.parent,
                prefix=f".{output_path.stem}-",
                suffix=".tmp.png",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            pixmap.save(temporary_path)
            if temporary_path.stat().st_size <= 0:
                raise OSError("figure crop is empty")
            temporary_path.replace(output_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
