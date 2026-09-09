from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType

from .domain import BBox, BlockType, ClassifiedBlock, LayoutDiagnosis, PageLayout, RawBlock


@dataclass(frozen=True)
class BodyStats:
    dominant_fonts: tuple[str, ...] = ()
    median_font_size: float = 0.0
    common_line_height: float = 0.0
    span_count: int = 0
    character_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dominant_fonts", tuple(self.dominant_fonts))


_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:[.)])?\s+(.+?)\s*$")
_ROMAN_HEADING = re.compile(
    r"^\s*([IVXLCDM]+)(?:\.(\d+(?:\.\d+)*))?\.?\s+(.+?)\s*$", re.I
)
_NUMBERED_LIST = re.compile(r"^\s*(?:\(\d+\)|\d+[.)])(?:\s+|$)")
_BULLET = re.compile(r"^\s*(?:[•▪◦‣●○–—*-]|\(?\d+[.)])\s+")
_MARGIN_PAGE_NUMBER = re.compile(r"^\s*\d+\s*$")
_CAPTION = re.compile(
    r"^\s*(figure|fig\.?|table|tab\.?)\s+"
    r"(?:\d+(?:\.\d+)*|[IVXLCDM]+|[A-Za-z]*\d+[A-Za-z]*)"
    r"(?=\s|[:.)-]|$)",
    re.I,
)
_AUTHOR_YEAR_REFERENCE = re.compile(
    r"^\s*[A-Z][A-Za-z'’\-]+,\s*[A-Z](?:\.[A-Z]?)*\.\s*"
    r"\(?(?:19|20)\d{2}[a-z]?\)?(?:[.),:]|\s|$)"
    r"|^\s*[A-Z][A-Za-z'’\-]+\s+et\s+al\.\s*"
    r"\(?(?:19|20)\d{2}[a-z]?\)?(?:[.),:]|\s|$)"
)
_HEADING_WORDS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "related works",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experiment",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "future work",
    "acknowledgments",
    "acknowledgements",
    "references",
    "bibliography",
}
_EQUATION_SYMBOLS = set("=∑∫√≤≥≠≈→←↔∂∆∇±×÷").union({"\\", "^", "_"})
_EQUATION_LABEL = re.compile(
    r"^\s*(?:eq(?:uation)?\.?\s*)?\(\d{1,3}\)\s*$", re.I
)


def heading_number_depth(text: str) -> int | None:
    """Return the dotted numeric prefix depth, if one is present."""

    match = _NUMBERED_HEADING.match(" ".join(text.split()))
    if match is None:
        roman = _ROMAN_HEADING.match(" ".join(text.split()))
        if roman is None:
            return None
        suffix = roman.group(2)
        return 1 + (len(suffix.split(".")) if suffix else 0)
    return len(match.group(1).split("."))


def looks_like_sentence(text: str) -> bool:
    """Conservatively identify sentence-like text, including numbered sentences."""

    normalized = " ".join(text.split())
    if not normalized:
        return False
    body = _NUMBERED_HEADING.match(normalized)
    roman = _ROMAN_HEADING.match(normalized) if body is None else None
    candidate = body.group(2) if body else roman.group(3) if roman else normalized
    words = re.findall(r"[\wÀ-ÖØ-öø-ÿ]+", candidate, flags=re.UNICODE)
    has_terminal = bool(re.search(r"[.!?。！？;；:]\s*$", candidate))
    verb_like = {
        "are",
        "be",
        "can",
        "contains",
        "demonstrates",
        "does",
        "explores",
        "has",
        "have",
        "indicates",
        "is",
        "may",
        "provides",
        "shows",
        "study",
        "studies",
        "uses",
        "was",
        "were",
        "will",
        "works",
    }
    if has_terminal and (
        len(words) >= 5
        or len(candidate) >= 28
        or bool(verb_like.intersection(word.casefold() for word in words))
    ):
        return True
    if len(words) >= 12 and any(mark in candidate for mark in ",;，；"):
        return True
    return False


@dataclass(frozen=True)
class MarginDetection:
    classifications: Mapping[tuple[int, int], BlockType]
    retained_blocks: tuple[RawBlock, ...]
    normalized_recurrence: Mapping[str, float] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "retained_blocks", tuple(self.retained_blocks))
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        object.__setattr__(
            self, "classifications", MappingProxyType(dict(self.classifications))
        )
        object.__setattr__(
            self,
            "normalized_recurrence",
            MappingProxyType(dict(self.normalized_recurrence)),
        )

    @property
    def excluded_keys(self) -> frozenset[tuple[int, int]]:
        return frozenset(
            key
            for key, block_type in self.classifications.items()
            if block_type in {BlockType.HEADER, BlockType.FOOTER}
        )


def _weighted_median(values: Counter[float]) -> float:
    total = sum(values.values())
    if total == 0:
        return 0.0
    threshold = total / 2
    cumulative = 0
    for value, weight in sorted(values.items()):
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return 0.0


def compute_body_stats(pages: Sequence[PageLayout]) -> BodyStats:
    font_weights: Counter[str] = Counter()
    size_weights: Counter[float] = Counter()
    line_height_weights: Counter[float] = Counter()
    span_count = 0
    character_count = 0

    for page in pages:
        for block in page.blocks:
            for line in block.lines:
                line_characters = sum(
                    1
                    for span in line.spans
                    for character in span.text
                    if not character.isspace()
                )
                if line_characters and isfinite(line.bbox.height):
                    line_height_weights[round(line.bbox.height, 3)] += line_characters
                for span in line.spans:
                    weight = sum(
                        1 for character in span.text if not character.isspace()
                    )
                    if weight == 0:
                        continue
                    span_count += 1
                    character_count += weight
                    font_weights[span.font] += weight
                    if isfinite(span.size) and span.size > 0:
                        size_weights[round(span.size, 3)] += weight

    maximum_font_weight = max(font_weights.values(), default=0)
    dominant_fonts = tuple(
        sorted(
            font
            for font, weight in font_weights.items()
            if weight == maximum_font_weight
        )
    )
    common_line_height = 0.0
    if line_height_weights:
        maximum_height_weight = max(line_height_weights.values())
        common_line_height = min(
            height
            for height, weight in line_height_weights.items()
            if weight == maximum_height_weight
        )

    return BodyStats(
        dominant_fonts=dominant_fonts,
        median_font_size=_weighted_median(size_weights),
        common_line_height=float(common_line_height),
        span_count=span_count,
        character_count=character_count,
    )


_STANDALONE_PAGE_NUMBER = re.compile(r"^\s*\d+\s*$")


def _normalize_margin_text(text: str) -> str:
    if _STANDALONE_PAGE_NUMBER.fullmatch(text):
        return "<page-number>"
    return " ".join(text.casefold().split())


def _maximum_font_size(block: RawBlock) -> float:
    return max(
        (
            span.size
            for line in block.lines
            for span in line.spans
            if isfinite(span.size)
        ),
        default=0.0,
    )


class RepeatedMarginDetector:
    def __init__(self, band_ratio: float = 0.12, recurrence_threshold: float = 0.6):
        if not 0 < band_ratio < 0.5:
            raise ValueError("band_ratio must be between 0 and 0.5")
        if not 0 < recurrence_threshold <= 1:
            raise ValueError("recurrence_threshold must be between 0 and 1")
        self.band_ratio = band_ratio
        self.recurrence_threshold = recurrence_threshold

    def detect(
        self,
        pages: Sequence[PageLayout],
        body_stats: BodyStats | None = None,
    ) -> MarginDetection:
        retained_blocks = tuple(block for page in pages for block in page.blocks)
        classifications = {
            (block.page_number, block.block_index): BlockType.UNKNOWN
            for block in retained_blocks
        }
        eligible_pages = [
            page
            for page in pages
            if isfinite(page.width)
            and isfinite(page.height)
            and page.width > 0
            and page.height > 0
        ]
        if not eligible_pages:
            return MarginDetection(
                classifications=classifications,
                retained_blocks=retained_blocks,
                reason_codes=("no_eligible_pages",),
            )

        stats = body_stats or compute_body_stats(pages)
        candidates: list[tuple[RawBlock, BlockType, str]] = []
        pages_by_candidate: defaultdict[tuple[BlockType, str], set[int]] = defaultdict(set)
        for page in eligible_pages:
            for block in page.blocks:
                normalized = _normalize_margin_text(block.text)
                if not normalized:
                    continue
                if block.bbox.y1 <= page.height * self.band_ratio:
                    block_type = BlockType.HEADER
                elif block.bbox.y0 >= page.height * (1 - self.band_ratio):
                    block_type = BlockType.FOOTER
                else:
                    continue
                if (
                    page.page_number == 1
                    and stats.median_font_size > 0
                    and _maximum_font_size(block) >= stats.median_font_size * 1.4
                ):
                    continue
                candidate = (block_type, normalized)
                candidates.append((block, block_type, normalized))
                pages_by_candidate[candidate].add(page.page_number)

        eligible_count = len(eligible_pages)
        repeated = {
            candidate: len(page_numbers) / eligible_count
            for candidate, page_numbers in pages_by_candidate.items()
            if eligible_count > 1
            and len(page_numbers) / eligible_count >= self.recurrence_threshold
        }
        normalized_recurrence: dict[str, float] = {}
        for (_block_type, normalized), recurrence in repeated.items():
            normalized_recurrence[normalized] = max(
                recurrence, normalized_recurrence.get(normalized, 0.0)
            )
        for block, block_type, normalized in candidates:
            if (block_type, normalized) in repeated:
                classifications[(block.page_number, block.block_index)] = block_type

        repeated_types = {block_type for block_type, _ in repeated}
        reason_codes = tuple(
            reason
            for block_type, reason in (
                (BlockType.HEADER, "repeated_header"),
                (BlockType.FOOTER, "repeated_footer"),
            )
            if block_type in repeated_types
        )
        if not reason_codes:
            reason_codes = ("below_recurrence_threshold",)
        return MarginDetection(
            classifications=classifications,
            retained_blocks=retained_blocks,
            normalized_recurrence=normalized_recurrence,
            reason_codes=reason_codes,
        )


def detect_repeated_margins(
    pages: Sequence[PageLayout], body_stats: BodyStats | None = None
) -> MarginDetection:
    return RepeatedMarginDetector().detect(pages, body_stats)


def _block_key(block: RawBlock) -> tuple[int, int]:
    return block.page_number, block.block_index


def _exclusion_keys(
    exclusions: Collection[object] | Mapping[object, object] | MarginDetection | None,
) -> frozenset[tuple[int, int]]:
    if exclusions is None:
        return frozenset()
    if isinstance(exclusions, MarginDetection):
        return exclusions.excluded_keys
    keys: set[tuple[int, int]] = set()
    if isinstance(exclusions, Mapping):
        items = exclusions.items()
        for key, value in items:
            if value not in {BlockType.HEADER, BlockType.FOOTER, True}:
                continue
            if isinstance(key, RawBlock):
                keys.add(_block_key(key))
            elif isinstance(key, tuple) and len(key) == 2:
                keys.add((int(key[0]), int(key[1])))
        return frozenset(keys)
    for item in exclusions:
        if isinstance(item, RawBlock):
            keys.add(_block_key(item))
        elif isinstance(item, tuple) and len(item) == 2:
            keys.add((int(item[0]), int(item[1])))
    return frozenset(keys)


def _non_whitespace_length(text: str) -> int:
    return sum(1 for character in text if not character.isspace())


def _merged_vertical_intervals(
    blocks: Sequence[RawBlock], page_height: float
) -> list[tuple[float, float]]:
    intervals = sorted(
        (max(0.0, block.bbox.y0), min(page_height, block.bbox.y1))
        for block in blocks
        if max(0.0, block.bbox.y0) < min(page_height, block.bbox.y1)
    )
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _interval_length(intervals: Sequence[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _overlap_length(
    left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]
) -> float:
    overlap = 0.0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        overlap += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap


@dataclass(frozen=True)
class _GapSample:
    left: float
    right: float
    support: float

    @property
    def midpoint(self) -> float:
        return (self.left + self.right) / 2

    @property
    def width(self) -> float:
        return self.right - self.left


def _merge_horizontal_intervals(
    intervals: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _select_supported_center_gap(
    blocks: Sequence[RawBlock], page_width: float, page_height: float
) -> tuple[float, float, float] | None:
    vertical_edges = sorted(
        {
            min(page_height, max(0.0, coordinate))
            for block in blocks
            for coordinate in (block.bbox.y0, block.bbox.y1)
        }
    )
    page_center = page_width / 2
    samples: list[_GapSample] = []
    for band_start, band_end in zip(vertical_edges, vertical_edges[1:], strict=False):
        if band_end <= band_start:
            continue
        horizontal = _merge_horizontal_intervals(
            [
                (max(0.0, block.bbox.x0), min(page_width, block.bbox.x1))
                for block in blocks
                if block.bbox.y0 < band_end
                and block.bbox.y1 > band_start
                and max(0.0, block.bbox.x0) < min(page_width, block.bbox.x1)
            ]
        )
        for left_interval, right_interval in zip(
            horizontal, horizontal[1:], strict=False
        ):
            gap_left = left_interval[1]
            gap_right = right_interval[0]
            gap_width = gap_right - gap_left
            gap_midpoint = (gap_left + gap_right) / 2
            if (
                gap_width >= page_width * 0.04
                and abs(gap_midpoint - page_center) <= page_width * 0.15
            ):
                samples.append(
                    _GapSample(gap_left, gap_right, band_end - band_start)
                )
    if not samples:
        return None

    clusters: list[list[_GapSample]] = []
    for sample in sorted(
        samples,
        key=lambda item: (item.midpoint, item.left, item.right, item.support),
    ):
        compatible = [
            cluster
            for cluster in clusters
            if abs(
                sample.midpoint
                - sum(item.midpoint * item.support for item in cluster)
                / sum(item.support for item in cluster)
            )
            <= page_width * 0.04
        ]
        if compatible:
            cluster = min(
                compatible,
                key=lambda items: (
                    abs(
                        sample.midpoint
                        - sum(item.midpoint * item.support for item in items)
                        / sum(item.support for item in items)
                    ),
                    min(item.left for item in items),
                ),
            )
            cluster.append(sample)
        else:
            clusters.append([sample])

    scored: list[tuple[float, float, float, float, float]] = []
    for cluster in clusters:
        support = sum(item.support for item in cluster)
        left = sum(item.left * item.support for item in cluster) / support
        right = sum(item.right * item.support for item in cluster) / support
        width = right - left
        midpoint = (left + right) / 2
        scored.append((support, width, -abs(midpoint - page_center), -left, right))
    support, _width, _center_score, negative_left, right = max(scored)
    return -negative_left, right, support


class LayoutAnalyzer:
    def detect_columns(
        self,
        page: PageLayout,
        exclusions: Collection[object]
        | Mapping[object, object]
        | MarginDetection
        | None = None,
    ) -> LayoutDiagnosis:
        if (
            not isfinite(page.width)
            or not isfinite(page.height)
            or page.width <= 0
            or page.height <= 0
        ):
            return self._single_column(page.page_number, "invalid_page_geometry")

        excluded = _exclusion_keys(exclusions)
        candidates = [
            block
            for block in page.blocks
            if _block_key(block) not in excluded
            and _non_whitespace_length(block.text) > 0
            and block.bbox.width > 0
            and block.bbox.height > 0
            and block.bbox.width < page.width * 0.65
        ]
        if not candidates:
            return self._single_column(page.page_number, "no_column_candidates")

        selected_gap = _select_supported_center_gap(
            candidates, page.width, page.height
        )
        if selected_gap is None:
            page_center = page.width / 2
            coarse_left = [
                block for block in candidates if block.bbox.x1 <= page_center
            ]
            coarse_right = [
                block for block in candidates if block.bbox.x0 >= page_center
            ]
            if len(coarse_left) >= 2 and len(coarse_right) >= 2:
                coarse_left_intervals = _merged_vertical_intervals(
                    coarse_left, page.height
                )
                coarse_right_intervals = _merged_vertical_intervals(
                    coarse_right, page.height
                )
                if (
                    _overlap_length(
                        coarse_left_intervals, coarse_right_intervals
                    )
                    < page.height * 0.03
                ):
                    return self._single_column(
                        page.page_number, "insufficient_vertical_overlap"
                    )
            return self._single_column(page.page_number, "no_stable_center_gap")
        left_edge, right_edge, gap_support = selected_gap
        edge_tolerance = page.width * 0.01
        left = [
            block for block in candidates if block.bbox.x1 <= left_edge + edge_tolerance
        ]
        right = [
            block for block in candidates if block.bbox.x0 >= right_edge - edge_tolerance
        ]
        if not left or not right:
            return self._single_column(
                page.page_number, "insufficient_bilateral_coverage"
            )

        gap_width = right_edge - left_edge
        if gap_support < page.height * 0.03:
            return self._single_column(page.page_number, "no_stable_center_gap")

        left_intervals = _merged_vertical_intervals(left, page.height)
        right_intervals = _merged_vertical_intervals(right, page.height)
        vertical_overlap = _overlap_length(left_intervals, right_intervals)
        left_vertical_support = _interval_length(left_intervals)
        right_vertical_support = _interval_length(right_intervals)
        smaller_vertical_support = min(left_vertical_support, right_vertical_support)
        if (
            vertical_overlap < page.height * 0.03
            or smaller_vertical_support == 0
            or vertical_overlap / smaller_vertical_support < 0.2
        ):
            return self._single_column(
                page.page_number, "insufficient_vertical_overlap"
            )
        left_characters = sum(_non_whitespace_length(block.text) for block in left)
        right_characters = sum(_non_whitespace_length(block.text) for block in right)
        total_characters = left_characters + right_characters
        smaller_coverage = min(left_characters, right_characters) / total_characters
        if smaller_coverage < 0.2:
            return self._single_column(
                page.page_number, "insufficient_bilateral_coverage"
            )
        if (
            left_vertical_support < page.height * 0.18
            or right_vertical_support < page.height * 0.18
        ):
            return self._single_column(page.page_number, "insufficient_side_coverage")

        left_bound = BBox(
            max(0.0, min(block.bbox.x0 for block in left)),
            0.0,
            min(page.width, left_edge),
            page.height,
        )
        right_bound = BBox(
            max(0.0, right_edge),
            0.0,
            min(page.width, max(block.bbox.x1 for block in right)),
            page.height,
        )
        central_gap = BBox(left_edge, 0.0, right_edge, page.height)
        balance_score = min(1.0, smaller_coverage * 2)
        gap_score = min(1.0, gap_width / (page.width * 0.12))
        stability_score = min(1.0, gap_support / (page.height * 0.18))
        confidence = round((balance_score + gap_score + stability_score) / 3, 3)
        return LayoutDiagnosis(
            page_number=page.page_number,
            column_count=2,
            column_bounds=(left_bound, right_bound),
            confidence=confidence,
            reason_codes=("stable_center_gap", "balanced_text_coverage"),
            central_gap=central_gap,
        )

    @staticmethod
    def _single_column(page_number: int, reason: str) -> LayoutDiagnosis:
        return LayoutDiagnosis(
            page_number=page_number,
            column_count=1,
            confidence=1.0,
            reason_codes=("fallback_single_column", reason),
        )

    def classify_blocks(
        self,
        pages_or_blocks: Sequence[PageLayout] | Sequence[RawBlock] | Sequence[object] | None = None,
        body_stats: BodyStats | None = None,
        ordered_blocks: Sequence[object] | None = None,
        *,
        pages: Sequence[PageLayout] | None = None,
    ) -> tuple[ClassifiedBlock, ...]:
        """Classify extracted blocks using deterministic layout and text evidence.

        The method accepts pages, raw blocks, or the ordered-block objects emitted by
        :class:`ReadingOrderResolver`.  It deliberately retains every input block;
        uncertain text is represented as ``UNKNOWN`` or ``PARAGRAPH``.
        """

        if pages is not None:
            pages_or_blocks = pages
        if (
            ordered_blocks is None
            and body_stats is not None
            and not isinstance(body_stats, BodyStats)
        ):
            ordered_blocks = body_stats  # type: ignore[assignment]
            body_stats = None
        elif isinstance(ordered_blocks, BodyStats):
            body_stats, ordered_blocks = ordered_blocks, body_stats  # type: ignore[assignment]
        if ordered_blocks is not None:
            page_layouts, _ = self._coerce_blocks(pages_or_blocks or ())
            _unused_pages, raw_blocks = self._coerce_blocks(ordered_blocks)
            pages = page_layouts
        else:
            pages, raw_blocks = self._coerce_blocks(pages_or_blocks or ())
        stats = body_stats or compute_body_stats(pages)
        margin_types = self._margin_types(pages, stats)
        if not pages:
            margin_types = self._inferred_margin_types(raw_blocks)
        page_widths = {page.page_number: page.width for page in pages}
        page_heights = {page.page_number: page.height for page in pages}
        first_page = min((block.page_number for block in raw_blocks), default=1)
        first_page_height = page_heights.get(
            first_page,
            max(
                (
                    block.bbox.y1
                    for block in raw_blocks
                    if block.page_number == first_page
                ),
                default=0.0,
            )
            * 1.1,
        )
        for block in raw_blocks:
            if (
                block.page_number == first_page
                and margin_types.get((block.page_number, block.block_index))
                is BlockType.HEADER
                and self._looks_like_title_candidate(block, stats, first_page_height)
            ):
                margin_types.pop((block.page_number, block.block_index), None)
        first_page_candidates = [
            block
            for block in raw_blocks
            if block.page_number == first_page
            and block.text.strip()
            and margin_types.get((block.page_number, block.block_index))
            not in {BlockType.HEADER, BlockType.FOOTER}
        ]
        title_candidate = next(
            (
                block
                for block in first_page_candidates
                if self._looks_like_title_candidate(block, stats, first_page_height)
            ),
            None,
        )
        author_candidate = None
        if title_candidate is not None:
            title_index = first_page_candidates.index(title_candidate)
            author_candidate = next(
                (
                    block
                    for block in first_page_candidates[title_index + 1 :]
                    if self._looks_like_author(block.text.strip())
                    and heading_number_depth(block.text) is None
                    and block.text.casefold().strip() not in _HEADING_WORDS
                ),
                None,
            )

        classified: list[ClassifiedBlock] = []
        reference_mode = False
        for ordinal, block in enumerate(raw_blocks):
            text = block.text
            normalized = " ".join(text.split())
            page_height = page_heights.get(block.page_number, 0.0)
            block_type, confidence, reasons = self._classify_one(
                block,
                normalized,
                stats,
                page_widths.get(block.page_number, 0.0),
                page_height,
                margin_types.get((block.page_number, block.block_index)),
                title_candidate is block,
                author_candidate is block,
                reference_mode,
                pages,
                (
                    block.bbox.y0 - raw_blocks[ordinal - 1].bbox.y1
                    if ordinal
                    else 0.0
                ),
                (
                    raw_blocks[ordinal + 1].bbox.y0 - block.bbox.y1
                    if ordinal + 1 < len(raw_blocks)
                    else 0.0
                ),
            )
            if ordinal:
                gap_before = block.bbox.y0 - raw_blocks[ordinal - 1].bbox.y1
                if stats.common_line_height and gap_before > stats.common_line_height * 1.8:
                    reasons = (*reasons, "large_before_whitespace")
            if ordinal + 1 < len(raw_blocks):
                gap_after = raw_blocks[ordinal + 1].bbox.y0 - block.bbox.y1
                if stats.common_line_height and gap_after > stats.common_line_height * 1.8:
                    reasons = (*reasons, "large_after_whitespace")
            if block_type is BlockType.HEADING and normalized.casefold().rstrip(" :;.-") in {
                "references",
                "bibliography",
            }:
                reference_mode = True
            if (
                reference_mode
                and block_type is BlockType.PARAGRAPH
                and self._looks_like_reference(normalized)
            ):
                block_type = BlockType.REFERENCE
                confidence = max(confidence, 0.72)
                reasons = (*reasons, "reference_mode")
            classified.append(
                ClassifiedBlock(
                    uid=f"block-{block.page_number}-{block.block_index}-{ordinal}",
                    block_type=block_type,
                    page_number=block.page_number,
                    bbox=block.bbox,
                    text=text,
                    confidence=round(min(1.0, max(0.0, confidence)), 3),
                    source_block_indices=(block.block_index,),
                    reason_codes=tuple(dict.fromkeys(reasons)),
                )
            )
        return tuple(classified)

    @staticmethod
    def _coerce_blocks(
        pages_or_blocks: Sequence[PageLayout] | Sequence[RawBlock] | Sequence[object],
    ) -> tuple[tuple[PageLayout, ...], tuple[RawBlock, ...]]:
        if hasattr(pages_or_blocks, "ordered_blocks"):
            items = tuple(pages_or_blocks.ordered_blocks)
        else:
            items = tuple(pages_or_blocks)
        if not items:
            return (), ()
        if isinstance(items[0], PageLayout):
            pages = tuple(item for item in items if isinstance(item, PageLayout))
            return pages, tuple(block for page in pages for block in page.blocks)
        raw_blocks = tuple(
            item.block
            if hasattr(item, "block") and isinstance(item.block, RawBlock)
            else item
            for item in items
            if isinstance(item, RawBlock)
            or (hasattr(item, "block") and isinstance(item.block, RawBlock))
        )
        return (), raw_blocks

    @staticmethod
    def _margin_types(
        pages: Sequence[PageLayout], body_stats: BodyStats
    ) -> dict[tuple[int, int], BlockType]:
        if len(pages) < 2:
            return {}
        detection = RepeatedMarginDetector().detect(pages, body_stats)
        return dict(detection.classifications)

    @staticmethod
    def _inferred_margin_types(
        blocks: Sequence[RawBlock],
    ) -> dict[tuple[int, int], BlockType]:
        if not blocks:
            return {}
        by_page: defaultdict[int, list[RawBlock]] = defaultdict(list)
        for block in blocks:
            by_page[block.page_number].append(block)
        page_heights = {
            page_number: max((block.bbox.y1 for block in page_blocks), default=0.0)
            for page_number, page_blocks in by_page.items()
        }
        labels: defaultdict[tuple[BlockType, str], set[int]] = defaultdict(set)
        candidates: list[tuple[RawBlock, BlockType, str]] = []
        for page_number, page_blocks in by_page.items():
            height = page_heights[page_number]
            for block in page_blocks:
                normalized = _normalize_margin_text(block.text)
                if not normalized or height <= 0:
                    continue
                if block.bbox.y0 <= height * 0.1:
                    block_type = BlockType.HEADER
                elif block.bbox.y1 >= height * 0.9:
                    block_type = BlockType.FOOTER
                else:
                    continue
                candidate = (block_type, normalized)
                labels[candidate].add(page_number)
                candidates.append((block, block_type, normalized))
        threshold = max(1, int(len(by_page) * 0.6 + 0.999))
        repeated = {
            key for key, page_numbers in labels.items() if len(page_numbers) >= threshold
        }
        return {
            (block.page_number, block.block_index): block_type
            for block, block_type, normalized in candidates
            if (block_type, normalized) in repeated
        }

    @staticmethod
    def _maximum_font_size(block: RawBlock) -> float:
        return max(
            (span.size for line in block.lines for span in line.spans if isfinite(span.size)),
            default=0.0,
        )

    @classmethod
    def _looks_like_title_candidate(
        cls, block: RawBlock, stats: BodyStats, page_height: float
    ) -> bool:
        text = " ".join(block.text.split())
        if (
            not text
            or heading_number_depth(text) is not None
            or looks_like_sentence(text)
            or text.casefold().rstrip(":") in _HEADING_WORDS
            or len(text) > 180
            or block.bbox.y0 > page_height * 0.35
        ):
            return False
        size = cls._maximum_font_size(block)
        relative_size = size / stats.median_font_size if stats.median_font_size else 1.0
        return size >= 14 or relative_size >= 1.35

    @staticmethod
    def _font_flags(block: RawBlock) -> tuple[bool, bool]:
        spans = [span for line in block.lines for span in line.spans]
        bold = any(span.flags & 16 or "bold" in span.font.casefold() for span in spans)
        italic = any(span.flags & 2 or "italic" in span.font.casefold() for span in spans)
        return bold, italic

    @classmethod
    def _classify_one(
        cls,
        block: RawBlock,
        normalized: str,
        stats: BodyStats,
        page_width: float,
        page_height: float,
        margin_type: BlockType | None,
        is_title_candidate: bool,
        is_author_candidate: bool,
        reference_mode: bool,
        pages: Sequence[PageLayout],
        before_gap: float,
        after_gap: float,
    ) -> tuple[BlockType, float, tuple[str, ...]]:
        if margin_type in {BlockType.HEADER, BlockType.FOOTER} and not (
            block.page_number == min((p.page_number for p in pages), default=1)
            and (is_title_candidate or is_author_candidate)
        ):
            return margin_type, 0.98, (f"margin_{margin_type.value}",)
        if not normalized:
            return BlockType.UNKNOWN, 0.05, ("empty_text",)

        max_size = cls._maximum_font_size(block)
        relative_size = max_size / stats.median_font_size if stats.median_font_size else 1.0
        bold, _ = cls._font_flags(block)
        line_count = max(1, len(block.lines))
        number_depth = heading_number_depth(normalized)
        sentence_like = looks_like_sentence(normalized)
        lower = normalized.casefold()
        reasons: list[str] = []
        if relative_size >= 1.35:
            reasons.append("large_relative_font")
        elif relative_size >= 1.1:
            reasons.append("elevated_relative_font")
        if bold:
            reasons.append("bold_or_bold_font")
        if line_count == 1:
            reasons.append("single_line")
        if page_height and block.bbox.y0 <= page_height * 0.12:
            reasons.append("top_margin_position")
        elif page_height and block.bbox.y1 >= page_height * 0.88:
            reasons.append("bottom_margin_position")
        centered = (
            page_width > 0
            and block.bbox.width <= page_width * 0.7
            and abs((block.bbox.x0 + block.bbox.x1) / 2 - page_width / 2)
            <= page_width * 0.08
        )
        if centered:
            reasons.append("center_alignment")
        else:
            reasons.append("left_alignment")
        spacing_bonus = 0.0
        if stats.common_line_height:
            if before_gap > stats.common_line_height * 1.8:
                reasons.append("large_before_whitespace")
                spacing_bonus += 0.04
            if after_gap > stats.common_line_height * 1.8:
                reasons.append("large_after_whitespace")
                spacing_bonus += 0.03
        if number_depth is not None:
            reasons.append(f"heading_number_depth_{number_depth}")
        if sentence_like:
            reasons.append("sentence_punctuation_or_length")

        # Explicit page elements take precedence over semantic labels.
        region_type = cls._element_region_type(block, pages)
        if region_type is not None:
            return region_type, 0.88, (*reasons, "element_region")

        caption_match = _CAPTION.match(normalized)
        if caption_match:
            caption_type = (
                BlockType.FIGURE_CAPTION
                if caption_match.group(1).casefold().startswith(("fig", "figure"))
                else BlockType.TABLE_CAPTION
            )
            return caption_type, 0.93, (*reasons, "caption_label")

        if _NUMBERED_LIST.match(normalized):
            return BlockType.LIST, 0.9, (*reasons, "numbered_list_marker")

        if cls._is_equation(normalized, block):
            return BlockType.EQUATION, 0.84, (*reasons, "equation_symbols_or_layout")
        if cls._is_table(normalized, block):
            return BlockType.TABLE, 0.82, (*reasons, "table_grid_or_columns")

        first_page = min((p.page_number for p in pages), default=1)
        if is_title_candidate and block.page_number == first_page:
            if (
                (relative_size >= 1.35 or max_size == 0)
                and line_count <= 3
                and len(normalized) <= 180
                and not sentence_like
            ):
                return (
                    BlockType.TITLE,
                    min(0.99, 0.78 + 0.08 * relative_size),
                    (*reasons, "first_page_title_position"),
                )
        if is_author_candidate and block.page_number == first_page:
            if cls._looks_like_author(normalized) and not sentence_like:
                return BlockType.AUTHOR, 0.81, (*reasons, "first_page_author_position")
        label = lower.rstrip(" :;.-")
        if label in {"abstract", "summary"}:
            return BlockType.ABSTRACT, 0.96, (*reasons, "known_abstract_label")

        known_heading = label in _HEADING_WORDS
        alignment_bonus = 0.07 if centered else 0.0
        if number_depth is not None and not sentence_like and len(normalized) <= 180:
            heading_type = BlockType.HEADING if number_depth == 1 else BlockType.SUBHEADING
            return (
                heading_type,
                min(
                    0.95,
                    0.68
                    + 0.08 * max(relative_size, 1.0)
                    + alignment_bonus
                    + spacing_bonus,
                ),
                (*reasons, "numbered_heading"),
            )
        if known_heading and len(normalized) <= 120:
            return (
                BlockType.HEADING,
                min(
                    0.96,
                    0.72
                    + 0.08 * max(relative_size, 1.0)
                    + alignment_bonus
                    + spacing_bonus,
                ),
                (*reasons, "known_heading_lexicon"),
            )
        if (
            relative_size >= 1.12
            and bold
            and line_count <= 2
            and len(normalized) <= 140
            and not sentence_like
        ):
            if relative_size < 1.4:
                reasons.append("font_rank_subheading")
            return (
                BlockType.SUBHEADING if relative_size < 1.35 else BlockType.HEADING,
                min(0.95, 0.76 + alignment_bonus + spacing_bonus),
                (*reasons, "font_rank_and_short_line"),
            )

        if cls._is_footnote(block, normalized, stats, page_height):
            return BlockType.FOOTNOTE, 0.72, (*reasons, "small_bottom_margin_text")
        if _BULLET.match(normalized):
            return BlockType.LIST, 0.9, (*reasons, "bullet_or_list_marker")
        if cls._looks_like_reference(normalized, allow_author_year=reference_mode):
            return BlockType.REFERENCE, 0.84, (*reasons, "strong_reference_citation")
        if sentence_like or len(normalized) > 90:
            return BlockType.PARAGRAPH, 0.7, (*reasons, "sentence_or_long_text")
        if re.search(r"\b(?:19|20)\d{2}\b", normalized):
            return BlockType.PARAGRAPH, 0.45, (*reasons, "year_without_citation_shape")
        return BlockType.UNKNOWN, 0.35, (*reasons, "insufficient_structural_evidence")

    @staticmethod
    def _element_region_type(block: RawBlock, pages: Sequence[PageLayout]) -> BlockType | None:
        page = next((item for item in pages if item.page_number == block.page_number), None)
        if page is None or not page.image_regions:
            return None
        for region in page.image_regions:
            intersection = max(
                0.0,
                min(block.bbox.x1, region.bbox.x1)
                - max(block.bbox.x0, region.bbox.x0),
            ) * max(
                0.0,
                min(block.bbox.y1, region.bbox.y1)
                - max(block.bbox.y0, region.bbox.y0),
            )
            if intersection > 0 and intersection / max(block.bbox.area, 1.0) >= 0.2:
                return BlockType.FIGURE
        return None

    @staticmethod
    def _is_equation(text: str, block: RawBlock) -> bool:
        symbols = sum(text.count(symbol) for symbol in _EQUATION_SYMBOLS)
        compact = re.sub(r"\s+", "", text)
        return (
            (
                symbols >= 1
                and len(compact) <= 80
                and len(re.findall(r"[A-Za-z]", compact)) < len(compact) * 0.7
            )
            or bool(_EQUATION_LABEL.fullmatch(text))
            or (symbols >= 2 and len(block.lines) <= 2 and len(text) <= 140)
        )

    @staticmethod
    def _is_table(text: str, block: RawBlock) -> bool:
        if "|" in text:
            return True
        if len(block.lines) < 2:
            return False
        rows = [line.text for line in block.lines if line.text.strip()]
        if len(rows) < 2:
            return False
        separated = [re.split(r"\s{2,}|\t+", row.strip()) for row in rows]
        return all(len(columns) >= 2 for columns in separated)

    @staticmethod
    def _looks_like_author(text: str) -> bool:
        words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)
        if not 2 <= len(words) <= 12 or any(char in text for char in ".!?;:()[]"):
            return False
        connectors = {"and", "et", "al"}
        name_words = [word for word in words if word.casefold() not in connectors]
        return bool(
            name_words
            and all(
                word[0].isupper() or (len(word) <= 2 and word.isupper())
                for word in name_words
            )
        )

    @staticmethod
    def _looks_like_reference(text: str, *, allow_author_year: bool = True) -> bool:
        if (
            re.match(r"^\s*\[\d+\](?:\s|$)", text)
            or re.match(r"^\s*doi\s*:\s*\S+", text, flags=re.I)
            or re.match(r"^\s*https?://doi\.org/\S+", text, flags=re.I)
        ):
            return True
        return allow_author_year and bool(_AUTHOR_YEAR_REFERENCE.match(text))

    @staticmethod
    def _is_footnote(
        block: RawBlock, text: str, stats: BodyStats, page_height: float
    ) -> bool:
        if page_height <= 0 or block.bbox.y0 < page_height * 0.82:
            return False
        size = LayoutAnalyzer._maximum_font_size(block)
        small = stats.median_font_size <= 0 or size <= stats.median_font_size * 0.86
        return small and bool(re.match(r"^(?:[*†‡]|\d+\s+)", text))


@dataclass(frozen=True)
class OrderedBlock:
    block: RawBlock
    page_order: int
    document_order: int | None = None

    @property
    def text(self) -> str:
        return self.block.text

    @property
    def page_number(self) -> int:
        return self.block.page_number

    @property
    def block_index(self) -> int:
        return self.block.block_index

    @property
    def bbox(self) -> BBox:
        return self.block.bbox


@dataclass(frozen=True)
class ReadingOrderResult:
    ordered_blocks: tuple[OrderedBlock, ...] = ()
    excluded_blocks: tuple[RawBlock, ...] = ()
    page_number: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordered_blocks", tuple(self.ordered_blocks))
        object.__setattr__(self, "excluded_blocks", tuple(self.excluded_blocks))

    def __iter__(self) -> Iterator[OrderedBlock]:
        return iter(self.ordered_blocks)

    def __len__(self) -> int:
        return len(self.ordered_blocks)

    def __getitem__(self, index: int | slice) -> OrderedBlock | tuple[OrderedBlock, ...]:
        return self.ordered_blocks[index]

    @property
    def blocks(self) -> tuple[OrderedBlock, ...]:
        return self.ordered_blocks


def _reading_sort_key(block: RawBlock) -> tuple[float, float, float, float, int, str]:
    return (
        block.bbox.y0,
        block.bbox.y1,
        block.bbox.x0,
        block.bbox.x1,
        block.block_index,
        block.text,
    )


@dataclass(frozen=True)
class _BlockOccurrence:
    position: int
    block: RawBlock


def _occurrence_sort_key(
    occurrence: _BlockOccurrence,
) -> tuple[float, float, float, float, int, str, int]:
    return (*_reading_sort_key(occurrence.block), occurrence.position)


class ReadingOrderResolver:
    def resolve(
        self,
        page: PageLayout,
        diagnosis: LayoutDiagnosis,
        excluded: Collection[object]
        | Mapping[object, object]
        | MarginDetection
        | None = None,
    ) -> ReadingOrderResult:
        excluded_keys = _exclusion_keys(excluded)
        excluded_blocks = tuple(
            sorted(
                (block for block in page.blocks if _block_key(block) in excluded_keys),
                key=_reading_sort_key,
            )
        )
        body_blocks = [
            block for block in page.blocks if _block_key(block) not in excluded_keys
        ]
        if diagnosis.column_count != 2:
            ordered_raw = sorted(body_blocks, key=_reading_sort_key)
        else:
            ordered_raw = self._resolve_two_column(page, body_blocks, diagnosis)
        ordered = tuple(
            OrderedBlock(block=block, page_order=index)
            for index, block in enumerate(ordered_raw)
        )
        return ReadingOrderResult(
            ordered_blocks=ordered,
            excluded_blocks=excluded_blocks,
            page_number=page.page_number,
        )

    def _resolve_two_column(
        self,
        page: PageLayout,
        blocks: Sequence[RawBlock],
        diagnosis: LayoutDiagnosis,
    ) -> list[RawBlock]:
        if diagnosis.central_gap is not None:
            gap_left = diagnosis.central_gap.x0
            gap_right = diagnosis.central_gap.x1
        elif len(diagnosis.column_bounds) >= 2:
            gap_left = diagnosis.column_bounds[0].x1
            gap_right = diagnosis.column_bounds[1].x0
        else:
            gap_left = gap_right = page.width / 2
        gap_midpoint = (gap_left + gap_right) / 2
        occurrences = [
            _BlockOccurrence(position, block) for position, block in enumerate(blocks)
        ]
        spanning = sorted(
            (
                occurrence
                for occurrence in occurrences
                if (
                    occurrence.block.bbox.x0 < gap_left
                    and occurrence.block.bbox.x1 > gap_right
                )
                or occurrence.block.bbox.width >= page.width * 0.65
            ),
            key=_occurrence_sort_key,
        )
        spanning_positions = {occurrence.position for occurrence in spanning}
        remaining = [
            occurrence
            for occurrence in occurrences
            if occurrence.position not in spanning_positions
        ]
        ordered: list[RawBlock] = []
        for boundary in spanning:
            boundary_center = (
                boundary.block.bbox.y0 + boundary.block.bbox.y1
            ) / 2
            band = [
                occurrence
                for occurrence in remaining
                if (occurrence.block.bbox.y0 + occurrence.block.bbox.y1) / 2
                < boundary_center
            ]
            band_positions = {occurrence.position for occurrence in band}
            remaining = [
                occurrence
                for occurrence in remaining
                if occurrence.position not in band_positions
            ]
            ordered.extend(self._order_band(band, gap_midpoint))
            ordered.append(boundary.block)
        ordered.extend(self._order_band(remaining, gap_midpoint))
        return ordered

    @staticmethod
    def _order_band(
        occurrences: Sequence[_BlockOccurrence], gap_midpoint: float
    ) -> list[RawBlock]:
        left = [
            occurrence
            for occurrence in occurrences
            if (occurrence.block.bbox.x0 + occurrence.block.bbox.x1) / 2
            <= gap_midpoint
        ]
        right = [
            occurrence
            for occurrence in occurrences
            if (occurrence.block.bbox.x0 + occurrence.block.bbox.x1) / 2
            > gap_midpoint
        ]
        return [
            occurrence.block
            for occurrence in (
                sorted(left, key=_occurrence_sort_key)
                + sorted(right, key=_occurrence_sort_key)
            )
        ]

    def resolve_document(
        self,
        pages: Sequence[PageLayout],
        diagnoses: Mapping[int, LayoutDiagnosis] | Sequence[LayoutDiagnosis] | None = None,
        excluded: Collection[object]
        | Mapping[object, object]
        | MarginDetection
        | None = None,
    ) -> ReadingOrderResult:
        if diagnoses is None:
            diagnosis_by_page: dict[int, LayoutDiagnosis] = {}
        elif isinstance(diagnoses, Mapping):
            diagnosis_by_page = dict(diagnoses)
        else:
            diagnosis_by_page = {
                diagnosis.page_number: diagnosis for diagnosis in diagnoses
            }

        ordered: list[OrderedBlock] = []
        excluded_blocks: list[RawBlock] = []
        for page in sorted(pages, key=lambda item: item.page_number):
            diagnosis = diagnosis_by_page.get(page.page_number)
            if diagnosis is None:
                diagnosis = LayoutAnalyzer().detect_columns(page, excluded)
            page_result = self.resolve(page, diagnosis, excluded)
            excluded_blocks.extend(page_result.excluded_blocks)
            for block in page_result:
                ordered.append(
                    OrderedBlock(
                        block=block.block,
                        page_order=block.page_order,
                        document_order=len(ordered),
                    )
                )
        return ReadingOrderResult(
            ordered_blocks=tuple(ordered),
            excluded_blocks=tuple(excluded_blocks),
        )
