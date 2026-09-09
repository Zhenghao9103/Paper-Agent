import pytest
from backend.app.ingestion.domain import (
    BBox,
    BlockType,
    LayoutDiagnosis,
    PageLayout,
    RawBlock,
    RawLine,
    RawSpan,
)
from backend.app.ingestion.layout import (
    LayoutAnalyzer,
    ReadingOrderResolver,
    RepeatedMarginDetector,
    compute_body_stats,
)


def _block(
    page_number: int,
    block_index: int,
    bbox: tuple[float, float, float, float],
    text: str,
    *,
    font: str = "Body",
    size: float = 10.0,
    line_height: float | None = None,
) -> RawBlock:
    box = BBox(*bbox)
    height = line_height if line_height is not None else box.height
    span_box = BBox(box.x0, box.y0, box.x1, min(box.y0 + height, box.y1))
    span = RawSpan(text=text, bbox=span_box, font=font, size=size)
    line = RawLine(text=text, bbox=span_box, spans=(span,))
    return RawBlock(
        page_number=page_number,
        block_index=block_index,
        bbox=box,
        text=text,
        lines=(line,),
    )


def _page(
    page_number: int,
    *blocks: RawBlock,
    width: float = 600.0,
    height: float = 800.0,
) -> PageLayout:
    return PageLayout(
        page_number=page_number,
        width=width,
        height=height,
        blocks=tuple(blocks),
    )


def test_body_stats_are_document_wide_and_weighted_by_non_whitespace_characters() -> None:
    pages = [
        _page(
            1,
            _block(1, 0, (60, 100, 250, 112), "abcdefghij", font="Alpha", size=9),
            _block(1, 1, (60, 130, 250, 144), "12345", font="Beta", size=14),
        ),
        _page(
            2,
            _block(2, 0, (60, 100, 250, 112), "klmno", font="Beta", size=12),
            _block(2, 1, (60, 130, 250, 142), "   ", font="Ignored", size=40),
        ),
    ]

    stats = compute_body_stats(pages)

    assert stats.dominant_fonts == ("Alpha", "Beta")
    assert stats.median_font_size == 9.0
    assert stats.common_line_height == 12.0
    assert stats.span_count == 3
    assert stats.character_count == 20


def test_body_stats_for_empty_pages_are_finite_and_deterministic() -> None:
    stats = compute_body_stats([_page(1), _page(2)])

    assert stats.dominant_fonts == ()
    assert stats.median_font_size == 0.0
    assert stats.common_line_height == 0.0
    assert stats.span_count == 0
    assert stats.character_count == 0


def test_common_line_height_uses_nonempty_spans_when_raw_line_text_is_empty() -> None:
    span = RawSpan(
        text="visible span",
        bbox=BBox(60, 100, 180, 114),
        font="Body",
        size=10,
    )
    line = RawLine(
        text="",
        bbox=BBox(60, 100, 180, 114),
        spans=(span,),
    )
    block = RawBlock(
        page_number=1,
        block_index=0,
        bbox=line.bbox,
        text="visible span",
        lines=(line,),
    )

    stats = compute_body_stats([_page(1, block)])

    assert stats.common_line_height == 14.0


def test_repeated_headers_and_changing_page_numbers_are_detected_without_removal() -> None:
    pages = []
    header_texts = (" Proceedings   2026 ", "proceedings 2026", "PROCEEDINGS 2026")
    for page_number, header_text in enumerate(header_texts, start=1):
        blocks = [
            _block(page_number, 0, (50, 20, 550, 32), header_text, size=8),
            _block(page_number, 1, (60, 150, 540, 180), f"Body {page_number}", size=10),
            _block(page_number, 2, (290, 760, 310, 772), str(page_number), size=8),
        ]
        if page_number == 1:
            blocks.append(
                _block(1, 3, (50, 45, 550, 75), "Unique Paper Title", size=24)
            )
        pages.append(_page(page_number, *blocks))

    detection = RepeatedMarginDetector().detect(pages)

    for page_number in range(1, 4):
        assert detection.classifications[(page_number, 0)] is BlockType.HEADER
        assert detection.classifications[(page_number, 1)] is BlockType.UNKNOWN
        assert detection.classifications[(page_number, 2)] is BlockType.FOOTER
    assert detection.classifications[(1, 3)] is BlockType.UNKNOWN
    assert detection.normalized_recurrence["proceedings 2026"] == 1.0
    assert detection.normalized_recurrence["<page-number>"] == 1.0
    assert len(detection.retained_blocks) == 10
    assert "repeated_header" in detection.reason_codes
    assert "repeated_footer" in detection.reason_codes


def test_margin_candidate_below_recurrence_threshold_remains_body() -> None:
    pages = [
        _page(
            page_number,
            _block(page_number, 0, (50, 20, 550, 32), "Only twice", size=8)
            if page_number <= 2
            else _block(page_number, 0, (60, 150, 540, 180), "Body", size=10),
        )
        for page_number in range(1, 5)
    ]

    detection = RepeatedMarginDetector().detect(pages)

    assert all(
        detection.classifications[(page_number, 0)] is BlockType.UNKNOWN
        for page_number in range(1, 5)
    )


def test_two_column_diagnosis_reports_stable_gap_balanced_coverage_and_geometry() -> None:
    blocks = [
        _block(1, index, (60, 100 + index * 70, 270, 155 + index * 70), label)
        for index, label in enumerate(("A", "B", "C"))
    ]
    blocks.extend(
        _block(1, index + 3, (330, 100 + index * 70, 540, 155 + index * 70), label)
        for index, label in enumerate(("D", "E", "F"))
    )

    diagnosis = LayoutAnalyzer().detect_columns(_page(1, *blocks))

    assert diagnosis.column_count == 2
    assert diagnosis.layout_type == "two_column"
    assert 0.0 <= diagnosis.confidence <= 1.0
    assert diagnosis.confidence > 0.5
    assert "stable_center_gap" in diagnosis.reason_codes
    assert "balanced_text_coverage" in diagnosis.reason_codes
    assert len(diagnosis.column_bounds) == 2
    assert diagnosis.central_gap == BBox(270, 0, 330, 800)
    assert all(bound.contains_within(600, 800) for bound in diagnosis.column_bounds)


def test_one_column_page_uses_explicit_fallback_reason() -> None:
    page = _page(
        1,
        _block(1, 0, (60, 100, 540, 180), "A full width body paragraph"),
        _block(1, 1, (60, 210, 540, 290), "Another full width paragraph"),
    )

    diagnosis = LayoutAnalyzer().detect_columns(page)

    assert diagnosis.column_count == 1
    assert diagnosis.layout_type == "single_column"
    assert diagnosis.confidence == 1.0
    assert diagnosis.central_gap is None
    assert diagnosis.reason_codes == ("fallback_single_column", "no_column_candidates")


def test_isolated_side_note_is_not_enough_for_two_columns() -> None:
    page = _page(
        1,
        _block(
            1,
            0,
            (60, 100, 270, 300),
            "Substantial left-column body content " * 8,
        ),
        _block(1, 1, (360, 120, 520, 150), "note"),
    )

    diagnosis = LayoutAnalyzer().detect_columns(page)

    assert diagnosis.column_count == 1
    assert "fallback_single_column" in diagnosis.reason_codes
    assert "insufficient_bilateral_coverage" in diagnosis.reason_codes


def test_bilateral_blocks_without_vertical_overlap_are_not_two_columns() -> None:
    page = _page(
        1,
        _block(1, 0, (60, 100, 270, 125), "Left upper one"),
        _block(1, 1, (60, 135, 270, 160), "Left upper two"),
        _block(1, 2, (330, 600, 540, 625), "Right lower one"),
        _block(1, 3, (330, 635, 540, 660), "Right lower two"),
    )

    diagnosis = LayoutAnalyzer().detect_columns(page)

    assert diagnosis.column_count == 1
    assert diagnosis.reason_codes == (
        "fallback_single_column",
        "insufficient_vertical_overlap",
    )


def test_isolated_center_intruder_does_not_erase_a_stable_main_column_gap() -> None:
    blocks = [
        _block(1, index, (60, 100 + index * 80, 270, 160 + index * 80), label)
        for index, label in enumerate(("A", "B", "C"))
    ]
    blocks.extend(
        _block(1, index + 3, (310, 100 + index * 80, 540, 160 + index * 80), label)
        for index, label in enumerate(("D", "E", "F"))
    )
    blocks.append(_block(1, 6, (275, 190, 295, 210), "center note"))

    diagnosis = LayoutAnalyzer().detect_columns(_page(1, *blocks))

    assert diagnosis.column_count == 2
    assert diagnosis.central_gap == BBox(270, 0, 310, 800)
    assert "stable_center_gap" in diagnosis.reason_codes


def test_sparse_aligned_bilateral_blocks_have_insufficient_side_coverage() -> None:
    page = _page(
        1,
        _block(1, 0, (60, 100, 270, 112), "A"),
        _block(1, 1, (60, 130, 270, 142), "B"),
        _block(1, 2, (330, 100, 540, 112), "D"),
        _block(1, 3, (330, 130, 540, 142), "E"),
    )

    diagnosis = LayoutAnalyzer().detect_columns(page)

    assert diagnosis.column_count == 1
    assert diagnosis.reason_codes == (
        "fallback_single_column",
        "insufficient_side_coverage",
    )


def test_three_sparse_rows_do_not_gain_side_coverage_from_gap_bridging() -> None:
    page = _page(
        1,
        _block(1, 0, (60, 100, 270, 110), "A"),
        _block(1, 1, (60, 200, 270, 210), "B"),
        _block(1, 2, (60, 300, 270, 310), "C"),
        _block(1, 3, (330, 100, 540, 110), "D"),
        _block(1, 4, (330, 200, 540, 210), "E"),
        _block(1, 5, (330, 300, 540, 310), "F"),
    )

    diagnosis = LayoutAnalyzer().detect_columns(page)

    assert diagnosis.column_count == 1
    assert diagnosis.reason_codes == (
        "fallback_single_column",
        "insufficient_side_coverage",
    )


def test_empty_page_column_diagnosis_is_finite_and_deterministic() -> None:
    diagnosis = LayoutAnalyzer().detect_columns(_page(7))

    assert diagnosis.page_number == 7
    assert diagnosis.column_count == 1
    assert diagnosis.confidence == 1.0
    assert diagnosis.column_bounds == ()


def test_two_column_reading_order_finishes_left_before_right() -> None:
    blocks = [
        _block(1, index, (60, 100 + index * 70, 270, 155 + index * 70), label)
        for index, label in enumerate(("A", "B", "C"))
    ]
    blocks.extend(
        _block(1, index + 3, (330, 100 + index * 70, 540, 155 + index * 70), label)
        for index, label in enumerate(("D", "E", "F"))
    )
    page = _page(1, *blocks)
    diagnosis = LayoutAnalyzer().detect_columns(page)

    result = ReadingOrderResolver().resolve(page, diagnosis)

    assert [block.text for block in result] == ["A", "B", "C", "D", "E", "F"]
    assert [block.page_order for block in result] == list(range(6))
    assert all(block.document_order is None for block in result)


def test_two_column_detection_accepts_one_tall_text_block_per_column() -> None:
    page = _page(
        1,
        _block(1, 0, (60, 100, 270, 650), "Left column body " * 80),
        _block(1, 1, (330, 100, 540, 650), "Right column body " * 80),
    )

    diagnosis = LayoutAnalyzer().detect_columns(page)
    result = ReadingOrderResolver().resolve(page, diagnosis)

    assert diagnosis.column_count == 2
    assert [block.text for block in result] == [page.blocks[0].text, page.blocks[1].text]


def test_full_width_heading_interrupts_upper_and_lower_column_bands() -> None:
    page = _page(
        1,
        _block(1, 0, (60, 60, 270, 120), "A"),
        _block(1, 1, (60, 130, 270, 190), "B"),
        _block(1, 2, (330, 60, 540, 120), "D"),
        _block(1, 3, (330, 130, 540, 190), "E"),
        _block(1, 4, (60, 230, 540, 265), "Full-width heading", size=16),
        _block(1, 5, (60, 300, 270, 360), "C"),
        _block(1, 6, (330, 300, 540, 360), "F"),
    )
    diagnosis = LayoutAnalyzer().detect_columns(page)

    result = ReadingOrderResolver().resolve(page, diagnosis)

    assert diagnosis.column_count == 2
    assert [block.text for block in result] == [
        "A",
        "B",
        "D",
        "E",
        "Full-width heading",
        "C",
        "F",
    ]


def test_reading_order_excludes_margins_from_body_and_retains_them_separately() -> None:
    page = _page(
        1,
        _block(1, 0, (60, 20, 540, 35), "Header", size=8),
        _block(1, 1, (60, 100, 270, 180), "A"),
        _block(1, 2, (60, 200, 270, 280), "B"),
        _block(1, 3, (330, 100, 540, 180), "D"),
        _block(1, 4, (330, 200, 540, 280), "E"),
        _block(1, 5, (290, 760, 310, 775), "1", size=8),
    )
    exclusions = {(1, 0): BlockType.HEADER, (1, 5): BlockType.FOOTER}
    diagnosis = LayoutAnalyzer().detect_columns(page, exclusions)

    result = ReadingOrderResolver().resolve(page, diagnosis, exclusions)

    assert [block.text for block in result] == ["A", "B", "D", "E"]
    assert [block.text for block in result.excluded_blocks] == ["Header", "1"]


def test_document_reading_order_assigns_monotonic_global_indices() -> None:
    pages = [
        _page(2, _block(2, 0, (60, 100, 540, 130), "Second page")),
        _page(1, _block(1, 0, (60, 100, 540, 130), "First page")),
    ]
    diagnoses = {
        page.page_number: LayoutDiagnosis(
            page_number=page.page_number,
            confidence=1.0,
            reason_codes=("fallback_single_column",),
        )
        for page in pages
    }

    result = ReadingOrderResolver().resolve_document(pages, diagnoses)

    assert [block.text for block in result] == ["First page", "Second page"]
    assert [block.page_order for block in result] == [0, 0]
    assert [block.document_order for block in result] == [0, 1]


def test_reading_order_preserves_duplicate_source_keys_by_occurrence() -> None:
    page = _page(
        1,
        _block(1, 7, (60, 100, 270, 180), "Body"),
        _block(1, 8, (330, 100, 540, 180), "Right"),
        _block(1, 7, (60, 230, 540, 265), "Heading", size=16),
    )
    diagnosis = LayoutDiagnosis(
        page_number=1,
        column_count=2,
        column_bounds=(BBox(60, 0, 270, 800), BBox(330, 0, 540, 800)),
        confidence=0.9,
        reason_codes=("stable_center_gap",),
        central_gap=BBox(270, 0, 330, 800),
    )

    result = ReadingOrderResolver().resolve(page, diagnosis)

    assert [block.text for block in result] == ["Body", "Right", "Heading"]
    assert [block.page_order for block in result] == [0, 1, 2]


def test_layout_diagnosis_snapshots_caller_owned_sequences() -> None:
    bounds = [BBox(60, 0, 270, 800), BBox(330, 0, 540, 800)]
    reasons = ["stable_center_gap"]
    diagnosis = LayoutDiagnosis(
        page_number=1,
        column_count=2,
        column_bounds=bounds,
        reason_codes=reasons,
        confidence=0.9,
        central_gap=BBox(270, 0, 330, 800),
    )

    bounds.clear()
    reasons.append("mutated")

    assert diagnosis.column_bounds == (
        BBox(60, 0, 270, 800),
        BBox(330, 0, 540, 800),
    )
    assert diagnosis.reason_codes == ("stable_center_gap",)


@pytest.mark.parametrize("column_count", [-1, 0, 3])
def test_layout_diagnosis_rejects_unsupported_column_count(column_count: int) -> None:
    with pytest.raises(ValueError, match="column_count must be 1 or 2"):
        LayoutDiagnosis(page_number=1, column_count=column_count)


def test_layout_diagnosis_rejects_incoherent_two_column_bounds() -> None:
    with pytest.raises(ValueError, match="two-column layout requires two column bounds"):
        LayoutDiagnosis(page_number=1, column_count=2)

    with pytest.raises(ValueError, match="column bounds must be ordered and non-overlapping"):
        LayoutDiagnosis(
            page_number=1,
            column_count=2,
            column_bounds=(BBox(60, 0, 350, 800), BBox(330, 0, 540, 800)),
        )
