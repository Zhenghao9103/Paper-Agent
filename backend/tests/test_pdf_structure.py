from __future__ import annotations

import pytest
from backend.app.ingestion.domain import (
    BBox,
    BlockType,
    ElementParseStatus,
    ElementType,
    PageLayout,
    ParsedElement,
    RawBlock,
    RawLine,
    RawSpan,
    StructuredChunk,
)
from backend.app.ingestion.layout import (
    BodyStats,
    LayoutAnalyzer,
    MarginDetection,
    OrderedBlock,
    ReadingOrderResult,
    compute_body_stats,
    heading_number_depth,
    looks_like_sentence,
)
from backend.app.ingestion.structure import DocumentStructureParser


def _block(
    page: int,
    index: int,
    text: str,
    *,
    y: float,
    size: float = 10,
    font: str = "Body",
    x0: float = 60,
    x1: float = 540,
) -> RawBlock:
    bbox = BBox(x0, y, x1, y + 18)
    span = RawSpan(
        text=text,
        bbox=bbox,
        font=font,
        size=size,
        flags=16 if "bold" in font.lower() else 0,
    )
    line = RawLine(text=text, bbox=bbox, spans=(span,))
    return RawBlock(page, index, bbox, text, (line,))


def _page(*blocks: RawBlock) -> PageLayout:
    return PageLayout(1, 600, 800, tuple(blocks))


def test_heading_helpers_distinguish_numbered_depth_and_sentence() -> None:
    assert heading_number_depth("3 Method") == 1
    assert heading_number_depth("3.1 Encoder") == 2
    assert heading_number_depth("3.1.2 Details") == 3
    assert heading_number_depth("3 This sentence explains the method.") == 1
    assert looks_like_sentence("3 This sentence explains the method.")
    assert not looks_like_sentence("3 Method")


def test_classify_blocks_uses_precedence_and_preserves_uncertain_text() -> None:
    blocks = (
        _block(1, 0, "A Study of Encoders", y=45, size=20, font="Helvetica-Bold"),
        _block(1, 1, "Alice Smith", y=75, size=11),
        _block(1, 2, "Abstract", y=110, size=14, font="Helvetica-Bold"),
        _block(1, 3, "We study robust encoders.", y=140),
        _block(1, 4, "3 Method", y=190, size=14, font="Helvetica-Bold"),
        _block(1, 5, "3 This sentence explains the method.", y=220),
        _block(1, 6, "References", y=270, size=14, font="Helvetica-Bold"),
        _block(1, 7, "[1] Doe, A. A paper. 2024.", y=300, size=9),
        _block(1, 8, "   ", y=350),
    )
    page = _page(*blocks)
    result = LayoutAnalyzer().classify_blocks([page], compute_body_stats([page]))
    by_text = {item.text: item for item in result}

    assert by_text["A Study of Encoders"].block_type is BlockType.TITLE
    assert by_text["Alice Smith"].block_type is BlockType.AUTHOR
    assert by_text["Abstract"].block_type is BlockType.ABSTRACT
    assert by_text["3 Method"].block_type is BlockType.HEADING
    assert by_text["3 This sentence explains the method."].block_type is BlockType.PARAGRAPH
    assert by_text["References"].block_type is BlockType.HEADING
    assert by_text["[1] Doe, A. A paper. 2024."].block_type is BlockType.REFERENCE
    assert by_text["   "].block_type in {BlockType.UNKNOWN, BlockType.PARAGRAPH}
    assert all(0 <= item.confidence <= 1 for item in result)
    assert all(isinstance(item.reason_codes, tuple) for item in result)


def test_structure_parser_builds_sections_and_assigns_active_section() -> None:
    raw = (
        _block(1, 0, "1 Introduction", y=50, size=15, font="Helvetica-Bold"),
        _block(1, 1, "Intro body", y=80),
        _block(1, 2, "3 Method", y=120, size=15, font="Helvetica-Bold"),
        _block(1, 3, "Method body", y=150),
        _block(1, 4, "3.1 Encoder", y=190, size=13, font="Helvetica-Bold"),
        _block(1, 5, "Encoder body", y=220),
        _block(1, 6, "3.2 Spectral Learning", y=260, size=13, font="Helvetica-Bold"),
        _block(1, 7, "Spectral body", y=290),
    )
    page = _page(*raw)
    classified = LayoutAnalyzer().classify_blocks([page], compute_body_stats([page]))
    roots, enriched = DocumentStructureParser().parse(classified)

    assert [root.title for root in roots] == ["1 Introduction", "3 Method"]
    method = roots[1]
    assert [child.title for child in method.children] == ["3.1 Encoder", "3.2 Spectral Learning"]
    assert enriched[1].section == "1 Introduction"
    assert enriched[3].section == "3 Method"
    assert enriched[5].section == "3 Method"
    assert enriched[5].subsection == "3.1 Encoder"
    assert enriched[7].subsection == "3.2 Spectral Learning"
    assert all(item.uid for item in enriched)


def test_structure_parser_handles_abstract_references_and_hierarchy_jumps() -> None:
    raw = (
        _block(1, 0, "Abstract", y=50, size=15, font="Helvetica-Bold"),
        _block(1, 1, "Summary body", y=80),
        _block(1, 2, "3.1 Too deep", y=120, size=13, font="Helvetica-Bold"),
        _block(1, 3, "Body", y=150),
        _block(1, 4, "References", y=190, size=15, font="Helvetica-Bold"),
        _block(1, 5, "[1] Smith. 2020.", y=220, size=9),
    )
    page = _page(*raw)
    classified = LayoutAnalyzer().classify_blocks([page], compute_body_stats([page]))
    roots, enriched = DocumentStructureParser().parse(classified)

    assert roots[0].title == "3.1 Too deep"
    assert any("hierarchy_jump" in code for code in enriched[2].reason_codes)
    assert enriched[5].block_type is BlockType.REFERENCE
    assert enriched[5].section == "References"


def test_structure_parser_assigns_context_to_following_elements() -> None:
    blocks = [
        {
            "uid": "heading-1",
            "block_type": BlockType.HEADING,
            "page_number": 1,
            "bbox": BBox(60, 100, 540, 118),
            "text": "3 Method",
        },
        {
            "uid": "heading-2",
            "block_type": BlockType.SUBHEADING,
            "page_number": 1,
            "bbox": BBox(60, 140, 540, 158),
            "text": "3.2 Spectral Learning",
        },
    ]
    classified = [
        __import__("backend.app.ingestion.domain", fromlist=["ClassifiedBlock"]).ClassifiedBlock(
            **payload
        )
        for payload in blocks
    ]
    element = ParsedElement(
        uid="element-1",
        element_type=ElementType.FIGURE,
        status=ElementParseStatus.SUCCESS,
        page_number=1,
        bbox=BBox(60, 180, 540, 320),
    )

    result = DocumentStructureParser().parse_with_elements(classified, elements=(element,))
    roots, enriched_blocks, enriched_elements = result

    assert roots[0].title == "3 Method"
    assert enriched_blocks[-1].subsection == "3.2 Spectral Learning"
    assert enriched_elements[0].section == "3 Method"
    assert enriched_elements[0].subsection == "3.2 Spectral Learning"
    assert enriched_elements[0].section_uid == roots[0].uid


def test_heading_score_uses_alignment_and_vertical_spacing_evidence() -> None:
    blocks = (
        _block(1, 0, "Paper title", y=20, size=20, font="Helvetica-Bold"),
        _block(1, 1, "Alice Smith", y=50),
        _block(1, 2, "Lead text.", y=80),
        _block(1, 3, "Methods", y=160, size=14, font="Helvetica-Bold", x0=180, x1=420),
        _block(1, 4, "Body after centered heading.", y=200),
        _block(1, 5, "Methods", y=240, size=14, font="Helvetica-Bold", x0=60, x1=540),
        _block(1, 6, "Body after left heading.", y=270),
    )
    page = _page(*blocks)
    result = LayoutAnalyzer().classify_blocks([page], compute_body_stats([page]))
    centered, left = result[3], result[5]

    assert centered.block_type is BlockType.HEADING
    assert left.block_type is BlockType.HEADING
    assert "center_alignment" in centered.reason_codes
    assert "left_alignment" in left.reason_codes
    assert centered.confidence > left.confidence
    assert "large_before_whitespace" in centered.reason_codes


def test_numbered_lists_precede_numbered_heading_detection() -> None:
    blocks = (
        _block(1, 0, "1. First item", y=40),
        _block(1, 1, "2. Second item", y=70),
        _block(1, 2, "(1)", y=100),
        _block(1, 3, "3 Method", y=140, size=14, font="Helvetica-Bold"),
        _block(1, 4, "3.1 Encoder", y=180, size=13, font="Helvetica-Bold"),
    )
    page = _page(*blocks)
    result = LayoutAnalyzer().classify_blocks([page], compute_body_stats([page]))

    assert [item.block_type for item in result[:3]] == [BlockType.LIST] * 3
    assert result[3].block_type is BlockType.HEADING
    assert result[4].block_type is BlockType.SUBHEADING


def test_short_numbered_sentence_with_terminal_punctuation_is_not_heading() -> None:
    assert looks_like_sentence("3 It works.")
    block = _block(1, 0, "3 It works.", y=80, size=14, font="Helvetica-Bold")
    page = _page(block)
    classified = LayoutAnalyzer().classify_blocks([page], compute_body_stats([page]))

    assert classified[0].block_type is BlockType.PARAGRAPH


def test_unnumbered_heading_font_rank_and_context_create_subsection() -> None:
    blocks = (
        _block(1, 0, "Paper title", y=20, size=20, font="Helvetica-Bold"),
        _block(1, 1, "Alice Smith", y=50, size=10),
        _block(1, 2, "Lead text.", y=80),
        _block(1, 3, "Methods", y=120, size=14, font="Helvetica-Bold"),
        _block(1, 4, "Encoder", y=160, size=13.5, font="Helvetica-Bold"),
        _block(1, 5, "Encoder body.", y=200),
    )
    page = _page(*blocks)
    classified = LayoutAnalyzer().classify_blocks([page], compute_body_stats([page]))
    roots, enriched = DocumentStructureParser().parse(classified)

    assert classified[3].block_type is BlockType.HEADING
    assert classified[4].block_type is BlockType.HEADING
    assert [root.title for root in roots] == ["Methods"]
    assert [child.title for child in roots[0].children] == ["Encoder"]
    assert enriched[5].subsection == "Encoder"


def test_numbered_first_page_section_is_not_title_or_author() -> None:
    blocks = (
        _block(1, 0, "1 Introduction", y=40, size=18, font="Helvetica-Bold"),
        _block(1, 1, "The introduction explains the problem.", y=80),
        _block(1, 2, "3 Method", y=140, size=16, font="Helvetica-Bold"),
    )
    page = _page(*blocks)
    result = LayoutAnalyzer().classify_blocks([page], compute_body_stats([page]))

    assert result[0].block_type is BlockType.HEADING
    assert result[1].block_type is BlockType.PARAGRAPH
    assert result[1].block_type is not BlockType.AUTHOR


def test_reference_year_is_gated_until_references_heading() -> None:
    blocks = (
        _block(1, 0, "Paper title", y=20, size=20, font="Helvetica-Bold"),
        _block(1, 1, "Alice Smith", y=50),
        _block(1, 2, "We report results in 2024.", y=90),
        _block(1, 3, "References", y=160, size=14, font="Helvetica-Bold"),
        _block(1, 4, "[1] Smith, A. 2024.", y=200, size=9),
    )
    page = _page(*blocks)
    result = LayoutAnalyzer().classify_blocks([page], compute_body_stats([page]))

    assert result[2].block_type is BlockType.PARAGRAPH
    assert result[4].block_type is BlockType.REFERENCE


def test_reference_heading_trailing_punctuation_and_strong_citations() -> None:
    blocks = (
        _block(1, 0, "Paper title", y=20, size=20, font="Helvetica-Bold"),
        _block(1, 1, "Alice Smith", y=50),
        _block(1, 2, "We report results in 2024.", y=90),
        _block(1, 3, "References:", y=140, size=14, font="Helvetica-Bold"),
        _block(1, 4, "[1] Smith, A. 2024.", y=180, size=9),
        _block(1, 5, "doi:10.1234/example", y=210, size=9),
        _block(1, 6, "Jones, B. (2023). A study.", y=240, size=9),
    )
    page = _page(*blocks)
    result = LayoutAnalyzer().classify_blocks([page], compute_body_stats([page]))

    assert result[2].block_type is BlockType.PARAGRAPH
    assert result[3].block_type is BlockType.HEADING
    assert [result[index].block_type for index in (4, 5, 6)] == [
        BlockType.REFERENCE,
        BlockType.REFERENCE,
        BlockType.REFERENCE,
    ]
    assert all("reference" in code for code in result[4].reason_codes if "reference" in code)


def test_bibliography_heading_trailing_colon_enters_reference_mode() -> None:
    blocks = (
        _block(1, 0, "Bibliography:", y=100, size=14, font="Helvetica-Bold"),
        _block(1, 1, "[1] Doe, C. 2020.", y=140, size=9),
    )
    page = _page(*blocks)
    result = LayoutAnalyzer().classify_blocks([page], compute_body_stats([page]))

    assert result[0].block_type is BlockType.HEADING
    assert result[1].block_type is BlockType.REFERENCE


def test_abstract_and_summary_labels_allow_trailing_punctuation() -> None:
    blocks = (
        _block(1, 0, "Abstract:", y=80, size=14, font="Helvetica-Bold"),
        _block(1, 1, "Summary;", y=120, size=14, font="Helvetica-Bold"),
    )
    page = _page(*blocks)
    result = LayoutAnalyzer().classify_blocks([page], compute_body_stats([page]))

    assert result[0].block_type is BlockType.ABSTRACT
    assert result[1].block_type is BlockType.ABSTRACT
    assert [item.text for item in result] == ["Abstract:", "Summary;"]


def test_roman_heading_punctuation_and_sentence_guard() -> None:
    assert heading_number_depth("I. Introduction") == 1
    assert heading_number_depth("III.2 Encoder") == 2
    assert looks_like_sentence("I study methods.")
    block = _block(1, 0, "I study methods.", y=80, size=14, font="Helvetica-Bold")
    result = LayoutAnalyzer().classify_blocks([_page(block)], compute_body_stats([_page(block)]))

    assert result[0].block_type is BlockType.PARAGRAPH


def test_caption_labels_require_canonical_numbering_boundaries() -> None:
    texts = (
        "Figure illustrates the workflow.",
        "Table of Contents",
        "Table reports the results.",
        "Fig. note for the reader.",
        "Figure 2: Workflow overview",
        "Table III Results",
        "Fig. A1. Ablation overview",
    )
    blocks = tuple(
        _block(1, index, text, y=40 + index * 30, size=10)
        for index, text in enumerate(texts)
    )
    page = _page(*blocks)
    result = LayoutAnalyzer().classify_blocks([page], compute_body_stats([page]))

    assert all(
        result[index].block_type not in {BlockType.FIGURE_CAPTION, BlockType.TABLE_CAPTION}
        for index in range(4)
    )
    assert result[4].block_type is BlockType.FIGURE_CAPTION
    assert result[5].block_type is BlockType.TABLE_CAPTION
    assert result[6].block_type is BlockType.FIGURE_CAPTION


def test_frozen_extraction_records_snapshot_constructor_lists() -> None:
    span = RawSpan("text", BBox(60, 100, 100, 112), "Body", 10)
    spans = [span]
    line = RawLine("text", BBox(60, 100, 100, 112), spans=spans)
    lines = [line]
    block = RawBlock(1, 0, BBox(60, 100, 100, 112), "text", lines=lines)
    blocks = [block]
    page = PageLayout(1, 600, 800, blocks=blocks, image_regions=[])
    chunk = StructuredChunk(
        uid="chunk-1",
        text="text",
        page_numbers=[1],
        section_path=["Methods"],
        block_uids=["block-1"],
        element_uids=["element-1"],
    )

    spans.append(span)
    lines.append(line)
    blocks.append(block)
    assert line.spans == (span,)
    assert block.lines == (line,)
    assert page.blocks == (block,)
    assert page.image_regions == ()
    assert chunk.page_numbers == (1,)
    assert chunk.section_path == ("Methods",)
    assert chunk.block_uids == ("block-1",)
    assert chunk.element_uids == ("element-1",)
    with pytest.raises(TypeError):
        page.blocks[0] = block
    retained = [block]
    detection = MarginDetection({}, retained)
    reading = ReadingOrderResult([OrderedBlock(block, 0)], retained)
    retained.append(block)
    assert detection.retained_blocks == (block,)
    assert reading.ordered_blocks[0].block is block
    assert reading.excluded_blocks == (block,)
    fonts = ["Body"]
    stats = BodyStats(dominant_fonts=fonts)
    fonts.append("Other")
    assert stats.dominant_fonts == ("Body",)


def test_body_like_author_year_text_is_not_a_reference() -> None:
    texts = ("Results 2024", "we 2024", "accuracy (2024)", "Smith, A. 2024.")
    blocks = tuple(_block(1, index, text, y=80 + index * 30) for index, text in enumerate(texts))
    page = _page(*blocks)
    result = LayoutAnalyzer().classify_blocks([page], compute_body_stats([page]))

    assert all(item.block_type is BlockType.PARAGRAPH for item in result)


def test_standalone_page_number_is_not_an_equation() -> None:
    block = _block(1, 0, "12", y=400, size=10)
    page = _page(block)
    result = LayoutAnalyzer().classify_blocks([page], compute_body_stats([page]))

    assert result[0].block_type is not BlockType.EQUATION


def test_parse_with_elements_is_explicit_and_section_uid_is_stable() -> None:
    blocks = [
        __import__("backend.app.ingestion.domain", fromlist=["ClassifiedBlock"]).ClassifiedBlock(
            uid="heading-1",
            block_type=BlockType.HEADING,
            page_number=1,
            bbox=BBox(60, 100, 540, 118),
            text="3 Method",
        )
    ]
    element = ParsedElement(
        uid="element-1",
        element_type=ElementType.FIGURE,
        status=ElementParseStatus.SUCCESS,
        page_number=1,
        bbox=BBox(60, 140, 540, 240),
    )
    parser = DocumentStructureParser()
    roots_a, enriched_a = parser.parse(blocks)
    roots_b, enriched_b, elements_b = parser.parse_with_elements(blocks, [element])

    assert len((roots_a, enriched_a)) == 2
    assert roots_a[0].uid == roots_b[0].uid == "section_heading-1"
    assert enriched_b[0].section_uid == roots_b[0].uid
    assert elements_b[0].section_uid == roots_b[0].uid


def test_section_uid_heading_based_ids_remain_unique_for_duplicate_block_uids() -> None:
    ClassifiedBlock = __import__(
        "backend.app.ingestion.domain", fromlist=["ClassifiedBlock"]
    ).ClassifiedBlock
    blocks = [
        ClassifiedBlock(
            uid="same-heading",
            block_type=BlockType.HEADING,
            page_number=1,
            bbox=BBox(60, 100, 540, 118),
            text="Methods",
        ),
        ClassifiedBlock(
            uid="same-heading",
            block_type=BlockType.HEADING,
            page_number=1,
            bbox=BBox(60, 160, 540, 178),
            text="Results",
        ),
    ]

    roots, _enriched = DocumentStructureParser().parse(blocks)

    assert [root.uid for root in roots] == ["section_same-heading", "section_same-heading_2"]


def test_roman_heading_numbering_reports_depth() -> None:
    assert heading_number_depth("II Methods") == 1
    assert heading_number_depth("III.2 Encoder") == 2
