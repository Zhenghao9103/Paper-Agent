from backend.app.ingestion.chunking import (
    Contextualizer,
    StructureAwareChunker,
)
from backend.app.ingestion.domain import (
    BBox,
    BlockType,
    ClassifiedBlock,
    ElementParseStatus,
    ElementType,
    ParsedElement,
)


def _block(uid: str, text: str, section: str, subsection: str | None = None) -> ClassifiedBlock:
    return ClassifiedBlock(
        uid=uid,
        block_type=BlockType.PARAGRAPH,
        page_number=1,
        bbox=BBox(72, 100, 540, 160),
        text=text,
        section=section,
        subsection=subsection,
        section_uid=f"section-{section}",
    )


def test_chunker_respects_section_boundaries_and_keeps_metadata() -> None:
    blocks = [
        _block("b1", "A short method paragraph.", "3 Method"),
        _block("b2", "A short result paragraph.", "4 Results"),
    ]

    chunks = StructureAwareChunker().chunk(blocks, title="Paper")

    assert len(chunks) == 2
    assert chunks[0].section_path == ("3 Method",)
    assert chunks[1].section_path == ("4 Results",)
    assert chunks[0].metadata["block_uids"] == ("b1",)


def test_chunker_contextual_prefix_is_separate_and_embedding_is_composed() -> None:
    block = _block(
        "b1",
        "The spectral loss preserves local neighborhoods.",
        "3 Method",
        "3.2 Spectral",
    )
    chunk = StructureAwareChunker(
        contextualizer=Contextualizer(
            generate=lambda title, section, subsection, text: "Local graph context."
        )
    ).chunk([block], title="GDSC")[0]

    assert chunk.text == block.text
    assert chunk.metadata["contextual_prefix"] == "Local graph context."
    assert "GDSC" in chunk.metadata["embedding_text"]
    assert "Local graph context." in chunk.metadata["embedding_text"]


def test_chunker_repairs_pdf_line_break_hyphenation() -> None:
    block = _block(
        "b1",
        "The hyper-\nparameters are selected while k-\nnearest remains hyphenated.",
        "3 Method",
    )

    chunk = StructureAwareChunker().chunk([block], title="Paper")[0]

    assert "hyperparameters" in chunk.text
    assert "k-nearest" in chunk.text
    assert "hyper-\nparameters" not in chunk.text


def test_equation_source_blocks_are_not_duplicated_as_text_chunks() -> None:
    formula_block = _block("formula", "x = y + 1", "3 Method")
    paragraph = _block("body", "The equation defines the update.", "3 Method")
    equation = ParsedElement(
        uid="equation-1",
        element_type=ElementType.EQUATION,
        status=ElementParseStatus.SUCCESS,
        page_number=1,
        bbox=BBox(10, 20, 300, 80),
        source_block_uids=("formula",),
        metadata={"latex": "x=y+1", "structured_data": {"latex": "x=y+1"}},
    )

    chunks = StructureAwareChunker().chunk(
        [formula_block, paragraph], elements=[equation]
    )

    assert len(chunks) == 2
    assert sum("x = y + 1" in chunk.text for chunk in chunks) == 0
    assert sum("LaTeX: x=y+1" in chunk.text for chunk in chunks) == 1


def test_chunker_splits_oversized_paragraph_within_same_section() -> None:
    text = " ".join(f"token{i}" for i in range(170))
    chunker = StructureAwareChunker(target_tokens=50, min_tokens=10, max_tokens=60)

    chunks = chunker.chunk([_block("b1", text, "3 Method")], title="Paper")

    assert len(chunks) > 1
    assert all(chunk.section_path == ("3 Method",) for chunk in chunks)
    assert all(chunk.metadata["token_count"] <= 60 for chunk in chunks)


def test_chunker_preserves_overlap_and_per_page_bboxes() -> None:
    first = ClassifiedBlock(
        uid="b1",
        block_type=BlockType.PARAGRAPH,
        page_number=1,
        bbox=BBox(10, 10, 30, 30),
        text=" ".join(["first."] * 60),
        section="3 Method",
        section_uid="section-method",
    )
    second = ClassifiedBlock(
        uid="b2",
        block_type=BlockType.PARAGRAPH,
        page_number=2,
        bbox=BBox(40, 40, 80, 80),
        text=" ".join(["second."] * 60),
        section="3 Method",
        section_uid="section-method",
    )
    chunks = StructureAwareChunker(
        target_tokens=100, min_tokens=10, max_tokens=200
    ).chunk([first, second], title="Paper")

    assert len(chunks) == 2
    assert "first." in chunks[1].text
    assert chunks[1].metadata["page_bboxes"] == {
        "1": (10.0, 10.0, 30.0, 30.0),
        "2": (40.0, 40.0, 80.0, 80.0),
    }


def test_element_chunk_preserves_element_and_source_provenance() -> None:
    element = ParsedElement(
        uid="figure-1",
        element_type=ElementType.FIGURE,
        status=ElementParseStatus.SUCCESS,
        page_number=5,
        bbox=BBox(100, 120, 300, 260),
        caption="Figure 1. Overview.",
        source_block_uids=("b10", "b11"),
        metadata={"image_path": "figures/page5.png"},
        section="3 Method",
        section_uid="section-method",
    )

    chunk = StructureAwareChunker().chunk([], title="Paper", elements=[element])[0]

    assert chunk.element_uids == ("figure-1",)
    assert chunk.block_uids == ("b10", "b11")
    assert chunk.metadata["bbox"] == (100.0, 120.0, 300.0, 260.0)
    assert chunk.metadata["source_block_uids"] == ("b10", "b11")
    assert chunk.metadata["page_bboxes"] == {
        "5": (100.0, 120.0, 300.0, 260.0)
    }


def test_figure_chunk_projects_structured_description_in_stable_order() -> None:
    element = ParsedElement(
        uid="figure-1",
        element_type=ElementType.FIGURE,
        status=ElementParseStatus.SUCCESS,
        page_number=2,
        bbox=BBox(10, 20, 300, 220),
        caption="Figure 1. Pipeline.",
        metadata={
            "image_path": "figures/pipeline.png",
            "structured_data": {
                "figure_type": "flowchart",
                "caption": "Figure 1. Pipeline.",
                "components": ["Input", "Encoder"],
                "relationships": ["Input enters Encoder"],
                "summary": "A processing pipeline.",
                "visible_text": ["Stage A"],
            },
            "warning_codes": [],
        },
    )

    chunk = StructureAwareChunker().chunk([], title="Paper", elements=[element])[0]

    assert chunk.text.splitlines() == [
        "Figure 1. Pipeline.",
        "Figure type: flowchart",
        "Components: Input; Encoder",
        "Relationships: Input enters Encoder",
        "Summary: A processing pipeline.",
        "Visible text: Stage A",
    ]
    assert chunk.metadata["structured_data"]["figure_type"] == "flowchart"
    assert chunk.metadata["image_path"] == "figures/pipeline.png"


def test_table_chunk_projects_columns_and_row_objects() -> None:
    element = ParsedElement(
        uid="table-1",
        element_type=ElementType.TABLE,
        status=ElementParseStatus.SUCCESS,
        page_number=3,
        bbox=BBox(10, 20, 300, 220),
        caption="Table 1. Results.",
        metadata={
            "structured_data": {
                "columns": ["Method", "ACC"],
                "rows": [{"Method": "STG", "ACC": "0.77"}],
            },
            "image_path": "tables/results.png",
        },
    )

    chunk = StructureAwareChunker().chunk([], title="Paper", elements=[element])[0]

    assert chunk.text.splitlines() == [
        "Table 1. Results.",
        "Columns: Method; ACC",
        "Method: STG | ACC: 0.77",
    ]


def test_partial_or_empty_table_does_not_create_element_chunk() -> None:
    element = ParsedElement(
        uid="table-empty",
        element_type=ElementType.TABLE,
        status=ElementParseStatus.PARTIAL,
        page_number=2,
        bbox=BBox(10, 10, 100, 100),
        content="Columns: column_1; column_2",
        metadata={
            "structured_data": {
                "columns": ["column_1", "column_2"],
                "rows": [],
                "raw_rows": [],
            }
        },
    )

    assert StructureAwareChunker(token_counter=lambda text: len(text.split())).chunk(
        [], elements=[element]
    ) == []


def test_equation_chunk_separates_label_latex_and_context() -> None:
    element = ParsedElement(
        uid="equation-1",
        element_type=ElementType.EQUATION,
        status=ElementParseStatus.SUCCESS,
        page_number=4,
        bbox=BBox(10, 20, 300, 80),
        metadata={
            "equation_id": "Eq. 16",
            "latex": r"\\sum_{k=1} P_{ik}",
            "surrounding_text": "The objective is minimized.",
            "structured_data": {"latex": r"\\sum_{k=1} P_{ik}"},
            "image_path": "equations/eq16.png",
        },
    )

    chunk = StructureAwareChunker().chunk([], title="Paper", elements=[element])[0]

    assert chunk.text.splitlines() == [
        "Equation: Eq. 16",
        r"LaTeX: \\sum_{k=1} P_{ik}",
        "Context: The objective is minimized.",
    ]


def test_reference_blocks_do_not_merge_with_body_without_section_context() -> None:
    body = ClassifiedBlock(
        uid="body",
        block_type=BlockType.PARAGRAPH,
        page_number=1,
        bbox=BBox(10, 10, 90, 30),
        text="Body text.",
    )
    reference = ClassifiedBlock(
        uid="reference",
        block_type=BlockType.REFERENCE,
        page_number=1,
        bbox=BBox(10, 40, 90, 60),
        text="[1] A paper reference.",
    )

    chunks = StructureAwareChunker().chunk([body, reference], title="Paper")

    assert len(chunks) == 2
    assert chunks[0].block_uids == ("body",)
    assert chunks[1].block_uids == ("reference",)
