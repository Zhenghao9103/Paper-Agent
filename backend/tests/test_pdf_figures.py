from __future__ import annotations

from pathlib import Path

import fitz
from backend.app.ingestion.extraction import TextExtractor
from backend.app.ingestion.figure_descriptions import (
    FigureDescriptionError,
    FigureDescriptionResult,
    normalize_figure_description,
)
from backend.app.ingestion.figures import FigureParser
from backend.tests.pdf_factory import insert_block, make_pdf, vector_figure_pdf


def _write_pdf(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(content)
    return path


def _figure_pdf(label: str = "Figure 2. Vector model overview") -> bytes:
    def populate(page: fitz.Page) -> None:
        page.draw_rect(fitz.Rect(140, 140, 472, 360), color=(0.1, 0.2, 0.6), width=2)
        page.draw_circle((240, 250), 45, color=(0.8, 0.2, 0.2), width=2)
        page.draw_circle((372, 250), 45, color=(0.2, 0.6, 0.2), width=2)
        page.draw_line((285, 250), (327, 250), color=(0, 0, 0), width=2)
        insert_block(page, (140, 385, 472, 425), label, size=10)

    return make_pdf([populate])


def test_figure_parser_detects_vector_region_and_links_caption(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path, _figure_pdf("Fig. 2. Vector model overview"))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    elements = FigureParser().parse(pdf_path, page, page.blocks, tmp_path / "figures")

    assert len(elements) == 1
    element = elements[0]
    assert element.metadata["figure_id"] == "Figure 2"
    assert element.caption == "Fig. 2. Vector model overview"
    assert element.metadata["image_path"]
    assert Path(element.metadata["image_path"]).exists()
    assert element.status.value == "success"
    assert element.parse_status.value == "success"
    assert Path(element.image_path).exists()


def test_figure_parser_uses_injected_vision_description(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path, vector_figure_pdf())
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    elements = FigureParser(
        describe=lambda request: "A two-stage encoder and clustering pipeline."
    ).parse(pdf_path, page, page.blocks, tmp_path / "figures")

    assert elements[0].metadata["vision_description"].startswith("A two-stage")
    assert elements[0].content.startswith("A two-stage")
    assert elements[0].vision_description.startswith("A two-stage")


def test_figure_parser_persists_structured_description(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path, vector_figure_pdf())
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    result = FigureDescriptionResult(
        description=normalize_figure_description(
            {
                "figure_type": "architecture",
                "caption": "Figure 1. Pipeline.",
                "visible_text": ["Encoder"],
                "components": ["Input", "Encoder"],
                "relationships": ["Input enters Encoder"],
                "summary": "An encoder pipeline.",
            }
        ),
        model="mineru-vlm",
        latency_ms=42,
    )
    element = FigureParser(describe=lambda _request: result).parse(
        pdf_path, page, page.blocks, tmp_path / "figures"
    )[0]

    assert element.structured_data["figure_type"] == "architecture"
    assert element.structured_data["components"] == ["Input", "Encoder"]
    assert element.vision_description == "An encoder pipeline."
    assert element.metadata["description_model"] == "mineru-vlm"
    assert element.metadata["description_latency_ms"] == 42


def test_figure_parser_keeps_crop_when_vision_fails(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path, vector_figure_pdf())
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    def fail(_request):
        raise RuntimeError("offline")

    elements = FigureParser(describe=fail).parse(
        pdf_path, page, page.blocks, tmp_path / "figures"
    )

    element = elements[0]
    assert element.status.value == "partial"
    assert Path(element.metadata["image_path"]).exists()
    assert element.metadata["vision_description"] is None
    assert "vision_description_failed" in element.metadata["warning_codes"]


def test_figure_parser_preserves_stable_structured_failure_code(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path, vector_figure_pdf())
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    def fail(_request):
        raise FigureDescriptionError("figure_description_timeout")

    element = FigureParser(describe=fail).parse(
        pdf_path, page, page.blocks, tmp_path / "figures"
    )[0]

    assert "figure_description_timeout" in element.metadata["warning_codes"]
    assert "vision_description_failed" not in element.metadata["warning_codes"]


def test_figure_parser_filters_tiny_vector_and_unrelated_text(tmp_path: Path) -> None:
    def populate(page: fitz.Page) -> None:
        page.draw_rect(fitz.Rect(100, 100, 102, 102), color=(0, 0, 0), width=1)
        insert_block(page, (72, 500, 540, 560), "This paragraph is not a figure.", size=10)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    assert FigureParser().parse(pdf_path, page, page.blocks, tmp_path / "figures") == []


def test_figure_parser_deduplicates_raster_and_vector_region(tmp_path: Path) -> None:
    def populate(page: fitz.Page) -> None:
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 80, 80), False)
        pixmap.clear_with(0x336699)
        page.insert_image(fitz.Rect(140, 140, 472, 360), pixmap=pixmap)
        page.draw_rect(fitz.Rect(140, 140, 472, 360), color=(0.1, 0.2, 0.6), width=2)
        insert_block(page, (140, 385, 472, 425), "Figure 3. Raster and vector", size=10)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    elements = FigureParser().parse(pdf_path, page, page.blocks, tmp_path / "figures")

    assert len(elements) == 1
    assert elements[0].metadata["figure_id"] == "Figure 3"


def test_figure_parser_leaves_non_overlapping_caption_unresolved(tmp_path: Path) -> None:
    def populate(page: fitz.Page) -> None:
        page.draw_rect(fitz.Rect(100, 140, 300, 300), color=(0.1, 0.2, 0.6), width=2)
        insert_block(page, (360, 310, 540, 350), "Figure 9. Unrelated caption", size=10)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    elements = FigureParser().parse(pdf_path, page, page.blocks, tmp_path / "figures")

    assert len(elements) == 1
    assert elements[0].caption is None
    assert elements[0].metadata["figure_id"] is None
    assert "caption_unresolved" in elements[0].metadata["warning_codes"]


def test_figure_parser_leaves_distant_caption_unresolved(tmp_path: Path) -> None:
    def populate(page: fitz.Page) -> None:
        page.draw_rect(fitz.Rect(140, 120, 472, 300), color=(0.1, 0.2, 0.6), width=2)
        insert_block(page, (140, 450, 472, 490), "Figure 10. Distant text", size=10)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    elements = FigureParser().parse(pdf_path, page, page.blocks, tmp_path / "figures")

    assert len(elements) == 1
    assert elements[0].caption is None


def test_figure_parser_filters_ruled_table_before_classification(tmp_path: Path) -> None:
    from backend.tests.pdf_factory import ruled_table_pdf

    pdf_path = _write_pdf(tmp_path, ruled_table_pdf())
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    assert FigureParser().parse(pdf_path, page, page.blocks, tmp_path / "figures") == []


def test_figure_parser_deduplicates_repeated_raster_xref(tmp_path: Path) -> None:
    def populate(page: fitz.Page) -> None:
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 80, 80), False)
        pixmap.clear_with(0x336699)
        page.insert_image(fitz.Rect(36, 20, 116, 100), pixmap=pixmap)
        page.insert_image(fitz.Rect(496, 20, 576, 100), pixmap=pixmap)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    elements = FigureParser().parse(pdf_path, page, page.blocks, tmp_path / "figures")

    assert len(elements) <= 1
