from __future__ import annotations

from pathlib import Path

import fitz
from backend.app.ingestion.domain import BlockType, ElementType
from backend.app.ingestion.equations import EquationParser
from backend.app.ingestion.extraction import TextExtractor
from backend.app.ingestion.layout import LayoutAnalyzer, compute_body_stats
from backend.app.ingestion.mineru_formula import FormulaRecognition
from backend.tests.pdf_factory import insert_block, make_pdf


def _write_pdf(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(content)
    return path


def _equation_pdf() -> bytes:
    def populate(page: fitz.Page) -> None:
        insert_block(
            page,
            (72, 145, 540, 180),
            "We define the energy relation below.",
            size=10,
        )
        insert_block(page, (170, 240, 420, 278), "E = mc^2", size=16)
        insert_block(page, (470, 240, 530, 278), "(7)", size=12)
        insert_block(
            page,
            (72, 305, 540, 340),
            "This relation links mass and energy.",
            size=10,
        )

    return make_pdf([populate])


def _classified(page):
    stats = compute_body_stats([page])
    return LayoutAnalyzer().classify_blocks([page], stats)


def test_equation_parser_extracts_label_expression_crop_and_context(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path, _equation_pdf())
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    elements = EquationParser().parse(
        pdf_path,
        page,
        _classified(page),
        tmp_path / "equations",
    )

    assert len(elements) == 1
    element = elements[0]
    assert element.element_type is ElementType.EQUATION
    assert element.metadata["equation_id"] == "Eq. 7"
    assert element.metadata["raw_expression"] == "E = mc^2"
    assert "mass and energy" in element.metadata["surrounding_text"]
    assert "energy relation" in element.metadata["surrounding_text"]
    assert element.bbox.x0 < 180 and element.bbox.x1 > 500
    assert element.image_path is not None and element.image_path.exists()


def test_equation_parser_does_not_promote_standalone_page_number(tmp_path: Path) -> None:
    def populate(page: fitz.Page) -> None:
        insert_block(page, (72, 200, 540, 230), "12", size=10)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    assert EquationParser().parse(
        pdf_path, page, _classified(page), tmp_path / "equations"
    ) == []


def test_equation_parser_preserves_unreliable_glyphs_as_partial(tmp_path: Path) -> None:
    def populate(page: fitz.Page) -> None:
        insert_block(page, (170, 240, 420, 278), "x = α + Ω", size=16)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    elements = EquationParser().parse(
        pdf_path, page, _classified(page), tmp_path / "equations"
    )

    assert len(elements) == 1
    assert elements[0].metadata["latex"] is None
    assert elements[0].status.value == "partial"
    assert elements[0].metadata["raw_expression"] == "x = ? + ?"


def test_equation_parser_ignores_heading_and_regular_text(tmp_path: Path) -> None:
    def populate(page: fitz.Page) -> None:
        insert_block(page, (72, 100, 540, 130), "3 Method", size=15, fontname="hebo")
        insert_block(page, (72, 160, 540, 210), "A = study of graph methods.", size=10)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    classified = _classified(page)
    assert all(item.block_type is not BlockType.EQUATION for item in classified)
    assert EquationParser().parse(pdf_path, page, classified, tmp_path / "equations") == []


def test_equation_parser_does_not_invent_id_for_unlabeled_expression(tmp_path: Path) -> None:
    def populate(page: fitz.Page) -> None:
        insert_block(page, (170, 240, 420, 278), "x^2 + y^2", size=16)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]
    elements = EquationParser().parse(
        pdf_path, page, _classified(page), tmp_path / "equations"
    )

    assert len(elements) == 1
    assert elements[0].equation_id is None


def test_equation_parser_merges_adjacent_aligned_lines(tmp_path: Path) -> None:
    def populate(page: fitz.Page) -> None:
        insert_block(page, (170, 220, 420, 260), "x^2 + y^2", size=14)
        insert_block(page, (170, 244, 420, 284), "= r^2", size=14)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]
    elements = EquationParser().parse(
        pdf_path, page, _classified(page), tmp_path / "equations"
    )

    assert len(elements) == 1
    assert "x^2 + y^2" in elements[0].raw_expression
    assert "= r^2" in elements[0].raw_expression


def test_equation_parser_accepts_single_operator_with_numbered_signal(tmp_path: Path) -> None:
    def populate(page: fitz.Page) -> None:
        insert_block(page, (180, 220, 420, 260), "E = mc", size=16)
        insert_block(page, (470, 220, 530, 260), "(1)", size=12)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]
    elements = EquationParser().parse(
        pdf_path, page, _classified(page), tmp_path / "equations"
    )

    assert len(elements) == 1
    assert elements[0].equation_id == "Eq. 1"


def test_equation_parser_accepts_pdf_encoded_inline_number(tmp_path: Path) -> None:
    def populate(page: fitz.Page) -> None:
        insert_block(page, (80, 220, 300, 260), "x = y + 1   ð5Þ", size=14)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]
    elements = EquationParser().parse(
        pdf_path, page, _classified(page), tmp_path / "equations"
    )

    assert len(elements) == 1
    assert elements[0].equation_id == "Eq. 5"
    assert elements[0].source_block_uids


def test_equation_parser_rejects_tiny_unlabelled_assignment(tmp_path: Path) -> None:
    def populate(page: fitz.Page) -> None:
        insert_block(page, (170, 240, 220, 265), "i=1", size=12)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    assert EquationParser().parse(
        pdf_path, page, _classified(page), tmp_path / "equations"
    ) == []


def test_equation_parser_rejects_bibliography_urls(tmp_path: Path) -> None:
    def populate(page: fitz.Page) -> None:
        insert_block(
            page,
            (72, 240, 540, 275),
            "1 http://www.cs.columbia.edu/CAVE/software/softlib/",
            size=10,
        )

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    assert EquationParser().parse(
        pdf_path, page, _classified(page), tmp_path / "equations"
    ) == []


def test_equation_parser_uses_expression_only_crop_for_recognition(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path, _equation_pdf())
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]
    recognized_sizes: list[tuple[int, int]] = []

    class Recognizer:
        def recognize(self, image_path: Path) -> FormulaRecognition:
            pixmap = fitz.Pixmap(str(image_path))
            recognized_sizes.append((pixmap.width, pixmap.height))
            return FormulaRecognition(
                latex=r"E=mc^2",
                model="fake-mfr",
                latency_ms=7,
                status="success",
            )

    element = EquationParser(recognizer=Recognizer()).parse(
        pdf_path, page, _classified(page), tmp_path / "equations"
    )[0]

    assert element.equation_id == "Eq. 7"
    assert element.latex == r"E=mc^2"
    assert element.metadata["recognition"]["model"] == "fake-mfr"
    assert element.metadata["recognition"]["latency_ms"] == 7
    stored_crop = fitz.Pixmap(str(element.image_path))
    assert recognized_sizes and recognized_sizes[0][0] < stored_crop.width


def test_equation_parser_preserves_raw_expression_when_recognition_fails(
    tmp_path: Path,
) -> None:
    pdf_path = _write_pdf(tmp_path, _equation_pdf())
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    class Recognizer:
        def recognize(self, _image_path: Path) -> FormulaRecognition:
            return FormulaRecognition(
                latex=None,
                model="fake-mfr",
                latency_ms=4,
                status="partial",
                warnings=("formula_recognition_failed",),
            )

    element = EquationParser(recognizer=Recognizer()).parse(
        pdf_path, page, _classified(page), tmp_path / "equations"
    )[0]

    assert element.raw_expression == "E = mc^2"
    assert element.status.value == "partial"
    assert "formula_recognition_failed" in element.metadata["warning_codes"]
