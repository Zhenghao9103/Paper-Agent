from __future__ import annotations

from collections.abc import Callable, Iterable

import fitz

PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0

PageCallback = Callable[[fitz.Page], None]
RectLike = fitz.Rect | tuple[float, float, float, float]


def make_pdf(pages: Iterable[PageCallback]) -> bytes:
    document = fitz.open()
    try:
        for populate_page in pages:
            page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
            populate_page(page)
        return document.tobytes(no_new_id=True)
    finally:
        document.close()


def insert_block(
    page: fitz.Page,
    rect: RectLike,
    text: str,
    size: float = 10,
    fontname: str | None = None,
) -> float:
    options = {"fontsize": size}
    if fontname is not None:
        options["fontname"] = fontname
    return page.insert_textbox(fitz.Rect(rect), text, **options)


def one_column_pdf() -> bytes:
    def populate(page: fitz.Page) -> None:
        insert_block(page, (72, 54, 540, 90), "One-column paper", size=18, fontname="hebo")
        insert_block(
            page,
            (72, 112, 540, 250),
            "This paragraph occupies a single readable column.\n"
            "Its lines provide deterministic extraction content.",
            size=11,
        )

    return make_pdf([populate])


def two_column_pdf() -> bytes:
    def populate(page: fitz.Page) -> None:
        for index, label in enumerate(("A", "B", "C")):
            top = 90 + index * 90
            insert_block(page, (72, top, 270, top + 45), label, size=14)
        for index, label in enumerate(("D", "E", "F")):
            top = 90 + index * 90
            insert_block(page, (342, top, 540, top + 45), label, size=14)

    return make_pdf([populate])


def headings_pdf() -> bytes:
    def populate(page: fitz.Page) -> None:
        insert_block(page, (72, 48, 540, 84), "Paper title", size=18, fontname="hebo")
        insert_block(page, (72, 105, 540, 130), "Abstract", size=13, fontname="hebo")
        insert_block(page, (72, 135, 540, 185), "A concise abstract paragraph.", size=10)
        insert_block(page, (72, 215, 540, 245), "1 Introduction", size=15, fontname="hebo")
        insert_block(page, (72, 270, 540, 300), "1.1 Motivation", size=12, fontname="hebo")

    return make_pdf([populate])


def ruled_table_pdf() -> bytes:
    def populate(page: fitz.Page) -> None:
        left, top, right, bottom = 90.0, 150.0, 522.0, 270.0
        middle_x, middle_y = 360.0, 210.0
        for start, end in (
            ((left, top), (right, top)),
            ((left, middle_y), (right, middle_y)),
            ((left, bottom), (right, bottom)),
            ((left, top), (left, bottom)),
            ((middle_x, top), (middle_x, bottom)),
            ((right, top), (right, bottom)),
        ):
            page.draw_line(start, end, color=(0, 0, 0), width=1)
        insert_block(page, (105, 166, 345, 198), "Method", size=11, fontname="hebo")
        insert_block(page, (375, 166, 505, 198), "Score", size=11, fontname="hebo")
        insert_block(page, (105, 226, 345, 258), "Proposed", size=11)
        insert_block(page, (375, 226, 505, 258), "0.95", size=11)

    return make_pdf([populate])


def vector_figure_pdf() -> bytes:
    def populate(page: fitz.Page) -> None:
        page.draw_rect(fitz.Rect(140, 140, 472, 360), color=(0.1, 0.2, 0.6), width=2)
        page.draw_circle((240, 250), 45, color=(0.8, 0.2, 0.2), width=2)
        page.draw_circle((372, 250), 45, color=(0.2, 0.6, 0.2), width=2)
        page.draw_line((285, 250), (327, 250), color=(0, 0, 0), width=2)
        insert_block(page, (140, 385, 472, 425), "Figure 1. Vector model overview", size=10)

    return make_pdf([populate])


def numbered_equation_pdf() -> bytes:
    def populate(page: fitz.Page) -> None:
        insert_block(page, (160, 220, 420, 260), "E = mc^2", size=16, fontname="tiro")
        insert_block(page, (470, 220, 520, 260), "(1)", size=12)

    return make_pdf([populate])
