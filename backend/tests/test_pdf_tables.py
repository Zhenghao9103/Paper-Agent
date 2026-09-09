from __future__ import annotations

from pathlib import Path

import fitz
from backend.app.ingestion.extraction import TextExtractor
from backend.app.ingestion.tables import TableParser
from backend.tests.pdf_factory import insert_block, make_pdf, ruled_table_pdf


def _write_pdf(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(content)
    return path


def _four_column_table_pdf() -> bytes:
    def populate(page: fitz.Page) -> None:
        left, top, right, bottom = 72.0, 150.0, 540.0, 300.0
        xs = [left, 190.0, 300.0, 410.0, right]
        ys = [top, 195.0, 245.0, bottom]
        for x in xs:
            page.draw_line((x, top), (x, bottom), color=(0, 0, 0), width=1)
        for y in ys:
            page.draw_line((left, y), (right, y), color=(0, 0, 0), width=1)
        for row, values in enumerate(
            (
                ("Method", "ACC", "NMI", "ARI"),
                ("KMeans", "62.3", "55.1", "48.3"),
                ("GDSC", "78.9", "69.2", "65.4"),
            )
        ):
            for col, value in enumerate(values):
                insert_block(
                    page,
                    (xs[col] + 8, ys[row] + 11, xs[col + 1] - 8, ys[row + 1] - 6),
                    value,
                    size=10,
                    fontname="hebo" if row == 0 else None,
                )
        insert_block(
            page,
            (72, 112, 540, 140),
            "Table 3: Clustering performance.",
            size=10,
        )

    return make_pdf([populate])


def test_table_parser_recovers_ruled_table_and_caption(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path, _four_column_table_pdf())
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    tables = TableParser().parse(pdf_path, page, page.blocks, tmp_path / "tables")

    assert len(tables) == 1
    table = tables[0]
    assert table.label == "Table 3"
    assert table.structured_data["headers"] == ["Method", "ACC", "NMI", "ARI"]
    assert table.structured_data["rows"][1] == {
        "Method": "GDSC",
        "ACC": "78.9",
        "NMI": "69.2",
        "ARI": "65.4",
    }
    assert table.structured_data["raw_rows"][1] == ["GDSC", "78.9", "69.2", "65.4"]
    assert "| GDSC | 78.9 | 69.2 | 65.4 |" in table.structured_data["markdown_text"]
    assert table.caption == "Table 3: Clustering performance."
    assert table.parse_status.value == "success"
    assert Path(table.image_path).exists()


def test_table_parser_falls_back_to_partial_without_inventing_cells(tmp_path: Path) -> None:
    def populate(page: fitz.Page) -> None:
        left, top, right, bottom = 90.0, 160.0, 522.0, 300.0
        page.draw_line((left, top), (right, top), color=(0, 0, 0), width=1)
        page.draw_line((left, bottom), (right, bottom), color=(0, 0, 0), width=1)
        page.draw_line((left, top), (left, bottom), color=(0, 0, 0), width=1)
        page.draw_line((right, top), (right, bottom), color=(0, 0, 0), width=1)
        page.draw_line((300, top), (300, bottom), color=(0, 0, 0), width=1)
        # Missing the middle row boundary and one value makes the grid unreliable.
        insert_block(page, (100, 180, 280, 205), "Method", size=10, fontname="hebo")
        insert_block(page, (315, 180, 500, 205), "Score", size=10, fontname="hebo")
        insert_block(page, (100, 235, 280, 260), "GDSC", size=10)
        insert_block(page, (100, 265, 280, 290), "Available text only", size=10)
        insert_block(page, (90, 120, 522, 145), "Table 4: Irregular results", size=10)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]
    tables = TableParser().parse(pdf_path, page, page.blocks, tmp_path / "tables")

    assert len(tables) == 1
    table = tables[0]
    assert table.parse_status.value == "partial"
    assert table.caption == "Table 4: Irregular results"
    assert Path(table.image_path).exists()
    assert "Available text only" in table.structured_data["raw_text"]
    assert "0.95" not in table.structured_data["markdown_text"]


def test_table_parser_ignores_non_table_page(tmp_path: Path) -> None:
    pdf_path = _write_pdf(
        tmp_path,
        make_pdf(
            [lambda page: insert_block(page, (72, 100, 540, 150), "A paragraph with aligned text.")]
        ),
    )
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]
    assert TableParser().parse(pdf_path, page, page.blocks, tmp_path / "tables") == []


def test_table_parser_handles_existing_two_column_fixture(tmp_path: Path) -> None:
    pdf_path = _write_pdf(tmp_path, ruled_table_pdf())
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]
    tables = TableParser().parse(pdf_path, page, page.blocks, tmp_path / "tables")
    assert len(tables) == 1
    assert tables[0].structured_data["headers"] == ["Method", "Score"]
    assert tables[0].structured_data["columns"] == ["Method", "Score"]
    assert tables[0].structured_data["rows"] == [
        {"Method": "Proposed", "Score": "0.95"}
    ]


def test_table_parser_recovers_text_aligned_table(tmp_path: Path) -> None:
    def populate(page: fitz.Page) -> None:
        rows = (
            (150, ("Method", "ACC", "NMI")),
            (180, ("A", "1", "2")),
            (210, ("B", "3", "4")),
        )
        for y, row in rows:
            for x, value in zip((100, 250, 380), row, strict=True):
                page.insert_text((x, y), value, fontsize=10)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]
    tables = TableParser().parse(pdf_path, page, page.blocks, tmp_path / "tables")

    assert len(tables) == 1
    assert tables[0].structured_data["headers"] == ["Method", "ACC", "NMI"]
    assert tables[0].structured_data["rows"] == [
        {"Method": "A", "ACC": "1", "NMI": "2"},
        {"Method": "B", "ACC": "3", "NMI": "4"},
    ]


def test_captioned_text_table_uses_only_following_same_column_blocks(
    tmp_path: Path,
) -> None:
    def populate(page: fitz.Page) -> None:
        page.insert_text((72, 120), "Table 2", fontsize=10)
        page.insert_text((72, 135), "Information of data sets.", fontsize=9)
        page.insert_text(
            (72, 155),
            "Name | Number of points | Dimension of data | Number of clusters",
            fontsize=8,
        )
        page.insert_text((72, 175), "Spiral | 312 | 2 | 3", fontsize=8)
        page.insert_text((72, 190), "Yale | 165 | 1024 | 15", fontsize=8)
        page.insert_text(
            (315, 120),
            "Unrelated right-column prose must not enter the table.",
            fontsize=9,
        )
        page.insert_text((72, 225), "4.3. Experimental results", fontsize=11)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    tables = TableParser().parse(pdf_path, page, page.blocks, tmp_path / "tables")

    assert len(tables) == 1
    assert tables[0].structured_data["columns"] == [
        "Name",
        "Number of points",
        "Dimension of data",
        "Number of clusters",
    ]
    assert tables[0].structured_data["rows"] == [
        {
            "Name": "Spiral",
            "Number of points": "312",
            "Dimension of data": "2",
            "Number of clusters": "3",
        },
        {
            "Name": "Yale",
            "Number of points": "165",
            "Dimension of data": "1024",
            "Number of clusters": "15",
        },
    ]
    assert "Unrelated" not in tables[0].content


def test_captioned_table_reconstructs_cells_from_pdf_line_geometry(
    tmp_path: Path,
) -> None:
    def populate(page: fitz.Page) -> None:
        headers = ("Name", "Number of points", "Dimension of data", "Number of clusters")
        xs = (45, 120, 225, 300)
        page.insert_text((40, 100), "Table 2")
        for x, value in zip(xs, headers, strict=True):
            page.insert_text((x, 125), value, fontsize=7)
        for y, row in (
            (145, ("Spiral", "312", "2", "3")),
            (160, ("Yale", "165", "1024", "15")),
        ):
            for x, value in zip(xs, row, strict=True):
                page.insert_text((x, y), value, fontsize=8)
        page.insert_text((520, 120), "Unrelated", fontsize=8)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]
    tables = TableParser().parse(pdf_path, page, page.blocks, tmp_path / "tables")

    table = next(item for item in tables if item.structured_data["table_id"] == "Table 2")
    assert table.structured_data["columns"] == [
        "Name",
        "Number of points",
        "Dimension of data",
        "Number of clusters",
    ]
    assert table.structured_data["rows"] == [
        {
            "Name": "Spiral",
            "Number of points": "312",
            "Dimension of data": "2",
            "Number of clusters": "3",
        },
        {
            "Name": "Yale",
            "Number of points": "165",
            "Dimension of data": "1024",
            "Number of clusters": "15",
        },
    ]
    assert "Unrelated" not in table.content


def test_table_parser_rejects_partial_uncaptioned_multicolumn_prose(tmp_path: Path) -> None:
    def populate(page: fitz.Page) -> None:
        for y, left, right in (
            (
                120,
                "Introduction and related work discuss clustering",
                "The proposed method learns representations",
            ),
            (
                145,
                "Prior approaches rely on pseudo labels",
                "Experiments compare several benchmark datasets",
            ),
            (
                170,
                "These paragraphs are prose rather than cells",
                "The conclusion summarizes future directions",
            ),
        ):
            page.insert_text((72, y), left, fontsize=9)
            page.insert_text((315, y), right, fontsize=9)

    pdf_path = _write_pdf(tmp_path, make_pdf([populate]))
    page = TextExtractor().extract(pdf_path, paper_id="paper-1")[0]

    assert TableParser().parse(pdf_path, page, page.blocks, tmp_path / "tables") == []


def test_table_columns_are_unique_and_rows_are_padded() -> None:
    columns, rows, warnings = TableParser.canonicalize_rows(
        ["Method", "", "Method"],
        [["STG", "0.77"], ["Ours", "0.91", "best"]],
    )

    assert columns == ["Method", "column_2", "Method_2"]
    assert rows == [
        {"Method": "STG", "column_2": "0.77", "Method_2": ""},
        {"Method": "Ours", "column_2": "0.91", "Method_2": "best"},
    ]
    assert warnings == []


def test_overwide_table_row_is_partial_and_preserves_raw_values() -> None:
    columns, rows, warnings = TableParser.canonicalize_rows(
        ["Method", "ACC"], [["STG", "0.77", "unexpected"]]
    )

    assert columns == ["Method", "ACC"]
    assert rows == [{"Method": "STG", "ACC": "0.77"}]
    assert warnings == ["table_row_overflow"]
