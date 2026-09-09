from __future__ import annotations

import json
from pathlib import Path

import fitz
from backend.app.ingestion.mineru_primary import MinerUPrimaryParser
from backend.app.ingestion.pipeline import PipelineResult
from backend.app.services.ingestion import run_primary_with_fallback


def _pdf(path: Path) -> None:
    document = fitz.open()
    document.new_page(width=600, height=800)
    document.save(path)
    document.close()


def _content() -> list[list[dict]]:
    return [
        [
            {
                "type": "title",
                "content": {
                    "title_content": [{"type": "text", "content": "1. Method"}],
                    "level": 2,
                },
                "bbox": [40, 50, 180, 70],
            },
            {
                "type": "paragraph",
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "We define "},
                        {"type": "equation_inline", "content": "k_{max}=20"},
                        {"type": "text", "content": " for every sample."},
                    ]
                },
                "bbox": [40, 90, 500, 140],
            },
            {
                "type": "table",
                "content": {
                    "image_source": {"path": "images/table.jpg"},
                    "table_caption": [{"type": "text", "content": "Table 2. Results"}],
                    "html": (
                        "<table><tr><td>Method</td><td>ACC</td></tr>"
                        "<tr><td>STG</td><td>0.77</td></tr></table>"
                    ),
                },
                "bbox": [40, 180, 400, 260],
            },
            {
                "type": "equation_interline",
                "content": {
                    "math_content": "s_{ij}=d_{ij}^{-1}",
                    "math_type": "latex",
                    "image_source": {"path": "images/equation.jpg"},
                },
                "bbox": [100, 270, 350, 290],
            },
            {
                "type": "image",
                "content": {
                    "image_source": {"path": "images/figure.jpg"},
                    "image_caption": [{"type": "text", "content": "Fig. 1. Overview"}],
                },
                "bbox": [40, 300, 400, 500],
            },
            {
                "type": "page_header",
                "content": {"page_header_content": [{"type": "text", "content": "Journal header"}]},
                "bbox": [40, 10, 200, 20],
            },
        ]
    ]


def test_mineru_primary_maps_v2_content_without_flattening_tables(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    _pdf(pdf_path)
    output = tmp_path / "mineru"
    parse_dir = output / "paper" / "txt"
    (parse_dir / "images").mkdir(parents=True)
    (parse_dir / "paper_content_list_v2.json").write_text(
        json.dumps(_content()), encoding="utf-8"
    )
    (parse_dir / "images" / "table.jpg").write_bytes(b"table")
    (parse_dir / "images" / "figure.jpg").write_bytes(b"figure")
    (parse_dir / "images" / "equation.jpg").write_bytes(b"equation")

    parsed = MinerUPrimaryParser(run_mineru=lambda *_args, **_kwargs: None).parse(
        pdf_path, paper_id="paper-1", title="Paper", output_dir=output
    )

    assert parsed.metadata["parser"] == "mineru-primary"
    assert [block.text for block in parsed.blocks] == [
        "1. Method",
        "We define $k_{max}=20$ for every sample.",
    ]
    table = next(element for element in parsed.elements if element.element_type.value == "table")
    assert table.structured_data["columns"] == ["Method", "ACC"]
    assert table.structured_data["rows"] == [{"Method": "STG", "ACC": "0.77"}]
    assert "<table>" not in table.content
    assert all("Journal header" not in chunk.text for chunk in parsed.chunks)
    assert any(chunk.chunk_type == "table" for chunk in parsed.chunks)
    assert any(chunk.chunk_type == "figure" for chunk in parsed.chunks)
    equation = next(
        element
        for element in parsed.elements
        if element.element_type.value == "equation"
    )
    assert equation.latex == "s_{ij}=d_{ij}^{-1}"
    assert any(chunk.chunk_type == "equation" for chunk in parsed.chunks)


def test_mineru_primary_rejects_missing_content_list(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    _pdf(pdf_path)

    parser = MinerUPrimaryParser(run_mineru=lambda *_args, **_kwargs: None)

    try:
        parser.parse(pdf_path, paper_id="paper-1", title="Paper", output_dir=tmp_path / "out")
    except RuntimeError as exc:
        assert "content_list_v2" in str(exc)
    else:
        raise AssertionError("missing MinerU output must trigger the fallback path")


def test_mineru_primary_rejects_truncated_page_output(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    document = fitz.open()
    document.new_page(width=600, height=800)
    document.new_page(width=600, height=800)
    document.save(pdf_path)
    document.close()
    output = tmp_path / "out"
    parse_dir = output / "paper" / "txt"
    parse_dir.mkdir(parents=True)
    (parse_dir / "paper_content_list_v2.json").write_text(
        json.dumps(_content()), encoding="utf-8"
    )

    parser = MinerUPrimaryParser(run_mineru=lambda *_args, **_kwargs: None)
    try:
        parser.parse(pdf_path, paper_id="paper-1", title="Paper", output_dir=output)
    except RuntimeError as exc:
        assert "page count" in str(exc)
    else:
        raise AssertionError("truncated MinerU output must trigger the fallback path")


def test_mineru_primary_preserves_colspan_and_duplicate_headers(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    _pdf(pdf_path)
    output = tmp_path / "out"
    parse_dir = output / "paper" / "txt"
    parse_dir.mkdir(parents=True)
    payload = [[
        {
            "type": "paragraph",
            "content": {
                "paragraph_content": [{"type": "text", "content": "Evaluation results."}]
            },
            "bbox": [40, 60, 500, 90],
        },
        {
            "type": "table",
            "content": {
                "table_caption": [],
                "html": (
                    "<table><tr><td colspan='2'>Metric</td><td></td></tr>"
                    "<tr><td rowspan='2'>A</td><td>ACC</td><td>0.8</td></tr>"
                    "<tr><td>NMI</td><td>0.7</td></tr></table>"
                ),
            },
            "bbox": [40, 100, 500, 300],
        },
    ]]
    (parse_dir / "paper_content_list_v2.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    parsed = MinerUPrimaryParser(run_mineru=lambda *_args: None).parse(
        pdf_path, paper_id="paper-1", title="Paper", output_dir=output
    )

    table = parsed.elements[0].structured_data
    assert table["table_id"] == "Table 1"
    assert table["columns"] == ["Metric", "Metric_2", "column_3"]
    assert table["rows"] == [
        {"Metric": "A", "Metric_2": "ACC", "column_3": "0.8"},
        {"Metric": "A", "Metric_2": "NMI", "column_3": "0.7"},
    ]


def test_mineru_primary_does_not_absorb_caption_text_into_table_id(tmp_path: Path) -> None:
    content = _content()
    content[0][2]["content"]["table_caption"] = [
        {"type": "text", "content": "Table 7Execution time (seconds)."}
    ]
    pdf_path = tmp_path / "paper.pdf"
    _pdf(pdf_path)
    output = tmp_path / "out"
    parse_dir = output / "paper" / "txt"
    (parse_dir / "images").mkdir(parents=True)
    (parse_dir / "paper_content_list_v2.json").write_text(
        json.dumps(content), encoding="utf-8"
    )

    parsed = MinerUPrimaryParser(run_mineru=lambda *_args: None).parse(
        pdf_path, paper_id="paper-1", title="Paper", output_dir=output
    )

    table = next(element for element in parsed.elements if element.element_type.value == "table")
    assert table.structured_data["table_id"] == "Table 7"


def test_primary_success_does_not_mix_fallback_output(tmp_path: Path) -> None:
    expected = object()
    calls: list[str] = []

    class Primary:
        def parse(self, *_args, **_kwargs):
            calls.append("primary")
            return expected

    class Fallback:
        def run(self, *_args, **_kwargs):
            calls.append("fallback")
            raise AssertionError("fallback must not run")

    result = run_primary_with_fallback(
        Primary(), Fallback(), tmp_path / "paper.pdf", "1", "Paper", tmp_path
    )

    assert result.document is expected
    assert calls == ["primary"]


def test_primary_failure_uses_existing_pipeline_once(tmp_path: Path) -> None:
    expected = object()
    calls: list[str] = []

    class Primary:
        def parse(self, *_args, **_kwargs):
            calls.append("primary")
            raise RuntimeError("mineru failed")

    class Fallback:
        def run(self, *_args, **_kwargs):
            calls.append("fallback")
            return PipelineResult(document=expected, staging_dir=tmp_path)

    result = run_primary_with_fallback(
        Primary(), Fallback(), tmp_path / "paper.pdf", "1", "Paper", tmp_path
    )

    assert result.document is expected
    assert calls == ["primary", "fallback"]
