from pathlib import Path

from backend.app.ingestion.pipeline import PDFIngestionPipeline
from backend.tests.pdf_factory import headings_pdf


def test_pdf_pipeline_builds_structured_document_and_staging(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(headings_pdf())

    result = PDFIngestionPipeline().run(
        pdf_path,
        paper_id="paper-1",
        title="Paper",
        output_dir=tmp_path / "staging",
    )

    assert result.document.report.status.value in {"success", "success_with_warnings"}
    assert result.document.pages
    assert result.document.blocks
    assert result.document.chunks
    assert result.staging_dir.exists()
    assert (result.staging_dir / "page-1.png").exists()
    assert (result.staging_dir / "manifest.json").exists()


def test_pdf_pipeline_keeps_local_element_failure_non_fatal(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(headings_pdf())

    pipeline = PDFIngestionPipeline()
    pipeline.figure_parser = type(
        "FailingFigures",
        (),
        {"parse": lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("figure"))},
    )()
    result = pipeline.run(
        pdf_path,
        paper_id="paper-2",
        title="Paper",
        output_dir=tmp_path / "staging",
    )

    assert result.document.report.status.value in {"success", "success_with_warnings"}
    assert any(w.code == "figure_parser_failed" for w in result.document.report.warnings)
