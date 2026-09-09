import json

from backend.app.cli.report_ingestion_timings import build_report


def test_timing_report_includes_per_document_and_percentiles(tmp_path) -> None:
    for document_id, total in ((1, 100), (2, 300)):
        directory = tmp_path / str(document_id)
        directory.mkdir()
        (directory / "pipeline-status.json").write_text(
            json.dumps(
                {
                    "pipeline_stage": "indexed",
                    "timings_ms": {"extract_render": total // 2, "total": total},
                }
            ),
            encoding="utf-8",
        )

    report = build_report(tmp_path)

    assert report["document_count"] == 2
    assert report["summary_ms"]["total"] == {"mean": 200.0, "p50": 200.0, "p95": 290.0}
