"""End-to-end, PDF-only academic document ingestion orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

from .chunking import StructureAwareChunker
from .domain import ParsedDocument, ParseReport, ParseStatus, ParseWarning
from .equations import EquationParser
from .extraction import TextExtractor
from .figures import FigureParser
from .layout import (
    LayoutAnalyzer,
    ReadingOrderResolver,
    RepeatedMarginDetector,
    compute_body_stats,
)
from .quality import ParseQualityChecker
from .references import CrossReferenceResolver, ReferenceParser
from .structure import DocumentStructureParser
from .tables import TableParser


@dataclass(frozen=True)
class PipelineResult:
    document: ParsedDocument
    staging_dir: Path
    timings_ms: dict[str, int] = field(default_factory=dict)


class PDFIngestionPipeline:
    """Run extraction, layout, structure, element parsing and chunking.

    Each figure/table/equation parser is isolated: a local failure is retained
    as a quality warning and never discards the PDF Text Layer result.
    """

    def __init__(
        self,
        *,
        extractor: TextExtractor | None = None,
        layout_analyzer: LayoutAnalyzer | None = None,
        margin_detector: RepeatedMarginDetector | None = None,
        reading_order: ReadingOrderResolver | None = None,
        structure_parser: DocumentStructureParser | None = None,
        figure_parser: FigureParser | None = None,
        table_parser: TableParser | None = None,
        equation_parser: EquationParser | None = None,
        reference_parser: ReferenceParser | None = None,
        reference_resolver: CrossReferenceResolver | None = None,
        chunker: StructureAwareChunker | None = None,
        quality_checker: ParseQualityChecker | None = None,
    ) -> None:
        self.extractor = extractor or TextExtractor()
        self.layout_analyzer = layout_analyzer or LayoutAnalyzer()
        self.margin_detector = margin_detector or RepeatedMarginDetector()
        self.reading_order = reading_order or ReadingOrderResolver()
        self.structure_parser = structure_parser or DocumentStructureParser()
        self.figure_parser = figure_parser or FigureParser()
        self.table_parser = table_parser or TableParser()
        self.equation_parser = equation_parser or EquationParser()
        self.reference_parser = reference_parser or ReferenceParser()
        self.reference_resolver = reference_resolver or CrossReferenceResolver()
        self.chunker = chunker or StructureAwareChunker()
        self.quality_checker = quality_checker or ParseQualityChecker()

    def run(
        self,
        pdf_path: Path,
        *,
        paper_id: str,
        title: str = "",
        output_dir: Path,
    ) -> PipelineResult:
        pdf_path = Path(pdf_path)
        staging_dir = Path(output_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        report = ParseReport(status=ParseStatus.PARSING)
        timings: dict[str, int] = {}
        total_started = perf_counter()
        extract_started = perf_counter()
        try:
            pages = self.extractor.extract(pdf_path, paper_id)
        except Exception as exc:
            report.status = ParseStatus.FAILED
            report.errors.append(f"text_extraction_failed: {exc}")
            document = ParsedDocument(
                uid=paper_id,
                source_path=pdf_path,
                title=title,
                report=report,
            )
            timings["extract_render"] = round((perf_counter() - extract_started) * 1000)
            timings["total"] = round((perf_counter() - total_started) * 1000)
            return PipelineResult(document=document, staging_dir=staging_dir, timings_ms=timings)

        for page in pages:
            self.extractor.render_page(
                pdf_path,
                page.page_number,
                staging_dir / f"page-{page.page_number}.png",
            )
        timings["extract_render"] = round((perf_counter() - extract_started) * 1000)

        layout_started = perf_counter()
        body_stats = compute_body_stats(pages)
        margins = self.margin_detector.detect(pages, body_stats)
        diagnoses = [
            self.layout_analyzer.detect_columns(page, margins) for page in pages
        ]
        ordered_result = self.reading_order.resolve_document(
            pages,
            diagnoses,
            margins,
        )
        ordered_raw = [item.block for item in ordered_result.ordered_blocks]
        classified = list(
            self.layout_analyzer.classify_blocks(
                pages=pages,
                ordered_blocks=ordered_raw,
                body_stats=body_stats,
            )
        )
        timings["layout_structure"] = round((perf_counter() - layout_started) * 1000)

        warnings: list[ParseWarning] = []
        elements = []
        parser_specs = (
            ("figure", self.figure_parser, "figures"),
            ("table", self.table_parser, "tables"),
            ("equation", self.equation_parser, "equations"),
        )
        element_timings = {"figure": 0, "table": 0, "equation": 0}
        for page in pages:
            page_blocks = [
                block for block in classified if block.page_number == page.page_number
            ]
            for kind, parser, directory in parser_specs:
                element_started = perf_counter()
                try:
                    elements.extend(
                        parser.parse(
                            pdf_path,
                            page,
                            page_blocks,
                            staging_dir / directory,
                            title=title,
                        )
                        if kind == "figure"
                        else parser.parse(
                            pdf_path,
                            page,
                            page_blocks,
                            staging_dir / directory,
                        )
                    )
                except Exception as exc:
                    warnings.append(
                        ParseWarning(
                            code=f"{kind}_parser_failed",
                            message=str(exc),
                            page_number=page.page_number,
                            stage=kind,
                        )
                    )
                finally:
                    element_timings[kind] += round(
                        (perf_counter() - element_started) * 1000
                    )
        timings["table_parse"] = element_timings["table"]
        timings["figure_detection"] = element_timings["figure"]
        timings["equation_detection"] = element_timings["equation"]

        structure_started = perf_counter()
        sections, enriched_blocks, enriched_elements = (
            self.structure_parser.parse_with_elements(classified, elements)
        )
        references = self.reference_parser.parse(enriched_blocks)
        references = self.reference_resolver.resolve(
            references,
            elements=enriched_elements,
            sections=sections,
        )
        timings["layout_structure"] += round(
            (perf_counter() - structure_started) * 1000
        )
        chunk_started = perf_counter()
        chunks = self.chunker.chunk(
            enriched_blocks,
            title=title,
            elements=enriched_elements,
        )
        timings["chunking"] = round((perf_counter() - chunk_started) * 1000)
        report.warnings.extend(warnings)
        document = ParsedDocument(
            uid=paper_id,
            source_path=pdf_path,
            title=title,
            pages=pages,
            blocks=enriched_blocks,
            layout_diagnoses=diagnoses,
            sections=sections,
            elements=enriched_elements,
            chunks=chunks,
            cross_references=references,
            report=report,
            metadata={"body_stats": asdict(body_stats)},
        )
        self.quality_checker.check(document)
        (staging_dir / "manifest.json").write_text(
            json.dumps(document.report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        timings["total"] = round((perf_counter() - total_started) * 1000)
        return PipelineResult(
            document=document,
            staging_dir=staging_dir,
            timings_ms=timings,
        )
