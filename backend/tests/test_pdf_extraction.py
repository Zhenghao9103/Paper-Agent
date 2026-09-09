import json
from importlib import import_module
from pathlib import Path

import fitz
import pytest
from backend.app.ingestion.domain import (
    BBox,
    BlockType,
    ElementParseStatus,
    ElementType,
    ParsedDocument,
    ParsedElement,
    ParseReport,
    ParseStatus,
    ParseWarning,
    RawSpan,
    SectionNode,
    StructuredChunk,
)
from pdf_factory import (
    PAGE_HEIGHT,
    PAGE_WIDTH,
    headings_pdf,
    insert_block,
    make_pdf,
    numbered_equation_pdf,
    one_column_pdf,
    ruled_table_pdf,
    two_column_pdf,
    vector_figure_pdf,
)


def _text_extractor_type():
    return import_module("backend.app.ingestion.extraction").TextExtractor


def _write_pdf(tmp_path: Path, content: bytes, name: str = "paper.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


@pytest.mark.parametrize(
    ("enum_type", "required_members"),
    [
        (
            BlockType,
            {
                "TITLE": "title",
                "AUTHOR": "author",
                "ABSTRACT": "abstract",
                "HEADING": "heading",
                "SUBHEADING": "subheading",
                "PARAGRAPH": "paragraph",
                "LIST": "list",
                "FIGURE": "figure",
                "FIGURE_CAPTION": "figure_caption",
                "TABLE": "table",
                "TABLE_CAPTION": "table_caption",
                "EQUATION": "equation",
                "FOOTNOTE": "footnote",
                "HEADER": "header",
                "FOOTER": "footer",
                "REFERENCE": "reference",
                "UNKNOWN": "unknown",
            },
        ),
        (ElementType, {"FIGURE": "figure", "TABLE": "table", "EQUATION": "equation"}),
        (
            ParseStatus,
            {
                "PARSING": "parsing",
                "SUCCESS": "success",
                "SUCCESS_WITH_WARNINGS": "success_with_warnings",
                "FAILED": "failed",
            },
        ),
        (
            ElementParseStatus,
            {"SUCCESS": "success", "PARTIAL": "partial", "FAILED": "failed"},
        ),
    ],
)
def test_enum_contains_required_unique_name_value_mappings(enum_type, required_members) -> None:
    actual_members = {
        name: member.value for name, member in enum_type.__members__.items()
    }

    assert required_members.items() <= actual_members.items()
    assert len(actual_members.values()) == len(set(actual_members.values()))


def test_bbox_rejects_inverted_coordinates() -> None:
    with pytest.raises(ValueError, match="x0 must not exceed x1"):
        BBox(10, 0, 5, 20)

    with pytest.raises(ValueError, match="y0 must not exceed y1"):
        BBox(0, 20, 10, 5)


@pytest.mark.parametrize("coordinate_index", range(4))
@pytest.mark.parametrize("non_finite", ["nan", "inf", "-inf"])
def test_bbox_rejects_non_finite_coordinates_after_float_coercion(
    coordinate_index: int,
    non_finite: str,
) -> None:
    coordinates: list[float | str] = [0, 0, 10, 10]
    coordinates[coordinate_index] = non_finite

    with pytest.raises(ValueError, match="coordinates must be finite"):
        BBox(*coordinates)


def test_bbox_union_and_page_bounds() -> None:
    first = BBox(10, 20, 30, 50)
    second = BBox(5, 25, 40, 60)

    assert first.width == 20.0
    assert first.height == 30.0
    assert first.area == 600.0
    assert first.union(second) == BBox(5, 20, 40, 60)
    assert first.as_list() == [10.0, 20.0, 30.0, 50.0]
    assert first.contains_within(612, 792)
    assert not BBox(-1, 0, 10, 10).contains_within(612, 792)
    assert not BBox(0, 0, 613, 10).contains_within(612, 792)


def test_raw_span_retains_font_metrics_and_serializes_bbox() -> None:
    span = RawSpan(
        text="Graph",
        bbox=BBox(72, 100, 110, 112),
        font="Times-Roman",
        size=10.5,
        flags=4,
        color=0,
    )

    assert span.font == "Times-Roman"
    assert span.size == 10.5
    assert span.bbox == BBox(72, 100, 110, 112)
    assert span.to_dict() == {
        "text": "Graph",
        "bbox": [72.0, 100.0, 110.0, 112.0],
        "font": "Times-Roman",
        "size": 10.5,
        "flags": 4,
        "color": 0,
        "origin": None,
    }


def test_raw_text_contract_retains_orientation_and_origin() -> None:
    from backend.app.ingestion import domain

    span = domain.RawSpan(
        text="Graph",
        bbox=BBox(72, 100, 110, 112),
        font="Times-Roman",
        size=10.5,
        origin=(72, 110),
    )
    line = domain.RawLine(
        text="Graph",
        bbox=BBox(72, 100, 110, 112),
        spans=(span,),
        direction=(0, -1),
        wmode=1,
    )

    assert span.origin == (72.0, 110.0)
    assert line.direction == (0.0, -1.0)
    assert line.wmode == 1
    assert line.to_dict()["direction"] == [0.0, -1.0]


@pytest.mark.parametrize(
    "origin",
    [(1,), (1, 2, 3), (float("nan"), 1), (float("inf"), 1), ("invalid", 1)],
)
def test_raw_span_rejects_malformed_origin(origin) -> None:
    from backend.app.ingestion import domain

    with pytest.raises(
        ValueError, match="origin must contain exactly two finite numeric values"
    ):
        domain.RawSpan(
            text="Graph",
            bbox=BBox(72, 100, 110, 112),
            font="Times-Roman",
            size=10.5,
            origin=origin,
        )


@pytest.mark.parametrize(
    "direction",
    [(1,), (1, 2, 3), (1, float("nan")), (1, float("inf")), (1, "invalid")],
)
def test_raw_line_rejects_malformed_direction(direction) -> None:
    from backend.app.ingestion import domain

    with pytest.raises(
        ValueError, match="direction must contain exactly two finite numeric values"
    ):
        domain.RawLine(
            text="Graph",
            bbox=BBox(72, 100, 110, 112),
            direction=direction,
        )


def test_image_region_and_page_layout_are_json_safe_and_immutable() -> None:
    from backend.app.ingestion import domain

    source_metadata = {"colorspace": 3, "transform": [40, 0, 0, 40, 100, 100]}
    image = domain.ImageRegion(
        source_index=2,
        bbox=BBox(100, 100, 140, 140),
        width=2,
        height=2,
        ext="png",
        metadata=source_metadata,
    )
    page = domain.PageLayout(
        page_number=1,
        width=PAGE_WIDTH,
        height=PAGE_HEIGHT,
        image_regions=(image,),
    )
    source_metadata["colorspace"] = 1

    assert page.image_regions == (image,)
    assert image.metadata["colorspace"] == 3
    with pytest.raises(TypeError):
        image.metadata["colorspace"] = 1
    assert json.loads(json.dumps(page.to_dict())) == page.to_dict()


def test_parse_report_serializes_status_counts_and_warnings() -> None:
    report = ParseReport(
        status=ParseStatus.SUCCESS_WITH_WARNINGS,
        page_count=2,
        block_count=12,
        element_count=3,
        chunk_count=5,
        warnings=[ParseWarning(code="cropped", message="Caption touches page edge", page_number=2)],
    )

    payload = report.to_dict()

    assert payload["status"] == "success_with_warnings"
    assert payload["page_count"] == 2
    assert payload["block_count"] == 12
    assert payload["element_count"] == 3
    assert payload["chunk_count"] == 5
    assert payload["warning_count"] == 1
    assert payload["warnings"][0]["code"] == "cropped"
    assert json.loads(json.dumps(payload)) == payload


def test_mutable_aggregation_lists_are_instance_isolated() -> None:
    first_section = SectionNode(uid="section-1", title="Introduction", level=1)
    second_section = SectionNode(uid="section-2", title="Methods", level=1)
    first_section.children.append(SectionNode(uid="section-1-1", title="Motivation", level=2))

    first_report = ParseReport()
    second_report = ParseReport()
    first_report.warnings.append(ParseWarning(code="test", message="first only"))

    first_document = ParsedDocument(uid="doc-1", source_path=Path("paper.pdf"))
    second_document = ParsedDocument(uid="doc-2", source_path=Path("other.pdf"))
    first_document.sections.append(first_section)

    assert second_section.children == []
    assert second_report.warnings == []
    assert second_document.sections == []
    assert first_document.to_dict()["source_path"] == "paper.pdf"


def _make_record_with_metadata(record_type, metadata):
    if record_type is ParsedElement:
        return ParsedElement(
            uid="element-1",
            element_type=ElementType.FIGURE,
            status=ElementParseStatus.SUCCESS,
            page_number=1,
            bbox=BBox(10, 20, 30, 40),
            metadata=metadata,
        )
    return StructuredChunk(
        uid="chunk-1",
        text="Chunk text",
        page_numbers=(1,),
        metadata=metadata,
    )


@pytest.mark.parametrize("record_type", [ParsedElement, StructuredChunk])
def test_frozen_record_metadata_is_detached_from_source_containers(record_type) -> None:
    source = {
        "nested": {"scores": [0.8, 0.9]},
        "labels": {"primary", "verified"},
    }
    record = _make_record_with_metadata(record_type, source)

    source["nested"]["scores"].append(1.0)
    source["labels"].add("mutated")
    source["new"] = "mutated"

    assert record.to_dict()["metadata"] == {
        "labels": ["primary", "verified"],
        "nested": {"scores": [0.8, 0.9]},
    }
    assert json.loads(json.dumps(record.to_dict())) == record.to_dict()


@pytest.mark.parametrize("record_type", [ParsedElement, StructuredChunk])
def test_frozen_record_metadata_rejects_direct_and_nested_mutation(record_type) -> None:
    record = _make_record_with_metadata(
        record_type,
        {"nested": {"score": 0.9}, "labels": ["primary"]},
    )

    with pytest.raises(TypeError):
        record.metadata["new"] = "mutated"
    with pytest.raises(TypeError):
        record.metadata["nested"]["score"] = 0.1
    with pytest.raises(TypeError):
        record.metadata["labels"][0] = "mutated"


@pytest.mark.parametrize(
    ("builder", "expected_text"),
    [
        (one_column_pdf, ["One-column paper", "single readable column"]),
        (two_column_pdf, ["A", "B", "C", "D", "E", "F"]),
        (headings_pdf, ["Abstract", "1 Introduction", "1.1 Motivation"]),
        (ruled_table_pdf, ["Method", "Score", "Proposed", "0.95"]),
        (vector_figure_pdf, ["Figure 1", "Vector model overview"]),
        (numbered_equation_pdf, ["E = mc", "(1)"]),
    ],
)
def test_named_pdf_builders_return_openable_single_page_pdf(builder, expected_text) -> None:
    content = builder()

    with fitz.open(stream=content, filetype="pdf") as document:
        assert document.page_count == 1
        text = document[0].get_text("text")
        for fragment in expected_text:
            assert fragment in text


def test_make_pdf_uses_one_fixed_size_page_per_callback() -> None:
    content = make_pdf(
        [
            lambda page: page.insert_text((72, 72), "Page one"),
            lambda page: page.insert_text((72, 72), "Page two"),
        ]
    )

    with fitz.open(stream=content, filetype="pdf") as document:
        assert document.page_count == 2
        assert [(page.rect.width, page.rect.height) for page in document] == [
            (612.0, 792.0),
            (612.0, 792.0),
        ]
        assert document[0].get_text("text").strip() == "Page one"
        assert document[1].get_text("text").strip() == "Page two"


def test_repeated_pdf_builds_are_byte_identical() -> None:
    assert one_column_pdf() == one_column_pdf()


def test_text_extractor_preserves_page_order_source_indices_and_span_metrics(
    tmp_path: Path,
) -> None:
    def first_page(page: fitz.Page) -> None:
        insert_block(page, (72, 54, 540, 90), "First page title", size=18, fontname="hebo")
        insert_block(page, (72, 112, 540, 160), "First line\nSecond line", size=11)

    def second_page(page: fitz.Page) -> None:
        insert_block(page, (72, 54, 540, 90), "Second page", size=14)

    path = _write_pdf(tmp_path, make_pdf([first_page, second_page]))

    pages = _text_extractor_type()().extract(path, paper_id="paper-1")

    assert [page.page_number for page in pages] == [1, 2]
    assert [(page.width, page.height) for page in pages] == [
        (PAGE_WIDTH, PAGE_HEIGHT),
        (PAGE_WIDTH, PAGE_HEIGHT),
    ]
    assert [block.block_index for block in pages[0].blocks] == [0, 1]
    assert pages[0].blocks[0].text == "First page title"
    assert pages[0].blocks[1].text == "First line\nSecond line"
    title_span = pages[0].blocks[0].lines[0].spans[0]
    assert title_span.font == "Helvetica-Bold"
    assert title_span.size == 18.0
    assert title_span.origin is not None
    assert pages[0].blocks[0].lines[0].direction == (1.0, 0.0)
    assert pages[0].blocks[0].lines[0].wmode == 0
    assert all(
        span.bbox.contains_within(page.width, page.height)
        for page in pages
        for block in page.blocks
        for line in block.lines
        for span in line.spans
    )


def test_text_extractor_calls_unsorted_dict_extraction_once_per_page(monkeypatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FakePage:
        def get_text(self, *args, **kwargs):
            calls.append((args, kwargs))
            return {"width": PAGE_WIDTH, "height": PAGE_HEIGHT, "blocks": []}

    class FakeDocument:
        page_count = 1
        closed = False

        def __iter__(self):
            return iter([FakePage()])

        def close(self) -> None:
            self.closed = True

    document = FakeDocument()
    extraction = import_module("backend.app.ingestion.extraction")
    monkeypatch.setattr(extraction.fitz, "open", lambda path: document)

    pages = extraction.TextExtractor().extract(Path("paper.pdf"), paper_id="paper-1")

    assert len(pages) == 1
    assert calls == [(('dict',), {"sort": False})]
    assert document.closed


def test_text_extractor_rejects_zero_page_pdf_and_closes_document(monkeypatch) -> None:
    class EmptyDocument:
        page_count = 0
        closed = False

        def close(self) -> None:
            self.closed = True

    document = EmptyDocument()
    extraction = import_module("backend.app.ingestion.extraction")
    monkeypatch.setattr(extraction.fitz, "open", lambda path: document)

    with pytest.raises(ValueError, match="PDF contains no pages"):
        extraction.TextExtractor().extract(Path("empty.pdf"), paper_id="paper-1")

    assert document.closed


def test_text_extractor_closes_document_when_page_extraction_fails(monkeypatch) -> None:
    class BrokenPage:
        def get_text(self, *args, **kwargs):
            raise RuntimeError("extraction failed")

    class BrokenDocument:
        page_count = 1
        closed = False

        def __iter__(self):
            return iter([BrokenPage()])

        def close(self) -> None:
            self.closed = True

    document = BrokenDocument()
    extraction = import_module("backend.app.ingestion.extraction")
    monkeypatch.setattr(extraction.fitz, "open", lambda path: document)

    with pytest.raises(RuntimeError, match="extraction failed"):
        extraction.TextExtractor().extract(Path("broken.pdf"), paper_id="paper-1")

    assert document.closed


def test_text_extractor_retains_image_regions_without_adding_image_bytes_to_text(
    tmp_path: Path,
) -> None:
    def populate(page: fitz.Page) -> None:
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 2, 2), False)
        pixmap.clear_with(0xFF0000)
        page.insert_image(fitz.Rect(100, 100, 140, 140), pixmap=pixmap)
        insert_block(page, (72, 180, 540, 220), "Caption text", size=10)

    path = _write_pdf(tmp_path, make_pdf([populate]), "image.pdf")

    page = _text_extractor_type()().extract(path, paper_id="paper-image")[0]

    assert [block.text for block in page.blocks] == ["Caption text"]
    assert len(page.image_regions) == 1
    image = page.image_regions[0]
    assert image.source_index == 0
    assert image.bbox == BBox(100, 100, 140, 140)
    assert (image.width, image.height, image.ext) == (2, 2, "png")
    assert image.metadata["colorspace"] == 3
    assert "image" not in image.metadata


def test_text_extractor_omits_binary_image_metadata(monkeypatch) -> None:
    class FakePage:
        def get_text(self, *args, **kwargs):
            return {
                "width": PAGE_WIDTH,
                "height": PAGE_HEIGHT,
                "blocks": [
                    {
                        "type": 1,
                        "number": 0,
                        "bbox": (100, 100, 140, 140),
                        "width": 2,
                        "height": 2,
                        "ext": "png",
                        "image": b"image bytes",
                        "mask": b"mask bytes",
                        "nested": {
                            "keep": "value",
                            "drop": bytearray(b"nested bytes"),
                        },
                        "sequence": [
                            1,
                            memoryview(b"sequence bytes"),
                            {"keep": 2, "drop": b"mapping bytes"},
                        ],
                    }
                ],
            }

    class FakeDocument:
        page_count = 1

        def __iter__(self):
            return iter([FakePage()])

        def close(self) -> None:
            pass

    extraction = import_module("backend.app.ingestion.extraction")
    monkeypatch.setattr(extraction.fitz, "open", lambda path: FakeDocument())

    page = extraction.TextExtractor().extract(Path("image.pdf"), paper_id="paper-image")[0]

    assert "image" not in page.image_regions[0].metadata
    assert "mask" not in page.image_regions[0].metadata
    assert page.image_regions[0].metadata["nested"] == {"keep": "value"}
    assert page.image_regions[0].metadata["sequence"] == (1, {"keep": 2})
    json.dumps(page.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("binary_collection", "expected_collection"),
    [
        ({"beta", b"set bytes", "alpha"}, ["alpha", "beta"]),
        (frozenset({"delta", b"frozenset bytes", "gamma"}), ["delta", "gamma"]),
        ({("nested", b"set tuple bytes"), "ok"}, ["ok", ["nested"]]),
        (
            frozenset({("nested", b"frozenset tuple bytes"), "ok"}),
            ["ok", ["nested"]],
        ),
    ],
    ids=["set", "frozenset", "set-with-tuple", "frozenset-with-tuple"],
)
def test_text_extractor_recursively_omits_binary_metadata_from_collections(
    monkeypatch, binary_collection, expected_collection
) -> None:
    class FakePage:
        def get_text(self, *args, **kwargs):
            return {
                "width": PAGE_WIDTH,
                "height": PAGE_HEIGHT,
                "blocks": [
                    {
                        "type": 1,
                        "number": 0,
                        "bbox": (100, 100, 140, 140),
                        "nested": {
                            "collection": binary_collection,
                            "sequence": [
                                bytearray(b"sequence bytes"),
                                {
                                    "keep": "value",
                                    "drop": memoryview(b"mapping bytes"),
                                },
                            ],
                        },
                    }
                ],
            }

    class FakeDocument:
        page_count = 1

        def __iter__(self):
            return iter([FakePage()])

        def close(self) -> None:
            pass

    extraction = import_module("backend.app.ingestion.extraction")
    monkeypatch.setattr(extraction.fitz, "open", lambda path: FakeDocument())

    page = extraction.TextExtractor().extract(Path("image.pdf"), paper_id="paper-image")[0]

    metadata = page.to_dict()["image_regions"][0]["metadata"]
    assert metadata["nested"] == {
        "collection": expected_collection,
        "sequence": [{"keep": "value"}],
    }
    json.dumps(page.to_dict(), allow_nan=False)


def test_render_page_closes_document_when_rendering_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenPage:
        def get_pixmap(self, *args, **kwargs):
            raise RuntimeError("render failed")

    class SpyDocument:
        page_count = 1
        closed = False

        def load_page(self, page_index: int):
            return BrokenPage()

        def close(self) -> None:
            self.closed = True

    output_path = tmp_path / "page.png"
    output_path.write_bytes(b"existing destination")
    document = SpyDocument()
    extraction = import_module("backend.app.ingestion.extraction")
    monkeypatch.setattr(extraction.fitz, "open", lambda path: document)

    with pytest.raises(RuntimeError, match="render failed"):
        extraction.TextExtractor().render_page(
            Path("paper.pdf"), 1, output_path
        )

    assert document.closed
    assert output_path.read_bytes() == b"existing destination"
    assert list(tmp_path.iterdir()) == [output_path]


def test_render_page_preserves_destination_and_cleans_temp_when_save_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingPixmap:
        def save(self, path: Path) -> None:
            Path(path).write_bytes(b"partial PNG")
            raise OSError("save failed")

    class FakePage:
        def get_pixmap(self, *args, **kwargs):
            return FailingPixmap()

    class SpyDocument:
        page_count = 1
        closed = False

        def load_page(self, page_index: int):
            return FakePage()

        def close(self) -> None:
            self.closed = True

    output_path = tmp_path / "page.png"
    output_path.write_bytes(b"existing destination")
    document = SpyDocument()
    extraction = import_module("backend.app.ingestion.extraction")
    monkeypatch.setattr(extraction.fitz, "open", lambda path: document)

    with pytest.raises(OSError, match="save failed"):
        extraction.TextExtractor().render_page(Path("paper.pdf"), 1, output_path)

    assert document.closed
    assert output_path.read_bytes() == b"existing destination"
    assert list(tmp_path.iterdir()) == [output_path]


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_render_page_rejects_non_positive_or_non_finite_scale_before_output_creation(
    tmp_path: Path, scale: float
) -> None:
    path = _write_pdf(tmp_path, one_column_pdf())
    output_parent = tmp_path / "nested"

    with pytest.raises(ValueError, match="scale must be finite and greater than 0"):
        _text_extractor_type()().render_page(
            path, 1, output_parent / "page.png", scale=scale
        )

    assert not output_parent.exists()


def test_render_page_creates_scaled_nonempty_png(tmp_path: Path) -> None:
    path = _write_pdf(tmp_path, one_column_pdf())
    output_path = tmp_path / "nested" / "page.png"

    _text_extractor_type()().render_page(path, 1, output_path, scale=1.5)

    content = output_path.read_bytes()
    assert content.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(content) > 8
    rendered = fitz.Pixmap(output_path)
    assert (rendered.width, rendered.height) == (918, 1188)
    assert list(output_path.parent.iterdir()) == [output_path]


@pytest.mark.parametrize("page_number", [0, 2])
def test_render_page_rejects_page_numbers_outside_pdf(tmp_path: Path, page_number: int) -> None:
    path = _write_pdf(tmp_path, one_column_pdf())

    with pytest.raises(ValueError, match="page_number must be between 1 and 1"):
        _text_extractor_type()().render_page(path, page_number, tmp_path / "page.png")
