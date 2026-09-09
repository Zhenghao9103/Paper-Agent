import pytest
from backend.app.db.base import Base
from backend.app.ingestion.domain import (
    BBox,
    BlockType,
    ClassifiedBlock,
    ElementParseStatus,
    ElementType,
    PageLayout,
    ParsedDocument,
    ParsedElement,
    ParseReport,
    ParseStatus,
    StructuredChunk,
)
from backend.app.ingestion.quality import ParseQualityChecker
from backend.app.models import (
    Document,
    DocumentBlock,
    DocumentChunk,
    DocumentChunkDetail,
    DocumentPage,
    DocumentParseRun,
)
from backend.app.services.document_store import DocumentStore
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session


@pytest.fixture()
def db(tmp_path, monkeypatch):
    documents_root = tmp_path / "documents"
    monkeypatch.setattr(
        "backend.app.services.document_store.documents_dir",
        lambda: documents_root,
    )
    monkeypatch.setattr(
        "backend.app.services.document_store.document_dir",
        lambda document_id: documents_root / str(document_id),
    )
    engine = create_engine(f"sqlite:///{tmp_path / 'store.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _document(uid: str = "doc") -> ParsedDocument:
    page = PageLayout(page_number=1, width=100, height=100)
    block = ClassifiedBlock(
        uid=f"{uid}-block-1",
        block_type=BlockType.PARAGRAPH,
        page_number=1,
        bbox=BBox(10, 10, 90, 30),
        text="A useful body paragraph.",
        section="1 Introduction",
        section_uid="section-1",
    )
    chunk = StructuredChunk(
        uid=f"{uid}-chunk-1",
        text=block.text,
        page_numbers=(1,),
        section_path=("1 Introduction",),
        block_uids=(block.uid,),
        metadata={
            "chunk_type": "text",
            "bbox": block.bbox.as_list(),
            "token_count": 5,
            "embedding_text": "Paper\n1 Introduction\nA useful body paragraph.",
        },
    )
    return ParsedDocument(
        uid=uid,
        source_path="paper.pdf",
        title="Paper",
        pages=[page],
        blocks=[block],
        chunks=[chunk],
        report=ParseReport(status=ParseStatus.SUCCESS, page_count=1),
    )


def test_quality_checker_reports_success_for_usable_document():
    result = ParseQualityChecker().check(_document())
    assert result.status is ParseStatus.SUCCESS
    assert result.page_count == 1
    assert result.block_count == 1
    assert result.chunk_count == 1


def test_quality_checker_warns_for_partial_element_and_invalid_bbox():
    parsed = _document()
    parsed.blocks[0] = ClassifiedBlock(
        uid="bad-block",
        block_type=BlockType.PARAGRAPH,
        page_number=1,
        bbox=BBox(90, 90, 120, 120),
        text="Body",
    )
    parsed.elements = [
        ParsedElement(
            uid="table-1",
            element_type=ElementType.TABLE,
            status=ElementParseStatus.PARTIAL,
            page_number=1,
            bbox=BBox(10, 40, 90, 80),
            caption=None,
        )
    ]
    result = ParseQualityChecker().check(parsed)
    assert result.status is ParseStatus.SUCCESS_WITH_WARNINGS
    codes = {warning.code for warning in result.warnings}
    assert "bbox_out_of_bounds" in codes
    assert "partial_element" in codes
    assert "missing_caption" in codes


def test_quality_checker_fails_without_body_text():
    parsed = _document()
    parsed.blocks.clear()
    parsed.chunks.clear()
    parsed.report = ParseReport(status=ParseStatus.PARSING, page_count=1)
    result = ParseQualityChecker().check(parsed)
    assert result.status is ParseStatus.FAILED
    assert any(w.code == "no_usable_text" for w in result.warnings)


def test_quality_checker_preserves_page_count_mismatch_warning():
    parsed = _document()
    parsed.report = ParseReport(status=ParseStatus.PARSING, page_count=2)

    result = ParseQualityChecker().check(parsed)

    assert any(w.code == "page_count_mismatch" for w in result.warnings)


def test_document_store_persists_structured_and_legacy_rows(db, tmp_path):
    document = Document(
        title="Paper",
        file_type="pdf",
        file_path="paper.pdf",
        status="uploaded",
        source_type="uploaded",
    )
    db.add(document)
    db.flush()
    parsed = _document()
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "page-1.png").write_bytes(b"png")
    (staging / "manifest.json").write_text("{}", encoding="utf-8")
    run = DocumentStore().save(db, document, parsed, staging)
    assert run.status == "success"
    assert db.scalar(select(DocumentPage).where(DocumentPage.document_id == document.id))
    assert db.scalar(select(DocumentBlock).where(DocumentBlock.document_id == document.id))
    assert db.scalar(select(DocumentChunk).where(DocumentChunk.document_id == document.id))
    assert db.scalar(select(DocumentChunkDetail))
    assert db.scalar(select(DocumentParseRun).where(DocumentParseRun.document_id == document.id))


def test_document_store_reparse_replaces_current_rows(db, tmp_path):
    document = Document(
        title="Paper",
        file_type="pdf",
        file_path="paper.pdf",
        status="uploaded",
        source_type="uploaded",
    )
    db.add(document)
    db.flush()
    store = DocumentStore()
    for index in range(2):
        staging = tmp_path / f"staging-{index}"
        staging.mkdir()
        (staging / "manifest.json").write_text("{}", encoding="utf-8")
        parsed = _document(uid=f"doc-{index}")
        store.save(db, document, parsed, staging)
    assert db.query(DocumentBlock).filter_by(document_id=document.id).count() == 1
    assert db.query(DocumentChunk).filter_by(document_id=document.id).count() == 1
    assert db.query(DocumentParseRun).filter_by(document_id=document.id).count() == 2


def test_document_store_rejects_missing_staging_without_rows(db, tmp_path):
    document = Document(
        title="Paper",
        file_type="pdf",
        file_path="paper.pdf",
        status="uploaded",
        source_type="uploaded",
    )
    db.add(document)
    db.flush()
    with pytest.raises(FileNotFoundError):
        DocumentStore().save(db, document, _document(), tmp_path / "missing")
    db.rollback()
    assert db.query(DocumentBlock).filter_by(document_id=document.id).count() == 0
