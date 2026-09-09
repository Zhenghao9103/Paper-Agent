import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool

from ...db.session import get_db
from ...models.analysis import PaperAnalysis
from ...models.chunk import DocumentChunk
from ...models.document import Document
from ...models.page import DocumentPage
from ...models.parsing import (
    DocumentBlock,
    DocumentChunkDetail,
    DocumentCrossReference,
    DocumentElement,
    DocumentParseRun,
)
from ...schemas.analysis import PaperAnalysisRead
from ...schemas.chunk import DocumentChunkRead
from ...schemas.document import DocumentRead
from ...schemas.page import DocumentPageRead
from ...schemas.parsing import (
    DocumentBlockRead,
    DocumentChunkDetailRead,
    DocumentCrossReferenceRead,
    DocumentElementRead,
    QualityReportRead,
)
from ...services.analysis import analyze_document
from ...services.document_jobs import enqueue_document
from ...services.figure_enrichment import enrich_figure
from ...services.ingestion import parse_and_store_document
from ...services.reset import clear_all_runtime_data
from ...services.storage import detect_file_type, infer_title, save_upload_file

router = APIRouter(prefix="/documents", tags=["documents"])


async def _create_uploaded_document(file: UploadFile, db: Session) -> Document:
    try:
        file_type = detect_file_type(file.filename or "")
        saved_path = await save_upload_file(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document = Document(
        title=infer_title(file.filename or "Untitled Document"),
        file_type=file_type,
        file_path=str(saved_path),
        status="uploaded",
        source_type="uploaded",
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


@router.get("", response_model=list[DocumentRead])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:
    statement = select(Document).order_by(Document.created_at.desc())
    return list(db.scalars(statement).all())


@router.post("/upload", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> Document:
    document = await _create_uploaded_document(file, db)
    # Parsing renders pages and runs BGE-M3, both CPU bound. Running it inline would
    # block the event loop for the whole upload, so hand it to the worker threadpool.
    await run_in_threadpool(parse_and_store_document, db, document)
    db.refresh(document)
    return document


@router.post(
    "/upload-async",
    response_model=DocumentRead,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> Document:
    document = await _create_uploaded_document(file, db)
    document.status = "parsing"
    db.add(document)
    db.commit()
    db.refresh(document)
    session_factory = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=db.get_bind(),
    )
    background_tasks.add_task(
        enqueue_document,
        document.id,
        session_factory=session_factory,
    )
    return document


@router.post("/{document_id}/reparse", response_model=DocumentRead)
async def reparse_document(
    document_id: int,
    db: Session = Depends(get_db),
) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.status == "parsing":
        raise HTTPException(
            status_code=409,
            detail="Document is already being parsed",
        )
    if not Path(document.file_path).is_file():
        raise HTTPException(status_code=409, detail="Source PDF not found")

    document.status = "parsing"
    db.add(document)
    db.commit()
    db.refresh(document)
    succeeded = await run_in_threadpool(parse_and_store_document, db, document)
    db.refresh(document)
    if not succeeded and document.status != "failed":
        document.status = "failed"
        db.add(document)
        db.commit()
        db.refresh(document)
    return document


@router.post("/clear")
def clear_documents_and_history(db: Session = Depends(get_db)) -> dict:
    deleted = clear_all_runtime_data(db)
    return {"ok": True, "deleted": deleted}


@router.get("/{document_id}", response_model=DocumentRead)
def get_document(document_id: int, db: Session = Depends(get_db)) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.get("/{document_id}/pages", response_model=list[DocumentPageRead])
def list_document_pages(document_id: int, db: Session = Depends(get_db)) -> list[DocumentPage]:
    if db.get(Document, document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")

    statement = (
        select(DocumentPage)
        .where(DocumentPage.document_id == document_id)
        .order_by(DocumentPage.page_number)
    )
    return list(db.scalars(statement).all())


@router.get("/{document_id}/chunks", response_model=list[DocumentChunkRead])
def list_document_chunks(document_id: int, db: Session = Depends(get_db)) -> list[dict]:
    if db.get(Document, document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")

    statement = (
        select(DocumentChunk, DocumentChunkDetail)
        .outerjoin(DocumentChunkDetail, DocumentChunkDetail.chunk_id == DocumentChunk.id)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.page_number, DocumentChunk.chunk_index)
    )
    return [
        {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "page_id": chunk.page_id,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "created_at": chunk.created_at,
            "detail": detail,
        }
        for chunk, detail in db.execute(statement).all()
    ]


@router.get("/{document_id}/blocks", response_model=list[DocumentBlockRead])
def list_document_blocks(
    document_id: int, db: Session = Depends(get_db)
) -> list[DocumentBlock]:
    if db.get(Document, document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    statement = (
        select(DocumentBlock)
        .where(DocumentBlock.document_id == document_id)
        .order_by(DocumentBlock.reading_order, DocumentBlock.id)
    )
    return list(db.scalars(statement).all())


@router.get("/{document_id}/elements", response_model=list[DocumentElementRead])
def list_document_elements(
    document_id: int, db: Session = Depends(get_db)
) -> list[dict]:
    if db.get(Document, document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    statement = (
        select(DocumentElement)
        .where(DocumentElement.document_id == document_id)
        .order_by(DocumentElement.page_number, DocumentElement.reading_order)
    )
    elements = list(db.scalars(statement).all())
    details = {
        detail.element_id: detail
        for detail in db.scalars(
            select(DocumentChunkDetail).where(
                DocumentChunkDetail.element_id.in_([element.id for element in elements])
            )
        ).all()
        if detail.element_id is not None
    } if elements else {}
    result = []
    for element in elements:
        detail = details.get(element.id)
        metadata = DocumentChunkDetailRead.model_validate(detail).metadata if detail else {}
        result.append(
            {
                **{
                    column.name: getattr(element, column.name)
                    for column in element.__table__.columns
                },
                "warning_codes": metadata.get("warning_codes", []),
                "metadata": metadata,
            }
        )
    return result


@router.get("/{document_id}/elements/{element_id}/image", response_class=FileResponse)
def get_document_element_image(
    document_id: int,
    element_id: int,
    db: Session = Depends(get_db),
) -> FileResponse:
    element = db.scalar(
        select(DocumentElement).where(
            DocumentElement.id == element_id,
            DocumentElement.document_id == document_id,
        )
    )
    if element is None or not element.image_path:
        raise HTTPException(status_code=404, detail="Element image not found")
    from ...core.paths import document_dir

    root = document_dir(document_id).resolve()
    image = Path(element.image_path).resolve()
    if not image.is_file() or not image.is_relative_to(root):
        raise HTTPException(status_code=404, detail="Element image not found")
    return FileResponse(image)


@router.post(
    "/{document_id}/elements/{element_id}/enrich",
    response_model=DocumentElementRead,
)
def enrich_document_figure(
    document_id: int,
    element_id: int,
    db: Session = Depends(get_db),
) -> dict:
    document = db.get(Document, document_id)
    element = db.scalar(
        select(DocumentElement).where(
            DocumentElement.id == element_id,
            DocumentElement.document_id == document_id,
        )
    )
    if document is None or element is None:
        raise HTTPException(status_code=404, detail="Figure element not found")
    if element.element_type != "figure":
        raise HTTPException(status_code=409, detail="Only figure elements can be enriched")
    try:
        enrich_figure(db, document, element)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return next(
        item
        for item in list_document_elements(document_id, db=db)
        if item["id"] == element_id
    )


@router.get(
    "/{document_id}/cross-references",
    response_model=list[DocumentCrossReferenceRead],
)
def list_document_cross_references(
    document_id: int, db: Session = Depends(get_db)
) -> list[DocumentCrossReference]:
    if db.get(Document, document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    statement = (
        select(DocumentCrossReference)
        .where(DocumentCrossReference.document_id == document_id)
        .order_by(DocumentCrossReference.id)
    )
    return list(db.scalars(statement).all())


@router.get("/{document_id}/quality", response_model=QualityReportRead)
def get_document_quality(document_id: int, db: Session = Depends(get_db)) -> dict:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    run = db.scalar(
        select(DocumentParseRun)
        .where(DocumentParseRun.document_id == document_id)
        .order_by(DocumentParseRun.id.desc())
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Parse report not found")
    from ...core.paths import document_dir

    manifest = {}
    try:
        manifest = json.loads(
            (document_dir(document_id) / "pipeline-status.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        pass
    chunk_count = db.scalar(
        select(func.count(DocumentChunk.id)).where(
            DocumentChunk.document_id == document_id
        )
    ) or 0
    return {
        "status": run.status,
        "page_count": run.page_count,
        "block_count": run.block_count,
        "element_count": run.element_count,
        "chunk_count": chunk_count,
        "text_chunk_count": run.text_chunk_count,
        "text_coverage": run.text_coverage,
        "warnings": run.warnings_json or [],
        "errors": [],
        "pipeline_stage": manifest.get("pipeline_stage", document.status),
        "timings_ms": manifest.get("timings_ms", {}),
        "counts": manifest.get("counts", {}),
    }


@router.post("/{document_id}/analyze", response_model=PaperAnalysisRead)
def analyze(document_id: int, db: Session = Depends(get_db)) -> PaperAnalysis:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if not _has_parsed_text(db, document.id):
        parse_and_store_document(db, document)
        db.refresh(document)
    if not _has_parsed_text(db, document.id):
        raise HTTPException(
            status_code=400,
            detail=(
                "No parsed text was extracted from this document. "
                "Please re-upload a text-based PDF."
            ),
        )
    return analyze_document(db, document)


@router.get("/{document_id}/analysis", response_model=PaperAnalysisRead | None)
def get_analysis(document_id: int, db: Session = Depends(get_db)) -> PaperAnalysis | None:
    if db.get(Document, document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return db.scalar(select(PaperAnalysis).where(PaperAnalysis.document_id == document_id))


def _has_parsed_text(db: Session, document_id: int) -> bool:
    page_texts = db.scalars(
        select(DocumentPage.text).where(DocumentPage.document_id == document_id)
    ).all()
    return any(text and text.strip() for text in page_texts)
