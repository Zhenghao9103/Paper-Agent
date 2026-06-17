from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...db.session import get_db
from ...models.document import Document
from ...models.chunk import DocumentChunk
from ...models.analysis import PaperAnalysis
from ...models.page import DocumentPage
from ...schemas.analysis import PaperAnalysisRead
from ...schemas.chunk import DocumentChunkRead
from ...schemas.document import DocumentRead
from ...schemas.page import DocumentPageRead
from ...services.analysis import analyze_document
from ...services.ingestion import parse_and_store_document
from ...services.reset import clear_all_runtime_data
from ...services.storage import detect_file_type, infer_title, save_upload_file

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentRead])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:
    statement = select(Document).order_by(Document.created_at.desc())
    return list(db.scalars(statement).all())


@router.post("/upload", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> Document:
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
    parse_and_store_document(db, document)
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
def list_document_chunks(document_id: int, db: Session = Depends(get_db)) -> list[DocumentChunk]:
    if db.get(Document, document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")

    statement = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.page_number, DocumentChunk.chunk_index)
    )
    return list(db.scalars(statement).all())


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
            detail="No parsed text was extracted from this document. Please re-upload a text-based PDF/PPTX or use OCR first.",
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
