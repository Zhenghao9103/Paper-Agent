from sqlalchemy import delete
from sqlalchemy.orm import Session

from ..ingestion.parsers import parse_document
from ..models.chunk import DocumentChunk
from ..models.document import Document
from ..models.page import DocumentPage
from ..rag.chunking import chunk_text
from ..rag.vector_store import upsert_chunk


def parse_and_store_document(db: Session, document: Document) -> bool:
    try:
        parsed_pages = parse_document(document.file_path, document.file_type, document.id)
    except Exception:
        document.status = "failed"
        db.add(document)
        db.commit()
        return False

    db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
    db.execute(delete(DocumentPage).where(DocumentPage.document_id == document.id))
    chunks_to_index: list[tuple[DocumentChunk, str, str | None]] = []
    for page in parsed_pages:
        page_record = DocumentPage(
            document_id=document.id,
            page_number=page.page_number,
            text=page.text,
            image_path=page.image_path,
        )
        db.add(page_record)
        db.flush()
        for chunk in chunk_text(page.text):
            chunk_record = DocumentChunk(
                document_id=document.id,
                page_id=page_record.id,
                page_number=page.page_number,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
            )
            db.add(chunk_record)
            db.flush()
            chunks_to_index.append((chunk_record, chunk.content, page.image_path))
    document.status = "indexed"
    db.add(document)
    db.commit()
    db.refresh(document)

    for chunk_record, content, image_path in chunks_to_index:
        try:
            upsert_chunk(
                chunk_id=str(chunk_record.id),
                content=content,
                metadata={
                    "chunk_id": chunk_record.id,
                    "document_id": document.id,
                    "title": document.title,
                    "page_number": chunk_record.page_number,
                    "chunk_index": chunk_record.chunk_index,
                    "file_type": document.file_type,
                    "image_path": image_path,
                },
            )
        except Exception:
            continue
    return True
