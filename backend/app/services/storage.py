from pathlib import Path
from uuid import uuid4

import fitz
from fastapi import UploadFile

from ..core.paths import uploads_dir

SUPPORTED_EXTENSIONS = {".pdf": "pdf"}


def _pymupdf_parse_errors() -> tuple[type[Exception], ...]:
    errors = [
        error
        for error in (
            getattr(fitz, "FileDataError", None),
            getattr(getattr(fitz, "mupdf", None), "FzErrorBase", None),
        )
        if isinstance(error, type) and issubclass(error, Exception)
    ]
    return tuple(errors) or (RuntimeError,)


PYMUPDF_PARSE_ERRORS = _pymupdf_parse_errors()


def detect_file_type(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only PDF files are supported")
    return SUPPORTED_EXTENSIONS[extension]


def validate_pdf_bytes(content: bytes) -> None:
    if not content.startswith(b"%PDF-"):
        raise ValueError("Uploaded file is not a valid PDF")

    try:
        pdf = fitz.open(stream=content, filetype="pdf")
        try:
            needs_password = bool(pdf.needs_pass)
            page_count = None if needs_password else pdf.page_count
        finally:
            pdf.close()
    except PYMUPDF_PARSE_ERRORS as exc:
        raise ValueError("Uploaded file is not a valid PDF") from exc

    if needs_password or page_count == 0:
        raise ValueError("Uploaded file is not a readable PDF")


def build_safe_filename(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    stem = Path(filename).stem.strip().replace(" ", "-")
    safe_stem = "".join(char for char in stem if char.isalnum() or char in {"-", "_"})
    if not safe_stem:
        safe_stem = "document"
    return f"{safe_stem}-{uuid4().hex}{extension}"


def infer_title(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip() or "Untitled Document"


async def save_upload_file(upload: UploadFile) -> Path:
    detect_file_type(upload.filename or "")
    content = await upload.read()
    validate_pdf_bytes(content)
    target_dir = uploads_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / build_safe_filename(upload.filename or "document")
    part_path = target_path.with_name(f".{target_path.name}.{uuid4().hex}.part")
    try:
        part_path.write_bytes(content)
        part_path.replace(target_path)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise
    return target_path
