from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from ..core.paths import uploads_dir

SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".pptx": "pptx"}


def detect_file_type(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only PDF and PPTX files are supported")
    return SUPPORTED_EXTENSIONS[extension]


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
    target_dir = uploads_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / build_safe_filename(upload.filename or "document")
    target_path.write_bytes(await upload.read())
    return target_path
