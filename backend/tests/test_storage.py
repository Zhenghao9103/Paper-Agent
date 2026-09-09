import asyncio
from pathlib import Path

import fitz
import pytest
from backend.app.core import paths
from backend.app.services.storage import (
    build_safe_filename,
    detect_file_type,
    save_upload_file,
    validate_pdf_bytes,
)


def make_pdf_bytes() -> bytes:
    pdf = fitz.open()
    pdf.new_page()
    data = pdf.tobytes()
    pdf.close()
    return data


def test_detect_file_type_accepts_pdf_case_insensitively() -> None:
    assert detect_file_type("paper.pdf") == "pdf"
    assert detect_file_type("paper.PDF") == "pdf"


@pytest.mark.parametrize("filename", ["slides.pptx", "notes.txt"])
def test_detect_file_type_rejects_non_pdf_extensions(filename: str) -> None:
    with pytest.raises(ValueError, match="^Only PDF files are supported$"):
        detect_file_type(filename)


def test_validate_pdf_bytes_rejects_spoofed_pdf() -> None:
    with pytest.raises(ValueError, match="^Uploaded file is not a valid PDF$"):
        validate_pdf_bytes(b"not really a PDF")


def test_validate_pdf_bytes_rejects_corrupt_signed_pdf() -> None:
    with pytest.raises(ValueError, match="^Uploaded file is not a valid PDF$"):
        validate_pdf_bytes(b"%PDF-not really a PDF")


def test_validate_pdf_bytes_accepts_generated_pdf() -> None:
    validate_pdf_bytes(make_pdf_bytes())


def test_validate_pdf_bytes_does_not_translate_unrelated_errors(monkeypatch) -> None:
    def fail_open(*args, **kwargs):
        raise KeyError("unrelated fault")

    monkeypatch.setattr("backend.app.services.storage.fitz.open", fail_open)

    with pytest.raises(KeyError, match="unrelated fault"):
        validate_pdf_bytes(b"%PDF-1.7")


def test_validate_pdf_bytes_rejects_password_protected_pdf() -> None:
    pdf = fitz.open()
    pdf.new_page()
    content = pdf.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="secret",
    )
    pdf.close()

    with pytest.raises(ValueError, match="^Uploaded file is not a readable PDF$"):
        validate_pdf_bytes(content)


def test_validate_pdf_bytes_checks_password_before_page_count(monkeypatch) -> None:
    class PasswordProtectedPdf:
        needs_pass = True

        @property
        def page_count(self) -> int:
            raise AssertionError("page count must not be read before password validation")

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "backend.app.services.storage.fitz.open",
        lambda *args, **kwargs: PasswordProtectedPdf(),
    )

    with pytest.raises(ValueError, match="^Uploaded file is not a readable PDF$"):
        validate_pdf_bytes(b"%PDF-1.7")


def test_validate_pdf_bytes_rejects_zero_page_pdf() -> None:
    content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [] /Count 0 >>
endobj
trailer
<< /Root 1 0 R >>
%%EOF
"""

    with pytest.raises(ValueError, match="^Uploaded file is not a readable PDF$"):
        validate_pdf_bytes(content)


def test_save_upload_file_reads_once_and_saves_valid_pdf(tmp_path, monkeypatch) -> None:
    content = make_pdf_bytes()

    class Upload:
        filename = "My Paper.pdf"
        read_count = 0

        async def read(self) -> bytes:
            self.read_count += 1
            return content

    upload = Upload()
    monkeypatch.setattr("backend.app.services.storage.uploads_dir", lambda: tmp_path)

    saved_path = asyncio.run(save_upload_file(upload))

    assert upload.read_count == 1
    assert saved_path.parent == tmp_path
    assert saved_path.read_bytes() == content


def test_save_upload_file_cleans_partial_file_when_write_fails(tmp_path, monkeypatch) -> None:
    content = make_pdf_bytes()

    class Upload:
        filename = "write-failure.pdf"
        read_count = 0

        async def read(self) -> bytes:
            self.read_count += 1
            return content

    attempted_paths: list[Path] = []

    def fail_after_prefix(path: Path, data: bytes) -> int:
        attempted_paths.append(path)
        with path.open("wb") as output:
            output.write(data[:16])
        raise OSError("simulated write failure")

    upload = Upload()
    monkeypatch.setattr("backend.app.services.storage.uploads_dir", lambda: tmp_path)
    monkeypatch.setattr(Path, "write_bytes", fail_after_prefix)

    with pytest.raises(OSError, match="simulated write failure"):
        asyncio.run(save_upload_file(upload))

    assert upload.read_count == 1
    assert attempted_paths[0].parent == tmp_path
    assert attempted_paths[0].suffix == ".part"
    assert list(tmp_path.iterdir()) == []


def test_save_upload_file_rejects_spoofed_pdf_before_writing(tmp_path, monkeypatch) -> None:
    class Upload:
        filename = "spoofed.pdf"

        async def read(self) -> bytes:
            return b"plain text"

    monkeypatch.setattr("backend.app.services.storage.uploads_dir", lambda: tmp_path)

    with pytest.raises(ValueError, match="^Uploaded file is not a valid PDF$"):
        asyncio.run(save_upload_file(Upload()))

    assert list(tmp_path.iterdir()) == []


def test_build_safe_filename_keeps_extension() -> None:
    filename = build_safe_filename("My Paper 2026.pdf")

    assert filename.endswith(".pdf")
    assert " " not in filename
    assert Path(filename).name == filename


def test_document_and_staging_paths_are_derived_from_storage_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths, "storage_root", lambda: tmp_path)

    assert paths.documents_dir() == tmp_path / "documents"
    assert paths.document_dir(17) == tmp_path / "documents" / "17"
    assert paths.staging_dir() == tmp_path / ".staging"
    assert paths.pages_dir() == tmp_path / "pages"


@pytest.mark.parametrize("document_id", [0, -1])
def test_document_dir_rejects_non_positive_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, document_id: int
) -> None:
    monkeypatch.setattr(paths, "storage_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="document_id must be positive"):
        paths.document_dir(document_id)


def test_ensure_runtime_dirs_creates_document_and_staging_roots(
    test_storage_root: Path,
) -> None:
    paths.ensure_runtime_dirs()

    assert paths.documents_dir().is_dir()
    assert paths.staging_dir().is_dir()
    assert paths.pages_dir().is_dir()
    assert paths.uploads_dir().is_dir()
    assert paths.exports_dir().is_dir()


def test_isolated_storage_fixture_preserves_document_id_validation(
    test_storage_root: Path,
) -> None:
    with pytest.raises(ValueError, match="document_id must be positive"):
        paths.document_dir(0)
