from pathlib import Path

from .config import get_settings


BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (BACKEND_ROOT / path).resolve()


def storage_root() -> Path:
    return resolve_project_path(get_settings().storage_dir)


def uploads_dir() -> Path:
    return storage_root() / "uploads"


def pages_dir() -> Path:
    return storage_root() / "pages"


def exports_dir() -> Path:
    return storage_root() / "exports"


def chroma_dir() -> Path:
    return resolve_project_path(get_settings().chroma_dir)


def ensure_runtime_dirs() -> None:
    for path in [uploads_dir(), pages_dir(), exports_dir(), chroma_dir(), BACKEND_ROOT / "data"]:
        path.mkdir(parents=True, exist_ok=True)
