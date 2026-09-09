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


def documents_dir() -> Path:
    return storage_root() / "documents"


def document_dir(document_id: int) -> Path:
    if document_id <= 0:
        raise ValueError("document_id must be positive")
    return documents_dir() / str(document_id)


def staging_dir() -> Path:
    return storage_root() / ".staging"


def exports_dir() -> Path:
    return storage_root() / "exports"


def chroma_dir() -> Path:
    return resolve_project_path(get_settings().chroma_dir)


def hf_model_snapshot(model_name: str) -> str:
    """Resolve a model name to its local .hf-cache snapshot when available.

    Loading by absolute path bypasses HF_* cache environment variables and
    hub connectivity checks entirely, which is what a local-first setup with
    pre-downloaded models needs. Falls back to the model name (normal hub
    resolution) when no complete snapshot exists.
    """

    repo_dir = "models--" + model_name.replace("/", "--")
    snapshots = PROJECT_ROOT / ".hf-cache" / "hub" / repo_dir / "snapshots"
    if not snapshots.is_dir():
        return model_name
    candidates = [
        path
        for path in snapshots.iterdir()
        if path.is_dir() and (path / "config.json").is_file()
    ]
    if not candidates:
        return model_name
    return str(max(candidates, key=lambda path: len(list(path.iterdir()))))


def ensure_runtime_dirs() -> None:
    runtime_dirs = [
        uploads_dir(),
        pages_dir(),
        documents_dir(),
        staging_dir(),
        exports_dir(),
        chroma_dir(),
        BACKEND_ROOT / "data",
    ]
    for path in runtime_dirs:
        path.mkdir(parents=True, exist_ok=True)
