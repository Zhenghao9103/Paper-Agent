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


def require_hf_model_snapshot(model_name: str) -> str:
    """Return the newest complete local snapshot or fail without Hub fallback."""
    repo_dir = "models--" + model_name.replace("/", "--")
    snapshots = PROJECT_ROOT / ".hf-cache" / "hub" / repo_dir / "snapshots"
    if snapshots.is_dir():
        candidates = [
            path
            for path in snapshots.iterdir()
            if path.is_dir()
            and (path / "config.json").is_file()
            and (
                (path / "model.safetensors").is_file()
                or (path / "pytorch_model.bin").is_file()
            )
        ]
        if candidates:
            selected = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
            return str(selected.resolve())

    raise RuntimeError(
        f"Required local model snapshot is missing or incomplete: {model_name}. "
        "Run setup.ps1 to install retrieval models before starting PaperMind."
    )


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
