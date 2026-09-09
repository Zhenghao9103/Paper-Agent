import os
import re
from pathlib import Path

from .config import get_settings

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
HF_MODEL_SEGMENT_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")


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
    """Return the installed local snapshot or fail without Hub fallback."""
    _validate_hf_model_name(model_name)
    hub_root = Path(os.environ["HUGGINGFACE_HUB_CACHE"]).expanduser().resolve()
    repo_dir = "models--" + model_name.replace("/", "--")
    repository = _resolve_within(hub_root, hub_root / repo_dir, "model repository")
    snapshots = _resolve_within(hub_root, repository / "snapshots", "snapshot directory")
    main_ref = _resolve_within(hub_root, repository / "refs" / "main", "refs/main")

    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        _validate_hf_revision(revision)
        selected = _resolve_within(snapshots, snapshots / revision, "refs/main revision")
        if _is_complete_hf_snapshot(selected):
            return str(selected)
        raise RuntimeError(
            f"Required local model snapshot is missing or incomplete: {model_name} "
            f"refs/main points to {revision!r}. Run setup.ps1 to install retrieval models."
        )

    if snapshots.is_dir():
        candidates: list[Path] = []
        for path in snapshots.iterdir():
            candidate = _resolve_within(snapshots, path, "snapshot")
            if _is_complete_hf_snapshot(candidate):
                candidates.append(candidate)
        if len(candidates) == 1:
            return str(candidates[0])
        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple complete local snapshots found for {model_name} without refs/main. "
                "Run setup.ps1 to lock the installed revision."
            )

    raise RuntimeError(
        f"Required local model snapshot is missing or incomplete: {model_name}. "
        "Run setup.ps1 to install retrieval models before starting PaperMind."
    )


def _is_complete_hf_snapshot(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (
            (path / "model.safetensors").is_file()
            or (path / "pytorch_model.bin").is_file()
        )
    )


def _validate_hf_model_name(model_name: str) -> None:
    segments = model_name.split("/")
    if "\\" in model_name or len(segments) != 2 or not all(
        _is_valid_hf_segment(segment) for segment in segments
    ):
        raise ValueError(f"Invalid HuggingFace model name {model_name!r}; expected owner/repo")


def _validate_hf_revision(revision: str) -> None:
    if not _is_valid_hf_segment(revision):
        raise ValueError(f"Invalid refs/main revision {revision!r}")


def _is_valid_hf_segment(value: str) -> bool:
    return (
        bool(HF_MODEL_SEGMENT_PATTERN.fullmatch(value))
        and value not in {".", ".."}
        and not value.endswith((".", "-"))
        and ".." not in value
        and "--" not in value
        and len(value) <= 96
    )


def _resolve_within(root: Path, path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Resolved {label} escapes configured HuggingFace hub cache")
    return resolved


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
