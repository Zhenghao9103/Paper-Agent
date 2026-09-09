from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable
from zipfile import ZipFile


LOCK_SCHEMA_VERSION = 1
INSTALL_STAGES = (
    "preflight",
    "venv",
    "dependencies",
    "git-lfs",
    "llama-cpp",
    "retrieval-models",
    "mineru-primary",
    "mineru-figure",
    "tiktoken",
    "configuration",
    "post-install-check",
)
ALLOWED_KINDS = {
    "git_lfs_file",
    "http_archive",
    "http_file",
    "huggingface_snapshot",
    "modelscope_snapshot",
}
INSTALLER_ENV = {
    "ROUTER_MANAGED": "true",
    "ROUTER_SERVER_PATH": ".tools/llama.cpp/llama-server.exe",
    "ROUTER_GGUF_PATH": (
        "models/router/qwen3-1.7b-router-sft-v3/"
        "Qwen3-1.7B-Router-SFT-V3-Q4_K_M.gguf"
    ),
    "ROUTER_HOST": "127.0.0.1",
    "ROUTER_PORT": "8089",
    "ROUTER_MODEL": "papermind-router-v3",
    "MINERU_ROOT": ".mineru",
    "MINERU_FORMULA_ENABLED": "true",
    "MINERU_FIGURE_ENABLED": "true",
    "MINERU_FIGURE_PYTHON": ".mineru/figure-env/Scripts/python.exe",
    "MINERU_FIGURE_MANIFEST": ".mineru/figure-model.json",
    "HF_HOME": ".hf-cache",
    "HUGGINGFACE_HUB_CACHE": ".hf-cache/hub",
    "TIKTOKEN_CACHE_DIR": ".cache/tiktoken",
}
MINERU_PIPELINE_REVISION = "05eaf85cc4ddab92c2be61e10abec4586d25c1a6"
MINERU_FIGURE_REVISION = "7a1ddf1dd3baa3c60507e33514c30e2cbfc1c3e2"


class ResourceLockError(ValueError):
    """The resource lock is malformed or unsafe."""


class ResourceInstallError(RuntimeError):
    """A locked resource could not be safely installed."""


@dataclass(frozen=True)
class Resource:
    id: str
    kind: str
    destination: Path
    url: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    sha256: str | None = None
    tree_sha256: str | None = None
    executable_sha256: str | None = None


@dataclass(frozen=True)
class ResourceLock:
    schema_version: int
    resources: tuple[Resource, ...]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_destination(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ResourceLockError("resource destination is required")
    normalized = PurePosixPath(value.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ResourceLockError(f"resource destination must remain inside repository: {value}")
    if normalized.parts and normalized.parts[0].endswith(":"):
        raise ResourceLockError(f"resource destination must be relative: {value}")
    return Path(*normalized.parts)


def load_resource_lock(path: Path) -> ResourceLock:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceLockError(f"cannot read resource lock: {exc}") from exc
    if payload.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise ResourceLockError("unsupported resource lock schema_version")
    raw_resources = payload.get("resources")
    if not isinstance(raw_resources, list):
        raise ResourceLockError("resources must be a list")

    resources: list[Resource] = []
    seen: set[str] = set()
    for raw in raw_resources:
        if not isinstance(raw, dict):
            raise ResourceLockError("each resource must be an object")
        resource_id = raw.get("id")
        kind = raw.get("kind")
        if not isinstance(resource_id, str) or not resource_id or resource_id in seen:
            raise ResourceLockError("resource IDs must be non-empty and unique")
        if kind not in ALLOWED_KINDS:
            raise ResourceLockError(f"unsupported resource kind: {kind}")
        destination = _safe_destination(raw.get("destination"))
        revision = raw.get("revision")
        if kind in {"huggingface_snapshot", "modelscope_snapshot"}:
            if not isinstance(raw.get("repo_id"), str) or not raw["repo_id"]:
                raise ResourceLockError(f"{resource_id} requires repo_id")
            if not isinstance(revision, str) or len(revision) < 40 or not revision.isalnum():
                raise ResourceLockError(f"{resource_id} requires an immutable revision")
        for field in ("sha256", "tree_sha256", "executable_sha256"):
            value = raw.get(field)
            if value is not None and not _is_sha256(value):
                raise ResourceLockError(f"{resource_id} has invalid lowercase {field}")
        if kind in {"http_archive", "http_file", "git_lfs_file"} and not _is_sha256(
            raw.get("sha256")
        ):
            raise ResourceLockError(f"{resource_id} requires sha256")
        if kind in {"http_archive", "http_file"} and not isinstance(raw.get("url"), str):
            raise ResourceLockError(f"{resource_id} requires url")
        resources.append(
            Resource(
                id=resource_id,
                kind=kind,
                destination=destination,
                url=raw.get("url"),
                repo_id=raw.get("repo_id"),
                revision=revision,
                sha256=raw.get("sha256"),
                tree_sha256=raw.get("tree_sha256"),
                executable_sha256=raw.get("executable_sha256"),
            )
        )
        seen.add(resource_id)
    return ResourceLock(schema_version=LOCK_SCHEMA_VERSION, resources=tuple(resources))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _download(url: str, target: Path) -> None:
    urllib.request.urlretrieve(url, target)


def install_file(
    resource: Resource,
    root: Path,
    *,
    download: Callable[[str, Path], None] = _download,
) -> Path:
    if not resource.url or not resource.sha256:
        raise ResourceInstallError(f"{resource.id} is missing URL or SHA256")
    target = root / resource.destination
    if target.is_file() and sha256_file(target) == resource.sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.download")
    try:
        if temporary.exists():
            temporary.unlink()
        download(resource.url, temporary)
        actual = sha256_file(temporary)
        if actual != resource.sha256:
            raise ResourceInstallError(
                f"SHA256 mismatch for {resource.id}: expected {resource.sha256}, got {actual}"
            )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def extract_verified_zip(archive: Path, destination: Path) -> Path:
    destination_parent = destination.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination_parent))
    try:
        with ZipFile(archive) as source:
            for member in source.infolist():
                member_path = PurePosixPath(member.filename.replace("\\", "/"))
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ResourceInstallError(
                        f"archive member is outside destination: {member.filename}"
                    )
            source.extractall(temporary)
        if destination.exists():
            backup = destination.with_name(f".{destination.name}.previous")
            if backup.exists():
                shutil.rmtree(backup)
            destination.replace(backup)
            try:
                temporary.replace(destination)
            except Exception:
                backup.replace(destination)
                raise
            shutil.rmtree(backup)
        else:
            temporary.replace(destination)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def install_snapshot(
    resource: Resource,
    root: Path,
    *,
    snapshot_download: Callable[..., str],
) -> Path:
    if not resource.repo_id or not resource.revision:
        raise ResourceInstallError(f"{resource.id} is missing repo_id or revision")
    cache_dir = (root / ".hf-cache" / "hub").resolve()
    snapshot = Path(
        snapshot_download(
            repo_id=resource.repo_id,
            revision=resource.revision,
            cache_dir=str(cache_dir),
            local_files_only=False,
        )
    )
    if resource.tree_sha256 and sha256_tree(snapshot) != resource.tree_sha256:
        raise ResourceInstallError(f"tree SHA256 mismatch for {resource.id}")
    return snapshot


def _install_archive(resource: Resource, root: Path) -> Path:
    downloads = root / ".cache" / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive_resource = Resource(
        id=resource.id,
        kind="http_file",
        destination=Path(".cache/downloads") / f"{resource.id}.zip",
        url=resource.url,
        sha256=resource.sha256,
    )
    archive = install_file(archive_resource, root)
    destination = root / resource.destination
    executable = destination / "llama-server.exe"
    if (
        executable.is_file()
        and resource.executable_sha256
        and sha256_file(executable) == resource.executable_sha256
    ):
        return destination
    extract_verified_zip(archive, destination)
    if resource.executable_sha256:
        candidates = list(destination.rglob("llama-server.exe"))
        if len(candidates) != 1 or sha256_file(candidates[0]) != resource.executable_sha256:
            raise ResourceInstallError(f"executable SHA256 mismatch for {resource.id}")
        if candidates[0].parent != destination:
            for item in candidates[0].parent.iterdir():
                shutil.move(str(item), destination / item.name)
    return destination


def _install_modelscope_snapshot(resource: Resource, root: Path) -> Path:
    from modelscope.hub.snapshot_download import snapshot_download

    if not resource.repo_id or not resource.revision:
        raise ResourceInstallError(f"{resource.id} is missing repo_id or revision")
    path = Path(
        snapshot_download(
            model_id=resource.repo_id,
            revision=resource.revision,
            cache_dir=str((root / ".mineru" / "modelscope").resolve()),
            local_files_only=False,
        )
    )
    if resource.tree_sha256 and sha256_tree(path) != resource.tree_sha256:
        raise ResourceInstallError(f"tree SHA256 mismatch for {resource.id}")
    return path


def _resource_digest(resource: Resource, root: Path) -> str:
    target = root / resource.destination
    if resource.kind in {"http_file", "git_lfs_file"}:
        if not target.is_file():
            raise ResourceInstallError(f"missing resource: {resource.id}")
        return sha256_file(target)
    if resource.kind == "http_archive":
        executable = target / "llama-server.exe"
        if not executable.is_file():
            raise ResourceInstallError(f"missing resource: {resource.id}")
        return sha256_file(executable)
    if resource.kind in {"huggingface_snapshot", "modelscope_snapshot"}:
        snapshots = target / "snapshots"
        expected = snapshots / str(resource.revision)
        candidate = expected if expected.is_dir() else None
        if candidate is None:
            matches = [item for item in snapshots.glob("*") if item.is_dir()] if snapshots.is_dir() else []
            candidate = matches[0] if len(matches) == 1 else None
        if candidate is None:
            raise ResourceInstallError(f"missing locked snapshot: {resource.id}")
        return sha256_tree(candidate)
    raise ResourceInstallError(f"unsupported resource kind: {resource.kind}")


def install_resources(root: Path) -> dict[str, str]:
    root = root.resolve()
    lock = load_resource_lock(root / "resources.lock.json")
    completed: dict[str, str] = {}
    for resource in lock.resources:
        if resource.kind == "git_lfs_file":
            digest = _resource_digest(resource, root)
            if digest != resource.sha256:
                raise ResourceInstallError(f"Git LFS resource mismatch: {resource.id}")
        elif resource.kind == "http_file":
            install_file(resource, root)
        elif resource.kind == "http_archive":
            _install_archive(resource, root)
        elif resource.kind == "huggingface_snapshot":
            from huggingface_hub import snapshot_download

            install_snapshot(resource, root, snapshot_download=snapshot_download)
        elif resource.kind == "modelscope_snapshot":
            _install_modelscope_snapshot(resource, root)
        completed[resource.id] = _resource_digest(resource, root)
        write_install_state(root, completed)
    return completed


def verify_resources(root: Path) -> dict[str, str]:
    root = root.resolve()
    lock = load_resource_lock(root / "resources.lock.json")
    verified: dict[str, str] = {}
    for resource in lock.resources:
        digest = _resource_digest(resource, root)
        expected = resource.sha256
        if resource.kind == "http_archive":
            expected = resource.executable_sha256
        elif resource.kind in {"huggingface_snapshot", "modelscope_snapshot"}:
            expected = resource.tree_sha256
        if expected and digest != expected:
            raise ResourceInstallError(f"resource digest mismatch: {resource.id}")
        verified[resource.id] = digest
    return verified


def write_install_state(root: Path, completed: dict[str, str]) -> None:
    target = root / ".cache" / "install-state.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"schema_version": 1, "completed": completed},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _atomic_write_text(target: Path, text: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    return target


def generate_env(root: Path) -> Path:
    target = root / ".env"
    existing = target.read_text(encoding="utf-8").splitlines() if target.exists() else []
    output: list[str] = []
    written: set[str] = set()
    for line in existing:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in line:
            key = line.split("=", 1)[0].strip()
            if key in INSTALLER_ENV:
                output.append(f"{key}={INSTALLER_ENV[key]}")
                written.add(key)
                continue
        output.append(line)
    missing = [key for key in INSTALLER_ENV if key not in written]
    if missing:
        if output and output[-1]:
            output.append("")
        output.append("# Managed by setup.ps1; provider credentials above are preserved.")
        output.extend(f"{key}={INSTALLER_ENV[key]}" for key in missing)
    return _atomic_write_text(target, "\n".join(output).rstrip() + "\n")


def generate_mineru_config(root: Path) -> Path:
    resolved_root = root.resolve()
    mineru_root = resolved_root / ".mineru"
    pipeline_snapshot = mineru_root / "modelscope" / "models" / (
        "OpenDataLab--PDF-Extract-Kit-1.0"
    ) / "snapshots" / MINERU_PIPELINE_REVISION
    figure_snapshot = mineru_root / "modelscope" / "models" / (
        "OpenDataLab--MinerU2.5-Pro-2605-1.2B"
    ) / "snapshots" / MINERU_FIGURE_REVISION
    config = {
        "config_version": "1.3.2",
        "model-source": "local",
        "models-dir": {"pipeline": str(pipeline_snapshot), "vlm": ""},
        "latex-delimiter-config": {
            "display": {"left": "$$", "right": "$$"},
            "inline": {"left": "$", "right": "$"},
        },
    }
    manifest = {
        "model_id": "OpenDataLab/MinerU2.5-Pro-2605-1.2B",
        "revision": MINERU_FIGURE_REVISION,
        "local_path": str(figure_snapshot),
    }
    text_options = {"indent": 2, "ensure_ascii": False, "sort_keys": True}
    _atomic_write_text(
        mineru_root / "figure-model.json",
        json.dumps(manifest, **text_options) + "\n",
    )
    return _atomic_write_text(
        mineru_root / "mineru.json",
        json.dumps(config, **text_options) + "\n",
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Paper-Agent resource bootstrap")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-lock", help="validate resources.lock.json")
    verify.add_argument("--root", type=Path, default=Path.cwd())
    install = subparsers.add_parser("install-resources", help="install locked resources")
    install.add_argument("--root", type=Path, default=Path.cwd())
    verify_installed = subparsers.add_parser(
        "verify-resources", help="verify installed resources"
    )
    verify_installed.add_argument("--root", type=Path, default=Path.cwd())
    configure = subparsers.add_parser("configure", help="write portable local config")
    configure.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == "verify-lock":
        lock = load_resource_lock(args.root / "resources.lock.json")
        for resource in lock.resources:
            print(f"{resource.id}: {resource.kind}")
        return 0
    if args.command == "install-resources":
        for resource_id, digest in install_resources(args.root).items():
            print(f"{resource_id}: {digest}")
        return 0
    if args.command == "verify-resources":
        for resource_id, digest in verify_resources(args.root).items():
            print(f"{resource_id}: {digest}")
        return 0
    if args.command == "configure":
        generate_env(args.root)
        generate_mineru_config(args.root)
        print("portable configuration written")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
