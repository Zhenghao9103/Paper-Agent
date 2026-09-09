from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile

_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")


def install_figure_model(
    *,
    mineru_root: Path | str,
    model_id: str,
    requested_revision: str,
    resolve_revision: Callable[[str, str], str],
    snapshot_download: Callable[..., str],
) -> Path:
    root = Path(mineru_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_revision = str(resolve_revision(model_id, requested_revision)).strip()
    if not source_revision:
        raise ValueError("resolved model revision is empty")
    local_path = Path(
        snapshot_download(
            model_id,
            revision=source_revision,
            cache_dir=str(root / "modelscope"),
        )
    ).resolve()
    if not local_path.is_dir() or not local_path.is_relative_to(root):
        raise ValueError("downloaded model path is outside MINERU_ROOT")
    revision = (
        source_revision
        if _COMMIT_RE.fullmatch(source_revision)
        else _snapshot_digest(local_path)
    )
    manifest = root / "figure-model.json"
    payload = {
        "model_id": model_id,
        "revision": revision,
        "source_revision": source_revision,
        "local_path": str(local_path),
    }
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=root,
            prefix=".figure-model-",
            suffix=".json",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, indent=2)
        temporary.replace(manifest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def _snapshot_digest(local_path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in local_path.rglob("*") if path.is_file())
    if not files:
        raise ValueError("downloaded model snapshot is empty")
    for path in files:
        digest.update(path.relative_to(local_path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _default_resolve_revision(model_id: str, revision: str) -> str:
    from huggingface_hub import HfApi

    return str(HfApi().model_info(model_id, revision=revision).sha)


def _default_snapshot_download(model_id: str, **kwargs: object) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_id, local_files_only=False, **kwargs)


def modelscope_download_adapters(*, api=None, download=None):
    """Return ModelScope adapters matching the installer's immutable contract."""
    if api is None:
        from modelscope.hub.api import HubApi

        api = HubApi()
    if download is None:
        from modelscope import snapshot_download

        download = snapshot_download

    def resolve_revision(model_id: str, requested_revision: str) -> str:
        revision = "master" if requested_revision in {"main", "master"} else requested_revision
        api.get_model_files(model_id, revision=revision)
        return revision

    def snapshot_download(model_id: str, **kwargs: object) -> str:
        return str(
            download(
                model_id,
                revision=kwargs["revision"],
                cache_dir=kwargs["cache_dir"],
            )
        )

    return resolve_revision, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicitly install the MinerU figure VLM")
    parser.add_argument("--mineru-root", required=True)
    parser.add_argument(
        "--model-id", default="opendatalab/MinerU2.5-Pro-2605-1.2B"
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--source", choices=("huggingface", "modelscope"), default="huggingface"
    )
    args = parser.parse_args()
    model_id = args.model_id
    resolve_revision = _default_resolve_revision
    snapshot_download = _default_snapshot_download
    if args.source == "modelscope":
        if model_id.casefold() == "opendatalab/mineru2.5-pro-2605-1.2b":
            model_id = "OpenDataLab/MinerU2.5-Pro-2605-1.2B"
        resolve_revision, snapshot_download = modelscope_download_adapters()
    manifest = install_figure_model(
        mineru_root=args.mineru_root,
        model_id=model_id,
        requested_revision=args.revision,
        resolve_revision=resolve_revision,
        snapshot_download=snapshot_download,
    )
    print(manifest)


if __name__ == "__main__":
    main()
