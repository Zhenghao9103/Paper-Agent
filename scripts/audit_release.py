from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


SELECTED_MODEL = (
    "models/router/qwen3-1.7b-router-sft-v3/"
    "Qwen3-1.7B-Router-SFT-V3-Q4_K_M.gguf"
)
ALLOWED_ROOTS = {
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "README.md",
    "backend",
    "frontend",
    "main.py",
    "models",
    "requirements.txt",
    "resources.lock.json",
    "scripts",
    "setup.ps1",
}
FORBIDDEN_COMPONENTS = {
    ".cache",
    ".hf-cache",
    ".mineru",
    ".tools",
    "chroma",
    "data",
    "docs",
    "reports",
    "storage",
}
SECRET_RE = re.compile(
    r"(?im)^(?:AGENT|OPENAI|ROUTER|JUDGE)_API_KEY\s*=\s*(?!$|your_|example|<)[^\s#]{8,}"
)


@dataclass(frozen=True)
class AuditResult:
    ok: bool
    forbidden_paths: tuple[str, ...]
    unexpected_roots: tuple[str, ...]
    unexpected_models: tuple[str, ...]
    missing_lfs: tuple[str, ...]
    large_blobs: tuple[str, ...]
    secret_findings: tuple[str, ...]


def audit_release_tree(
    *,
    root: Path,
    tracked_files: Sequence[str],
    lfs_files: Sequence[str],
    blob_sizes: Mapping[str, int],
    staged_texts: Mapping[str, str],
) -> AuditResult:
    normalized = tuple(path.replace("\\", "/") for path in tracked_files)
    forbidden = tuple(
        sorted(
            path
            for path in normalized
            if any(part in FORBIDDEN_COMPONENTS for part in PurePosixPath(path).parts)
        )
    )
    unexpected_roots = tuple(
        sorted(path for path in normalized if PurePosixPath(path).parts[0] not in ALLOWED_ROOTS)
    )
    models = tuple(sorted(path for path in normalized if path.lower().endswith(".gguf")))
    unexpected_models = tuple(path for path in models if path != SELECTED_MODEL)
    lfs = {path.replace("\\", "/") for path in lfs_files}
    missing_lfs = (SELECTED_MODEL,) if SELECTED_MODEL in normalized and SELECTED_MODEL not in lfs else ()
    if SELECTED_MODEL not in normalized:
        missing_lfs = (SELECTED_MODEL,)
    large_blobs = tuple(
        sorted(path for path, size in blob_sizes.items() if size > 100 * 1024 * 1024)
    )
    secrets = tuple(sorted(path for path, text in staged_texts.items() if SECRET_RE.search(text)))
    fields = (forbidden, unexpected_roots, unexpected_models, missing_lfs, large_blobs, secrets)
    return AuditResult(not any(fields), forbidden, unexpected_roots, unexpected_models, missing_lfs, large_blobs, secrets)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout


def audit_repository(root: Path) -> AuditResult:
    tracked = tuple(line for line in _git(root, "ls-files").splitlines() if line)
    lfs_output = _git(root, "lfs", "ls-files", "--name-only")
    lfs = tuple(line for line in lfs_output.splitlines() if line)
    sizes: dict[str, int] = {}
    texts: dict[str, str] = {}
    for relative in tracked:
        path = root / relative
        if not path.is_file() or relative in lfs:
            continue
        sizes[relative] = path.stat().st_size
        if path.stat().st_size <= 2 * 1024 * 1024:
            try:
                texts[relative] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                pass
    return audit_release_tree(
        root=root,
        tracked_files=tracked,
        lfs_files=lfs,
        blob_sizes=sizes,
        staged_texts=texts,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the public release tree")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = audit_repository(args.root.resolve())
    payload = asdict(result)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
