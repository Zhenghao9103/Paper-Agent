from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from scripts.bootstrap import verify_resources


@dataclass(frozen=True)
class CheckResults:
    ok: bool
    failed: tuple[str, ...]
    details: dict[str, str]


def run_checks(*, checks: Mapping[str, Callable[[], bool]]) -> CheckResults:
    failed: list[str] = []
    details: dict[str, str] = {}
    for name, check in checks.items():
        try:
            if check():
                details[name] = "ok"
            else:
                failed.append(name)
                details[name] = "failed"
        except Exception as exc:  # Each probe must report without hiding later probes.
            failed.append(name)
            details[name] = str(exc)
    return CheckResults(not failed, tuple(failed), details)


def _imports_ok(router_provider: str = "local") -> bool:
    names = ["chromadb", "fastapi", "mineru", "sentence_transformers", "FlagEmbedding"]
    if router_provider == "jev":
        names.append("typesafe_sdk")
    for name in names:
        importlib.import_module(name)
    return True


def _configuration_ok(root: Path) -> bool:
    return (root / ".env").is_file() and (root / ".mineru" / "mineru.json").is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-Agent post-install checks")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--router-provider", choices=("local", "jev"), default="local")
    args = parser.parse_args()
    root = args.root.resolve()
    results = run_checks(
        checks={
            "imports": lambda: _imports_ok(args.router_provider),
            "resources": lambda: bool(
                verify_resources(root, router_provider=args.router_provider)
            ),
            "configuration": lambda: _configuration_ok(root),
        }
    )
    print(json.dumps({"ok": results.ok, "failed": results.failed, "details": results.details}, indent=2))
    return 0 if results.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
