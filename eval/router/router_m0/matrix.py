from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .backends import LlamaRouterBackend, RouterBackend
from .contract import RouterCase, load_router_cases
from .dataset_validation import sha256_file as dataset_sha256_file
from .dataset_validation import verify_sha256
from .prompt import ROUTER_M0_SYSTEM_PROMPT, build_router_messages
from .reporting import write_scorecard
from .runner import run_router_dataset
from .server import LlamaServerManager, ModelSpec, load_model_registry, sha256_file

INFERENCE = {
    "temperature": 0,
    "max_tokens": 64,
    "context_size": 4096,
    "threads": 8,
    "parallel": 1,
    "thinking": False,
}
REQUIRED_LINEAGE_STRINGS = (
    "model_id",
    "display_name",
    "quantization",
    "model_path",
    "model_sha256",
    "dataset_path",
    "dataset_sha256",
    "prompt_sha256",
    "llama_server_path",
    "llama_server_sha256",
    "code_revision",
    "started_at",
    "completed_at",
)


class _PredictionSetError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _find_git() -> str | None:
    found = shutil.which("git")
    if found is not None or os.name != "nt":
        return found
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\GitForWindows"
        ) as key:
            install_path, _ = winreg.QueryValueEx(key, "InstallPath")
    except OSError:
        return None
    candidate = Path(install_path) / "cmd" / "git.exe"
    return str(candidate) if candidate.is_file() else None


def _git_revision(repo_root: Path) -> tuple[str, bool]:
    git = _find_git()
    if git is None:
        return "unknown", True
    try:
        revision = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [git, "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown", True
    return revision or "unknown", bool(status.strip())


def _machine() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def _prediction_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.append(str(json.loads(line)["case_id"]))
    return ids


def valid_complete_lineage(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("status") == "complete"
        and all(
            isinstance(payload.get(key), str) and payload[key]
            for key in REQUIRED_LINEAGE_STRINGS
        )
        and isinstance(payload.get("dirty"), bool)
        and isinstance(payload.get("machine"), dict)
        and isinstance(payload.get("launch_args"), list)
        and payload.get("inference") == INFERENCE
    )


def _can_skip(
    output_dir: Path,
    expected_ids: Sequence[str],
    *,
    model_sha256: str,
    dataset_sha256: str,
    prompt_sha256: str,
    llama_server_sha256: str,
) -> bool:
    lineage_path = output_dir / "lineage.json"
    predictions_path = output_dir / "predictions.jsonl"
    scorecard_path = output_dir / "scorecard.json"
    if not (
        lineage_path.is_file()
        and predictions_path.is_file()
        and scorecard_path.is_file()
    ):
        return False
    try:
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        prediction_ids = _prediction_ids(predictions_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return (
        valid_complete_lineage(lineage)
        and lineage["model_sha256"] == model_sha256
        and lineage["dataset_sha256"] == dataset_sha256
        and lineage["prompt_sha256"] == prompt_sha256
        and lineage["llama_server_sha256"] == llama_server_sha256
        and len(prediction_ids) == len(expected_ids)
        and set(prediction_ids) == set(expected_ids)
    )


def _failure_category(stage: str, exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "artifact_missing"
    if isinstance(exc, _PredictionSetError):
        return "prediction_set_mismatch"
    if stage == "startup":
        return "startup_failed"
    if stage == "readiness" and isinstance(exc, TimeoutError):
        return "startup_timeout"
    if stage == "readiness":
        return "readiness_failed"
    if stage == "warmup":
        return "warmup_failed"
    return "evaluation_failed"


def run_model_matrix(
    *,
    cases: Sequence[RouterCase],
    dataset_path: Path,
    dataset_sha256: str,
    specs: Sequence[ModelSpec],
    llama_server_path: Path,
    output_root: Path,
    port: int = 8089,
    startup_timeout: float = 180.0,
    skip_complete: bool = False,
    server_extra_args: tuple[str, ...] = (),
    manager_factory: Callable[..., Any] = LlamaServerManager,
    backend_factory: Callable[[str, str], RouterBackend] = LlamaRouterBackend,
    dataset_runner: Callable[..., Mapping[str, Any]] = run_router_dataset,
    revision_provider: Callable[[], tuple[str, bool]] | None = None,
    machine_provider: Callable[[], dict[str, str]] = _machine,
) -> list[dict[str, str]]:
    sorted_cases = sorted(cases, key=lambda case: case.id)
    expected_ids = [case.id for case in sorted_cases]
    prompt_sha256 = hashlib.sha256(ROUTER_M0_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    llama_server_path = llama_server_path.resolve()
    llama_sha256 = sha256_file(llama_server_path)
    revision, dirty = (
        revision_provider()
        if revision_provider is not None
        else _git_revision(Path.cwd())
    )
    machine = machine_provider()
    results: list[dict[str, str]] = []

    for spec in specs:
        output_dir = output_root / spec.id
        model_sha256 = sha256_file(spec.path)
        if skip_complete and _can_skip(
            output_dir,
            expected_ids,
            model_sha256=model_sha256,
            dataset_sha256=dataset_sha256,
            prompt_sha256=prompt_sha256,
            llama_server_sha256=llama_sha256,
        ):
            results.append({"model_id": spec.id, "status": "skipped"})
            continue

        lineage: dict[str, Any] = {
            "model_id": spec.id,
            "display_name": spec.display_name,
            "quantization": spec.quantization,
            "model_path": str(spec.path.resolve()),
            "model_sha256": model_sha256,
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": dataset_sha256,
            "prompt_sha256": prompt_sha256,
            "llama_server_path": str(llama_server_path),
            "llama_server_sha256": llama_sha256,
            "code_revision": revision,
            "started_at": _utc_now(),
            "completed_at": "",
            "inference": dict(INFERENCE),
            "server_extra_args": list(server_extra_args),
            "machine": dict(machine),
            "launch_args": [],
            "dirty": dirty,
            "status": "failed",
        }
        if server_extra_args:
            manager = manager_factory(
                llama_server_path,
                spec,
                output_dir,
                port,
                extra_args=server_extra_args,
            )
        else:
            manager = manager_factory(llama_server_path, spec, output_dir, port)
        stage = "startup"
        try:
            lineage["launch_args"] = manager.start()
            stage = "readiness"
            manager.wait_ready(startup_timeout)
            backend = backend_factory(spec.served_model, f"http://127.0.0.1:{port}/v1")
            stage = "warmup"
            warmup = backend.generate(build_router_messages(sorted_cases[0]))
            if warmup.error_code is not None:
                raise RuntimeError("warmup backend failure")
            stage = "evaluation"
            dataset_runner(sorted_cases, backend, output_dir, lineage)
            prediction_ids = _prediction_ids(output_dir / "predictions.jsonl")
            if len(prediction_ids) != len(expected_ids) or set(prediction_ids) != set(
                expected_ids
            ):
                raise _PredictionSetError("prediction IDs do not match dataset")
            lineage["status"] = "complete"
            lineage["completed_at"] = _utc_now()
            write_scorecard(output_dir / "lineage.json", lineage)
            results.append({"model_id": spec.id, "status": "complete"})
        except Exception as exc:
            lineage["status"] = "failed"
            lineage["failure_category"] = _failure_category(stage, exc)
            lineage["completed_at"] = _utc_now()
            write_scorecard(output_dir / "lineage.json", lineage)
            results.append({"model_id": spec.id, "status": "failed"})
        finally:
            manager.stop()
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the sequential Router M0 matrix")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sha", type=Path)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--model-id")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument(
        "--server-extra-args-json",
        default="[]",
        help="JSON string array appended to llama-server arguments",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        parsed_extra_args = json.loads(args.server_extra_args_json)
    except json.JSONDecodeError as exc:
        raise SystemExit("--server-extra-args-json must be valid JSON") from exc
    if not isinstance(parsed_extra_args, list) or not all(
        isinstance(item, str) for item in parsed_extra_args
    ):
        raise SystemExit("--server-extra-args-json must be a JSON string array")
    if args.sha is not None:
        verify_sha256(args.dataset, args.sha)
    dataset_sha256 = dataset_sha256_file(args.dataset)
    cases = load_router_cases(args.dataset)
    specs = load_model_registry(args.registry, Path.cwd(), args.model_id)
    results = run_model_matrix(
        cases=cases,
        dataset_path=args.dataset,
        dataset_sha256=dataset_sha256,
        specs=specs,
        llama_server_path=args.llama_server,
        output_root=args.output_root,
        port=args.port,
        startup_timeout=args.startup_timeout,
        skip_complete=args.skip_complete,
        server_extra_args=tuple(parsed_extra_args),
    )
    for result in results:
        print(f"{result['model_id']}: {result['status']}")
    return 1 if any(result["status"] == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
