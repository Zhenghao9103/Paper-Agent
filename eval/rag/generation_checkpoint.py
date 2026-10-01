"""Atomic checkpoints for per-case answer generation."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def run_component_generation_cases(
    *,
    cases: Sequence[Any],
    contexts_by_case: Mapping[str, Sequence[Mapping[str, Any]]],
    output: Path,
    signature: str,
    answer_fn: Callable[[Any, Sequence[Mapping[str, Any]]], Any],
    resume: bool,
) -> dict[str, Any]:
    """Generate answers from frozen contexts with per-case atomic checkpoints."""

    output = Path(output)
    rows_by_id: dict[str, dict[str, Any]] = {}
    if resume and output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("signature") != signature:
            raise ValueError("generation checkpoint signature does not match")
        rows_by_id = {
            str(row["case_id"]): dict(row)
            for row in payload.get("cases", [])
            if isinstance(row, Mapping) and row.get("case_id")
        }

    target_ids = [str(case.case_id) for case in cases]
    for case in cases:
        case_id = str(case.case_id)
        if case_id in rows_by_id and not rows_by_id[case_id].get("error"):
            continue
        contexts = [dict(item) for item in contexts_by_case.get(case_id, [])]
        started = time.perf_counter()
        try:
            result = answer_fn(case, contexts)
            citations = [
                item.model_dump(mode="json")
                if hasattr(item, "model_dump")
                else dict(item)
                for item in getattr(result, "citations", [])
            ]
            answer = str(getattr(result, "answer", ""))
            error = None
        except Exception as exc:  # noqa: BLE001 - preserve per-case failures
            answer = ""
            citations = []
            error = f"{type(exc).__name__}: {exc}"
        rows_by_id[case_id] = {
            "case_id": case_id,
            "question": str(case.question),
            "reference_answer": str(case.reference_answer),
            "answer_points": list(case.answer_points),
            "expect_insufficient_evidence": bool(case.expect_insufficient_evidence),
            "unanswerable_reason": case.unanswerable_reason,
            "contexts": contexts,
            "answer": answer,
            "citations": citations,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "error": error,
        }
        ordered = [rows_by_id[item] for item in target_ids if item in rows_by_id]
        _write_json_atomic(
            output,
            {
                "signature": signature,
                "target_case_count": len(target_ids),
                "completed_case_count": len(ordered),
                "complete": len(ordered) == len(target_ids),
                "cases": ordered,
            },
        )

    ordered = [rows_by_id[item] for item in target_ids if item in rows_by_id]
    report = {
        "signature": signature,
        "target_case_count": len(target_ids),
        "completed_case_count": len(ordered),
        "complete": len(ordered) == len(target_ids),
        "cases": ordered,
    }
    _write_json_atomic(output, report)
    return report
