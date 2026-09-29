"""Evaluate TypeSafe Jev as an isolated three-way intent router.

Install ``typesafe-sdk`` in the evaluation environment for live runs. The
``--validate-only`` path needs no SDK or API key. Gold labels and metadata are
never sent to Jev. No production routing files are changed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from backend.app.core.config import BACKEND_ROOT, PROJECT_ROOT, _parse_env_file
from backend.app.services.jev_router import CRITERIA, QUESTION
from backend.app.services.router_contract import VALID_INTENTS, build_router_user_content
from .routing_quality import evaluate_routes
from .jev_metrics import (
    latency_summary,
    per_class_metrics,
    slice_metrics,
    load_holdout_cases,
)

MODEL = "jev-1.13.0"
GATEWAY_MODEL = "typesafe-ai/jev"
GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/typesafe"
OPENCODE_MODEL = "jev-1.13"
OPENCODE_BASE_URL = "https://opencode.ai/zen"
DEFAULT_HOLDOUT = Path(__file__).parent / "datasets" / "router_m0_test_v1.jsonl"


def _local_jev_settings() -> dict[str, str]:
    """Match the project's local .env precedence without exporting credentials."""
    values: dict[str, str] = {}
    for path in (PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"):
        values.update(_parse_env_file(path))
    for name in ("JEV_API_KEY", "JEV_BASE_URL", "JEV_MODEL"):
        if os.environ.get(name):
            values[name] = os.environ[name]
    return values


def jev_connection() -> tuple[str, str | None, str]:
    settings = _local_jev_settings()
    key = settings.get("JEV_API_KEY")
    if not key:
        raise ValueError("set JEV_API_KEY in backend/.env")
    return key, settings.get("JEV_BASE_URL") or None, settings.get("JEV_MODEL") or MODEL


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = load_holdout_cases(path)
    seen: set[str] = set()
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        if case["expected"] not in VALID_INTENTS:
            raise ValueError(f"invalid expected intent for {case_id}: {case['expected']}")
        if not str(case["question"]).strip():
            raise ValueError(f"empty question for {case_id}")
    return cases


def state_for_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Use the production Router's bounded input representation, without gold."""
    return json.loads(
        build_router_user_content(case.get("session_context", []), str(case["question"]))
    )


def _usage_value(usage: Any, name: str) -> int:
    value = usage.get(name, 0) if isinstance(usage, Mapping) else getattr(usage, name, 0)
    return int(value or 0)


def evaluate_cases(
    cases: Sequence[Mapping[str, Any]],
    decide: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    input_tokens = output_tokens = 0
    actual_models: set[str] = set()
    for case in cases:
        started = time.perf_counter()
        predicted: str | None = None
        confidence: float | None = None
        probabilities: dict[str, float] | None = None
        error_type: str | None = None
        try:
            response = decide(state_for_case(case))
            answer = response.choices["intent"]
            predicted = str(answer.choice)
            if predicted not in VALID_INTENTS:
                raise ValueError("Jev returned an invalid route")
            confidence = float(answer.confidence)
            probabilities = {label: float(answer.probabilities[label]) for label in VALID_INTENTS}
            if not 0 <= confidence <= 1 or any(
                not 0 <= value <= 1 for value in probabilities.values()
            ):
                raise ValueError("Jev returned invalid probabilities")
            actual_models.add(str(response.model))
            input_tokens += _usage_value(response.usage, "input_tokens")
            output_tokens += _usage_value(response.usage, "output_tokens")
        except Exception as exc:
            # Do not persist provider exception text: it may include request data.
            predicted = None
            confidence = None
            probabilities = None
            error_type = type(exc).__name__

        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        rows.append(
            {
                "case_id": case["case_id"],
                "question": case["question"],
                "expected": case["expected"],
                "predicted": predicted,
                "correct": predicted == case["expected"],
                "degraded": predicted is None,
                "error_type": error_type,
                "confidence": confidence,
                "probabilities": probabilities,
                "latency_ms": latency_ms,
                "language": case.get("language", "unknown"),
                "difficulty": case.get("difficulty", "unknown"),
                "scenario": case.get("scenario", "unknown"),
                "has_context": bool(case.get("session_context")),
            }
        )

    valid = [row for row in rows if row["predicted"] is not None]
    gold = [str(row["expected"]) for row in valid]
    predicted = [str(row["predicted"]) for row in valid]
    metrics = evaluate_routes(gold, predicted)
    per_class = per_class_metrics(gold, predicted)
    directional = {
        f"{source}->{target}": sum(
            row["expected"] == source and row["predicted"] == target for row in rows
        )
        for source in VALID_INTENTS
        for target in VALID_INTENTS
        if source != target
    }
    agentic = [row for row in rows if row["expected"] == "agentic_rag"]
    return {
        "case_count": len(rows),
        "valid_count": len(valid),
        "error_count": len(rows) - len(valid),
        "accuracy_all": round(sum(row["correct"] for row in rows) / len(rows), 4) if rows else 0,
        "accuracy_valid": metrics.accuracy,
        "macro_f1_valid": round(
            sum(item["f1"] for item in per_class.values()) / len(VALID_INTENTS), 4
        ),
        "per_class_valid": per_class,
        "confusion_valid": metrics.confusion,
        "directional_errors": directional,
        "high_risk_gates": {
            "agentic_to_direct_zero": directional["agentic_rag->direct"] == 0,
            "agentic_to_simple_le_2": directional["agentic_rag->simple_rag"] <= 2,
            "simple_to_direct_le_2": directional["simple_rag->direct"] <= 2,
        },
        "agentic_recall_all": round(sum(row["correct"] for row in agentic) / len(agentic), 4)
        if agentic
        else 0,
        "latency_ms": latency_summary([row["latency_ms"] for row in rows]),
        "latency_ms_valid": latency_summary([row["latency_ms"] for row in valid]),
        "slices": slice_metrics(rows),
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "actual_models": sorted(actual_models),
        "cases": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    cases = load_cases(args.dataset)
    dataset_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    question_sha256 = hashlib.sha256(
        json.dumps({"instructions": QUESTION, "criteria": CRITERIA}, sort_keys=True).encode()
    ).hexdigest()
    if args.validate_only:
        print(json.dumps({"case_count": len(cases), "dataset_sha256": dataset_sha256}))
        return
    if not args.output:
        parser.error("--output is required for a live run")
    try:
        api_key, base_url, default_model = jev_connection()
    except ValueError as exc:
        parser.error(str(exc))
    model = args.model or default_model
    try:
        from typesafe_sdk import Choice, TypeSafeClient
    except ImportError as exc:
        parser.error(f"typesafe-sdk is required for a live run: {exc}")

    choice = Choice(instructions=QUESTION, criteria=CRITERIA)
    client_options: dict[str, str] = {"model": model, "api_key": api_key}
    if base_url:
        client_options["base_url"] = base_url
    with TypeSafeClient(**client_options) as client:
        report = evaluate_cases(
            cases,
            lambda state: client.system_one(state=state, questions={"intent": choice}),
        )
    report.update(
        {
            "dataset": str(args.dataset),
            "dataset_sha256": dataset_sha256,
            "requested_model": model,
            "provider": (
                "opencode_zen" if base_url == OPENCODE_BASE_URL
                else "vercel_ai_gateway" if base_url == GATEWAY_BASE_URL
                else "custom" if base_url else "typesafe_direct"
            ),
            "question_sha256": question_sha256,
            "question": {"instructions": QUESTION, "criteria": CRITERIA},
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "case_count",
                    "valid_count",
                    "accuracy_all",
                    "accuracy_valid",
                    "agentic_recall_all",
                    "latency_ms",
                    "latency_ms_valid",
                    "usage",
                    "actual_models",
                )
            },
            ensure_ascii=False,
        )
    )
    if report["error_count"]:
        print(
            f"{report['error_count']} Jev calls failed; inspect per-case error_type",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
