"""Shared M0 case loading and metrics for the Jev evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from .routing_quality import ROUTE_LABELS


def load_holdout_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid holdout JSON at line {line_number}") from exc
        cases.append(
            {
                "case_id": str(record.get("id", f"holdout-{line_number}")),
                "question": str(record["input"]["current_question"]),
                "session_context": list(record["input"].get("session_context", [])),
                "expected": str(record["expected"]["intent"]),
                "language": str(record.get("metadata", {}).get("language", "unknown")),
                "difficulty": str(
                    record.get("metadata", {}).get("difficulty", "unknown")
                ),
                "scenario": str(record.get("metadata", {}).get("scenario", "unknown")),
            }
        )
    if not cases:
        raise ValueError(f"holdout dataset is empty: {path}")
    return cases


def per_class_metrics(
    gold: Sequence[str], predicted: Sequence[str]
) -> dict[str, dict[str, float]]:
    per_class: dict[str, dict[str, float]] = {}
    for label in ROUTE_LABELS:
        tp = sum(
            1 for g, p in zip(gold, predicted, strict=True) if g == label and p == label
        )
        fp = sum(
            1 for g, p in zip(gold, predicted, strict=True) if g != label and p == label
        )
        fn = sum(
            1 for g, p in zip(gold, predicted, strict=True) if g == label and p != label
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    return per_class


def latency_summary(latencies: Sequence[float]) -> dict[str, float]:
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    ordered = sorted(latencies)

    def quantile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 3)

    return {
        "p50": quantile(0.50),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "mean": round(mean(ordered), 3),
        "max": round(ordered[-1], 3),
    }


def slice_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    slices: dict[str, Any] = {}
    for key in ("language", "difficulty", "scenario", "has_context"):
        grouped: dict[Any, list[Mapping[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row.get(key), []).append(row)
        slices[key] = {
            str(value): {
                "count": len(group),
                "accuracy": round(
                    sum(1 for row in group if row["correct"]) / len(group), 4
                ),
                "degraded_count": sum(1 for row in group if row["degraded"]),
                "latency_p50_ms": latency_summary([row["latency_ms"] for row in group])[
                    "p50"
                ],
            }
            for value, group in sorted(grouped.items(), key=lambda item: str(item[0]))
        }
    return slices
