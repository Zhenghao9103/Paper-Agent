from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .backends import RouterGeneration
from .contract import RouterCase, RouterProtocolError, parse_router_output

LABELS = ("direct", "simple_rag", "agentic_rag")
PREDICTION_COLUMNS = (*LABELS, "invalid")
SHORT_LABELS = {
    "direct": "direct",
    "simple_rag": "simple",
    "agentic_rag": "agentic",
}


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    expected_intent: str
    predicted_intent: str | None
    protocol_pass: bool
    intent_correct: bool
    failure_code: str | None
    latency_ms: float
    first_token_ms: float | None
    language: str
    difficulty: str
    scenario: str
    pair_id: str | None
    has_context: bool
    raw_output: str = ""


def score_generation(case: RouterCase, generation: RouterGeneration) -> CaseScore:
    predicted: str | None = None
    protocol_pass = False
    failure_code = generation.error_code
    if failure_code is None:
        try:
            output = parse_router_output(generation.raw_output)
        except RouterProtocolError as exc:
            failure_code = exc.code
        else:
            predicted = output.intent
            protocol_pass = True
            if predicted != case.expected.intent:
                expected_name = SHORT_LABELS[case.expected.intent]
                predicted_name = SHORT_LABELS[predicted]
                failure_code = f"{expected_name}_as_{predicted_name}"

    intent_correct = protocol_pass and predicted == case.expected.intent
    return CaseScore(
        case_id=case.id,
        expected_intent=case.expected.intent,
        predicted_intent=predicted,
        protocol_pass=protocol_pass,
        intent_correct=intent_correct,
        failure_code=failure_code,
        latency_ms=generation.total_ms,
        first_token_ms=generation.first_token_ms,
        language=case.metadata.language,
        difficulty=case.metadata.difficulty,
        scenario=case.metadata.scenario,
        pair_id=case.metadata.pair_id,
        has_context=bool(case.input.session_context),
        raw_output=generation.raw_output,
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _nearest_rank(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = max(0, math.ceil(percentile * len(sorted_values)) - 1)
    return sorted_values[index]


def _latency_summary(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": _nearest_rank(values, 0.50),
        "p95": _nearest_rank(values, 0.95),
    }


def _slice_metrics(rows: Sequence[CaseScore]) -> dict[str, float | int]:
    support = len(rows)
    return {
        "support": support,
        "effective_accuracy": _safe_rate(
            sum(row.intent_correct for row in rows), support
        ),
        "protocol_rate": _safe_rate(sum(row.protocol_pass for row in rows), support),
    }


def _group_slices(
    rows: Sequence[CaseScore], key_name: str
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[CaseScore]] = defaultdict(list)
    for row in rows:
        if key_name == "context":
            key = "with_context" if row.has_context else "single_turn"
        else:
            key = str(getattr(row, key_name))
        grouped[key].append(row)
    return {key: _slice_metrics(grouped[key]) for key in sorted(grouped)}


def aggregate_scores(rows: Sequence[CaseScore]) -> dict[str, Any]:
    case_count = len(rows)
    confusion = {
        label: {column: 0 for column in PREDICTION_COLUMNS} for label in LABELS
    }
    for row in rows:
        column = row.predicted_intent if row.protocol_pass else "invalid"
        confusion[row.expected_intent][column] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        support = sum(confusion[label].values())
        fn = support - tp
        precision = _safe_rate(tp, tp + fp)
        recall = _safe_rate(tp, tp + fn)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    pairs: dict[str, list[CaseScore]] = defaultdict(list)
    for row in rows:
        if row.pair_id is not None:
            pairs[row.pair_id].append(row)
    pair_accuracy = _safe_rate(
        sum(all(row.intent_correct for row in pair_rows) for pair_rows in pairs.values()),
        len(pairs),
    )

    first_token_values = [
        row.first_token_ms for row in rows if row.first_token_ms is not None
    ]
    failure_counts = Counter(
        row.failure_code for row in rows if row.failure_code is not None
    )
    return {
        "case_count": case_count,
        "protocol_rate": _safe_rate(
            sum(row.protocol_pass for row in rows), case_count
        ),
        "effective_accuracy": _safe_rate(
            sum(row.intent_correct for row in rows), case_count
        ),
        "macro_f1": statistics.fmean(
            per_class[label]["f1"] for label in LABELS
        ),
        "agentic_recall": per_class["agentic_rag"]["recall"],
        "pair_accuracy": pair_accuracy,
        "per_class": per_class,
        "confusion": confusion,
        "latency": {
            "total_ms": _latency_summary([row.latency_ms for row in rows]),
            "first_token_ms": _latency_summary(first_token_values),
        },
        "slices": {
            "language": _group_slices(rows, "language"),
            "difficulty": _group_slices(rows, "difficulty"),
            "context": _group_slices(rows, "context"),
            "scenario": _group_slices(rows, "scenario"),
        },
        "failure_counts": dict(sorted(failure_counts.items())),
    }
