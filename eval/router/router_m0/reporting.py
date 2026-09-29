from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .scoring import LABELS, PREDICTION_COLUMNS, CaseScore


def write_predictions(path: Path, rows: Sequence[CaseScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(asdict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def write_scorecard(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_slice_table(
    title: str, slices: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Slice | Support | Effective Accuracy | Protocol |",
        "|---|---:|---:|---:|",
    ]
    for name, metrics in slices.items():
        lines.append(
            "| {name} | {support} | {accuracy} | {protocol} |".format(
                name=name,
                support=metrics["support"],
                accuracy=_format_value(metrics["effective_accuracy"]),
                protocol=_format_value(metrics["protocol_rate"]),
            )
        )
    lines.append("")
    return lines


def render_model_readme(model: str, scorecard: Mapping[str, Any]) -> str:
    lines = [
        f"# Router M0 report: {model}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "case_count",
        "protocol_rate",
        "effective_accuracy",
        "macro_f1",
        "agentic_recall",
        "pair_accuracy",
    ):
        lines.append(f"| {key} | {_format_value(scorecard[key])} |")

    lines.extend(
        [
            "",
            "## Per class",
            "",
            "| Intent | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in LABELS:
        metrics = scorecard["per_class"][label]
        lines.append(
            "| {label} | {precision} | {recall} | {f1} | {support} |".format(
                label=label,
                precision=_format_value(metrics["precision"]),
                recall=_format_value(metrics["recall"]),
                f1=_format_value(metrics["f1"]),
                support=metrics["support"],
            )
        )

    lines.extend(
        [
            "",
            "## Confusion matrix",
            "",
            "| Gold / Prediction | " + " | ".join(PREDICTION_COLUMNS) + " |",
            "|---|" + "---:|" * len(PREDICTION_COLUMNS),
        ]
    )
    for label in LABELS:
        counts = scorecard["confusion"][label]
        lines.append(
            f"| {label} | "
            + " | ".join(str(counts[column]) for column in PREDICTION_COLUMNS)
            + " |"
        )

    lines.extend(["", "## Slices", ""])
    for slice_name, values in scorecard["slices"].items():
        lines.extend(_render_slice_table(slice_name, values))

    lines.extend(
        [
            "## Failures",
            "",
            "| Failure code | Count |",
            "|---|---:|",
        ]
    )
    if scorecard["failure_counts"]:
        for code, count in scorecard["failure_counts"].items():
            lines.append(f"| {code} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(["", "## Timing", "", "| Measure | Mean | P50 | P95 |", "|---|---:|---:|---:|"])
    for timing_name, values in scorecard["latency"].items():
        lines.append(
            "| {name} | {mean} | {p50} | {p95} |".format(
                name=timing_name,
                mean=_format_value(values["mean"]),
                p50=_format_value(values["p50"]),
                p95=_format_value(values["p95"]),
            )
        )
    return "\n".join(lines) + "\n"
