from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .matrix import valid_complete_lineage
from .reporting import write_scorecard
from .scoring import LABELS, PREDICTION_COLUMNS

STABLE_FAILURE_CODES = {
    "invalid_json",
    "not_json_object",
    "missing_intent",
    "extra_fields",
    "invalid_intent",
    "empty_output",
    "timeout",
    "server_error",
    "direct_as_simple",
    "direct_as_agentic",
    "simple_as_direct",
    "simple_as_agentic",
    "agentic_as_direct",
    "agentic_as_simple",
}
QUANTIZATION_NOTE = (
    "Qwen3-0.6B and Qwen3-1.7B use Q8_0 while Qwen3-4B-Instruct-2507 uses "
    "Q4_K_M. This is an engineering baseline, not a pure parameter-size comparison."
)


def eligibility_reasons(
    scorecard: Mapping[str, Any], lineage: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    if scorecard.get("case_count") != 60:
        reasons.append("case_count_not_60")
    if not valid_complete_lineage(lineage):
        reasons.append("lineage_incomplete")
    if float(scorecard.get("protocol_rate", 0.0)) < 0.98:
        reasons.append("protocol_below_0.98")
    per_class = scorecard.get("per_class", {})
    for label in LABELS:
        recall = float(per_class.get(label, {}).get("recall", 0.0))
        if recall < 0.80:
            reasons.append(f"class_recall_below_0.80:{label}")
    if float(scorecard.get("agentic_recall", 0.0)) < 0.90:
        reasons.append("agentic_recall_below_0.90")
    return reasons


def _parameter_b(model_id: str) -> float:
    match = re.search(r"(?<!\d)(0\.6|1\.7|4)b", model_id.lower())
    return float(match.group(1)) if match else float("inf")


def select_model(models: Sequence[Mapping[str, Any]]) -> str | None:
    eligible = [model for model in models if model.get("eligible")]
    if not eligible:
        return None
    max_passes = max(int(model["effective_pass_count"]) for model in eligible)
    quality_ties = [
        model
        for model in eligible
        if int(model["effective_pass_count"]) >= max_passes - 1
    ]
    selected = min(
        quality_ties,
        key=lambda model: (
            -float(model["macro_f1"]),
            -float(model["agentic_recall"]),
            float(model["parameter_b"]),
            float(model["p95_ms"]),
            -int(model["effective_pass_count"]),
        ),
    )
    return str(selected["model_id"])


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _failure_rows(
    model_id: str, predictions: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    prediction_ids = {str(row["case_id"]) for row in predictions}
    for row in predictions:
        if row.get("intent_correct"):
            continue
        code = row.get("failure_code")
        if code not in STABLE_FAILURE_CODES:
            raise ValueError(
                f"failure for {row.get('case_id')} must use a stable failure code"
            )
        case_id = str(row["case_id"])
        if case_id not in prediction_ids:
            raise ValueError(f"failure row does not link to prediction ID: {case_id}")
        failures.append(
            {
                "model_id": model_id,
                "case_id": case_id,
                "expected_intent": row["expected_intent"],
                "predicted_intent": row.get("predicted_intent"),
                "failure_code": code,
                "language": row["language"],
                "difficulty": row["difficulty"],
                "scenario": row["scenario"],
            }
        )
    return failures


def _recommendation(selected_model_id: str | None) -> str:
    if selected_model_id is None:
        return "audit_contract_prompt_and_labels_before_training"
    if "4b" in selected_model_id.lower():
        return "train_qwen3_1.7b_first_keep_4b_as_baseline"
    if "1.7b" in selected_model_id.lower():
        return "prioritize_qwen3_1.7b_sft_keep_0.6b_as_lightweight_control"
    return "include_qwen3_0.6b_and_qwen3_1.7b_as_sft_candidates"


def build_comparison(
    report_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    models: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    failure_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model_dir in sorted(path for path in report_root.iterdir() if path.is_dir()):
        scorecard_path = model_dir / "scorecard.json"
        predictions_path = model_dir / "predictions.jsonl"
        lineage_path = model_dir / "lineage.json"
        if not (
            scorecard_path.is_file()
            and predictions_path.is_file()
            and lineage_path.is_file()
        ):
            continue
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        if lineage.get("status") != "complete":
            continue
        predictions = _read_predictions(predictions_path)
        if len(predictions) != int(scorecard["case_count"]):
            raise ValueError(f"prediction count mismatch for {model_dir.name}")
        prediction_ids = [str(row["case_id"]) for row in predictions]
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ValueError(f"duplicate prediction ID for {model_dir.name}")

        model_id = str(lineage.get("model_id", model_dir.name))
        model_failures = _failure_rows(model_id, predictions)
        failures.extend(model_failures)
        failure_by_model[model_id].extend(model_failures)
        reasons = eligibility_reasons(scorecard, lineage)
        effective_pass_count = sum(
            bool(row.get("intent_correct")) for row in predictions
        )
        latency = scorecard["latency"]["total_ms"]
        models.append(
            {
                "model_id": model_id,
                "display_name": str(lineage.get("display_name", model_id)),
                "quantization": str(lineage.get("quantization", "unknown")),
                "parameter_b": _parameter_b(model_id),
                "case_count": scorecard["case_count"],
                "protocol_rate": scorecard["protocol_rate"],
                "effective_accuracy": scorecard["effective_accuracy"],
                "effective_pass_count": effective_pass_count,
                "macro_f1": scorecard["macro_f1"],
                "agentic_recall": scorecard["agentic_recall"],
                "pair_accuracy": scorecard["pair_accuracy"],
                "p50_ms": latency["p50"],
                "p95_ms": latency["p95"],
                "per_class": scorecard["per_class"],
                "confusion": scorecard["confusion"],
                "slices": scorecard["slices"],
                "eligible": not reasons,
                "eligibility_reasons": reasons,
            }
        )

    selected_model_id = select_model(models)
    failure_analysis: dict[str, dict[str, Any]] = {}
    for model in models:
        model_id = model["model_id"]
        rows = failure_by_model[model_id]
        counts = Counter(row["failure_code"] for row in rows)
        representatives: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            code = row["failure_code"]
            if len(representatives[code]) < 3:
                representatives[code].append(row["case_id"])
        failure_analysis[model_id] = {
            "counts": dict(sorted(counts.items())),
            "representative_case_ids": dict(sorted(representatives.items())),
        }

    comparison = {
        "quantization_note": QUANTIZATION_NOTE,
        "models": models,
        "selected_model_id": selected_model_id,
        "training_recommendation": _recommendation(selected_model_id),
        "failure_analysis": failure_analysis,
    }
    return comparison, failures


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_comparison_readme(comparison: Mapping[str, Any]) -> str:
    models = comparison["models"]
    lines = [
        "# Router M0 model comparison",
        "",
        comparison["quantization_note"],
        "",
        "## Main metrics",
        "",
        "| Model | Protocol | Effective Accuracy | Macro-F1 | Agentic Recall | "
        "Pair Accuracy | P50 | P95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        lines.append(
            (
                "| {model} | {protocol} | {effective} | {macro} | {agentic} | "
                "{pair} | {p50} | {p95} |"
            ).format(
                model=model["display_name"],
                protocol=_fmt(model["protocol_rate"]),
                effective=_fmt(model["effective_accuracy"]),
                macro=_fmt(model["macro_f1"]),
                agentic=_fmt(model["agentic_recall"]),
                pair=_fmt(model["pair_accuracy"]),
                p50=_fmt(model["p50_ms"]),
                p95=_fmt(model["p95_ms"]),
            )
        )

    lines.extend(
        [
            "",
            "## Class metrics",
            "",
            "| Model | Direct F1 | Simple RAG F1 | Agentic RAG F1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for model in models:
        per_class = model["per_class"]
        lines.append(
            f"| {model['display_name']} | {_fmt(per_class['direct']['f1'])} | "
            f"{_fmt(per_class['simple_rag']['f1'])} | "
            f"{_fmt(per_class['agentic_rag']['f1'])} |"
        )

    for model in models:
        lines.extend(
            [
                "",
                f"## Confusion matrix: {model['display_name']}",
                "",
                "| Gold / Prediction | " + " | ".join(PREDICTION_COLUMNS) + " |",
                "|---|" + "---:|" * len(PREDICTION_COLUMNS),
            ]
        )
        for label in LABELS:
            counts = model["confusion"][label]
            lines.append(
                f"| {label} | "
                + " | ".join(str(counts[column]) for column in PREDICTION_COLUMNS)
                + " |"
            )

    lines.extend(["", "## Slices", ""])
    for slice_name in ("language", "difficulty", "context", "scenario"):
        lines.extend(
            [
                f"### {slice_name}",
                "",
                "| Model | Slice | Support | Effective Accuracy | Protocol |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for model in models:
            for name, metrics in model["slices"].get(slice_name, {}).items():
                lines.append(
                    f"| {model['display_name']} | {name} | {metrics['support']} | "
                    f"{_fmt(metrics['effective_accuracy'])} | {_fmt(metrics['protocol_rate'])} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Eligibility and recommendation",
            "",
            "| Model | Eligible | Reasons | Selected |",
            "|---|---|---|---|",
        ]
    )
    for model in models:
        reasons = ", ".join(model["eligibility_reasons"]) or "none"
        selected = model["model_id"] == comparison["selected_model_id"]
        lines.append(
            f"| {model['display_name']} | {model['eligible']} | {reasons} | {selected} |"
        )
    lines.extend(
        [
            "",
            f"Training recommendation: `{comparison['training_recommendation']}`.",
            "",
            "## Failure analysis",
            "",
            "| Model | Failure code | Count | Representative case IDs |",
            "|---|---|---:|---|",
        ]
    )
    for model in models:
        analysis = comparison["failure_analysis"][model["model_id"]]
        if not analysis["counts"]:
            lines.append(f"| {model['display_name']} | none | 0 | none |")
            continue
        for code, count in analysis["counts"].items():
            case_ids = ", ".join(analysis["representative_case_ids"][code])
            lines.append(f"| {model['display_name']} | {code} | {count} | {case_ids} |")
    return "\n".join(lines) + "\n"


def write_comparison(report_root: Path) -> dict[str, Any]:
    comparison, failures = build_comparison(report_root)
    write_scorecard(report_root / "comparison.json", comparison)
    with (report_root / "failures.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for failure in failures:
            handle.write(
                json.dumps(failure, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    (report_root / "README.md").write_text(
        render_comparison_readme(comparison), encoding="utf-8", newline="\n"
    )
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare completed Router M0 runs")
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    comparison = write_comparison(args.report_root)
    print(
        f"models={len(comparison['models'])} "
        f"selected={comparison['selected_model_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
