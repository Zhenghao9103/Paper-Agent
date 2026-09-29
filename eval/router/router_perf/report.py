from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _metric(record: dict[str, Any], group: str, statistic: str) -> str:
    value = record.get(group, {}).get(statistic)
    return "-" if value is None else f"{float(value):.2f}"


def render_comparison_markdown(
    candidates: list[dict[str, Any]], *, selected_name: str | None
) -> str:
    lines = [
        "# Router Phase 6 Performance Comparison",
        "",
        "Target: warm loaded Router mean <= 3000 ms and P95 <= 5000 ms; "
        "protocol rate and intent match rate must both equal 100%.",
        "",
        f"Selected: `{selected_name}`" if selected_name else "Selected: none",
        "",
        "| Candidate | Eligible | Total mean ms | Total P95 ms | TTFT mean ms | "
        "Decode tok/s | Protocol | Intent match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in candidates:
        lines.append(
            (
                "| {name} | {eligible} | {mean} | {p95} | {ttft} | {decode} | "
                "{protocol:.3f} | {match:.3f} |"
            ).format(
                name=item.get("name", "unnamed"),
                eligible="yes" if item.get("eligible") else "no",
                mean=_metric(item, "total_ms", "mean"),
                p95=_metric(item, "total_ms", "p95"),
                ttft=_metric(item, "ttft_ms", "mean"),
                decode=_metric(item, "decode_tps", "mean"),
                protocol=float(item.get("protocol_rate", 0.0)),
                match=float(item.get("intent_match_rate", 0.0)),
            )
        )
    return "\n".join(lines) + "\n"


def write_comparison_report(
    output_dir: Path,
    candidates: list[dict[str, Any]],
    *,
    selected_name: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"selected": selected_name, "candidates": candidates}
    (output_dir / "selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "README.md").write_text(
        render_comparison_markdown(candidates, selected_name=selected_name),
        encoding="utf-8",
        newline="\n",
    )
