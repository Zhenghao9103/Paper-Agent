"""Summarize persisted PDF pipeline timings across documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, median
from typing import Any

from ..core.paths import documents_dir


def build_report(root: Path) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    values: dict[str, list[int]] = {}
    for path in sorted(root.glob("*/pipeline-status.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            document_id = int(path.parent.name)
            timings = {
                str(key): max(0, int(value))
                for key, value in payload.get("timings_ms", {}).items()
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        documents.append(
            {
                "document_id": document_id,
                "pipeline_stage": payload.get("pipeline_stage"),
                "timings_ms": timings,
            }
        )
        for stage, duration in timings.items():
            values.setdefault(stage, []).append(duration)
    summary = {
        stage: {
            "mean": round(fmean(durations), 1),
            "p50": round(median(durations), 1),
            "p95": round(_percentile(durations, 0.95), 1),
        }
        for stage, durations in sorted(values.items())
    }
    return {"document_count": len(documents), "documents": documents, "summary_ms": summary}


def _percentile(values: list[int], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-root", type=Path, default=documents_dir())
    args = parser.parse_args()
    print(json.dumps(build_report(args.documents_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
