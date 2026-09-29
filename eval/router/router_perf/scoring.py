from __future__ import annotations

import json
import math
from collections.abc import Iterable
from statistics import fmean
from typing import Any

from eval.router.router_perf.contract import PerfMeasurement


def _summary(values: Iterable[float | None]) -> dict[str, float | None]:
    available = sorted(float(value) for value in values if value is not None)
    if not available:
        return {"mean": None, "p50": None, "p95": None}

    def nearest_rank(percentile: float) -> float:
        rank = max(1, math.ceil(percentile * len(available)))
        return available[rank - 1]

    return {
        "mean": fmean(available),
        "p50": nearest_rank(0.50),
        "p95": nearest_rank(0.95),
    }


def score_measurements(measurements: list[PerfMeasurement]) -> dict[str, Any]:
    if not measurements:
        raise ValueError("cannot score an empty performance run")
    return {
        "measurement_count": len(measurements),
        "protocol_rate": sum(row.protocol_pass for row in measurements)
        / len(measurements),
        "intent_match_rate": sum(row.intent_correct for row in measurements)
        / len(measurements),
        "eligible": all(
            row.protocol_pass and row.intent_correct and row.error_code is None
            for row in measurements
        ),
        "ttft_ms": _summary(row.ttft_ms for row in measurements),
        "total_ms": _summary(row.total_ms for row in measurements),
        "prefill_ms": _summary(row.prefill_ms for row in measurements),
        "prefill_tps": _summary(row.prefill_tps for row in measurements),
        "decode_ms": _summary(row.decode_ms for row in measurements),
        "decode_tps": _summary(row.decode_tps for row in measurements),
        "cache_hit_ratio_estimate": _summary(
            row.cache_hit_ratio_estimate for row in measurements
        ),
        "rss_peak_mb": _summary(row.rss_peak_mb for row in measurements),
    }


def rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def value(record: dict[str, Any], metric: str, statistic: str, default: float) -> float:
        group = record.get(metric)
        if not isinstance(group, dict):
            return default
        raw = group.get(statistic)
        return float(raw) if isinstance(raw, (int, float)) else default

    return sorted(
        candidates,
        key=lambda item: (
            not bool(item.get("eligible")),
            value(item, "total_ms", "p95", math.inf),
            value(item, "total_ms", "mean", math.inf),
            -value(item, "decode_tps", "mean", -math.inf),
            value(item, "rss_peak_mb", "mean", math.inf),
            json.dumps(item.get("config", {}), sort_keys=True, separators=(",", ":")),
        ),
    )
