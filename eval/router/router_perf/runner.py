from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from eval.router.router_perf.contract import PerfFixture, PerfMeasurement
from eval.router.router_perf.scoring import score_measurements

SCENARIOS = (
    "cold_first",
    "warm_no_hit",
    "warm_prefix_hit",
    "exact_repeat",
)


class PerfBackend(Protocol):
    def generate(
        self,
        fixture: PerfFixture,
        *,
        scenario: str,
        repeat: int,
        rss_peak_mb: float | None,
    ) -> PerfMeasurement: ...


@dataclass(frozen=True)
class PerfRun:
    scenario_order: tuple[str, ...]
    measurements: tuple[PerfMeasurement, ...]


class PerfRunner:
    def __init__(self, *, fixtures: list[PerfFixture], backend: PerfBackend) -> None:
        self._fixtures = fixtures
        self._backend = backend

    def run(self, repeats: int) -> PerfRun:
        if repeats < 1:
            raise ValueError("repeats must be positive")
        measurements: list[PerfMeasurement] = []
        for scenario in SCENARIOS:
            for fixture in self._fixtures:
                for repeat in range(1, repeats + 1):
                    measurements.append(
                        self._backend.generate(
                            fixture,
                            scenario=scenario,
                            repeat=repeat,
                            rss_peak_mb=None,
                        )
                    )
        return PerfRun(SCENARIOS, tuple(measurements))


def classify_run(run: PerfRun, *, fixture_count: int, repeats: int) -> str:
    expected_count = len(run.scenario_order) * fixture_count * repeats
    keys = {
        (row.scenario, row.fixture_id, row.repeat) for row in run.measurements
    }
    if len(run.measurements) != expected_count or len(keys) != expected_count:
        return "incomplete"
    if any(row.error_code is not None for row in run.measurements):
        return "incomplete"
    return "complete"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_run_artifacts(
    output_dir: Path,
    *,
    run: PerfRun,
    config: dict[str, object],
    model_sha256: str,
    fixture_sha256: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    measurement_records = [asdict(row) for row in run.measurements]
    _write_json(output_dir / "config.json", config)
    (output_dir / "measurements.jsonl").write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for record in measurement_records
        ),
        encoding="utf-8",
        newline="\n",
    )
    scorecard = score_measurements(list(run.measurements))
    _write_json(output_dir / "scorecard.json", scorecard)
    digest = hashlib.sha256()
    for value in (
        model_sha256,
        fixture_sha256,
        json.dumps(config, sort_keys=True, separators=(",", ":")),
        json.dumps(measurement_records, sort_keys=True, separators=(",", ":")),
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    fixture_count = len({row.fixture_id for row in run.measurements})
    repeats = max((row.repeat for row in run.measurements), default=0)
    manifest: dict[str, object] = {
        "status": classify_run(run, fixture_count=fixture_count, repeats=repeats),
        "model_sha256": model_sha256,
        "fixture_sha256": fixture_sha256,
        "run_sha256": digest.hexdigest(),
        "measurement_count": len(run.measurements),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest
