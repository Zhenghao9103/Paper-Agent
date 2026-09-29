from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from eval.router.router_m0.server import sha256_file
from eval.router.router_perf.contract import (
    PerfFixture,
    PerfMeasurement,
    ServerConfig,
    load_perf_fixtures,
)
from eval.router.router_perf.runner import PerfRun, write_run_artifacts
from eval.router.router_perf.server import PerfServerManager
from eval.router.router_perf.streaming import StreamingRouterClient

SCENARIOS = ("cold_first", "warm_no_hit", "warm_prefix_hit", "exact_repeat")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _measure(
    client: StreamingRouterClient,
    fixture: PerfFixture,
    *,
    scenario: str,
    repeat: int,
    manager: PerfServerManager,
) -> PerfMeasurement:
    row = client.generate(
        fixture,
        scenario=scenario,
        repeat=repeat,
        rss_peak_mb=manager.rss_peak_mb,
    )
    return replace(row, rss_peak_mb=manager.rss_peak_mb)


def _start_server(
    *,
    executable: Path,
    model: Path,
    output_dir: Path,
    port: int,
    config: ServerConfig,
) -> tuple[PerfServerManager, StreamingRouterClient, float]:
    manager = PerfServerManager(
        executable=executable,
        model_path=model,
        output_dir=output_dir,
        port=port,
        config=config,
    )
    manager.start()
    ready_ms = manager.wait_ready(180.0)
    client = StreamingRouterClient(
        model=model.name,
        base_url=f"http://127.0.0.1:{port}/v1",
    )
    return manager, client, ready_ms


def run_live(
    *,
    executable: Path,
    model: Path,
    output_dir: Path,
    port: int,
    config: ServerConfig,
    fixtures: list[PerfFixture],
    scenarios: tuple[str, ...],
    repeats: int,
) -> tuple[PerfRun, list[float]]:
    rows: list[PerfMeasurement] = []
    ready_times: list[float] = []
    for scenario in scenarios:
        scenario_config = replace(
            config,
            cache_prompt=scenario not in {"cold_first", "warm_no_hit"},
        )
        if scenario == "cold_first":
            for fixture in fixtures:
                for repeat in range(1, repeats + 1):
                    run_dir = output_dir / "servers" / scenario / f"{fixture.id}-{repeat}"
                    manager, client, ready_ms = _start_server(
                        executable=executable,
                        model=model,
                        output_dir=run_dir,
                        port=port,
                        config=scenario_config,
                    )
                    ready_times.append(ready_ms)
                    try:
                        rows.append(
                            _measure(
                                client,
                                fixture,
                                scenario=scenario,
                                repeat=repeat,
                                manager=manager,
                            )
                        )
                    finally:
                        manager.stop()
            continue

        manager, client, ready_ms = _start_server(
            executable=executable,
            model=model,
            output_dir=output_dir / "servers" / scenario,
            port=port,
            config=scenario_config,
        )
        ready_times.append(ready_ms)
        try:
            if scenario == "warm_prefix_hit":
                client.generate(
                    fixtures[0], scenario="warmup", repeat=0, rss_peak_mb=None
                )
            for fixture in fixtures:
                if scenario == "exact_repeat":
                    client.generate(
                        fixture, scenario="warmup", repeat=0, rss_peak_mb=None
                    )
                for repeat in range(1, repeats + 1):
                    rows.append(
                        _measure(
                            client,
                            fixture,
                            scenario=scenario,
                            repeat=repeat,
                            manager=manager,
                        )
                    )
        finally:
            manager.stop()
    return PerfRun(scenarios, tuple(rows)), ready_times


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Router Phase 6 CPU configuration")
    parser.add_argument("stage", choices=("baseline", "toolchain", "threads", "prefill", "prompt-cache", "kv-cache", "final"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--threads-batch", type=int, default=4)
    parser.add_argument("--cache-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--scenario", action="append", choices=SCENARIOS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--port", type=int, default=18091)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    executable = args.llama_server.resolve()
    model = args.model.resolve()
    dataset = args.dataset.resolve()
    output_dir = args.output_root.resolve() / args.stage / args.name
    config = ServerConfig(
        threads=args.threads,
        threads_batch=args.threads_batch,
        cache_prompt=args.cache_prompt,
        cache_type_k=args.cache_type_k,
        cache_type_v=args.cache_type_v,
    )
    scenarios = tuple(args.scenario or ("warm_prefix_hit",))
    fixtures = load_perf_fixtures(dataset)
    run, ready_times = run_live(
        executable=executable,
        model=model,
        output_dir=output_dir,
        port=args.port,
        config=config,
        fixtures=fixtures,
        scenarios=scenarios,
        repeats=args.repeats,
    )
    config_payload = {
        "stage": args.stage,
        "name": args.name,
        "server": asdict(config),
        "scenarios": scenarios,
        "repeats": args.repeats,
        "ready_ms": ready_times,
        "llama_server": str(executable),
        "model": str(model),
        "dataset": str(dataset),
    }
    manifest = write_run_artifacts(
        output_dir,
        run=run,
        config=config_payload,
        model_sha256=sha256_file(model),
        fixture_sha256=_file_sha256(dataset),
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
