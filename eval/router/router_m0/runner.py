from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .backends import LlamaRouterBackend, MockRouterBackend, RouterBackend
from .contract import RouterCase, load_router_cases
from .prompt import build_router_messages
from .reporting import render_model_readme, write_predictions, write_scorecard
from .scoring import CaseScore, aggregate_scores, score_generation


def run_router_dataset(
    cases: Sequence[RouterCase],
    backend: RouterBackend | MockRouterBackend,
    output_dir: Path,
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    sorted_cases = sorted(cases, key=lambda case: case.id)
    rows: list[CaseScore] = []
    for case in sorted_cases:
        messages = build_router_messages(case)
        if isinstance(backend, MockRouterBackend):
            generation = backend.generate_for_case(case.id, messages)
        else:
            generation = backend.generate(messages)
        rows.append(score_generation(case, generation))

    expected_ids = [case.id for case in sorted_cases]
    result_ids = [row.case_id for row in rows]
    if result_ids != expected_ids:
        raise ValueError("Router result IDs do not match dataset case IDs")

    scorecard = aggregate_scores(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir / "predictions.jsonl", rows)
    write_scorecard(output_dir / "scorecard.json", scorecard)
    write_scorecard(output_dir / "lineage.json", lineage)
    (output_dir / "README.md").write_text(
        render_model_readme(backend.model, scorecard),
        encoding="utf-8",
        newline="\n",
    )
    return scorecard


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an isolated Router M0 dataset")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("mock", "llama"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    cases = load_router_cases(args.dataset)
    if args.backend == "mock":
        responses = {
            case.id: json.dumps(
                case.expected.model_dump(), ensure_ascii=False, separators=(",", ":")
            )
            for case in cases
        }
        backend: RouterBackend | MockRouterBackend = MockRouterBackend(responses)
    else:
        if not args.model or not args.base_url:
            raise SystemExit("--model and --base-url are required for llama backend")
        backend = LlamaRouterBackend(args.model, args.base_url)

    scorecard = run_router_dataset(
        cases,
        backend,
        args.output_dir,
        {"backend": args.backend, "model": backend.model, "status": "complete"},
    )
    print(
        f"cases={scorecard['case_count']} "
        f"protocol={scorecard['protocol_rate']:.4f} "
        f"effective_accuracy={scorecard['effective_accuracy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
