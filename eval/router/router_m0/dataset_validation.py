from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from .contract import RouterCase, load_router_cases

LABELS = ("direct", "simple_rag", "agentic_rag")
SCENARIO_QUOTAS = {
    "direct": {
        "library_metadata": 6,
        "system_capability": 4,
        "non_paper_interaction": 3,
        "direct_vs_simple_boundary": 4,
        "direct_vs_agentic_boundary": 2,
        "prompt_injection": 1,
    },
    "simple_rag": {
        "single_paper_fact": 4,
        "single_paper_explanation": 3,
        "general_academic_evidence": 2,
        "insufficient_evidence_probe": 2,
        "direct_vs_simple_boundary": 4,
        "simple_vs_agentic_boundary": 4,
        "prompt_injection": 1,
    },
    "agentic_rag": {
        "cross_paper_comparison": 4,
        "implicit_multistep": 3,
        "multi_aspect_synthesis": 2,
        "evidence_conflict": 2,
        "explicit_arxiv": 2,
        "simple_vs_agentic_boundary": 4,
        "direct_vs_agentic_boundary": 2,
        "prompt_injection": 1,
    },
}
PAIR_QUOTAS = {
    frozenset({"direct", "simple_rag"}): 4,
    frozenset({"simple_rag", "agentic_rag"}): 4,
    frozenset({"direct", "agentic_rag"}): 2,
}


def validate_phase01(cases: Sequence[RouterCase]) -> list[str]:
    errors: list[str] = []
    if len(cases) != 3:
        errors.append(f"Phase 1 must contain exactly 3 cases, found {len(cases)}")
    counts = Counter(case.expected.intent for case in cases)
    expected_counts = Counter({label: 1 for label in LABELS})
    if counts != expected_counts:
        errors.append(f"Phase 1 intent counts must be 1/1/1, found {dict(counts)}")
    if any(case.metadata.pair_id is not None for case in cases):
        errors.append("Phase 1 cases must not have pair IDs")
    return errors


def _check_counter(
    errors: list[str],
    name: str,
    actual: Counter,
    expected: Counter,
    scope: str = "global",
) -> None:
    if actual != expected:
        errors.append(
            f"{scope} {name} must be {dict(expected)}, found {dict(actual)}"
        )


def _normalized_input(case: RouterCase) -> tuple:
    def normalize(text: str) -> str:
        return " ".join(text.split())

    context = tuple(
        (message.role, normalize(message.content)) for message in case.input.session_context
    )
    return context, normalize(case.input.current_question)


def validate_phase02(cases: Sequence[RouterCase]) -> list[str]:
    errors: list[str] = []
    if len(cases) != 60:
        errors.append(f"Phase 2 must contain exactly 60 cases, found {len(cases)}")

    intent_counts = Counter(case.expected.intent for case in cases)
    _check_counter(
        errors,
        "intent counts",
        intent_counts,
        Counter({label: 20 for label in LABELS}),
    )
    _check_counter(
        errors,
        "language counts",
        Counter(case.metadata.language for case in cases),
        Counter({"zh": 30, "en": 30}),
    )
    _check_counter(
        errors,
        "difficulty counts",
        Counter(case.metadata.difficulty for case in cases),
        Counter({"easy": 18, "medium": 24, "hard": 18}),
    )
    _check_counter(
        errors,
        "context counts",
        Counter(
            "with_context" if case.input.session_context else "single_turn"
            for case in cases
        ),
        Counter({"single_turn": 36, "with_context": 24}),
    )

    for label in LABELS:
        class_cases = [case for case in cases if case.expected.intent == label]
        _check_counter(
            errors,
            "language counts",
            Counter(case.metadata.language for case in class_cases),
            Counter({"zh": 10, "en": 10}),
            label,
        )
        _check_counter(
            errors,
            "difficulty counts",
            Counter(case.metadata.difficulty for case in class_cases),
            Counter({"easy": 6, "medium": 8, "hard": 6}),
            label,
        )
        _check_counter(
            errors,
            "context counts",
            Counter(
                "with_context" if case.input.session_context else "single_turn"
                for case in class_cases
            ),
            Counter({"single_turn": 12, "with_context": 8}),
            label,
        )
        actual_scenarios = Counter(case.metadata.scenario for case in class_cases)
        expected_scenarios = Counter(SCENARIO_QUOTAS[label])
        if actual_scenarios != expected_scenarios:
            errors.append(
                f"{label} scenario quotas must be {dict(expected_scenarios)}, "
                f"found {dict(actual_scenarios)}"
            )

    pair_groups: dict[str, list[RouterCase]] = defaultdict(list)
    for case in cases:
        if case.metadata.pair_id is not None:
            pair_groups[case.metadata.pair_id].append(case)
    pair_combinations: Counter[frozenset[str]] = Counter()
    for pair_id, pair_cases in sorted(pair_groups.items()):
        if len(pair_cases) != 2:
            errors.append(
                f"pair {pair_id} must appear exactly twice, found {len(pair_cases)}"
            )
            continue
        combination = frozenset(case.expected.intent for case in pair_cases)
        if combination not in PAIR_QUOTAS:
            errors.append(
                f"pair {pair_id} has invalid label combination {sorted(combination)}"
            )
            continue
        pair_combinations[combination] += 1
    if pair_combinations != Counter(PAIR_QUOTAS):
        errors.append(
            "pair label quotas must be "
            f"{dict(PAIR_QUOTAS)}, found {dict(pair_combinations)}"
        )

    seen_inputs: dict[tuple, str] = {}
    for case in cases:
        normalized = _normalized_input(case)
        if normalized in seen_inputs:
            errors.append(
                "duplicate normalized input: "
                f"{seen_inputs[normalized]} and {case.id}"
            )
        else:
            seen_inputs[normalized] = case.id
    return errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(dataset_path: Path, sha_path: Path) -> None:
    parts = sha_path.read_text(encoding="ascii").strip().split()
    if len(parts) != 2 or len(parts[0]) != 64:
        raise ValueError("invalid SHA256 file format")
    expected, filename = parts
    if filename != dataset_path.name:
        raise ValueError(
            f"SHA256 filename mismatch: expected {dataset_path.name}, found {filename}"
        )
    actual = sha256_file(dataset_path)
    if actual.lower() != expected.lower():
        raise ValueError(f"SHA256 mismatch: expected {expected}, found {actual}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Router M0 datasets")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--phase", choices=("phase01", "phase02"), required=True)
    parser.add_argument("--sha", type=Path)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        cases = load_router_cases(args.dataset)
        errors = (
            validate_phase01(cases)
            if args.phase == "phase01"
            else validate_phase02(cases)
        )
        if args.sha is not None:
            verify_sha256(args.dataset, args.sha)
    except (OSError, ValueError) as exc:
        print(str(exc))
        return 2
    if errors:
        for error in errors:
            print(error)
        return 2
    print(f"cases={len(cases)} sha256={sha256_file(args.dataset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
