"""Deterministic split and selection helpers for retrieval Top-K tuning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from .rag_quality import RetrievalEvalCase, RetrievalRunResult, evaluate_retrieval_results


def build_paper_stratified_split(dataset: Any) -> dict[str, list[str]]:
    """Select one single-paper case per corpus paper without reading outcomes."""

    dev: list[str] = []
    test: list[str] = []
    for paper_index, paper in enumerate(dataset.corpus):
        paper_cases = sorted(
            (
                case
                for case in dataset.cases
                if not case.expect_insufficient_evidence
                and list(case.required_documents) == [paper.key]
            ),
            key=lambda case: str(case.case_id),
        )
        if len(paper_cases) != 3:
            raise ValueError(
                f"expected exactly three single-paper cases for {paper.key}, "
                f"found {len(paper_cases)}"
            )
        dev_index = paper_index % 3
        for case_index, case in enumerate(paper_cases):
            target = dev if case_index == dev_index else test
            target.append(str(case.case_id))
    return {"dev_case_ids": dev, "test_case_ids": test}


def select_smallest_qualifying_k(
    curve: Mapping[int, Mapping[str, float]],
    *,
    minimum_hit: float = 0.90,
    minimum_document_recall: float = 0.95,
) -> int | None:
    for k in sorted(curve):
        metrics = curve[k]
        if (
            float(metrics["hit_at_k"]) >= minimum_hit
            and float(metrics["required_document_recall_at_k"])
            >= minimum_document_recall
        ):
            return int(k)
    return None


def validate_complete_case_ids(
    actual: set[str], expected: set[str], *, label: str
) -> None:
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{label} case IDs are incomplete: missing={missing}, unexpected={unexpected}"
        )


def evaluate_arm_topk(
    cases: list[RetrievalEvalCase],
    results: list[RetrievalRunResult],
    *,
    dev_case_ids: list[str],
    test_case_ids: list[str],
    candidate_ks: tuple[int, ...],
) -> dict[str, Any]:
    case_by_id = {case.case_id: case for case in cases}
    result_by_id = {result.case_id: result for result in results}

    def evaluate(case_ids: list[str], ks: tuple[int, ...]) -> dict[str, Any]:
        return asdict(
            evaluate_retrieval_results(
                [case_by_id[case_id] for case_id in case_ids],
                [result_by_id[case_id] for case_id in case_ids],
                top_ks=ks,
            )
        )

    dev_metrics = evaluate(dev_case_ids, candidate_ks)
    curve: dict[int, dict[str, float]] = {}
    for k in candidate_ks:
        curve[k] = {
            "hit_at_k": float(dev_metrics["hit_at_k"][k]),
            "recall_at_k": float(dev_metrics["recall_at_k"][k]),
            "mrr_at_k": float(dev_metrics["mrr_at_k"][k]),
            "map_at_k": float(dev_metrics["map_at_k"][k]),
            "ndcg_at_k": float(dev_metrics["ndcg_at_k"][k]),
            "required_document_recall_at_k": float(
                dev_metrics["all_documents_recall_at_k"][k]
            ),
        }
    selected_k = select_smallest_qualifying_k(curve)
    test_metrics = None
    if selected_k is not None:
        scored = evaluate(test_case_ids, (selected_k,))
        test_metrics = {
            "evaluated_k": selected_k,
            "hit_at_k": float(scored["hit_at_k"][selected_k]),
            "recall_at_k": float(scored["recall_at_k"][selected_k]),
            "mrr_at_k": float(scored["mrr_at_k"][selected_k]),
            "map_at_k": float(scored["map_at_k"][selected_k]),
            "ndcg_at_k": float(scored["ndcg_at_k"][selected_k]),
            "required_document_recall_at_k": float(
                scored["all_documents_recall_at_k"][selected_k]
            ),
        }
    return {
        "selected_k": selected_k,
        "selection_gates": {
            "minimum_hit_at_k": 0.90,
            "minimum_required_document_recall_at_k": 0.95,
        },
        "dev_curve": {str(k): metrics for k, metrics in curve.items()},
        "test_metrics": test_metrics,
    }
