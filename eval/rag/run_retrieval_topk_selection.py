"""Select retrieval Top-K offline from frozen reranked Top-20 ranks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .query_rewrite_rerank import extract_raw_rerank_cases
from .rag_gold import RAGGoldDataset
from .retrieval_topk_selection import (
    build_paper_stratified_split,
    evaluate_arm_topk,
    validate_complete_case_ids,
)
from .run_query_rewrite_ablation import _evaluation_cases, _run_results
from .run_query_rewrite_rerank import (
    _load_chunks,
    _write_json_atomic,
    _write_text_atomic,
)
from .run_rag_component_eval import (
    AUDITED_GOLD_NOTES,
    AUDITED_GOLD_OVERRIDES,
    resolve_frozen_document_ids,
    select_simple_rag_cases,
)

CANDIDATE_KS = (3, 5, 8, 10, 15, 20)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rewrite_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("rewrite rerank report must contain a cases list")
    rows = {str(row["case_id"]): dict(row) for row in cases}
    failed = sorted(case_id for case_id, row in rows.items() if row.get("error"))
    if failed:
        raise ValueError(f"rewrite rerank report contains failed cases: {failed}")
    return rows


def _distribution(dataset: Any, case_ids: list[str]) -> dict[str, Any]:
    wanted = set(case_ids)
    rows = [case for case in dataset.cases if str(case.case_id) in wanted]

    def counts(attribute: str) -> dict[str, int]:
        output: dict[str, int] = {}
        for case in rows:
            key = str(getattr(case, attribute))
            output[key] = output.get(key, 0) + 1
        return dict(sorted(output.items()))

    return {
        "case_count": len(rows),
        "language": counts("language"),
        "difficulty": counts("difficulty"),
        "papers": len({case.required_documents[0] for case in rows}),
    }


def _markdown(scorecard: dict[str, Any]) -> str:
    lines = [
        "# Retrieval Top-K Selection",
        "",
        (
            "Selection uses the 20-case paper-stratified dev split only. "
            "The 40-case test split is evaluated once at the selected K."
        ),
        "",
    ]
    for arm, report in scorecard["arms"].items():
        lines.extend(
            [
                f"## {arm}",
                "",
                "| K | Hit@K | Recall@K | MRR@K | MAP@K | nDCG@K | Required-doc Recall@K |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for k in CANDIDATE_KS:
            metrics = report["dev_curve"][str(k)]
            lines.append(
                f"| {k} | {metrics['hit_at_k']:.4f} | {metrics['recall_at_k']:.4f} | "
                f"{metrics['mrr_at_k']:.4f} | {metrics['map_at_k']:.4f} | "
                f"{metrics['ndcg_at_k']:.4f} | "
                f"{metrics['required_document_recall_at_k']:.4f} |"
            )
        lines.extend(["", f"Selected K: **{report['selected_k']}**.", ""])
        if report["test_metrics"] is not None:
            test = report["test_metrics"]
            lines.append(
                f"Held-out test at K={test['evaluated_k']}: Hit={test['hit_at_k']:.4f}, "
                f"Recall={test['recall_at_k']:.4f}, MRR={test['mrr_at_k']:.4f}, "
                f"Required-doc Recall={test['required_document_recall_at_k']:.4f}."
            )
            lines.append("")
    lines.extend(
        [
            f"Primary recommendation for Rewrite + Rerank: **K={scorecard['recommended_k']}**.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = args.dataset.resolve()
    raw_report_path = args.raw_report.resolve()
    rewrite_path = args.rewrite_rerank_report.resolve()
    db_path = args.db.resolve()
    output_dir = args.output_dir.resolve()
    for path in (dataset_path, raw_report_path, rewrite_path, db_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    dataset = RAGGoldDataset.model_validate_json(
        dataset_path.read_text(encoding="utf-8")
    )
    simple_cases = select_simple_rag_cases(dataset)
    expected_ids = {str(case.case_id) for case in simple_cases}
    raw_report = json.loads(raw_report_path.read_text(encoding="utf-8"))
    rewrite_payload = json.loads(rewrite_path.read_text(encoding="utf-8"))
    raw_rows = extract_raw_rerank_cases(raw_report)
    rewrite_rows = _rewrite_rows(rewrite_payload)
    validate_complete_case_ids(set(raw_rows), expected_ids, label="raw rerank")
    validate_complete_case_ids(set(rewrite_rows), expected_ids, label="rewrite rerank")

    split = build_paper_stratified_split(dataset)
    chunks = _load_chunks(db_path)
    metadata = {
        chunk_id: (chunk.document_id, chunk.page_number)
        for chunk_id, chunk in chunks.items()
    }
    gold_by_case = {
        str(case_id): [int(value) for value in values]
        for case_id, values in (raw_report.get("gold_chunks") or {}).items()
    }
    gold_by_case.update(AUDITED_GOLD_OVERRIDES)
    document_ids = resolve_frozen_document_ids(dataset, db_path)
    evaluation_cases = _evaluation_cases(
        simple_cases,
        gold_by_case=gold_by_case,
        document_ids=document_ids,
        chunk_metadata=metadata,
    )
    arms = {
        "raw_rerank": evaluate_arm_topk(
            evaluation_cases,
            _run_results(raw_rows, chunk_metadata=metadata),
            dev_case_ids=split["dev_case_ids"],
            test_case_ids=split["test_case_ids"],
            candidate_ks=CANDIDATE_KS,
        ),
        "rewrite_rerank": evaluate_arm_topk(
            evaluation_cases,
            _run_results(rewrite_rows, chunk_metadata=metadata),
            dev_case_ids=split["dev_case_ids"],
            test_case_ids=split["test_case_ids"],
            candidate_ks=CANDIDATE_KS,
        ),
    }
    scorecard = {
        "evaluation": "retrieval-topk-selection-v1",
        "candidate_ks": list(CANDIDATE_KS),
        "primary_arm": "rewrite_rerank",
        "recommended_k": arms["rewrite_rerank"]["selected_k"],
        "arms": arms,
        "split": {
            **split,
            "dev_distribution": _distribution(dataset, split["dev_case_ids"]),
            "test_distribution": _distribution(dataset, split["test_case_ids"]),
        },
        "leakage_controls": {
            "split_uses_retrieval_outcomes": False,
            "selection_uses_dev_only": True,
            "test_evaluated_once_at_selected_k": True,
        },
        "gold_audit_overrides": AUDITED_GOLD_NOTES,
        "source_signature": rewrite_payload.get("signature"),
    }
    _write_json_atomic(output_dir / "topk-scorecard.json", scorecard)
    _write_text_atomic(output_dir / "topk-scorecard.md", _markdown(scorecard))
    _write_json_atomic(
        output_dir / "split-manifest.json",
        {
            "strategy": "one-case-per-paper-corpus-index-mod-3",
            "dataset_sha256": _sha256_file(dataset_path),
            "raw_report_sha256": _sha256_file(raw_report_path),
            "rewrite_rerank_sha256": _sha256_file(rewrite_path),
            **scorecard["split"],
        },
    )
    dev_ids = set(split["dev_case_ids"])
    test_ids = set(split["test_case_ids"])
    case_by_id = {str(case.case_id): case for case in simple_cases}
    lines = []
    for case_id in [*split["dev_case_ids"], *split["test_case_ids"]]:
        case = case_by_id[case_id]
        lines.append(
            json.dumps(
                {
                    "case_id": case_id,
                    "split": "dev" if case_id in dev_ids else "test",
                    "language": case.language,
                    "difficulty": case.difficulty,
                    "document": case.required_documents[0],
                    "raw_rerank_top20": [
                        int(item["chunk_id"])
                        for item in raw_rows[case_id]["ranked_chunks"]
                    ],
                    "rewrite_rerank_top20": [
                        int(item["chunk_id"])
                        for item in rewrite_rows[case_id]["ranked_chunks"]
                    ],
                },
                ensure_ascii=False,
            )
        )
    if dev_ids & test_ids:
        raise ValueError("dev and test splits overlap")
    _write_text_atomic(output_dir / "topk-cases.jsonl", "\n".join(lines) + "\n")
    return scorecard


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select reranked retrieval Top-K from frozen Top-20 ranks"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--rewrite-rerank-report", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    scorecard = run(_parser().parse_args(argv))
    print(json.dumps({"recommended_k": scorecard["recommended_k"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
