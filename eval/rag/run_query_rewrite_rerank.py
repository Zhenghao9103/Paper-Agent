"""Evaluate production reranking over frozen Raw/Rewrite Fusion candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.app.rag.rerank import _get_bge_reranker
from .query_rewrite_ablation import extract_raw_fusion_cases, paired_hit_outcomes
from .query_rewrite_rerank import extract_raw_rerank_cases, rerank_frozen_case
from .rag_gold import RAGGoldDataset
from .rag_quality import RetrievalEvalCase, RetrievalRunResult, evaluate_retrieval_results
from .run_query_rewrite_ablation import _evaluation_cases, _run_results
from .run_rag_component_eval import (
    AUDITED_GOLD_NOTES,
    AUDITED_GOLD_OVERRIDES,
    resolve_frozen_document_ids,
    select_simple_rag_cases,
)


@dataclass(frozen=True)
class FrozenChunk:
    document_id: int
    page_number: int
    content: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(paths: Mapping[str, Path]) -> str:
    payload = {
        "contract": "query-rewrite-rerank-only-v1",
        "inputs": {name: _sha256_file(path) for name, path in sorted(paths.items())},
        "reranker": "BAAI/bge-reranker-base",
        "fusion_top_k": 20,
        "metadata_filter": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _load_resume(path: Path, signature: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("signature") != signature:
        raise ValueError(f"resume signature mismatch: {path}")
    return [row for row in payload.get("cases", []) if not row.get("error")]


def _persist_cases(
    path: Path, *, signature: str, cases: Sequence[Mapping[str, Any]]
) -> None:
    _write_json_atomic(path, {"signature": signature, "cases": list(cases)})


def _load_chunks(db_path: Path) -> dict[int, FrozenChunk]:
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT id, document_id, page_number, content FROM document_chunks"
        ).fetchall()
    return {
        int(row[0]): FrozenChunk(
            document_id=int(row[1]), page_number=int(row[2]), content=str(row[3])
        )
        for row in rows
    }


def _run_rewrite_rerank(
    *,
    cases: Sequence[Any],
    rewrite_fusion_rows: Mapping[str, Mapping[str, Any]],
    rewrites_by_id: Mapping[str, Mapping[str, Any]],
    chunks_by_id: Mapping[int, FrozenChunk],
    output_path: Path,
    signature: str,
    resume: bool,
) -> list[dict[str, Any]]:
    rows = _load_resume(output_path, signature) if resume else []
    completed = {str(row["case_id"]) for row in rows}
    _get_bge_reranker()
    for case in cases:
        case_id = str(case.case_id)
        if case_id in completed:
            continue
        query = str(rewrites_by_id[case_id]["english_query"])
        frozen = list(rewrite_fusion_rows[case_id].get("ranked_chunks") or [])
        try:
            result = rerank_frozen_case(query, frozen, chunks_by_id=chunks_by_id)
        except Exception as exc:  # noqa: BLE001 - stable per-case failure artifact
            result = {
                "latency_ms": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
                "degraded": None,
                "rerank_applied": False,
                "ranked_chunks": [],
            }
        rows.append(
            {
                "case_id": case_id,
                "question": str(case.question),
                "rewrite_query": query,
                **result,
            }
        )
        _persist_cases(output_path, signature=signature, cases=rows)
    return rows


def _rankings(results: Sequence[RetrievalRunResult]) -> dict[str, list[int]]:
    return {
        result.case_id: [chunk.chunk_id for chunk in result.ranked_chunks]
        for result in results
    }


def _at(metrics: Mapping[str, Any], name: str, k: int) -> float:
    values = metrics[name]
    return float(values.get(k, values.get(str(k), 0.0)))


def score_four_arms(
    evaluation_cases: Sequence[RetrievalEvalCase],
    arms: Mapping[str, Sequence[RetrievalRunResult]],
    *,
    gold_by_case: Mapping[str, Sequence[int]],
    rerank_error_count: int,
) -> dict[str, Any]:
    expected = {"raw_fusion", "rewrite_fusion", "raw_rerank", "rewrite_rerank"}
    if set(arms) != expected:
        raise ValueError(f"four-arm evaluation requires {sorted(expected)}")
    metrics = {
        name: asdict(
            evaluate_retrieval_results(
                evaluation_cases, results, top_ks=(3, 5, 10, 20)
            )
        )
        for name, results in arms.items()
    }
    rankings = {name: _rankings(results) for name, results in arms.items()}

    def compare(left: str, right: str, k: int) -> dict[str, Any]:
        return paired_hit_outcomes(
            rankings[left], rankings[right], gold_by_case, k=k
        )

    comparisons = {
        "query_before_rerank_at_5": compare("raw_fusion", "rewrite_fusion", 5),
        "query_before_rerank_at_20": compare("raw_fusion", "rewrite_fusion", 20),
        "rerank_on_raw_at_5": compare("raw_fusion", "raw_rerank", 5),
        "rerank_on_raw_at_20": compare("raw_fusion", "raw_rerank", 20),
        "rerank_on_rewrite_at_5": compare("rewrite_fusion", "rewrite_rerank", 5),
        "rerank_on_rewrite_at_20": compare("rewrite_fusion", "rewrite_rerank", 20),
        "final_at_5": compare("raw_rerank", "rewrite_rerank", 5),
        "final_at_20": compare("raw_rerank", "rewrite_rerank", 20),
    }
    raw_final = metrics["raw_rerank"]
    rewrite_final = metrics["rewrite_rerank"]
    gates = {
        "final_hit5_improves": _at(rewrite_final, "hit_at_k", 5)
        > _at(raw_final, "hit_at_k", 5),
        "final_hit20_non_decreasing": _at(rewrite_final, "hit_at_k", 20)
        >= _at(raw_final, "hit_at_k", 20),
        "zero_rewrite_rerank_errors": rerank_error_count == 0,
    }
    return {
        "arms": {name: {"metrics": value} for name, value in metrics.items()},
        "comparisons": comparisons,
        "decision": {"adopt_rewrite_rerank": all(gates.values()), "gates": gates},
    }


def _extract_rewrite_fusion_cases(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("rewrite report must contain a cases list")
    return {str(row["case_id"]): dict(row) for row in cases}


def _extract_rewrites(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("rewrite file must contain a cases list")
    return {str(row["case_id"]): dict(row) for row in cases}


def _raw_rerank_latency(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    values = [float(row.get("latency_ms") or 0.0) for row in rows.values()]
    ordered = sorted(values)

    def percentile(q: float) -> float:
        if not ordered:
            return 0.0
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    return {"p50": percentile(0.5), "p95": percentile(0.95), "p99": percentile(0.99)}


def _markdown(scorecard: Mapping[str, Any]) -> str:
    names = (
        ("Raw Fusion", "raw_fusion"),
        ("Rewrite Fusion", "rewrite_fusion"),
        ("Raw + Rerank", "raw_rerank"),
        ("Rewrite + Rerank", "rewrite_rerank"),
    )
    lines = [
        "# Query Rewrite + Rerank Evaluation",
        "",
        "| Arm | Hit@5 | Hit@20 | Recall@5 | MRR@5 | Required-doc Recall@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in names:
        metrics = scorecard["arms"][key]["metrics"]
        lines.append(
            f"| {label} | {_at(metrics, 'hit_at_k', 5):.4f} | "
            f"{_at(metrics, 'hit_at_k', 20):.4f} | "
            f"{_at(metrics, 'recall_at_k', 5):.4f} | "
            f"{_at(metrics, 'mrr_at_k', 5):.4f} | "
            f"{_at(metrics, 'all_documents_recall_at_k', 5):.4f} |"
        )
    final = scorecard["comparisons"]["final_at_5"]["counts"]
    lines.extend(
        [
            "",
            (
                f"Final paired Hit@5: improved={final['improved']}, "
                f"regressed={final['regressed']}, both_hit={final['both_hit']}, "
                f"both_miss={final['both_miss']}."
            ),
            "",
            (
                "Decision: **ADOPT Rewrite + Rerank**."
                if scorecard["decision"]["adopt_rewrite_rerank"]
                else "Decision: **DO NOT ADOPT Rewrite + Rerank**."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "dataset": args.dataset.resolve(),
        "raw_report": args.raw_report.resolve(),
        "rewrite_report": args.rewrite_report.resolve(),
        "rewrite_file": args.rewrite_file.resolve(),
        "db": args.db.resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.resolve()
    signature = _signature(paths)

    dataset = RAGGoldDataset.model_validate_json(
        paths["dataset"].read_text(encoding="utf-8")
    )
    cases = select_simple_rag_cases(dataset)
    expected_ids = {str(case.case_id) for case in cases}
    raw_report = json.loads(paths["raw_report"].read_text(encoding="utf-8"))
    rewrite_report = json.loads(paths["rewrite_report"].read_text(encoding="utf-8"))
    rewrite_file = json.loads(paths["rewrite_file"].read_text(encoding="utf-8"))
    raw_fusion_rows = extract_raw_fusion_cases(raw_report)
    raw_rerank_rows = extract_raw_rerank_cases(raw_report)
    rewrite_fusion_rows = _extract_rewrite_fusion_cases(rewrite_report)
    rewrites_by_id = _extract_rewrites(rewrite_file)
    for name, rows in (
        ("raw fusion", raw_fusion_rows),
        ("raw rerank", raw_rerank_rows),
        ("rewrite fusion", rewrite_fusion_rows),
        ("rewrite file", rewrites_by_id),
    ):
        if set(rows) != expected_ids:
            raise ValueError(f"{name} does not contain exactly the 70 Simple-RAG cases")

    chunks = _load_chunks(paths["db"])
    rerank_rows_list = _run_rewrite_rerank(
        cases=cases,
        rewrite_fusion_rows=rewrite_fusion_rows,
        rewrites_by_id=rewrites_by_id,
        chunks_by_id=chunks,
        output_path=output_dir / "rewrite-rerank-results.json",
        signature=signature,
        resume=args.resume,
    )
    rewrite_rerank_rows = {
        str(row["case_id"]): row for row in rerank_rows_list
    }
    metadata = {
        chunk_id: (chunk.document_id, chunk.page_number)
        for chunk_id, chunk in chunks.items()
    }
    gold_by_case = {
        str(case_id): [int(value) for value in values]
        for case_id, values in (raw_report.get("gold_chunks") or {}).items()
    }
    gold_by_case.update(AUDITED_GOLD_OVERRIDES)
    document_ids = resolve_frozen_document_ids(dataset, paths["db"])
    evaluation_cases = _evaluation_cases(
        cases,
        gold_by_case=gold_by_case,
        document_ids=document_ids,
        chunk_metadata=metadata,
    )
    arm_rows = {
        "raw_fusion": raw_fusion_rows,
        "rewrite_fusion": rewrite_fusion_rows,
        "raw_rerank": raw_rerank_rows,
        "rewrite_rerank": rewrite_rerank_rows,
    }
    arms = {
        name: _run_results(rows, chunk_metadata=metadata)
        for name, rows in arm_rows.items()
    }
    rerank_error_count = sum(bool(row.get("error")) for row in rerank_rows_list)
    scorecard = score_four_arms(
        evaluation_cases,
        arms,
        gold_by_case=gold_by_case,
        rerank_error_count=rerank_error_count,
    )
    scorecard.update(
        {
            "evaluation": "query-rewrite-rerank-v1",
            "signature": signature,
            "case_count": len(cases),
            "answerable_case_count": len(gold_by_case),
            "rewrite_rerank_error_count": rerank_error_count,
            "rewrite_rerank_degraded_count": sum(
                bool(row.get("degraded")) for row in rerank_rows_list
            ),
            "rerank_latency_ms": {
                "raw_frozen": _raw_rerank_latency(raw_rerank_rows),
                "rewrite_new": asdict(
                    evaluate_retrieval_results(
                        evaluation_cases,
                        arms["rewrite_rerank"],
                        top_ks=(5,),
                    ).latency_ms
                ),
            },
            "gold_audit_overrides": AUDITED_GOLD_NOTES,
        }
    )
    _write_json_atomic(output_dir / "rerank-scorecard.json", scorecard)
    _write_text_atomic(output_dir / "rerank-scorecard.md", _markdown(scorecard))

    comparisons = scorecard["comparisons"]
    comparison_by_name = {
        name: {row["case_id"]: row for row in comparison["cases"]}
        for name, comparison in comparisons.items()
    }
    case_lines: list[str] = []
    for case in cases:
        case_id = str(case.case_id)
        case_lines.append(
            json.dumps(
                {
                    "case_id": case_id,
                    "question": str(case.question),
                    "rewrite_query": rewrites_by_id[case_id]["english_query"],
                    "arms_top20": {
                        name: [
                            int(item["chunk_id"])
                            for item in rows[case_id].get("ranked_chunks") or []
                        ]
                        for name, rows in arm_rows.items()
                    },
                    "comparisons": {
                        name: rows[case_id]
                        for name, rows in comparison_by_name.items()
                    },
                    "rewrite_rerank_latency_ms": rewrite_rerank_rows[case_id].get(
                        "latency_ms"
                    ),
                    "rewrite_rerank_error": rewrite_rerank_rows[case_id].get("error"),
                },
                ensure_ascii=False,
            )
        )
    _write_text_atomic(output_dir / "rerank-cases.jsonl", "\n".join(case_lines) + "\n")
    _write_json_atomic(
        output_dir / "manifest.json",
        {
            "evaluation": "query-rewrite-rerank-v1",
            "signature": signature,
            "inputs": {name: str(path) for name, path in paths.items()},
            "reranker": "BAAI/bge-reranker-base",
            "rewrite_source": rewrite_file.get("source"),
            "chroma_initialized": False,
            "router_called": False,
            "agent_called": False,
            "generation_called": False,
            "writes_confined_to_d_drive": output_dir.drive.casefold() == "d:",
        },
    )
    return scorecard


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rerank frozen GPT-5.5 Rewrite Fusion candidates and score four arms"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--rewrite-report", type=Path, required=True)
    parser.add_argument("--rewrite-file", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    scorecard = run(_parser().parse_args(argv))
    print(json.dumps(scorecard["decision"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
