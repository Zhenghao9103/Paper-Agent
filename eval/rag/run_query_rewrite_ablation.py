"""Run the Raw-vs-Rewrite ablation at the pre-rerank Hybrid fusion stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.rag import vector_store
from backend.app.services.model_clients import answer_json
from .query_rewrite_ablation import (
    adoption_decision,
    build_query_plan,
    build_rewrite_messages,
    extract_raw_fusion_cases,
    paired_hit_outcomes,
    parse_rewrite_payload,
    retrieve_fusion,
)
from .rag_gold import RAGGoldDataset
from .rag_quality import (
    RankedChunk,
    RetrievalEvalCase,
    RetrievalRunResult,
    evaluate_retrieval_results,
)
from .run_rag_component_eval import (
    AUDITED_GOLD_NOTES,
    AUDITED_GOLD_OVERRIDES,
    resolve_frozen_document_ids,
    select_simple_rag_cases,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(
    *,
    dataset_path: Path,
    raw_report_path: Path,
    db_path: Path,
    chroma_path: Path,
    offline_rewrites_path: Path | None = None,
) -> str:
    bin_files = sorted(chroma_path.rglob("*.bin"))
    payload = {
        "dataset_sha256": _sha256_file(dataset_path),
        "raw_report_sha256": _sha256_file(raw_report_path),
        "db_sha256": _sha256_file(db_path),
        "chroma_sqlite_sha256": _sha256_file(chroma_path / "chroma.sqlite3"),
        "chroma_bin_sha256": [
            [str(path.relative_to(chroma_path)), _sha256_file(path)] for path in bin_files
        ],
        "contract": "raw-vs-one-shot-rewrite-hybrid-fusion-v1",
        "top_k": 20,
        "rerank": False,
        "metadata_filter": False,
        "offline_rewrites_sha256": (
            _sha256_file(offline_rewrites_path) if offline_rewrites_path else None
        ),
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


def _load_resume(
    path: Path, signature: str, *, retry_errors: bool = False
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("signature") != signature:
        raise ValueError(f"resume signature mismatch: {path}")
    rows = list(payload.get("cases") or [])
    if retry_errors:
        rows = [row for row in rows if not row.get("error")]
    return rows


def _persist_cases(
    path: Path, *, signature: str, cases: Sequence[Mapping[str, Any]]
) -> None:
    _write_json_atomic(path, {"signature": signature, "cases": list(cases)})


def _generate_rewrites(
    cases: Sequence[Any], *, path: Path, signature: str, resume: bool
) -> list[dict[str, Any]]:
    rows = _load_resume(path, signature, retry_errors=True) if resume else []
    completed = {str(row["case_id"]) for row in rows}
    consecutive_errors = 0
    for case in cases:
        case_id = str(case.case_id)
        if case_id in completed:
            continue
        import time

        started = time.perf_counter()
        payload = answer_json(build_rewrite_messages(str(case.question)))
        latency_ms = (time.perf_counter() - started) * 1000
        error = None
        rewrite = None
        try:
            if payload is None:
                raise ValueError("rewrite provider returned no JSON object")
            parsed = parse_rewrite_payload(str(case.question), payload)
            rewrite = {
                "english_query": parsed.english_query,
                "lexical_terms": list(parsed.lexical_terms),
            }
        except Exception as exc:  # noqa: BLE001 - stable per-case error artifact
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "case_id": case_id,
                "question": str(case.question),
                "rewrite": rewrite,
                "latency_ms": latency_ms,
                "error": error,
            }
        )
        _persist_cases(path, signature=signature, cases=rows)
        if error:
            consecutive_errors += 1
            if consecutive_errors >= 3:
                raise RuntimeError(
                    "rewrite provider failed for 3 consecutive cases; aborting early"
                )
        else:
            consecutive_errors = 0
    return rows


def _load_offline_rewrites(
    cases: Sequence[Any], *, path: Path
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_rows = payload.get("cases") if isinstance(payload, Mapping) else None
    if not isinstance(source_rows, list):
        raise ValueError("offline rewrite file must contain a cases list")
    by_id = {str(row.get("case_id")): row for row in source_rows}
    expected = {str(case.case_id) for case in cases}
    if set(by_id) != expected:
        raise ValueError("offline rewrite file must contain exactly the 70 Simple-RAG cases")
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.case_id)
        source = by_id[case_id]
        rewrite = parse_rewrite_payload(str(case.question), source)
        rows.append(
            {
                "case_id": case_id,
                "question": str(case.question),
                "rewrite": {
                    "english_query": rewrite.english_query,
                    "lexical_terms": list(rewrite.lexical_terms),
                },
                "latency_ms": 0.0,
                "error": None,
                "source": str(payload.get("source") or "offline"),
            }
        )
    return rows


@contextmanager
def _read_only_session(db_path: Path):
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    engine = create_engine(
        "sqlite://",
        creator=lambda: sqlite3.connect(uri, uri=True),
    )
    with Session(engine) as session:
        yield session
    engine.dispose()


@contextmanager
def _chroma_runtime(chroma_path: Path):
    original = vector_store.chroma_dir
    vector_store._persistent_client.cache_clear()
    vector_store.chroma_dir = lambda: chroma_path.resolve()
    try:
        yield
    finally:
        vector_store._persistent_client.cache_clear()
        vector_store.chroma_dir = original


def _run_rewrite_retrieval(
    cases: Sequence[Any],
    rewrites: Sequence[Mapping[str, Any]],
    *,
    db_path: Path,
    chroma_path: Path,
    path: Path,
    signature: str,
    resume: bool,
    limit: int = 20,
) -> list[dict[str, Any]]:
    rows = _load_resume(path, signature, retry_errors=True) if resume else []
    completed = {str(row["case_id"]) for row in rows}
    rewrite_by_id = {str(row["case_id"]): row for row in rewrites}
    with _chroma_runtime(chroma_path), _read_only_session(db_path) as db:
        count = vector_store.get_paper_chunks_collection().count()
        if count != 1371:
            raise ValueError(f"expected 1371 persisted vectors, found {count}")
        for case in cases:
            case_id = str(case.case_id)
            if case_id in completed:
                continue
            rewrite_row = rewrite_by_id[case_id]
            if rewrite_row.get("error"):
                result = {
                    "latency_ms": 0.0,
                    "error": "rewrite unavailable",
                    "degraded": None,
                    "rerank_applied": False,
                    "ranked_chunks": [],
                    "diagnostics": {},
                }
            else:
                rewrite = parse_rewrite_payload(
                    str(case.question), rewrite_row["rewrite"]
                )
                try:
                    result = retrieve_fusion(
                        db,
                        build_query_plan(str(case.question), rewrite),
                        limit=limit,
                    )
                except Exception as exc:  # noqa: BLE001 - stable per-case error artifact
                    result = {
                        "latency_ms": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                        "degraded": None,
                        "rerank_applied": False,
                        "ranked_chunks": [],
                        "diagnostics": {},
                    }
            rows.append(
                {
                    "case_id": case_id,
                    "question": str(case.question),
                    "rewrite": rewrite_row.get("rewrite"),
                    **result,
                }
            )
            _persist_cases(path, signature=signature, cases=rows)
    return rows


def _chunk_metadata(db_path: Path) -> dict[int, tuple[int, int]]:
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            "SELECT id, document_id, page_number FROM document_chunks"
        ).fetchall()
    return {int(row[0]): (int(row[1]), int(row[2])) for row in rows}


def _evaluation_cases(
    cases: Sequence[Any],
    *,
    gold_by_case: Mapping[str, Sequence[int]],
    document_ids: Mapping[str, int],
    chunk_metadata: Mapping[int, tuple[int, int]],
) -> list[RetrievalEvalCase]:
    output: list[RetrievalEvalCase] = []
    for case in cases:
        case_id = str(case.case_id)
        if case.expect_insufficient_evidence:
            output.append(
                RetrievalEvalCase(case_id=case_id, expect_insufficient_evidence=True)
            )
            continue
        gold = [int(value) for value in gold_by_case[case_id]]
        output.append(
            RetrievalEvalCase(
                case_id=case_id,
                relevance_by_chunk={chunk_id: 1 for chunk_id in gold},
                chunk_document_ids={
                    chunk_id: chunk_metadata[chunk_id][0] for chunk_id in gold
                },
                required_document_ids={
                    int(document_ids[key]) for key in case.required_documents
                },
            )
        )
    return output


def _run_results(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    chunk_metadata: Mapping[int, tuple[int, int]],
) -> list[RetrievalRunResult]:
    output: list[RetrievalRunResult] = []
    for case_id, row in rows.items():
        ranked: list[RankedChunk] = []
        for rank, item in enumerate(row.get("ranked_chunks") or [], start=1):
            chunk_id = int(item["chunk_id"])
            document_id, page_number = chunk_metadata[chunk_id]
            ranked.append(
                RankedChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    page_number=page_number,
                    rank=rank,
                    score=float(item.get("score") or 0.0),
                )
            )
        output.append(
            RetrievalRunResult(
                case_id=case_id,
                ranked_chunks=ranked,
                latency_ms=float(row.get("latency_ms") or 0.0),
                error=row.get("error"),
                degraded=row.get("degraded"),
            )
        )
    return output


def _rankings(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, list[int]]:
    return {
        case_id: [int(item["chunk_id"]) for item in row.get("ranked_chunks") or []]
        for case_id, row in rows.items()
    }


def _metric(metrics: Mapping[str, Any], name: str, k: int) -> float:
    values = metrics[name]
    return float(values.get(k, values.get(str(k), 0.0)))


def _markdown(scorecard: Mapping[str, Any]) -> str:
    raw = scorecard["raw_fusion"]["metrics"]
    rewritten = scorecard["rewrite_fusion"]["metrics"]
    paired = scorecard["paired_at_5"]["counts"]
    decision = scorecard["decision"]
    return "\n".join(
        [
            "# Query Rewrite Ablation",
            "",
            "Reranker: **OFF**. Metadata filter: **OFF**. Retrieval: BM25 + Vector + RRF.",
            "",
            "| Metric | Raw | Rewrite | Delta |",
            "|---|---:|---:|---:|",
            *[
                (
                    f"| {name}@{k} | {_metric(raw, key, k):.4f} | "
                    f"{_metric(rewritten, key, k):.4f} | "
                    f"{_metric(rewritten, key, k) - _metric(raw, key, k):+.4f} |"
                )
                for name, key, k in (
                    ("Hit", "hit_at_k", 5),
                    ("Hit", "hit_at_k", 20),
                    ("Strict Recall", "recall_at_k", 5),
                    ("Strict Recall", "recall_at_k", 20),
                    ("MRR", "mrr_at_k", 5),
                    ("MRR", "mrr_at_k", 20),
                    ("Required-document Recall", "all_documents_recall_at_k", 5),
                    ("Required-document Recall", "all_documents_recall_at_k", 20),
                )
            ],
            "",
            (
                f"Paired Hit@5: improved={paired['improved']}, "
                f"regressed={paired['regressed']}, both_hit={paired['both_hit']}, "
                f"both_miss={paired['both_miss']}."
            ),
            "",
            f"Decision: **{'ADOPT' if decision['adopt'] else 'REJECT'}**.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = args.dataset.resolve()
    raw_report_path = args.raw_report.resolve()
    db_path = args.db.resolve()
    chroma_path = args.chroma_dir.resolve()
    output_dir = args.output_dir.resolve()
    offline_rewrites_path = (
        args.offline_rewrites.resolve() if args.offline_rewrites else None
    )
    for path in (dataset_path, raw_report_path, db_path, chroma_path / "chroma.sqlite3"):
        if not path.exists():
            raise FileNotFoundError(path)
    if not list(chroma_path.rglob("header.bin")):
        raise ValueError("persisted Chroma index has no header.bin")

    signature = _signature(
        dataset_path=dataset_path,
        raw_report_path=raw_report_path,
        db_path=db_path,
        chroma_path=chroma_path,
        offline_rewrites_path=offline_rewrites_path,
    )
    dataset = RAGGoldDataset.model_validate_json(dataset_path.read_text(encoding="utf-8"))
    cases = select_simple_rag_cases(dataset)
    raw_report = json.loads(raw_report_path.read_text(encoding="utf-8"))
    raw_rows = extract_raw_fusion_cases(raw_report)
    expected_ids = {str(case.case_id) for case in cases}
    if set(raw_rows) != expected_ids:
        raise ValueError("Raw fusion report does not contain exactly the 70 Simple-RAG cases")

    rewrite_output_path = output_dir / "query-rewrites.json"
    if offline_rewrites_path is not None:
        rewrites = _load_offline_rewrites(cases, path=offline_rewrites_path)
        _persist_cases(rewrite_output_path, signature=signature, cases=rewrites)
    else:
        rewrites = _generate_rewrites(
            cases,
            path=rewrite_output_path,
            signature=signature,
            resume=args.resume,
        )
    rewrite_rows_list = _run_rewrite_retrieval(
        cases,
        rewrites,
        db_path=db_path,
        chroma_path=chroma_path,
        path=output_dir / "rewrite-retrieval.json",
        signature=signature,
        resume=args.resume,
    )
    rewrite_rows = {str(row["case_id"]): row for row in rewrite_rows_list}

    metadata = _chunk_metadata(db_path)
    gold_by_case = {
        str(case_id): [int(value) for value in values]
        for case_id, values in (raw_report.get("gold_chunks") or {}).items()
    }
    gold_by_case.update(AUDITED_GOLD_OVERRIDES)
    document_ids = resolve_frozen_document_ids(dataset, db_path)
    eval_cases = _evaluation_cases(
        cases,
        gold_by_case=gold_by_case,
        document_ids=document_ids,
        chunk_metadata=metadata,
    )
    raw_metrics = asdict(
        evaluate_retrieval_results(
            eval_cases,
            _run_results(raw_rows, chunk_metadata=metadata),
            top_ks=(3, 5, 10, 20),
        )
    )
    rewrite_metrics = asdict(
        evaluate_retrieval_results(
            eval_cases,
            _run_results(rewrite_rows, chunk_metadata=metadata),
            top_ks=(3, 5, 10, 20),
        )
    )
    raw_rankings = _rankings(raw_rows)
    rewrite_rankings = _rankings(rewrite_rows)
    paired5 = paired_hit_outcomes(raw_rankings, rewrite_rankings, gold_by_case, k=5)
    paired20 = paired_hit_outcomes(raw_rankings, rewrite_rankings, gold_by_case, k=20)
    rewrite_error_count = sum(bool(row.get("error")) for row in rewrites)
    retrieval_error_count = sum(bool(row.get("error")) for row in rewrite_rows_list)
    decision = adoption_decision(
        improved=paired5["counts"]["improved"],
        regressed=paired5["counts"]["regressed"],
        raw_hit20=_metric(raw_metrics, "hit_at_k", 20),
        rewrite_hit20=_metric(rewrite_metrics, "hit_at_k", 20),
        error_count=rewrite_error_count + retrieval_error_count,
    )
    scorecard = {
        "evaluation": "query-rewrite-ablation-v1",
        "signature": signature,
        "case_count": len(cases),
        "answerable_case_count": len(gold_by_case),
        "retrieval": "BM25 + Vector + RRF fusion",
        "rerank_applied": False,
        "metadata_filter_applied": False,
        "raw_fusion": {"metrics": raw_metrics},
        "rewrite_fusion": {"metrics": rewrite_metrics},
        "paired_at_5": paired5,
        "paired_at_20": paired20,
        "rewrite_error_count": rewrite_error_count,
        "retrieval_error_count": retrieval_error_count,
        "decision": decision,
        "gold_audit_overrides": AUDITED_GOLD_NOTES,
    }
    _write_json_atomic(output_dir / "query-ablation-scorecard.json", scorecard)
    _write_text_atomic(output_dir / "query-ablation-scorecard.md", _markdown(scorecard))

    paired5_by_id = {row["case_id"]: row for row in paired5["cases"]}
    paired20_by_id = {row["case_id"]: row for row in paired20["cases"]}
    rewrite_by_id = {str(row["case_id"]): row for row in rewrites}
    case_lines = []
    for case in cases:
        case_id = str(case.case_id)
        row = {
            "case_id": case_id,
            "question": str(case.question),
            "rewrite": rewrite_by_id[case_id].get("rewrite"),
            "rewrite_error": rewrite_by_id[case_id].get("error"),
            "retrieval_error": rewrite_rows[case_id].get("error"),
            "rewrite_latency_ms": rewrite_by_id[case_id].get("latency_ms"),
            "raw_retrieval_latency_ms": raw_rows[case_id].get("latency_ms"),
            "rewrite_retrieval_latency_ms": rewrite_rows[case_id].get("latency_ms"),
            "raw_top20": raw_rankings[case_id],
            "rewrite_top20": rewrite_rankings[case_id],
            "paired_at_5": paired5_by_id.get(case_id),
            "paired_at_20": paired20_by_id.get(case_id),
        }
        case_lines.append(json.dumps(row, ensure_ascii=False))
    _write_text_atomic(
        output_dir / "query-ablation-cases.jsonl", "\n".join(case_lines) + "\n"
    )
    _write_json_atomic(
        output_dir / "manifest.json",
        {
            "evaluation": "query-rewrite-ablation-v1",
            "signature": signature,
            "dataset": str(dataset_path),
            "raw_report": str(raw_report_path),
            "db": str(db_path),
            "chroma_dir": str(chroma_path),
            "chroma_count": 1371,
            "agent_model": get_settings().resolved_agent_model,
            "rewrite_source": (
                str(rewrites[0].get("source") or "offline")
                if offline_rewrites_path
                else "agent-api"
            ),
            "offline_rewrites": (
                str(offline_rewrites_path) if offline_rewrites_path else None
            ),
            "rerank_applied": False,
            "metadata_filter_applied": False,
            "writes_confined_to_d_drive": all(
                path.drive.casefold() == "d:" for path in (output_dir,)
            ),
        },
    )
    return scorecard


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Raw and one-shot rewritten queries before reranking"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--chroma-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offline-rewrites", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    scorecard = run(_parser().parse_args(argv))
    print(json.dumps(scorecard["decision"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
