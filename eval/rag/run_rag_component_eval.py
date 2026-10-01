"""Offline Retrieval and Generation component evaluation for frozen Hybrid results."""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def build_component_signature(
    *,
    dataset_sha256: str,
    retrieval_sha256: str,
    top_k: int,
    model: str,
) -> str:
    payload = json.dumps(
        {
            "dataset_sha256": dataset_sha256,
            "retrieval_sha256": retrieval_sha256,
            "top_k": int(top_k),
            "model": model,
            "prompt_contract": "evidence-bound-answer-v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_generation_model(
    *, configured_model: str | None, process_model: str | None
) -> str | None:
    override = process_model.strip() if process_model else ""
    return override or configured_model


def select_simple_rag_cases(dataset: Any, case_ids: Sequence[str] | None = None) -> list[Any]:
    eligible = [
        case
        for case in dataset.cases
        if case.task_type in {"single_paper", "insufficient_evidence"}
    ]
    if case_ids is None:
        return eligible
    by_id = {str(case.case_id): case for case in eligible}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise ValueError(f"unknown or non-Simple-RAG case ID: {missing}")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be unique")
    return [by_id[case_id] for case_id in case_ids]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("embedding vectors must be non-empty and equal length")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def response_relevancy_from_embeddings(
    question_embedding: Sequence[float],
    reconstructed_question_embeddings: Sequence[Sequence[float]],
) -> float:
    if len(reconstructed_question_embeddings) != 3:
        raise ValueError("response relevancy requires exactly three reconstructed questions")
    return (
        sum(
            _cosine(question_embedding, embedding)
            for embedding in reconstructed_question_embeddings
        )
        / 3
    )


def validate_hybrid_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    retrieval = payload.get("retrieval")
    if not isinstance(retrieval, Mapping) or "hybrid" not in retrieval:
        raise ValueError("retrieval report must contain hybrid results")
    return {
        "source_report_mode": "hybrid",
        "channels": ["bm25", "vector"],
        "fusion": "production_hybrid_search",
        "rerank": True,
    }


def materialize_ranked_contexts(
    retrieval: Mapping[str, Any],
    *,
    chunks_by_id: Mapping[int, Any],
    titles_by_document_id: Mapping[int, str],
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    hybrid = retrieval.get("hybrid")
    if not isinstance(hybrid, Mapping):
        raise ValueError("retrieval report must contain hybrid results")
    output: dict[str, list[dict[str, Any]]] = {}
    for case in hybrid.get("cases", []):
        case_id = str(case["case_id"])
        contexts: list[dict[str, Any]] = []
        for ranked in case.get("ranked_chunks", [])[:top_k]:
            chunk_id = int(ranked["chunk_id"])
            if chunk_id not in chunks_by_id:
                raise ValueError(f"retrieval chunk is missing from SQLite: {chunk_id}")
            chunk = chunks_by_id[chunk_id]
            document_id = int(chunk.document_id)
            contexts.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "title": titles_by_document_id.get(document_id, "Unknown document"),
                    "page_number": int(chunk.page_number),
                    "chunk_index": int(chunk.chunk_index),
                    "content": str(chunk.content),
                    "fusion_score": float(ranked.get("score", 0.0)),
                }
            )
        output[case_id] = contexts
    return output


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _average_precision(flags: Sequence[bool]) -> float:
    relevant = 0
    precision_sum = 0.0
    for rank, flag in enumerate(flags, 1):
        if not flag:
            continue
        relevant += 1
        precision_sum += relevant / rank
    return _ratio(precision_sum, relevant)


def score_ragas_judgment(judgment: Mapping[str, Any]) -> dict[str, Any]:
    context_flags = [bool(value) for value in judgment.get("context_relevance", [])]
    answer_claims = list(judgment.get("answer_claims", []))
    reference_claims = list(judgment.get("reference_claims", []))
    context_supported = sum(bool(row.get("context_supported")) for row in answer_claims)
    reference_supported = sum(bool(row.get("reference_supported")) for row in answer_claims)
    covered = sum(bool(row.get("answer_covered")) for row in reference_claims)
    context_covered = sum(bool(row.get("context_supported")) for row in reference_claims)
    factual_precision = _ratio(reference_supported, len(answer_claims))
    factual_recall = _ratio(covered, len(reference_claims))
    factual_f1 = (
        2 * factual_precision * factual_recall / (factual_precision + factual_recall)
        if factual_precision + factual_recall
        else 0.0
    )
    return {
        "context_precision": _average_precision(context_flags),
        "context_recall": _ratio(context_covered, len(reference_claims)),
        "faithfulness": _ratio(context_supported, len(answer_claims)),
        "response_relevancy": float(judgment.get("response_relevancy", 0.0)),
        "factual_precision": factual_precision,
        "factual_recall": factual_recall,
        "factual_correctness_f1": factual_f1,
        "citation_validity": bool(judgment.get("citation_validity")),
        "abstention_correct": judgment.get("abstention_correct"),
        "fabrication": bool(judgment.get("fabrication")),
    }


def _metric_at_k(metrics: Mapping[str, Any], name: str, top_k: int) -> float:
    values = metrics.get(name, {})
    if not isinstance(values, Mapping):
        return 0.0
    return float(values.get(str(top_k), values.get(top_k, 0.0)))


def build_retrieval_scorecard(report: Mapping[str, Any], *, top_k: int) -> dict[str, Any]:
    lineage = validate_hybrid_report(report)
    hybrid = report["retrieval"]["hybrid"]
    metrics = hybrid.get("metrics", {})
    recall = _metric_at_k(metrics, "recall_at_k", top_k)
    hit_rate = _metric_at_k(metrics, "hit_at_k", top_k)
    error_rate = float(metrics.get("error_rate", 0.0))
    degraded_rate = float(metrics.get("degraded_rate", 0.0))
    return {
        "dataset": report.get("dataset"),
        "version": report.get("version"),
        "top_k": top_k,
        "hybrid_pipeline": lineage,
        "metrics": {
            "case_count": int(metrics.get("case_count", 0)),
            "recall_at_k": recall,
            "strict_chunk_recall_at_k": recall,
            "hit_at_k": hit_rate,
            "mrr_at_k": _metric_at_k(metrics, "mrr_at_k", top_k),
            "map_at_k": _metric_at_k(metrics, "map_at_k", top_k),
            "ndcg_at_k": _metric_at_k(metrics, "ndcg_at_k", top_k),
            "required_document_recall_at_k": _metric_at_k(
                metrics, "all_documents_recall_at_k", top_k
            ),
            "latency_ms": dict(metrics.get("latency_ms", {})),
            "error_rate": error_rate,
            "degraded_rate": degraded_rate,
            "context_precision": None,
            "context_recall": None,
        },
        "mapping_errors": list(report.get("mapping_errors", [])),
        "gates": {
            "strict_chunk_recall_at_k": {
                "status": "diagnostic",
                "value": recall,
            },
            "hit_at_k": {
                "threshold": 0.80,
                "value": hit_rate,
                "passed": hit_rate >= 0.80,
            },
            "error_rate": {"threshold": 0.0, "value": error_rate, "passed": error_rate == 0.0},
            "degraded_rate": {
                "threshold": 0.0,
                "value": degraded_rate,
                "passed": degraded_rate == 0.0,
            },
            "ragas_context_metrics": {"status": "pending_offline_judgment"},
        },
    }


def citations_are_valid(
    generation: Mapping[str, Any], answer_claims: Sequence[Mapping[str, Any]]
) -> bool:
    context_ids = {int(item["chunk_id"]) for item in generation.get("contexts", [])}
    try:
        citation_ids = {int(item["chunk_id"]) for item in generation.get("citations", [])}
    except (KeyError, TypeError, ValueError):
        return False
    if not citation_ids.issubset(context_ids):
        return False
    assessed_ids: set[int] = set()
    for claim in answer_claims:
        claim_ids = {int(value) for value in claim.get("citation_chunk_ids", [])}
        if not claim_ids.issubset(citation_ids):
            return False
        if claim_ids and claim.get("citation_supported") is not True:
            return False
        assessed_ids.update(claim_ids)
    if assessed_ids != citation_ids:
        return False
    return True


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_generation_scores(
    rows: Sequence[Mapping[str, Any]], *, expected_count: int
) -> dict[str, Any]:
    judged = [row for row in rows if isinstance(row.get("metrics"), Mapping)]
    metric_names = (
        "faithfulness",
        "response_relevancy",
        "factual_correctness_f1",
        "context_precision",
        "context_recall",
    )
    output = {
        "case_count": len(rows),
        "expected_count": expected_count,
        "judge_coverage": len(judged) / expected_count if expected_count else 0.0,
        **{
            name: _mean([float(row["metrics"].get(name, 0.0)) for row in judged])
            for name in metric_names
        },
        "citation_validity_rate": _mean(
            [1.0 if row["metrics"].get("citation_validity") else 0.0 for row in judged]
        ),
        "abstention_case_count": sum(
            bool(row.get("expect_insufficient_evidence")) for row in judged
        ),
        "abstention_pass_count": sum(
            bool(row.get("expect_insufficient_evidence"))
            and row["metrics"].get("abstention_correct") is True
            for row in judged
        ),
        "fabrication_count": sum(row["metrics"].get("fabrication") is True for row in judged),
        "generation_error_count": sum(bool(row.get("error")) for row in rows),
    }
    output["gates"] = {
        "faithfulness": output["faithfulness"] >= 0.90,
        "response_relevancy": output["response_relevancy"] >= 0.80,
        "factual_correctness_f1": output["factual_correctness_f1"] >= 0.80,
        "citation_validity": output["citation_validity_rate"] == 1.0,
        "abstention": output["abstention_pass_count"] == output["abstention_case_count"],
        "fabrication": output["fabrication_count"] == 0,
        "generation_errors": output["generation_error_count"] == 0,
    }
    return output


def evaluate_generation_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    effective = 0
    insufficient_pass = 0
    insufficient_total = 0
    for row in rows:
        case_id = str(row.get("case_id"))
        metrics = row.get("metrics") or {}
        passed = (
            not row.get("error")
            and float(metrics.get("faithfulness", 0.0)) >= 0.90
            and float(metrics.get("response_relevancy", 0.0)) >= 0.80
            and float(metrics.get("factual_correctness_f1", 0.0)) >= 0.80
            and metrics.get("citation_validity") is True
            and metrics.get("fabrication") is False
        )
        if row.get("expect_insufficient_evidence"):
            insufficient_total += 1
            passed = passed and metrics.get("abstention_correct") is True
            if passed:
                insufficient_pass += 1
        if passed:
            effective += 1
        else:
            failures.append(case_id)
    passed = (
        len(rows) == 10 and effective >= 8 and insufficient_total == 2 and insufficient_pass == 2
    )
    return {
        "passed": passed,
        "case_count": len(rows),
        "effective_pass_count": effective,
        "insufficient_pass_count": insufficient_pass,
        "failures": failures,
    }


# Historic chunk IDs depend on a particular local ingestion run. Fresh runs
# never apply those IDs to a newly created corpus.
AUDITED_GOLD_OVERRIDES: dict[str, list[int]] = {}
AUDITED_GOLD_NOTES: dict[str, Any] = {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    value = "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)
    _write_text_atomic(path, value)


def recompute_audited_recall(
    report: Mapping[str, Any],
    *,
    top_ks: Sequence[int],
    gold_overrides: Mapping[str, Sequence[int]] | None = None,
    required_document_ids_by_case: Mapping[str, Sequence[int]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Recompute strict chunk recall and Hit@K from frozen audited ranks."""

    gold_by_case = {
        str(case_id): [int(value) for value in values]
        for case_id, values in (report.get("gold_chunks") or {}).items()
    }
    for case_id, values in (gold_overrides or {}).items():
        gold_by_case[str(case_id)] = [int(value) for value in values]

    hybrid = report.get("retrieval", {}).get("hybrid", {})
    cases = {
        str(case["case_id"]): case
        for case in hybrid.get("cases", [])
        if isinstance(case, Mapping) and case.get("case_id")
    }
    missing_cases = sorted(set(gold_by_case) - set(cases))
    if missing_cases:
        raise ValueError(f"Gold cases are absent from Hybrid report: {missing_cases}")

    recalls: dict[str, float] = {}
    hits: dict[str, float] = {}
    document_recalls: dict[str, float] = {}
    for top_k in top_ks:
        per_case: list[float] = []
        per_case_hit: list[float] = []
        per_case_document_recall: list[float] = []
        for case_id, gold_ids in gold_by_case.items():
            ranked = cases[case_id].get("ranked_chunks", [])[:top_k]
            ranked_ids = {int(row["chunk_id"]) for row in ranked}
            gold_hits = ranked_ids & set(gold_ids)
            per_case.append(_ratio(len(gold_hits), len(gold_ids)))
            per_case_hit.append(float(bool(gold_hits)))
            if required_document_ids_by_case is not None:
                required_ids = {
                    int(value)
                    for value in required_document_ids_by_case.get(case_id, [])
                }
                if required_ids:
                    ranked_document_ids = {
                        int(row["document_id"])
                        for row in ranked
                        if row.get("document_id") is not None
                    }
                    per_case_document_recall.append(
                        _ratio(len(ranked_document_ids & required_ids), len(required_ids))
                    )
        recalls[str(top_k)] = _mean(per_case)
        hits[str(top_k)] = _mean(per_case_hit)
        if required_document_ids_by_case is not None:
            document_recalls[str(top_k)] = _mean(per_case_document_recall)

    failures: list[dict[str, Any]] = []
    for case_id, gold_ids in gold_by_case.items():
        ranked = [int(row["chunk_id"]) for row in cases[case_id].get("ranked_chunks", [])]
        top5_hits = set(ranked[:5]) & set(gold_ids)
        top20_hits = set(ranked[:20]) & set(gold_ids)
        if top5_hits:
            continue
        failures.append(
            {
                "case_id": case_id,
                "failure_type": ("ranking_miss_at_5" if top20_hits else "retrieval_miss_at_20"),
                "gold_chunk_ids": gold_ids,
                "top_5_chunk_ids": ranked[:5],
                "first_gold_rank": next(
                    (rank for rank, chunk_id in enumerate(ranked, 1) if chunk_id in set(gold_ids)),
                    None,
                ),
            }
        )
    return (
        {
            "answerable_case_count": len(gold_by_case),
            "recall_at_k": recalls,
            "strict_chunk_recall_at_k": recalls,
            "hit_at_k": hits,
            "required_document_recall_at_k": document_recalls,
        },
        failures,
    )


def resolve_frozen_document_ids(dataset: Any, db_path: Path) -> dict[str, int]:
    """Map corpus keys to existing SQLite document IDs without modifying the DB."""

    resolved_path = db_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Evaluation database does not exist: {resolved_path}")
    uri = f"file:{resolved_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute("SELECT id, file_path FROM documents ORDER BY id").fetchall()

    mapping: dict[str, int] = {}
    claimed: set[int] = set()
    corpus = sorted(
        dataset.corpus,
        key=lambda document: (-len(document.file_prefix), document.key),
    )
    for paper in corpus:
        prefix = paper.file_prefix.casefold()
        matches = [
            (int(document_id), str(file_path))
            for document_id, file_path in rows
            if int(document_id) not in claimed
            and Path(str(file_path)).name.casefold().startswith(prefix)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Corpus key {paper.key!r} expected one frozen document, found {len(matches)}"
            )
        document_id, _ = matches[0]
        mapping[str(paper.key)] = document_id
        claimed.add(document_id)
    return mapping


def _load_dataset(path: Path) -> Any:
    from .rag_gold import RAGGoldDataset

    return RAGGoldDataset.model_validate_json(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_case_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not values or any(not value for value in values):
        raise ValueError("case ID file must contain non-empty lines")
    if len(values) != len(set(values)):
        raise ValueError("case ID file contains duplicates")
    return values


def _validate_frozen_inputs(dataset: Any, report: Mapping[str, Any]) -> list[Any]:
    cases = select_simple_rag_cases(dataset)
    expected_ids = {str(case.case_id) for case in cases}
    hybrid_rows = report.get("retrieval", {}).get("hybrid", {}).get("cases", [])
    actual_ids = {str(row.get("case_id")) for row in hybrid_rows if isinstance(row, Mapping)}
    if expected_ids != actual_ids:
        raise ValueError("Hybrid report case IDs do not match the 70-case Simple-RAG slice")
    if report.get("dataset") != dataset.name or report.get("version") != dataset.version:
        raise ValueError("dataset identity does not match retrieval report")
    validate_hybrid_report(report)
    return cases


def _retrieval_markdown(scorecard: Mapping[str, Any]) -> str:
    metrics = scorecard["metrics"]
    gates = scorecard["gates"]
    lines = [
        "# RAG Retrieval Component Scorecard",
        "",
        "Frozen production Hybrid ranks were reused. No vector query or embedding ran.",
        "",
        "| Metric | Value | Gate |",
        "| --- | ---: | --- |",
        f"| Strict Chunk Recall@5 | {metrics['strict_chunk_recall_at_k']:.4f} | diagnostic |",
        f"| Hit@5 | {metrics['hit_at_k']:.4f} | {'PASS' if gates['hit_at_k']['passed'] else 'FAIL'} |",
        f"| MRR@5 | {metrics['mrr_at_k']:.4f} | diagnostic |",
        f"| MAP@5 | {metrics['map_at_k']:.4f} | diagnostic |",
        f"| nDCG@5 | {metrics['ndcg_at_k']:.4f} | diagnostic |",
        f"| Required-document Recall@5 | {metrics['required_document_recall_at_k']:.4f} | diagnostic |",
        f"| Error rate | {metrics['error_rate']:.4f} | {'PASS' if gates['error_rate']['passed'] else 'FAIL'} |",
        f"| Degradation rate | {metrics['degraded_rate']:.4f} | {'PASS' if gates['degraded_rate']['passed'] else 'FAIL'} |",
        "",
        "Context Precision@5 and Context Recall@5 remain pending until offline judgments are supplied.",
        "",
        "Review any unresolved gold-to-chunk mappings before interpreting recall.",
        "",
    ]
    return "\n".join(lines)


def run_retrieval_stage(
    *,
    dataset_path: Path,
    retrieval_path: Path,
    db_path: Path,
    top_k: int,
    output_dir: Path,
) -> dict[str, Any]:
    dataset = _load_dataset(dataset_path)
    report = _load_json(retrieval_path)
    _validate_frozen_inputs(dataset, report)
    document_ids = resolve_frozen_document_ids(dataset, db_path)
    required_document_ids_by_case = {
        str(case.case_id): [document_ids[key] for key in case.required_documents]
        for case in select_simple_rag_cases(dataset)
        if not case.expect_insufficient_evidence
    }
    audited, failures = recompute_audited_recall(
        report,
        top_ks=(3, 5, 10, 20),
        required_document_ids_by_case=required_document_ids_by_case,
    )
    scorecard = build_retrieval_scorecard(report, top_k=top_k)
    source_recall = scorecard["metrics"]["recall_at_k"]
    audited_recall = float(audited["recall_at_k"][str(top_k)])
    scorecard["metrics"]["source_recall_at_k"] = source_recall
    scorecard["metrics"]["recall_at_k"] = audited_recall
    scorecard["metrics"]["strict_chunk_recall_at_k"] = audited_recall
    scorecard["metrics"]["audited_recall_at_k"] = audited["recall_at_k"]
    scorecard["metrics"]["audited_strict_chunk_recall_at_k"] = audited[
        "strict_chunk_recall_at_k"
    ]
    scorecard["metrics"]["hit_at_k"] = float(audited["hit_at_k"][str(top_k)])
    scorecard["metrics"]["audited_hit_at_k"] = audited["hit_at_k"]
    scorecard["metrics"]["required_document_recall_at_k"] = float(
        audited["required_document_recall_at_k"][str(top_k)]
    )
    scorecard["metrics"]["audited_required_document_recall_at_k"] = audited[
        "required_document_recall_at_k"
    ]
    scorecard["metrics"]["audited_answerable_case_count"] = audited["answerable_case_count"]
    scorecard["gold_audit_overrides"] = {}
    scorecard["resolved_mapping_errors"] = []
    scorecard["unresolved_mapping_errors"] = report.get("mapping_errors", [])
    scorecard["gates"]["strict_chunk_recall_at_k"] = {
        "status": "diagnostic",
        "value": audited_recall,
    }
    audited_hit = scorecard["metrics"]["hit_at_k"]
    scorecard["gates"]["hit_at_k"] = {
        "threshold": 0.80,
        "value": audited_hit,
        "passed": audited_hit >= 0.80,
    }

    _write_json_atomic(output_dir / "retrieval-scorecard.json", scorecard)
    _write_text_atomic(
        output_dir / "retrieval-scorecard.md",
        _retrieval_markdown(scorecard),
    )
    _write_jsonl_atomic(output_dir / "retrieval-failures.jsonl", failures)
    _update_manifest(
        output_dir,
        {
            "retrieval": {
                "status": "complete",
                "dataset_sha256": _sha256_file(dataset_path),
                "retrieval_report_sha256": _sha256_file(retrieval_path),
                "source_case_count": 70,
                "fresh_retrieval_executed": False,
                "chroma_initialized": False,
                "gold_audit_overrides": AUDITED_GOLD_NOTES,
            }
        },
    )
    return scorecard


def _open_sqlite_snapshot(
    db_path: Path,
) -> tuple[dict[int, Any], dict[int, str]]:
    from types import SimpleNamespace

    if not db_path.is_file():
        raise FileNotFoundError(db_path)
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        chunks = {
            int(row[0]): SimpleNamespace(
                id=int(row[0]),
                document_id=int(row[1]),
                page_number=int(row[2]),
                chunk_index=int(row[3]),
                content=str(row[4]),
            )
            for row in connection.execute(
                "SELECT id, document_id, page_number, chunk_index, content FROM document_chunks"
            )
        }
        titles = {
            int(row[0]): str(row[1])
            for row in connection.execute("SELECT id, title FROM documents")
        }
    finally:
        connection.close()
    return chunks, titles


def _production_answer(
    case: Any,
    contexts: Sequence[Mapping[str, Any]],
    *,
    model: str | None = None,
) -> Any:
    from backend.app.schemas.retrieval import RetrievalCandidate
    from backend.app.services.answering import generate_answer
    from backend.app.services.model_clients import complex_answer_json

    def strict_answer(messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = complex_answer_json(messages, model=model)
        if payload is None:
            raise RuntimeError("Agent provider returned no valid JSON response")
        return payload

    candidates = [RetrievalCandidate.model_validate(row) for row in contexts]
    return generate_answer(
        str(case.question),
        candidates,
        [],
        None,
        answer_callable=strict_answer,
    )


def require_generation_gate(path: Path) -> None:
    if not path.is_file():
        raise ValueError("full Generation requires a passed 10-case Gate")
    payload = _load_json(path)
    if payload.get("passed") is not True:
        raise ValueError("full Generation requires a passed 10-case Gate")


def run_generation_stage(
    *,
    dataset_path: Path,
    retrieval_path: Path,
    db_path: Path,
    top_k: int,
    output_dir: Path,
    case_ids_path: Path | None,
    resume: bool,
) -> dict[str, Any]:
    from backend.app.core.config import get_settings
    from .generation_checkpoint import run_component_generation_cases

    if top_k != 5:
        raise ValueError("formal Generation evaluation requires --top-k 5")
    if case_ids_path is None:
        require_generation_gate(output_dir / "generation-gate.json")
    dataset = _load_dataset(dataset_path)
    report = _load_json(retrieval_path)
    all_cases = _validate_frozen_inputs(dataset, report)
    selected = (
        select_simple_rag_cases(dataset, _read_case_ids(case_ids_path))
        if case_ids_path
        else all_cases
    )
    chunks, titles = _open_sqlite_snapshot(db_path)
    contexts = materialize_ranked_contexts(
        report["retrieval"],
        chunks_by_id=chunks,
        titles_by_document_id=titles,
        top_k=top_k,
    )
    settings = get_settings()
    model = resolve_generation_model(
        configured_model=settings.resolved_agent_model,
        process_model=os.environ.get("AGENT_MODEL"),
    )
    if not (settings.resolved_agent_api_key and settings.resolved_agent_base_url and model):
        raise ValueError("Agent provider configuration is incomplete")
    signature = build_component_signature(
        dataset_sha256=_sha256_file(dataset_path),
        retrieval_sha256=_sha256_file(retrieval_path),
        top_k=top_k,
        model=model,
    )
    result = run_component_generation_cases(
        cases=selected,
        contexts_by_case=contexts,
        output=output_dir / "generation-results.json",
        signature=signature,
        answer_fn=lambda case, ranked_contexts: _production_answer(
            case, ranked_contexts, model=model
        ),
        resume=resume,
    )
    error_count = sum(bool(row.get("error")) for row in result["cases"])
    if len(selected) == 10 and error_count:
        _write_json_atomic(
            output_dir / "generation-gate.json",
            {
                "passed": False,
                "case_count": 10,
                "generation_error_count": error_count,
                "reason": "Generation/provider errors must be zero.",
            },
        )
    _update_manifest(
        output_dir,
        {
            "generation": {
                "status": "complete" if result["complete"] else "partial",
                "signature": signature,
                "target_case_count": len(selected),
                "completed_case_count": result["completed_case_count"],
                "top_k": top_k,
                "router_called": False,
                "agent_tool_loop_called": False,
                "chroma_initialized": False,
                "case_ids_file": str(case_ids_path) if case_ids_path else None,
            }
        },
    )
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _generation_markdown(scorecard: Mapping[str, Any]) -> str:
    def status(name: str) -> str:
        return "PASS" if scorecard["gates"].get(name) else "FAIL"

    return "\n".join(
        [
            "# RAG Generation Component Scorecard",
            "",
            "| Metric | Value | Gate |",
            "| --- | ---: | --- |",
            f"| Faithfulness | {scorecard['faithfulness']:.4f} | {status('faithfulness')} |",
            f"| Response Relevancy | {scorecard['response_relevancy']:.4f} | {status('response_relevancy')} |",
            f"| Factual Correctness F1 | {scorecard['factual_correctness_f1']:.4f} | {status('factual_correctness_f1')} |",
            f"| Citation Validity | {scorecard['citation_validity_rate']:.4f} | {status('citation_validity')} |",
            f"| Abstention | {scorecard['abstention_pass_count']}/{scorecard['abstention_case_count']} | {status('abstention')} |",
            f"| Fabrication count | {scorecard['fabrication_count']} | {status('fabrication')} |",
            f"| Generation errors | {scorecard['generation_error_count']} | {status('generation_errors')} |",
            "",
            f"Latency P50/P95/P99: {scorecard['latency_ms']['p50']:.1f} / "
            f"{scorecard['latency_ms']['p95']:.1f} / "
            f"{scorecard['latency_ms']['p99']:.1f} ms",
            "",
        ]
    )


def _failure_attribution(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    metrics = row["metrics"]
    failures: list[dict[str, Any]] = []
    case_id = str(row["case_id"])
    checks = (
        ("faithfulness", 0.90, "hallucination"),
        ("response_relevancy", 0.80, "generation_irrelevant"),
        ("factual_correctness_f1", 0.80, "generation_incorrect_or_incomplete"),
        ("context_precision", 0.80, "context_noise"),
        ("context_recall", 0.80, "retrieval_context_gap"),
    )
    for metric, threshold, category in checks:
        value = float(metrics.get(metric, 0.0))
        if value < threshold:
            failures.append(
                {
                    "case_id": case_id,
                    "category": category,
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                }
            )
    if not metrics.get("citation_validity"):
        failures.append({"case_id": case_id, "category": "citation_error"})
    if metrics.get("fabrication"):
        failures.append({"case_id": case_id, "category": "fabrication"})
    if row.get("expect_insufficient_evidence") and metrics.get("abstention_correct") is not True:
        failures.append({"case_id": case_id, "category": "abstention_failure"})
    if row.get("error"):
        failures.append({"case_id": case_id, "category": "generation_provider_error"})
    return failures


def run_score_stage(
    *,
    generation_path: Path,
    judgments_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    generation = _load_json(generation_path)
    judgment_payload = _load_json(judgments_path)
    judgments = (
        judgment_payload.get("cases", [])
        if isinstance(judgment_payload, Mapping)
        else judgment_payload
    )
    if not isinstance(judgments, list):
        raise ValueError("judgments must be a list or an object with cases")
    generation_by_id = {str(row["case_id"]): row for row in generation.get("cases", [])}
    judgment_by_id = {
        str(row["case_id"]): row
        for row in judgments
        if isinstance(row, Mapping) and row.get("case_id")
    }
    if set(judgment_by_id) != set(generation_by_id):
        raise ValueError("judgment case IDs must exactly match generation results")

    embedding_texts: list[str] = []
    embedding_slices: dict[str, tuple[int, int]] = {}
    for case_id, judgment in judgment_by_id.items():
        questions = list(judgment.get("reconstructed_questions", []))
        if len(questions) != 3:
            raise ValueError(f"{case_id}: exactly three reconstructed questions are required")
        start = len(embedding_texts)
        embedding_texts.extend([str(generation_by_id[case_id]["question"]), *map(str, questions)])
        embedding_slices[case_id] = (start, start + 4)

    from backend.app.rag.vector_store import embed_texts

    embeddings = embed_texts(embedding_texts, input_type="query")
    judged_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case_id, generation_row in generation_by_id.items():
        judgment = dict(judgment_by_id[case_id])
        start, end = embedding_slices[case_id]
        vectors = embeddings[start:end]
        judgment["response_relevancy"] = response_relevancy_from_embeddings(
            vectors[0],
            vectors[1:],
        )
        answer_claims = list(judgment.get("answer_claims", []))
        judgment["citation_validity"] = citations_are_valid(
            generation_row,
            answer_claims,
        )
        metrics = score_ragas_judgment(judgment)
        row = {
            "case_id": case_id,
            "question": generation_row["question"],
            "answer": generation_row.get("answer", ""),
            "reference_answer": generation_row.get("reference_answer", ""),
            "expect_insufficient_evidence": generation_row.get(
                "expect_insufficient_evidence", False
            ),
            "error": generation_row.get("error"),
            "judgment": judgment,
            "metrics": metrics,
        }
        judged_rows.append(row)
        failures.extend(_failure_attribution(row))

    scorecard = aggregate_generation_scores(
        judged_rows,
        expected_count=len(generation_by_id),
    )
    latencies = [float(row.get("latency_ms", 0.0)) for row in generation_by_id.values()]
    scorecard["latency_ms"] = {
        "p50": _percentile(latencies, 0.50),
        "p95": _percentile(latencies, 0.95),
        "p99": _percentile(latencies, 0.99),
    }
    _write_json_atomic(
        output_dir / "generation-judged.json",
        {"cases": judged_rows},
    )
    _write_json_atomic(output_dir / "generation-scorecard.json", scorecard)
    _write_text_atomic(
        output_dir / "generation-scorecard.md",
        _generation_markdown(scorecard),
    )
    _write_jsonl_atomic(output_dir / "failure-attribution.jsonl", failures)

    if len(judged_rows) == 10:
        gate = evaluate_generation_gate(judged_rows)
        _write_json_atomic(output_dir / "generation-gate.json", gate)
    else:
        gate = None

    retrieval_scorecard_path = output_dir / "retrieval-scorecard.json"
    if retrieval_scorecard_path.is_file():
        retrieval_scorecard = _load_json(retrieval_scorecard_path)
        retrieval_scorecard["metrics"]["context_precision"] = scorecard["context_precision"]
        retrieval_scorecard["metrics"]["context_recall"] = scorecard["context_recall"]
        retrieval_scorecard["gates"]["ragas_context_metrics"] = {
            "context_precision": {
                "threshold": 0.80,
                "value": scorecard["context_precision"],
                "passed": scorecard["context_precision"] >= 0.80,
            },
            "context_recall": {
                "threshold": 0.80,
                "value": scorecard["context_recall"],
                "passed": scorecard["context_recall"] >= 0.80,
            },
        }
        _write_json_atomic(retrieval_scorecard_path, retrieval_scorecard)
        _write_text_atomic(
            output_dir / "retrieval-scorecard.md",
            _retrieval_markdown(retrieval_scorecard),
        )

    _write_text_atomic(
        output_dir / "interview-summary.md",
        _interview_summary(scorecard, gate),
    )
    _update_manifest(
        output_dir,
        {
            "judge": {
                "status": "complete",
                "method": "RAGAS-style offline judgments plus local embeddings",
                "external_judge_api_called": False,
                "judgment_case_count": len(judged_rows),
                "judgments_sha256": _sha256_file(judgments_path),
            }
        },
    )
    return scorecard


def _interview_summary(
    scorecard: Mapping[str, Any],
    gate: Mapping[str, Any] | None,
) -> str:
    gate_text = (
        f"10-case Gate passed: {gate['passed']}."
        if gate is not None
        else "The full component scorecard is reported below."
    )
    return "\n".join(
        [
            "# RAG Component Evaluation Interview Summary",
            "",
            "The evaluation separates retrieval from generation and does not connect Router or Agent tools.",
            "Retrieval reuses frozen BM25 plus vector Hybrid ranks, production fusion, and reranking.",
            "Generation uses only the frozen Hybrid Top-5 contexts.",
            "Judging reproduces RAGAS-style definitions without an external Judge API.",
            "",
            gate_text,
            "",
            f"Faithfulness: {scorecard['faithfulness']:.4f}",
            f"Response Relevancy: {scorecard['response_relevancy']:.4f}",
            f"Factual Correctness F1: {scorecard['factual_correctness_f1']:.4f}",
            f"Citation Validity: {scorecard['citation_validity_rate']:.4f}",
            "",
        ]
    )


def _update_manifest(output_dir: Path, update: Mapping[str, Any]) -> None:
    path = output_dir / "manifest.json"
    payload = (
        _load_json(path)
        if path.is_file()
        else {
            "evaluation": "rag-component-v1",
            "writes_confined_to_workspace": True,
            "components": {},
        }
    )
    payload["components"].update(update)
    _write_json_atomic(path, payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen Hybrid retrieval and Top-5 answer generation"
    )
    parser.add_argument(
        "--stage",
        choices=("retrieval", "generation", "score"),
        required=True,
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--retrieval-report", type=Path)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-ids-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generation-results", type=Path)
    parser.add_argument("--judgments", type=Path)
    return parser


def _require_paths(args: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        flags = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise ValueError(f"stage {args.stage} requires {flags}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.stage == "retrieval":
        _require_paths(args, "dataset", "retrieval_report", "db")
        run_retrieval_stage(
            dataset_path=args.dataset,
            retrieval_path=args.retrieval_report,
            db_path=args.db,
            top_k=args.top_k,
            output_dir=args.output_dir,
        )
    elif args.stage == "generation":
        _require_paths(args, "dataset", "retrieval_report", "db")
        run_generation_stage(
            dataset_path=args.dataset,
            retrieval_path=args.retrieval_report,
            db_path=args.db,
            top_k=args.top_k,
            output_dir=args.output_dir,
            case_ids_path=args.case_ids_file,
            resume=args.resume,
        )
    else:
        _require_paths(args, "judgments")
        generation_path = args.generation_results or args.output_dir / "generation-results.json"
        run_score_stage(
            generation_path=generation_path,
            judgments_path=args.judgments,
            output_dir=args.output_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
