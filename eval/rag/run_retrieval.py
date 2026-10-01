"""Run fresh RAG retrieval over locally supplied PDFs and the 20-paper gold set."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from sqlalchemy.orm import Session

from backend.app.rag.hybrid import hybrid_search
from backend.app.schemas.retrieval import QueryPlan
from .corpus_runtime import (
    _isolated_runtime,
    create_eval_database,
    prepare_corpus,
    resolve_corpus,
)
from .evidence_mapping import load_case_chunks, map_gold_evidence
from .rag_gold import RAGGoldCase, RAGGoldDataset
from .rag_quality import (
    RankedChunk,
    RetrievalEvalCase,
    RetrievalRunResult,
    evaluate_retrieval_results,
)

RetrievalMode = Literal["bm25", "vector", "hybrid"]
Retriever = Callable[
    ["PreparedRetrievalCase", RetrievalMode, int],
    tuple[list[RankedChunk], float, str | None, str | None],
]


@dataclass(frozen=True)
class PreparedRetrievalCase:
    question: str
    evaluation: RetrievalEvalCase
    gold: RAGGoldCase | None = None


def filter_cases_simple_rag(dataset: RAGGoldDataset) -> list[RAGGoldCase]:
    """Select exactly the simple_rag slice: 60 single_paper + 10 no-evidence."""

    return [
        case
        for case in dataset.cases
        if case.task_type in {"single_paper", "insufficient_evidence"}
    ]


def run_retrieval_sweep(
    cases: Sequence[PreparedRetrievalCase],
    *,
    retriever: Retriever,
    modes: Sequence[RetrievalMode] = ("bm25", "vector", "hybrid"),
    top_ks: Sequence[int] = (3, 5, 10, 20),
) -> dict[str, dict[str, Any]]:
    ks = _normalize_top_ks(top_ks)
    invalid = set(modes) - {"bm25", "vector", "hybrid"}
    if invalid:
        raise ValueError(f"unknown retrieval mode: {sorted(invalid)[0]}")
    output: dict[str, dict[str, Any]] = {}
    for mode in dict.fromkeys(modes):
        results: list[RetrievalRunResult] = []
        details: list[dict[str, Any]] = []
        for case in cases:
            diagnostics: Mapping[str, Any] = {}
            try:
                retrieved = retriever(case, mode, max(ks))
                if len(retrieved) == 3:
                    ranked, latency_ms, error = retrieved
                    degraded = None
                elif len(retrieved) == 4:
                    ranked, latency_ms, error, degraded = retrieved
                elif len(retrieved) == 5:
                    ranked, latency_ms, error, degraded, diagnostics = retrieved
                else:
                    raise ValueError("retriever must return 3, 4, or 5 values")
            except Exception as exc:  # noqa: BLE001 - stable per-case error record
                ranked, latency_ms, error, degraded = (
                    [],
                    0.0,
                    (f"{type(exc).__name__}: {exc}"),
                    None,
                )
            result = RetrievalRunResult(
                case_id=case.evaluation.case_id,
                ranked_chunks=ranked,
                latency_ms=latency_ms,
                error=error,
                degraded=degraded,
            )
            results.append(result)
            details.append(
                {
                    "case_id": result.case_id,
                    "question": case.question,
                    "latency_ms": latency_ms,
                    "error": error,
                    "degraded": degraded,
                    "diagnostics": dict(diagnostics),
                    "ranked_chunks": [asdict(item) for item in ranked],
                }
            )
        metrics = evaluate_retrieval_results(
            [case.evaluation for case in cases],
            results,
            top_ks=ks,
        )
        output[mode] = {"metrics": asdict(metrics), "cases": details}
    return output


def prepare_retrieval_cases(
    db: Session,
    dataset: RAGGoldDataset,
    document_ids: Mapping[str, int],
    *,
    minimum_mapping_score: float = 0.35,
) -> tuple[list[PreparedRetrievalCase], list[dict[str, Any]]]:
    prepared: list[PreparedRetrievalCase] = []
    mapping_errors: list[dict[str, Any]] = []
    for case in dataset.cases:
        if case.expect_insufficient_evidence:
            prepared.append(
                PreparedRetrievalCase(
                    question=case.question,
                    gold=case,
                    evaluation=RetrievalEvalCase(
                        case_id=case.case_id,
                        expect_insufficient_evidence=True,
                    ),
                )
            )
            continue
        chunks = load_case_chunks(db, case, document_ids)
        mapping = map_gold_evidence(
            case=case,
            chunks=chunks,
            document_ids=document_ids,
            minimum_score=minimum_mapping_score,
        )
        if mapping.unmapped:
            mapping_errors.append(
                {"case_id": case.case_id, "unmapped_evidence_indices": mapping.unmapped}
            )
        chunk_document_ids = {
            int(chunk.id): int(chunk.document_id)
            for chunk in chunks
            if int(chunk.id) in mapping.relevance_by_chunk
        }
        prepared.append(
            PreparedRetrievalCase(
                question=case.question,
                gold=case,
                evaluation=RetrievalEvalCase(
                    case_id=case.case_id,
                    relevance_by_chunk=mapping.relevance_by_chunk,
                    chunk_document_ids=chunk_document_ids,
                    required_document_ids={
                        document_ids[key] for key in case.required_documents
                    },
                ),
            )
        )
    return prepared, mapping_errors


def build_live_retriever(db: Session) -> Retriever:
    from backend.app.rag.tokenization import tokenize_mixed

    def retrieve(
        case: PreparedRetrievalCase,
        mode: RetrievalMode,
        limit: int,
    ) -> tuple[list[RankedChunk], float, str | None, str | None]:
        # Query construction must use the question only: injecting gold
        # keywords would leak the expected answer into the BM25 channel.
        # tokenize_mixed gives deterministic zh/en terms without an LLM.
        lexical = tokenize_mixed(case.question)[:24]
        plan = QueryPlan(
            intent="simple_rag",
            confidence=1,
            standalone_query=case.question,
            lexical_terms=lexical,
            semantic_queries=[case.question],
        )
        started = time.perf_counter()
        result = hybrid_search(
            db,
            plan,
            evidence_limit=limit,
            mode=mode,
            rerank=(mode == "hybrid"),
        )
        latency_ms = (time.perf_counter() - started) * 1000
        ranked = [
            RankedChunk(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                page_number=item.page_number,
                rank=index,
                score=(
                    item.rerank_score
                    if item.rerank_score is not None
                    else item.fusion_score
                ),
            )
            for index, item in enumerate(result.candidates, 1)
        ]
        degraded = (
            ",".join(result.diagnostics.degraded_channels)
            if result.diagnostics.degraded_channels
            else None
        )
        return (
            ranked,
            latency_ms,
            None,
            degraded,
            result.diagnostics.model_dump(mode="json"),
        )

    return retrieve


def _normalize_top_ks(values: Sequence[int]) -> tuple[int, ...]:
    ks = tuple(sorted(set(int(value) for value in values)))
    if not ks or ks[0] <= 0:
        raise ValueError("top-k values must be positive")
    return ks


def ensure_vector_index(
    db: Session,
    *,
    force_rebuild: bool = False,
    allow_rebuild: bool = True,
) -> int:
    """Guarantee a queryable vector index inside the benchmark process.

    The local chromadb build cannot reload a persisted hnsw segment across
    processes at this corpus size, so the benchmark rebuilds the index from
    SQLite chunks in-process whenever a probe query fails. Runs must use a
    single process for all stages; the persisted copy stays best effort.

    With ``allow_rebuild=False`` (``--vector-policy reuse-required``) a failed
    probe is fatal: the run stops instead of silently paying for a rebuild.
    """

    import tempfile

    from sqlalchemy import select

    from backend.app.models import Document
    from backend.app.models.chunk import DocumentChunk
    from backend.app.rag import vector_store

    if not force_rebuild:
        try:
            vector_store.query_chunks("vector index health probe", limit=1)
            return -1
        except Exception as exc:  # noqa: BLE001 - any failure triggers rebuild
            if not allow_rebuild:
                raise RuntimeError(
                    "vector index is unavailable and --vector-policy "
                    "reuse-required forbids a rebuild; run the vector-index "
                    f"stage first (probe error: {type(exc).__name__}: {exc})"
                ) from exc
            print(
                f"vector index unavailable ({type(exc).__name__}); rebuilding in-process"
            )

    # The probe client keeps sqlite handles that cannot be released on this
    # platform, so instead of wiping the locked directory we switch to a
    # fresh one. The stale directory stays behind as dead weight only.
    fresh = Path(
        tempfile.mkdtemp(
            prefix="chroma-fresh-", dir=Path(vector_store.chroma_dir()).parent
        )
    )
    vector_store.chroma_dir = lambda: fresh
    vector_store._persistent_client.cache_clear()

    total = 0
    for document in db.scalars(select(Document).order_by(Document.id)):
        rows = list(
            db.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document.id)
                .order_by(DocumentChunk.id)
            )
        )
        payload = [
            {
                "chunk_id": str(chunk.id),
                "content": chunk.content,
                "metadata": {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "title": document.title,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                },
            }
            for chunk in rows
        ]
        vector_store.upsert_chunks(payload, purge_existing=False)
        total += len(payload)
        print(f"vector rebuild: doc {document.id} -> {len(payload)} chunks", flush=True)
    print(f"in-process vector index rebuilt: {total} chunks")
    return total


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def benchmark_signature(
    *,
    dataset_name: str,
    dataset_version: str,
    top_ks: Sequence[int],
    modes: Sequence[str],
    slice_name: str = "all",
    case_ids: Sequence[str] | None = None,
) -> str:
    payload = json.dumps(
        {
            "dataset": dataset_name,
            "version": dataset_version,
            "top_k": list(_normalize_top_ks(top_ks)),
            "modes": list(dict.fromkeys(modes)),
            "slice": slice_name,
            "case_ids": sorted(case_ids) if case_ids else [],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent / "datasets" / "papers_20_gold_v1.json",
    )
    parser.add_argument(
        "--uploads",
        type=Path,
        required=True,
        help="Directory containing the 20 corpus PDFs",
    )
    parser.add_argument("--workdir", type=Path, default=Path(".tmp/rag-eval"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".tmp/rag-eval/reports")
    )
    parser.add_argument(
        "--case-slice", choices=("simple-rag", "all"), default="simple-rag"
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("bm25", "vector", "hybrid"),
        nargs="+",
        default=("hybrid",),
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=(3, 5, 10, 20))
    parser.add_argument("--reuse-index", action="store_true")
    args = parser.parse_args()

    dataset = RAGGoldDataset.model_validate_json(
        args.dataset.read_text(encoding="utf-8")
    )
    if args.case_slice == "simple-rag":
        dataset = dataset.model_copy(update={"cases": filter_cases_simple_rag(dataset)})
    corpus = resolve_corpus(dataset, args.uploads)
    with _isolated_runtime(args.workdir.resolve()):
        engine, db = create_eval_database(args.workdir.resolve())
        try:
            document_ids = prepare_corpus(db, corpus, reuse_index=args.reuse_index)
            if any(mode in {"vector", "hybrid"} for mode in args.retrieval_mode):
                ensure_vector_index(db)
            cases, mapping_errors = prepare_retrieval_cases(db, dataset, document_ids)
            retrieval = run_retrieval_sweep(
                cases,
                retriever=build_live_retriever(db),
                modes=args.retrieval_mode,
                top_ks=args.top_k,
            )
            report = {
                "dataset": dataset.name,
                "version": dataset.version,
                "signature": benchmark_signature(
                    dataset_name=dataset.name,
                    dataset_version=dataset.version,
                    top_ks=args.top_k,
                    modes=args.retrieval_mode,
                    slice_name=args.case_slice,
                ),
                "top_k": list(_normalize_top_ks(args.top_k)),
                "slice": args.case_slice,
                "case_ids": None,
                "mapping_errors": mapping_errors,
                "gold_chunks": {
                    case.evaluation.case_id: sorted(
                        set(case.evaluation.relevance_by_chunk)
                    )
                    for case in cases
                    if case.evaluation.relevance_by_chunk
                },
                "retrieval": retrieval,
            }
            _write_json_atomic(args.output_dir / "retrieval.json", report)
            print(
                json.dumps(
                    {
                        "case_count": len(cases),
                        "mapping_error_count": len(mapping_errors),
                        "modes": {
                            mode: values["metrics"]
                            for mode, values in retrieval.items()
                        },
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        finally:
            db.close()
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
