"""Query-rewrite ablation helpers for pre-rerank Hybrid retrieval."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from backend.app.rag.hybrid import hybrid_search
from backend.app.rag.tokenization import tokenize_mixed
from backend.app.schemas.retrieval import QueryPlan

_PROTECTED_TOKEN = re.compile(
    r"[A-Za-z][A-Za-z0-9]*(?:[-_.+][A-Za-z0-9]+)*|\d+(?:\.\d+)?%?"
)


@dataclass(frozen=True)
class QueryRewrite:
    english_query: str
    lexical_terms: tuple[str, ...]


def build_rewrite_messages(question: str) -> list[dict[str, str]]:
    """Build the frozen one-shot rewrite request used by this ablation."""

    return [
        {
            "role": "system",
            "content": (
                "Rewrite an academic retrieval question for hybrid search. Return JSON "
                "with exactly: english_query (one concise English semantic query) and "
                "lexical_terms (1-12 Chinese/English search terms). Preserve every named "
                "method, dataset, number, equation, metric, and constraint verbatim. Do not "
                "answer the question, add facts, or broaden its scope."
            ),
        },
        {"role": "user", "content": question},
    ]


def parse_rewrite_payload(question: str, payload: Mapping[str, Any]) -> QueryRewrite:
    english_query = " ".join(str(payload.get("english_query") or "").split())
    raw_terms = payload.get("lexical_terms")
    if not english_query:
        raise ValueError("english_query is required")
    if not isinstance(raw_terms, list):
        raise ValueError("lexical_terms must be a list")

    terms: list[str] = []
    seen: set[str] = set()
    for value in raw_terms:
        term = " ".join(str(value).split())
        key = term.casefold()
        if not term or key in seen:
            continue
        seen.add(key)
        terms.append(term)
    if not terms or len(terms) > 12:
        raise ValueError("lexical_terms must contain 1-12 unique terms")

    searchable = " ".join([english_query, *terms]).casefold()
    for token in _protected_tokens(question):
        if token.casefold() not in searchable:
            raise ValueError(f"rewrite dropped protected token: {token}")
    return QueryRewrite(english_query=english_query, lexical_terms=tuple(terms))


def build_query_plan(question: str, rewrite: QueryRewrite | None = None) -> QueryPlan:
    if rewrite is None:
        standalone_query = question
        lexical_terms = tokenize_mixed(question)[:24]
    else:
        standalone_query = rewrite.english_query
        lexical_terms = list(rewrite.lexical_terms)
    return QueryPlan(
        intent="simple_rag",
        confidence=1,
        standalone_query=standalone_query,
        lexical_terms=lexical_terms,
        semantic_queries=[standalone_query],
        document_id=None,
    )


def extract_raw_fusion_cases(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract pre-rerank ranks and latency from a prior live Hybrid report."""

    output: dict[str, dict[str, Any]] = {}
    cases = report.get("retrieval", {}).get("hybrid", {}).get("cases", [])
    for case in cases:
        case_id = str(case.get("case_id") or "")
        diagnostics = case.get("diagnostics") or {}
        fusion = diagnostics.get("fusion_candidates") or []
        if not case_id or not fusion:
            raise ValueError(f"Raw report lacks fusion candidates for {case_id or '<unknown>'}")
        timings = diagnostics.get("timings_ms") or {}
        output[case_id] = {
            "case_id": case_id,
            "question": case.get("question"),
            "latency_ms": sum(
                float(timings.get(name) or 0.0) for name in ("bm25", "vector", "fusion")
            ),
            "error": case.get("error"),
            "degraded": case.get("degraded"),
            "rerank_applied": False,
            "ranked_chunks": [dict(row) for row in fusion],
            "diagnostics": {
                "channel_status": diagnostics.get("channel_status") or {},
                "channel_candidates": diagnostics.get("channel_candidates") or {},
                "fusion_candidates": [dict(row) for row in fusion],
            },
        }
    return output


def retrieve_fusion(
    db: Any,
    plan: QueryPlan,
    *,
    limit: int = 20,
    search_fn: Callable[..., Any] = hybrid_search,
) -> dict[str, Any]:
    """Run production Hybrid retrieval and stop at RRF fusion."""

    started = time.perf_counter()
    result = search_fn(
        db,
        plan,
        evidence_limit=limit,
        mode="hybrid",
        rerank=False,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    diagnostics = result.diagnostics.model_dump(mode="json")
    ranked = [
        {
            "chunk_id": int(candidate.chunk_id),
            "document_id": int(candidate.document_id),
            "page_number": int(candidate.page_number),
            "rank": rank,
            "score": float(candidate.fusion_score),
            "bm25_rank": candidate.bm25_rank,
            "vector_rank": candidate.vector_rank,
        }
        for rank, candidate in enumerate(result.candidates, start=1)
    ]
    return {
        "latency_ms": latency_ms,
        "error": None,
        "degraded": (
            ",".join(result.diagnostics.degraded_channels)
            if result.diagnostics.degraded_channels
            else None
        ),
        "rerank_applied": False,
        "ranked_chunks": ranked,
        "diagnostics": diagnostics,
    }


def paired_hit_outcomes(
    raw_rankings: Mapping[str, Sequence[int]],
    rewrite_rankings: Mapping[str, Sequence[int]],
    gold_by_case: Mapping[str, Sequence[int]],
    *,
    k: int,
) -> dict[str, Any]:
    counts = {"improved": 0, "regressed": 0, "both_hit": 0, "both_miss": 0}
    cases: list[dict[str, Any]] = []
    for case_id, gold_values in gold_by_case.items():
        gold = {int(value) for value in gold_values}
        raw = [int(value) for value in raw_rankings.get(case_id, [])]
        rewritten = [int(value) for value in rewrite_rankings.get(case_id, [])]
        raw_hit = bool(set(raw[:k]) & gold)
        rewrite_hit = bool(set(rewritten[:k]) & gold)
        if not raw_hit and rewrite_hit:
            outcome = "improved"
        elif raw_hit and not rewrite_hit:
            outcome = "regressed"
        elif raw_hit:
            outcome = "both_hit"
        else:
            outcome = "both_miss"
        counts[outcome] += 1
        cases.append(
            {
                "case_id": case_id,
                "outcome": outcome,
                "raw_hit": raw_hit,
                "rewrite_hit": rewrite_hit,
                "raw_first_gold_rank": _first_gold_rank(raw, gold),
                "rewrite_first_gold_rank": _first_gold_rank(rewritten, gold),
            }
        )
    return {"k": k, "counts": counts, "cases": cases}


def adoption_decision(
    *,
    improved: int,
    regressed: int,
    raw_hit20: float,
    rewrite_hit20: float,
    error_count: int,
) -> dict[str, Any]:
    gates = {
        "hit5_gain_at_least_3_cases": improved >= 3,
        "hit20_non_decreasing": rewrite_hit20 >= raw_hit20,
        "hit5_regressions_at_most_1": regressed <= 1,
        "zero_errors": error_count == 0,
    }
    return {"adopt": all(gates.values()), "gates": gates}


def _protected_tokens(question: str) -> list[str]:
    protected: list[str] = []
    for token in _PROTECTED_TOKEN.findall(question):
        letters = [character for character in token if character.isalpha()]
        upper = sum(character.isupper() for character in letters)
        if any(character.isdigit() for character in token) or upper >= 2:
            protected.append(token)
    return list(dict.fromkeys(protected))


def _first_gold_rank(ranking: Sequence[int], gold: set[int]) -> int | None:
    return next((rank for rank, chunk_id in enumerate(ranking, 1) if chunk_id in gold), None)
