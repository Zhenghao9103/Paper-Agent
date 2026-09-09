"""Top-level Agentic-RAG research flow.

The service owns routing and orchestration; the QueryPlanner, retrieval, and
the bounded tool loop are ordinary Python calls.  There is deliberately no
graph runtime here so the public trace remains small, deterministic and safe
to expose through the chat API.

Execution paths (the Router only classifies intent):
- ``direct``: deterministic database answers for metadata questions; any other
  non-paper interaction is answered by the Agent model without retrieval.
- ``simple_rag``: the QueryPlanner builds one fixed query plan, one hybrid
  retrieval round runs, and the Agent model answers from that evidence.
- ``agentic_rag``: no separate planner; the question goes straight into the
  Agent tool loop.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from sqlalchemy.orm import Session

from ..agents.orchestrator import AgentResearchResult, run_agent
from ..agents.tools import search_arxiv  # noqa: F401 - compatibility patch point
from ..core.config import get_settings
from ..rag.hybrid import HybridSearchResult, RetrievalUnavailable, hybrid_search
from ..rag.tokenization import tokenize_mixed
from ..schemas.chat import Citation, WebSource
from ..schemas.retrieval import QueryPlan, ResearchResult, RetrievalCandidate
from .answering import (
    AnswerResult,
    answer_direct,
    answer_general,
    generate_agentic_answer,
    generate_answer,
    is_document_operation,
)
from .context_checkpoint import load_active_context
from .intent_router import RoutingOutcome, route_question
from .llm import ContextWindowExceededError
from .memory import MemoryHit, query_relevant_memories
from .progress import ProgressCallback, emit_progress
from .query_planner import QueryPlanningOutcome, plan_queries


@dataclass(frozen=True)
class _RetrievalState:
    candidates: list[RetrievalCandidate]
    diagnostics: Mapping[str, Any]
    degraded: list[str]


def safe_query_relevant_memories(
    db: Session, question: str, *, limit: int = 3
) -> list[MemoryHit]:
    try:
        return query_relevant_memories(db, question, limit=limit)
    except Exception:
        return []


def _memory_context(db: Session, session_id: int | None) -> dict[str, Any]:
    if session_id is None:
        return {}
    try:
        context = load_active_context(db, int(session_id))
    except Exception:
        return {}
    return context if isinstance(context, dict) else {}


def _route_event(
    outcome: RoutingOutcome,
    *,
    timing_ms: float = 0.0,
    source: str | None = None,
) -> dict[str, Any]:
    event = {
        "type": "route_decision",
        "intent": outcome.intent,
        "timing_ms": round(float(timing_ms), 3),
        "degraded": bool(outcome.degraded),
        "error": outcome.error,
    }
    if source is not None:
        event["source"] = source
    return event


def _planning_event(outcome: QueryPlanningOutcome, *, timing_ms: float) -> dict[str, Any]:
    plan = outcome.plan
    return {
        "type": "query_planning",
        "degraded": bool(outcome.degraded),
        "error": outcome.error,
        "timing_ms": round(float(timing_ms), 3),
        "semantic_query_count": len(plan.semantic_queries),
        "lexical_term_count": len(plan.lexical_terms),
        "synonym_count": len(plan.synonyms),
    }


def _retrieval_state(
    db: Session,
    plan: QueryPlan,
    *,
    evidence_limit: int,
    trace_events: list[dict[str, Any]],
) -> _RetrievalState:
    started = perf_counter()
    try:
        result = hybrid_search(db, plan, evidence_limit=evidence_limit)
        result = _normalise_hybrid_result(result, evidence_limit, plan.document_id)
    except RetrievalUnavailable:
        elapsed = (perf_counter() - started) * 1000
        trace_events.append(
            {
                "type": "retrieval_degraded",
                "degraded": True,
                "reason": "all retrieval channels unavailable",
                "timings_ms": {"total": round(elapsed, 3)},
            }
        )
        return _RetrievalState([], {}, ["bm25", "vector"])
    except Exception:
        elapsed = (perf_counter() - started) * 1000
        trace_events.append(
            {
                "type": "retrieval_degraded",
                "degraded": True,
                "reason": "retrieval error",
                "timings_ms": {"total": round(elapsed, 3)},
            }
        )
        return _RetrievalState([], {}, ["retrieval"])

    diagnostics = result.diagnostics
    timings = dict(getattr(diagnostics, "timings_ms", {}) or {})
    degraded = list(getattr(diagnostics, "degraded_channels", []) or [])
    trace_events.extend(
        [
            {
                "type": "bm25_retrieval",
                "candidate_count": sum(
                    1 for candidate in result.candidates if candidate.bm25_rank is not None
                ),
                "timing_ms": round(float(timings.get("bm25", 0.0)), 3),
                "degraded": "bm25" in degraded,
            },
            {
                "type": "vector_retrieval",
                "candidate_count": sum(
                    1 for candidate in result.candidates if candidate.vector_rank is not None
                ),
                "timing_ms": round(float(timings.get("vector", 0.0)), 3),
                "degraded": "vector" in degraded,
            },
            {
                "type": "retrieval_fusion",
                "candidate_count": len(result.candidates),
                "timing_ms": round(float(timings.get("fusion", 0.0)), 3),
                "degraded": bool(degraded),
            },
        ]
    )
    if degraded:
        trace_events.append(
            {
                "type": "retrieval_degraded",
                "degraded": True,
                "reason": ",".join(degraded),
                "degraded_channels": degraded,
                "timings_ms": timings,
            }
        )
    return _RetrievalState(list(result.candidates), timings, degraded)


def _normalise_hybrid_result(
    result: Any,
    evidence_limit: int,
    document_id: int | None = None,
) -> HybridSearchResult:
    """Accept the typed result and the simple list shape used by API tests."""

    if isinstance(result, HybridSearchResult):
        if result.candidates:
            return result
        return result
    if isinstance(result, list):
        candidates: list[RetrievalCandidate] = []
        for index, item in enumerate(result):
            if isinstance(item, RetrievalCandidate):
                candidates.append(item)
                continue
            if not isinstance(item, Mapping):
                continue
            try:
                if float(item.get("score", 0.0)) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
            metadata = item.get("metadata")
            if not isinstance(metadata, Mapping):
                metadata = {}
            try:
                candidates.append(
                    RetrievalCandidate(
                        chunk_id=int(metadata.get("chunk_id", item.get("chunk_id", index + 1))),
                        document_id=int(metadata.get("document_id", document_id or 1)),
                        title=str(metadata.get("title", "Untitled")),
                        page_number=int(metadata.get("page_number", 0)),
                        chunk_index=int(metadata.get("chunk_index", index)),
                        content=str(item.get("content", "")),
                        fusion_rank=index + 1,
                        fusion_score=float(item.get("score", 0.0)),
                    )
                )
            except (TypeError, ValueError):
                continue
        return HybridSearchResult(candidates=candidates[:evidence_limit])
    return HybridSearchResult()


def _agent_events(agent_result: AgentResearchResult) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current_round: int | None = None
    for event in agent_result.trace_events:
        event_type = str(event.get("event", event.get("type", "")))
        if event_type == "research_plan_created":
            events.append(
                {
                    "type": "agent_research_plan",
                    "protocol_version": 2,
                    "round": event.get("round", 0),
                    "status": event.get("status", "ok"),
                    "claim_ids": list(event.get("claim_ids") or []),
                    "action_ids": list(event.get("action_ids") or []),
                    "error_category": event.get("error_category"),
                    "degraded": event.get("status") == "error",
                }
            )
            continue
        if event_type == "search_action_started":
            events.append(
                {
                    "type": "agent_tool_call",
                    "protocol_version": 2,
                    "round": event.get("round"),
                    "tool": event.get("tool", "unknown"),
                    "call_id": event.get("action_id"),
                    "claim_ids": list(event.get("claim_ids") or []),
                    "query": str(event.get("query") or "")[:1000],
                    "status": "ok",
                    "executed": True,
                    "degraded": False,
                }
            )
            continue
        if event_type == "search_action_result":
            events.append(
                {
                    "type": "agent_tool_result",
                    "protocol_version": 2,
                    "round": event.get("round"),
                    "tool": event.get("tool", "unknown"),
                    "call_id": event.get("action_id"),
                    "status": event.get("status", "ok"),
                    "validation_error_category": event.get("error_category"),
                    "degraded": event.get("status") == "error",
                }
            )
            continue
        if event_type in {
            "retrieval_round_started",
            "evidence_judge_started",
            "evidence_judge_result",
            "claim_coverage_updated",
        }:
            events.append({"type": event_type, **event})
            continue
        if event_type == "evidence_pool_updated":
            events.append({"type": "agent_evidence_pool_update", **event})
            continue
        if event_type == "evidence_gap_identified":
            events.append({"type": "agent_evidence_gap", **event})
            continue
        if event_type == "research_sufficiency_decided":
            events.append({"type": "agent_sufficiency_decision", **event})
            continue
        if event_type == "evidence_pack_built":
            events.append({"type": "agent_evidence_pack", **event})
            continue
        if event_type == "agent_research_completed":
            events.append(
                {
                    "type": "agent_research_finish",
                    **event,
                    "status": event.get("completion_status"),
                    "degraded": event.get("completion_status") != "completed",
                }
            )
            continue
        if event_type == "model_start":
            current_round = event.get("round")
            continue
        if event_type == "model_result" and event.get("status") == "error":
            category = event.get("error_category")
            if category not in {
                "client_error",
                "connection_error",
                "provider_error",
                "rate_limit",
                "server_error",
                "timeout",
            }:
                category = "provider_error"
            events.append(
                {
                    "type": "agent_model_result",
                    "round": event.get("round", current_round),
                    "status": "error",
                    "error_category": category,
                    "provider_attempts": max(
                        1, min(2, int(event.get("provider_attempts") or 1))
                    ),
                    "degraded": True,
                }
            )
            continue
        if event_type == "tool_start":
            status = event.get("status", "ok")
            events.append(
                {
                    "type": "agent_tool_call",
                    "tool": event.get("tool", "unknown"),
                    "call_id": event.get("call_id"),
                    "round": event.get("round", current_round),
                    "argument_fingerprint": event.get("argument_fingerprint", ""),
                    "timing_ms": _event_timing_ms(event),
                    "status": status,
                    "executed": status in (None, "ok"),
                    "degraded": status not in (None, "ok"),
                }
            )
        elif event_type == "tool_result":
            events.append(
                {
                    "type": "agent_tool_result",
                    "tool": event.get("tool", "unknown"),
                    "call_id": event.get("call_id"),
                    "round": event.get("round", current_round),
                    "timing_ms": _event_timing_ms(event),
                    "status": event.get("status", "ok"),
                    "result_count": int(event.get("result_count") or 0),
                    "evidence_delta_count": int(
                        event.get("evidence_delta_count") or 0
                    ),
                    "returned_document_ids": _event_document_ids(event),
                    "validation_error_category": _event_validation_error_category(
                        event
                    ),
                    "degraded": event.get("status") not in (None, "ok"),
                }
            )
        elif event_type == "task_search":
            events.append(
                {
                    "type": "agent_task_search",
                    "round": event.get("round", current_round),
                    "task_id": str(event.get("task_id") or "")[:32],
                    "subquestion": str(event.get("subquestion") or "")[:1000],
                    "query": str(event.get("query") or "")[:1000],
                    "status": event.get("status", "ok"),
                    "result_count": int(event.get("result_count") or 0),
                    "evidence_delta_count": int(
                        event.get("evidence_delta_count") or 0
                    ),
                    "task_evidence_count": int(
                        event.get("task_evidence_count") or 0
                    ),
                    "degraded": event.get("status") not in (None, "ok"),
                }
            )
        elif event_type == "degraded":
            events.append(
                {
                    "type": "retrieval_degraded",
                    "degraded": True,
                    "reason": event.get("reason", "agent degraded"),
                    "completion_status": event.get("completion_status"),
                    "round": event.get("round", current_round),
                    "timing_ms": _event_timing_ms(event),
                }
            )
        elif event_type == "research_finish_result":
            events.append(
                {
                    "type": "agent_research_finish",
                    "round": event.get("round", current_round),
                    "status": event.get("status", "error"),
                    "subquestion_count": int(event.get("subquestion_count") or 0),
                    "supported_count": int(event.get("supported_count") or 0),
                    "unsupported_count": int(event.get("unsupported_count") or 0),
                    "available_evidence_count": int(
                        event.get("available_evidence_count") or 0
                    ),
                    "accepted_evidence_count": int(
                        event.get("accepted_evidence_count") or 0
                    ),
                    "accepted_chunk_ids": _event_chunk_ids(
                        event, "accepted_chunk_ids"
                    ),
                    "task_count": int(event.get("task_count") or 0),
                    "omitted_task_ids": [
                        str(value)[:32]
                        for value in (event.get("omitted_task_ids") or [])[:8]
                    ],
                    "error_category": event.get("error_category"),
                    "normalization_categories": [
                        str(value)[:64]
                        for value in (event.get("normalization_categories") or [])[:5]
                    ],
                    "terminal_response_category": (
                        event.get("terminal_response_category")
                        if event.get("terminal_response_category")
                        in {
                            "no_tool_call",
                            "multiple_tool_calls",
                            "unexpected_tool_name",
                        }
                        else None
                    ),
                    "terminal_tool_count": max(
                        0, min(4, int(event.get("terminal_tool_count") or 0))
                    ),
                    "terminal_tool_names": [
                        value if value == "finish_research" else "unknown"
                        for value in (event.get("terminal_tool_names") or [])[:4]
                    ],
                    "terminal_attempt": max(
                        0, min(2, int(event.get("terminal_attempt") or 0))
                    ),
                    "degraded": event.get("status")
                    not in {"completed", "insufficient_evidence"},
                }
            )
        elif event_type == "subquestion_coverage":
            events.append(
                {
                    "type": "agent_subquestion_coverage",
                    "round": event.get("round", current_round),
                    "subquestion_count": int(event.get("subquestion_count") or 0),
                    "supported_count": int(event.get("supported_count") or 0),
                    "unsupported_count": int(event.get("unsupported_count") or 0),
                    "degraded": False,
                }
            )
        elif event_type == "accepted_evidence":
            events.append(
                {
                    "type": "agent_accepted_evidence",
                    "round": event.get("round", current_round),
                    "count": int(event.get("count") or 0),
                    "chunk_ids": _event_chunk_ids(event, "chunk_ids"),
                    "degraded": False,
                }
            )
        elif event_type == "fallback_selected_evidence":
            events.append(
                {
                    "type": "agent_fallback_selected_evidence",
                    "round": event.get("round", current_round),
                    "count": int(event.get("count") or 0),
                    "chunk_ids": _event_chunk_ids(event, "chunk_ids"),
                    "degraded": True,
                }
            )
    return events


def _event_timing_ms(event: dict[str, Any]) -> float:
    raw = event.get("timing_ms", event.get("duration_ms", event.get("elapsed_ms", 0.0)))
    try:
        return round(float(raw or 0.0), 3)
    except (TypeError, ValueError):
        return 0.0


def _event_document_ids(event: Mapping[str, Any]) -> list[int]:
    values = event.get("returned_document_ids")
    if not isinstance(values, list):
        return []
    return sorted(
        {
            value
            for value in values[:20]
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        }
    )


def _event_chunk_ids(event: Mapping[str, Any], key: str) -> list[int]:
    values = event.get(key)
    if not isinstance(values, list):
        return []
    return list(
        dict.fromkeys(
            value
            for value in values[:8]
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        )
    )


def _event_validation_error_category(event: Mapping[str, Any]) -> str | None:
    value = event.get("validation_error_category")
    allowed = {
        "missing_required_field",
        "extra_field",
        "limit_exceeded",
        "invalid_range",
        "invalid_type",
        "malformed_json",
        "invalid_value",
        "task_limit_exceeded",
        "invalid_task_query",
        "duplicate_task_query",
        "task_query_limit_exceeded",
        "unknown_task_id",
    }
    return value if isinstance(value, str) and value in allowed else None


def _evidence_event(
    candidates: Iterable[RetrievalCandidate], *, timing_ms: float = 0.0
) -> dict[str, Any]:
    items = list(candidates)
    return {
        "type": "evidence_selected",
        "count": len(items),
        "chunk_ids": [candidate.chunk_id for candidate in items[:8]],
        "document_ids": sorted({candidate.document_id for candidate in items}),
        "timing_ms": round(float(timing_ms), 3),
    }


def _fallback_answer(question: str, citations: list[Citation]) -> str:
    if not citations:
        return (
            "\u5f53\u524d\u8bc1\u636e\u4e0d\u8db3\uff0c"
            "\u65e0\u6cd5\u53ef\u9760\u56de\u7b54\u8be5\u95ee\u9898\u3002"
        )
    content = " ".join(" ".join(citation.content.split()) for citation in citations[:3])
    if any(
        marker in question.casefold()
        for marker in (
            "innovation",
            "novelty",
            "contribution",
            "\u521b\u65b0",
            "\u8d21\u732e",
        )
    ):
        parts = []
        for sentence in re.split(r"(?<=[.!?])\s+", content):
            lowered = sentence.casefold()
            if "journal of latex class files" in lowered:
                continue
            if sum(token.isdigit() for token in sentence) > max(8, len(sentence) // 3):
                continue
            if any(
                marker in lowered
                for marker in ("propose", "method", "optimal transport", "stability", "innovation")
            ):
                parts.append(sentence.strip())
        selected = " ".join(parts) or content
        return f"\u521b\u65b0\u70b9\uff1a\n{selected[:800]}"
    return f"\u672c\u5730\u77e5\u8bc6\u5e93\u8bc1\u636e\u5982\u4e0b\uff1a\n{content[:900]}"


def _is_operational_question(question: str) -> bool:
    if is_document_operation(question):
        return True
    lowered = question.casefold()
    return any(
        marker in lowered
        for marker in (
            "how many papers",
            "how many documents",
            "current database",
            "list papers",
            "papers in database",
            "当前数据库",
            "当前知识库",
            "列出论文",
            "论文数量",
        )
    )


def _agentic_plan(question: str, document_id: int | None) -> QueryPlan:
    """Deterministic entry plan for the Agent tool loop (no planner call)."""

    bounded = " ".join(str(question).split())[:1000] or "(empty question)"
    return QueryPlan(
        intent="agentic_rag",
        confidence=1.0,
        standalone_query=bounded,
        lexical_terms=tokenize_mixed(bounded)[:8] or [bounded],
        synonyms=[],
        semantic_queries=[bounded],
        document_id=document_id,
    )


def _memory_trace(trace: list[str], short_term_memory: Mapping[str, Any]) -> None:
    if short_term_memory.get("session_summary") or short_term_memory.get(
        "recent_messages"
    ) or short_term_memory.get("session_memory"):
        trace.extend(["load_recent_chat_history", "inject_memory_into_prompt"])


def run_research(
    db: Session,
    *,
    question: str,
    document_id: int | None = None,
    session_id: int | None = None,
    forced_intent: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ResearchResult:
    """Run route -> plan/retrieve/agent -> evidence-bound answer generation."""

    if forced_intent not in {None, "agentic_rag"}:
        raise ValueError("forced_intent must be agentic_rag when provided")
    short_term_memory = _memory_context(db, session_id)
    memory_hits = safe_query_relevant_memories(db, question, limit=3)
    route_source = None
    emit_progress(progress_callback, "routing")
    if forced_intent is None:
        route_started = perf_counter()
        outcome = route_question(question, document_id, short_term_memory)
        route_timing_ms = (perf_counter() - route_started) * 1000
    else:
        outcome = RoutingOutcome(intent="agentic_rag")
        route_timing_ms = 0.0
        route_source = "evaluation_override"
    intent = outcome.intent
    trace_events: list[dict[str, Any]] = [
        _route_event(outcome, timing_ms=route_timing_ms, source=route_source)
    ]
    trace: list[str] = ["load_short_term_memory", f"route_{intent}"]
    if outcome.degraded:
        trace.append("route_degraded")
    if memory_hits:
        trace.append("load_long_term_vector_memory")
    _memory_trace(trace, short_term_memory)

    settings = get_settings()

    if intent == "direct":
        emit_progress(progress_callback, "generation")
        if _is_operational_question(question):
            answer_started = perf_counter()
            answer = answer_direct(db, question, document_id)
            answer_timing_ms = (perf_counter() - answer_started) * 1000
            trace.append("database_document_summary")
            trace.append("answer_direct")
            return ResearchResult(
                answer=answer,
                citations=[],
                web_sources=[],
                trace=trace,
                memory_hits=memory_hits,
                trace_events=trace_events
                + [
                    _evidence_event([]),
                    {
                        "type": "answer_generation",
                        "model_tier": "deterministic",
                        "timing_ms": round(answer_timing_ms, 3),
                        "degraded": False,
                    },
                ],
            )
        answer_started = perf_counter()
        answer = answer_general(question)
        answer_timing_ms = (perf_counter() - answer_started) * 1000
        trace.append("answer_direct_agent")
        return ResearchResult(
            answer=answer,
            citations=[],
            web_sources=[],
            trace=trace,
            memory_hits=memory_hits,
            trace_events=trace_events
            + [
                _evidence_event([]),
                {
                    "type": "answer_generation",
                    "model_tier": "agent",
                    "timing_ms": round(answer_timing_ms, 3),
                    "degraded": False,
                },
            ],
        )

    if intent == "simple_rag":
        emit_progress(progress_callback, "planning")
        planning_started = perf_counter()
        planning = plan_queries(question, document_id, short_term_memory)
        planning_ms = (perf_counter() - planning_started) * 1000
        trace_events.append(
            _planning_event(planning, timing_ms=planning_ms)
        )
        trace.append("plan_queries")
        if planning.degraded:
            trace.append("query_planning_degraded")
        plan = planning.plan
        emit_progress(progress_callback, "retrieval")
        retrieved = _retrieval_state(
            db,
            plan,
            evidence_limit=settings.simple_evidence_limit,
            trace_events=trace_events,
        )
        candidates = retrieved.candidates
        web_sources: list[WebSource] = []
        trace.extend(["local_retrieve", "answer_synthesizer_zh"])
        if retrieved.degraded:
            trace.append("retrieval_degraded")
            emit_progress(progress_callback, "degraded")
        else:
            trace.append("bge_rerank")
        agent_result: AgentResearchResult | None = None
    else:
        trace.append("agentic_research")
        plan = _agentic_plan(question, document_id)
        try:
            agent_result = run_agent(
                db,
                question,
                plan,
                short_term_memory,
                progress_callback=progress_callback,
            )
        except ContextWindowExceededError:
            raise
        except Exception:
            agent_result = AgentResearchResult(
                degraded_reason="agent provider error",
                completion_status="tool_error",
            )
        candidates = list(agent_result.evidence)
        web_sources = list(agent_result.web_sources)
        trace.extend(
            [
                "agent_tool_loop",
                f"agent_rounds_{agent_result.rounds}",
                f"agent_tool_calls_{agent_result.tool_calls}",
            ]
        )
        if agent_result.completion_status is not None:
            trace.append(f"agent_completion_{agent_result.completion_status}")
        trace_events.extend(_agent_events(agent_result))
        if agent_result.degraded_reason:
            trace.append("retrieval_degraded")
            emit_progress(progress_callback, "degraded")
            trace_events.append(
                {
                    "type": "retrieval_degraded",
                    "degraded": True,
                    "reason": agent_result.degraded_reason,
                    "round": agent_result.rounds,
                    "timing_ms": 0.0,
                }
            )

    emit_progress(progress_callback, "evidence")
    evidence_started = perf_counter()
    trace_events.append(
        _evidence_event(
            candidates,
            timing_ms=(perf_counter() - evidence_started) * 1000,
        )
    )
    emit_progress(progress_callback, "generation")
    answer_started = perf_counter()
    answer_memory: dict[str, Any] = {
        "short_term": short_term_memory,
        "long_term": memory_hits,
    }
    if agent_result is not None:
        answer_memory["agent_research"] = {
            "completion_status": agent_result.completion_status,
            "claims": [item.model_dump(mode="json") for item in agent_result.claims],
        }
    if agent_result is not None and agent_result.evidence_pack is not None:
        generated = generate_agentic_answer(
            question,
            agent_result.evidence_pack,
            answer_memory,
        )
        trace_events.append(
            {
                "type": "answer_verification_result",
                "protocol_version": 2,
                "status": generated.verification_status,
                "passed": generated.verification_status
                in {"passed", "passed_after_repair"},
                "citation_identity_valid": bool(generated.citations)
                or generated.verification_status != "verification_failed",
                "verification_attempt_count": len(generated.verification_attempts),
                "verification_attempts": list(generated.verification_attempts),
            }
        )
        if generated.verification_status == "verification_failed":
            agent_result = agent_result.model_copy(
                update={"completion_status": "verification_failed"}
            )
            trace.append("agent_completion_verification_failed")
            trace_events.append(
                {
                    "type": "agent_research_finish",
                    "protocol_version": 2,
                    "round": agent_result.rounds,
                    "status": "verification_failed",
                    "completion_status": "verification_failed",
                    "degraded": True,
                }
            )
    elif agent_result is not None:
        generated = AnswerResult(
            answer=(
                "当前 Evidence Pool 尚未覆盖全部必要问题，"
                "因此停止生成结论性回答。"
            )
        )
    else:
        generated = generate_answer(
            question,
            candidates,
            web_sources,
            answer_memory,
        )
    trace_events.append(
        {
            "type": "answer_generation",
            "model_tier": "agent",
            "timing_ms": round((perf_counter() - answer_started) * 1000, 3),
            "degraded": False,
        }
    )
    if web_sources and not generated.web_sources:
        generated = generated.model_copy(update={"web_sources": web_sources})
    citations = list(generated.citations)
    answer = generated.answer
    if answer and citations and "\u8bc1\u636e\u4e0d\u8db3" not in answer:
        trace.append("llm_answer_synthesizer")
    # Preserve a useful offline answer when the provider is unavailable.  The
    # answer generator still validates and bounds every citation before returning.
    if citations and (not answer.strip() or "\u8bc1\u636e\u4e0d\u8db3" in answer):
        answer = _fallback_answer(question, citations)
    return ResearchResult(
        answer=answer,
        citations=citations,
        web_sources=list(generated.web_sources),
        trace=trace,
        memory_hits=memory_hits,
        trace_events=trace_events,
    )


def _format_history_section(short_term_memory: Mapping[str, Any] | None) -> str:
    """Compatibility formatter used by the chat trace tests and prompt audits."""

    memory = short_term_memory or {}
    lines: list[str] = []
    if session_memory := memory.get("session_memory"):
        lines.extend(
            [
                "\u4f1a\u8bdd\u4efb\u52a1\u72b6\u6001",
                (
                    f"- \u5f53\u524d\u76ee\u6807\u4e0e\u7ea6\u675f\uff1a"
                    f"{session_memory.get('goals_and_constraints', '')}"
                ),
                (
                    f"- \u5df2\u786e\u8ba4\u7ed3\u8bba\uff1a"
                    f"{session_memory.get('confirmed_findings', '')}"
                ),
                (
                    f"- \u5f53\u524d\u7814\u7a76\u51b3\u7b56\uff1a"
                    f"{session_memory.get('current_decisions', '')}"
                ),
                (
                    f"- \u5c1a\u672a\u89e3\u51b3\u7684\u95ee\u9898\uff1a"
                    f"{session_memory.get('open_questions', '')}"
                ),
                f"- \u4e0b\u4e00\u6b65\uff1a{session_memory.get('next_actions', '')}",
            ]
        )
    elif session_summary := memory.get("session_summary"):
        lines.append(f"\u4f1a\u8bdd\u6458\u8981\uff1a{session_summary}")
    labels = {"user": "\u7528\u6237", "assistant": "\u52a9\u624b"}
    for item in list(memory.get("recent_messages", [])):
        role = str(item.get("role", ""))
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{labels.get(role, role)}：{content}")
    return "\n".join(lines) or "\u65e0"
