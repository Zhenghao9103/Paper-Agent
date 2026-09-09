"""Bounded claim-aware Agentic RAG research loop."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..schemas.chat import WebSource
from ..schemas.evidence import EvidencePack, EvidencePool, ResearchClaim, SearchAction
from ..schemas.retrieval import EvidenceLedger, QueryPlan, RetrievalCandidate
from ..services.model_clients import ModelClientError
from ..services.progress import ProgressCallback, emit_progress
from .evidence_judge import (
    EvidenceCandidateGroup,
    EvidenceJudgeError,
    judge_evidence_round,
)
from .evidence_pack import EvidencePackError, build_evidence_pack
from .evidence_pool import EvidencePoolUpdateError, apply_judge_result, pool_is_sufficient
from .research_planner import (
    ResearchPlannerError,
    create_followup_actions,
    create_initial_plan,
)
from .tools import execute_tool_result

CompletionStatus = Literal[
    "completed",
    "insufficient_evidence",
    "conflicting_evidence",
    "budget_exhausted",
    "stalled",
    "timeout",
    "planner_error",
    "judge_error",
    "tool_error",
    "verification_failed",
]


class AgentResearchResult(BaseModel):
    protocol_version: Literal[2] = 2
    evidence: list[RetrievalCandidate] = Field(default_factory=list)
    web_sources: list[WebSource] = Field(default_factory=list)
    rounds: int = 0
    tool_calls: int = 0
    validation_errors: int = 0
    trace_events: list[dict[str, Any]] = Field(default_factory=list)
    degraded_reason: str | None = None
    completion_status: CompletionStatus
    claims: list[ResearchClaim] = Field(default_factory=list)
    evidence_pool: EvidencePool | None = None
    evidence_pack: EvidencePack | None = None


def _action_arguments(
    action: SearchAction,
    claims: dict[str, ResearchClaim],
) -> dict[str, Any]:
    if action.tool == "hybrid_search":
        return {
            "task_id": action.action_id,
            "subquestion": " / ".join(
                claims[claim_id].question for claim_id in action.claim_ids
            ),
            "query": action.query,
            "document_id": action.document_id,
        }
    if action.tool == "get_chunk_neighbors":
        return {"chunk_id": action.chunk_id}
    if action.tool == "inspect_document":
        return {"document_id": action.document_id}
    return {"query": action.query}


def _active_source_count(pool: EvidencePool) -> int:
    return sum(len(item.source_refs) for item in pool.evidence.values())


def _candidate_group_from_result(
    action: SearchAction,
    claims: dict[str, ResearchClaim],
    result: Mapping[str, Any],
) -> EvidenceCandidateGroup:
    local_payloads = result.get("candidates") or result.get("chunks") or []
    web_payloads = result.get("web_sources") or []
    return EvidenceCandidateGroup(
        action_id=action.action_id,
        target_claim_ids=action.claim_ids,
        query=action.query,
        subquestion=" / ".join(claims[claim_id].question for claim_id in action.claim_ids),
        local_candidates=[RetrievalCandidate.model_validate(item) for item in local_payloads],
        web_sources=[WebSource.model_validate(item) for item in web_payloads],
    )


def _selected_sources(
    pool: EvidencePool,
    ledger: EvidenceLedger,
) -> tuple[list[RetrievalCandidate], list[WebSource]]:
    local: list[RetrievalCandidate] = []
    web: list[WebSource] = []
    seen_chunks: set[int] = set()
    seen_urls: set[str] = set()
    for item in pool.evidence.values():
        for source in item.source_refs:
            if source.kind == "local":
                if source.chunk_id in seen_chunks:
                    continue
                candidate = ledger.local.get(source.chunk_id)
                if candidate is not None:
                    seen_chunks.add(source.chunk_id)
                    local.append(candidate)
                continue
            if source.url in seen_urls:
                continue
            web_source = ledger.web.get(source.url)
            if web_source is not None:
                seen_urls.add(source.url)
                web.append(web_source)
    return local, web


def _expired(started: float, timeout_seconds: float) -> bool:
    return time.monotonic() - started >= timeout_seconds


def _trace(events: list[dict[str, Any]], event: str, **payload: Any) -> None:
    payload.setdefault("protocol_version", 2)
    payload.setdefault("round", 0)
    payload.setdefault("timing_ms", 0.0)
    events.append({"event": event, "name": event, "node": event, **payload})


def _failure_result(
    status: CompletionStatus,
    trace_events: list[dict[str, Any]],
    *,
    validation_errors: int = 0,
) -> AgentResearchResult:
    return AgentResearchResult(
        completion_status=status,
        degraded_reason=status,
        validation_errors=validation_errors,
        trace_events=trace_events,
    )


def run_agent(
    db: Session,
    question: str,
    query_plan: QueryPlan,
    short_term_memory: Mapping[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AgentResearchResult:
    """Plan, retrieve, judge and refine until evidence is sufficient or bounded."""

    del short_term_memory
    settings = get_settings()
    started = time.monotonic()
    trace_events: list[dict[str, Any]] = []
    ledger = EvidenceLedger()
    rounds = 0
    tool_calls = 0
    validation_errors = 0

    try:
        emit_progress(progress_callback, "agent_planning", round_number=1)
        planner_started = time.monotonic()
        plan = create_initial_plan(question, query_plan.document_id)
    except (ResearchPlannerError, ModelClientError, ValidationError, ValueError) as exc:
        _trace(
            trace_events,
            "research_plan_created",
            protocol_version=2,
            status="error",
            error_category=getattr(exc, "category", "planner_schema_error"),
            timing_ms=round((time.monotonic() - planner_started) * 1000, 3),
        )
        return _failure_result("planner_error", trace_events, validation_errors=1)

    pool = EvidencePool(claims=plan.claims)
    claims = {claim.claim_id: claim for claim in plan.claims}
    actions = list(plan.actions)
    attempted_queries: list[str] = []
    no_progress_rounds = 0
    tool_error_seen = False
    _trace(
        trace_events,
        "research_plan_created",
        protocol_version=2,
        status="ok",
        claim_ids=list(claims),
            action_ids=[action.action_id for action in actions],
            timing_ms=round((time.monotonic() - planner_started) * 1000, 3),
    )

    max_rounds = min(3, int(settings.agent_max_rounds))
    max_tool_calls = min(5, int(settings.agent_max_tool_calls))
    completion_status: CompletionStatus | None = None

    for round_number in range(1, max_rounds + 1):
        if _expired(started, settings.agent_total_timeout_seconds):
            completion_status = "timeout"
            _trace(trace_events, "timeout", protocol_version=2, round=round_number)
            break
        if not actions:
            completion_status = "insufficient_evidence"
            break

        rounds = round_number
        _trace(
            trace_events,
            "retrieval_round_started",
            protocol_version=2,
            round=round_number,
            action_count=len(actions),
        )
        successful_actions = 0
        candidate_groups: list[EvidenceCandidateGroup] = []
        emit_progress(
            progress_callback,
            "agent_retrieval",
            round_number=round_number,
        )

        for action in actions:
            if tool_calls >= max_tool_calls:
                completion_status = "budget_exhausted"
                break
            if _expired(started, settings.agent_total_timeout_seconds):
                completion_status = "timeout"
                break
            _trace(
                trace_events,
                "search_action_started",
                protocol_version=2,
                round=round_number,
                action_id=action.action_id,
                claim_ids=action.claim_ids,
                tool=action.tool,
                query=action.query,
            )
            tool_calls += 1
            if action.query:
                attempted_queries.append(action.query)
            tool_started = time.monotonic()
            tool_outcome = execute_tool_result(
                db,
                action.tool,
                _action_arguments(action, claims),
                ledger=ledger,
                request_document_id=query_plan.document_id,
            )
            failure = tool_outcome.failure
            _trace(
                trace_events,
                "search_action_result",
                protocol_version=2,
                round=round_number,
                action_id=action.action_id,
                tool=action.tool,
                status=tool_outcome.status,
                error_category=failure.code if failure is not None else None,
                failure_category=failure.category if failure is not None else None,
                retryable=failure.retryable if failure is not None else False,
                attempts=tool_outcome.attempts,
                timing_ms=round((time.monotonic() - tool_started) * 1000, 3),
            )
            if tool_outcome.status == "failed":
                tool_error_seen = True
                validation_errors += int(
                    failure is not None and failure.category == "validation"
                )
                continue

            tool_result = dict(tool_outcome.value or {})
            candidate_groups.append(
                _candidate_group_from_result(action, claims, tool_result)
            )
            successful_actions += 1

        if completion_status in {"timeout", "budget_exhausted"} and not successful_actions:
            break
        if successful_actions == 0 and tool_error_seen:
            completion_status = "tool_error"
            break
        if _expired(started, settings.agent_total_timeout_seconds):
            completion_status = "timeout"
            break

        _trace(
            trace_events,
            "evidence_judge_started",
            protocol_version=2,
            round=round_number,
            local_candidate_count=sum(
                len(group.local_candidates) for group in candidate_groups
            ),
            web_candidate_count=sum(len(group.web_sources) for group in candidate_groups),
            candidate_group_count=len(candidate_groups),
        )
        judge_diagnostics: dict[str, Any] = {}
        try:
            judge_started = time.monotonic()
            judge_result = judge_evidence_round(
                question,
                pool,
                candidate_groups,
                round_number=round_number,
                diagnostics=judge_diagnostics,
            )
            allowed_source_refs = {
                *(candidate.evidence_id for candidate in ledger.local.values()),
                *(f"url:{source.entry_url}" for source in ledger.web.values()),
            }
            source_count_before = _active_source_count(pool)
            evidence_count_before = len(pool.evidence)
            pool = apply_judge_result(
                pool,
                judge_result,
                allowed_source_refs=allowed_source_refs,
                round_number=round_number,
            )
        except (EvidenceJudgeError, EvidencePoolUpdateError, ValidationError, ValueError) as exc:
            validation_errors += 1
            _trace(
                trace_events,
                "evidence_judge_result",
                protocol_version=2,
                round=round_number,
                status="error",
                error_category=getattr(exc, "category", "judge_schema_error"),
                attempts=getattr(exc, "attempts", 1),
                provider_category=getattr(exc, "provider_category", None),
                validation_errors=getattr(exc, "validation_errors", []),
                **judge_diagnostics,
                timing_ms=round((time.monotonic() - judge_started) * 1000, 3),
            )
            completion_status = "judge_error"
            break

        source_count_after = _active_source_count(pool)
        no_progress_rounds = (
            no_progress_rounds + 1
            if source_count_after == source_count_before
            else 0
        )
        _trace(
            trace_events,
            "evidence_judge_result",
            protocol_version=2,
            round=round_number,
            status="ok",
            decision_count=len(judge_result.decisions),
            candidate_count=sum(
                len(group.local_candidates) + len(group.web_sources)
                for group in candidate_groups
            ),
            decision_action_counts={
                action: sum(decision.action == action for decision in judge_result.decisions)
                for action in ("keep", "drop", "merge", "conflict")
            },
            **judge_diagnostics,
            timing_ms=round((time.monotonic() - judge_started) * 1000, 3),
        )
        _trace(
            trace_events,
            "evidence_pool_updated",
            protocol_version=2,
            round=round_number,
            evidence_count=len(pool.evidence),
            growth=len(pool.evidence) - evidence_count_before,
            active_source_count=source_count_after,
            unresolved_conflict_count=len(pool.unresolved_conflicts),
        )
        _trace(
            trace_events,
            "claim_coverage_updated",
            protocol_version=2,
            round=round_number,
            claim_statuses={claim.claim_id: claim.status for claim in pool.claims},
        )
        for claim in pool.claims:
            if claim.status != "covered":
                _trace(
                    trace_events,
                    "evidence_gap_identified",
                    protocol_version=2,
                    round=round_number,
                    claim_id=claim.claim_id,
                    status=claim.status,
                    gap=claim.gap,
                )
        sufficient = pool_is_sufficient(pool)
        _trace(
            trace_events,
            "research_sufficiency_decided",
            protocol_version=2,
            round=round_number,
            sufficient=sufficient,
        )
        if sufficient:
            completion_status = "completed"
            break
        if no_progress_rounds >= 2:
            completion_status = "stalled"
            break
        if round_number >= max_rounds or tool_calls >= max_tool_calls:
            completion_status = (
                "conflicting_evidence" if pool.unresolved_conflicts else "budget_exhausted"
            )
            break

        try:
            emit_progress(
                progress_callback,
                "agent_planning",
                round_number=round_number + 1,
            )
            planner_started = time.monotonic()
            actions = create_followup_actions(
                question,
                pool,
                attempted_queries,
                remaining_tool_calls=max_tool_calls - tool_calls,
                document_id=query_plan.document_id,
            )
        except (ResearchPlannerError, ModelClientError, ValidationError, ValueError) as exc:
            validation_errors += 1
            _trace(
                trace_events,
                "research_plan_created",
                protocol_version=2,
                round=round_number + 1,
                status="error",
                error_category=getattr(exc, "category", "planner_schema_error"),
                timing_ms=round((time.monotonic() - planner_started) * 1000, 3),
            )
            completion_status = "planner_error"
            break

    if completion_status is None:
        completion_status = "tool_error" if tool_error_seen else "insufficient_evidence"
    evidence_pack: EvidencePack | None = None
    if completion_status == "completed":
        try:
            pack_started = time.monotonic()
            evidence_pack = build_evidence_pack(
                pool,
                ledger,
                token_budget=int(settings.agent_generation_evidence_tokens),
                model=str(settings.resolved_agent_model or "unknown"),
            )
            _trace(
                trace_events,
                "evidence_pack_built",
                protocol_version=2,
                status="ok",
                item_count=len(evidence_pack.items),
                token_count=evidence_pack.token_count,
                claim_count=len(evidence_pack.claims),
                covered_claim_count=sum(
                    claim.status == "covered" for claim in evidence_pack.claims
                ),
                timing_ms=round((time.monotonic() - pack_started) * 1000, 3),
            )
        except EvidencePackError as exc:
            completion_status = "insufficient_evidence"
            _trace(
                trace_events,
                "evidence_pack_built",
                protocol_version=2,
                status="error",
                error_category=exc.category,
                timing_ms=round((time.monotonic() - pack_started) * 1000, 3),
            )

    selected_local, selected_web = _selected_sources(pool, ledger)
    _trace(
        trace_events,
        "agent_research_completed",
        protocol_version=2,
        completion_status=completion_status,
        rounds=rounds,
        tool_calls=tool_calls,
        claim_count=len(pool.claims),
        covered_claim_count=sum(claim.status == "covered" for claim in pool.claims),
        evidence_count=len(pool.evidence),
        unresolved_conflict_count=len(pool.unresolved_conflicts),
    )
    return AgentResearchResult(
        evidence=selected_local,
        web_sources=selected_web,
        rounds=rounds,
        tool_calls=tool_calls,
        validation_errors=validation_errors,
        trace_events=trace_events,
        degraded_reason=None if completion_status == "completed" else completion_status,
        completion_status=completion_status,
        claims=pool.claims,
        evidence_pool=pool,
        evidence_pack=evidence_pack,
    )
