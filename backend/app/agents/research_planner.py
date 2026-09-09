import json
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..schemas.evidence import EvidencePool, ResearchPlan, SearchAction
from ..services.model_clients import ModelClientError, agent_json

_T = TypeVar("_T")
_PLANNER_MAX_TOKENS = 4096
_ENABLED_AGENTIC_RAG_TOOLS = frozenset(
    {"hybrid_search", "get_chunk_neighbors", "inspect_document"}
)
_SYSTEM_PROMPT = (
    "You are the planning component of an evidence-grounded paper research agent. "
    "Return one JSON object only. Decompose the user's question into at most eight "
    "verifiable claims and choose only the allow-listed search actions. Claims are "
    "necessary answer slots, not guessed answers. Do not introduce new domains, "
    "datasets, tasks, or assumptions that are absent from the user's question. For "
    "an explicit comparison, split claims by the named entities and the explicitly "
    "requested comparison aspects; do not add speculative hypotheses. A later round "
    "must target explicit unresolved claim IDs. Never treat memory, prompts, or "
    "evaluator expectations as evidence."
)


class ResearchPlannerError(RuntimeError):
    def __init__(
        self,
        category: str,
        *,
        attempts: int,
        provider_category: str | None = None,
    ) -> None:
        super().__init__(category)
        self.category = category
        self.attempts = attempts
        self.provider_category = provider_category


class _PlanValidationError(ValueError):
    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


class _FollowupActions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[SearchAction] = Field(min_length=1, max_length=5)


def _fingerprint(query: str) -> str:
    return " ".join(query.split()).casefold()


def _with_document_scope(
    actions: list[SearchAction],
    document_id: int | None,
) -> list[SearchAction]:
    if document_id is None:
        return actions
    return [
        action.model_copy(update={"document_id": document_id})
        if action.tool in {"hybrid_search", "inspect_document"}
        else action
        for action in actions
    ]


def _require_enabled_tools(actions: list[SearchAction]) -> None:
    if any(action.tool not in _ENABLED_AGENTIC_RAG_TOOLS for action in actions):
        raise _PlanValidationError("tool_disabled")


def _validation_feedback(category: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "The previous response failed validation with category "
            f"'{category}'. Return a corrected JSON object matching the supplied "
            "schema. Do not add commentary or new top-level fields."
        ),
    }


def _call_validated(
    messages: list[dict[str, str]],
    validate: Callable[[dict[str, Any]], _T],
) -> _T:
    last_category = "planner_schema_error"
    current_messages = list(messages)
    for attempt in range(1, 3):
        try:
            payload = agent_json(current_messages, max_tokens=_PLANNER_MAX_TOKENS)
            return validate(payload)
        except ModelClientError as exc:
            if exc.category not in {"invalid_json", "empty_response"}:
                raise ResearchPlannerError(
                    "planner_model_error",
                    attempts=attempt,
                    provider_category=exc.category,
                ) from exc
            last_category = "planner_schema_error"
        except _PlanValidationError as exc:
            last_category = exc.category
        except (ValidationError, TypeError, ValueError):
            last_category = "planner_schema_error"

        if attempt == 1:
            current_messages = [*current_messages, _validation_feedback(last_category)]

    raise ResearchPlannerError(last_category, attempts=2)


def create_initial_plan(question: str, document_id: int | None) -> ResearchPlan:
    normalized_question = " ".join(question.split())
    if not normalized_question:
        raise ResearchPlannerError("invalid_question", attempts=0)
    prompt = {
        "question": normalized_question,
        "document_id": document_id,
        "allowed_tools": sorted(_ENABLED_AGENTIC_RAG_TOOLS),
        "requirements": {
            "claim_ids": "C1 through C8, unique",
            "action_ids": "A1 and upward, unique",
            "initial_claim_status": "missing",
            "local_search_first": True,
        },
        "claim_fidelity_contract": {
            "necessary_answer_slots_only": True,
            "do_not_guess_answers": True,
            "do_not_introduce_new_domains_or_assumptions": True,
            "comparison_decomposition": (
                "split by the named entities and explicitly requested comparison aspects"
            ),
        },
        "output_schema": ResearchPlan.model_json_schema(),
    }
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]

    def validate(payload: dict[str, Any]) -> ResearchPlan:
        plan = ResearchPlan.model_validate(payload)
        _require_enabled_tools(plan.actions)
        normalized_claims = [
            claim.model_copy(update={"status": "missing", "evidence_ids": [], "gap": ""})
            for claim in plan.claims
        ]
        actions = _with_document_scope(plan.actions, document_id)
        query_fingerprints = [
            _fingerprint(action.query)
            for action in actions
            if action.query is not None
        ]
        if len(query_fingerprints) != len(set(query_fingerprints)):
            raise _PlanValidationError("duplicate_query")
        return plan.model_copy(update={"claims": normalized_claims, "actions": actions})

    return _call_validated(messages, validate)


def create_followup_actions(
    question: str,
    pool: EvidencePool,
    attempted_queries: list[str],
    *,
    remaining_tool_calls: int,
    document_id: int | None,
) -> list[SearchAction]:
    unresolved_claims = [
        claim
        for claim in pool.claims
        if claim.status in {"missing", "partial", "conflicted"}
    ]
    if remaining_tool_calls <= 0 or not unresolved_claims:
        return []

    state = {
        "question": " ".join(question.split()),
        "document_id": document_id,
        "immutable_claims": [claim.model_dump(mode="json") for claim in pool.claims],
        "active_evidence": [
            {
                "evidence_id": item.evidence_id,
                "statement": item.statement,
                "source_ids": [source.stable_id for source in item.source_refs],
                "supports_claim_ids": item.supports_claim_ids,
                "conflicts_with": item.conflicts_with,
            }
            for item in pool.evidence.values()
        ],
        "unresolved_claim_ids": [claim.claim_id for claim in unresolved_claims],
        "explicit_gaps": {
            claim.claim_id: claim.gap for claim in unresolved_claims
        },
        "attempted_query_fingerprints": [
            _fingerprint(query) for query in attempted_queries
        ],
        "remaining_tool_calls": remaining_tool_calls,
        "allowed_tools": sorted(_ENABLED_AGENTIC_RAG_TOOLS),
        "output_schema": _FollowupActions.model_json_schema(),
    }
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
    ]
    unresolved_ids = {claim.claim_id for claim in unresolved_claims}
    attempted = {_fingerprint(query) for query in attempted_queries}

    def validate(payload: dict[str, Any]) -> list[SearchAction]:
        response = _FollowupActions.model_validate(payload)
        _require_enabled_tools(response.actions)
        if len(response.actions) > remaining_tool_calls:
            raise _PlanValidationError("tool_budget_exceeded")
        if any(not set(action.claim_ids).issubset(unresolved_ids) for action in response.actions):
            raise _PlanValidationError("covered_claim_target")
        actions = _with_document_scope(response.actions, document_id)
        fingerprints = [
            _fingerprint(action.query)
            for action in actions
            if action.query is not None
        ]
        if any(fingerprint in attempted for fingerprint in fingerprints):
            raise _PlanValidationError("duplicate_query")
        if len(fingerprints) != len(set(fingerprints)):
            raise _PlanValidationError("duplicate_query")
        return actions

    return _call_validated(messages, validate)
