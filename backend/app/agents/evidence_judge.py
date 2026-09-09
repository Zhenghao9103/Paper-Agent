import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..core.config import get_settings
from ..schemas.chat import WebSource
from ..schemas.evidence import EvidenceJudgeResult, EvidencePool
from ..schemas.retrieval import RetrievalCandidate
from ..services.context_checkpoint import count_text_tokens
from ..services.model_clients import ModelClientError, agent_json

_JUDGE_MAX_OUTPUT_TOKENS = 6144
_MIN_CANDIDATE_EXCERPT_CHARS = 256
_MAX_CANDIDATE_EXCERPT_CHARS = 4000
_DECISION_ACTION_CONTRACT = {
    "keep": ["statement", "supports_claim_ids"],
    "drop": [],
    "merge": ["target_evidence_id", "supports_claim_ids"],
    "conflict": ["target_evidence_id", "statement", "supports_claim_ids"],
}
_SAFE_ACTION_VALIDATION_CODES = {
    "merge requires target_evidence_id": "merge_requires_target_evidence_id",
    "conflict requires target_evidence_id": "conflict_requires_target_evidence_id",
    "keep requires statement": "keep_requires_statement",
    "conflict requires statement": "conflict_requires_statement",
    "keep requires supports_claim_ids": "keep_requires_supports_claim_ids",
    "merge requires supports_claim_ids": "merge_requires_supports_claim_ids",
    "conflict requires supports_claim_ids": "conflict_requires_supports_claim_ids",
}
_SYSTEM_PROMPT = (
    "You are the Evidence Judge inside an academic Agentic RAG loop. Return one "
    "JSON object with decisions, claim_assessments, overall_sufficient, "
    "unresolved_gaps, and next_search_focus. Judge only supplied sources. Relevance "
    "is not entailment. Use keep, drop, merge, or conflict. Merge only materially "
    "equivalent or complementary propositions. Preserve contradictions. 'Not found' "
    "is not 'contradicted'. Assess every claim. overall_sufficient may be true only "
    "when every required claim is covered and no required claim is conflicted. Never "
    "invent a claim ID, evidence ID, chunk ID, or URL. Candidate-group claim IDs "
    "describe retrieval intent only; independently judge whether each source really "
    "supports those claims. The supplied decision_action_contract lists additional "
    "required fields for each action; follow it exactly."
)


class EvidenceCandidateGroup(BaseModel):
    """Candidates returned by one search action with its original claim intent."""

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(pattern=r"^A[1-9][0-9]*$")
    target_claim_ids: list[str] = Field(min_length=1, max_length=8)
    query: str | None = Field(default=None, max_length=1000)
    subquestion: str = Field(min_length=1, max_length=2000)
    local_candidates: list[RetrievalCandidate] = Field(default_factory=list)
    web_sources: list[WebSource] = Field(default_factory=list)


class EvidenceJudgeError(RuntimeError):
    def __init__(
        self,
        category: str,
        *,
        attempts: int = 1,
        provider_category: str | None = None,
        validation_errors: list[dict[str, str]] | None = None,
    ) -> None:
        super().__init__(category)
        self.category = category
        self.attempts = attempts
        self.provider_category = provider_category
        self.validation_errors = validation_errors or []


def _bounded_validation_errors(exc: ValidationError) -> list[dict[str, str]]:
    bounded: list[dict[str, str]] = []
    for error in exc.errors(include_url=False)[:8]:
        error_type = str(error.get("type") or "validation_error")
        if error_type == "value_error":
            context = error.get("ctx")
            cause = context.get("error") if isinstance(context, dict) else None
            error_type = _SAFE_ACTION_VALIDATION_CODES.get(str(cause), error_type)
        bounded.append(
            {
                "loc": ".".join(str(part) for part in error.get("loc", ()))
                or "result",
                "type": error_type,
            }
        )
    return bounded


def _candidate_rank(candidate: RetrievalCandidate) -> tuple[float, int, int]:
    score = candidate.rerank_score
    return (
        -(score if score is not None else float("-inf")),
        candidate.fusion_rank or 10**9,
        candidate.chunk_id,
    )


def _pool_payload(pool: EvidencePool) -> dict[str, Any]:
    return {
        "protocol_version": pool.protocol_version,
        "claims": [claim.model_dump(mode="json") for claim in pool.claims],
        "active_evidence": [
            {
                "evidence_id": item.evidence_id,
                "statement": item.statement,
                "source_ids": [source.stable_id for source in item.source_refs],
                "supports_claim_ids": item.supports_claim_ids,
                "confidence": item.confidence,
                "conflicts_with": item.conflicts_with,
            }
            for item in pool.evidence.values()
        ],
        "unresolved_conflicts": pool.unresolved_conflicts,
    }


def _candidate_payload(candidate: RetrievalCandidate) -> dict[str, Any]:
    return {
        "source_id": candidate.evidence_id,
        "kind": "local",
        "chunk_id": candidate.chunk_id,
        "document_id": candidate.document_id,
        "title": candidate.title,
        "page_number": candidate.page_number,
        "chunk_index": candidate.chunk_index,
        "content": candidate.content[:_MAX_CANDIDATE_EXCERPT_CHARS],
        "rerank_score": candidate.rerank_score,
        "fusion_rank": candidate.fusion_rank,
    }


def _web_payload(source: WebSource) -> dict[str, Any]:
    return {
        "source_id": f"url:{source.entry_url}",
        "kind": "web",
        "url": source.entry_url,
        "title": source.title,
        "authors": source.authors,
        "published": source.published,
        "summary": source.summary[:_MAX_CANDIDATE_EXCERPT_CHARS],
    }


def _serialize_bounded_payload(
    question: str,
    pool: EvidencePool,
    candidate_groups: list[EvidenceCandidateGroup],
    *,
    round_number: int,
    token_budget: int,
    model: str,
    diagnostics: dict[str, Any] | None = None,
) -> str:
    candidate_count_before = sum(
        len(group.local_candidates) + len(group.web_sources)
        for group in candidate_groups
    )
    payload = {
        "question": " ".join(question.split()),
        "round": round_number,
        "current_pool": _pool_payload(pool),
        "output_schema": EvidenceJudgeResult.model_json_schema(),
        "decision_action_contract": _DECISION_ACTION_CONTRACT,
        "candidate_groups": [
            {
                "action_id": group.action_id,
                "target_claim_ids": group.target_claim_ids,
                "query": group.query,
                "subquestion": group.subquestion,
                "local_candidates": [
                    _candidate_payload(candidate)
                    for candidate in sorted(group.local_candidates, key=_candidate_rank)
                ],
                "web_sources": [_web_payload(source) for source in group.web_sources],
            }
            for group in candidate_groups
        ],
    }

    def serialize() -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def record_diagnostics(serialized_value: str, token_count: int) -> None:
        if diagnostics is None:
            return
        candidate_count_after = sum(
            len(group["local_candidates"]) + len(group["web_sources"])
            for group in payload["candidate_groups"]
        )
        diagnostics.update(
            {
                "input_token_count": token_count,
                "serialized_char_count": len(serialized_value),
                "candidate_group_count": len(payload["candidate_groups"]),
                "candidate_count_before": candidate_count_before,
                "candidate_count_after": candidate_count_after,
                "candidate_count_removed": candidate_count_before
                - candidate_count_after,
            }
        )

    serialized = serialize()
    input_token_count = count_text_tokens(serialized, model)
    while input_token_count > token_budget:
        reduced = False
        for group in reversed(payload["candidate_groups"]):
            for candidate in reversed(group["local_candidates"]):
                content = candidate["content"]
                if len(content) > _MIN_CANDIDATE_EXCERPT_CHARS:
                    shortened_length = max(
                        _MIN_CANDIDATE_EXCERPT_CHARS,
                        len(content) // 2,
                    )
                    candidate["content"] = content[:shortened_length]
                    reduced = True
                    break
            if reduced:
                break
        if not reduced:
            for group in reversed(payload["candidate_groups"]):
                for source in reversed(group["web_sources"]):
                    summary = source["summary"]
                    if len(summary) > _MIN_CANDIDATE_EXCERPT_CHARS:
                        shortened_length = max(
                            _MIN_CANDIDATE_EXCERPT_CHARS,
                            len(summary) // 2,
                        )
                        source["summary"] = summary[:shortened_length]
                        reduced = True
                        break
                if reduced:
                    break
        if not reduced:
            for group in reversed(payload["candidate_groups"]):
                if group["local_candidates"]:
                    group["local_candidates"].pop()
                    reduced = True
                    break
        if not reduced:
            for group in reversed(payload["candidate_groups"]):
                if group["web_sources"]:
                    group["web_sources"].pop()
                    reduced = True
                    break
        serialized = serialize()
        input_token_count = count_text_tokens(serialized, model)
        if not reduced:
            record_diagnostics(serialized, input_token_count)
            raise EvidenceJudgeError("judge_context_too_large", attempts=0)
    record_diagnostics(serialized, input_token_count)
    return serialized


def _source_ids(
    pool: EvidencePool,
    serialized_payload: str,
) -> set[str]:
    allowed = {
        source.stable_id
        for item in pool.evidence.values()
        for source in item.source_refs
    }
    payload = json.loads(serialized_payload)
    for group in payload["candidate_groups"]:
        allowed.update(
            candidate["source_id"] for candidate in group["local_candidates"]
        )
        allowed.update(source["source_id"] for source in group["web_sources"])
    return allowed


def _validate_result(
    payload: dict[str, Any],
    pool: EvidencePool,
    allowed_source_ids: set[str],
) -> EvidenceJudgeResult:
    result = EvidenceJudgeResult.model_validate(payload)
    known_claim_ids = {claim.claim_id for claim in pool.claims}
    assessed_ids = {assessment.claim_id for assessment in result.claim_assessments}
    if assessed_ids != known_claim_ids:
        raise EvidenceJudgeError("incomplete_claim_assessment")
    for decision in result.decisions:
        if any(source.stable_id not in allowed_source_ids for source in decision.source_refs):
            raise EvidenceJudgeError("unknown_source_ref")
        if not set(decision.supports_claim_ids).issubset(known_claim_ids):
            raise EvidenceJudgeError("unknown_claim_id")
        if (
            decision.target_evidence_id is not None
            and decision.target_evidence_id not in pool.evidence
        ):
            raise EvidenceJudgeError("unknown_target_evidence_id")
    assessment_by_id = {
        assessment.claim_id: assessment for assessment in result.claim_assessments
    }
    if result.overall_sufficient and any(
        claim.required and assessment_by_id[claim.claim_id].status != "covered"
        for claim in pool.claims
    ):
        raise EvidenceJudgeError("false_sufficiency")
    return result


def judge_evidence_round(
    question: str,
    pool: EvidencePool,
    candidate_groups: list[EvidenceCandidateGroup],
    *,
    round_number: int,
    diagnostics: dict[str, Any] | None = None,
) -> EvidenceJudgeResult:
    settings = get_settings()
    model = settings.resolved_agent_model or "unknown"
    user_payload = _serialize_bounded_payload(
        question,
        pool,
        candidate_groups,
        round_number=round_number,
        token_budget=settings.agent_judge_context_tokens,
        model=model,
        diagnostics=diagnostics,
    )
    base_messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]
    messages = list(base_messages)
    allowed_source_ids = _source_ids(pool, user_payload)
    last_category = "judge_schema_error"
    last_provider_category: str | None = None
    last_validation_errors: list[dict[str, str]] = []
    for attempt in range(1, 3):
        try:
            payload = agent_json(messages, max_tokens=_JUDGE_MAX_OUTPUT_TOKENS)
            return _validate_result(payload, pool, allowed_source_ids)
        except ModelClientError as exc:
            if exc.category not in {"invalid_json", "empty_response"}:
                raise EvidenceJudgeError(
                    "judge_model_error",
                    attempts=attempt,
                    provider_category=exc.category,
                ) from exc
            last_category = "judge_schema_error"
            last_provider_category = exc.category
            last_validation_errors = [{"loc": "response", "type": exc.category}]
        except EvidenceJudgeError as exc:
            last_category = exc.category
            last_provider_category = exc.provider_category
            last_validation_errors = exc.validation_errors or [
                {"loc": "result", "type": exc.category}
            ]
        except ValidationError as exc:
            last_category = "judge_schema_error"
            last_provider_category = None
            last_validation_errors = _bounded_validation_errors(exc)
        except (TypeError, ValueError) as exc:
            last_category = "judge_schema_error"
            last_provider_category = None
            last_validation_errors = [
                {"loc": "result", "type": type(exc).__name__}
            ]

        if attempt == 1:
            messages = [
                *base_messages,
                {
                    "role": "user",
                    "content": (
                        "The previous Evidence Judge response failed validation with "
                        f"category '{last_category}'. Return a corrected JSON object "
                        "using only the supplied claim and source IDs. Validation "
                        f"details: {json.dumps(last_validation_errors, separators=(',', ':'))}."
                    ),
                },
            ]
    raise EvidenceJudgeError(
        last_category,
        attempts=2,
        provider_category=last_provider_category,
        validation_errors=last_validation_errors,
    )
