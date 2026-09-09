import pytest
from backend.app.resilience.models import (
    ExecutionBudget,
    ExecutionResult,
    FailureInfo,
    RequestContext,
)
from pydantic import ValidationError


def test_success_result_cannot_include_failure() -> None:
    with pytest.raises(ValidationError):
        ExecutionResult(
            status="success",
            value={"items": []},
            failure=FailureInfo(
                category="transient",
                code="timeout",
                retryable=True,
            ),
        )


def test_failed_result_requires_failure() -> None:
    with pytest.raises(ValidationError):
        ExecutionResult(status="failed", value=None)


def test_only_failed_result_can_have_zero_attempts() -> None:
    with pytest.raises(ValidationError):
        ExecutionResult.success({"ok": True}, attempts=0)

    result = ExecutionResult.failed(
        FailureInfo(category="transient", code="circuit_open", retryable=True),
        attempts=0,
    )
    assert result.attempts == 0


def test_degraded_result_records_fallback_lineage() -> None:
    result = ExecutionResult.degraded(
        {"items": [1]},
        failure=FailureInfo(
            category="transient",
            code="vector_timeout",
            retryable=True,
        ),
        fallback_from="vector",
        fallback_to="bm25",
    )
    assert result.status == "degraded"
    assert result.fallback_from == "vector"
    assert result.fallback_to == "bm25"


def test_request_context_defaults_to_bounded_production_budget() -> None:
    context = RequestContext()
    assert context.mode == "production"
    assert context.request_id
    assert context.budget == ExecutionBudget()
    assert context.budget.max_agent_rounds == 3
    assert context.budget.max_tool_calls == 5


def test_request_context_shares_one_retry_budget_across_consumers() -> None:
    context = RequestContext(
        budget=ExecutionBudget(max_retries=1),
        started_at_monotonic=10.0,
    )
    first_consumer = context
    second_consumer = context

    assert first_consumer.reserve_retry() is True
    assert second_consumer.reserve_retry() is False
    assert context.usage.retries == 1


def test_request_context_remaining_time_never_becomes_negative() -> None:
    context = RequestContext(
        budget=ExecutionBudget(total_timeout_seconds=5.0),
        started_at_monotonic=10.0,
    )

    assert context.remaining_seconds(12.0) == 3.0
    assert context.remaining_seconds(20.0) == 0.0


def test_request_context_rejects_reservations_past_each_limit() -> None:
    context = RequestContext(
        budget=ExecutionBudget(
            max_agent_rounds=1,
            max_tool_calls=1,
            max_fallbacks=1,
        )
    )

    for reserve in (
        context.reserve_agent_round,
        context.reserve_tool_call,
        context.reserve_fallback,
    ):
        assert reserve() is True
        assert reserve() is False
