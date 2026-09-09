from collections.abc import Callable

import pytest
from backend.app.resilience.circuit_breaker import (
    CircuitBreakerPolicy,
    CircuitKey,
    HealthRegistry,
)
from backend.app.resilience.executor import AttemptContext, ResilientExecutor
from backend.app.resilience.models import (
    ExecutionBudget,
    FailureInfo,
    RequestContext,
)
from backend.app.resilience.retry import RetryPolicy


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _executor(
    clock: FakeClock,
    *,
    registry: HealthRegistry | None = None,
    random_value: Callable[[], float] = lambda: 1.0,
) -> ResilientExecutor:
    return ResilientExecutor(
        registry=registry,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        random_value=random_value,
    )


def _context(*, timeout: float = 10.0, retries: int = 2) -> RequestContext:
    return RequestContext(
        budget=ExecutionBudget(total_timeout_seconds=timeout, max_retries=retries),
        started_at_monotonic=0.0,
    )


def test_transient_failure_retries_then_returns_success() -> None:
    clock = FakeClock()
    executor = _executor(clock)
    calls = 0

    def operation(attempt: AttemptContext) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("private detail")
        return f"ok-{attempt.attempt}"

    result = executor.execute(
        operation,
        context=_context(),
        component="agent",
        component_timeout_seconds=5.0,
    )

    assert result.status == "success"
    assert result.value == "ok-2"
    assert result.attempts == 2
    assert clock.sleeps == [0.5]


def test_retry_after_is_honored() -> None:
    clock = FakeClock()
    executor = _executor(clock, random_value=lambda: 0.0)
    calls = 0

    class RateLimitedError(TimeoutError):
        retry_after_seconds = 2.0

    def operation(_: AttemptContext) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RateLimitedError("rate limited")
        return "ok"

    result = executor.execute(
        operation,
        context=_context(),
        component="judge",
        component_timeout_seconds=5.0,
    )

    assert result.status == "success"
    assert clock.sleeps == [2.0]


def test_two_executions_share_one_retry_budget() -> None:
    clock = FakeClock()
    executor = _executor(clock, random_value=lambda: 0.0)
    context = _context(retries=1)
    calls = 0

    def always_timeout(_: AttemptContext) -> None:
        nonlocal calls
        calls += 1
        raise TimeoutError("offline")

    first = executor.execute(
        always_timeout,
        context=context,
        component="first",
        component_timeout_seconds=1.0,
    )
    second = executor.execute(
        always_timeout,
        context=context,
        component="second",
        component_timeout_seconds=1.0,
    )

    assert first.attempts == 2
    assert second.attempts == 1
    assert second.failure is not None
    assert second.failure.code == "retry_budget_exhausted"
    assert context.usage.retries == 1
    assert calls == 3


@pytest.mark.parametrize(
    ("failure", "exception"),
    [
        (None, ValueError("bad argument")),
        (None, RuntimeError("broken")),
        (
            FailureInfo(
                category="semantic",
                code="insufficient_evidence",
                retryable=False,
            ),
            LookupError("semantic"),
        ),
    ],
)
def test_non_transient_failure_does_not_retry(
    failure: FailureInfo | None,
    exception: Exception,
) -> None:
    clock = FakeClock()
    executor = _executor(clock)
    calls = 0

    def operation(_: AttemptContext) -> None:
        nonlocal calls
        calls += 1
        raise exception

    result = executor.execute(
        operation,
        context=_context(),
        component="planner",
        component_timeout_seconds=1.0,
        classifier=(lambda _: failure) if failure is not None else None,
    )

    assert result.status == "failed"
    assert result.attempts == 1
    assert calls == 1
    assert clock.sleeps == []


def test_deadline_prevents_retry_and_sleep() -> None:
    clock = FakeClock()
    executor = _executor(clock)

    class SlowRetryError(TimeoutError):
        retry_after_seconds = 1.0

    result = executor.execute(
        lambda _: (_ for _ in ()).throw(SlowRetryError("offline")),
        context=_context(timeout=1.0),
        component="vector",
        component_timeout_seconds=1.0,
    )

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "deadline_exceeded"
    assert result.attempts == 1
    assert clock.sleeps == []


def test_open_circuit_skips_the_operation() -> None:
    clock = FakeClock(now=5.0)
    registry = HealthRegistry(CircuitBreakerPolicy(failure_threshold=1))
    permit = registry.acquire_call(
        CircuitKey("agent", "provider-a", "model-a"), now=4.0
    )
    assert permit is not None
    registry.record_failure(
        CircuitKey("agent", "provider-a", "model-a"),
        FailureInfo(category="transient", code="timeout", retryable=True),
        permit=permit,
        now=4.0,
    )
    calls = 0

    def operation(_: AttemptContext) -> str:
        nonlocal calls
        calls += 1
        return "must not run"

    result = _executor(clock, registry=registry).execute(
        operation,
        context=RequestContext(started_at_monotonic=5.0),
        component="agent",
        provider="provider-a",
        endpoint="model-a",
        component_timeout_seconds=5.0,
    )

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "circuit_open"
    assert result.attempts == 0
    assert calls == 0


def test_operation_receives_smaller_component_or_request_timeout() -> None:
    clock = FakeClock(now=3.0)
    executor = _executor(clock)
    received: list[float] = []

    executor.execute(
        lambda attempt: received.append(attempt.timeout_seconds),
        context=_context(timeout=10.0),
        component="local",
        component_timeout_seconds=5.0,
    )
    executor.execute(
        lambda attempt: received.append(attempt.timeout_seconds),
        context=_context(timeout=4.0),
        component="remote",
        component_timeout_seconds=5.0,
    )

    assert received == [5.0, 1.0]


def test_transient_failure_opens_registry_and_probe_success_closes_it() -> None:
    clock = FakeClock()
    registry = HealthRegistry(
        CircuitBreakerPolicy(failure_threshold=1, open_seconds=30.0)
    )
    executor = _executor(clock, registry=registry)
    key = CircuitKey("agent", "provider-a", "model-a")

    failed = executor.execute(
        lambda _: (_ for _ in ()).throw(TimeoutError("offline")),
        context=_context(retries=0),
        component=key.component,
        provider=key.provider,
        endpoint=key.endpoint,
        component_timeout_seconds=5.0,
        retry_policy=RetryPolicy(max_retries=0),
    )
    assert failed.status == "failed"
    assert registry.state(key, now=0.0) == "open"

    clock.now = 30.0
    succeeded = executor.execute(
        lambda _: "ok",
        context=RequestContext(started_at_monotonic=30.0),
        component=key.component,
        provider=key.provider,
        endpoint=key.endpoint,
        component_timeout_seconds=5.0,
    )

    assert succeeded.status == "success"
    assert registry.state(key, now=30.0) == "closed"


def test_failed_half_open_probe_does_not_sleep_or_consume_retry() -> None:
    clock = FakeClock()
    registry = HealthRegistry(CircuitBreakerPolicy(failure_threshold=1))
    key = CircuitKey("agent", "provider-a", "model-a")
    permit = registry.acquire_call(key, now=0.0)
    assert permit is not None
    registry.record_failure(
        key,
        FailureInfo(category="transient", code="timeout", retryable=True),
        permit=permit,
        now=0.0,
    )
    clock.now = 30.0
    context = RequestContext(
        budget=ExecutionBudget(total_timeout_seconds=10.0, max_retries=2),
        started_at_monotonic=30.0,
    )

    result = _executor(clock, registry=registry).execute(
        lambda _: (_ for _ in ()).throw(TimeoutError("probe failed")),
        context=context,
        component=key.component,
        provider=key.provider,
        endpoint=key.endpoint,
        component_timeout_seconds=5.0,
    )

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "timeout"
    assert result.attempts == 1
    assert context.usage.retries == 0
    assert clock.sleeps == []
