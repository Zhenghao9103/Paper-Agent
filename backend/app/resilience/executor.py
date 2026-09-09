from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from .circuit_breaker import CircuitKey, HealthRegistry
from .classifier import classify_exception
from .models import ExecutionResult, FailureInfo, RequestContext
from .retry import RetryPolicy, parse_retry_after

T = TypeVar("T")


@dataclass(frozen=True)
class AttemptContext:
    attempt: int
    timeout_seconds: float


class ResilientExecutor:
    def __init__(
        self,
        *,
        registry: HealthRegistry | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self.registry = registry or HealthRegistry()
        self.monotonic = monotonic
        self.sleep = sleep
        self.random_value = random_value

    def execute(
        self,
        operation: Callable[[AttemptContext], T],
        *,
        context: RequestContext,
        component: str,
        provider: str = "local",
        endpoint: str = "default",
        component_timeout_seconds: float,
        retry_policy: RetryPolicy | None = None,
        retry_after: Callable[[Exception], float | None] | None = None,
        classifier: Callable[[Exception], FailureInfo] | None = None,
    ) -> ExecutionResult:
        if component_timeout_seconds <= 0:
            raise ValueError("component_timeout_seconds must be positive")
        policy = retry_policy or RetryPolicy()
        failure_classifier = classifier or classify_exception
        retry_after_extractor = retry_after or _retry_after_from_exception
        key = CircuitKey(component, provider, endpoint)
        attempts = 0
        retries = 0

        while True:
            now = self.monotonic()
            remaining = context.remaining_seconds(now)
            if remaining <= 0:
                return ExecutionResult.failed(
                    _deadline_failure(),
                    attempts=attempts,
                )
            permit = self.registry.acquire_call(key, now=now)
            if permit is None:
                return ExecutionResult.failed(
                    FailureInfo(
                        category="transient",
                        code="circuit_open",
                        retryable=True,
                    ),
                    attempts=attempts,
                )

            attempts += 1
            attempt_context = AttemptContext(
                attempt=attempts,
                timeout_seconds=min(component_timeout_seconds, remaining),
            )
            try:
                value = operation(attempt_context)
            except Exception as exc:
                failure = failure_classifier(exc)
                self.registry.record_failure(
                    key,
                    failure,
                    permit=permit,
                    now=self.monotonic(),
                )
                if failure.category != "transient" or not failure.retryable:
                    return ExecutionResult.failed(failure, attempts=attempts)
                if retries >= policy.max_retries:
                    return ExecutionResult.failed(failure, attempts=attempts)
                if self.registry.state(key, now=self.monotonic()) != "closed":
                    return ExecutionResult.failed(failure, attempts=attempts)

                remaining = context.remaining_seconds(self.monotonic())
                delay = policy.delay_seconds(
                    retries,
                    retry_after=retry_after_extractor(exc),
                    remaining_seconds=remaining,
                    random_value=self.random_value,
                )
                if delay is None:
                    return ExecutionResult.failed(
                        _deadline_failure(),
                        attempts=attempts,
                    )
                if not context.reserve_retry():
                    return ExecutionResult.failed(
                        FailureInfo(
                            category="transient",
                            code="retry_budget_exhausted",
                            retryable=False,
                        ),
                        attempts=attempts,
                    )
                retries += 1
                self.sleep(delay)
                continue

            self.registry.record_success(key, permit=permit)
            return ExecutionResult.success(value, attempts=attempts)


def _deadline_failure() -> FailureInfo:
    return FailureInfo(
        category="transient",
        code="deadline_exceeded",
        retryable=False,
    )


def _retry_after_from_exception(exc: Exception) -> float | None:
    for attribute in ("retry_after_seconds", "retry_after"):
        value = getattr(exc, attribute, None)
        parsed = parse_retry_after(value)
        if parsed is not None:
            return parsed
    response = getattr(exc, "response", None)
    headers: Any = getattr(response, "headers", None)
    if headers is not None:
        return parse_retry_after(headers.get("Retry-After"))
    return None
