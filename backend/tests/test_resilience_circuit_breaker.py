from backend.app.resilience.circuit_breaker import (
    CircuitBreakerPolicy,
    CircuitKey,
    HealthRegistry,
)
from backend.app.resilience.models import FailureInfo

KEY = CircuitKey(component="agent", provider="provider-a", endpoint="model-a")
TRANSIENT = FailureInfo(category="transient", code="timeout", retryable=True)


def _record_failure(
    registry: HealthRegistry,
    failure: FailureInfo,
    *,
    now: float,
) -> None:
    permit = registry.acquire_call(KEY, now=now)
    assert permit is not None
    registry.record_failure(KEY, failure, permit=permit, now=now)


def _open_registry(registry: HealthRegistry) -> None:
    for now in range(5):
        _record_failure(registry, TRANSIENT, now=float(now))


def test_five_transient_failures_in_window_open_the_circuit() -> None:
    registry = HealthRegistry()

    _open_registry(registry)

    assert registry.state(KEY, now=4.0) == "open"
    assert registry.acquire_call(KEY, now=4.0) is None


def test_failures_outside_window_do_not_open_the_circuit() -> None:
    registry = HealthRegistry()
    for now in (0.0, 1.0, 2.0, 3.0):
        _record_failure(registry, TRANSIENT, now=now)
    _record_failure(registry, TRANSIENT, now=65.0)

    assert registry.state(KEY, now=65.0) == "closed"


def test_half_open_allows_exactly_one_probe() -> None:
    registry = HealthRegistry()
    _open_registry(registry)

    assert registry.acquire_call(KEY, now=34.0) is not None
    assert registry.state(KEY, now=34.0) == "half_open"
    assert registry.acquire_call(KEY, now=34.0) is None


def test_successful_half_open_probe_closes_the_circuit() -> None:
    registry = HealthRegistry()
    _open_registry(registry)
    probe = registry.acquire_call(KEY, now=34.0)
    assert probe is not None

    registry.record_success(KEY, permit=probe)

    assert registry.state(KEY, now=34.0) == "closed"
    assert registry.acquire_call(KEY, now=34.0) is not None


def test_failed_half_open_probe_reopens_for_another_open_period() -> None:
    registry = HealthRegistry()
    _open_registry(registry)
    probe = registry.acquire_call(KEY, now=34.0)
    assert probe is not None

    registry.record_failure(KEY, TRANSIENT, permit=probe, now=35.0)

    assert registry.state(KEY, now=64.0) == "open"
    assert registry.acquire_call(KEY, now=64.0) is None
    assert registry.acquire_call(KEY, now=65.0) is not None


def test_validation_and_semantic_failures_do_not_count() -> None:
    registry = HealthRegistry()
    validation = FailureInfo(
        category="validation", code="schema_invalid", retryable=False
    )
    semantic = FailureInfo(
        category="semantic", code="insufficient_evidence", retryable=False
    )

    for _ in range(10):
        _record_failure(registry, validation, now=1.0)
        _record_failure(registry, semantic, now=1.0)

    assert registry.state(KEY, now=1.0) == "closed"


def test_stale_success_cannot_close_a_newly_opened_circuit() -> None:
    registry = HealthRegistry(CircuitBreakerPolicy(failure_threshold=1))
    stale_permit = registry.acquire_call(KEY, now=0.0)
    failed_permit = registry.acquire_call(KEY, now=0.0)
    assert stale_permit is not None
    assert failed_permit is not None

    registry.record_failure(KEY, TRANSIENT, permit=failed_permit, now=1.0)
    registry.record_success(KEY, permit=stale_permit)

    assert registry.state(KEY, now=1.0) == "open"


def test_out_of_order_failure_completion_preserves_window_order() -> None:
    registry = HealthRegistry(CircuitBreakerPolicy(failure_threshold=3))
    later_permit = registry.acquire_call(KEY, now=0.0)
    earlier_permit = registry.acquire_call(KEY, now=0.0)
    assert later_permit is not None
    assert earlier_permit is not None
    registry.record_failure(KEY, TRANSIENT, permit=later_permit, now=11.0)
    registry.record_failure(KEY, TRANSIENT, permit=earlier_permit, now=10.0)
    current_permit = registry.acquire_call(KEY, now=70.5)
    assert current_permit is not None

    registry.record_failure(KEY, TRANSIENT, permit=current_permit, now=70.5)

    assert registry.state(KEY, now=70.5) == "closed"


def test_out_of_order_threshold_uses_latest_failure_as_open_time() -> None:
    registry = HealthRegistry(CircuitBreakerPolicy(failure_threshold=3))
    permits = [registry.acquire_call(KEY, now=0.0) for _ in range(3)]
    assert all(permit is not None for permit in permits)
    for permit, now in zip(permits, (11.0, 10.0, 9.0), strict=True):
        assert permit is not None
        registry.record_failure(KEY, TRANSIENT, permit=permit, now=now)

    assert registry.state(KEY, now=39.0) == "open"
    assert registry.state(KEY, now=41.0) == "half_open"


def test_non_transient_half_open_failure_releases_without_closing() -> None:
    registry = HealthRegistry(CircuitBreakerPolicy(failure_threshold=1))
    initial = registry.acquire_call(KEY, now=0.0)
    assert initial is not None
    registry.record_failure(KEY, TRANSIENT, permit=initial, now=0.0)
    probe = registry.acquire_call(KEY, now=30.0)
    assert probe is not None
    validation = FailureInfo(
        category="validation", code="schema_invalid", retryable=False
    )

    registry.record_failure(KEY, validation, permit=probe, now=31.0)

    assert registry.state(KEY, now=31.0) == "half_open"
    assert registry.acquire_call(KEY, now=31.0) is not None
