from __future__ import annotations

from bisect import insort
from dataclasses import dataclass, field
from threading import RLock
from typing import Literal

from .models import FailureInfo

CircuitState = Literal["closed", "open", "half_open"]


@dataclass(frozen=True)
class CircuitKey:
    component: str
    provider: str
    endpoint: str


@dataclass(frozen=True)
class CircuitPermit:
    generation: int
    half_open_probe: bool = False


@dataclass(frozen=True)
class CircuitBreakerPolicy:
    window_seconds: float = 60.0
    failure_threshold: int = 5
    open_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if self.open_seconds <= 0:
            raise ValueError("open_seconds must be positive")


@dataclass
class _CircuitRecord:
    state: CircuitState = "closed"
    failures: list[float] = field(default_factory=list)
    opened_at: float | None = None
    probe_in_flight: bool = False
    generation: int = 0


class HealthRegistry:
    def __init__(self, policy: CircuitBreakerPolicy | None = None) -> None:
        self.policy = policy or CircuitBreakerPolicy()
        self._records: dict[CircuitKey, _CircuitRecord] = {}
        self._lock = RLock()

    def acquire_call(self, key: CircuitKey, *, now: float) -> CircuitPermit | None:
        with self._lock:
            record = self._record(key)
            if record.state == "closed":
                self._purge_old_failures(record, now)
                return CircuitPermit(generation=record.generation)
            if record.state == "open":
                if (
                    record.opened_at is None
                    or now - record.opened_at < self.policy.open_seconds
                ):
                    return None
                record.state = "half_open"
                record.probe_in_flight = True
                return CircuitPermit(
                    generation=record.generation,
                    half_open_probe=True,
                )
            if record.probe_in_flight:
                return None
            record.probe_in_flight = True
            return CircuitPermit(
                generation=record.generation,
                half_open_probe=True,
            )

    def record_success(self, key: CircuitKey, *, permit: CircuitPermit) -> None:
        with self._lock:
            record = self._record(key)
            if permit.generation != record.generation:
                return
            record.state = "closed"
            record.failures.clear()
            record.opened_at = None
            record.probe_in_flight = False
            if permit.half_open_probe:
                record.generation += 1

    def record_failure(
        self,
        key: CircuitKey,
        failure: FailureInfo,
        *,
        permit: CircuitPermit,
        now: float,
    ) -> None:
        with self._lock:
            record = self._record(key)
            if permit.generation != record.generation:
                return
            if failure.category != "transient":
                if permit.half_open_probe:
                    record.probe_in_flight = False
                return
            if permit.half_open_probe:
                self._open(record, now)
                return
            if record.state == "open":
                return
            self._purge_old_failures(record, now)
            insort(record.failures, now)
            if len(record.failures) >= self.policy.failure_threshold:
                self._open(record, record.failures[-1])

    def state(self, key: CircuitKey, *, now: float) -> CircuitState:
        with self._lock:
            record = self._record(key)
            if (
                record.state == "open"
                and record.opened_at is not None
                and now - record.opened_at >= self.policy.open_seconds
            ):
                return "half_open"
            if record.state == "closed":
                self._purge_old_failures(record, now)
            return record.state

    def _record(self, key: CircuitKey) -> _CircuitRecord:
        return self._records.setdefault(key, _CircuitRecord())

    def _purge_old_failures(self, record: _CircuitRecord, now: float) -> None:
        cutoff = now - self.policy.window_seconds
        first_current = 0
        while (
            first_current < len(record.failures)
            and record.failures[first_current] < cutoff
        ):
            first_current += 1
        if first_current:
            del record.failures[:first_current]

    @staticmethod
    def _open(record: _CircuitRecord, now: float) -> None:
        record.state = "open"
        record.opened_at = now
        record.probe_in_flight = False
        record.generation += 1
