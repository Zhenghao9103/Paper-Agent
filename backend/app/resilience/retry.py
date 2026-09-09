import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    base_seconds: float = 0.5
    cap_seconds: float = 8.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.base_seconds < 0:
            raise ValueError("base_seconds must be non-negative")
        if self.cap_seconds <= 0:
            raise ValueError("cap_seconds must be positive")

    def delay_seconds(
        self,
        retry_index: int,
        *,
        retry_after: float | None,
        remaining_seconds: float,
        random_value: Callable[[], float],
    ) -> float | None:
        if retry_index < 0:
            raise ValueError("retry_index must be non-negative")
        if remaining_seconds <= 0:
            return None
        if retry_after is not None:
            ceiling = retry_after
            delay = retry_after
        else:
            ceiling = min(self.cap_seconds, self.base_seconds * (2**retry_index))
            delay = random_value() * ceiling
        if not math.isfinite(delay) or delay < 0 or delay >= remaining_seconds:
            return None
        return delay


def parse_retry_after(
    value: object,
    *,
    now: datetime | None = None,
) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = None
    if seconds is not None:
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    if not isinstance(value, str):
        return None
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    delay = (retry_at - current).total_seconds()
    return delay if math.isfinite(delay) and delay >= 0 else None
