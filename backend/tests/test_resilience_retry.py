from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest
from backend.app.resilience.retry import RetryPolicy, parse_retry_after


def test_parse_retry_after_accepts_non_negative_seconds() -> None:
    assert parse_retry_after("2.5") == 2.5
    assert parse_retry_after(0) == 0.0


def test_parse_retry_after_accepts_future_http_date() -> None:
    now = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
    value = format_datetime(now + timedelta(seconds=5), usegmt=True)

    assert parse_retry_after(value, now=now) == 5.0


@pytest.mark.parametrize("value", [-1, "-2", "not-a-delay", None])
def test_parse_retry_after_rejects_invalid_values(value: object) -> None:
    assert parse_retry_after(value) is None


def test_full_jitter_uses_exponential_ceiling() -> None:
    policy = RetryPolicy(base_seconds=0.5, cap_seconds=8.0)

    delay = policy.delay_seconds(
        1,
        retry_after=None,
        remaining_seconds=10.0,
        random_value=lambda: 0.25,
    )

    assert delay == 0.25


def test_full_jitter_caps_the_exponential_ceiling() -> None:
    policy = RetryPolicy(base_seconds=0.5, cap_seconds=8.0)

    delay = policy.delay_seconds(
        10,
        retry_after=None,
        remaining_seconds=10.0,
        random_value=lambda: 0.5,
    )

    assert delay == 4.0


def test_retry_after_takes_precedence_without_jitter() -> None:
    delay = RetryPolicy().delay_seconds(
        0,
        retry_after=3.0,
        remaining_seconds=5.0,
        random_value=lambda: 0.0,
    )

    assert delay == 3.0


def test_delay_returns_none_when_it_would_exhaust_deadline() -> None:
    policy = RetryPolicy()

    assert (
        policy.delay_seconds(
            0,
            retry_after=3.0,
            remaining_seconds=3.0,
            random_value=lambda: 0.0,
        )
        is None
    )
