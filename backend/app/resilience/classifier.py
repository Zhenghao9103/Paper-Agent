import sqlite3

from pydantic import ValidationError

from ..services.model_clients import ModelClientError
from .models import FailureInfo

_TRANSIENT_PROVIDER_CODES = {
    "connection_error",
    "rate_limit",
    "server_error",
    "timeout",
}


def classify_exception(exc: Exception) -> FailureInfo:
    if isinstance(exc, ValidationError):
        return FailureInfo(
            category="validation", code="schema_invalid", retryable=False
        )
    if isinstance(exc, ModelClientError):
        code = exc.category or "model_error"
        return FailureInfo(
            category="transient" if code in _TRANSIENT_PROVIDER_CODES else "permanent",
            code=code,
            retryable=code in _TRANSIENT_PROVIDER_CODES,
        )
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return FailureInfo(category="transient", code="timeout", retryable=True)
    if isinstance(exc, sqlite3.OperationalError):
        normalized = str(exc).casefold()
        if "locked" in normalized or "busy" in normalized:
            return FailureInfo(
                category="transient", code="database_busy", retryable=True
            )
        return FailureInfo(
            category="permanent", code="database_error", retryable=False
        )
    if isinstance(exc, ValueError):
        return FailureInfo(
            category="validation", code="invalid_argument", retryable=False
        )
    return FailureInfo(
        category="permanent", code="internal_error", retryable=False
    )
