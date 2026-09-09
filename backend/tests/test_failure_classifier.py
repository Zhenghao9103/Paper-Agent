import sqlite3

from backend.app.resilience.classifier import classify_exception
from backend.app.services.model_clients import ModelClientError
from pydantic import BaseModel, ValidationError


class _RequiredValue(BaseModel):
    value: int


def test_classifies_schema_validation_as_non_retryable() -> None:
    try:
        _RequiredValue.model_validate({})
    except ValidationError as exc:
        failure = classify_exception(exc)
    assert failure.model_dump() == {
        "category": "validation",
        "code": "schema_invalid",
        "retryable": False,
    }


def test_classifies_provider_timeout_as_transient() -> None:
    failure = classify_exception(
        ModelClientError("redacted", category="timeout", attempts=2)
    )
    assert failure.category == "transient"
    assert failure.code == "timeout"
    assert failure.retryable is True


def test_classifies_sqlite_lock_as_transient() -> None:
    failure = classify_exception(sqlite3.OperationalError("database is locked"))
    assert failure.category == "transient"
    assert failure.code == "database_busy"


def test_unknown_exception_is_non_retryable_internal_error() -> None:
    failure = classify_exception(RuntimeError("secret provider detail"))
    assert failure.model_dump() == {
        "category": "permanent",
        "code": "internal_error",
        "retryable": False,
    }
