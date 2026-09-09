from types import SimpleNamespace

import pytest
from backend.app.services import model_clients


class _Completions:
    def __init__(self, contents: list[str] | None = None, errors: list[Exception] | None = None):
        self.contents = list(contents or [])
        self.errors = list(errors or [])
        self.requests: list[dict] = []

    def create(self, **request):
        self.requests.append(request)
        if self.errors:
            raise self.errors.pop(0)
        content = self.contents.pop(0)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _client(completions: _Completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_agent_json_uses_agent_model_and_strict_generation_settings(monkeypatch) -> None:
    completions = _Completions(contents=['{"claims": []}'])
    settings = SimpleNamespace(resolved_agent_model="agent-model")
    monkeypatch.setattr(model_clients, "get_settings", lambda: settings)
    monkeypatch.setattr(model_clients, "_agent_client", lambda: _client(completions))

    payload = model_clients.agent_json(
        [{"role": "user", "content": "plan"}],
        max_tokens=2048,
    )

    assert payload == {"claims": []}
    assert completions.requests == [
        {
            "model": "agent-model",
            "messages": [{"role": "user", "content": "plan"}],
            "temperature": 0,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
        }
    ]


def test_agent_json_retries_one_retryable_provider_failure(monkeypatch) -> None:
    class ConnectionFailure(Exception):
        pass

    completions = _Completions(
        contents=['{"ok": true}'],
        errors=[ConnectionFailure("secret provider detail")],
    )
    monkeypatch.setattr(
        model_clients,
        "get_settings",
        lambda: SimpleNamespace(resolved_agent_model="agent-model"),
    )
    monkeypatch.setattr(model_clients, "_agent_client", lambda: _client(completions))

    assert model_clients.agent_json([], max_tokens=32) == {"ok": True}
    assert len(completions.requests) == 2


def test_agent_json_reports_stable_invalid_json_without_raw_content(monkeypatch) -> None:
    completions = _Completions(contents=["not-json secret-token"])
    monkeypatch.setattr(
        model_clients,
        "get_settings",
        lambda: SimpleNamespace(resolved_agent_model="agent-model"),
    )
    monkeypatch.setattr(model_clients, "_agent_client", lambda: _client(completions))

    with pytest.raises(model_clients.ModelClientError) as exc_info:
        model_clients.agent_json([], max_tokens=32)

    assert exc_info.value.category == "invalid_json"
    assert "secret-token" not in str(exc_info.value)


def test_agent_json_rejects_empty_choices(monkeypatch) -> None:
    class EmptyCompletions:
        def create(self, **request):
            return SimpleNamespace(choices=[])

    monkeypatch.setattr(
        model_clients,
        "get_settings",
        lambda: SimpleNamespace(resolved_agent_model="agent-model"),
    )
    monkeypatch.setattr(
        model_clients,
        "_agent_client",
        lambda: _client(EmptyCompletions()),
    )

    with pytest.raises(model_clients.ModelClientError) as exc_info:
        model_clients.agent_json([], max_tokens=32)

    assert exc_info.value.category == "empty_response"
