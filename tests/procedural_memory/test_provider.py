from types import SimpleNamespace

import pytest

from backend.app.services import model_clients


def test_procedural_json_accepts_one_complete_fence(monkeypatch):
    content = ['```json\n{"action":"add"}\n```']

    def complete(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content[0]))]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=complete))
    )
    monkeypatch.setattr(model_clients, "_agent_client", lambda: client)
    assert model_clients.agent_json([], max_tokens=10, allow_json_fence=True) == {
        "action": "add"
    }
    with pytest.raises(model_clients.ModelClientError):
        model_clients.agent_json([], max_tokens=10)
    content[0] += " extra"
    with pytest.raises(model_clients.ModelClientError):
        model_clients.agent_json([], max_tokens=10, allow_json_fence=True)
