import json

import pytest
from backend.app.services import model_clients, query_planner
from backend.app.services.query_planner import (
    fallback_plan,
    plan_queries,
    planner_json,
)


def _planner_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "standalone_query": "SpecNet2 如何避免正交化？",
        "lexical_terms": ["SpecNet2", "正交化"],
        "synonyms": ["orthogonality constraint"],
        "semantic_queries": [
            "SpecNet2 如何避免正交化？",
            "SpecNet2 orthogonalization avoidance",
        ],
    }
    payload.update(overrides)
    return payload


def test_plan_queries_builds_plan_from_model_output(monkeypatch) -> None:
    captured: list[list[dict[str, str]]] = []

    def fake_planner(messages):
        captured.append(messages)
        return _planner_payload()

    monkeypatch.setattr(query_planner, "planner_json", fake_planner)

    outcome = plan_queries("它如何避免正交化？", document_id=7)

    assert outcome.degraded is False
    assert outcome.error is None
    plan = outcome.plan
    assert plan.intent == "simple_rag"
    assert plan.document_id == 7
    assert plan.standalone_query == "SpecNet2 如何避免正交化？"
    assert plan.lexical_terms == ["SpecNet2", "正交化"]
    assert plan.synonyms == ["orthogonality constraint"]
    assert plan.semantic_queries[0] == "SpecNet2 如何避免正交化？"
    assert len(plan.semantic_queries) <= 3
    user_payload = json.loads(captured[0][1]["content"])
    assert set(user_payload) == {"session_context", "current_question"}


def test_planner_prompt_hides_document_scope_and_gold_answers(monkeypatch) -> None:
    captured: list[list[dict[str, str]]] = []

    def fake_planner(messages):
        captured.append(messages)
        return _planner_payload()

    monkeypatch.setattr(query_planner, "planner_json", fake_planner)

    plan_queries("它如何避免正交化？", document_id=7)

    system = captured[0][0]["content"]
    assert "standalone_query" in system
    assert "lexical_terms" in system
    assert "semantic_queries" in system
    assert "document_id" not in captured[0][1]["content"]


def test_plan_queries_injects_request_document_id_over_model_value(monkeypatch) -> None:
    monkeypatch.setattr(
        query_planner,
        "planner_json",
        lambda messages: _planner_payload(document_id=999),
    )

    outcome = plan_queries("它如何避免正交化？", document_id=7)

    assert outcome.plan.document_id == 7


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"standalone_query": ""},
        {"standalone_query": "   "},
        {"semantic_queries": ["unused because standalone is missing"]},
    ],
)
def test_plan_queries_falls_back_on_invalid_payloads(monkeypatch, payload) -> None:
    monkeypatch.setattr(query_planner, "planner_json", lambda messages: payload)

    outcome = plan_queries("original question", document_id=8)

    assert outcome.degraded is True
    assert outcome.error
    assert outcome.plan.standalone_query == "original question"
    assert outcome.plan.semantic_queries == ["original question"]
    assert outcome.plan.document_id == 8


def test_plan_queries_tolerates_garbage_secondary_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        query_planner,
        "planner_json",
        lambda messages: {
            "standalone_query": "query",
            "lexical_terms": "not-a-list",
            "synonyms": {"nested": True},
            "semantic_queries": [42, ""],
        },
    )

    outcome = plan_queries("original question", document_id=8)

    assert outcome.degraded is False
    assert outcome.plan.semantic_queries == ["query", "42"]
    assert outcome.plan.lexical_terms == []


def test_plan_queries_falls_back_on_timeout(monkeypatch) -> None:
    def fail(messages):
        raise TimeoutError("planner call timed out")

    monkeypatch.setattr(query_planner, "planner_json", fail)

    outcome = plan_queries("original question", document_id=8)

    assert outcome.degraded is True
    assert outcome.error == "Query planner timeout."
    assert outcome.plan.semantic_queries == ["original question"]


def test_plan_queries_falls_back_on_model_configuration_error(monkeypatch) -> None:
    def fail(messages):
        raise model_clients.ModelConfigurationError(
            "Agent model is not configured: AGENT_API_KEY."
        )

    monkeypatch.setattr(query_planner, "planner_json", fail)

    outcome = plan_queries("original question", document_id=None)

    assert outcome.degraded is True
    assert outcome.error == "Query planner configuration error."
    assert outcome.plan.document_id is None


def test_plan_queries_bounds_oversized_model_fields(monkeypatch) -> None:
    huge = "z" * 100_000
    monkeypatch.setattr(
        query_planner,
        "planner_json",
        lambda messages: _planner_payload(
            standalone_query=huge,
            lexical_terms=[huge],
            synonyms=[huge],
            semantic_queries=[huge, "second", "third", "fourth"],
        ),
    )

    outcome = plan_queries("bounded question", document_id=8)

    plan = outcome.plan
    assert len(plan.standalone_query) <= 1000
    assert len(plan.lexical_terms[0]) <= 256
    assert len(plan.synonyms[0]) <= 256
    assert len(plan.semantic_queries) <= 3
    assert all(len(query) <= 1000 for query in plan.semantic_queries)


def test_plan_queries_deduplicates_and_keeps_standalone_first(monkeypatch) -> None:
    monkeypatch.setattr(
        query_planner,
        "planner_json",
        lambda messages: _planner_payload(
            standalone_query="foo",
            semantic_queries=["foo", " bar ", "bar", "baz", "qux"],
            lexical_terms=["foo", "FOO", "bar"],
        ),
    )

    outcome = plan_queries("original", document_id=None)

    plan = outcome.plan
    assert plan.semantic_queries == ["foo", "bar", "baz"]
    assert plan.lexical_terms == ["foo", "bar"]


def test_fallback_plan_tokenizes_question_without_model_call() -> None:
    plan = fallback_plan("SpecNet2 的损失函数是什么？", document_id=3)

    assert plan.intent == "simple_rag"
    assert plan.confidence == 1.0
    assert plan.semantic_queries == ["SpecNet2 的损失函数是什么？"]
    assert plan.lexical_terms
    assert all(term for term in plan.lexical_terms)
    assert plan.document_id == 3


def test_planner_json_uses_agent_transport(monkeypatch) -> None:
    settings_stub = type(
        "S",
        (),
        {
            "resolved_agent_api_key": "agent-key",
            "resolved_agent_base_url": "https://agent.example/v1",
            "resolved_agent_model": "agent-model",
            "agent_timeout_seconds": 5.0,
        },
    )()
    calls: list[dict[str, object]] = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return _response('{"standalone_query":"q"}')

    class Client:
        chat = _Namespace(completions=Completions())

    monkeypatch.setattr(model_clients, "get_settings", lambda: settings_stub)
    monkeypatch.setattr(model_clients, "_agent_client", lambda: Client())

    payload = planner_json([{"role": "user", "content": "question"}])

    assert payload == {"standalone_query": "q"}
    assert calls[0]["model"] == "agent-model"
    assert calls[0]["response_format"] == {"type": "json_object"}


class _Namespace:
    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def _response(content: str) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
