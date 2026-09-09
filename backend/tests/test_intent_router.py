import json
from types import SimpleNamespace

import pytest
from backend.app.core.config import Settings
from backend.app.services import intent_router, model_clients
from backend.app.services.router_contract import (
    FALLBACK_INTENT,
    RouterDecision,
    RouterProtocolError,
    RoutingOutcome,
    parse_router_decision,
)


def _router_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {"intent": "simple_rag"}
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("intent", "question"),
    [
        ("direct", "How many documents are indexed?"),
        ("simple_rag", "What loss does SpecNet2 use?"),
        ("agentic_rag", "Compare SpecNet2 and SpectralNet and explain why."),
    ],
)
def test_route_question_accepts_each_supported_intent(monkeypatch, intent, question) -> None:
    monkeypatch.setattr(
        intent_router, "router_json", lambda messages: _router_payload(intent=intent)
    )

    outcome = intent_router.route_question(question, document_id=None)

    assert outcome == RoutingOutcome(intent=intent)
    assert outcome.degraded is False
    assert outcome.error is None


def test_route_question_uses_context_and_ignores_document_scope(monkeypatch) -> None:
    captured: list[list[dict[str, str]]] = []

    def fake_router(messages):
        captured.append(messages)
        return _router_payload(intent="simple_rag")

    monkeypatch.setattr(intent_router, "router_json", fake_router)

    outcome = intent_router.route_question(
        "它如何避免正交化？",
        document_id=7,
        short_term_memory={
            "summary": "正在讨论 SpecNet2",
            "recent_messages": [{"role": "user", "content": "SpecNet2 的损失是什么？"}],
        },
    )

    assert outcome.intent == "simple_rag"
    assert outcome.degraded is False
    user_payload = json.loads(captured[0][1]["content"])
    assert set(user_payload) == {"session_context", "current_question"}
    assert user_payload["session_context"][0]["content"].endswith("正在讨论 SpecNet2")
    assert user_payload["current_question"] == "它如何避免正交化？"


def test_route_question_falls_back_on_extra_fields(monkeypatch) -> None:
    monkeypatch.setattr(
        intent_router,
        "router_json",
        lambda messages: _router_payload(confidence=0.9, standalone_query="x"),
    )

    outcome = intent_router.route_question("What is SpecNet2?", document_id=4)

    assert outcome.degraded is True
    assert outcome.intent == FALLBACK_INTENT
    assert outcome.error == "Router response schema error."


def test_route_question_falls_back_on_invalid_enum(monkeypatch) -> None:
    monkeypatch.setattr(
        intent_router,
        "router_json",
        lambda messages: {"intent": "complex_search"},
    )

    outcome = intent_router.route_question("What is SpecNet2?", document_id=4)

    assert outcome.degraded is True
    assert outcome.intent == FALLBACK_INTENT
    assert outcome.error == "Router response schema error."


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"intent": "direct"}\n```',
        'The intent is {"intent": "direct"} as requested.',
        '{"intent": "direct", "reason": "operational"}',
        '{"confidence": 0.9, "intent": "direct"}',
        "",
        "   ",
        "null",
        "[1, 2, 3]",
        '{"intent": ""}',
    ],
)
def test_parse_router_decision_rejects_protocol_violations(raw) -> None:
    with pytest.raises(RouterProtocolError):
        parse_router_decision(raw)


def test_parse_router_decision_accepts_canonical_output() -> None:
    decision = parse_router_decision('{"intent": "agentic_rag"}')
    assert decision == RouterDecision(intent="agentic_rag")


@pytest.mark.parametrize(
    "failure",
    [
        model_clients.ModelClientError("Router returned invalid JSON."),
        model_clients.ModelClientError("Router returned an empty response."),
        model_clients.ModelConfigurationError("Router model is not configured."),
    ],
)
def test_route_question_falls_back_on_provider_errors(monkeypatch, failure) -> None:
    def fail(messages):
        raise failure

    monkeypatch.setattr(intent_router, "router_json", fail)

    outcome = intent_router.route_question("What is SpecNet2?", document_id=4)

    assert outcome.degraded is True
    assert outcome.intent == FALLBACK_INTENT
    assert outcome.error


def test_route_question_falls_back_on_timeout(monkeypatch) -> None:
    def fail(messages):
        raise TimeoutError("router call timed out")

    monkeypatch.setattr(intent_router, "router_json", fail)

    outcome = intent_router.route_question("What is SpecNet2?", document_id=4)

    assert outcome.degraded is True
    assert outcome.intent == FALLBACK_INTENT
    assert outcome.error == "Router timeout."


def test_router_error_does_not_echo_long_question_or_provider_details(monkeypatch) -> None:
    question = "private-question-" + ("x" * 1200)
    leaked_prefix = question[:500]

    def fail(messages):
        raise ValueError(f"provider echoed {leaked_prefix} api_key=router-secret")

    monkeypatch.setattr(intent_router, "router_json", fail)

    outcome = intent_router.route_question(question, document_id=4)

    assert outcome.degraded is True
    assert outcome.error
    assert "private-question-" not in outcome.error
    assert "x" * 100 not in outcome.error
    assert "router-secret" not in outcome.error


def test_router_prompt_keeps_only_the_last_four_session_messages(monkeypatch) -> None:
    captured: list[list[dict[str, str]]] = []

    def fake_router(messages):
        captured.append(messages)
        return _router_payload()

    monkeypatch.setattr(intent_router, "router_json", fake_router)
    memory = {
        "recent_messages": [
            {"role": "user", "content": f"message-{index}"} for index in range(1, 7)
        ]
    }
    intent_router.route_question("继续。", short_term_memory=memory)

    context = json.loads(captured[0][1]["content"])["session_context"]
    assert [item["content"] for item in context] == [
        "message-3",
        "message-4",
        "message-5",
        "message-6",
    ]


def test_router_prompt_bounds_question_and_context_entries(monkeypatch) -> None:
    captured: list[list[dict[str, str]]] = []

    def fake_router(messages):
        captured.append(messages)
        return _router_payload()

    monkeypatch.setattr(intent_router, "router_json", fake_router)
    memory = {"recent_messages": [{"role": "user", "content": "c" * 5000}]}
    intent_router.route_question("q" * 5000, short_term_memory=memory)

    user_payload = json.loads(captured[0][1]["content"])
    assert len(user_payload["current_question"]) <= 1000
    assert len(user_payload["session_context"][0]["content"]) <= 900


def test_routing_outcome_carries_no_plan_or_confidence() -> None:
    outcome = RoutingOutcome(intent="direct")

    assert not hasattr(outcome, "plan")
    assert not hasattr(outcome, "confidence")


def test_model_client_requires_complete_router_configuration(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key=None,
        agent_api_key=None,
        router_api_key=None,
        router_base_url=None,
        router_model=None,
    )
    monkeypatch.setattr(model_clients, "get_settings", lambda: settings)
    model_clients._router_client.cache_clear()

    with pytest.raises(model_clients.ModelConfigurationError):
        model_clients._router_client()


def test_model_client_builds_openai_client_with_timeout(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        router_api_key="router-key",
        router_base_url="https://router.example/v1",
        router_model="small-router",
        router_timeout_seconds=3.5,
    )
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(model_clients, "get_settings", lambda: settings)
    monkeypatch.setattr(model_clients, "OpenAI", FakeOpenAI)
    model_clients._router_client.cache_clear()

    model_clients._router_client()

    assert captured == {
        "api_key": "router-key",
        "base_url": "https://router.example/v1",
        "timeout": 3.5,
        "max_retries": 0,
    }


def test_router_does_not_reuse_legacy_agent_provider(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="shared-key",
        openai_base_url="https://shared.example/v1",
        openai_chat_model="shared-model",
        router_api_key=None,
        router_base_url=None,
        router_model=None,
    )
    monkeypatch.setattr(model_clients, "get_settings", lambda: settings)
    model_clients._router_client.cache_clear()

    with pytest.raises(model_clients.ModelConfigurationError):
        model_clients._router_client()


def test_router_json_matches_frozen_generation_protocol(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        router_api_key="router-key",
        router_base_url="https://router.example/v1",
        router_model="small-router",
    )
    calls: list[dict[str, object]] = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"intent":"direct"}'))]
            )

    class Client:
        chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(model_clients, "get_settings", lambda: settings)
    monkeypatch.setattr(model_clients, "_router_client", lambda: Client())

    first = model_clients.router_json([{"role": "user", "content": "question"}])
    second = model_clients.router_json([{"role": "user", "content": "question"}])

    assert first == {"intent": "direct"} == second
    assert len(calls) == 2
    assert calls[0]["model"] == "small-router"
    assert calls[0]["temperature"] == 0
    # Frozen M0/Phase 6 protocol: bounded output, thinking disabled, and no
    # grammar forcing that would mask the trained stop behaviour.
    assert calls[0]["max_tokens"] == 64
    assert calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "response_format" not in calls[0]


def test_router_json_redacts_key_and_does_not_log_prompt(monkeypatch, caplog) -> None:
    settings = Settings(
        _env_file=None,
        router_api_key="router-secret",
        router_base_url="https://router.example/v1",
        router_model="small-router",
    )

    class Client:
        chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: (_ for _ in ()).throw(
                    RuntimeError("authorization api_key=router-secret")
                )
            )
        )

    monkeypatch.setattr(model_clients, "get_settings", lambda: settings)
    monkeypatch.setattr(model_clients, "_router_client", lambda: Client())
    secret_prompt = "private question with router-secret"

    with pytest.raises(model_clients.ModelClientError) as exc_info:
        model_clients.router_json([{"role": "user", "content": secret_prompt}])

    assert "router-secret" not in str(exc_info.value)
    assert secret_prompt not in caplog.text


def test_router_json_does_not_return_long_provider_error_or_key(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        router_api_key="router-secret",
        router_base_url="https://router.example/v1",
        router_model="small-router",
    )
    long_prefix = "provider-detail-" + ("p" * 1000)

    class Client:
        chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: (_ for _ in ()).throw(
                    RuntimeError(f"{long_prefix} router-secret")
                )
            )
        )

    monkeypatch.setattr(model_clients, "get_settings", lambda: settings)
    monkeypatch.setattr(model_clients, "_router_client", lambda: Client())

    with pytest.raises(model_clients.ModelClientError) as exc_info:
        model_clients.router_json([{"role": "user", "content": "question"}])

    assert "router-secret" not in str(exc_info.value)
    assert "provider-detail-" not in str(exc_info.value)
