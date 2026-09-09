import json
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from backend.app.core.config import Settings
from backend.app.db.base import Base
from backend.app.models import ContextCheckpoint, SessionMemory
from backend.app.models.chat import ChatMessage, ChatSession
from backend.app.services import context_checkpoint as checkpoint_service
from backend.app.services import llm as llm_service
from backend.app.services import model_clients
from backend.app.services.context_checkpoint import (
    CheckpointOutcome,
    compact_session_context,
    count_message_tokens,
    count_text_tokens,
    get_latest_checkpoint,
    select_first_kept_index,
)
from backend.app.services.llm import (
    ContextSummaryError,
    ContextWindowExceededError,
    complete_context_summary,
    complete_research,
    is_context_window_error,
)
from openai import OpenAIError
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session


class CodedOpenAIError(OpenAIError):
    def __init__(
        self,
        message: str,
        *,
        code: object,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        if status_code is not None:
            self.status_code = status_code


class StringableContextLengthCode:
    def __str__(self) -> str:
        return "context_length_exceeded"


VALID_CONTEXT_SUMMARY = (
    "## 用户目标与偏好\n"
    "研究 RAG\n\n"
    "## 已确认结论\n"
    "- 已保留关键结论\n\n"
    "## 当前研究决策\n"
    "继续验证\n\n"
    "## 尚未解决的问题\n"
    "无\n\n"
    "## 下一步\n"
    "继续实验"
)


def _patch_llm_client(
    monkeypatch,
    *,
    content: str | None = "response",
    error: OpenAIError | None = None,
    choices: list[object] | None = None,
) -> tuple[list[dict], dict]:
    settings = SimpleNamespace(
        openai_api_key="sk-test-key",
        openai_base_url="https://provider.example/v1",
        openai_chat_model="test-model",
    )
    requests: list[dict] = []
    client_kwargs: dict = {}

    def create(**kwargs):
        requests.append(kwargs)
        if error is not None:
            raise error
        return SimpleNamespace(
            choices=choices
            if choices is not None
            else [SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def openai(**kwargs):
        client_kwargs.update(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

    monkeypatch.setattr(llm_service, "get_settings", lambda: settings)
    monkeypatch.setattr(llm_service, "OpenAI", openai)
    return requests, client_kwargs


@pytest.mark.parametrize(
    "error",
    [
        CodedOpenAIError("provider rejected the request", code="context_length_exceeded"),
        CodedOpenAIError("provider rejected the request", code="CONTEXT_LENGTH_EXCEEDED"),
        CodedOpenAIError(
            "provider rejected the request",
            code=StringableContextLengthCode(),
        ),
        CodedOpenAIError(
            "provider rejected the request",
            code="context_length_exceeded",
            status_code=429,
        ),
        RuntimeError("CONTEXT_LENGTH_EXCEEDED"),
        RuntimeError("Maximum context length reached"),
        RuntimeError("The context window is full"),
        RuntimeError("Too many tokens in this request"),
    ],
)
def test_context_window_error_detects_provider_code_and_messages(error: Exception) -> None:
    assert is_context_window_error(error) is True


@pytest.mark.parametrize(
    "error",
    [
        CodedOpenAIError("rate limited", code="rate_limit_exceeded"),
        CodedOpenAIError("Too many tokens per minute", code="rate_limit_exceeded"),
        CodedOpenAIError("Context window request rate", code="rate_limit_error"),
        CodedOpenAIError("Maximum context length quota", code="insufficient_quota"),
        CodedOpenAIError("Too many tokens per minute", code="", status_code=429),
        RuntimeError("request timed out"),
        RuntimeError("invalid API key"),
    ],
)
def test_context_window_error_rejects_unrelated_failures(error: Exception) -> None:
    assert is_context_window_error(error) is False


@pytest.mark.parametrize(
    "error",
    [
        OpenAIError("Maximum context length reached"),
        CodedOpenAIError("provider rejected the request", code="context_length_exceeded"),
    ],
)
def test_complete_research_maps_openai_overflow_to_domain_error(
    monkeypatch,
    error: OpenAIError,
) -> None:
    _patch_llm_client(monkeypatch, error=error)

    with pytest.raises(ContextWindowExceededError) as caught:
        complete_research("research prompt")

    assert caught.value.__cause__ is error


def test_complete_research_returns_none_for_other_openai_errors(monkeypatch) -> None:
    _patch_llm_client(monkeypatch, error=OpenAIError("rate limit exceeded"))

    assert complete_research("research prompt") is None


def test_complete_research_returns_none_without_api_key(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_service,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key=None),
    )
    monkeypatch.setattr(
        llm_service,
        "OpenAI",
        lambda **kwargs: pytest.fail("OpenAI client must not be created without a key"),
    )

    assert complete_research("research prompt") is None


def test_complete_research_preserves_analysis_call_configuration(monkeypatch) -> None:
    requests, client_kwargs = _patch_llm_client(monkeypatch, content="answer")

    assert complete_research("research prompt") == "answer"
    assert client_kwargs == {
        "api_key": "sk-test-key",
        "base_url": "https://provider.example/v1",
    }
    assert requests == [
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a concise Chinese academic paper assistant.",
                },
                {"role": "user", "content": "research prompt"},
            ],
            "temperature": 0.2,
        }
    ]


def test_complete_context_summary_requires_agent_configuration(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        agent_api_key=None,
        agent_base_url=None,
        agent_model=None,
    )
    monkeypatch.setattr(llm_service, "get_settings", lambda: settings)
    monkeypatch.setattr(
        model_clients,
        "complex_text_completion",
        lambda messages, **kwargs: pytest.fail(
            "Agent completion must not be attempted without configuration"
        ),
    )

    with pytest.raises(ContextSummaryError):
        complete_context_summary("summarize", max_tokens=128)


def test_complete_context_summary_uses_agent_text_completion(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        agent_api_key="agent-key",
        agent_base_url="https://agent.example/v1",
        agent_model="agent-model",
    )
    calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def fake_completion(messages, **kwargs):
        calls.append((messages, kwargs))
        return VALID_CONTEXT_SUMMARY

    monkeypatch.setattr(llm_service, "get_settings", lambda: settings)
    monkeypatch.setattr(model_clients, "complex_text_completion", fake_completion)

    assert complete_context_summary("summarize", max_tokens=128) == VALID_CONTEXT_SUMMARY
    assert calls[0][0][-1]["content"] == "summarize"
    assert calls[0][1]["max_tokens"] == 128
    assert calls[0][1]["temperature"] == 0


def test_complete_context_summary_returns_stripped_provider_result(monkeypatch) -> None:
    requests, client_kwargs = _patch_llm_client(
        monkeypatch,
        content=f"  {VALID_CONTEXT_SUMMARY}  ",
    )

    result = complete_context_summary("summarize this", max_tokens=321)

    assert result == VALID_CONTEXT_SUMMARY
    assert client_kwargs == {
        "api_key": "sk-test-key",
        "base_url": "https://provider.example/v1",
    }
    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == "test-model"
    assert request["temperature"] == 0
    assert request["max_tokens"] == 321
    assert request["messages"][0]["role"] == "system"
    assert "中文学术上下文检查点" in request["messages"][0]["content"]
    assert "引用标识" in request["messages"][0]["content"]
    assert request["messages"][1] == {"role": "user", "content": "summarize this"}
    assert len(request["messages"]) == 2


def test_complete_context_summary_rejects_nonempty_malformed_result(
    monkeypatch,
) -> None:
    _patch_llm_client(monkeypatch, content="非空但缺少固定标题的摘要")

    with pytest.raises(ContextSummaryError, match="required headings"):
        complete_context_summary("summarize", max_tokens=128)


def test_complete_context_summary_requires_an_input_reference_marker(
    monkeypatch,
) -> None:
    _patch_llm_client(monkeypatch, content=VALID_CONTEXT_SUMMARY)

    with pytest.raises(ContextSummaryError, match="citation marker"):
        complete_context_summary(
            "summarize",
            max_tokens=128,
            required_reference_markers=("[doc:7|page:3|chunk:1]",),
        )


def test_complete_context_summary_accepts_an_input_reference_marker(
    monkeypatch,
) -> None:
    summary = (
        f"{VALID_CONTEXT_SUMMARY}\n"
        "依据：[doc:7|page:3|chunk:1]"
    )
    _patch_llm_client(monkeypatch, content=summary)

    result = complete_context_summary(
        "summarize",
        max_tokens=128,
        required_reference_markers=("[doc:7|page:3|chunk:1]",),
    )

    assert result == summary


@pytest.mark.parametrize("content", [None, "", "   "])
def test_complete_context_summary_rejects_empty_provider_result(
    monkeypatch,
    content: str | None,
) -> None:
    _patch_llm_client(monkeypatch, content=content)

    with pytest.raises(ContextSummaryError):
        complete_context_summary("summarize", max_tokens=128)


@pytest.mark.parametrize(
    "choices",
    [
        [],
        [SimpleNamespace()],
        [SimpleNamespace(message=SimpleNamespace())],
    ],
)
def test_complete_context_summary_rejects_missing_response_content(
    monkeypatch,
    choices: list[object],
) -> None:
    _patch_llm_client(monkeypatch, choices=choices)

    with pytest.raises(ContextSummaryError):
        complete_context_summary("summarize", max_tokens=128)


def test_complete_context_summary_wraps_provider_error_without_leaking_key(
    monkeypatch,
) -> None:
    _patch_llm_client(
        monkeypatch,
        error=OpenAIError("provider rejected sk-secret-value"),
    )

    with pytest.raises(ContextSummaryError) as caught:
        complete_context_summary("summarize", max_tokens=128)

    assert isinstance(caught.value.__cause__, OpenAIError)
    assert "sk-secret-value" not in str(caught.value)
    assert "sk-***" in str(caught.value)


def test_context_checkpoint_table_is_registered(tmp_path: Path) -> None:
    from backend.app.models import ContextCheckpoint

    database_path = (tmp_path / "context-checkpoint.db").as_posix()
    engine = create_engine(f"sqlite:///{database_path}")
    try:
        Base.metadata.create_all(bind=engine)

        assert ContextCheckpoint.__tablename__ == "context_checkpoints"
        assert "context_checkpoints" in inspect(engine).get_table_names()
    finally:
        engine.dispose()


def test_session_memory_table_is_registered(tmp_path: Path) -> None:
    from backend.app.models import SessionMemory

    database_path = (tmp_path / "session-memory.db").as_posix()
    engine = create_engine(f"sqlite:///{database_path}")
    try:
        Base.metadata.create_all(bind=engine)

        assert SessionMemory.__tablename__ == "session_memories"
        columns = {column["name"] for column in inspect(engine).get_columns("session_memories")}
        assert columns == {
            "session_id",
            "goals_and_constraints",
            "confirmed_findings",
            "current_decisions",
            "open_questions",
            "next_actions",
            "source_checkpoint_id",
            "version",
            "updated_at",
        }
    finally:
        engine.dispose()


def test_parse_session_memory_sections_maps_checkpoint_headings() -> None:
    assert checkpoint_service.parse_session_memory_sections(VALID_CONTEXT_SUMMARY) == {
        "goals_and_constraints": "研究 RAG",
        "confirmed_findings": "- 已保留关键结论",
        "current_decisions": "继续验证",
        "open_questions": "无",
        "next_actions": "继续实验",
    }


def test_count_text_tokens_supports_unknown_models() -> None:
    model = "unknown-local-model"

    assert count_text_tokens("English text", model) > 0
    assert count_text_tokens("中文文本", model) > 0


def test_message_tokens_include_role_content_and_citations() -> None:
    model = "unknown-local-model"
    citations = '{"local":[{"document_id":1,"page_number":2}]}'
    message = ChatMessage(
        session_id=1,
        role="assistant",
        content="结论",
        citations=citations,
    )

    with_citations = count_message_tokens(message, model)
    message.citations = None
    without_citations = count_message_tokens(message, model)

    assert with_citations == count_text_tokens(f"assistant\n结论\n{citations}", model)
    assert without_citations == count_text_tokens("assistant\n结论", model)
    assert with_citations > without_citations


def test_cut_point_keeps_complete_turns(monkeypatch) -> None:
    messages = [
        ChatMessage(id=1, session_id=1, role="user", content="u1"),
        ChatMessage(id=2, session_id=1, role="assistant", content="a1"),
        ChatMessage(id=3, session_id=1, role="user", content="u2"),
        ChatMessage(id=4, session_id=1, role="assistant", content="a2"),
        ChatMessage(id=5, session_id=1, role="user", content="u3"),
        ChatMessage(id=6, session_id=1, role="assistant", content="a3"),
    ]
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.count_message_tokens",
        lambda message, model: 10,
    )

    assert select_first_kept_index(messages, keep_recent_tokens=25, model="test") == 4


def test_cut_point_skips_index_zero_and_keeps_scanning(monkeypatch) -> None:
    messages = [
        ChatMessage(id=1, session_id=1, role="user", content="u1"),
        ChatMessage(id=2, session_id=1, role="assistant", content="a1"),
        ChatMessage(id=3, session_id=1, role="user", content="u2"),
        ChatMessage(id=4, session_id=1, role="assistant", content="a2"),
    ]
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.count_message_tokens",
        lambda message, model: 10,
    )

    assert select_first_kept_index(messages, keep_recent_tokens=35, model="test") == 2


def test_cut_point_returns_none_when_no_newer_user_message_exists(monkeypatch) -> None:
    messages = [
        ChatMessage(id=1, session_id=1, role="user", content="u1"),
        ChatMessage(id=2, session_id=1, role="assistant", content="a1"),
        ChatMessage(id=3, session_id=1, role="assistant", content="a2"),
        ChatMessage(id=4, session_id=1, role="assistant", content="a3"),
    ]
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.count_message_tokens",
        lambda message, model: 10,
    )

    assert select_first_kept_index(messages, keep_recent_tokens=25, model="test") is None


@pytest.fixture()
def checkpoint_db(tmp_path: Path) -> Generator[Session, None, None]:
    database_path = (tmp_path / "checkpoint-tests.db").as_posix()
    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "openai_api_key": "test-key",
        "openai_chat_model": "test-model",
        "context_window_tokens": 100,
        "context_reserve_tokens": 20,
        "context_keep_recent_tokens": 25,
        "context_summary_max_tokens": 32,
        "agent_judge_context_tokens": 40,
        "agent_generation_evidence_tokens": 40,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _add_messages(
    db: Session,
    session_id: int,
    roles: list[str],
) -> list[ChatMessage]:
    messages = [
        ChatMessage(
            session_id=session_id,
            role=role,
            content=f"message-{index}",
        )
        for index, role in enumerate(roles, start=1)
    ]
    db.add_all(messages)
    db.flush()
    return messages


def test_load_active_context_uses_latest_checkpoint_boundary(
    checkpoint_db: Session,
) -> None:
    session = ChatSession(
        title="active-context",
        session_summary="legacy summary must be ignored",
    )
    checkpoint_db.add(session)
    checkpoint_db.flush()
    messages = _add_messages(
        checkpoint_db,
        session.id,
        ["user", "assistant", "user", "assistant"],
    )
    checkpoint_db.add(
        ContextCheckpoint(
            session_id=session.id,
            summary="latest checkpoint summary",
            first_kept_message_id=messages[2].id,
            tokens_before=100,
            model="test-model",
            details="{}",
        )
    )
    checkpoint_db.flush()

    active_context = checkpoint_service.load_active_context(
        checkpoint_db,
        session.id,
    )

    assert active_context == {
        "session_memory": None,
        "session_summary": "latest checkpoint summary",
        "recent_messages": [
            {"role": "user", "content": "message-3"},
            {"role": "assistant", "content": "message-4"},
        ],
    }
    assert "legacy summary must be ignored" not in str(active_context)
    assert "message-1" not in str(active_context)
    assert "message-2" not in str(active_context)


def test_load_active_context_without_checkpoint_returns_all_messages(
    checkpoint_db: Session,
) -> None:
    session = ChatSession(
        title="legacy-context",
        session_summary="legacy summary",
    )
    checkpoint_db.add(session)
    checkpoint_db.flush()
    _add_messages(
        checkpoint_db,
        session.id,
        ["user", "assistant", "user", "assistant", "user", "assistant", "user"],
    )

    active_context = checkpoint_service.load_active_context(
        checkpoint_db,
        session.id,
    )

    assert active_context["session_summary"] == "legacy summary"
    assert active_context["session_memory"] is None
    assert active_context["recent_messages"] == [
        {
            "role": "user" if index % 2 else "assistant",
            "content": f"message-{index}",
        }
        for index in range(1, 8)
    ]


def test_load_active_context_without_session_is_empty(
    checkpoint_db: Session,
) -> None:
    assert checkpoint_service.load_active_context(checkpoint_db, 999_999) == {
        "session_memory": None,
        "session_summary": None,
        "recent_messages": [],
    }


def test_load_active_context_prefers_structured_session_memory(
    checkpoint_db: Session,
) -> None:
    session = ChatSession(title="structured-context")
    checkpoint_db.add(session)
    checkpoint_db.flush()
    messages = _add_messages(
        checkpoint_db,
        session.id,
        ["user", "assistant", "user", "assistant"],
    )
    checkpoint = ContextCheckpoint(
        session_id=session.id,
        summary="checkpoint summary must not be injected",
        first_kept_message_id=messages[2].id,
        tokens_before=100,
        model="test-model",
        details="{}",
    )
    checkpoint_db.add(checkpoint)
    checkpoint_db.flush()
    checkpoint_db.add(
        SessionMemory(
            session_id=session.id,
            goals_and_constraints="finish the comparison",
            confirmed_findings="the baseline is stable",
            current_decisions="use rule scoring",
            open_questions="which ablation remains",
            next_actions="run the final ablation",
            source_checkpoint_id=checkpoint.id,
            version=2,
        )
    )
    checkpoint_db.flush()

    active_context = checkpoint_service.load_active_context(
        checkpoint_db,
        session.id,
    )

    assert active_context == {
        "session_memory": {
            "goals_and_constraints": "finish the comparison",
            "confirmed_findings": "the baseline is stable",
            "current_decisions": "use rule scoring",
            "open_questions": "which ablation remains",
            "next_actions": "run the final ablation",
            "source_checkpoint_id": checkpoint.id,
            "version": 2,
        },
        "session_summary": None,
        "recent_messages": [
            {"role": "user", "content": "message-3"},
            {"role": "assistant", "content": "message-4"},
        ],
    }


def test_checkpoint_outcome_payload_omits_internal_error() -> None:
    outcome = CheckpointOutcome(
        status="failed",
        checkpoint_id=None,
        tokens_before=91,
        first_kept_message_id=None,
        error="provider details",
    )

    assert outcome.as_payload() == {
        "status": "failed",
        "checkpoint_id": None,
        "tokens_before": 91,
        "first_kept_message_id": None,
    }


def test_get_latest_checkpoint_breaks_created_at_ties_by_id(
    checkpoint_db: Session,
) -> None:
    session = ChatSession(title="latest")
    checkpoint_db.add(session)
    checkpoint_db.flush()
    created_at = datetime(2026, 1, 1)
    checkpoints = [
        ContextCheckpoint(
            session_id=session.id,
            summary=summary,
            first_kept_message_id=1,
            tokens_before=100,
            model="test-model",
            details="{}",
            created_at=created_at,
        )
        for summary in ("older", "newer")
    ]
    checkpoint_db.add_all(checkpoints)
    checkpoint_db.flush()

    assert get_latest_checkpoint(checkpoint_db, session.id) is checkpoints[1]


def test_compaction_appends_checkpoint_without_deleting_messages(
    checkpoint_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = ChatSession(title="test", session_summary="legacy goal")
    checkpoint_db.add(session)
    checkpoint_db.flush()
    messages = _add_messages(
        checkpoint_db,
        session.id,
        ["user", "assistant", "user", "assistant", "user", "assistant"],
    )
    document_ref = {
        "document_id": 7,
        "title": "Local Paper",
        "page_number": 3,
        "chunk_index": 1,
    }
    messages[1].citations = json.dumps(
        {
            "local": [
                {**document_ref, "content": "evidence", "score": 0.9},
                document_ref,
                {"document_id": 99},
            ]
        }
    )
    messages[2].citations = "{invalid json"
    messages[3].citations = json.dumps({"local": {"document_id": 8}})
    checkpoint_db.flush()
    prompts: list[str] = []
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.count_message_tokens",
        lambda message, model: 20,
    )
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.count_text_tokens",
        lambda text, model: len(text.split()),
    )

    def summarize(
        prompt: str,
        *,
        max_tokens: int,
        required_reference_markers: tuple[str, ...],
    ) -> str:
        prompts.append(prompt)
        assert max_tokens == 32
        assert required_reference_markers == ("[doc:7|page:3|chunk:1]",)
        return (
            f"{VALID_CONTEXT_SUMMARY}\n"
            "依据：[doc:7|page:3|chunk:1]"
        )

    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.complete_context_summary",
        summarize,
    )

    outcome = compact_session_context(
        checkpoint_db,
        session.id,
        settings=_settings(),
    )

    assert outcome.status == "created"
    assert outcome.tokens_before == 122
    assert outcome.first_kept_message_id == messages[4].id
    assert (
        checkpoint_db.query(ChatMessage).filter_by(session_id=session.id).count()
        == 6
    )
    checkpoint = checkpoint_db.get(ContextCheckpoint, outcome.checkpoint_id)
    assert checkpoint is not None
    assert checkpoint.summary == (
        f"{VALID_CONTEXT_SUMMARY}\n"
        "依据：[doc:7|page:3|chunk:1]"
    )
    assert checkpoint.first_kept_message_id == messages[4].id
    assert checkpoint.tokens_before == 122
    assert checkpoint.model == "test-model"
    assert checkpoint.version == 1
    session_memory = checkpoint_db.get(SessionMemory, session.id)
    assert session_memory is not None
    assert session_memory.goals_and_constraints == "研究 RAG"
    assert session_memory.confirmed_findings == "- 已保留关键结论"
    assert session_memory.current_decisions == "继续验证"
    assert session_memory.open_questions == "无"
    assert session_memory.next_actions.startswith("继续实验")
    assert "[doc:7|page:3|chunk:1]" in session_memory.next_actions
    assert session_memory.source_checkpoint_id == checkpoint.id
    assert session_memory.version == 1
    details = json.loads(checkpoint.details)
    assert details["summarized_from_message_id"] == messages[0].id
    assert details["summarized_through_message_id"] == messages[3].id
    assert details["document_refs"] == [document_ref]
    assert details["kept_message_count"] == 2
    assert details["summary_input_tokens"] == len(prompts[0].split())
    assert session.session_summary == "legacy goal"
    assert "legacy goal" in prompts[0]
    for heading in (
        "用户目标与偏好",
        "已确认结论",
        "当前研究决策",
        "尚未解决的问题",
        "下一步",
    ):
        assert f"## {heading}" in prompts[0]
    assert '"role": "assistant"' in prompts[0]
    assert '"citations":' in prompts[0]
    assert "web" in prompts[0]
    assert "长期兴趣" in prompts[0]
    assert "本地" in prompts[0]
    assert '"marker": "[doc:7|page:3|chunk:1]"' in prompts[0]


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        (ContextSummaryError("offline"), "offline"),
        (RuntimeError("malformed provider response"), "RuntimeError"),
    ],
)
def test_compaction_summary_failure_keeps_messages_and_writes_no_checkpoint(
    checkpoint_db: Session,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected_error: str,
) -> None:
    session = ChatSession(title="test")
    checkpoint_db.add(session)
    checkpoint_db.flush()
    messages = _add_messages(
        checkpoint_db,
        session.id,
        ["user", "assistant", "user"],
    )
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.count_message_tokens",
        lambda message, model: 50,
    )

    def fail_summary(
        prompt: str,
        *,
        max_tokens: int,
        **kwargs,
    ) -> str:
        raise failure

    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.complete_context_summary",
        fail_summary,
    )

    outcome = compact_session_context(
        checkpoint_db,
        session.id,
        settings=_settings(),
        force=True,
    )

    assert outcome.status == "failed"
    assert expected_error in (outcome.error or "")
    assert checkpoint_db.query(ContextCheckpoint).count() == 0
    assert checkpoint_db.query(ChatMessage).count() == len(messages)


def test_compaction_tokenizer_failure_is_a_non_destructive_failed_outcome(
    checkpoint_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = ChatSession(title="tokenizer failure")
    checkpoint_db.add(session)
    checkpoint_db.flush()
    messages = _add_messages(
        checkpoint_db,
        session.id,
        ["user", "assistant", "user"],
    )
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.count_text_tokens",
        lambda text, model: (_ for _ in ()).throw(RuntimeError("cache unavailable")),
    )

    outcome = compact_session_context(
        checkpoint_db,
        session.id,
        settings=_settings(),
        force=True,
    )

    assert outcome.status == "failed"
    assert outcome.tokens_before == 0
    assert outcome.error == "Token counting failed: RuntimeError"
    assert checkpoint_db.query(ContextCheckpoint).count() == 0
    assert checkpoint_db.query(ChatMessage).count() == len(messages)


def test_repeated_compaction_uses_previous_checkpoint_and_appends_new_one(
    checkpoint_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = ChatSession(title="test", session_summary="stale legacy summary")
    checkpoint_db.add(session)
    checkpoint_db.flush()
    messages = _add_messages(
        checkpoint_db,
        session.id,
        [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
        ],
    )
    previous_ref = {
        "document_id": 11,
        "title": "Previous Paper",
        "page_number": 4,
        "chunk_index": 2,
    }
    previous_marker = "[doc:11|page:4|chunk:2]"
    first = ContextCheckpoint(
        session_id=session.id,
        summary=f"{VALID_CONTEXT_SUMMARY}\n依据：{previous_marker}",
        first_kept_message_id=messages[2].id,
        tokens_before=100,
        model="test-model",
        version=1,
        details=json.dumps(
            {"document_refs": [previous_ref]},
            ensure_ascii=False,
        ),
    )
    checkpoint_db.add(first)
    checkpoint_db.flush()
    session_memory = SessionMemory(
        session_id=session.id,
        goals_and_constraints="old goal",
        confirmed_findings="old finding",
        current_decisions="old decision",
        open_questions="old question",
        next_actions="old action",
        source_checkpoint_id=first.id,
        version=3,
    )
    checkpoint_db.add(session_memory)
    checkpoint_db.flush()
    prompts: list[str] = []
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.count_message_tokens",
        lambda message, model: 20,
    )
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.count_text_tokens",
        lambda text, model: len(text.split()),
    )

    def summarize(
        prompt: str,
        *,
        max_tokens: int,
        required_reference_markers: tuple[str, ...],
    ) -> str:
        prompts.append(prompt)
        assert required_reference_markers == (previous_marker,)
        return f"{VALID_CONTEXT_SUMMARY}\n依据：{previous_marker}"

    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.complete_context_summary",
        summarize,
    )

    outcome = compact_session_context(
        checkpoint_db,
        session.id,
        settings=_settings(),
    )

    assert outcome.status == "created"
    assert prompts and previous_marker in prompts[0]
    assert "stale legacy summary" not in prompts[0]
    assert "message-3" in prompts[0]
    assert "message-6" in prompts[0]
    assert "message-1" not in prompts[0]
    assert "message-2" not in prompts[0]
    assert outcome.first_kept_message_id == messages[6].id
    assert checkpoint_db.query(ContextCheckpoint).count() == 2
    checkpoint_db.refresh(first)
    assert first.summary == f"{VALID_CONTEXT_SUMMARY}\n依据：{previous_marker}"
    latest = get_latest_checkpoint(checkpoint_db, session.id)
    assert latest is not None
    assert latest.id == outcome.checkpoint_id
    checkpoint_db.refresh(session_memory)
    assert checkpoint_db.query(SessionMemory).count() == 1
    assert session_memory.goals_and_constraints == "研究 RAG"
    assert session_memory.confirmed_findings == "- 已保留关键结论"
    assert session_memory.current_decisions == "继续验证"
    assert session_memory.open_questions == "无"
    assert session_memory.next_actions.startswith("继续实验")
    assert session_memory.source_checkpoint_id == latest.id
    assert session_memory.version == 4
    repeated_details = json.loads(latest.details)
    assert repeated_details["summarized_from_message_id"] == messages[2].id
    assert repeated_details["summarized_through_message_id"] == messages[5].id
    assert repeated_details["document_refs"] == [previous_ref]


def test_repeated_compaction_rejects_dropping_a_previous_summary_marker(
    checkpoint_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = ChatSession(title="marker regression")
    checkpoint_db.add(session)
    checkpoint_db.flush()
    messages = _add_messages(
        checkpoint_db,
        session.id,
        ["user", "assistant", "user", "assistant", "user", "assistant"],
    )
    previous_marker = "[doc:21|page:5|chunk:8]"
    checkpoint_db.add(
        ContextCheckpoint(
            session_id=session.id,
            summary=f"{VALID_CONTEXT_SUMMARY}\n依据：{previous_marker}",
            first_kept_message_id=messages[0].id,
            tokens_before=100,
            model="test-model",
            version=1,
            details="{}",
        )
    )
    checkpoint_db.flush()
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.count_message_tokens",
        lambda message, model: 20,
    )
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.count_text_tokens",
        lambda text, model: len(text.split()),
    )
    _patch_llm_client(monkeypatch, content=VALID_CONTEXT_SUMMARY)

    outcome = compact_session_context(
        checkpoint_db,
        session.id,
        settings=_settings(),
    )

    assert outcome.status == "failed"
    assert "citation marker" in (outcome.error or "")
    assert checkpoint_db.query(ContextCheckpoint).count() == 1
    assert checkpoint_db.query(ChatMessage).count() == len(messages)


def test_compaction_at_soft_threshold_is_not_needed(
    checkpoint_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = ChatSession(title="threshold")
    checkpoint_db.add(session)
    checkpoint_db.flush()
    _add_messages(
        checkpoint_db,
        session.id,
        ["user", "assistant", "user", "assistant"],
    )
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.count_message_tokens",
        lambda message, model: 20,
    )
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.complete_context_summary",
        lambda prompt, max_tokens: pytest.fail("summary must not be requested"),
    )

    outcome = compact_session_context(
        checkpoint_db,
        session.id,
        settings=_settings(),
    )

    assert outcome.status == "not_needed"
    assert outcome.tokens_before == 80
    assert checkpoint_db.query(ContextCheckpoint).count() == 0


def test_disabled_compaction_is_not_needed_even_above_threshold(
    checkpoint_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = ChatSession(title="disabled")
    checkpoint_db.add(session)
    checkpoint_db.flush()
    _add_messages(
        checkpoint_db,
        session.id,
        ["user", "assistant", "user", "assistant", "user", "assistant"],
    )
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.count_message_tokens",
        lambda message, model: 20,
    )
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.complete_context_summary",
        lambda prompt, max_tokens: pytest.fail("summary must not be requested"),
    )

    outcome = compact_session_context(
        checkpoint_db,
        session.id,
        settings=_settings(context_compaction_enabled=False),
    )

    assert outcome.status == "not_needed"
    assert outcome.tokens_before == 120
    assert checkpoint_db.query(ContextCheckpoint).count() == 0


def test_compaction_without_complete_turn_fails_without_writing(
    checkpoint_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = ChatSession(title="incomplete")
    checkpoint_db.add(session)
    checkpoint_db.flush()
    messages = _add_messages(
        checkpoint_db,
        session.id,
        ["user", "assistant", "assistant", "assistant"],
    )
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.count_message_tokens",
        lambda message, model: 10,
    )
    monkeypatch.setattr(
        "backend.app.services.context_checkpoint.complete_context_summary",
        lambda prompt, max_tokens: pytest.fail("summary must not be requested"),
    )

    outcome = compact_session_context(
        checkpoint_db,
        session.id,
        settings=_settings(),
        force=True,
    )

    assert outcome.status == "failed"
    assert outcome.first_kept_message_id is None
    assert checkpoint_db.query(ContextCheckpoint).count() == 0
    assert checkpoint_db.query(ChatMessage).count() == len(messages)
