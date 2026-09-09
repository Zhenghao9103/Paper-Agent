import json
from collections.abc import Generator
from pathlib import Path

import pytest
from backend.app.core.config import Settings
from backend.app.db.base import Base
from backend.app.models.chat import ChatMessage, ChatSession, TranscriptEvent
from backend.app.models.context_checkpoint import ContextCheckpoint
from backend.app.models.trace import AgentTrace
from backend.app.schemas.chat import Citation
from backend.app.services import chat as chat_service
from backend.app.services import context_checkpoint as checkpoint_service
from backend.app.services.context_checkpoint import (
    CheckpointOutcome,
    ContextCompactionUnavailable,
    is_hard_overflow,
)
from backend.app.services.context_compression import compress_evidence
from backend.app.services.llm import ContextSummaryError, ContextWindowExceededError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

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


@pytest.fixture()
def chat_db(tmp_path: Path) -> Generator[Session, None, None]:
    database_path = (tmp_path / "chat-context.db").as_posix()
    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False)
    with session_factory() as db:
        yield db
    engine.dispose()


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-key",
        openai_chat_model="test-model",
        context_window_tokens=100,
        context_reserve_tokens=20,
        context_keep_recent_tokens=25,
        context_summary_max_tokens=32,
        agent_judge_context_tokens=40,
        agent_generation_evidence_tokens=40,
    )


def _graph_result(answer: str = "recovered answer") -> dict:
    return {
        "answer": answer,
        "citations": [],
        "web_sources": [],
        "trace": ["load_short_term_memory"],
        "memory_hits": [],
        "errors": [],
    }


class SequenceResearch:
    def __init__(self, *results: dict | Exception) -> None:
        self.results = list(results)
        self.calls = 0

    def invoke(self, state: dict) -> dict:
        result = self.results[self.calls]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


class _ResearchPayload:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def model_dump(self) -> dict:
        return self.payload


def _patch_research(
    monkeypatch: pytest.MonkeyPatch,
    research: SequenceResearch,
) -> None:
    monkeypatch.setattr(chat_service, "get_settings", _settings, raising=False)
    monkeypatch.setattr(
        chat_service,
        "run_research",
        lambda db, **kwargs: _ResearchPayload(research.invoke(kwargs)),
    )


def _add_history(
    db: Session,
    session_id: int,
    roles: list[str],
) -> list[ChatMessage]:
    messages = [
        ChatMessage(
            session_id=session_id,
            role=role,
            content=f"history-{index}",
        )
        for index, role in enumerate(roles, start=1)
    ]
    db.add_all(messages)
    db.commit()
    return messages


def test_compress_evidence_keeps_top_citations_and_truncates_content() -> None:
    citations = [
        Citation(
            document_id=1,
            title="Paper A",
            page_number=1,
            chunk_index=0,
            score=0.9,
            content="A" * 50,
        ),
        Citation(
            document_id=1,
            title="Paper A",
            page_number=2,
            chunk_index=1,
            score=0.7,
            content="B" * 50,
        ),
        Citation(
            document_id=1,
            title="Paper A",
            page_number=3,
            chunk_index=2,
            score=0.2,
            content="C" * 50,
        ),
    ]

    compressed, stats = compress_evidence(citations, max_items=2, max_chars_per_item=12)

    assert [citation.chunk_index for citation in compressed] == [0, 1]
    assert compressed[0].content == "AAAAAAAAAAA..."
    assert stats == {
        "original_items": 3,
        "kept_items": 2,
        "original_chars": 150,
        "compressed_chars": 28,
        "dropped_items": 1,
    }


def test_hard_overflow_includes_the_exact_window_boundary() -> None:
    settings = _settings()

    assert is_hard_overflow(
        CheckpointOutcome("failed", None, 100, None),
        settings,
    )
    assert not is_hard_overflow(
        CheckpointOutcome("failed", None, 99, None),
        settings,
    )


def test_chat_appends_soft_checkpoint_and_keeps_raw_messages(
    chat_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = ChatSession(title="soft", session_summary="legacy summary")
    chat_db.add(session)
    chat_db.commit()
    graph = SequenceResearch(
        _graph_result("answer-1"),
        _graph_result("answer-2"),
        _graph_result("answer-3"),
    )
    _patch_research(monkeypatch, graph)
    monkeypatch.setattr(
        checkpoint_service,
        "count_message_tokens",
        lambda message, model: 20,
    )
    monkeypatch.setattr(
        checkpoint_service,
        "count_text_tokens",
        lambda text, model: 0,
    )
    monkeypatch.setattr(
        checkpoint_service,
        "complete_context_summary",
        lambda prompt, *, max_tokens, **kwargs: VALID_CONTEXT_SUMMARY,
    )

    last_result = None
    for question in ("first", "second", "third"):
        last_result = chat_service.answer_question(
            chat_db,
            question=question,
            document_id=None,
            session_id=session.id,
        )

    assert last_result is not None
    assert graph.calls == 3
    messages = list(
        chat_db.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session.id)
            .order_by(ChatMessage.id)
        )
    )
    assert [message.content for message in messages] == [
        "first",
        "answer-1",
        "second",
        "answer-2",
        "third",
        "answer-3",
    ]
    assert chat_db.query(ChatSession).count() == 1
    assert messages[4].role == "user"
    checkpoints = list(
        chat_db.scalars(
            select(ContextCheckpoint)
            .where(ContextCheckpoint.session_id == session.id)
            .order_by(ContextCheckpoint.id)
        )
    )
    assert len(checkpoints) == 1
    assert checkpoints[0].first_kept_message_id == messages[4].id
    chat_db.refresh(session)
    assert session.session_summary == "legacy summary"
    assert "context_compaction_completed" in last_result[4]
    assert "compress_short_term_memory" not in last_result[4]
    assert any(
        event.get("type") == "context_checkpoint"
        and event.get("status") == "created"
        for event in last_result[6]
    )
    qa_event = chat_db.scalar(
        select(TranscriptEvent)
        .where(
            TranscriptEvent.session_id == session.id,
            TranscriptEvent.event_type == "qa",
        )
        .order_by(TranscriptEvent.id.desc())
        .limit(1)
    )
    assert qa_event is not None
    assert json.loads(qa_event.payload)["context_checkpoint"]["status"] == "created"


def test_chat_forwards_forced_agentic_route(
    chat_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received = []
    monkeypatch.setattr(chat_service, "get_settings", _settings, raising=False)

    def fake_research(db, **kwargs):
        received.append(kwargs.get("forced_intent"))
        return _ResearchPayload(_graph_result("forced answer"))

    monkeypatch.setattr(chat_service, "run_research", fake_research)

    result = chat_service.answer_question(
        chat_db,
        question="compare papers",
        document_id=None,
        session_id=None,
        forced_intent="agentic_rag",
    )

    assert result[1] == "forced answer"
    assert received == ["agentic_rag"]


def test_soft_failure_continues_and_postflight_hard_failure_keeps_answer(
    chat_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = ChatSession(title="soft failure")
    chat_db.add(session)
    chat_db.commit()
    original_messages = _add_history(
        chat_db,
        session.id,
        ["user", "assistant", "user", "assistant"],
    )
    graph = SequenceResearch(_graph_result("answer survives"))
    _patch_research(monkeypatch, graph)
    monkeypatch.setattr(
        checkpoint_service,
        "count_message_tokens",
        lambda message, model: 22,
    )

    def fail_summary(
        prompt: str,
        *,
        max_tokens: int,
        **kwargs,
    ) -> str:
        raise RuntimeError("unexpected parser failure")

    monkeypatch.setattr(
        checkpoint_service,
        "complete_context_summary",
        fail_summary,
    )

    result = chat_service.answer_question(
        chat_db,
        question="new question",
        document_id=None,
        session_id=session.id,
    )

    assert result[1] == "answer survives"
    assert graph.calls == 1
    assert "context_compaction_failed" in result[4]
    assert chat_db.query(ContextCheckpoint).count() == 0
    assert chat_db.query(ChatMessage).count() == len(original_messages) + 2


def test_chat_retries_once_after_provider_overflow(
    chat_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = SequenceResearch(
        ContextWindowExceededError("overflow"),
        _graph_result(),
    )
    _patch_research(monkeypatch, graph)
    outcomes = [
        CheckpointOutcome("not_needed", None, 10, None),
        CheckpointOutcome("created", 1, 100, 3),
        CheckpointOutcome("not_needed", None, 10, None),
    ]

    def fake_compact(db, session_id, *, settings, force=False):
        return outcomes.pop(0)

    monkeypatch.setattr(
        chat_service,
        "compact_session_context",
        fake_compact,
        raising=False,
    )

    result = chat_service.answer_question(
        chat_db,
        question="recover",
        document_id=None,
        session_id=None,
    )

    assert result[1] == "recovered answer"
    assert graph.calls == 2
    assert "context_overflow_retry" in result[4]
    assert "context_compaction_completed" in result[4]
    assert chat_db.query(ChatMessage).count() == 2
    checkpoint_events = [
        event
        for event in result[6]
        if event.get("type") == "context_checkpoint"
    ]
    assert len(checkpoint_events) == 1
    assert {
        "type": "context_checkpoint",
        "status": "created",
        "checkpoint_id": 1,
        "tokens_before": 100,
        "first_kept_message_id": 3,
    }.items() <= checkpoint_events[0].items()
    assert checkpoint_events[0]["turn_id"].endswith("turn-1")
    assert checkpoint_events[0]["event_id"].endswith(
        f":{checkpoint_events[0]['sequence']}"
    )


def test_force_failure_records_context_error_without_pending_messages(
    chat_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = SequenceResearch(ContextWindowExceededError("overflow"))
    _patch_research(monkeypatch, graph)
    outcomes = [
        CheckpointOutcome("not_needed", None, 10, None),
        CheckpointOutcome("failed", None, 90, None, "offline"),
    ]
    monkeypatch.setattr(
        chat_service,
        "compact_session_context",
        lambda db, session_id, *, settings, force=False: outcomes.pop(0),
        raising=False,
    )

    with pytest.raises(ContextCompactionUnavailable):
        chat_service.answer_question(
            chat_db,
            question="cannot recover",
            document_id=None,
            session_id=None,
        )

    assert graph.calls == 1
    assert chat_db.query(ChatMessage).count() == 0
    assert chat_db.query(AgentTrace).count() == 0
    context_errors = list(
        chat_db.scalars(
            select(TranscriptEvent).where(
                TranscriptEvent.event_type == "context_error"
            )
        )
    )
    assert len(context_errors) == 1
    assert json.loads(context_errors[0].payload) == {
        "trace": "context_overflow_recovery_failed",
        "message": "上下文压缩未能恢复模型窗口。",
    }


def test_second_overflow_keeps_committed_checkpoint_and_no_pending_messages(
    chat_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = SequenceResearch(
        ContextWindowExceededError("first overflow"),
        ContextWindowExceededError("second overflow"),
    )
    _patch_research(monkeypatch, graph)

    def fake_compact(db, session_id, *, settings, force=False):
        if not force:
            return CheckpointOutcome("not_needed", None, 10, None)
        checkpoint = ContextCheckpoint(
            session_id=session_id,
            summary="committed recovery checkpoint",
            first_kept_message_id=1,
            tokens_before=100,
            model="test-model",
            details="{}",
        )
        db.add(checkpoint)
        db.flush()
        return CheckpointOutcome("created", checkpoint.id, 100, 1)

    monkeypatch.setattr(
        chat_service,
        "compact_session_context",
        fake_compact,
        raising=False,
    )

    with pytest.raises(
        ContextCompactionUnavailable,
        match="压缩后仍超过",
    ):
        chat_service.answer_question(
            chat_db,
            question="still too long",
            document_id=None,
            session_id=None,
        )

    assert graph.calls == 2
    assert chat_db.query(ContextCheckpoint).count() == 1
    assert chat_db.query(ChatMessage).count() == 0
    assert chat_db.query(AgentTrace).count() == 0
    assert (
        chat_db.query(TranscriptEvent)
        .filter(TranscriptEvent.event_type == "qa")
        .count()
        == 0
    )
    assert (
        chat_db.query(TranscriptEvent)
        .filter(TranscriptEvent.event_type == "context_error")
        .count()
        == 1
    )


def test_hard_preflight_failure_stops_before_graph_and_records_error(
    chat_db: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = ChatSession(title="hard failure")
    chat_db.add(session)
    chat_db.commit()
    original_messages = _add_history(
        chat_db,
        session.id,
        ["user", "assistant", "user", "assistant", "user", "assistant"],
    )
    graph = SequenceResearch(_graph_result("must not run"))
    _patch_research(monkeypatch, graph)
    monkeypatch.setattr(
        checkpoint_service,
        "count_message_tokens",
        lambda message, model: 20,
    )
    monkeypatch.setattr(
        checkpoint_service,
        "count_text_tokens",
        lambda text, model: 0,
    )

    def fail_summary(
        prompt: str,
        *,
        max_tokens: int,
        **kwargs,
    ) -> str:
        raise ContextSummaryError("offline")

    monkeypatch.setattr(
        checkpoint_service,
        "complete_context_summary",
        fail_summary,
    )

    with pytest.raises(ContextCompactionUnavailable):
        chat_service.answer_question(
            chat_db,
            question="blocked question",
            document_id=None,
            session_id=session.id,
        )

    assert graph.calls == 0
    assert chat_db.query(ChatMessage).count() == len(original_messages)
    assert chat_db.query(ContextCheckpoint).count() == 0
    assert (
        chat_db.query(TranscriptEvent)
        .filter(TranscriptEvent.event_type == "context_error")
        .count()
        == 1
    )
