import asyncio
import json
import threading

import fitz
from backend.app.api.routes import chat as chat_routes
from backend.app.schemas.chat import ChatRequest, Citation
from backend.app.services.memory import build_trace_events
from backend.app.services.research import _format_history_section
from fastapi.testclient import TestClient


class _FakeSession:
    def __init__(self, *, rollback_fails: bool = False) -> None:
        self.rolled_back = False
        self.closed = False
        self.rollback_fails = rollback_fails

    def rollback(self) -> None:
        self.rolled_back = True
        if self.rollback_fails:
            raise RuntimeError("rollback failed")

    def close(self) -> None:
        self.closed = True


def test_chat_stream_emits_status_and_heartbeat_before_worker_finishes(monkeypatch) -> None:
    release = threading.Event()
    session = _FakeSession()

    def fake_answer(*args, progress_callback=None, **kwargs):
        assert progress_callback is not None
        release.wait(timeout=2)
        return 7, "answer", [], [], [], [], []

    monkeypatch.setattr(chat_routes, "answer_question", fake_answer)

    async def exercise() -> None:
        stream = chat_routes._stream_events(
            ChatRequest(question="slow"),
            session_factory=lambda: session,
            heartbeat_seconds=0.02,
        )
        first = await anext(stream)
        assert first["event"] == "status"
        assert json.loads(first["data"])["code"] == "request_received"
        second = await anext(stream)
        assert second["event"] == "heartbeat"
        heartbeat = json.loads(second["data"])
        assert set(heartbeat) == {"type", "code", "message", "elapsed_seconds"}
        release.set()
        remaining = [event async for event in stream]
        assert [event["event"] for event in remaining] == ["session", "final"]

    asyncio.run(exercise())
    assert session.closed is True


def test_chat_stream_forwards_bounded_status_and_raw_final(monkeypatch) -> None:
    session = _FakeSession()
    citation = Citation(
        document_id=8,
        chunk_id=1330,
        title="Paper",
        page_number=3,
        chunk_index=2,
        score=0.9,
        content="Evidence",
    )

    def fake_answer(*args, progress_callback=None, **kwargs):
        progress_callback({"code": "routing", "message": "正在判断问题类型"})
        return 11, "**answer** [E2] [1330]", [citation], [], [], [], []

    monkeypatch.setattr(chat_routes, "answer_question", fake_answer)

    async def exercise() -> list[dict[str, str]]:
        return [
            event
            async for event in chat_routes._stream_events(
                ChatRequest(question="question"),
                session_factory=lambda: session,
                heartbeat_seconds=1,
            )
        ]

    events = asyncio.run(exercise())
    assert [event["event"] for event in events] == [
        "status",
        "status",
        "session",
        "final",
    ]
    final = json.loads(events[-1]["data"])
    assert final == {
        "type": "final_answer",
        "session_id": 11,
        "answer": "**answer** [E2] [1330]",
        "citation_chunk_ids": [1330],
    }


def test_chat_stream_sanitizes_worker_error_and_closes_session(monkeypatch) -> None:
    session = _FakeSession()

    def fake_answer(*args, **kwargs):
        raise RuntimeError("sk-secret must not leak")

    monkeypatch.setattr(chat_routes, "answer_question", fake_answer)

    async def exercise() -> list[dict[str, str]]:
        return [
            event
            async for event in chat_routes._stream_events(
                ChatRequest(question="question"),
                session_factory=lambda: session,
                heartbeat_seconds=1,
            )
        ]

    events = asyncio.run(exercise())
    assert [event["event"] for event in events] == ["status", "error"]
    error = json.loads(events[-1]["data"])
    assert error == {
        "type": "error",
        "code": "request_failed",
        "message": "模型调用失败，请稍后重试。",
    }
    assert "sk-secret" not in events[-1]["data"]
    assert session.rolled_back is True
    assert session.closed is True


def test_chat_stream_reports_failure_even_when_rollback_also_fails(monkeypatch) -> None:
    session = _FakeSession(rollback_fails=True)

    def fake_answer(*args, **kwargs):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(chat_routes, "answer_question", fake_answer)

    async def exercise() -> list[dict[str, str]]:
        return [
            event
            async for event in chat_routes._stream_events(
                ChatRequest(question="question"),
                session_factory=lambda: session,
                heartbeat_seconds=0.02,
            )
        ]

    events = asyncio.run(asyncio.wait_for(exercise(), timeout=1))
    assert [event["event"] for event in events] == ["status", "error"]
    assert session.rolled_back is True
    assert session.closed is True


def make_pdf_bytes(text: str) -> bytes:
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    data = pdf.tobytes()
    pdf.close()
    return data


def test_chat_ask_returns_structured_trace_events(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr("backend.app.agents.tools.search_arxiv", lambda *args, **kwargs: [])
    client.post(
        "/api/documents/upload",
        files={
            "file": (
                "trace.pdf",
                make_pdf_bytes("Retrieval augmented generation cites local evidence."),
                "application/pdf",
            )
        },
    )

    response = client.post(
        "/api/chat/ask",
        json={"question": "What does retrieval augmented generation cite?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["trace_events"][0]["type"] == "node"
    assert payload["trace_events"][0]["node"] == "load_short_term_memory"
    assert payload["trace_events"][-1]["type"] == "final_answer"
    assert payload["citations"][0]["score"] >= 0

    transcript = client.get(f"/api/transcripts/session/{payload['session_id']}").json()
    event_types = [event["event_type"] for event in transcript]
    assert "trace_event" in event_types
    assert "qa" in event_types

    event_ids = [event["event_id"] for event in payload["trace_events"]]
    turn_ids = {event["turn_id"] for event in payload["trace_events"]}
    assert len(turn_ids) == 1
    assert len(event_ids) == len(set(event_ids))
    assert [event["sequence"] for event in payload["trace_events"]] == list(
        range(1, len(payload["trace_events"]) + 1)
    )


def test_chat_stream_outputs_status_session_and_bounded_final_events(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr("backend.app.agents.tools.search_arxiv", lambda *args, **kwargs: [])
    client.post(
        "/api/documents/upload",
        files={
            "file": (
                "stream.pdf",
                make_pdf_bytes("Agent trace output includes similarity scores."),
                "application/pdf",
            )
        },
    )

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"question": "What includes similarity scores?"},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: status" in body
    assert "event: session" in body
    assert "event: final" in body
    assert "event: trace" not in body
    assert "event: citation" not in body
    data_lines = [
        line.removeprefix("data: ")
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    decoded = [json.loads(line) for line in data_lines]
    final = next(event for event in decoded if event.get("type") == "final_answer")
    assert final["answer"]
    assert isinstance(final["citation_chunk_ids"], list)


def test_chat_ask_injects_short_and_long_term_memory_into_the_llm_prompt(
    client: TestClient,
    monkeypatch,
) -> None:
    """Loading memory is not enough: it has to reach the prompt to affect the answer."""
    monkeypatch.setattr("backend.app.agents.tools.search_arxiv", lambda *args, **kwargs: [])
    captured_prompts: list[str] = []

    def fake_complete(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "基于本地证据的回答。"

    monkeypatch.setattr(
        "backend.app.services.answering.answer_json",
        lambda messages: {
            "answer": fake_complete(str(messages)),
            "local_chunk_ids": [1],
            "web_source_urls": [],
        },
    )
    client.post(
        "/api/documents/upload",
        files={
            "file": (
                "memory-prompt.pdf",
                make_pdf_bytes("Spectral clustering builds graph Laplacian evidence."),
                "application/pdf",
            )
        },
    )

    first = client.post(
        "/api/chat/ask",
        json={"question": "What does spectral clustering build in its pipeline?"},
    ).json()
    second = client.post(
        "/api/chat/ask",
        json={
            "session_id": first["session_id"],
            "question": "Why does that matter for stability?",
        },
    ).json()

    assert len(captured_prompts) == 2

    # The opening turn has no history and no stored interests yet.
    assert "inject_memory_into_prompt" not in first["trace"]

    follow_up_prompt = captured_prompts[1]
    assert "Why does that matter for stability?" in follow_up_prompt
    assert "What does spectral clustering build in its pipeline?" in follow_up_prompt
    assert "User asked about" in follow_up_prompt
    assert "User asked about" in follow_up_prompt
    assert "inject_memory_into_prompt" in second["trace"]
    assert "load_long_term_vector_memory" in second["trace"]


def test_chat_ask_injects_complete_active_context_into_the_llm_prompt(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr("backend.app.agents.tools.search_arxiv", lambda *args, **kwargs: [])
    captured_prompts: list[str] = []

    def fake_complete(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "基于本地证据的回答。"

    monkeypatch.setattr(
        "backend.app.services.answering.answer_json",
        lambda messages: {
            "answer": fake_complete(str(messages)),
            "local_chunk_ids": [1],
            "web_source_urls": [],
        },
    )
    client.post(
        "/api/documents/upload",
        files={
            "file": (
                "active-context-prompt.pdf",
                make_pdf_bytes("Spectral clustering preserves active context evidence."),
                "application/pdf",
            )
        },
    )
    first = client.post(
        "/api/chat/ask",
        json={"question": "What evidence does spectral clustering preserve?"},
    ).json()

    summary = (
        "CHECKPOINT-SUMMARY-START-"
        + ("完整摘要内容" * 45)
        + "-CHECKPOINT-SUMMARY-END"
    )
    long_message = (
        "RETAINED-HISTORY-5-START-"
        + ("full-message-content-" * 12)
        + "-RETAINED-HISTORY-5-END"
    )
    retained_messages = [
        {"role": "user", "content": "RETAINED-HISTORY-1"},
        {"role": "assistant", "content": "RETAINED-HISTORY-2"},
        {"role": "user", "content": "RETAINED-HISTORY-3"},
        {"role": "assistant", "content": "RETAINED-HISTORY-4"},
        {"role": "user", "content": long_message},
    ]
    loaded_session_ids: list[int] = []

    def fake_load_active_context(db, session_id: int) -> dict:
        loaded_session_ids.append(session_id)
        return {
            "session_summary": summary,
            "recent_messages": retained_messages,
        }

    monkeypatch.setattr(
        "backend.app.services.research.load_active_context",
        fake_load_active_context,
        raising=False,
    )

    second = client.post(
        "/api/chat/ask",
        json={
            "session_id": first["session_id"],
            "question": "How does the retained context affect stability?",
        },
    ).json()

    assert second["session_id"] == first["session_id"]
    assert loaded_session_ids == [first["session_id"]]
    assert len(captured_prompts) == 2
    follow_up_prompt = captured_prompts[1]
    assert summary in follow_up_prompt
    for index in range(1, 5):
        assert f"RETAINED-HISTORY-{index}" in follow_up_prompt
    assert long_message in follow_up_prompt


def test_history_prompt_prefers_structured_session_memory() -> None:
    history = _format_history_section(
        {
            "session_memory": {
                "goals_and_constraints": "完成论文对比",
                "confirmed_findings": "基线结果稳定",
                "current_decisions": "使用规则评分",
                "open_questions": "消融实验尚未完成",
                "next_actions": "运行最终消融",
            },
            "session_summary": "旧摘要不得重复注入",
            "recent_messages": [
                {"role": "user", "content": "继续上一项任务"},
            ],
        }
    )

    assert "会话任务状态" in history
    assert "当前目标与约束：完成论文对比" in history
    assert "已确认结论：基线结果稳定" in history
    assert "当前研究决策：使用规则评分" in history
    assert "尚未解决的问题：消融实验尚未完成" in history
    assert "下一步：运行最终消融" in history
    assert "旧摘要不得重复注入" not in history
    assert "用户：继续上一项任务" in history


def test_chat_ask_loads_recent_messages_as_short_term_memory(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr("backend.app.agents.tools.search_arxiv", lambda *args, **kwargs: [])
    client.post(
        "/api/documents/upload",
        files={
            "file": (
                "history.pdf",
                make_pdf_bytes("Spectral clustering builds graph Laplacian evidence."),
                "application/pdf",
            )
        },
    )

    first = client.post(
        "/api/chat/ask",
        json={"question": "What does spectral clustering build?"},
    ).json()
    second = client.post(
        "/api/chat/ask",
        json={
            "session_id": first["session_id"],
            "question": "What did I ask about before?",
        },
    ).json()

    assert "load_recent_chat_history" in second["trace"]


def test_trace_events_include_context_checkpoint_payload() -> None:
    checkpoint = {
        "status": "created",
        "checkpoint_id": 7,
        "tokens_before": 128000,
        "first_kept_message_id": 21,
    }

    events = build_trace_events(
        trace=[],
        citations=[],
        web_sources=[],
        memory_hits=[],
        context_checkpoint=checkpoint,
        answer="answer",
    )

    assert {"type": "context_checkpoint", **checkpoint} in events
