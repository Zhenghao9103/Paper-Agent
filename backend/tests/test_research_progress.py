from backend.app.agents.orchestrator import AgentResearchResult
from backend.app.rag.hybrid import HybridSearchResult
from backend.app.services import research
from backend.app.services.answering import AnswerResult
from backend.app.services.intent_router import RoutingOutcome
from backend.app.services.progress import emit_progress
from backend.app.services.query_planner import QueryPlanningOutcome, fallback_plan


def _codes(events: list[dict]) -> list[str]:
    return [event["code"] for event in events]


def _quiet_memory(monkeypatch) -> None:
    monkeypatch.setattr(research, "_memory_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        research, "safe_query_relevant_memories", lambda *args, **kwargs: []
    )


def test_emit_progress_none_is_a_noop() -> None:
    emit_progress(None, "routing")


def test_emit_progress_rejects_unknown_codes() -> None:
    try:
        emit_progress(lambda event: None, "raw_internal_trace")
    except ValueError as exc:
        assert str(exc) == "Unsupported progress code: raw_internal_trace"
    else:
        raise AssertionError("unknown progress code was accepted")


def test_progress_callback_failure_does_not_change_research_result(
    db_session,
    monkeypatch,
) -> None:
    _quiet_memory(monkeypatch)
    monkeypatch.setattr(
        research,
        "route_question",
        lambda *args, **kwargs: RoutingOutcome(intent="direct"),
    )
    monkeypatch.setattr(research, "answer_general", lambda question: "direct answer")

    def broken_callback(event: dict) -> None:
        raise RuntimeError("UI listener stopped")

    result = research.run_research(
        db_session,
        question="hello",
        progress_callback=broken_callback,
    )

    assert result.answer == "direct answer"


def test_direct_progress_reports_routing_then_generation(db_session, monkeypatch) -> None:
    events: list[dict] = []
    _quiet_memory(monkeypatch)
    monkeypatch.setattr(
        research,
        "route_question",
        lambda *args, **kwargs: RoutingOutcome(intent="direct"),
    )
    monkeypatch.setattr(research, "answer_general", lambda question: "direct answer")

    result = research.run_research(
        db_session,
        question="hello",
        progress_callback=events.append,
    )

    assert result.answer == "direct answer"
    assert _codes(events) == ["routing", "generation"]


def test_simple_rag_progress_reports_real_phase_order(db_session, monkeypatch) -> None:
    events: list[dict] = []
    _quiet_memory(monkeypatch)
    monkeypatch.setattr(
        research,
        "route_question",
        lambda *args, **kwargs: RoutingOutcome(intent="simple_rag"),
    )
    monkeypatch.setattr(
        research,
        "plan_queries",
        lambda question, document_id, memory: QueryPlanningOutcome(
            plan=fallback_plan(question, document_id)
        ),
    )
    monkeypatch.setattr(
        research,
        "hybrid_search",
        lambda *args, **kwargs: HybridSearchResult(),
    )
    monkeypatch.setattr(
        research,
        "generate_answer",
        lambda *args, **kwargs: AnswerResult(answer="simple answer"),
    )

    result = research.run_research(
        db_session,
        question="compare methods",
        progress_callback=events.append,
    )

    assert result.answer == "simple answer"
    assert _codes(events) == [
        "routing",
        "planning",
        "retrieval",
        "evidence",
        "generation",
    ]


def test_agentic_progress_is_bounded_and_forwards_callback(db_session, monkeypatch) -> None:
    events: list[dict] = []
    _quiet_memory(monkeypatch)

    def fake_agent(*args, progress_callback=None, **kwargs):
        emit_progress(progress_callback, "agent_planning", round_number=1)
        emit_progress(progress_callback, "agent_retrieval", round_number=1)
        return AgentResearchResult(
            completion_status="insufficient_evidence",
            degraded_reason="insufficient_evidence",
        )

    monkeypatch.setattr(research, "run_agent", fake_agent)

    result = research.run_research(
        db_session,
        question="compare two algorithms",
        forced_intent="agentic_rag",
        progress_callback=events.append,
    )

    assert result.answer
    assert _codes(events) == [
        "routing",
        "agent_planning",
        "agent_retrieval",
        "degraded",
        "evidence",
        "generation",
    ]
    assert set().union(*(event.keys() for event in events)) <= {
        "code",
        "message",
        "round",
    }
