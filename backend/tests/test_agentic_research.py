from backend.app.schemas.chat import Citation
from backend.app.schemas.retrieval import RetrievalCandidate
from backend.app.services import research
from backend.app.services.answering import AnswerResult
from backend.app.services.intent_router import RoutingOutcome
from backend.app.services.query_planner import QueryPlanningOutcome, fallback_plan


def _outcome(intent: str) -> RoutingOutcome:
    return RoutingOutcome(intent=intent)


def _planning(question: str, document_id=None) -> QueryPlanningOutcome:
    return QueryPlanningOutcome(plan=fallback_plan(question, document_id))


def _candidate() -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=1,
        document_id=1,
        title="RAG",
        page_number=1,
        chunk_index=0,
        content="Retrieval uses external evidence.",
        fusion_score=0.9,
    )


def test_direct_route_skips_retrieval_and_agent(db_session, monkeypatch) -> None:
    monkeypatch.setattr(research, "route_question", lambda *args, **kwargs: _outcome("direct"))
    monkeypatch.setattr(research, "answer_direct", lambda *args, **kwargs: "direct")
    monkeypatch.setattr(
        research,
        "hybrid_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval")),
    )
    monkeypatch.setattr(
        research,
        "run_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("agent")),
    )

    result = research.run_research(db_session, question="how many papers?")

    assert result.answer == "direct"
    assert result.citations == []
    assert "route_direct" in result.trace


def test_simple_route_plans_retrieves_and_generates_once(db_session, monkeypatch) -> None:
    calls = {"plan": 0, "retrieve": 0, "generate": 0}
    candidate = _candidate()
    monkeypatch.setattr(
        research, "route_question", lambda *args, **kwargs: _outcome("simple_rag")
    )

    def fake_plan(question, document_id=None, short_term_memory=None):
        calls["plan"] += 1
        return _planning(question, document_id)

    def fake_retrieve(*args, **kwargs):
        calls["retrieve"] += 1
        return research.HybridSearchResult(candidates=[candidate])

    def fake_generate(*args, **kwargs):
        calls["generate"] += 1
        return AnswerResult(
            answer="grounded",
            citations=[
                Citation(
                    document_id=1,
                    chunk_id=1,
                    title="RAG",
                    page_number=1,
                    chunk_index=0,
                    score=0.9,
                    content=candidate.content,
                )
            ],
        )

    monkeypatch.setattr(research, "plan_queries", fake_plan)
    monkeypatch.setattr(research, "hybrid_search", fake_retrieve)
    monkeypatch.setattr(research, "generate_answer", fake_generate)
    monkeypatch.setattr(
        research,
        "run_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("agent")),
    )

    result = research.run_research(db_session, question="what is retrieval")

    assert calls == {"plan": 1, "retrieve": 1, "generate": 1}
    assert result.answer == "grounded"
    assert result.citations[0].chunk_id == 1


def test_list_retrieval_preserves_document_scope_when_metadata_is_missing() -> None:
    result = research._normalise_hybrid_result(
        [{"content": "scoped evidence", "score": 0.8}],
        evidence_limit=5,
        document_id=42,
    )

    assert result.candidates[0].document_id == 42
