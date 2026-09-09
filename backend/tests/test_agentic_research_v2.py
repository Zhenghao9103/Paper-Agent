from backend.app.agents.orchestrator import AgentResearchResult
from backend.app.schemas.chat import Citation
from backend.app.schemas.evidence import (
    EvidencePack,
    EvidencePackItem,
    EvidencePool,
    LocalSourceRef,
    PackedLocalSource,
    ResearchClaim,
)
from backend.app.schemas.retrieval import RetrievalCandidate
from backend.app.services import research
from backend.app.services.answering import AnswerResult
from backend.app.services.intent_router import RoutingOutcome


def _agent_result(*, completed: bool = True) -> AgentResearchResult:
    claim = ResearchClaim(
        claim_id="C1",
        question="Why use OT?",
        status="covered" if completed else "missing",
        evidence_ids=["E1"] if completed else [],
        gap="" if completed else "Need mechanism evidence.",
    )
    pool = EvidencePool(
        claims=[claim],
        evidence={
            "E1": {
                "evidence_id": "E1",
                "statement": "OT creates balanced assignments.",
                "source_refs": [LocalSourceRef(chunk_id=21)],
                "supports_claim_ids": ["C1"],
                "confidence": 0.9,
                "round_added": 1,
            }
        }
        if completed
        else {},
        next_evidence_number=2 if completed else 1,
    )
    candidate = RetrievalCandidate(
        chunk_id=21,
        document_id=7,
        title="OT Paper",
        page_number=3,
        chunk_index=1,
        content="OT creates balanced assignments.",
        rerank_score=0.9,
    )
    pack = None
    if completed:
        pack = EvidencePack(
            claims=[claim],
            items=[
                EvidencePackItem(
                    evidence_id="E1",
                    statement="OT creates balanced assignments.",
                    supports_claim_ids=["C1"],
                    confidence=0.9,
                    sources=[
                        PackedLocalSource(
                            chunk_id=21,
                            document_id=7,
                            title="OT Paper",
                            page_number=3,
                            chunk_index=1,
                            excerpt=candidate.content,
                            score=0.9,
                        )
                    ],
                )
            ],
            token_count=100,
        )
    return AgentResearchResult(
        protocol_version=2,
        evidence=[candidate] if completed else [],
        rounds=1,
        tool_calls=1,
        completion_status="completed" if completed else "budget_exhausted",
        claims=[claim],
        evidence_pool=pool,
        evidence_pack=pack,
    )


def test_forced_agentic_v2_bypasses_router_and_generator_gets_only_pack(
    db_session, monkeypatch
) -> None:
    result_v2 = _agent_result()
    captured = {}
    monkeypatch.setattr(
        research,
        "route_question",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("router called")),
    )
    monkeypatch.setattr(research, "run_agent", lambda *args, **kwargs: result_v2)
    monkeypatch.setattr(
        research,
        "generate_answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("simple generator called")
        ),
    )

    def fake_agentic_generator(question, evidence_pack, memory_context):
        captured["pack"] = evidence_pack
        captured["memory"] = memory_context
        return AnswerResult(
            answer="OT creates balanced assignments.",
            citations=[
                Citation(
                    document_id=7,
                    chunk_id=21,
                    title="OT Paper",
                    page_number=3,
                    chunk_index=1,
                    score=0.9,
                    content="OT creates balanced assignments.",
                )
            ],
            used_evidence_ids=["E1"],
            verification_status="passed",
            verification_attempts=[
                {
                    "attempt": 1,
                    "stage": "verification",
                    "error_category": None,
                    "provider_category": None,
                    "provider_attempts": 1,
                    "validation_errors": [],
                    "expected_claim_count": 1,
                    "assessed_claim_count": 1,
                    "claim_status_counts": {"supported": 1},
                    "citation_error_count": 0,
                    "unsupported_span_count": 0,
                    "repair_instruction_present": False,
                    "reported_passed": True,
                    "passed": True,
                }
            ],
        )

    monkeypatch.setattr(research, "generate_agentic_answer", fake_agentic_generator)

    result = research.run_research(
        db_session,
        question="Why OT?",
        document_id=7,
        forced_intent="agentic_rag",
    )

    assert captured["pack"] is result_v2.evidence_pack
    assert "session_memory" not in str(captured["pack"])
    assert result.answer == "OT creates balanced assignments."
    assert result.citations[0].chunk_id == 21
    event = next(
        event
        for event in result.trace_events
        if event["type"] == "answer_verification_result"
    )
    assert event["passed"] is True
    assert event["verification_attempt_count"] == 1
    assert event["verification_attempts"] == [
        {
            "attempt": 1,
            "stage": "verification",
            "error_category": None,
            "provider_category": None,
            "provider_attempts": 1,
            "validation_errors": [],
            "expected_claim_count": 1,
            "assessed_claim_count": 1,
            "claim_status_counts": {"supported": 1},
            "citation_error_count": 0,
            "unsupported_span_count": 0,
            "repair_instruction_present": False,
            "reported_passed": True,
            "passed": True,
        }
    ]


def test_normally_routed_agentic_v2_uses_same_active_path(db_session, monkeypatch) -> None:
    calls = {"agent": 0}
    monkeypatch.setattr(
        research,
        "route_question",
        lambda *args, **kwargs: RoutingOutcome(intent="agentic_rag"),
    )

    def fake_agent(*args, **kwargs):
        calls["agent"] += 1
        return _agent_result()

    monkeypatch.setattr(research, "run_agent", fake_agent)
    monkeypatch.setattr(
        research,
        "generate_agentic_answer",
        lambda *args, **kwargs: AnswerResult(
            answer="grounded",
            verification_status="passed",
        ),
    )

    result = research.run_research(db_session, question="Why OT?")

    assert calls["agent"] == 1
    assert result.answer == "grounded"


def test_insufficient_agentic_v2_does_not_call_any_generator(
    db_session, monkeypatch
) -> None:
    monkeypatch.setattr(
        research,
        "run_agent",
        lambda *args, **kwargs: _agent_result(completed=False),
    )
    monkeypatch.setattr(
        research,
        "generate_agentic_answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generator called")),
    )
    monkeypatch.setattr(
        research,
        "generate_answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generator called")),
    )

    result = research.run_research(
        db_session,
        question="Why OT?",
        forced_intent="agentic_rag",
    )

    assert result.citations == []
    assert "尚未覆盖全部必要问题" in result.answer
