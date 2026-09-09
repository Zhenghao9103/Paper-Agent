from types import SimpleNamespace

from backend.app.agents import orchestrator
from backend.app.resilience import ExecutionResult, FailureInfo
from backend.app.schemas.evidence import (
    CandidateDecision,
    ClaimAssessment,
    EvidenceJudgeResult,
    ResearchClaim,
    ResearchPlan,
    SearchAction,
)
from backend.app.schemas.retrieval import QueryPlan, RetrievalCandidate


def _query_plan() -> QueryPlan:
    return QueryPlan(
        intent="agentic_rag",
        confidence=1,
        standalone_query="Why OT instead of KMeans, and is it validated?",
        semantic_queries=["Why OT instead of KMeans, and is it validated?"],
        document_id=7,
    )


def _settings(**overrides):
    values = {
        "agent_max_rounds": 3,
        "agent_max_tool_calls": 5,
        "agent_total_timeout_seconds": 180.0,
        "agent_generation_evidence_tokens": 12000,
        "resolved_agent_model": "test-model",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _candidate(chunk_id: int, content: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=7,
        title="OT Paper",
        page_number=4,
        chunk_index=chunk_id,
        content=content,
    )


def _initial_plan() -> ResearchPlan:
    return ResearchPlan(
        claims=[
            ResearchClaim(claim_id="C1", question="What role does OT play?"),
            ResearchClaim(claim_id="C2", question="Is there an ablation?"),
        ],
        actions=[
            SearchAction(
                action_id="A1",
                claim_ids=["C1"],
                tool="hybrid_search",
                query="optimal transport balanced assignment",
                document_id=7,
            )
        ],
    )


def _keep_result(
    chunk_id: int,
    claim_id: str,
    *,
    c1_status: str,
    c2_status: str,
    c2_gap: str = "",
    sufficient: bool,
) -> EvidenceJudgeResult:
    return EvidenceJudgeResult(
        decisions=[
            CandidateDecision(
                source_refs=[{"kind": "local", "chunk_id": chunk_id}],
                action="keep",
                statement=f"Evidence from chunk {chunk_id}.",
                supports_claim_ids=[claim_id],
                confidence=0.9,
                reason="Direct evidence.",
            )
        ],
        claim_assessments=[
            ClaimAssessment(claim_id="C1", status=c1_status),
            ClaimAssessment(claim_id="C2", status=c2_status, gap=c2_gap),
        ],
        overall_sufficient=sufficient,
    )


def test_evidence_v2_next_round_query_is_derived_from_missing_claim(
    db_session, monkeypatch
) -> None:
    monkeypatch.setattr(orchestrator, "get_settings", _settings)
    monkeypatch.setattr(
        orchestrator,
        "create_initial_plan",
        lambda *args, **kwargs: _initial_plan(),
    )
    followup_snapshots = []

    def fake_followup(question, pool, attempted_queries, **kwargs):
        followup_snapshots.append(pool.model_copy(deep=True))
        return [
            SearchAction(
                action_id="A2",
                claim_ids=["C2"],
                tool="hybrid_search",
                query="optimal transport ablation clustering objective",
                document_id=7,
            )
        ]

    monkeypatch.setattr(orchestrator, "create_followup_actions", fake_followup)

    def fake_execute(db, tool_name, raw_args, *, ledger, **kwargs):
        chunk_id = 21 if raw_args["task_id"] == "A1" else 71
        candidate = _candidate(chunk_id, raw_args["query"])
        ledger.add_local(candidate)
        return ExecutionResult.success({"candidates": [candidate.model_dump()]})

    monkeypatch.setattr(orchestrator, "execute_tool_result", fake_execute)
    judge_results = iter(
        [
            _keep_result(
                21,
                "C1",
                c1_status="covered",
                c2_status="missing",
                c2_gap="Need an experiment or ablation result.",
                sufficient=False,
            ),
            _keep_result(
                71,
                "C2",
                c1_status="covered",
                c2_status="covered",
                sufficient=True,
            ),
        ]
    )
    monkeypatch.setattr(
        orchestrator,
        "judge_evidence_round",
        lambda *args, **kwargs: next(judge_results),
    )

    result = orchestrator.run_agent(
        db_session,
        "Why OT?",
        _query_plan(),
        {},
    )

    assert result.completion_status == "completed"
    assert result.rounds == 2
    assert result.tool_calls == 2
    assert followup_snapshots[0].claims[1].gap == "Need an experiment or ablation result."
    assert set(result.evidence_pool.evidence) == {"E1", "E2"}
    assert [item.chunk_id for item in result.evidence] == [21, 71]


def test_evidence_v2_loop_stops_immediately_when_pool_is_sufficient(
    db_session, monkeypatch
) -> None:
    monkeypatch.setattr(orchestrator, "get_settings", _settings)
    plan = _initial_plan().model_copy(update={"claims": _initial_plan().claims[:1]})
    plan = plan.model_copy(
        update={"actions": [plan.actions[0].model_copy(update={"claim_ids": ["C1"]})]}
    )
    monkeypatch.setattr(orchestrator, "create_initial_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        orchestrator,
        "create_followup_actions",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("followup called")),
    )

    def fake_execute(db, tool_name, raw_args, *, ledger, **kwargs):
        ledger.add_local(_candidate(21, "balanced assignment"))
        return ExecutionResult.success({})

    monkeypatch.setattr(orchestrator, "execute_tool_result", fake_execute)
    monkeypatch.setattr(
        orchestrator,
        "judge_evidence_round",
        lambda *args, **kwargs: EvidenceJudgeResult(
            decisions=[
                CandidateDecision(
                    source_refs=[{"kind": "local", "chunk_id": 21}],
                    action="keep",
                    statement="OT creates balanced assignments.",
                    supports_claim_ids=["C1"],
                    confidence=0.9,
                    reason="Direct evidence.",
                )
            ],
            claim_assessments=[ClaimAssessment(claim_id="C1", status="covered")],
            overall_sufficient=True,
        ),
    )

    result = orchestrator.run_agent(db_session, "Why OT?", _query_plan(), {})

    assert result.completion_status == "completed"
    assert result.rounds == 1


def test_evidence_v2_preserves_action_claim_candidate_groups_for_judge(
    db_session, monkeypatch
) -> None:
    monkeypatch.setattr(orchestrator, "get_settings", _settings)
    plan = _initial_plan().model_copy(
        update={
            "actions": [
                _initial_plan().actions[0],
                SearchAction(
                    action_id="A2",
                    claim_ids=["C2"],
                    tool="hybrid_search",
                    query="optimal transport ablation",
                    document_id=7,
                ),
            ]
        }
    )
    monkeypatch.setattr(orchestrator, "create_initial_plan", lambda *args, **kwargs: plan)

    def fake_execute(db, tool_name, raw_args, *, ledger, **kwargs):
        chunk_id = 21 if raw_args["task_id"] == "A1" else 71
        candidate = _candidate(chunk_id, raw_args["query"])
        ledger.add_local(candidate)
        return ExecutionResult.success({"candidates": [candidate.model_dump()]})

    monkeypatch.setattr(orchestrator, "execute_tool_result", fake_execute)
    captured: list[object] = []

    def fake_judge(question, pool, candidate_groups, **kwargs):
        captured.extend(candidate_groups)
        return EvidenceJudgeResult(
            decisions=[
                CandidateDecision(
                    source_refs=[{"kind": "local", "chunk_id": 21}],
                    action="keep",
                    statement="OT creates balanced assignments.",
                    supports_claim_ids=["C1", "C2"],
                    confidence=0.9,
                    reason="Direct evidence.",
                )
            ],
            claim_assessments=[
                ClaimAssessment(claim_id="C1", status="covered"),
                ClaimAssessment(claim_id="C2", status="covered"),
            ],
            overall_sufficient=True,
        )

    monkeypatch.setattr(orchestrator, "judge_evidence_round", fake_judge)

    result = orchestrator.run_agent(db_session, "Why OT?", _query_plan(), {})

    assert result.completion_status == "completed"
    assert [getattr(group, "action_id", None) for group in captured] == ["A1", "A2"]
    assert [getattr(group, "target_claim_ids", None) for group in captured] == [
        ["C1"],
        ["C2"],
    ]
    assert [
        [candidate.chunk_id for candidate in getattr(group, "local_candidates", [])]
        for group in captured
    ] == [[21], [71]]


def test_evidence_v2_no_progress_for_two_rounds_is_stalled(
    db_session, monkeypatch
) -> None:
    monkeypatch.setattr(orchestrator, "get_settings", _settings)
    plan = _initial_plan().model_copy(update={"claims": _initial_plan().claims[:1]})
    plan = plan.model_copy(
        update={"actions": [plan.actions[0].model_copy(update={"claim_ids": ["C1"]})]}
    )
    monkeypatch.setattr(orchestrator, "create_initial_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        orchestrator,
        "create_followup_actions",
        lambda *args, **kwargs: [
            SearchAction(
                action_id="A2",
                claim_ids=["C1"],
                tool="hybrid_search",
                query="refined query",
            )
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "execute_tool_result",
        lambda *args, **kwargs: ExecutionResult.success({}),
    )
    monkeypatch.setattr(
        orchestrator,
        "judge_evidence_round",
        lambda *args, **kwargs: EvidenceJudgeResult(
            decisions=[],
            claim_assessments=[
                ClaimAssessment(claim_id="C1", status="missing", gap="Still missing.")
            ],
            overall_sufficient=False,
        ),
    )

    result = orchestrator.run_agent(db_session, "Why OT?", _query_plan(), {})

    assert result.completion_status == "stalled"
    assert result.rounds == 2


def test_evidence_v2_distinguishes_planner_and_judge_failures(
    db_session, monkeypatch
) -> None:
    monkeypatch.setattr(orchestrator, "get_settings", _settings)
    monkeypatch.setattr(
        orchestrator,
        "create_initial_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            orchestrator.ResearchPlannerError("planner_schema_error", attempts=2)
        ),
    )
    planner_result = orchestrator.run_agent(
        db_session, "Why OT?", _query_plan(), {}
    )
    assert planner_result.completion_status == "planner_error"

    monkeypatch.setattr(
        orchestrator,
        "create_initial_plan",
        lambda *args, **kwargs: _initial_plan(),
    )
    monkeypatch.setattr(
        orchestrator,
        "execute_tool_result",
        lambda *args, **kwargs: ExecutionResult.success({}),
    )
    monkeypatch.setattr(
        orchestrator,
        "judge_evidence_round",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            orchestrator.EvidenceJudgeError("judge_schema_error", attempts=2)
        ),
    )
    judge_result = orchestrator.run_agent(db_session, "Why OT?", _query_plan(), {})
    assert judge_result.completion_status == "judge_error"


def test_evidence_v2_judge_error_trace_preserves_safe_diagnostics(
    db_session, monkeypatch
) -> None:
    monkeypatch.setattr(orchestrator, "get_settings", _settings)
    monkeypatch.setattr(
        orchestrator,
        "create_initial_plan",
        lambda *args, **kwargs: _initial_plan(),
    )
    monkeypatch.setattr(
        orchestrator,
        "execute_tool_result",
        lambda *args, **kwargs: ExecutionResult.success({}),
    )

    def failing_judge(*args, diagnostics, **kwargs):
        diagnostics.update(
            {
                "input_token_count": 4210,
                "serialized_char_count": 16840,
                "candidate_group_count": 1,
                "candidate_count_before": 20,
                "candidate_count_after": 12,
                "candidate_count_removed": 8,
            }
        )
        raise orchestrator.EvidenceJudgeError(
            "judge_model_error",
            attempts=2,
            provider_category="timeout",
            validation_errors=[{"loc": "response", "type": "timeout"}],
        )

    monkeypatch.setattr(orchestrator, "judge_evidence_round", failing_judge)

    result = orchestrator.run_agent(db_session, "Why OT?", _query_plan(), {})

    event = next(
        item
        for item in result.trace_events
        if item.get("event") == "evidence_judge_result"
    )
    assert event["status"] == "error"
    assert event["error_category"] == "judge_model_error"
    assert event["attempts"] == 2
    assert event["provider_category"] == "timeout"
    assert event["validation_errors"] == [{"loc": "response", "type": "timeout"}]
    assert event["input_token_count"] == 4210
    assert event["candidate_count_before"] == 20
    assert event["candidate_count_after"] == 12
    assert event["candidate_count_removed"] == 8


def test_failed_tool_result_is_not_sent_to_evidence_judge(
    db_session, monkeypatch
) -> None:
    monkeypatch.setattr(orchestrator, "get_settings", _settings)
    monkeypatch.setattr(
        orchestrator, "create_initial_plan", lambda *args, **kwargs: _initial_plan()
    )
    monkeypatch.setattr(
        orchestrator,
        "execute_tool_result",
        lambda *args, **kwargs: ExecutionResult.failed(
            FailureInfo(
                category="transient",
                code="local_retrieval_failed",
                retryable=True,
            )
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "judge_evidence_round",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("judge must not receive a failed action")
        ),
    )

    result = orchestrator.run_agent(db_session, "Why OT?", _query_plan(), {})

    assert result.completion_status == "tool_error"
    event = next(
        item
        for item in result.trace_events
        if item["event"] == "search_action_result"
    )
    assert event["status"] == "failed"
    assert event["failure_category"] == "transient"
    assert event["error_category"] == "local_retrieval_failed"


def test_degraded_tool_result_with_candidates_can_continue(
    db_session, monkeypatch
) -> None:
    monkeypatch.setattr(orchestrator, "get_settings", _settings)
    plan = _initial_plan().model_copy(update={"claims": _initial_plan().claims[:1]})
    plan = plan.model_copy(
        update={"actions": [plan.actions[0].model_copy(update={"claim_ids": ["C1"]})]}
    )
    monkeypatch.setattr(
        orchestrator, "create_initial_plan", lambda *args, **kwargs: plan
    )
    candidate = _candidate(21, "balanced assignment")

    def degraded_result(*args, ledger, **kwargs):
        ledger.add_local(candidate)
        return ExecutionResult.degraded(
            {"candidates": [candidate.model_dump()]},
            failure=FailureInfo(
                category="transient",
                code="retrieval_channel_degraded",
                retryable=True,
            ),
        )

    monkeypatch.setattr(orchestrator, "execute_tool_result", degraded_result)
    monkeypatch.setattr(
        orchestrator,
        "judge_evidence_round",
        lambda *args, **kwargs: _keep_result(
            21,
            "C1",
            c1_status="covered",
            c2_status="missing",
            sufficient=True,
        ).model_copy(
            update={
                "claim_assessments": [
                    ClaimAssessment(claim_id="C1", status="covered")
                ]
            }
        ),
    )

    result = orchestrator.run_agent(db_session, "Why OT?", _query_plan(), {})

    assert result.completion_status == "completed"
    event = next(
        item
        for item in result.trace_events
        if item["event"] == "search_action_result"
    )
    assert event["status"] == "degraded"
    assert event["error_category"] == "retrieval_channel_degraded"
