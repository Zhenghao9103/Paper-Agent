import json

import pytest
from backend.app.agents import research_planner
from backend.app.schemas.evidence import (
    EvidenceItem,
    EvidencePool,
    LocalSourceRef,
    ResearchClaim,
)
from backend.app.services.model_clients import ModelClientError


def _initial_payload() -> dict:
    return {
        "claims": [
            {"claim_id": "C1", "question": "What role does OT play?"},
            {"claim_id": "C2", "question": "Is there an ablation?"},
        ],
        "actions": [
            {
                "action_id": "A1",
                "claim_ids": ["C1"],
                "tool": "hybrid_search",
                "query": "optimal transport balanced assignment",
            }
        ],
        "rationale": "Find mechanism evidence first.",
    }


def _partial_pool() -> EvidencePool:
    return EvidencePool(
        claims=[
            ResearchClaim(
                claim_id="C1",
                question="What role does OT play?",
                status="covered",
                evidence_ids=["E1"],
            ),
            ResearchClaim(
                claim_id="C2",
                question="Is there an ablation?",
                status="missing",
                gap="Need an experiment or ablation result.",
            ),
        ],
        evidence={
            "E1": EvidenceItem(
                evidence_id="E1",
                statement="OT creates balanced assignments.",
                source_refs=[LocalSourceRef(chunk_id=21)],
                supports_claim_ids=["C1"],
                confidence=0.9,
                round_added=1,
            )
        },
        next_evidence_number=2,
    )


def test_initial_plan_creates_required_claims_and_queries(monkeypatch) -> None:
    monkeypatch.setattr(
        research_planner,
        "agent_json",
        lambda messages, **kwargs: _initial_payload(),
    )

    plan = research_planner.create_initial_plan("Why OT?", document_id=7)

    assert [claim.claim_id for claim in plan.claims] == ["C1", "C2"]
    assert plan.actions[0].query == "optimal transport balanced assignment"
    assert plan.actions[0].document_id == 7


def test_initial_plan_receives_claim_fidelity_contract(monkeypatch) -> None:
    captured: list[list[dict[str, str]]] = []

    def fake_agent_json(messages, **kwargs):
        captured.append(messages)
        return _initial_payload()

    monkeypatch.setattr(research_planner, "agent_json", fake_agent_json)

    research_planner.create_initial_plan(
        "BootSC 与 P2OT 分别用 OT 监督什么对象？",
        document_id=None,
    )

    system_prompt = captured[0][0]["content"]
    request = json.loads(captured[0][1]["content"])
    contract = request["claim_fidelity_contract"]
    assert contract["necessary_answer_slots_only"] is True
    assert contract["do_not_guess_answers"] is True
    assert contract["do_not_introduce_new_domains_or_assumptions"] is True
    assert contract["comparison_decomposition"] == (
        "split by the named entities and explicitly requested comparison aspects"
    )
    assert "answer slots" in system_prompt
    assert "new domains" in system_prompt


def test_initial_plan_retries_when_model_selects_disabled_arxiv(monkeypatch) -> None:
    calls: list[list[dict[str, str]]] = []
    arxiv_payload = _initial_payload()
    arxiv_payload["actions"][0] = {
        "action_id": "A1",
        "claim_ids": ["C1"],
        "tool": "search_arxiv",
        "query": "optimal transport balanced assignment",
    }

    def fake_agent_json(messages, **kwargs):
        calls.append(messages)
        return arxiv_payload if len(calls) == 1 else _initial_payload()

    monkeypatch.setattr(research_planner, "agent_json", fake_agent_json)

    plan = research_planner.create_initial_plan("Why OT?", document_id=7)

    initial_prompt = json.loads(calls[0][-1]["content"])
    assert "search_arxiv" not in initial_prompt["allowed_tools"]
    assert "tool_disabled" in calls[1][-1]["content"]
    assert [action.tool for action in plan.actions] == ["hybrid_search"]


def test_followup_plan_targets_only_missing_partial_or_conflicted_claims(monkeypatch) -> None:
    captured: list[list[dict[str, str]]] = []

    def fake_agent_json(messages, **kwargs):
        captured.append(messages)
        return {
            "actions": [
                {
                    "action_id": "A2",
                    "claim_ids": ["C2"],
                    "tool": "hybrid_search",
                    "query": "optimal transport ablation clustering objective",
                }
            ]
        }

    monkeypatch.setattr(research_planner, "agent_json", fake_agent_json)

    actions = research_planner.create_followup_actions(
        "Why OT?",
        _partial_pool(),
        ["optimal transport balanced assignment"],
        remaining_tool_calls=2,
        document_id=7,
    )

    assert actions[0].claim_ids == ["C2"]
    assert actions[0].document_id == 7
    prompt = captured[0][-1]["content"]
    assert "Need an experiment or ablation result" in prompt
    assert "OT creates balanced assignments" in prompt


def test_followup_plan_cannot_change_claim_definitions(monkeypatch) -> None:
    payload = {
        "claims": [{"claim_id": "C3", "question": "Invented claim"}],
        "actions": [
            {
                "action_id": "A2",
                "claim_ids": ["C2"],
                "tool": "hybrid_search",
                "query": "ablation",
            }
        ],
    }
    monkeypatch.setattr(research_planner, "agent_json", lambda messages, **kwargs: payload)

    with pytest.raises(research_planner.ResearchPlannerError) as exc_info:
        research_planner.create_followup_actions(
            "Why OT?",
            _partial_pool(),
            [],
            remaining_tool_calls=2,
            document_id=None,
        )

    assert exc_info.value.category == "planner_schema_error"
    assert exc_info.value.attempts == 2


def test_duplicate_query_is_rejected_before_execution(monkeypatch) -> None:
    payload = {
        "actions": [
            {
                "action_id": "A2",
                "claim_ids": ["C2"],
                "tool": "hybrid_search",
                "query": "  OPTIMAL   transport ABLATION ",
            }
        ]
    }
    monkeypatch.setattr(research_planner, "agent_json", lambda messages, **kwargs: payload)

    with pytest.raises(research_planner.ResearchPlannerError) as exc_info:
        research_planner.create_followup_actions(
            "Why OT?",
            _partial_pool(),
            ["optimal transport ablation"],
            remaining_tool_calls=2,
            document_id=None,
        )

    assert exc_info.value.category == "duplicate_query"


def test_planner_invalid_json_retries_once_with_validation_feedback(monkeypatch) -> None:
    calls: list[list[dict[str, str]]] = []

    def fake_agent_json(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            raise ModelClientError("invalid", category="invalid_json")
        return _initial_payload()

    monkeypatch.setattr(research_planner, "agent_json", fake_agent_json)

    plan = research_planner.create_initial_plan("Why OT?", document_id=None)

    assert plan.claims[0].claim_id == "C1"
    assert len(calls) == 2
    assert "previous response failed validation" in calls[1][-1]["content"].lower()


def test_planner_second_invalid_response_has_stable_error_category(monkeypatch) -> None:
    monkeypatch.setattr(
        research_planner,
        "agent_json",
        lambda messages, **kwargs: {"claims": [], "actions": []},
    )

    with pytest.raises(research_planner.ResearchPlannerError) as exc_info:
        research_planner.create_initial_plan("Why OT?", document_id=None)

    assert exc_info.value.category == "planner_schema_error"
    assert exc_info.value.attempts == 2


def test_followup_plan_does_not_call_model_without_remaining_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        research_planner,
        "agent_json",
        lambda messages, **kwargs: pytest.fail("model must not be called"),
    )

    assert research_planner.create_followup_actions(
        "Why OT?",
        _partial_pool(),
        [],
        remaining_tool_calls=0,
        document_id=None,
    ) == []
