import json
from types import SimpleNamespace

import pytest
from backend.app.agents import evidence_judge
from backend.app.schemas.chat import WebSource
from backend.app.schemas.evidence import (
    EvidenceItem,
    EvidenceJudgeResult,
    EvidencePool,
    LocalSourceRef,
    ResearchClaim,
)
from backend.app.schemas.retrieval import RetrievalCandidate


def _candidate(chunk_id: int, content: str, rerank_score: float = 0.8) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=7,
        title="OT Paper",
        page_number=3,
        chunk_index=chunk_id,
        content=content,
        rerank_score=rerank_score,
    )


def _pool(*, two_claims: bool = False) -> EvidencePool:
    claims = [
        ResearchClaim(
            claim_id="C1",
            question="What role does OT play?",
            status="covered",
            evidence_ids=["E1"],
        )
    ]
    if two_claims:
        claims.append(
            ResearchClaim(
                claim_id="C2",
                question="Is there an ablation?",
                gap="Need ablation evidence.",
            )
        )
    return EvidencePool(
        claims=claims,
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


def _valid_payload(*, sufficient: bool = False) -> dict:
    return {
        "decisions": [
            {
                "source_refs": [{"kind": "local", "chunk_id": 71}],
                "action": "keep",
                "statement": "Removing the clustering objective lowers ACC.",
                "supports_claim_ids": ["C2"],
                "confidence": 0.95,
                "reason": "Direct ablation result.",
            }
        ],
        "claim_assessments": [
            {"claim_id": "C1", "status": "covered", "evidence_ids": ["E1"]},
            {"claim_id": "C2", "status": "covered"},
        ],
        "overall_sufficient": sufficient,
        "unresolved_gaps": [],
        "next_search_focus": [],
    }


def _settings(token_budget: int = 24000):
    return SimpleNamespace(
        agent_judge_context_tokens=token_budget,
        resolved_agent_model="agent-model",
    )


def _group(
    candidates: list[RetrievalCandidate] | None = None,
    web_sources: list[WebSource] | None = None,
) -> evidence_judge.EvidenceCandidateGroup:
    return evidence_judge.EvidenceCandidateGroup(
        action_id="A1",
        target_claim_ids=["C2"],
        query="optimal transport ablation",
        subquestion="Is there an ablation?",
        local_candidates=candidates or [],
        web_sources=web_sources or [],
    )


def test_judge_receives_only_current_pool_and_new_candidates(monkeypatch) -> None:
    captured = []

    def fake_agent_json(messages, **kwargs):
        captured.append(messages)
        return _valid_payload()

    monkeypatch.setattr(evidence_judge, "agent_json", fake_agent_json)
    monkeypatch.setattr(evidence_judge, "get_settings", _settings)

    evidence_judge.judge_evidence_round(
        "Why OT?",
        _pool(two_claims=True),
        [_group([_candidate(71, "Table 4 ablation result")])],
        round_number=2,
    )

    prompt = captured[0][-1]["content"]
    assert "OT creates balanced assignments" in prompt
    assert "Table 4 ablation result" in prompt
    payload = json.loads(prompt)
    assert payload["output_schema"] == EvidenceJudgeResult.model_json_schema()
    group = payload["candidate_groups"][0]
    assert group["action_id"] == "A1"
    assert group["target_claim_ids"] == ["C2"]
    assert group["local_candidates"][0]["source_id"] == "chunk:71"


def test_judge_receives_explicit_action_field_contract(monkeypatch) -> None:
    captured = []

    def fake_agent_json(messages, **kwargs):
        captured.append(messages)
        return _valid_payload()

    monkeypatch.setattr(evidence_judge, "agent_json", fake_agent_json)
    monkeypatch.setattr(evidence_judge, "get_settings", _settings)

    evidence_judge.judge_evidence_round(
        "Why OT?",
        _pool(two_claims=True),
        [_group([_candidate(71, "Table 4 ablation result")])],
        round_number=2,
    )

    payload = json.loads(captured[0][-1]["content"])
    assert payload["decision_action_contract"] == {
        "keep": ["statement", "supports_claim_ids"],
        "drop": [],
        "merge": ["target_evidence_id", "supports_claim_ids"],
        "conflict": ["target_evidence_id", "statement", "supports_claim_ids"],
    }


def test_judge_can_keep_drop_merge_and_mark_conflict(monkeypatch) -> None:
    payload = {
        "decisions": [
            {
                "source_refs": [{"kind": "local", "chunk_id": 71}],
                "action": "keep",
                "statement": "Ablation supports OT.",
                "supports_claim_ids": ["C2"],
                "confidence": 0.9,
                "reason": "Direct result.",
            },
            {
                "source_refs": [{"kind": "local", "chunk_id": 72}],
                "action": "drop",
                "confidence": 0.1,
                "reason": "Irrelevant.",
            },
            {
                "source_refs": [{"kind": "local", "chunk_id": 73}],
                "action": "merge",
                "target_evidence_id": "E1",
                "supports_claim_ids": ["C1"],
                "confidence": 0.8,
                "reason": "Complementary support.",
            },
            {
                "source_refs": [{"kind": "local", "chunk_id": 74}],
                "action": "conflict",
                "target_evidence_id": "E1",
                "statement": "Assignments are not balanced.",
                "supports_claim_ids": ["C1"],
                "confidence": 0.6,
                "reason": "Contradiction.",
            },
        ],
        "claim_assessments": [
            {"claim_id": "C1", "status": "conflicted", "gap": "Resolve conflict."},
            {"claim_id": "C2", "status": "covered"},
        ],
        "overall_sufficient": False,
    }
    monkeypatch.setattr(evidence_judge, "agent_json", lambda messages, **kwargs: payload)
    monkeypatch.setattr(evidence_judge, "get_settings", _settings)

    result = evidence_judge.judge_evidence_round(
        "Why OT?",
        _pool(two_claims=True),
        [
            _group(
                [_candidate(chunk_id, f"candidate {chunk_id}") for chunk_id in range(71, 75)]
            )
        ],
        round_number=2,
    )

    assert [decision.action for decision in result.decisions] == [
        "keep",
        "drop",
        "merge",
        "conflict",
    ]


def test_judge_cannot_reference_unseen_chunk_or_url(monkeypatch) -> None:
    payload = _valid_payload()
    payload["decisions"][0]["source_refs"] = [
        {"kind": "web", "url": "https://example.com/invented"}
    ]
    monkeypatch.setattr(evidence_judge, "agent_json", lambda messages, **kwargs: payload)
    monkeypatch.setattr(evidence_judge, "get_settings", _settings)

    with pytest.raises(evidence_judge.EvidenceJudgeError) as exc_info:
        evidence_judge.judge_evidence_round(
            "Why OT?",
            _pool(two_claims=True),
            [_group([_candidate(71, "ablation")])],
            round_number=2,
        )

    assert exc_info.value.category == "unknown_source_ref"


def test_judge_must_assess_every_required_claim(monkeypatch) -> None:
    payload = _valid_payload()
    payload["claim_assessments"] = payload["claim_assessments"][:1]
    monkeypatch.setattr(evidence_judge, "agent_json", lambda messages, **kwargs: payload)
    monkeypatch.setattr(evidence_judge, "get_settings", _settings)

    with pytest.raises(evidence_judge.EvidenceJudgeError) as exc_info:
        evidence_judge.judge_evidence_round(
            "Why OT?",
            _pool(two_claims=True),
            [_group([_candidate(71, "ablation")])],
            round_number=2,
        )

    assert exc_info.value.category == "incomplete_claim_assessment"


def test_judge_cannot_call_sufficient_with_missing_claim(monkeypatch) -> None:
    payload = _valid_payload(sufficient=True)
    payload["claim_assessments"][1] = {
        "claim_id": "C2",
        "status": "missing",
        "gap": "Need ablation.",
    }
    monkeypatch.setattr(evidence_judge, "agent_json", lambda messages, **kwargs: payload)
    monkeypatch.setattr(evidence_judge, "get_settings", _settings)

    with pytest.raises(evidence_judge.EvidenceJudgeError) as exc_info:
        evidence_judge.judge_evidence_round(
            "Why OT?",
            _pool(two_claims=True),
            [_group([_candidate(71, "ablation")])],
            round_number=2,
        )

    assert exc_info.value.category == "false_sufficiency"


def test_judge_context_is_trimmed_by_tokens_not_item_count(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(evidence_judge, "count_text_tokens", lambda text, model: len(text))
    monkeypatch.setattr(evidence_judge, "get_settings", lambda: _settings(3500))

    def fake_agent_json(messages, **kwargs):
        captured.append(messages)
        return {
            "decisions": [],
            "claim_assessments": [
                {"claim_id": "C1", "status": "covered", "evidence_ids": ["E1"]}
            ],
            "overall_sufficient": True,
        }

    monkeypatch.setattr(evidence_judge, "agent_json", fake_agent_json)
    candidates = [
        _candidate(71, "HIGH" * 900, rerank_score=0.9),
        _candidate(72, "LOW" * 900, rerank_score=0.1),
    ]

    evidence_judge.judge_evidence_round(
        "Why OT?", _pool(), [_group(candidates)], round_number=2
    )

    prompt = captured[0][-1]["content"]
    assert len(prompt) <= 3500
    assert "OT creates balanced assignments" in prompt
    assert prompt.count("HIGH") >= prompt.count("LOW")


def test_judge_reports_final_bounded_request_diagnostics(monkeypatch) -> None:
    captured = []
    diagnostics = {}
    token_budget = 4300
    monkeypatch.setattr(evidence_judge, "count_text_tokens", lambda text, model: len(text))
    monkeypatch.setattr(
        evidence_judge,
        "get_settings",
        lambda: _settings(token_budget),
    )

    def fake_agent_json(messages, **kwargs):
        captured.append(messages)
        return {
            "decisions": [],
            "claim_assessments": [
                {"claim_id": "C1", "status": "covered", "evidence_ids": ["E1"]}
            ],
            "overall_sufficient": True,
        }

    monkeypatch.setattr(evidence_judge, "agent_json", fake_agent_json)
    candidates = [
        _candidate(chunk_id, f"candidate-{chunk_id}-" * 400)
        for chunk_id in range(71, 75)
    ]

    evidence_judge.judge_evidence_round(
        "Why OT?",
        _pool(),
        [_group(candidates)],
        round_number=2,
        diagnostics=diagnostics,
    )

    prompt = captured[0][-1]["content"]
    assert diagnostics == {
        "input_token_count": len(prompt),
        "serialized_char_count": len(prompt),
        "candidate_group_count": 1,
        "candidate_count_before": 4,
        "candidate_count_after": diagnostics["candidate_count_after"],
        "candidate_count_removed": 4 - diagnostics["candidate_count_after"],
    }
    assert 0 <= diagnostics["candidate_count_after"] < 4
    assert diagnostics["input_token_count"] <= token_budget


def test_judge_rejects_candidate_removed_from_bounded_payload(monkeypatch) -> None:
    token_budget = 4300
    all_chunk_ids = set(range(71, 75))
    monkeypatch.setattr(evidence_judge, "count_text_tokens", lambda text, model: len(text))
    monkeypatch.setattr(
        evidence_judge,
        "get_settings",
        lambda: _settings(token_budget),
    )

    def fake_agent_json(messages, **kwargs):
        prompt = json.loads(messages[1]["content"])
        visible_chunk_ids = {
            candidate["chunk_id"]
            for group in prompt["candidate_groups"]
            for candidate in group["local_candidates"]
        }
        removed_chunk_ids = sorted(all_chunk_ids - visible_chunk_ids)
        assert removed_chunk_ids
        return {
            "decisions": [
                {
                    "source_refs": [
                        {"kind": "local", "chunk_id": removed_chunk_ids[0]}
                    ],
                    "action": "drop",
                    "confidence": 0.1,
                    "reason": "This source was not present in the bounded prompt.",
                }
            ],
            "claim_assessments": [
                {"claim_id": "C1", "status": "covered", "evidence_ids": ["E1"]}
            ],
            "overall_sufficient": True,
        }

    monkeypatch.setattr(evidence_judge, "agent_json", fake_agent_json)
    candidates = [
        _candidate(chunk_id, f"candidate-{chunk_id}-" * 400)
        for chunk_id in sorted(all_chunk_ids)
    ]

    with pytest.raises(evidence_judge.EvidenceJudgeError) as exc_info:
        evidence_judge.judge_evidence_round(
            "Why OT?",
            _pool(),
            [_group(candidates)],
            round_number=2,
        )

    assert exc_info.value.category == "unknown_source_ref"


def test_judge_context_too_large_still_reports_final_diagnostics(monkeypatch) -> None:
    diagnostics = {}
    monkeypatch.setattr(evidence_judge, "count_text_tokens", lambda text, model: len(text))
    monkeypatch.setattr(evidence_judge, "get_settings", lambda: _settings(1))

    with pytest.raises(evidence_judge.EvidenceJudgeError) as exc_info:
        evidence_judge.judge_evidence_round(
            "Why OT?",
            _pool(),
            [_group([_candidate(71, "ablation")])],
            round_number=2,
            diagnostics=diagnostics,
        )

    assert exc_info.value.category == "judge_context_too_large"
    assert diagnostics["candidate_count_before"] == 1
    assert diagnostics["candidate_count_after"] == 0
    assert diagnostics["candidate_count_removed"] == 1
    assert diagnostics["input_token_count"] > 1


def test_judge_invalid_schema_retries_once(monkeypatch) -> None:
    calls = []

    def fake_agent_json(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return {"decisions": []}
        return _valid_payload()

    monkeypatch.setattr(evidence_judge, "agent_json", fake_agent_json)
    monkeypatch.setattr(evidence_judge, "get_settings", _settings)

    result = evidence_judge.judge_evidence_round(
        "Why OT?",
        _pool(two_claims=True),
        [_group([_candidate(71, "ablation")])],
        round_number=2,
    )

    assert result.decisions[0].action == "keep"
    assert len(calls) == 2
    assert "failed validation" in calls[1][-1]["content"].lower()


def test_judge_schema_error_reports_bounded_validation_details(monkeypatch) -> None:
    monkeypatch.setattr(
        evidence_judge,
        "agent_json",
        lambda messages, **kwargs: {
            "decisions": [{"action": "keep"}],
            "claim_assessments": [],
            "overall_sufficient": False,
        },
    )
    monkeypatch.setattr(evidence_judge, "get_settings", _settings)

    with pytest.raises(evidence_judge.EvidenceJudgeError) as exc_info:
        evidence_judge.judge_evidence_round(
            "Why OT?",
            _pool(),
            [_group([_candidate(71, "ablation")])],
            round_number=2,
        )

    error = exc_info.value
    assert error.category == "judge_schema_error"
    assert error.attempts == 2
    assert error.validation_errors
    assert len(error.validation_errors) <= 8
    assert all(set(item) == {"loc", "type"} for item in error.validation_errors)
    assert any(item["loc"] == "decisions.0.source_refs" for item in error.validation_errors)


def test_judge_reports_safe_action_rule_code_and_retries_with_it(monkeypatch) -> None:
    calls = []
    invalid_payload = _valid_payload()
    invalid_payload["decisions"][0].pop("statement")

    def fake_agent_json(messages, **kwargs):
        calls.append(messages)
        return invalid_payload

    monkeypatch.setattr(evidence_judge, "agent_json", fake_agent_json)
    monkeypatch.setattr(evidence_judge, "get_settings", _settings)

    with pytest.raises(evidence_judge.EvidenceJudgeError) as exc_info:
        evidence_judge.judge_evidence_round(
            "Why OT?",
            _pool(two_claims=True),
            [_group([_candidate(71, "ablation")])],
            round_number=2,
        )

    error = exc_info.value
    assert error.validation_errors == [
        {"loc": "decisions.0", "type": "keep_requires_statement"}
    ]
    assert "keep_requires_statement" in calls[1][-1]["content"]


def test_judge_does_not_expose_unknown_validator_message(monkeypatch) -> None:
    invalid_payload = _valid_payload()
    invalid_payload["decisions"][0]["reason"] = ""
    monkeypatch.setattr(
        evidence_judge,
        "agent_json",
        lambda messages, **kwargs: invalid_payload,
    )
    monkeypatch.setattr(evidence_judge, "get_settings", _settings)

    with pytest.raises(evidence_judge.EvidenceJudgeError) as exc_info:
        evidence_judge.judge_evidence_round(
            "Why OT?",
            _pool(two_claims=True),
            [_group([_candidate(71, "ablation")])],
            round_number=2,
        )

    assert exc_info.value.validation_errors == [
        {"loc": "decisions.0.reason", "type": "string_too_short"}
    ]


def test_judge_accepts_only_supplied_web_source(monkeypatch) -> None:
    source = WebSource(
        title="OT paper",
        authors=["A"],
        summary="A web abstract.",
        published="2024-01-01",
        entry_url="https://arxiv.org/abs/2401.00001",
    )
    payload = {
        "decisions": [
            {
                "source_refs": [{"kind": "web", "url": source.entry_url}],
                "action": "keep",
                "statement": "The paper studies balanced OT.",
                "supports_claim_ids": ["C1"],
                "confidence": 0.7,
                "reason": "Relevant abstract.",
            }
        ],
        "claim_assessments": [{"claim_id": "C1", "status": "covered"}],
        "overall_sufficient": True,
    }
    monkeypatch.setattr(evidence_judge, "agent_json", lambda messages, **kwargs: payload)
    monkeypatch.setattr(evidence_judge, "get_settings", _settings)

    result = evidence_judge.judge_evidence_round(
        "Why OT?", _pool(), [_group(web_sources=[source])], round_number=2
    )

    assert result.decisions[0].source_refs[0].url == source.entry_url
