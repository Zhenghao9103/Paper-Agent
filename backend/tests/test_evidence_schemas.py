import pytest
from backend.app.schemas.evidence import (
    CandidateDecision,
    EvidenceItem,
    EvidenceJudgeResult,
    EvidencePool,
    LocalSourceRef,
    ResearchClaim,
    ResearchPlan,
    SearchAction,
    WebSourceRef,
)
from pydantic import ValidationError


def _claim(claim_id: str = "C1") -> ResearchClaim:
    return ResearchClaim(claim_id=claim_id, question=f"Question for {claim_id}")


def _local_evidence(
    evidence_id: str = "E1",
    claim_id: str = "C1",
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        statement="OT produces balanced assignments.",
        source_refs=[LocalSourceRef(chunk_id=21)],
        supports_claim_ids=[claim_id],
        confidence=0.9,
        round_added=1,
    )


def test_claim_ids_are_unique_and_bounded() -> None:
    with pytest.raises(ValidationError, match="claim IDs must be unique"):
        ResearchPlan(
            claims=[_claim("C1"), _claim("C1")],
            actions=[
                SearchAction(
                    action_id="A1",
                    claim_ids=["C1"],
                    tool="hybrid_search",
                    query="optimal transport",
                )
            ],
        )

    with pytest.raises(ValidationError):
        ResearchPlan(
            claims=[_claim(f"C{index}") for index in range(1, 9)] + [_claim("C9")],
            actions=[
                SearchAction(
                    action_id="A1",
                    claim_ids=["C1"],
                    tool="hybrid_search",
                    query="optimal transport",
                )
            ],
        )


@pytest.mark.parametrize("chunk_id", [0, -1, True])
def test_local_source_requires_positive_chunk_id(chunk_id: object) -> None:
    with pytest.raises(ValidationError):
        LocalSourceRef(chunk_id=chunk_id)


@pytest.mark.parametrize(
    "url",
    ["ftp://example.com/paper", "example.com/paper", "", "javascript:alert(1)"],
)
def test_web_source_requires_http_url(url: str) -> None:
    with pytest.raises(ValidationError, match="HTTP"):
        WebSourceRef(url=url)


@pytest.mark.parametrize("tool", ["hybrid_search", "search_arxiv"])
def test_search_action_requires_query_for_search_tools(tool: str) -> None:
    with pytest.raises(ValidationError, match="query"):
        SearchAction(action_id="A1", claim_ids=["C1"], tool=tool)


def test_neighbor_action_requires_chunk_id() -> None:
    with pytest.raises(ValidationError, match="chunk_id"):
        SearchAction(
            action_id="A1",
            claim_ids=["C1"],
            tool="get_chunk_neighbors",
        )


def test_evidence_item_keeps_source_provenance() -> None:
    item = EvidenceItem(
        evidence_id="E1",
        statement="OT is used for balanced assignments.",
        source_refs=[
            LocalSourceRef(chunk_id=21),
            WebSourceRef(url="https://arxiv.org/abs/2401.00001"),
        ],
        supports_claim_ids=["C1"],
        confidence=0.8,
        round_added=1,
    )

    assert item.source_refs[0].kind == "local"
    assert item.source_refs[0].chunk_id == 21
    assert item.source_refs[1].kind == "web"
    assert item.source_refs[1].url == "https://arxiv.org/abs/2401.00001"


def test_pool_rejects_unknown_claim_references() -> None:
    with pytest.raises(ValidationError, match="unknown claim"):
        EvidencePool(
            claims=[_claim("C1")],
            evidence={"E1": _local_evidence(claim_id="C2")},
        )


def test_pool_rejects_mismatched_evidence_dictionary_key() -> None:
    with pytest.raises(ValidationError, match="evidence dictionary key"):
        EvidencePool(
            claims=[_claim("C1")],
            evidence={"E2": _local_evidence(evidence_id="E1")},
        )


def test_judge_result_rejects_more_than_eight_gaps() -> None:
    with pytest.raises(ValidationError):
        EvidenceJudgeResult(
            decisions=[],
            claim_assessments=[],
            overall_sufficient=False,
            unresolved_gaps=[f"gap {index}" for index in range(9)],
        )


def test_candidate_decision_enforces_action_specific_fields() -> None:
    with pytest.raises(ValidationError, match="target_evidence_id"):
        CandidateDecision(
            source_refs=[LocalSourceRef(chunk_id=21)],
            action="merge",
            supports_claim_ids=["C1"],
            confidence=0.8,
            reason="Same proposition.",
        )

    with pytest.raises(ValidationError, match="statement"):
        CandidateDecision(
            source_refs=[LocalSourceRef(chunk_id=21)],
            action="keep",
            supports_claim_ids=["C1"],
            confidence=0.8,
            reason="Relevant evidence.",
        )


def test_research_plan_rejects_actions_for_unknown_claims() -> None:
    with pytest.raises(ValidationError, match="unknown claim"):
        ResearchPlan(
            claims=[_claim("C1")],
            actions=[
                SearchAction(
                    action_id="A1",
                    claim_ids=["C2"],
                    tool="hybrid_search",
                    query="ablation",
                )
            ],
        )


def test_structured_agent_contracts_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ResearchClaim.model_validate(
            {
                "claim_id": "C1",
                "question": "What role does OT play?",
                "unexpected": "must not be ignored",
            }
        )
