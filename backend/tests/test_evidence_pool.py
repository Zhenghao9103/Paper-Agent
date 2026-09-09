import pytest
from backend.app.agents.evidence_pool import (
    EvidencePoolUpdateError,
    apply_judge_result,
    pool_is_sufficient,
)
from backend.app.schemas.evidence import (
    CandidateDecision,
    ClaimAssessment,
    EvidenceItem,
    EvidenceJudgeResult,
    EvidencePool,
    LocalSourceRef,
    ResearchClaim,
)


def _pool(*, two_claims: bool = False, with_evidence: bool = False) -> EvidencePool:
    claims = [ResearchClaim(claim_id="C1", question="What does OT do?")]
    evidence = {}
    if with_evidence:
        evidence["E1"] = EvidenceItem(
            evidence_id="E1",
            statement="OT creates balanced assignments.",
            source_refs=[LocalSourceRef(chunk_id=21)],
            supports_claim_ids=["C1"],
            confidence=0.9,
            round_added=1,
        )
        claims[0] = claims[0].model_copy(
            update={"status": "covered", "evidence_ids": ["E1"]}
        )
    if two_claims:
        claims.append(ResearchClaim(claim_id="C2", question="Is there an ablation?"))
    return EvidencePool(
        claims=claims,
        evidence=evidence,
        next_evidence_number=2 if with_evidence else 1,
    )


def _result(
    decision: CandidateDecision,
    *,
    assessments: list[ClaimAssessment] | None = None,
    sufficient: bool = False,
) -> EvidenceJudgeResult:
    return EvidenceJudgeResult(
        decisions=[decision],
        claim_assessments=assessments
        if assessments is not None
        else [
            ClaimAssessment(
                claim_id="C1",
                status="covered" if decision.action != "drop" else "missing",
                gap="" if decision.action != "drop" else "Need direct evidence.",
            )
        ],
        overall_sufficient=sufficient,
    )


def test_keep_assigns_system_evidence_id_and_updates_claim() -> None:
    pool = _pool()
    result = _result(
        CandidateDecision(
            source_refs=[LocalSourceRef(chunk_id=21)],
            action="keep",
            statement="OT creates balanced assignments.",
            supports_claim_ids=["C1"],
            confidence=0.9,
            reason="Directly supports the mechanism.",
        ),
        sufficient=True,
    )

    updated = apply_judge_result(
        pool,
        result,
        allowed_source_refs={"chunk:21"},
        round_number=1,
    )

    assert pool.evidence == {}
    assert list(updated.evidence) == ["E1"]
    assert updated.claims[0].status == "covered"
    assert updated.claims[0].evidence_ids == ["E1"]
    assert updated.next_evidence_number == 2
    assert pool_is_sufficient(updated) is True


def test_drop_keeps_audit_ref_but_not_active_evidence() -> None:
    updated = apply_judge_result(
        _pool(),
        _result(
            CandidateDecision(
                source_refs=[LocalSourceRef(chunk_id=22)],
                action="drop",
                confidence=0.1,
                reason="Does not answer the claim.",
            )
        ),
        allowed_source_refs={"chunk:22"},
        round_number=1,
    )

    assert updated.evidence == {}
    assert [ref.stable_id for ref in updated.rejected_source_refs] == ["chunk:22"]
    assert updated.claims[0].status == "missing"


def test_merge_adds_unique_sources_without_losing_original_refs() -> None:
    result = _result(
        CandidateDecision(
            source_refs=[LocalSourceRef(chunk_id=21), LocalSourceRef(chunk_id=22)],
            action="merge",
            target_evidence_id="E1",
            supports_claim_ids=["C1"],
            confidence=0.8,
            reason="Adds complementary support.",
        )
    )

    updated = apply_judge_result(
        _pool(with_evidence=True),
        result,
        allowed_source_refs={"chunk:21", "chunk:22"},
        round_number=2,
    )

    assert [ref.stable_id for ref in updated.evidence["E1"].source_refs] == [
        "chunk:21",
        "chunk:22",
    ]
    assert updated.evidence["E1"].statement == "OT creates balanced assignments."


def test_conflict_preserves_both_items_and_marks_claim_conflicted() -> None:
    result = _result(
        CandidateDecision(
            source_refs=[LocalSourceRef(chunk_id=23)],
            action="conflict",
            target_evidence_id="E1",
            statement="The assignments are not constrained to be balanced.",
            supports_claim_ids=["C1"],
            confidence=0.7,
            reason="Contradicts E1.",
        ),
        assessments=[
            ClaimAssessment(
                claim_id="C1",
                status="conflicted",
                gap="Resolve the assignment constraint conflict.",
            )
        ],
    )

    updated = apply_judge_result(
        _pool(with_evidence=True),
        result,
        allowed_source_refs={"chunk:23"},
        round_number=2,
    )

    assert set(updated.evidence) == {"E1", "E2"}
    assert updated.evidence["E1"].conflicts_with == ["E2"]
    assert updated.evidence["E2"].conflicts_with == ["E1"]
    assert updated.unresolved_conflicts == [("E1", "E2")]
    assert updated.claims[0].status == "conflicted"
    assert pool_is_sufficient(updated) is False


def test_unknown_chunk_reference_rejects_entire_judge_update() -> None:
    pool = _pool()
    before = pool.model_dump()
    result = _result(
        CandidateDecision(
            source_refs=[LocalSourceRef(chunk_id=999)],
            action="keep",
            statement="Unsupported source reference.",
            supports_claim_ids=["C1"],
            confidence=0.9,
            reason="Invalid source.",
        )
    )

    with pytest.raises(EvidencePoolUpdateError) as exc_info:
        apply_judge_result(
            pool,
            result,
            allowed_source_refs={"chunk:21"},
            round_number=1,
        )

    assert exc_info.value.category == "unknown_source_ref"
    assert pool.model_dump() == before


def test_unknown_claim_reference_rejects_entire_judge_update() -> None:
    pool = _pool()
    before = pool.model_dump()
    result = _result(
        CandidateDecision(
            source_refs=[LocalSourceRef(chunk_id=21)],
            action="keep",
            statement="Evidence for an invented claim.",
            supports_claim_ids=["C2"],
            confidence=0.8,
            reason="Invalid claim.",
        )
    )

    with pytest.raises(EvidencePoolUpdateError) as exc_info:
        apply_judge_result(
            pool,
            result,
            allowed_source_refs={"chunk:21"},
            round_number=1,
        )

    assert exc_info.value.category == "unknown_claim_id"
    assert pool.model_dump() == before


def test_unknown_target_rejects_entire_judge_update() -> None:
    pool = _pool()
    result = _result(
        CandidateDecision(
            source_refs=[LocalSourceRef(chunk_id=21)],
            action="merge",
            target_evidence_id="E9",
            supports_claim_ids=["C1"],
            confidence=0.8,
            reason="Invalid merge target.",
        )
    )

    with pytest.raises(EvidencePoolUpdateError) as exc_info:
        apply_judge_result(
            pool,
            result,
            allowed_source_refs={"chunk:21"},
            round_number=1,
        )

    assert exc_info.value.category == "unknown_target_evidence_id"


def test_false_sufficiency_is_normalized_to_false() -> None:
    pool = _pool(two_claims=True)
    result = _result(
        CandidateDecision(
            source_refs=[LocalSourceRef(chunk_id=21)],
            action="keep",
            statement="OT creates balanced assignments.",
            supports_claim_ids=["C1"],
            confidence=0.9,
            reason="Covers only C1.",
        ),
        assessments=[
            ClaimAssessment(claim_id="C1", status="covered"),
            ClaimAssessment(claim_id="C2", status="missing", gap="Need ablation."),
        ],
        sufficient=True,
    )

    updated = apply_judge_result(
        pool,
        result,
        allowed_source_refs={"chunk:21"},
        round_number=1,
    )

    assert pool_is_sufficient(updated) is False
    assert updated.claims[1].status == "missing"


def test_all_required_claims_covered_is_sufficient() -> None:
    pool = _pool(two_claims=True, with_evidence=True)
    result = _result(
        CandidateDecision(
            source_refs=[LocalSourceRef(chunk_id=71)],
            action="keep",
            statement="The ablation degrades clustering accuracy.",
            supports_claim_ids=["C2"],
            confidence=0.95,
            reason="Direct ablation result.",
        ),
        assessments=[
            ClaimAssessment(claim_id="C1", status="covered", evidence_ids=["E1"]),
            ClaimAssessment(claim_id="C2", status="covered"),
        ],
        sufficient=True,
    )

    updated = apply_judge_result(
        pool,
        result,
        allowed_source_refs={"chunk:71"},
        round_number=2,
    )

    assert pool_is_sufficient(updated) is True
    assert [claim.status for claim in updated.claims] == ["covered", "covered"]
