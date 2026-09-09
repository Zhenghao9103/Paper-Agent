from backend.app.schemas.evidence import (
    CandidateDecision,
    EvidenceItem,
    EvidenceJudgeResult,
    EvidencePool,
    EvidenceSourceRef,
    ResearchClaim,
)


class EvidencePoolUpdateError(ValueError):
    """Stable validation failure for an atomic Evidence Pool update."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


def _source_id(source: EvidenceSourceRef) -> str:
    return source.stable_id


def _append_unique_sources(
    existing: list[EvidenceSourceRef],
    incoming: list[EvidenceSourceRef],
) -> list[EvidenceSourceRef]:
    merged = list(existing)
    seen = {_source_id(source) for source in merged}
    for source in incoming:
        if _source_id(source) not in seen:
            merged.append(source)
            seen.add(_source_id(source))
    return merged


def _validate_result(
    pool: EvidencePool,
    result: EvidenceJudgeResult,
    allowed_source_refs: set[str],
) -> None:
    known_claims = {claim.claim_id for claim in pool.claims}
    known_evidence = set(pool.evidence)
    for decision in result.decisions:
        if any(_source_id(source) not in allowed_source_refs for source in decision.source_refs):
            raise EvidencePoolUpdateError("unknown_source_ref")
        if set(decision.supports_claim_ids) - known_claims:
            raise EvidencePoolUpdateError("unknown_claim_id")
        if (
            decision.action in {"merge", "conflict"}
            and decision.target_evidence_id not in known_evidence
        ):
            raise EvidencePoolUpdateError("unknown_target_evidence_id")
    if any(
        assessment.claim_id not in known_claims
        for assessment in result.claim_assessments
    ):
        raise EvidencePoolUpdateError("unknown_claim_id")


def _new_item(
    pool: EvidencePool,
    decision: CandidateDecision,
    *,
    round_number: int,
    conflicts_with: list[str] | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"E{pool.next_evidence_number}",
        statement=decision.statement or "",
        source_refs=decision.source_refs,
        supports_claim_ids=decision.supports_claim_ids,
        confidence=decision.confidence,
        conflicts_with=conflicts_with or [],
        round_added=round_number,
    )


def _apply_decision(
    pool: EvidencePool,
    decision: CandidateDecision,
    *,
    round_number: int,
) -> None:
    if decision.action == "drop":
        pool.rejected_source_refs = _append_unique_sources(
            pool.rejected_source_refs,
            decision.source_refs,
        )
        return

    if decision.action == "merge":
        target_id = decision.target_evidence_id
        if target_id is None:
            raise EvidencePoolUpdateError("unknown_target_evidence_id")
        target = pool.evidence[target_id]
        pool.evidence[target_id] = target.model_copy(
            update={
                "source_refs": _append_unique_sources(
                    target.source_refs,
                    decision.source_refs,
                ),
                "supports_claim_ids": list(
                    dict.fromkeys(target.supports_claim_ids + decision.supports_claim_ids)
                ),
                "confidence": max(target.confidence, decision.confidence),
            }
        )
        return

    target_id = decision.target_evidence_id if decision.action == "conflict" else None
    item = _new_item(
        pool,
        decision,
        round_number=round_number,
        conflicts_with=[target_id] if target_id else [],
    )
    pool.evidence[item.evidence_id] = item
    pool.next_evidence_number += 1

    if target_id is not None:
        target = pool.evidence[target_id]
        pool.evidence[target_id] = target.model_copy(
            update={
                "conflicts_with": list(
                    dict.fromkeys(target.conflicts_with + [item.evidence_id])
                )
            }
        )
        pair = (target_id, item.evidence_id)
        if pair not in pool.unresolved_conflicts:
            pool.unresolved_conflicts.append(pair)


def _update_claims(
    pool: EvidencePool,
    result: EvidenceJudgeResult,
) -> list[ResearchClaim]:
    assessments = {
        assessment.claim_id: assessment for assessment in result.claim_assessments
    }
    conflicted_ids = {
        evidence_id
        for pair in pool.unresolved_conflicts
        for evidence_id in pair
    }
    updated: list[ResearchClaim] = []
    for claim in pool.claims:
        evidence_ids = [
            evidence_id
            for evidence_id, item in pool.evidence.items()
            if claim.claim_id in item.supports_claim_ids
        ]
        assessment = assessments.get(claim.claim_id)
        has_conflict = any(evidence_id in conflicted_ids for evidence_id in evidence_ids)
        if has_conflict:
            status = "conflicted"
        elif not evidence_ids:
            status = "missing"
        elif assessment is not None and assessment.status == "covered":
            status = "covered"
        elif assessment is None and claim.status == "covered":
            status = "covered"
        else:
            status = "partial"
        gap = assessment.gap if assessment is not None else claim.gap
        if status == "covered":
            gap = ""
        updated.append(
            claim.model_copy(
                update={
                    "status": status,
                    "evidence_ids": evidence_ids,
                    "gap": gap,
                }
            )
        )
    return updated


def apply_judge_result(
    pool: EvidencePool,
    result: EvidenceJudgeResult,
    *,
    allowed_source_refs: set[str],
    round_number: int,
) -> EvidencePool:
    """Return a new pool after an all-or-nothing Judge update."""

    _validate_result(pool, result, allowed_source_refs)
    updated = pool.model_copy(deep=True)
    for decision in result.decisions:
        _apply_decision(updated, decision, round_number=round_number)
    updated.claims = _update_claims(updated, result)
    try:
        return EvidencePool.model_validate(updated.model_dump())
    except ValueError as exc:
        raise EvidencePoolUpdateError("invalid_pool_update") from exc


def pool_is_sufficient(pool: EvidencePool) -> bool:
    return all(
        not claim.required or claim.status == "covered"
        for claim in pool.claims
    ) and not pool.unresolved_conflicts
