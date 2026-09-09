import pytest
from backend.app.agents.evidence_pack import EvidencePackError, build_evidence_pack
from backend.app.schemas.evidence import (
    EvidenceItem,
    EvidencePool,
    LocalSourceRef,
    ResearchClaim,
)
from backend.app.schemas.retrieval import EvidenceLedger, RetrievalCandidate


def _candidate(chunk_id: int, content: str | None = None) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=7,
        title="OT Paper",
        page_number=4,
        chunk_index=chunk_id,
        content=content or (f"Evidence content {chunk_id}. " * 20),
        rerank_score=1 - chunk_id / 1000,
    )


def _pool_and_ledger(count: int = 2) -> tuple[EvidencePool, EvidenceLedger]:
    evidence = {}
    ledger = EvidenceLedger()
    for index in range(1, count + 1):
        evidence_id = f"E{index}"
        chunk_id = 20 + index
        evidence[evidence_id] = EvidenceItem(
            evidence_id=evidence_id,
            statement=f"Evidence statement {index}.",
            source_refs=[LocalSourceRef(chunk_id=chunk_id)],
            supports_claim_ids=["C1" if index == 1 else "C2"],
            confidence=1 - index / 100,
            round_added=1,
        )
        ledger.add_local(_candidate(chunk_id))
    claims = [
        ResearchClaim(
            claim_id="C1",
            question="What does OT do?",
            status="covered",
            evidence_ids=["E1"],
        )
    ]
    if count >= 2:
        claims.append(
            ResearchClaim(
                claim_id="C2",
                question="Is it experimentally validated?",
                status="covered",
                evidence_ids=[f"E{index}" for index in range(2, count + 1)],
            )
        )
    return EvidencePool(
        claims=claims,
        evidence=evidence,
        next_evidence_number=count + 1,
    ), ledger


def test_pack_includes_every_covered_required_claim() -> None:
    pool, ledger = _pool_and_ledger()

    pack = build_evidence_pack(pool, ledger, token_budget=12000, model="unknown-model")

    covered = {claim_id for item in pack.items for claim_id in item.supports_claim_ids}
    assert covered == {"C1", "C2"}
    assert pack.token_count <= 12000


def test_pack_preserves_both_sides_of_a_disclosed_conflict() -> None:
    pool, ledger = _pool_and_ledger()
    pool.evidence["E1"].conflicts_with = ["E2"]
    pool.evidence["E2"].conflicts_with = ["E1"]
    pool.unresolved_conflicts = [("E1", "E2")]
    pool.claims[0].status = "conflicted"

    pack = build_evidence_pack(pool, ledger, token_budget=12000, model="unknown-model")

    assert {item.evidence_id for item in pack.items} >= {"E1", "E2"}


def test_pack_uses_more_than_eight_items_when_they_fit() -> None:
    pool, ledger = _pool_and_ledger(count=9)

    pack = build_evidence_pack(pool, ledger, token_budget=50000, model="unknown-model")

    assert len(pack.items) == 9


def test_pack_compresses_excerpts_before_dropping_evidence() -> None:
    pool, ledger = _pool_and_ledger(count=3)
    for candidate in ledger.local.values():
        candidate.content = f"chunk-{candidate.chunk_id} " * 2000

    pack = build_evidence_pack(pool, ledger, token_budget=1600, model="unknown-model")

    assert len(pack.items) == 3
    assert all(len(source.excerpt) < 4000 for item in pack.items for source in item.sources)
    assert pack.token_count <= 1600


def test_pack_never_drops_the_only_evidence_for_a_claim() -> None:
    pool, ledger = _pool_and_ledger(count=2)

    pack = build_evidence_pack(pool, ledger, token_budget=900, model="unknown-model")

    assert {item.evidence_id for item in pack.items} == {"E1", "E2"}


def test_pack_fails_when_minimum_claim_coverage_cannot_fit() -> None:
    pool, ledger = _pool_and_ledger(count=2)

    with pytest.raises(EvidencePackError) as exc_info:
        build_evidence_pack(pool, ledger, token_budget=10, model="unknown-model")

    assert exc_info.value.category == "minimum_claim_coverage_exceeds_budget"


def test_pack_token_count_is_reproducible_for_unknown_model_name() -> None:
    pool, ledger = _pool_and_ledger(count=2)

    first = build_evidence_pack(pool, ledger, token_budget=12000, model="vendor-new-model")
    second = build_evidence_pack(pool, ledger, token_budget=12000, model="vendor-new-model")

    assert first.token_count == second.token_count
    assert first.model_dump() == second.model_dump()


def test_pack_claims_never_reference_dropped_optional_evidence() -> None:
    pool, ledger = _pool_and_ledger(count=3)
    for candidate in ledger.local.values():
        candidate.content = f"chunk-{candidate.chunk_id} " * 2000

    pack = build_evidence_pack(pool, ledger, token_budget=450, model="unknown-model")

    packed_ids = {item.evidence_id for item in pack.items}
    assert len(pack.items) == 2
    assert all(
        set(claim.evidence_ids) <= packed_ids
        for claim in pack.claims
    )
