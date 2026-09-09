import json

from ..schemas.evidence import (
    EvidenceItem,
    EvidencePack,
    EvidencePackItem,
    EvidencePool,
    PackedEvidenceSource,
    PackedLocalSource,
    PackedWebSource,
)
from ..schemas.retrieval import EvidenceLedger
from ..services.context_checkpoint import count_text_tokens

_MAX_EXCERPT_CHARS = 4000
_MIN_EXCERPT_CHARS = 160


class EvidencePackError(ValueError):
    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


def _score(item) -> float:
    return float(
        item.rerank_score
        if item.rerank_score is not None
        else item.fusion_score
    )


def _pack_sources(
    item: EvidenceItem,
    ledger: EvidenceLedger,
) -> list[PackedEvidenceSource]:
    sources: list[PackedEvidenceSource] = []
    for source_ref in item.source_refs:
        if source_ref.kind == "local":
            candidate = ledger.local.get(source_ref.chunk_id)
            if candidate is None:
                raise EvidencePackError("missing_local_source")
            sources.append(
                PackedLocalSource(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    title=candidate.title[:500],
                    page_number=candidate.page_number,
                    chunk_index=candidate.chunk_index,
                    excerpt=candidate.content[:_MAX_EXCERPT_CHARS],
                    score=_score(candidate),
                )
            )
            continue
        web_source = ledger.web.get(source_ref.url)
        if web_source is None:
            raise EvidencePackError("missing_web_source")
        sources.append(
            PackedWebSource(
                url=web_source.entry_url,
                pdf_url=web_source.pdf_url,
                title=web_source.title[:500],
                authors=web_source.authors,
                published=web_source.published,
                excerpt=web_source.summary[:_MAX_EXCERPT_CHARS],
            )
        )
    return sources


def _pack_item(item: EvidenceItem, ledger: EvidenceLedger) -> EvidencePackItem:
    return EvidencePackItem(
        evidence_id=item.evidence_id,
        statement=item.statement,
        supports_claim_ids=item.supports_claim_ids,
        confidence=item.confidence,
        conflicts_with=item.conflicts_with,
        sources=_pack_sources(item, ledger),
    )


def _minimum_evidence_ids(pool: EvidencePool) -> list[str]:
    selected: list[str] = []
    for claim in pool.claims:
        if not claim.required:
            continue
        if claim.status not in {"covered", "conflicted"} or not claim.evidence_ids:
            raise EvidencePackError("insufficient_claim_coverage")
        candidates = [pool.evidence[evidence_id] for evidence_id in claim.evidence_ids]
        best = sorted(candidates, key=lambda item: (-item.confidence, item.evidence_id))[0]
        if best.evidence_id not in selected:
            selected.append(best.evidence_id)
    return selected


def _ordered_evidence_ids(pool: EvidencePool) -> tuple[list[str], set[str]]:
    minimum = _minimum_evidence_ids(pool)
    mandatory = set(minimum)
    ordered = list(minimum)
    for left, right in pool.unresolved_conflicts:
        for evidence_id in (left, right):
            mandatory.add(evidence_id)
            if evidence_id not in ordered:
                ordered.append(evidence_id)
    optional = sorted(
        (item for item in pool.evidence.values() if item.evidence_id not in ordered),
        key=lambda item: (-item.confidence, item.evidence_id),
    )
    ordered.extend(item.evidence_id for item in optional)
    return ordered, mandatory


def _pack_token_count(pack: EvidencePack, model: str) -> int:
    payload = pack.model_dump(mode="json", exclude={"token_count"})
    return count_text_tokens(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        model,
    )


def _shorten_one_excerpt(pack: EvidencePack) -> bool:
    for item_index in range(len(pack.items) - 1, -1, -1):
        item = pack.items[item_index]
        for source_index in range(len(item.sources) - 1, -1, -1):
            source = item.sources[source_index]
            if len(source.excerpt) <= _MIN_EXCERPT_CHARS:
                continue
            shortened_length = max(_MIN_EXCERPT_CHARS, len(source.excerpt) // 2)
            shortened = source.model_copy(
                update={"excerpt": source.excerpt[:shortened_length]}
            )
            sources = list(item.sources)
            sources[source_index] = shortened
            pack.items[item_index] = item.model_copy(update={"sources": sources})
            return True
    return False


def _sync_claim_evidence_ids(pack: EvidencePack) -> None:
    packed_ids = {item.evidence_id for item in pack.items}
    pack.claims = [
        claim.model_copy(
            update={
                "evidence_ids": [
                    evidence_id
                    for evidence_id in claim.evidence_ids
                    if evidence_id in packed_ids
                ]
            }
        )
        for claim in pack.claims
    ]


def build_evidence_pack(
    pool: EvidencePool,
    ledger: EvidenceLedger,
    *,
    token_budget: int,
    model: str,
) -> EvidencePack:
    if token_budget <= 0:
        raise EvidencePackError("invalid_token_budget")
    ordered_ids, mandatory_ids = _ordered_evidence_ids(pool)
    pack = EvidencePack(
        claims=pool.claims,
        items=[_pack_item(pool.evidence[evidence_id], ledger) for evidence_id in ordered_ids],
        token_count=1,
    )
    pack.token_count = _pack_token_count(pack, model)
    while pack.token_count > token_budget:
        if _shorten_one_excerpt(pack):
            pack.token_count = _pack_token_count(pack, model)
            continue
        removable_index = next(
            (
                index
                for index in range(len(pack.items) - 1, -1, -1)
                if pack.items[index].evidence_id not in mandatory_ids
            ),
            None,
        )
        if removable_index is None:
            raise EvidencePackError("minimum_claim_coverage_exceeds_budget")
        pack.items.pop(removable_index)
        _sync_claim_evidence_ids(pack)
        pack.token_count = _pack_token_count(pack, model)
    return pack
