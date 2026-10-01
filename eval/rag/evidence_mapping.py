"""Deterministically bind stable gold evidence text to runtime chunks."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from backend.app.models.chunk import DocumentChunk
from .rag_gold import RAGGoldCase


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MappedEvidence(_StrictModel):
    evidence_index: int = Field(ge=0)
    document_key: str
    page_number: int = Field(ge=1)
    chunk_ids: list[int]
    best_score: float = Field(ge=0, le=1)


class EvidenceMappingResult(_StrictModel):
    evidence_units: list[MappedEvidence]
    relevance_by_chunk: dict[int, int]
    unmapped: list[int]


def normalize_evidence_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _tokens(value: str) -> set[str]:
    normalized = normalize_evidence_text(value)
    return set(normalized.split()) if normalized else set()


def _match_score(evidence: str, content: str) -> float:
    left = normalize_evidence_text(evidence)
    right = normalize_evidence_text(content)
    if not left or not right:
        return 0.0
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    overlap = len(left_tokens & right_tokens)
    precision = overlap / len(right_tokens) if right_tokens else 0.0
    recall = overlap / len(left_tokens) if left_tokens else 0.0
    token_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return max(token_f1, SequenceMatcher(None, left, right).ratio())


def map_gold_evidence(
    *,
    case: RAGGoldCase | Any,
    chunks: Sequence[DocumentChunk | Any],
    document_ids: Mapping[str, int],
    minimum_score: float = 0.35,
) -> EvidenceMappingResult:
    if not 0 <= minimum_score <= 1:
        raise ValueError("minimum_score must be between 0 and 1")

    relevance: dict[int, int] = {}
    mapped: list[MappedEvidence] = []
    unmapped: list[int] = []
    keywords = [normalize_evidence_text(item) for item in case.gold_keywords]

    for index, evidence in enumerate(case.evidence):
        document_id = document_ids.get(evidence.doc)
        eligible = [
            chunk
            for chunk in chunks
            if document_id is not None
            and int(chunk.document_id) == int(document_id)
            and int(chunk.page_number) == int(evidence.page)
        ]
        scored = sorted(
            ((_match_score(evidence.text, chunk.content), chunk) for chunk in eligible),
            key=lambda item: (-item[0], int(item[1].chunk_index), int(item[1].id)),
        )
        if not scored or scored[0][0] < minimum_score:
            # Annotated evidence sentences are frequently paraphrases that
            # never overlap chunk text; fall back to gold keywords, binding
            # every same-page chunk that mentions one as weakly relevant.
            keyword_chunk_ids = [
                int(chunk.id)
                for chunk in eligible
                if any(
                    keyword and keyword in normalize_evidence_text(chunk.content)
                    for keyword in keywords
                )
            ]
            if keyword_chunk_ids:
                for chunk_id in keyword_chunk_ids:
                    relevance[chunk_id] = max(relevance.get(chunk_id, 0), 1)
                mapped.append(
                    MappedEvidence(
                        evidence_index=index,
                        document_key=evidence.doc,
                        page_number=evidence.page,
                        chunk_ids=keyword_chunk_ids,
                        best_score=scored[0][0] if scored else 0.0,
                    )
                )
            else:
                unmapped.append(index)
            continue

        best_score, best = scored[0]
        best_id = int(best.id)
        relevance[best_id] = max(relevance.get(best_id, 0), 3)
        mapped.append(
            MappedEvidence(
                evidence_index=index,
                document_key=evidence.doc,
                page_number=evidence.page,
                chunk_ids=[best_id],
                best_score=best_score,
            )
        )

        for _, chunk in scored[1:]:
            content = normalize_evidence_text(chunk.content)
            if any(keyword and keyword in content for keyword in keywords):
                chunk_id = int(chunk.id)
                relevance[chunk_id] = max(relevance.get(chunk_id, 0), 1)

    return EvidenceMappingResult(
        evidence_units=mapped,
        relevance_by_chunk=relevance,
        unmapped=unmapped,
    )


def load_case_chunks(
    db: Session,
    case: RAGGoldCase,
    document_ids: Mapping[str, int],
) -> list[DocumentChunk]:
    clauses = [
        (DocumentChunk.document_id == document_ids[reference.doc])
        & (DocumentChunk.page_number == page)
        for reference in case.gold
        if reference.doc in document_ids
        for page in reference.pages
    ]
    if not clauses:
        return []
    return list(
        db.scalars(
            select(DocumentChunk)
            .where(or_(*clauses))
            .order_by(
                DocumentChunk.document_id,
                DocumentChunk.page_number,
                DocumentChunk.chunk_index,
                DocumentChunk.id,
            )
        )
    )
