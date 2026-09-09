from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

ClaimStatus = Literal["missing", "partial", "covered", "conflicted"]
EvidenceAction = Literal["keep", "drop", "merge", "conflict"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _unique(values: list[str], *, label: str) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    return values


class LocalSourceRef(_StrictModel):
    kind: Literal["local"] = "local"
    chunk_id: StrictInt = Field(gt=0)

    @property
    def stable_id(self) -> str:
        return f"chunk:{self.chunk_id}"


class WebSourceRef(_StrictModel):
    kind: Literal["web"] = "web"
    url: str = Field(min_length=1, max_length=2000)

    @field_validator("url", mode="before")
    @classmethod
    def require_http_url(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("web source must use an HTTP(S) URL")
        return normalized

    @property
    def stable_id(self) -> str:
        return f"url:{self.url}"


EvidenceSourceRef = Annotated[
    LocalSourceRef | WebSourceRef,
    Field(discriminator="kind"),
]


class ResearchClaim(_StrictModel):
    claim_id: str = Field(pattern=r"^C[1-8]$")
    question: str = Field(min_length=1, max_length=1000)
    required: bool = True
    status: ClaimStatus = "missing"
    evidence_ids: list[str] = Field(default_factory=list)
    gap: str = Field(default="", max_length=1000)

    @field_validator("question", "gap")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence_ids(cls, values: list[str]) -> list[str]:
        return _unique(values, label="claim evidence IDs")


class EvidenceItem(_StrictModel):
    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    statement: str = Field(min_length=1, max_length=1500)
    source_refs: list[EvidenceSourceRef] = Field(min_length=1)
    supports_claim_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    conflicts_with: list[str] = Field(default_factory=list)
    round_added: int = Field(ge=1, le=3)

    @field_validator("statement")
    @classmethod
    def strip_statement(cls, value: str) -> str:
        return value.strip()

    @field_validator("supports_claim_ids", "conflicts_with")
    @classmethod
    def unique_ids(cls, values: list[str], info: Any) -> list[str]:
        return _unique(values, label=info.field_name.replace("_", " "))

    @field_validator("source_refs")
    @classmethod
    def unique_sources(cls, values: list[EvidenceSourceRef]) -> list[EvidenceSourceRef]:
        stable_ids = [value.stable_id for value in values]
        _unique(stable_ids, label="source references")
        return values


class EvidencePool(_StrictModel):
    protocol_version: Literal[2] = 2
    claims: list[ResearchClaim] = Field(min_length=1, max_length=8)
    evidence: dict[str, EvidenceItem] = Field(default_factory=dict)
    rejected_source_refs: list[EvidenceSourceRef] = Field(default_factory=list)
    unresolved_conflicts: list[tuple[str, str]] = Field(default_factory=list)
    next_evidence_number: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_references(self) -> "EvidencePool":
        claim_ids = [claim.claim_id for claim in self.claims]
        _unique(claim_ids, label="claim IDs")
        known_claims = set(claim_ids)
        known_evidence = set(self.evidence)

        for key, item in self.evidence.items():
            if key != item.evidence_id:
                raise ValueError("evidence dictionary key must match evidence_id")
            unknown_claims = set(item.supports_claim_ids) - known_claims
            if unknown_claims:
                raise ValueError("evidence references an unknown claim")
            if set(item.conflicts_with) - known_evidence:
                raise ValueError("evidence conflicts with an unknown evidence item")

        for claim in self.claims:
            if set(claim.evidence_ids) - known_evidence:
                raise ValueError("claim references an unknown evidence item")
            if claim.status == "covered" and not claim.evidence_ids:
                raise ValueError("covered claim requires active evidence")

        for left, right in self.unresolved_conflicts:
            if left == right or left not in known_evidence or right not in known_evidence:
                raise ValueError("unresolved conflict references invalid evidence")
            if right not in self.evidence[left].conflicts_with:
                raise ValueError("unresolved conflict must be symmetric")
            if left not in self.evidence[right].conflicts_with:
                raise ValueError("unresolved conflict must be symmetric")
        return self


class SearchAction(_StrictModel):
    action_id: str = Field(pattern=r"^A[1-9][0-9]*$")
    claim_ids: list[str] = Field(min_length=1, max_length=8)
    tool: Literal[
        "hybrid_search",
        "get_chunk_neighbors",
        "inspect_document",
        "search_arxiv",
    ]
    query: str | None = Field(default=None, max_length=1000)
    document_id: int | None = Field(default=None, gt=0)
    chunk_id: int | None = Field(default=None, gt=0)

    @field_validator("claim_ids")
    @classmethod
    def unique_claim_ids(cls, values: list[str]) -> list[str]:
        return _unique(values, label="action claim IDs")

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized or None

    @model_validator(mode="after")
    def validate_tool_arguments(self) -> "SearchAction":
        if self.tool in {"hybrid_search", "search_arxiv"} and not self.query:
            raise ValueError(f"{self.tool} requires query")
        if self.tool == "get_chunk_neighbors" and self.chunk_id is None:
            raise ValueError("get_chunk_neighbors requires chunk_id")
        return self


class ResearchPlan(_StrictModel):
    claims: list[ResearchClaim] = Field(min_length=1, max_length=8)
    actions: list[SearchAction] = Field(min_length=1, max_length=5)
    rationale: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def validate_ids(self) -> "ResearchPlan":
        claim_ids = [claim.claim_id for claim in self.claims]
        _unique(claim_ids, label="claim IDs")
        _unique([action.action_id for action in self.actions], label="action IDs")
        known_claims = set(claim_ids)
        if any(set(action.claim_ids) - known_claims for action in self.actions):
            raise ValueError("search action references an unknown claim")
        return self


class CandidateDecision(_StrictModel):
    source_refs: list[EvidenceSourceRef] = Field(min_length=1)
    action: EvidenceAction
    target_evidence_id: str | None = Field(default=None, pattern=r"^E[1-9][0-9]*$")
    statement: str | None = Field(default=None, max_length=1500)
    supports_claim_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=1000)

    @field_validator("statement", "reason")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip()

    @field_validator("supports_claim_ids")
    @classmethod
    def unique_claim_ids(cls, values: list[str]) -> list[str]:
        return _unique(values, label="decision claim IDs")

    @model_validator(mode="after")
    def validate_action_fields(self) -> "CandidateDecision":
        if self.action in {"merge", "conflict"} and self.target_evidence_id is None:
            raise ValueError(f"{self.action} requires target_evidence_id")
        if self.action in {"keep", "conflict"} and not self.statement:
            raise ValueError(f"{self.action} requires statement")
        if self.action != "drop" and not self.supports_claim_ids:
            raise ValueError(f"{self.action} requires supports_claim_ids")
        return self


class ClaimAssessment(_StrictModel):
    claim_id: str = Field(pattern=r"^C[1-8]$")
    status: ClaimStatus
    evidence_ids: list[str] = Field(default_factory=list)
    gap: str = Field(default="", max_length=1000)

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence_ids(cls, values: list[str]) -> list[str]:
        return _unique(values, label="assessment evidence IDs")


class EvidenceJudgeResult(_StrictModel):
    decisions: list[CandidateDecision]
    claim_assessments: list[ClaimAssessment]
    overall_sufficient: bool
    unresolved_gaps: list[str] = Field(default_factory=list, max_length=8)
    next_search_focus: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_assessment_ids(self) -> "EvidenceJudgeResult":
        _unique(
            [assessment.claim_id for assessment in self.claim_assessments],
            label="claim assessment IDs",
        )
        return self


class PackedLocalSource(_StrictModel):
    kind: Literal["local"] = "local"
    chunk_id: StrictInt = Field(gt=0)
    document_id: StrictInt = Field(gt=0)
    title: str = Field(max_length=500)
    page_number: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    excerpt: str = Field(min_length=1)
    score: float = 0.0


class PackedWebSource(_StrictModel):
    kind: Literal["web"] = "web"
    url: str = Field(min_length=1, max_length=2000)
    pdf_url: str | None = Field(default=None, max_length=2000)
    title: str = Field(max_length=500)
    authors: list[str] = Field(default_factory=list)
    published: str = ""
    excerpt: str = Field(min_length=1)


PackedEvidenceSource = Annotated[
    PackedLocalSource | PackedWebSource,
    Field(discriminator="kind"),
]


class EvidencePackItem(_StrictModel):
    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    statement: str = Field(min_length=1, max_length=1500)
    supports_claim_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    conflicts_with: list[str] = Field(default_factory=list)
    sources: list[PackedEvidenceSource] = Field(min_length=1)


class EvidencePack(_StrictModel):
    protocol_version: Literal[2] = 2
    claims: list[ResearchClaim] = Field(min_length=1, max_length=8)
    items: list[EvidencePackItem] = Field(min_length=1)
    token_count: int = Field(ge=1)
