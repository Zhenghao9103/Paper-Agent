from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .chat import WebSource

IntentName = Literal["direct", "simple_rag", "agentic_rag"]


def _unique_casefold(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


class QueryPlan(BaseModel):
    intent: IntentName
    confidence: float = Field(ge=0, le=1)
    standalone_query: str = Field(min_length=1, max_length=1000)
    lexical_terms: list[str] = Field(default_factory=list, max_length=24)
    synonyms: list[str] = Field(default_factory=list, max_length=24)
    semantic_queries: list[str] = Field(min_length=1, max_length=3)
    document_id: int | None = Field(default=None, gt=0)

    @field_validator("lexical_terms", "synonyms", mode="before")
    @classmethod
    def validate_raw_term_count(cls, values: Any) -> Any:
        if not isinstance(values, (list, tuple)):
            return values
        if len(values) > 24:
            raise ValueError("term lists may contain at most 24 raw entries")
        return values

    @field_validator("lexical_terms", "synonyms")
    @classmethod
    def deduplicate_terms(cls, values: list[str]) -> list[str]:
        return _unique_casefold(values)

    @field_validator("semantic_queries", mode="before")
    @classmethod
    def validate_raw_semantic_query_count(cls, values: Any) -> Any:
        if not isinstance(values, (list, tuple)):
            return values
        if len(values) > 3:
            raise ValueError("semantic_queries may contain at most 3 raw entries")
        return values

    @field_validator("semantic_queries")
    @classmethod
    def deduplicate_semantic_queries(cls, values: list[str]) -> list[str]:
        unique = _unique_casefold(values)
        if not unique:
            raise ValueError("semantic_queries must contain at least one non-empty query")
        return unique


class RetrievalCandidate(BaseModel):
    chunk_id: int
    document_id: int
    title: str
    page_number: int
    chunk_index: int
    content: str
    bm25_rank: int | None = None
    vector_rank: int | None = None
    fusion_rank: int | None = None
    fusion_score: float = 0.0
    rerank_score: float | None = None
    matched_queries: list[str] = Field(default_factory=list)

    @property
    def evidence_id(self) -> str:
        return f"chunk:{self.chunk_id}"


class EvidenceSearchBatch(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    chunk_ids: list[int] = Field(default_factory=list, max_length=20)


class ResearchTaskError(ValueError):
    """Stable taskboard validation error returned to the Agent loop."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


class ResearchTaskState(BaseModel):
    task_id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    question: str = Field(min_length=1, max_length=1000)
    queries_attempted: list[str] = Field(default_factory=list, max_length=5)
    evidence_chunk_ids: list[int] = Field(default_factory=list, max_length=40)

    @field_validator("task_id", "question", mode="before")
    @classmethod
    def strip_task_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class EvidenceLedger(BaseModel):
    """The bounded evidence set an agent is allowed to cite.

    Observations are deliberately separate from evidence: document metadata and
    planning hints can be recorded without becoming citations in the final answer.
    """

    protocol_version: Literal[2] = 2
    local: dict[int, RetrievalCandidate] = Field(default_factory=dict)
    web: dict[str, WebSource] = Field(default_factory=dict)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    local_searches: int = Field(default=0, ge=0)
    search_batches: list[EvidenceSearchBatch] = Field(default_factory=list, max_length=5)
    tasks: dict[str, ResearchTaskState] = Field(default_factory=dict, max_length=8)

    def add_local(self, candidate: RetrievalCandidate) -> None:
        self.local.setdefault(candidate.chunk_id, candidate)

    def add_web(self, source: WebSource) -> None:
        self.web.setdefault(source.entry_url, source)

    def record_search(self, query: str, chunk_ids: list[int]) -> None:
        if len(self.search_batches) >= 5:
            return
        normalized_query = " ".join(str(query).split())[:1000]
        if not normalized_query:
            return
        bounded: list[int] = []
        for value in chunk_ids:
            if type(value) is not int or value <= 0 or value in bounded:
                continue
            bounded.append(value)
            if len(bounded) == 20:
                break
        if bounded:
            self.search_batches.append(
                EvidenceSearchBatch(query=normalized_query, chunk_ids=bounded)
            )

    def begin_task_search(self, task_id: str, question: str, query: str) -> None:
        candidate = ResearchTaskState(task_id=task_id, question=question)
        task = self.tasks.get(candidate.task_id)
        if task is None:
            if len(self.tasks) >= 8:
                raise ResearchTaskError("task_limit_exceeded")
            task = candidate
            self.tasks[task.task_id] = task

        normalized_query = " ".join(str(query).split())
        if not normalized_query or len(normalized_query) > 1000:
            raise ResearchTaskError("invalid_task_query")
        if any(
            attempted.casefold() == normalized_query.casefold()
            for attempted in task.queries_attempted
        ):
            raise ResearchTaskError("duplicate_task_query")
        if len(task.queries_attempted) >= 5:
            raise ResearchTaskError("task_query_limit_exceeded")
        task.queries_attempted.append(normalized_query)

    def bind_task_evidence(self, task_id: str, chunk_ids: list[int]) -> None:
        task = self.tasks.get(task_id)
        if task is None:
            raise ResearchTaskError("unknown_task_id")
        for chunk_id in chunk_ids:
            if type(chunk_id) is not int or chunk_id <= 0:
                continue
            if chunk_id in task.evidence_chunk_ids:
                continue
            if len(task.evidence_chunk_ids) == 40:
                break
            task.evidence_chunk_ids.append(chunk_id)

    def task_ids_for_chunk(self, chunk_id: int) -> list[str]:
        return [
            task_id
            for task_id, task in self.tasks.items()
            if chunk_id in task.evidence_chunk_ids
        ]


class RetrievalDiagnostics(BaseModel):
    degraded_channels: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    channel_status: dict[str, dict[str, Any]] = Field(default_factory=dict)
    channel_candidates: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    fusion_candidates: list[dict[str, Any]] = Field(default_factory=list)
    rerank_candidates: list[dict[str, Any]] = Field(default_factory=list)


class ResearchResult(BaseModel):
    answer: str
    citations: list[Any] = Field(default_factory=list)
    web_sources: list[Any] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    memory_hits: list[Any] = Field(default_factory=list)
    trace_events: list[dict[str, Any]] = Field(default_factory=list)
