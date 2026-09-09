"""Constrained, deterministic tools exposed to the research agent.

The module intentionally keeps a small explicit allow-list.  Tool arguments are
validated locally before any database or network operation is attempted, and all
evidence enters the caller-owned :class:`EvidenceLedger` through this boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..models.chunk import DocumentChunk
from ..models.document import Document
from ..rag.hybrid import RetrievalUnavailable, hybrid_search
from ..rag.tokenization import tokenize_mixed
from ..resilience import ExecutionResult, FailureInfo, classify_exception
from ..schemas.arxiv import ArxivPaper
from ..schemas.chat import WebSource
from ..schemas.retrieval import EvidenceLedger, QueryPlan, RetrievalCandidate
from ..services.arxiv_search import search_arxiv


class UnknownToolError(ValueError):
    """Raised when a model attempts to call a tool outside the allow-list."""


class ToolPreconditionError(ValueError):
    """Raised when a constrained tool is called before its required prerequisite."""


class _ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HybridSearchArgs(_ToolArgs):
    task_id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_-]+$")
    subquestion: str = Field(min_length=1, max_length=1000)
    query: str = Field(min_length=1, max_length=1000)
    document_id: int | None = Field(default=None, gt=0)


class ChunkNeighborsArgs(_ToolArgs):
    chunk_id: int = Field(gt=0)
    before: int = Field(default=1, ge=0, le=2)
    after: int = Field(default=1, ge=0, le=2)


class InspectDocumentArgs(_ToolArgs):
    document_id: int = Field(gt=0)


class ArxivSearchArgs(_ToolArgs):
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=3, ge=1, le=3)


_ArgModel: TypeAlias = type[BaseModel]
_ARG_MODELS: dict[str, _ArgModel] = {
    "hybrid_search": HybridSearchArgs,
    "get_chunk_neighbors": ChunkNeighborsArgs,
    "inspect_document": InspectDocumentArgs,
    "search_arxiv": ArxivSearchArgs,
}


def _spec(name: str, description: str, model: _ArgModel) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": model.model_json_schema(),
        },
    }


# OpenAI-compatible tool definitions.  Keep this list as the sole public model
# surface; arbitrary callables, shell commands, and unrestricted HTTP are absent.
TOOL_SPECS: list[dict[str, Any]] = [
    _spec(
        "hybrid_search",
        "Search local SQLite BM25 and vector indexes with deterministic fusion.",
        HybridSearchArgs,
    ),
    _spec(
        "get_chunk_neighbors",
        "Read bounded adjacent chunks from the same document.",
        ChunkNeighborsArgs,
    ),
    _spec(
        "inspect_document",
        "Inspect document metadata and indexed coverage without adding evidence.",
        InspectDocumentArgs,
    ),
    _spec(
        "search_arxiv",
        "Search arXiv after local retrieval found or attempted local evidence.",
        ArxivSearchArgs,
    ),
]


def execute_tool(
    db: Session,
    tool_name: str,
    raw_args: Mapping[str, Any] | None,
    *,
    ledger: EvidenceLedger,
    forced_document_id: int | None = None,
    request_document_id: int | None = None,
) -> dict[str, Any]:
    """Validate and execute one allow-listed tool using a caller-owned session."""

    if forced_document_id is not None and request_document_id is not None:
        raise ValueError("provide only one forced document scope")
    scope_document_id = (
        forced_document_id if forced_document_id is not None else request_document_id
    )

    model = _ARG_MODELS.get(tool_name)
    if model is None:
        raise UnknownToolError(f"unknown tool: {tool_name}")
    try:
        args = model.model_validate(dict(raw_args or {}))
    except ValidationError:
        # Keep Pydantic's useful validation details for the orchestrator to relay,
        # while ensuring no operation ran before validation completed.
        raise

    if tool_name == "hybrid_search":
        return _execute_hybrid_search(
            db,
            args,
            ledger=ledger,
            forced_document_id=scope_document_id,
        )
    if tool_name == "get_chunk_neighbors":
        return _execute_neighbors(
            db,
            args,
            ledger=ledger,
            forced_document_id=scope_document_id,
        )
    if tool_name == "inspect_document":
        return _execute_inspect(
            db,
            args,
            ledger=ledger,
            forced_document_id=scope_document_id,
        )
    return _execute_arxiv(db, args, ledger=ledger)


def execute_tool_result(
    db: Session,
    tool_name: str,
    raw_args: Mapping[str, Any] | None,
    *,
    ledger: EvidenceLedger,
    forced_document_id: int | None = None,
    request_document_id: int | None = None,
) -> ExecutionResult:
    try:
        payload = execute_tool(
            db,
            tool_name,
            raw_args,
            ledger=ledger,
            forced_document_id=forced_document_id,
            request_document_id=request_document_id,
        )
    except Exception as exc:
        return ExecutionResult.failed(classify_exception(exc))

    error = payload.get("error")
    candidates = (
        payload.get("candidates")
        or payload.get("chunks")
        or payload.get("web_sources")
        or []
    )
    diagnostics = payload.get("diagnostics") or {}
    degraded_channels = diagnostics.get("degraded_channels") or []
    if error and not candidates:
        code = (
            "local_retrieval_unavailable"
            if "unavailable" in str(error).casefold()
            else "local_retrieval_failed"
        )
        return ExecutionResult.failed(
            FailureInfo(category="transient", code=code, retryable=True)
        )
    if error or degraded_channels:
        return ExecutionResult.degraded(
            payload,
            failure=FailureInfo(
                category="transient",
                code="retrieval_channel_degraded",
                retryable=True,
            ),
        )
    return ExecutionResult.success(payload)


def _execute_hybrid_search(
    db: Session,
    args: HybridSearchArgs,
    *,
    ledger: EvidenceLedger,
    forced_document_id: int | None,
) -> dict[str, Any]:
    ledger.begin_task_search(args.task_id, args.subquestion, args.query)
    ledger.local_searches += 1
    document_id = forced_document_id if forced_document_id is not None else args.document_id
    lexical_terms = tokenize_mixed(args.query)[:24] or [args.query]
    plan = QueryPlan(
        intent="agentic_rag",
        confidence=1.0,
        standalone_query=args.query,
        lexical_terms=lexical_terms,
        synonyms=[],
        semantic_queries=[args.query],
        document_id=document_id,
    )
    candidate_limit = get_settings().agent_retrieval_candidate_k
    try:
        result = hybrid_search(db, plan, evidence_limit=candidate_limit)
    except RetrievalUnavailable:
        observation = {"type": "hybrid_search", "status": "unavailable"}
        ledger.observations.append(observation)
        return {
            "candidates": [],
            "diagnostics": {"degraded_channels": ["bm25", "vector"], "timings_ms": {}},
            "error": "local retrieval unavailable",
        }
    except Exception:
        ledger.observations.append({"type": "hybrid_search", "status": "degraded"})
        return {
            "candidates": [],
            "diagnostics": {"degraded_channels": ["hybrid"], "timings_ms": {}},
            "error": "local retrieval failed",
        }

    candidates = [
        item if isinstance(item, RetrievalCandidate) else RetrievalCandidate.model_validate(item)
        for item in getattr(result, "candidates", [])
    ][:candidate_limit]
    for candidate in candidates:
        ledger.add_local(candidate)
    ledger.bind_task_evidence(
        args.task_id,
        [candidate.chunk_id for candidate in candidates],
    )
    ledger.record_search(args.query, [candidate.chunk_id for candidate in candidates])
    diagnostics = getattr(result, "diagnostics", None)
    if isinstance(diagnostics, BaseModel):
        diagnostics_payload = diagnostics.model_dump()
    elif isinstance(diagnostics, Mapping):
        diagnostics_payload = dict(diagnostics)
    elif diagnostics is None:
        diagnostics_payload = {}
    else:
        diagnostics_payload = {
            "degraded_channels": list(getattr(diagnostics, "degraded_channels", [])),
            "timings_ms": dict(getattr(diagnostics, "timings_ms", {})),
        }
    return {
        "candidates": [candidate.model_dump() for candidate in candidates],
        "diagnostics": diagnostics_payload,
        "coverage": {
            "document_ids": sorted({candidate.document_id for candidate in candidates}),
            "chunk_ids": [candidate.chunk_id for candidate in candidates[:20]],
        },
        "task": {
            "task_id": args.task_id,
            "subquestion": args.subquestion,
            "query": args.query,
            "evidence_chunk_ids": list(
                ledger.tasks[args.task_id].evidence_chunk_ids
            ),
        },
    }


def _execute_neighbors(
    db: Session,
    args: ChunkNeighborsArgs,
    *,
    ledger: EvidenceLedger,
    forced_document_id: int | None,
) -> dict[str, Any]:
    task_ids = ledger.task_ids_for_chunk(args.chunk_id)
    if not task_ids:
        raise ToolPreconditionError("task_binding_missing")
    chunk = db.get(DocumentChunk, args.chunk_id)
    if chunk is None or (
        forced_document_id is not None and chunk.document_id != forced_document_id
    ):
        return {"chunks": [], "error": "chunk not found"}

    document = db.get(Document, chunk.document_id)
    if document is None:
        return {"chunks": [], "error": "document not found"}
    before_rows: list[DocumentChunk] = []
    if args.before:
        before_rows = list(
            db.execute(
                select(DocumentChunk)
                .where(
                    DocumentChunk.document_id == chunk.document_id,
                    _chunk_order_before(chunk),
                )
                .order_by(
                    DocumentChunk.page_number.desc(),
                    DocumentChunk.chunk_index.desc(),
                    DocumentChunk.id.desc(),
                )
                .limit(args.before)
            ).scalars()
        )
        before_rows.reverse()

    after_rows: list[DocumentChunk] = []
    if args.after:
        after_rows = list(
            db.execute(
                select(DocumentChunk)
                .where(
                    DocumentChunk.document_id == chunk.document_id,
                    _chunk_order_after(chunk),
                )
                .order_by(
                    DocumentChunk.page_number.asc(),
                    DocumentChunk.chunk_index.asc(),
                    DocumentChunk.id.asc(),
                )
                .limit(args.after)
            ).scalars()
        )

    selected = [*before_rows, chunk, *after_rows]
    candidates = [_candidate_from_chunk(item, document) for item in selected]
    for candidate in candidates:
        ledger.add_local(candidate)
    chunk_ids = [candidate.chunk_id for candidate in candidates]
    for task_id in task_ids:
        ledger.bind_task_evidence(task_id, chunk_ids)
    return {"chunks": [candidate.model_dump() for candidate in candidates]}


def _execute_inspect(
    db: Session,
    args: InspectDocumentArgs,
    *,
    ledger: EvidenceLedger,
    forced_document_id: int | None,
) -> dict[str, Any]:
    document_id = forced_document_id if forced_document_id is not None else args.document_id
    document = db.get(Document, document_id)
    if document is None:
        payload = {"document": None, "error": "document not found"}
        ledger.observations.append(
            {"type": "document_inspection", "document_id": document_id, "status": "missing"}
        )
        return payload
    chunk_count = int(
        db.scalar(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == document_id)
        )
        or 0
    )
    summary = {
        "chunk_count": chunk_count,
        "abstract": document.abstract or "",
    }
    metadata = {
        "id": document.id,
        "title": document.title,
        "authors": document.authors or "",
        "year": document.year,
        "file_type": document.file_type,
        "status": document.status,
        "source_type": document.source_type,
        "summary": summary,
    }
    ledger.observations.append(
        {"type": "document_inspection", "document_id": document_id, "summary": summary}
    )
    return {"document": metadata}


def _execute_arxiv(
    db: Session,
    args: ArxivSearchArgs,
    *,
    ledger: EvidenceLedger,
) -> dict[str, Any]:
    del db  # The network helper is intentionally independent of the local session.
    if ledger.local_searches <= 0:
        raise ToolPreconditionError("local search required before arXiv")
    try:
        papers = search_arxiv(args.query, max_results=min(args.max_results, 3))
    except Exception:
        ledger.observations.append({"type": "arxiv_search", "status": "unavailable"})
        return {"web_sources": [], "error": "arXiv unavailable"}

    sources: list[WebSource] = []
    for paper in papers[:3]:
        source = _web_source_from_paper(paper)
        if source is None or _duplicate_web_source(source, ledger):
            continue
        ledger.add_web(source)
        sources.append(source)
    ledger.observations.append(
        {"type": "arxiv_search", "status": "ok", "count": len(sources)}
    )
    return {"web_sources": [source.model_dump() for source in sources]}


def _candidate_from_chunk(chunk: DocumentChunk, document: Document) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        title=document.title,
        page_number=chunk.page_number,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
    )


def _chunk_order_before(chunk: DocumentChunk):
    return or_(
        DocumentChunk.page_number < chunk.page_number,
        and_(
            DocumentChunk.page_number == chunk.page_number,
            DocumentChunk.chunk_index < chunk.chunk_index,
        ),
        and_(
            DocumentChunk.page_number == chunk.page_number,
            DocumentChunk.chunk_index == chunk.chunk_index,
            DocumentChunk.id < chunk.id,
        ),
    )


def _chunk_order_after(chunk: DocumentChunk):
    return or_(
        DocumentChunk.page_number > chunk.page_number,
        and_(
            DocumentChunk.page_number == chunk.page_number,
            DocumentChunk.chunk_index > chunk.chunk_index,
        ),
        and_(
            DocumentChunk.page_number == chunk.page_number,
            DocumentChunk.chunk_index == chunk.chunk_index,
            DocumentChunk.id > chunk.id,
        ),
    )


def _web_source_from_paper(paper: ArxivPaper | Mapping[str, Any] | Any) -> WebSource | None:
    if isinstance(paper, Mapping):
        payload = dict(paper)
    else:
        payload = {
            "title": getattr(paper, "title", ""),
            "authors": getattr(paper, "authors", []),
            "summary": getattr(paper, "summary", ""),
            "published": getattr(paper, "published", ""),
            "entry_url": getattr(paper, "entry_url", ""),
            "pdf_url": getattr(paper, "pdf_url", None),
        }
    try:
        source = WebSource.model_validate(payload)
    except ValidationError:
        return None
    return source if source.entry_url else None


def _duplicate_web_source(source: WebSource, ledger: EvidenceLedger) -> bool:
    return any(
        existing.entry_url == source.entry_url
        or (
            source.pdf_url is not None
            and existing.pdf_url is not None
            and existing.pdf_url == source.pdf_url
        )
        for existing in ledger.web.values()
    )
