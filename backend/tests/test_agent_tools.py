from types import SimpleNamespace

import pytest
from backend.app.agents.tools import (
    TOOL_SPECS,
    ArxivSearchArgs,
    ChunkNeighborsArgs,
    EvidenceLedger,
    HybridSearchArgs,
    InspectDocumentArgs,
    ToolPreconditionError,
    UnknownToolError,
    execute_tool,
    execute_tool_result,
)
from backend.app.models.chunk import DocumentChunk
from backend.app.models.document import Document
from backend.app.schemas.chat import WebSource
from backend.app.schemas.retrieval import RetrievalCandidate
from pydantic import ValidationError


def _candidate(chunk_id: int, *, document_id: int = 1) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=document_id,
        title="Paper",
        page_number=1,
        chunk_index=chunk_id - 1,
        content=f"chunk {chunk_id}",
    )


def _seed_three_adjacent_chunks(db_session) -> int:
    document = Document(
        title="Paper",
        file_type="pdf",
        file_path="paper.pdf",
        status="indexed",
    )
    db_session.add(document)
    db_session.flush()
    db_session.add_all(
        [
            DocumentChunk(
                id=1,
                document_id=document.id,
                page_id=None,
                page_number=1,
                chunk_index=0,
                content="one",
            ),
            DocumentChunk(
                id=2,
                document_id=document.id,
                page_id=None,
                page_number=1,
                chunk_index=1,
                content="two",
            ),
            DocumentChunk(
                id=3,
                document_id=document.id,
                page_id=None,
                page_number=1,
                chunk_index=2,
                content="three",
            ),
        ]
    )
    db_session.commit()
    return document.id


def test_registry_rejects_unknown_tool(db_session) -> None:
    with pytest.raises(UnknownToolError):
        execute_tool(db_session, "run_shell", {}, ledger=EvidenceLedger())


def test_registry_exposes_exactly_four_safe_tools() -> None:
    assert {spec["function"]["name"] for spec in TOOL_SPECS} == {
        "hybrid_search",
        "get_chunk_neighbors",
        "inspect_document",
        "search_arxiv",
    }
    assert all("shell" not in name and "http" not in name for name in TOOL_SPECS)


def test_hybrid_search_schema_requires_agent_task_metadata() -> None:
    schema = next(
        spec["function"]["parameters"]
        for spec in TOOL_SPECS
        if spec["function"]["name"] == "hybrid_search"
    )

    assert set(schema["properties"]) == {
        "task_id",
        "subquestion",
        "query",
        "document_id",
    }
    assert set(schema["required"]) == {"task_id", "subquestion", "query"}
    assert HybridSearchArgs(
        task_id="sq1",
        subquestion="How does the entity scale?",
        query="entity scalability",
    ).model_dump() == {
        "task_id": "sq1",
        "subquestion": "How does the entity scale?",
        "query": "entity scalability",
        "document_id": None,
    }
    with pytest.raises(ValidationError):
        HybridSearchArgs(
            task_id="sq1",
            subquestion="How does the entity scale?",
            query="entity scalability",
            lexical_terms=["entity"],
            semantic_queries=["entity scalability"],
        )


def test_hybrid_search_binds_returned_chunks_to_agent_task(
    db_session, monkeypatch
) -> None:
    candidate = _candidate(11, document_id=7)
    monkeypatch.setattr(
        "backend.app.agents.tools.hybrid_search",
        lambda *args, **kwargs: SimpleNamespace(
            candidates=[candidate],
            diagnostics=SimpleNamespace(degraded_channels=[], timings_ms={}),
        ),
    )
    ledger = EvidenceLedger()

    execute_tool(
        db_session,
        "hybrid_search",
        {
            "task_id": "sq1",
            "subquestion": "What does BootSC supervise?",
            "query": "BootSC optimal transport target",
        },
        ledger=ledger,
    )

    assert ledger.tasks["sq1"].queries_attempted == ["BootSC optimal transport target"]
    assert ledger.tasks["sq1"].evidence_chunk_ids == [11]


def test_hybrid_search_derives_internal_query_plan_from_query(
    db_session, monkeypatch
) -> None:
    captured = []

    def fake_hybrid(db, plan, **kwargs):
        captured.append(plan)
        return SimpleNamespace(
            candidates=[],
            diagnostics=SimpleNamespace(degraded_channels=[], timings_ms={}),
        )

    monkeypatch.setattr("backend.app.agents.tools.hybrid_search", fake_hybrid)
    execute_tool(
        db_session,
        "hybrid_search",
        {
            "task_id": "sq1",
            "subquestion": "How does the entity scale?",
            "query": "Entity scalability with partial samples",
            "document_id": 999,
        },
        ledger=EvidenceLedger(),
        forced_document_id=7,
    )

    plan = captured[0]
    assert plan.standalone_query == "Entity scalability with partial samples"
    assert 1 <= len(plan.lexical_terms) <= 24
    assert plan.semantic_queries == ["Entity scalability with partial samples"]
    assert plan.synonyms == []
    assert plan.document_id == 7


def test_agent_hybrid_search_uses_explicit_candidate_window_and_audits_once(
    db_session, monkeypatch
) -> None:
    captured_kwargs = []

    def fake_hybrid(db, plan, **kwargs):
        captured_kwargs.append(kwargs)
        return SimpleNamespace(
            candidates=[_candidate(chunk_id) for chunk_id in range(1, 26)],
            diagnostics=SimpleNamespace(degraded_channels=[], timings_ms={}),
        )

    monkeypatch.setattr("backend.app.agents.tools.hybrid_search", fake_hybrid)
    monkeypatch.setattr(
        "backend.app.agents.tools.get_settings",
        lambda: SimpleNamespace(agent_retrieval_candidate_k=20),
    )
    ledger = EvidenceLedger()

    payload = execute_tool(
        db_session,
        "hybrid_search",
        {
            "task_id": "sq1",
            "subquestion": "What evidence supports OT?",
            "query": "optimal transport evidence",
        },
        ledger=ledger,
    )

    assert captured_kwargs == [{"evidence_limit": 20}]
    assert len(payload["candidates"]) == 20
    assert list(ledger.local) == list(range(1, 21))
    assert ledger.tasks["sq1"].evidence_chunk_ids == list(range(1, 21))


def test_chunk_neighbors_are_bounded_and_added_to_ledger(db_session) -> None:
    document_id = _seed_three_adjacent_chunks(db_session)
    ledger = EvidenceLedger()
    ledger.begin_task_search("sq1", "Read local context", "seed query")
    ledger.add_local(_candidate(2, document_id=document_id))
    ledger.bind_task_evidence("sq1", [2])
    payload = execute_tool(
        db_session,
        "get_chunk_neighbors",
        {"chunk_id": 2, "before": 2, "after": 2},
        ledger=ledger,
    )
    assert [item["chunk_id"] for item in payload["chunks"]] == [1, 2, 3]
    assert len(ledger.local) == 3
    assert all(item["document_id"] == document_id for item in payload["chunks"])
    assert ledger.tasks["sq1"].evidence_chunk_ids == [2, 1, 3]


def test_chunk_neighbors_order_pages_before_page_local_chunk_indices(db_session) -> None:
    document = Document(
        title="Paged Paper",
        file_type="pdf",
        file_path="paged.pdf",
        status="indexed",
    )
    db_session.add(document)
    db_session.flush()
    db_session.add_all(
        [
            DocumentChunk(
                id=11,
                document_id=document.id,
                page_id=None,
                page_number=1,
                chunk_index=0,
                content="page one, first",
            ),
            DocumentChunk(
                id=12,
                document_id=document.id,
                page_id=None,
                page_number=1,
                chunk_index=1,
                content="page one, second",
            ),
            DocumentChunk(
                id=13,
                document_id=document.id,
                page_id=None,
                page_number=2,
                chunk_index=0,
                content="page two, first",
            ),
            DocumentChunk(
                id=14,
                document_id=document.id,
                page_id=None,
                page_number=2,
                chunk_index=1,
                content="page two, second",
            ),
        ]
    )
    db_session.commit()

    ledger = EvidenceLedger()
    ledger.begin_task_search("sq1", "Read page context", "seed query")
    ledger.add_local(_candidate(13, document_id=document.id))
    ledger.bind_task_evidence("sq1", [13])
    payload = execute_tool(
        db_session,
        "get_chunk_neighbors",
        {"chunk_id": 13, "before": 2, "after": 1},
        ledger=ledger,
    )

    assert [item["chunk_id"] for item in payload["chunks"]] == [11, 12, 13, 14]


def test_chunk_neighbors_fetches_bounded_sql_windows(db_session, monkeypatch) -> None:
    document = Document(
        title="Large Paper",
        file_type="pdf",
        file_path="large.pdf",
        status="indexed",
    )
    db_session.add(document)
    db_session.flush()
    db_session.add_all(
        [
            DocumentChunk(
                id=100 + index,
                document_id=document.id,
                page_id=None,
                page_number=(index // 2) + 1,
                chunk_index=index % 2,
                content=f"chunk {index}",
            )
            for index in range(100)
        ]
    )
    db_session.commit()

    original_execute = db_session.execute
    captured_statements: list[str] = []

    def spy_execute(statement, *args, **kwargs):
        sql = str(statement).lower()
        if "document_chunks" in sql:
            captured_statements.append(sql)
        return original_execute(statement, *args, **kwargs)

    monkeypatch.setattr(db_session, "execute", spy_execute)
    ledger = EvidenceLedger()
    ledger.begin_task_search("sq1", "Read bounded context", "seed query")
    ledger.add_local(_candidate(150, document_id=document.id))
    ledger.bind_task_evidence("sq1", [150])
    payload = execute_tool(
        db_session,
        "get_chunk_neighbors",
        {"chunk_id": 150, "before": 2, "after": 2},
        ledger=ledger,
    )

    assert [item["chunk_id"] for item in payload["chunks"]] == [148, 149, 150, 151, 152]
    bounded = [sql for sql in captured_statements if "order by" in sql]
    assert len(bounded) == 2
    assert all("limit" in sql for sql in bounded)


def test_chunk_neighbor_args_reject_unsafe_window_and_ids() -> None:
    with pytest.raises(ValidationError):
        ChunkNeighborsArgs(chunk_id=0)
    with pytest.raises(ValidationError):
        ChunkNeighborsArgs(chunk_id=1, before=3)
    with pytest.raises(ValidationError):
        ChunkNeighborsArgs(chunk_id=1, after=-1)


def test_chunk_neighbors_require_task_bound_seed(db_session) -> None:
    _seed_three_adjacent_chunks(db_session)

    with pytest.raises(ToolPreconditionError, match="task_binding_missing"):
        execute_tool(
            db_session,
            "get_chunk_neighbors",
            {"chunk_id": 2},
            ledger=EvidenceLedger(),
        )


def test_ledger_deduplicates_chunks_and_arxiv_urls() -> None:
    ledger = EvidenceLedger()
    ledger.add_local(_candidate(7))
    ledger.add_local(_candidate(7, document_id=9))
    source = WebSource(
        title="A",
        authors=[],
        summary="S",
        published="2026",
        entry_url="https://arxiv.org/abs/1",
        pdf_url="https://arxiv.org/pdf/1",
    )
    ledger.add_web(source)
    ledger.add_web(source.model_copy(update={"title": "duplicate"}))
    ledger.observations.append({"document_id": 1, "summary": "metadata"})
    assert len(ledger.local) == 1
    assert len(ledger.web) == 1
    assert len(ledger.observations) == 1


def test_ledger_records_bounded_deduplicated_search_batches() -> None:
    ledger = EvidenceLedger()

    ledger.record_search("first entity", [1, 1, *range(2, 30)])
    for batch_id in range(2, 7):
        ledger.record_search(f"entity {batch_id}", [batch_id * 100])
    ledger.record_search("empty", [])

    assert ledger.search_batches[0].query == "first entity"
    assert ledger.search_batches[0].chunk_ids == list(range(1, 21))
    assert [batch.chunk_ids for batch in ledger.search_batches[1:]] == [
        [200],
        [300],
        [400],
        [500],
    ]
    assert len(ledger.search_batches) == 5


def test_arxiv_requires_a_local_search_attempt_first(db_session, monkeypatch) -> None:
    ledger = EvidenceLedger()
    monkeypatch.setattr(
        "backend.app.agents.tools.search_arxiv",
        lambda *args, **kwargs: [],
    )
    with pytest.raises(ToolPreconditionError, match="local search required"):
        execute_tool(
            db_session,
            "search_arxiv",
            {"query": "agentic RAG", "max_results": 3},
            ledger=ledger,
        )


def test_tool_argument_models_enforce_lengths_and_ranges() -> None:
    with pytest.raises(ValidationError):
        HybridSearchArgs(task_id="sq1", subquestion="question", query="")
    with pytest.raises(ValidationError):
        HybridSearchArgs(task_id="sq1", subquestion="question", query="q" * 1001)
    with pytest.raises(ValidationError):
        HybridSearchArgs(
            task_id="sq1", subquestion="question", query="q", document_id=0
        )
    with pytest.raises(ValidationError):
        InspectDocumentArgs(document_id=0)
    with pytest.raises(ValidationError):
        ArxivSearchArgs(query="q", max_results=4)


def test_hybrid_search_forces_request_document_scope_and_counts_only_hybrid(
    db_session, monkeypatch
) -> None:
    calls: list[tuple[object, object]] = []
    candidate = _candidate(11, document_id=7)

    def fake_hybrid(db, plan, **kwargs):
        calls.append((plan, kwargs))
        return SimpleNamespace(
            candidates=[candidate],
            diagnostics=SimpleNamespace(
                degraded_channels=["vector"],
                timings_ms={"bm25": 1.0},
            ),
        )

    monkeypatch.setattr("backend.app.agents.tools.hybrid_search", fake_hybrid)
    ledger = EvidenceLedger()
    payload = execute_tool(
        db_session,
        "hybrid_search",
        {
            "task_id": "sq1",
            "subquestion": "Scoped search",
            "query": "q",
            "document_id": 999,
        },
        ledger=ledger,
        forced_document_id=7,
    )
    assert calls[0][0].document_id == 7
    assert ledger.local_searches == 1
    assert list(ledger.local) == [11]
    assert [batch.chunk_ids for batch in ledger.search_batches] == [[11]]
    assert payload["candidates"][0]["chunk_id"] == 11
    assert payload["diagnostics"]["degraded_channels"] == ["vector"]
    assert payload["coverage"] == {"document_ids": [7], "chunk_ids": [11]}
    assert payload["task"]["task_id"] == "sq1"


def test_inspect_document_is_observation_only(db_session) -> None:
    document_id = _seed_three_adjacent_chunks(db_session)
    ledger = EvidenceLedger()
    payload = execute_tool(
        db_session,
        "inspect_document",
        {"document_id": document_id},
        ledger=ledger,
    )
    assert payload["document"]["id"] == document_id
    assert len(ledger.local) == 0
    assert ledger.observations[-1]["document_id"] == document_id


def test_search_arxiv_is_bounded_safe_and_deduplicated(db_session, monkeypatch) -> None:
    _seed_three_adjacent_chunks(db_session)
    papers = [
        SimpleNamespace(
            title="Paper",
            authors=["A"],
            summary="Summary",
            published="2026-01-01",
            entry_url="https://arxiv.org/abs/1",
            pdf_url="https://arxiv.org/pdf/1",
        ),
        SimpleNamespace(
            title="Duplicate",
            authors=[],
            summary="S",
            published="2026-01-02",
            entry_url="https://arxiv.org/abs/1",
            pdf_url="https://arxiv.org/pdf/1",
        ),
    ]
    monkeypatch.setattr("backend.app.agents.tools.search_arxiv", lambda *args, **kwargs: papers)
    ledger = EvidenceLedger(local_searches=1)
    payload = execute_tool(
        db_session,
        "search_arxiv",
        {"query": "agentic", "max_results": 3},
        ledger=ledger,
    )
    assert len(payload["web_sources"]) == 1
    assert len(ledger.web) == 1
    assert payload["web_sources"][0]["entry_url"] == "https://arxiv.org/abs/1"


def test_search_arxiv_failures_return_safe_empty_payload(db_session, monkeypatch) -> None:
    _seed_three_adjacent_chunks(db_session)

    def broken(*args, **kwargs):
        raise RuntimeError("secret api key should not leak")

    monkeypatch.setattr("backend.app.agents.tools.search_arxiv", broken)
    ledger = EvidenceLedger(local_searches=1)
    payload = execute_tool(
        db_session,
        "search_arxiv",
        {"query": "agentic", "max_results": 3},
        ledger=ledger,
    )
    assert payload["web_sources"] == []
    assert "secret api key" not in str(payload)
    assert ledger.observations[-1]["status"] == "unavailable"


def test_hybrid_args_scope_cannot_be_overridden_by_tool_document_id(
    db_session, monkeypatch
) -> None:
    plans = []
    monkeypatch.setattr(
        "backend.app.agents.tools.hybrid_search",
        lambda db, plan, **kwargs: plans.append(plan)
        or SimpleNamespace(
            candidates=[],
            diagnostics=SimpleNamespace(degraded_channels=[], timings_ms={}),
        ),
    )
    execute_tool(
        db_session,
        "hybrid_search",
        {
            "task_id": "sq1",
            "subquestion": "Scoped search",
            "query": "q",
            "document_id": 2,
        },
        ledger=EvidenceLedger(),
        forced_document_id=1,
    )
    assert plans[0].document_id == 1


def test_tool_result_marks_error_payload_as_failed(db_session, monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.app.agents.tools.hybrid_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    result = execute_tool_result(
        db_session,
        "hybrid_search",
        {
            "task_id": "A1",
            "subquestion": "Find evidence",
            "query": "evidence query",
        },
        ledger=EvidenceLedger(),
    )
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "local_retrieval_failed"
    assert result.value is None


def test_tool_result_preserves_degraded_candidates(db_session, monkeypatch) -> None:
    candidate = _candidate(11)
    monkeypatch.setattr(
        "backend.app.agents.tools.hybrid_search",
        lambda *args, **kwargs: SimpleNamespace(
            candidates=[candidate],
            diagnostics=SimpleNamespace(
                degraded_channels=["vector"], timings_ms={}
            ),
        ),
    )
    result = execute_tool_result(
        db_session,
        "hybrid_search",
        {
            "task_id": "A1",
            "subquestion": "Find evidence",
            "query": "evidence query",
        },
        ledger=EvidenceLedger(),
    )
    assert result.status == "degraded"
    assert result.value["candidates"][0]["chunk_id"] == 11
    assert result.failure.code == "retrieval_channel_degraded"


def test_tool_result_converts_validation_exception(db_session) -> None:
    result = execute_tool_result(
        db_session,
        "get_chunk_neighbors",
        {"chunk_id": 0},
        ledger=EvidenceLedger(),
    )
    assert result.status == "failed"
    assert result.failure.category == "validation"
    assert result.failure.code == "schema_invalid"
