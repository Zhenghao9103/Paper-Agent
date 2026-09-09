import pytest
from backend.app.schemas.chat import Citation
from backend.app.schemas.retrieval import (
    EvidenceLedger,
    QueryPlan,
    ResearchResult,
    ResearchTaskError,
    ResearchTaskState,
    RetrievalCandidate,
    RetrievalDiagnostics,
)
from pydantic import ValidationError


def test_query_plan_deduplicates_lexical_terms_and_preserves_document_id() -> None:
    plan = QueryPlan(
        intent="simple_rag",
        confidence=0.91,
        standalone_query="SpecNet 如何构图？",
        lexical_terms=["SpecNet", "specnet"],
        synonyms=["谱网络", "spectral network"],
        semantic_queries=["SpecNet 如何构图？", "SpecNet graph construction"],
        document_id=7,
    )

    assert plan.lexical_terms == ["SpecNet"]
    assert plan.document_id == 7


@pytest.mark.parametrize("intent", ["direct", "agentic_rag"])
def test_query_plan_accepts_other_valid_intents(intent: str) -> None:
    plan = QueryPlan(
        intent=intent,
        confidence=0.5,
        standalone_query="query",
        semantic_queries=["query"],
    )

    assert plan.intent == intent


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_query_plan_accepts_confidence_boundaries(confidence: float) -> None:
    plan = QueryPlan(
        intent="simple_rag",
        confidence=confidence,
        standalone_query="query",
        semantic_queries=["query"],
    )

    assert plan.confidence == confidence


@pytest.mark.parametrize("standalone_query", ["q", "q" * 1000])
def test_query_plan_accepts_standalone_query_length_boundaries(
    standalone_query: str,
) -> None:
    plan = QueryPlan(
        intent="simple_rag",
        confidence=0.5,
        standalone_query=standalone_query,
        semantic_queries=["query"],
    )

    assert plan.standalone_query == standalone_query


def test_query_plan_accepts_collection_size_boundaries() -> None:
    lexical_terms = [f"lexical {index}" for index in range(24)]
    synonyms = [f"synonym {index}" for index in range(24)]
    semantic_queries = [f"query {index}" for index in range(3)]

    plan = QueryPlan(
        intent="simple_rag",
        confidence=0.5,
        standalone_query="query",
        lexical_terms=lexical_terms,
        synonyms=synonyms,
        semantic_queries=semantic_queries,
    )

    assert plan.lexical_terms == lexical_terms
    assert plan.synonyms == synonyms
    assert plan.semantic_queries == semantic_queries


def test_query_plan_rejects_more_than_three_semantic_queries() -> None:
    with pytest.raises(ValidationError):
        QueryPlan(
            intent="simple_rag",
            confidence=0.91,
            standalone_query="SpecNet 如何构图？",
            semantic_queries=["query 1", "query 2", "query 3", "query 4"],
        )


def test_query_plan_rejects_duplicate_heavy_raw_semantic_query_overflow() -> None:
    with pytest.raises(ValidationError):
        QueryPlan(
            intent="simple_rag",
            confidence=0.91,
            standalone_query="SpecNet 如何构图？",
            semantic_queries=["q", " Q ", "q", "Q"],
        )


def test_query_plan_rejects_semantic_queries_that_normalize_to_empty() -> None:
    with pytest.raises(ValidationError):
        QueryPlan(
            intent="simple_rag",
            confidence=0.91,
            standalone_query="SpecNet 如何构图？",
            semantic_queries=["   "],
        )


@pytest.mark.parametrize("field_name", ["lexical_terms", "synonyms", "semantic_queries"])
@pytest.mark.parametrize("malformed_value", [1, None, {}])
def test_query_plan_reports_malformed_list_elements_as_validation_errors(
    field_name: str,
    malformed_value: object,
) -> None:
    values = {"semantic_queries": ["query"]}
    values[field_name] = [malformed_value]

    with pytest.raises(ValidationError):
        QueryPlan(
            intent="simple_rag",
            confidence=0.5,
            standalone_query="query",
            **values,
        )


@pytest.mark.parametrize("intent", ["unknown", "rag"])
def test_query_plan_rejects_invalid_intent(intent: str) -> None:
    with pytest.raises(ValidationError):
        QueryPlan(
            intent=intent,
            confidence=0.5,
            standalone_query="query",
            semantic_queries=["query"],
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_query_plan_rejects_confidence_outside_unit_interval(
    confidence: float,
) -> None:
    with pytest.raises(ValidationError):
        QueryPlan(
            intent="simple_rag",
            confidence=confidence,
            standalone_query="query",
            semantic_queries=["query"],
        )


@pytest.mark.parametrize("standalone_query", ["", "q" * 1001])
def test_query_plan_rejects_invalid_standalone_query_length(
    standalone_query: str,
) -> None:
    with pytest.raises(ValidationError):
        QueryPlan(
            intent="simple_rag",
            confidence=0.5,
            standalone_query=standalone_query,
            semantic_queries=["query"],
        )


@pytest.mark.parametrize("field_name", ["lexical_terms", "synonyms"])
def test_query_plan_rejects_duplicate_heavy_raw_term_overflow(
    field_name: str,
) -> None:
    with pytest.raises(ValidationError):
        QueryPlan(
            intent="simple_rag",
            confidence=0.5,
            standalone_query="query",
            semantic_queries=["query"],
            **{field_name: ["term", " TERM "] * 13},
        )


def test_query_plan_deduplicates_synonyms_and_semantic_queries() -> None:
    plan = QueryPlan(
        intent="simple_rag",
        confidence=0.5,
        standalone_query="query",
        synonyms=["Spectral Network", " spectral network ", "谱网络"],
        semantic_queries=["Graph Query", " graph query ", "Other"],
    )

    assert plan.synonyms == ["Spectral Network", "谱网络"]
    assert plan.semantic_queries == ["Graph Query", "Other"]


@pytest.mark.parametrize("document_id", [0, -1])
def test_query_plan_rejects_nonpositive_document_id(document_id: int) -> None:
    with pytest.raises(ValidationError):
        QueryPlan(
            intent="simple_rag",
            confidence=0.5,
            standalone_query="query",
            semantic_queries=["query"],
            document_id=document_id,
        )


def test_retrieval_candidate_builds_evidence_id_from_chunk_id() -> None:
    candidate = RetrievalCandidate(
        chunk_id=11,
        document_id=7,
        title="SpecNet",
        page_number=1,
        chunk_index=2,
        content="Graph construction details.",
    )

    assert candidate.evidence_id == "chunk:11"
    assert candidate.bm25_rank is None
    assert candidate.vector_rank is None
    assert candidate.fusion_rank is None
    assert candidate.fusion_score == 0.0
    assert candidate.rerank_score is None
    assert candidate.matched_queries == []


def test_retrieval_candidate_preserves_optional_ranking_values() -> None:
    candidate = RetrievalCandidate(
        chunk_id=11,
        document_id=7,
        title="SpecNet",
        page_number=1,
        chunk_index=2,
        content="Graph construction details.",
        bm25_rank=1,
        vector_rank=2,
        fusion_rank=3,
        fusion_score=0.75,
        rerank_score=0.88,
        matched_queries=["SpecNet", "graph construction"],
    )

    assert (
        candidate.bm25_rank,
        candidate.vector_rank,
        candidate.fusion_rank,
        candidate.fusion_score,
        candidate.rerank_score,
    ) == (1, 2, 3, 0.75, 0.88)
    assert candidate.matched_queries == ["SpecNet", "graph construction"]


def test_retrieval_diagnostics_defaults_are_empty() -> None:
    diagnostics = RetrievalDiagnostics()

    assert diagnostics.degraded_channels == []
    assert diagnostics.timings_ms == {}
    assert diagnostics.channel_status == {}
    assert diagnostics.channel_candidates == {}
    assert diagnostics.fusion_candidates == []
    assert diagnostics.rerank_candidates == []


def test_evidence_ledger_marks_agentic_protocol_v2() -> None:
    ledger = EvidenceLedger()

    assert ledger.protocol_version == 2


def test_research_result_collection_defaults_are_empty() -> None:
    result = ResearchResult(answer="answer")

    assert result.citations == []
    assert result.web_sources == []
    assert result.trace == []
    assert result.memory_hits == []
    assert result.trace_events == []


def test_citation_accepts_optional_chunk_id_without_breaking_legacy_payloads() -> None:
    payload = {
        "document_id": 7,
        "title": "SpecNet",
        "page_number": 1,
        "chunk_index": 2,
        "score": 0.8,
        "content": "Graph construction details.",
    }

    assert Citation(**payload).chunk_id is None
    assert Citation(**payload, chunk_id=11).chunk_id == 11


def test_evidence_ledger_registers_task_queries_and_chunks() -> None:
    ledger = EvidenceLedger()

    ledger.begin_task_search("sq1", "BootSC uses OT for what?", "BootSC OT target")
    ledger.bind_task_evidence("sq1", [7, 8, 7])

    task = ledger.tasks["sq1"]
    assert task.question == "BootSC uses OT for what?"
    assert task.queries_attempted == ["BootSC OT target"]
    assert task.evidence_chunk_ids == [7, 8]
    assert ledger.task_ids_for_chunk(7) == ["sq1"]


def test_evidence_ledger_keeps_canonical_question_and_rejects_duplicate_query() -> None:
    ledger = EvidenceLedger()
    ledger.begin_task_search("sq1", "BootSC target", "query one")

    ledger.begin_task_search("sq1", "BootSC target phrased differently", "query two")

    assert ledger.tasks["sq1"].question == "BootSC target"
    assert ledger.tasks["sq1"].queries_attempted == ["query one", "query two"]
    with pytest.raises(ResearchTaskError, match="duplicate_task_query"):
        ledger.begin_task_search("sq1", "BootSC target", " QUERY   ONE ")


@pytest.mark.parametrize("task_id", ["", "space id", "x" * 33, "../sq1"])
def test_research_task_rejects_invalid_ids(task_id: str) -> None:
    with pytest.raises(ValidationError):
        ResearchTaskState(task_id=task_id, question="question")


def test_evidence_ledger_rejects_ninth_task() -> None:
    ledger = EvidenceLedger()
    for index in range(8):
        ledger.begin_task_search(f"sq{index}", f"question {index}", f"query {index}")

    with pytest.raises(ResearchTaskError, match="task_limit_exceeded"):
        ledger.begin_task_search("sq8", "question 8", "query 8")
