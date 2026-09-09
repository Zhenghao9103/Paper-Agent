import json
from types import SimpleNamespace

import pytest
from backend.app.core.config import Settings
from backend.app.models.document import Document
from backend.app.schemas.chat import WebSource
from backend.app.schemas.evidence import (
    EvidencePack,
    EvidencePackItem,
    PackedLocalSource,
    ResearchClaim,
)
from backend.app.schemas.retrieval import RetrievalCandidate
from backend.app.services import answering, model_clients
from backend.app.services.llm import ContextWindowExceededError


def _candidate(chunk_id: int, *, score: float = 0.5) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        document_id=1,
        title="Paper",
        page_number=1,
        chunk_index=chunk_id,
        content=f"grounded evidence {chunk_id}",
        fusion_rank=1,
        fusion_score=score,
    )


def _web(url: str = "https://arxiv.org/abs/2401.00001") -> WebSource:
    return WebSource(
        title="External paper",
        authors=["A. Author"],
        summary="external evidence",
        published="2024-01-01",
        entry_url=url,
        pdf_url=url.replace("/abs/", "/pdf/") + ".pdf",
    )


def test_direct_handler_lists_database_documents_with_status(db_session) -> None:
    db_session.add_all(
        [
            Document(title="First", file_type="pdf", file_path="1.pdf", status="indexed"),
            Document(title="Second", file_type="pdf", file_path="2.pdf", status="failed"),
        ]
    )
    db_session.commit()

    answer = answering.answer_direct(
        db_session,
        question="当前知识库有哪些论文？",
        document_id=None,
    )

    assert "2 篇" in answer
    assert "First" in answer and "indexed" in answer
    assert "Second" in answer and "failed" in answer


def test_direct_handler_can_select_one_document(db_session) -> None:
    db_session.add_all(
        [
            Document(title="First", file_type="pdf", file_path="1.pdf", status="indexed"),
            Document(title="Second", file_type="pdf", file_path="2.pdf", status="failed"),
        ]
    )
    db_session.commit()

    answer = answering.answer_direct(db_session, question="status", document_id=2)

    assert "Second" in answer and "failed" in answer
    assert "First" not in answer


def test_direct_handler_handles_empty_database(db_session) -> None:
    answer = answering.answer_direct(db_session, question="list", document_id=None)
    assert "当前知识库中还没有论文" in answer


def test_direct_handler_lists_only_failed_indexing_documents(db_session) -> None:
    db_session.add_all(
        [
            Document(title="Indexed", file_type="pdf", file_path="ok.pdf", status="indexed"),
            Document(title="Failed", file_type="pdf", file_path="bad.pdf", status="failed"),
        ]
    )
    db_session.commit()

    answer = answering.answer_direct(
        db_session,
        question="Which documents failed indexing?",
        document_id=None,
    )

    assert "Failed" in answer and "failed" in answer
    assert "Indexed" not in answer


def test_direct_handler_reports_when_no_document_failed_indexing(db_session) -> None:
    db_session.add(
        Document(title="Indexed", file_type="pdf", file_path="ok.pdf", status="indexed")
    )
    db_session.commit()

    answer = answering.answer_direct(
        db_session,
        question="Which documents failed indexing?",
        document_id=None,
    )

    assert answer == "No documents have failed indexing."


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Which documents failed indexing?", "failed_indexing"),
        ("How many documents are in the current database?", "count"),
        ("List all uploaded documents.", "inventory"),
        ("What is the status of this document?", "document_status"),
    ],
)
def test_direct_operation_classifier(question, expected) -> None:
    assert answering.classify_direct_operation(question) == expected


def test_direct_handler_count_does_not_expand_inventory(db_session) -> None:
    db_session.add_all(
        [
            Document(title="First", file_type="pdf", file_path="1.pdf", status="indexed"),
            Document(title="Second", file_type="pdf", file_path="2.pdf", status="failed"),
        ]
    )
    db_session.commit()

    answer = answering.answer_direct(
        db_session,
        question="How many documents are in the current database?",
        document_id=None,
    )

    assert answer == "There are 2 documents in the current database."


def test_general_direct_answer_uses_agent_text_completion(monkeypatch) -> None:
    captured: list[list[dict[str, str]]] = []

    def fake_completion(messages, **kwargs):
        captured.append(messages)
        return "agent answer"

    monkeypatch.setattr(answering, "complex_text_completion", fake_completion)

    assert answering.answer_general("what is a hash table?") == "agent answer"
    assert captured[0][1]["content"].startswith(
        "Answer the user's general question concisely."
    )


def test_general_direct_answer_falls_back_when_agent_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(answering, "complex_text_completion", lambda messages, **kwargs: None)

    assert answering.answer_general("what is a hash table?") == "当前无法生成回答。"


def test_answer_generator_retries_unknown_citation_ids(monkeypatch) -> None:
    responses = iter(
        [
            {"answer": "bad", "local_chunk_ids": [999], "web_source_urls": []},
            {"answer": "good", "local_chunk_ids": [7], "web_source_urls": []},
        ]
    )
    monkeypatch.setattr(answering, "answer_json", lambda messages: next(responses))

    result = answering.generate_answer(
        question="q", evidence=[_candidate(7)], web_sources=[], memory_context={}
    )

    assert result.answer == "good"
    assert [item.chunk_id for item in result.citations] == [7]


def test_answer_generator_uses_conservative_fallback_after_second_invalid_reference(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        answering,
        "answer_json",
        lambda messages: {"answer": "bad", "local_chunk_ids": [999], "web_source_urls": []},
    )

    result = answering.generate_answer(
        question="q", evidence=[_candidate(7)], web_sources=[], memory_context={}
    )

    assert "当前证据" in result.answer
    assert [citation.chunk_id for citation in result.citations] == [7]


def test_answer_generator_validates_web_urls_and_deduplicates_evidence(monkeypatch) -> None:
    source = _web()
    evidence = [_candidate(7, score=0.2), _candidate(11, score=0.8)]
    responses = iter(
        [
            {
                "answer": "retry",
                "local_chunk_ids": [11],
                "web_source_urls": ["https://evil.example"],
            },
            {
                "answer": "supported",
                "local_chunk_ids": [11, 7, 11],
                "web_source_urls": [source.entry_url, source.entry_url],
            },
        ]
    )
    monkeypatch.setattr(
        answering,
        "answer_json", lambda messages: next(responses)
    )

    result = answering.generate_answer(
        question="q", evidence=evidence, web_sources=[source], memory_context={}
    )

    assert [citation.chunk_id for citation in result.citations] == [11, 7]
    assert result.citations[0].score == 0.8
    assert [item.entry_url for item in result.web_sources] == [source.entry_url]


def test_answer_generator_uses_rerank_score_and_preserves_metadata(monkeypatch) -> None:
    candidate = _candidate(7, score=0.2).model_copy(
        update={
            "document_id": 3,
            "title": "Title",
            "page_number": 4,
            "chunk_index": 9,
            "rerank_score": 0.91,
        }
    )
    monkeypatch.setattr(
        answering,
        "answer_json",
        lambda messages: {"answer": "supported", "local_chunk_ids": [7], "web_source_urls": []},
    )

    result = answering.generate_answer("q", [candidate], [], {})

    assert result.citations[0].model_dump() == {
        "document_id": 3,
        "chunk_id": 7,
        "title": "Title",
        "page_number": 4,
        "chunk_index": 9,
        "score": 0.91,
        "content": "grounded evidence 7",
    }


def test_answer_generator_empty_evidence_is_conservative(monkeypatch) -> None:
    monkeypatch.setattr(
        answering,
        "answer_json",
        lambda messages: {"answer": "hallucinated", "local_chunk_ids": [7], "web_source_urls": []},
    )
    result = answering.generate_answer("q", [], [], {})
    assert "当前证据" in result.answer
    assert result.citations == []


def test_memory_is_context_only_and_prompt_is_bounded_and_secret_safe(monkeypatch) -> None:
    captured: list[list[dict[str, str]]] = []

    def fake_answer(messages):
        captured.append(messages)
        return {"answer": "supported", "local_chunk_ids": [7], "web_source_urls": []}

    monkeypatch.setattr(answering, "answer_json", fake_answer)
    result = answering.generate_answer(
        "q" * 10_000,
        [_candidate(7)],
        [],
        {"summary": "memory-only-secret", "messages": ["remembered fact"]},
    )

    prompt = json.dumps(captured[0], ensure_ascii=False)
    assert result.answer == "supported"
    assert "memory-only-secret" in prompt
    assert "memory is context only" in prompt.lower()
    assert len(prompt) < 30_000


def test_nested_short_and_long_memory_is_bounded_context_only(monkeypatch) -> None:
    captured: list[list[dict[str, str]]] = []

    def fake_answer(messages):
        captured.append(messages)
        return {"answer": "supported", "local_chunk_ids": [7], "web_source_urls": []}

    monkeypatch.setattr(answering, "answer_json", fake_answer)
    answering.generate_answer(
        "q",
        [_candidate(7)],
        [],
        {
            "short_term": {"summary": "session context"},
            "long_term": [{"content": "long-term context api_key=secret"}],
        },
    )

    prompt = json.dumps(captured[0], ensure_ascii=False)
    assert "session context" in prompt
    assert "long-term context" in prompt
    assert "api_key=secret" not in prompt
    assert len(prompt) < 30_000


def test_nested_memory_dict_redacts_quoted_secret_values(monkeypatch) -> None:
    captured: list[list[dict[str, str]]] = []

    def fake_answer(messages):
        captured.append(messages)
        return {"answer": "supported", "local_chunk_ids": [7], "web_source_urls": []}

    monkeypatch.setattr(answering, "answer_json", fake_answer)
    answering.generate_answer(
        "q",
        [_candidate(7)],
        [],
        {"short_term": {"api_key": "quoted-secret", "token": "quoted-token"}},
    )

    prompt = json.dumps(captured[0], ensure_ascii=False)
    assert "quoted-secret" not in prompt
    assert "quoted-token" not in prompt


def test_memory_redaction_handles_authorization_credentials_and_bare_api_tokens() -> None:
    raw = (
        "Authorization: Bearer bearer-secret "
        "Authorization: Basic basic-secret sk-test-secret"
    )

    for redacted in (
        answering._redact_and_bound(raw, 1000),
        answering._bound_memory({"short_term": raw}),
    ):
        assert "bearer-secret" not in redacted
        assert "basic-secret" not in redacted
        assert "sk-test-secret" not in redacted
        assert "Authorization: Bearer ***" in redacted
        assert "Authorization: Basic ***" in redacted


def test_bound_memory_preserves_agent_research_completion_state() -> None:
    bounded = answering._bound_memory(
        {
            "short_term": {},
            "long_term": [],
            "agent_research": {
                "completion_status": "insufficient_evidence",
                "subquestions": [
                    {
                        "question": "entity B",
                        "status": "unsupported",
                        "evidence_chunk_ids": [],
                    }
                ],
                "accepted_chunk_ids": [],
            },
        }
    )

    assert "insufficient_evidence" in bounded
    assert "unsupported" in bounded
    assert "entity B" in bounded


def test_redaction_consumes_scheme_credentials_for_all_secret_keys() -> None:
    redacted = answering._redact_and_bound(
        "token=Bearer token-secret api_key=Basic key-secret authorization=Bearer auth-secret",
        1000,
    )

    assert "token-secret" not in redacted
    assert "key-secret" not in redacted
    assert "auth-secret" not in redacted
    assert "token=***" in redacted
    assert "api_key=***" in redacted
    assert "authorization=Bearer ***" in redacted


def test_sensitive_web_url_is_not_prompted_or_citable(monkeypatch) -> None:
    captured: list[list[dict[str, str]]] = []
    source = _web("https://arxiv.org/abs/2401.00001?api_key=url-secret")

    def fake_answer(messages):
        captured.append(messages)
        return {
            "answer": "supported",
            "local_chunk_ids": [7],
            "web_source_urls": [source.entry_url],
        }

    monkeypatch.setattr(answering, "answer_json", fake_answer)
    result = answering.generate_answer("q", [_candidate(7)], [source], {})

    prompt = json.dumps(captured[0], ensure_ascii=False)
    assert "url-secret" not in prompt
    assert len(captured) == 2
    assert result.web_sources == []


def test_answer_generator_caps_validation_allowlist_to_prompt_evidence_budget(monkeypatch) -> None:
    evidence = [_candidate(index) for index in range(1, 10)]
    calls: list[list[dict[str, str]]] = []

    def fake_answer(messages):
        calls.append(messages)
        return {"answer": "bad", "local_chunk_ids": [9], "web_source_urls": []}

    monkeypatch.setattr(answering, "answer_json", fake_answer)
    result = answering.generate_answer("q", evidence, [], {})

    assert len(calls) == 2
    assert [citation.chunk_id for citation in result.citations] == list(range(1, 9))
    assert 9 not in [citation.chunk_id for citation in result.citations]
    assert all('"chunk_id": 9' not in message[1]["content"] for message in calls)


def test_answer_generator_reraises_context_window_error(monkeypatch) -> None:
    monkeypatch.setattr(
        answering,
        "answer_json",
        lambda messages: (_ for _ in ()).throw(ContextWindowExceededError("too large")),
    )
    with pytest.raises(ContextWindowExceededError):
        answering.generate_answer("q", [_candidate(7)], [], {})


def test_answer_generator_falls_back_on_ordinary_model_failure(monkeypatch) -> None:
    monkeypatch.setattr(answering, "answer_json", lambda messages: None)
    result = answering.generate_answer("q", [_candidate(7)], [], {})
    assert "当前证据" in result.answer
    assert [citation.chunk_id for citation in result.citations] == [7]


def test_answer_json_uses_agent_settings_and_json_mode_without_response_cache(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        agent_api_key="agent-key",
        agent_base_url="https://agent.example/v1",
        agent_model="deepseek-chat",
        agent_timeout_seconds=12.5,
    )
    calls: list[dict[str, object]] = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"ok"}'))]
            )

    class Client:
        chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(model_clients, "get_settings", lambda: settings)
    monkeypatch.setattr(model_clients, "_agent_client", lambda: Client())
    assert model_clients.answer_json([{"role": "user", "content": "q"}]) == {"answer": "ok"}
    assert model_clients.answer_json([{"role": "user", "content": "q"}]) == {"answer": "ok"}
    assert len(calls) == 2
    assert calls[0]["model"] == "deepseek-chat"
    assert calls[0]["temperature"] == 0.2
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_answer_json_uses_explicit_model_override(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        agent_api_key="agent-key",
        agent_base_url="https://agent.example/v1",
        agent_model="deepseek-chat",
    )
    calls: list[dict[str, object]] = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"ok"}'))]
            )

    class Client:
        chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(model_clients, "get_settings", lambda: settings)
    monkeypatch.setattr(model_clients, "_agent_client", lambda: Client())

    assert model_clients.answer_json(
        [{"role": "user", "content": "q"}], model="qwen3.7-flash"
    ) == {"answer": "ok"}
    assert calls[0]["model"] == "qwen3.7-flash"


def test_answer_json_disables_thinking_for_glm_5_2_json_mode(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        agent_api_key="agent-key",
        agent_base_url="https://agent.example/v1",
        agent_model="deepseek-chat",
    )
    calls: list[dict[str, object]] = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"ok"}'))]
            )

    class Client:
        chat = SimpleNamespace(completions=Completions())

    monkeypatch.setattr(model_clients, "get_settings", lambda: settings)
    monkeypatch.setattr(model_clients, "_agent_client", lambda: Client())

    assert model_clients.answer_json(
        [{"role": "user", "content": "q"}], model="glm-5.2"
    ) == {"answer": "ok"}
    assert calls[0]["model"] == "glm-5.2"
    assert calls[0]["extra_body"] == {"enable_thinking": False}


def test_answer_json_context_window_error_is_preserved(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        agent_api_key="agent-key",
        agent_base_url="https://agent.example/v1",
        agent_model="deepseek-chat",
    )

    class Client:
        chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: (_ for _ in ()).throw(
                    RuntimeError("maximum context length exceeded")
                )
            )
        )

    monkeypatch.setattr(model_clients, "get_settings", lambda: settings)
    monkeypatch.setattr(model_clients, "_agent_client", lambda: Client())
    with pytest.raises(ContextWindowExceededError):
        model_clients.answer_json([{"role": "user", "content": "q"}])


def test_answer_json_returns_none_for_non_context_api_failure(monkeypatch) -> None:
    settings = Settings(
        _env_file=None,
        agent_api_key="agent-key",
        agent_base_url="https://agent.example/v1",
        agent_model="deepseek-chat",
    )

    class Client:
        chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("provider down"))
            )
        )

    monkeypatch.setattr(model_clients, "get_settings", lambda: settings)
    monkeypatch.setattr(model_clients, "_agent_client", lambda: Client())
    assert model_clients.answer_json([{"role": "user", "content": "q"}]) is None


def _agentic_pack(count: int = 2) -> EvidencePack:
    items = [
        EvidencePackItem(
            evidence_id=f"E{index}",
            statement=f"Evidence statement {index}.",
            supports_claim_ids=["C1"],
            confidence=0.9,
            sources=[
                PackedLocalSource(
                    chunk_id=20 + index,
                    document_id=7,
                    title="OT Paper",
                    page_number=index,
                    chunk_index=index,
                    excerpt=f"Evidence excerpt {index}.",
                    score=0.9,
                )
            ],
        )
        for index in range(1, count + 1)
    ]
    return EvidencePack(
        claims=[
            ResearchClaim(
                claim_id="C1",
                question="Why use OT?",
                status="covered",
                evidence_ids=[item.evidence_id for item in items],
            )
        ],
        items=items,
        token_count=100,
    )


def _supported_verification(messages):
    return {
        "passed": True,
        "claims": [
            {
                "claim_id": "C1",
                "status": "supported",
                "evidence_ids": ["E1"],
                "reason": "Supported by E1.",
            }
        ],
        "unsupported_spans": [],
        "citation_errors": [],
        "repair_instruction": "",
    }


def test_agentic_generator_uses_more_than_eight_evidence_items() -> None:
    pack = _agentic_pack(count=9)
    captured = []

    def generate(messages):
        captured.append(messages)
        return {
            "answer": "Grounded answer.",
            "used_evidence_ids": [f"E{index}" for index in range(1, 10)],
            "local_chunk_ids": list(range(21, 30)),
            "web_source_urls": [],
        }

    result = answering.generate_agentic_answer(
        "Why OT?",
        pack,
        {},
        answer_callable=generate,
        verifier_callable=_supported_verification,
    )

    payload = json.loads(captured[0][-1]["content"])
    assert len(payload["evidence_pack"]["items"]) == 9
    assert len(result.citations) == 9
    assert result.verification_status == "passed"


def test_agentic_generator_rejects_unknown_citation_before_verifier() -> None:
    verifier_calls = []

    result = answering.generate_agentic_answer(
        "Why OT?",
        _agentic_pack(),
        {},
        answer_callable=lambda messages: {
            "answer": "Invented.",
            "used_evidence_ids": ["E99"],
            "local_chunk_ids": [999],
            "web_source_urls": [],
        },
        verifier_callable=lambda messages: verifier_calls.append(messages),
    )

    assert verifier_calls == []
    assert result.verification_status == "verification_failed"
    assert "证据校验失败" in result.answer
    assert result.verification_attempts == [
        {
            "attempt": 1,
            "stage": "generation",
            "error_category": "generator_schema_error",
            "passed": False,
        },
        {
            "attempt": 2,
            "stage": "generation_retry",
            "error_category": "generator_schema_error",
            "passed": False,
        },
    ]


def test_agentic_generator_repairs_once_with_same_pack() -> None:
    answers = iter(
        [
            {
                "answer": "OT always prevents collapse.",
                "used_evidence_ids": ["E1"],
                "local_chunk_ids": [21],
                "web_source_urls": [],
            },
            {
                "answer": "OT provides balanced assignments.",
                "used_evidence_ids": ["E1"],
                "local_chunk_ids": [21],
                "web_source_urls": [],
            },
        ]
    )
    generator_messages = []

    def generate(messages):
        generator_messages.append(messages)
        return next(answers)

    verifications = iter(
        [
            {
                "passed": False,
                "claims": [
                    {
                        "claim_id": "C1",
                        "status": "unsupported",
                        "evidence_ids": ["E1"],
                        "reason": "Absolute wording is unsupported.",
                    }
                ],
                "unsupported_spans": ["always prevents"],
                "citation_errors": [],
                "repair_instruction": "Remove the absolute wording.",
            },
            _supported_verification([]),
        ]
    )

    result = answering.generate_agentic_answer(
        "Why OT?",
        _agentic_pack(),
        {},
        answer_callable=generate,
        verifier_callable=lambda messages: next(verifications),
    )

    assert result.answer == "OT provides balanced assignments."
    assert result.verification_status == "passed_after_repair"
    assert [attempt["stage"] for attempt in result.verification_attempts] == [
        "verification",
        "repair_verification",
    ]
    assert [attempt["error_category"] for attempt in result.verification_attempts] == [
        "semantic_verification_failed",
        None,
    ]
    assert result.verification_attempts[0]["unsupported_span_count"] == 1
    assert result.verification_attempts[1]["passed"] is True
    assert len(generator_messages) == 2
    assert "Remove the absolute wording" in generator_messages[1][-1]["content"]
    assert (
        json.loads(generator_messages[0][-1]["content"])["evidence_pack"]
        == json.loads(generator_messages[1][-1]["content"])["evidence_pack"]
    )


def test_agentic_generator_returns_conservative_answer_after_failed_repair() -> None:
    failed = {
        "passed": False,
        "claims": [
            {
                "claim_id": "C1",
                "status": "unsupported",
                "evidence_ids": ["E1"],
                "reason": "Unsupported.",
            }
        ],
        "unsupported_spans": ["unsupported"],
        "citation_errors": [],
        "repair_instruction": "Use only E1.",
    }

    result = answering.generate_agentic_answer(
        "Why OT?",
        _agentic_pack(),
        {},
        answer_callable=lambda messages: {
            "answer": "Unsupported answer.",
            "used_evidence_ids": ["E1"],
            "local_chunk_ids": [21],
            "web_source_urls": [],
        },
        verifier_callable=lambda messages: failed,
    )

    assert result.verification_status == "verification_failed"
    assert result.citations
    assert "无法生成通过一致性校验的完整回答" in result.answer
    assert [attempt["stage"] for attempt in result.verification_attempts] == [
        "verification",
        "repair_verification",
    ]
    assert all(
        attempt["error_category"] == "semantic_verification_failed"
        for attempt in result.verification_attempts
    )


def test_agentic_generator_distinguishes_repair_generation_failure() -> None:
    answers = iter(
        [
            {
                "answer": "OT always prevents collapse.",
                "used_evidence_ids": ["E1"],
                "local_chunk_ids": [21],
                "web_source_urls": [],
            },
            {
                "answer": "Invented repair.",
                "used_evidence_ids": ["E99"],
                "local_chunk_ids": [999],
                "web_source_urls": [],
            },
        ]
    )
    failed = {
        "passed": False,
        "claims": [
            {
                "claim_id": "C1",
                "status": "unsupported",
                "evidence_ids": ["E1"],
                "reason": "Unsupported.",
            }
        ],
        "unsupported_spans": ["always"],
        "citation_errors": [],
        "repair_instruction": "Remove the absolute wording.",
    }

    result = answering.generate_agentic_answer(
        "Why OT?",
        _agentic_pack(),
        {},
        answer_callable=lambda messages: next(answers),
        verifier_callable=lambda messages: failed,
    )

    assert result.verification_status == "verification_failed"
    assert [attempt["stage"] for attempt in result.verification_attempts] == [
        "verification",
        "repair_generation",
    ]
    assert result.verification_attempts[1]["error_category"] == (
        "generator_schema_error"
    )
