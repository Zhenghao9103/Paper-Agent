import fitz
import pytest
from backend.app.schemas.arxiv import ArxivPaper
from backend.app.services.context_checkpoint import ContextCompactionUnavailable
from fastapi.testclient import TestClient


def make_pdf_bytes(text: str) -> bytes:
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    data = pdf.tobytes()
    pdf.close()
    return data


def test_chat_ask_returns_cited_chinese_answer(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr("backend.app.agents.tools.search_arxiv", lambda *args, **kwargs: [])
    files = {
        "file": (
            "rag.pdf",
            make_pdf_bytes("Retrieval augmented generation uses external evidence."),
            "application/pdf",
        )
    }
    client.post("/api/documents/upload", files=files)

    response = client.post(
        "/api/chat/ask",
        json={"question": "What does retrieval augmented generation use?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] > 0
    assert "\u672c\u5730\u77e5\u8bc6\u5e93" in payload["answer"]
    assert payload["citations"]
    assert payload["citations"][0]["page_number"] == 1
    assert "answer_synthesizer_zh" in payload["trace"]
    assert "bge_rerank" in payload["trace"]

    memories = client.get("/api/memories").json()
    assert memories
    assert memories[0]["memory_type"] == "research_topic"
    assert memories[0]["status"] == "active"
    assert memories[0]["importance_score"] == 0.5
    assert "confidence" not in memories[0]

    transcript = client.get(f"/api/transcripts/session/{payload['session_id']}").json()
    assert transcript
    assert "qa" in [event["event_type"] for event in transcript]


def test_chat_ask_reports_retrieval_degraded_when_vector_search_fails(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr("backend.app.agents.tools.search_arxiv", lambda *args, **kwargs: [])
    files = {
        "file": (
            "sqlite-fallback.pdf",
            make_pdf_bytes("SpecNet2 studies spectral embedding without orthogonalization."),
            "application/pdf",
        )
    }
    client.post("/api/documents/upload", files=files)

    def broken_query(*args, **kwargs):
        raise RuntimeError("broken vector index")

    monkeypatch.setattr("backend.app.services.research.hybrid_search", broken_query)
    response = client.post(
        "/api/chat/ask",
        json={"question": "What does SpecNet2 study?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"] == []
    assert "当前证据不足" in payload["answer"]
    assert "retrieval_degraded" in payload["trace"]


def test_chat_ask_overview_question_does_not_scan_full_database(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr("backend.app.agents.tools.search_arxiv", lambda *args, **kwargs: [])
    papers = [
        (
            "deep-spectral.pdf",
            "Deep spectral clustering learns robust cluster assignments with neural networks.",
        ),
        (
            "subspace.pdf",
            "Self-expressive subspace clustering studies sparse representations.",
        ),
    ]
    for filename, text in papers:
        client.post(
            "/api/documents/upload",
            files={"file": (filename, make_pdf_bytes(text), "application/pdf")},
        )

    monkeypatch.setattr("backend.app.services.research.hybrid_search", lambda *args, **kwargs: [])
    response = client.post(
        "/api/chat/ask",
        json={
            "question": (
                "\u5f53\u524d\u77e5\u8bc6\u5e93\u7684"
                "\u8bba\u6587\u4e3b\u8981\u96c6\u4e2d\u5728\u54ea\u4e9b\u65b9\u9762"
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"] == []
    assert "当前证据不足" in payload["answer"]
    assert "retrieval_degraded" not in payload["trace"]


def test_chat_ask_counts_current_database_documents_without_vector_rag(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr("backend.app.agents.tools.search_arxiv", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "backend.app.services.research.hybrid_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("vector search should not run")
        ),
    )
    papers = [
        ("first.pdf", "First paper studies retrieval augmented generation."),
        ("second.pdf", "Second paper studies spectral clustering."),
    ]
    for filename, text in papers:
        client.post(
            "/api/documents/upload",
            files={"file": (filename, make_pdf_bytes(text), "application/pdf")},
        )

    response = client.post(
        "/api/chat/ask",
        json={"question": "当前数据库中有几篇论文"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "当前知识库共有 2 篇论文" in payload["answer"]
    assert "first" in payload["answer"]
    assert "second" in payload["answer"]
    assert payload["citations"] == []
    assert "database_document_summary" in payload["trace"]
    assert "local_retrieve" not in payload["trace"]


def test_chat_ask_ignores_zero_score_vector_matches(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr("backend.app.agents.tools.search_arxiv", lambda *args, **kwargs: [])
    client.post(
        "/api/documents/upload",
        files={
            "file": (
                "zero-score.pdf",
                make_pdf_bytes("Relevant paper content that should not matter here."),
                "application/pdf",
            )
        },
    )
    monkeypatch.setattr(
        "backend.app.services.research.hybrid_search",
        lambda *args, **kwargs: [
            {
                "content": (
                    "JOURNAL OF LATEX CLASS FILES author biography and unrelated table values."
                ),
                "metadata": {
                    "document_id": 1,
                    "title": "Zero Score Paper",
                    "page_number": 14,
                    "chunk_index": 1,
                },
                "score": 0.0,
            }
        ],
    )

    response = client.post(
        "/api/chat/ask",
        json={"question": "这个问题没有相关证据"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"] == []
    assert "当前证据不足" in payload["answer"]
    assert "JOURNAL OF LATEX CLASS FILES" not in payload["answer"]
    assert any(event.get("type") == "evidence_selected" for event in payload["trace_events"])


def test_chat_ask_uses_llm_prompt_when_evidence_is_available(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr("backend.app.agents.tools.search_arxiv", lambda *args, **kwargs: [])
    captured_prompts: list[str] = []

    def fake_complete(messages):
        captured_prompts.append(str(messages))
        return {
            "answer": "LLM 基于证据回答：RAG 会使用外部证据。",
            "local_chunk_ids": [1],
            "web_source_urls": [],
        }

    monkeypatch.setattr("backend.app.services.answering.answer_json", fake_complete)
    monkeypatch.setattr(
        "backend.app.services.research.hybrid_search",
        lambda *args, **kwargs: [
            {
                "content": (
                    "Retrieval augmented generation uses external evidence to ground answers."
                ),
                "metadata": {
                    "document_id": 1,
                    "title": "RAG Paper",
                    "page_number": 1,
                    "chunk_index": 0,
                },
                "score": 0.82,
            }
        ],
    )
    client.post(
        "/api/documents/upload",
        files={"file": ("rag-prompt.pdf", make_pdf_bytes("placeholder"), "application/pdf")},
    )

    response = client.post(
        "/api/chat/ask",
        json={"question": "RAG 使用什么来回答问题？"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "LLM 基于证据回答：RAG 会使用外部证据。"
    assert "llm_answer_synthesizer" in payload["trace"]
    assert captured_prompts
    assert "RAG" in captured_prompts[0]
    assert "Retrieval augmented generation uses external evidence" in captured_prompts[0]


def test_simple_route_does_not_search_arxiv(
    client: TestClient,
    monkeypatch,
) -> None:
    captured_prompts: list[str] = []

    def fake_complete(messages):
        captured_prompts.append(str(messages))
        return {
            "answer": "LLM 联网总结：RAG 与 agent 的关系是通过外部检索增强任务执行。",
            "local_chunk_ids": [1],
            "web_source_urls": [],
        }

    monkeypatch.setattr("backend.app.services.answering.answer_json", fake_complete)
    client.post(
        "/api/documents/upload",
        files={
            "file": (
                "local-rag.pdf",
                make_pdf_bytes("Local RAG evidence discusses retrieval augmented generation."),
                "application/pdf",
            )
        },
    )

    response = client.post(
        "/api/chat/ask",
        json={"question": "How does retrieval augmented generation relate to agents?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "LLM 联网总结：RAG 与 agent 的关系是通过外部检索增强任务执行。"
    assert captured_prompts
    assert payload["web_sources"] == []


def test_chat_overview_uses_llm_to_summarize_web_context(
    client: TestClient,
    monkeypatch,
) -> None:
    captured_prompts: list[str] = []

    def fake_complete(messages):
        captured_prompts.append(str(messages))
        return {
            "answer": "LLM 概览总结：当前论文集中在深度聚类，并结合联网结果补充相关背景。",
            "local_chunk_ids": [1],
            "web_source_urls": [],
        }

    monkeypatch.setattr("backend.app.services.answering.answer_json", fake_complete)
    monkeypatch.setattr("backend.app.services.research.hybrid_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "backend.app.agents.tools.search_arxiv",
        lambda query, max_results=3: [
            ArxivPaper(
                title="Design Principles of Zero-Shot Self-Supervised Unknown Emitter Detectors",
                authors=["A. Researcher"],
                summary="Emitter detection systems need robust self-supervised recognition.",
                published="2025-11-10",
                pdf_url="https://arxiv.org/pdf/2511.00001",
                entry_url="https://arxiv.org/abs/2511.00001",
            )
        ],
        raising=False,
    )
    for filename, text in [
        ("deep.pdf", "Deep clustering learns neural cluster assignments."),
        ("spectral.pdf", "Spectral clustering uses graph Laplacian constraints."),
    ]:
        client.post(
            "/api/documents/upload",
            files={"file": (filename, make_pdf_bytes(text), "application/pdf")},
        )

    response = client.post(
        "/api/chat/ask",
        json={
            "question": "当前知识库的论文主要集中在哪些方面",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "当前证据不足" in payload["answer"]
    assert "llm_answer_synthesizer" not in payload["trace"]
    assert captured_prompts == []
    assert "依据来源" not in payload["answer"]


def test_chat_ask_keeps_local_chinese_answer_when_arxiv_fails(
    client: TestClient,
    monkeypatch,
) -> None:
    client.post(
        "/api/documents/upload",
        files={
            "file": (
                "local-only.pdf",
                make_pdf_bytes("Spectral clustering builds graph Laplacian representations."),
                "application/pdf",
            )
        },
    )

    def broken_search(*args, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("backend.app.agents.tools.search_arxiv", broken_search, raising=False)
    response = client.post(
        "/api/chat/ask",
        json={"question": "\u8c31\u805a\u7c7b\u65b9\u6cd5\u7684\u6838\u5fc3\u662f\u4ec0\u4e48"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "\u672c\u5730\u77e5\u8bc6\u5e93" in payload["answer"]
    assert payload["web_sources"] == []


def test_chat_ask_extracts_innovation_answer_instead_of_echoing_first_page(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr("backend.app.agents.tools.search_arxiv", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "backend.app.services.research.hybrid_search",
        lambda *args, **kwargs: [
            {
                "content": (
                    "JOURNAL OF LATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 "
                    "Bootstrap Deep Spectral Clustering with Optimal Transport."
                ),
                "metadata": {
                    "document_id": 1,
                    "title": "Bootstrap Deep Spectral Clustering with Optimal Transport",
                    "page_number": 1,
                    "chunk_index": 0,
                },
                "score": 0.9,
            },
            {
                "content": (
                    "To handle the above issues, we propose a bootstrapped deep spectral "
                    "clustering model using optimal transport. The method aligns multiple "
                    "sampled clustering assignments and improves stability."
                ),
                "metadata": {
                    "document_id": 1,
                    "title": "Bootstrap Deep Spectral Clustering with Optimal Transport",
                    "page_number": 2,
                    "chunk_index": 1,
                },
                "score": 0.8,
            },
            {
                "content": "0.498 0.710 0.815 0.675 BootSC Ours table values and sample sizes.",
                "metadata": {
                    "document_id": 1,
                    "title": "Bootstrap Deep Spectral Clustering with Optimal Transport",
                    "page_number": 11,
                    "chunk_index": 2,
                },
                "score": 0.7,
            },
        ],
    )

    response = client.post(
        "/api/chat/ask",
        json={
            "question": (
                "Bootstrap Deep Spectral Clustering with Optimal Transport"
                "这篇论文的创新点是什么"
            )
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]
    assert "创新点" in answer
    assert "optimal transport" in answer.lower()
    assert "bootstrapped deep spectral clustering" in answer.lower()
    assert "JOURNAL OF LATEX CLASS FILES" not in answer
    assert "0.498 0.710" not in answer


def test_chat_ask_exposes_rerank_error_in_trace_events(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr("backend.app.agents.tools.search_arxiv", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "backend.app.services.research.hybrid_search",
        lambda *args, **kwargs: [
            {
                "content": "The method proposes a stable graph model.",
                "metadata": {
                    "document_id": 1,
                    "title": "Rerank Error Paper",
                    "page_number": 2,
                    "chunk_index": 0,
                },
                "score": 0.8,
            }
        ],
    )

    def broken_rerank(*args, **kwargs):
        raise RuntimeError("reranker model missing")

    monkeypatch.setattr("backend.app.rag.rerank.rerank_chunks", broken_rerank)

    response = client.post(
        "/api/chat/ask",
        json={"question": "这篇论文的创新点是什么"},
    )

    assert response.status_code == 200
    trace_events = response.json()["trace_events"]
    assert any(event.get("type") == "evidence_selected" for event in trace_events)


def test_chat_ask_maps_context_compaction_failure_to_503(
    client: TestClient,
    monkeypatch,
) -> None:
    def fail_question(*args, **kwargs):
        raise ContextCompactionUnavailable("上下文溢出恢复失败。")

    monkeypatch.setattr(
        "backend.app.api.routes.chat.answer_question",
        fail_question,
    )

    response = client.post("/api/chat/ask", json={"question": "too much context"})

    assert response.status_code == 503
    assert response.json()["detail"] == "上下文溢出恢复失败。"


def test_chat_stream_reports_context_compaction_failure_as_safe_sse(
    client: TestClient,
    monkeypatch,
) -> None:
    def fail_question(*args, **kwargs):
        raise ContextCompactionUnavailable("internal context detail")

    monkeypatch.setattr(
        "backend.app.api.routes.chat.answer_question",
        fail_question,
    )

    response = client.post("/api/chat/stream", json={"question": "too much context"})

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "上下文处理失败，请缩短问题后重试。" in response.text
    assert "internal context detail" not in response.text


def test_chat_route_does_not_map_unrelated_errors_to_503(
    client: TestClient,
    monkeypatch,
) -> None:
    def fail_question(*args, **kwargs):
        raise RuntimeError("unrelated failure")

    monkeypatch.setattr(
        "backend.app.api.routes.chat.answer_question",
        fail_question,
    )

    with pytest.raises(RuntimeError, match="unrelated failure"):
        client.post("/api/chat/ask", json={"question": "ordinary failure"})
