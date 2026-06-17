import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models.chat import ChatMessage, ChatSession
from ..models.chunk import DocumentChunk
from ..models.document import Document
from ..rag.vector_store import query_chunks
from ..rag.rerank import rerank_chunks
from ..schemas.arxiv import ArxivPaper
from ..schemas.chat import Citation, WebSource
from ..services.arxiv_search import search_arxiv
from ..services.llm import complete
from ..services.memory import MemoryHit, query_relevant_memories

MIN_RELEVANCE_SCORE = 0.15


class ResearchState(TypedDict, total=False):
    question: str
    document_id: int | None
    db: Session
    matches: list[dict]
    citations: list[Citation]
    web_sources: list[WebSource]
    web_error: str
    answer: str
    trace: list[str]
    memory_hits: list[MemoryHit]
    session_id: int | None
    short_term_memory: dict
    errors: list[dict]
    direct_answer: str


def load_memory(state: ResearchState) -> ResearchState:
    state.setdefault("trace", []).append("load_short_term_memory")
    if "db" in state and state.get("session_id") is not None:
        short_term_memory = _load_short_term_memory(state["db"], int(state["session_id"]))
        state["short_term_memory"] = short_term_memory
        if short_term_memory.get("session_summary") or short_term_memory.get("recent_messages"):
            state["trace"].append("load_recent_chat_history")
    try:
        state["memory_hits"] = query_relevant_memories(state["question"], limit=3)
        if state["memory_hits"]:
            state["trace"].append("load_long_term_vector_memory")
    except Exception:
        state["memory_hits"] = []
        state["trace"].append("long_term_memory_unavailable")
    return state


def _load_short_term_memory(db: Session, session_id: int) -> dict:
    session = db.get(ChatSession, session_id)
    recent_messages = list(
        db.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(6)
        )
    )
    recent_messages.reverse()
    return {
        "session_summary": session.session_summary if session else None,
        "recent_messages": [
            {"role": message.role, "content": message.content}
            for message in recent_messages
        ],
    }


def _tokenize(text: str) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]", text)
        if token.strip()
    }
    lowered = text.lower()
    expansions = {
        "\u805a\u7c7b": {"cluster", "clustering"},
        "\u8c31\u805a\u7c7b": {"spectral", "clustering"},
        "\u65b9\u6cd5": {"method", "model", "framework", "propose", "proposed"},
        "\u5b9e\u9a8c": {"experiment", "experiments", "evaluation", "results"},
        "\u7ed3\u679c": {"result", "results", "performance"},
        "\u56fe\u8868": {"figure", "table", "chart"},
        "\u521b\u65b0": {"novel", "new", "propose", "proposed", "contribution"},
        "\u6458\u8981": {"abstract", "summary"},
        "\u5c40\u9650": {"limitation", "limitations", "future"},
    }
    for phrase, related_tokens in expansions.items():
        if phrase in lowered:
            tokens.update(related_tokens)
    return tokens


def _is_overview_question(question: str) -> bool:
    lowered = question.lower()
    chinese_markers = [
        "\u77e5\u8bc6\u5e93",
        "\u8bba\u6587",
        "\u4e3b\u8981",
        "\u96c6\u4e2d",
        "\u65b9\u9762",
        "\u65b9\u5411",
        "\u9886\u57df",
        "\u54ea\u4e9b",
    ]
    overview_markers = {"overview", "topics", "themes", "areas", "directions"}
    return sum(marker in question for marker in chinese_markers) >= 3 or any(
        marker in lowered for marker in overview_markers
    )


def _sqlite_matches(
    db: Session,
    *,
    question: str,
    document_id: int | None,
    limit: int,
) -> list[dict[str, Any]]:
    statement = (
        select(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .order_by(DocumentChunk.document_id.desc(), DocumentChunk.page_number, DocumentChunk.chunk_index)
        .limit(300)
    )
    if document_id is not None:
        statement = statement.where(DocumentChunk.document_id == document_id)

    question_tokens = _tokenize(question)
    scored_matches: list[tuple[float, DocumentChunk, Document]] = []
    for chunk, document in db.execute(statement).all():
        chunk_tokens = _tokenize(chunk.content)
        overlap = len(question_tokens & chunk_tokens)
        if overlap > 0:
            score = overlap / max(len(question_tokens), 1)
            scored_matches.append((score, chunk, document))

    scored_matches.sort(key=lambda item: item[0], reverse=True)
    return [_match_from_chunk(score, chunk, document) for score, chunk, document in scored_matches[:limit]]


def _document_overview_matches(
    db: Session,
    *,
    document_id: int | None,
    limit: int,
) -> list[dict[str, Any]]:
    statement = (
        select(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .order_by(DocumentChunk.document_id.desc(), DocumentChunk.page_number, DocumentChunk.chunk_index)
        .limit(600)
    )
    if document_id is not None:
        statement = statement.where(DocumentChunk.document_id == document_id)

    matches: list[dict[str, Any]] = []
    seen_documents: set[int] = set()
    for chunk, document in db.execute(statement).all():
        if document.id in seen_documents:
            continue
        seen_documents.add(document.id)
        match = _match_from_chunk(0.25, chunk, document)
        match["metadata"]["retrieval_mode"] = "overview"
        matches.append(match)
        if len(matches) >= limit:
            break
    return matches


def _match_from_chunk(score: float, chunk: DocumentChunk, document: Document) -> dict[str, Any]:
    return {
        "content": chunk.content,
        "metadata": {
            "document_id": chunk.document_id,
            "title": document.title,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "file_type": document.file_type,
        },
        "score": score,
    }


def handle_database_query(state: ResearchState) -> ResearchState:
    if "db" not in state or not _is_database_document_summary_question(state["question"]):
        return state

    state.setdefault("trace", []).append("database_document_summary")
    documents = list(
        state["db"].scalars(
            select(Document).order_by(Document.created_at.asc(), Document.id.asc())
        )
    )
    if not documents:
        state["direct_answer"] = "当前知识库中还没有论文。"
        return state

    papers = "\n".join(
        f"{index}. {document.title}" for index, document in enumerate(documents, start=1)
    )
    state["direct_answer"] = f"当前知识库共有 {len(documents)} 篇论文：\n{papers}"
    return state


def _is_database_document_summary_question(question: str) -> bool:
    lowered = question.lower()
    count_markers = ("几篇", "多少篇", "多少个", "有几篇", "有哪些", "列出", "列表")
    corpus_markers = ("数据库", "知识库", "当前", "已上传", "库里")
    paper_markers = ("论文", "文档", "paper", "papers", "document", "documents")
    english_markers = ("how many papers", "what papers", "which papers", "list papers")
    return (
        any(marker in lowered for marker in english_markers)
        or (
            any(marker in question for marker in count_markers)
            and any(marker in question for marker in corpus_markers)
            and any(marker in lowered for marker in paper_markers)
        )
    )


def _filter_relevant_matches(
    matches: list[dict],
    trace: list[str],
    *,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> list[dict]:
    filtered = [match for match in matches if float(match.get("score", 0.0)) >= min_score]
    if len(filtered) < len(matches):
        trace.append("filter_low_relevance_matches")
    return filtered


def local_retrieve(state: ResearchState) -> ResearchState:
    state.setdefault("trace", []).append("local_retrieve")
    try:
        matches = query_chunks(
            state["question"],
            document_id=state.get("document_id"),
            limit=5,
        )
        matches = _filter_relevant_matches(matches, state["trace"])
    except Exception:
        matches = []
        state.setdefault("trace", []).append("sqlite_fallback_retrieve")

    if not matches and "db" in state:
        if "sqlite_fallback_retrieve" not in state.setdefault("trace", []):
            state["trace"].append("sqlite_fallback_retrieve")
        matches = _sqlite_matches(
            state["db"],
            question=state["question"],
            document_id=state.get("document_id"),
            limit=5,
        )
        if not matches and _is_overview_question(state["question"]):
            state["trace"].append("sqlite_overview_retrieve")
            matches = _document_overview_matches(
                state["db"],
                document_id=state.get("document_id"),
                limit=8,
            )
    state["matches"] = matches
    return state


def rerank_retrieval(state: ResearchState) -> ResearchState:
    matches = state.get("matches", [])
    if not matches:
        state.setdefault("trace", []).append("bge_rerank_skipped")
        return state
    try:
        state["matches"] = _filter_relevant_matches(
            rerank_chunks(state["question"], matches, top_k=5),
            state["trace"],
        )
        state.setdefault("trace", []).append("bge_rerank")
    except Exception as exc:
        state.setdefault("trace", []).append("bge_rerank_unavailable")
        state.setdefault("errors", []).append(
            {
                "node": "rerank_retrieval",
                "message": str(exc),
                "recoverable": True,
            }
        )
    return state


def web_search(state: ResearchState) -> ResearchState:
    if state.get("direct_answer"):
        return state
    state.setdefault("trace", []).append("web_search")
    query = _build_web_query(state["question"], state.get("matches", []))
    try:
        papers = search_arxiv(query, max_results=3)
    except Exception as exc:
        state.setdefault("trace", []).append("web_search_failed")
        state["web_sources"] = []
        state["web_error"] = str(exc)
        return state

    state["web_sources"] = [_web_source_from_arxiv(paper) for paper in papers]
    return state


def _build_web_query(question: str, matches: list[dict]) -> str:
    words = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question)
        if word.lower() not in {"what", "does", "how", "the", "and", "with", "for", "from"}
    ]
    if len(words) >= 2:
        return " ".join(words[:8])

    title_words: list[str] = []
    for match in matches[:3]:
        title = str(match.get("metadata", {}).get("title", ""))
        title_words.extend(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", title))
    if title_words:
        return " ".join(title_words[:8])

    tokens = _tokenize(question)
    fallback_terms = [
        "spectral clustering",
        "deep clustering",
        "retrieval augmented generation",
        "paper agent",
    ]
    if {"spectral", "clustering"} & tokens or {"cluster", "clustering"} & tokens:
        return fallback_terms[0]
    return question


def _web_source_from_arxiv(paper: ArxivPaper) -> WebSource:
    return WebSource(
        title=paper.title,
        authors=paper.authors,
        summary=paper.summary,
        published=paper.published,
        entry_url=paper.entry_url,
        pdf_url=paper.pdf_url,
    )


def synthesize_answer(state: ResearchState) -> ResearchState:
    if state.get("direct_answer"):
        state["citations"] = []
        state["web_sources"] = []
        state["answer"] = state["direct_answer"]
        return state

    state.setdefault("trace", []).append("answer_synthesizer_zh")
    citations = [
        Citation(
            document_id=int(match["metadata"]["document_id"]),
            title=str(match["metadata"].get("title", "Untitled")),
            page_number=int(match["metadata"].get("page_number", 0)),
            chunk_index=int(match["metadata"].get("chunk_index", 0)),
            score=round(float(match["score"]), 4),
            content=str(match["content"]),
        )
        for match in state.get("matches", [])
    ]
    state["citations"] = citations
    web_sources = state.get("web_sources", [])
    if not citations:
        if web_sources and (llm_answer := _synthesize_with_llm(state["question"], citations, web_sources)):
            state.setdefault("trace", []).append("llm_answer_synthesizer")
            state["answer"] = llm_answer
        else:
            state["answer"] = _synthesize_no_local_answer(web_sources, state.get("web_error"))
    elif llm_answer := _synthesize_with_llm(state["question"], citations, web_sources):
        state.setdefault("trace", []).append("llm_answer_synthesizer")
        state["answer"] = llm_answer
    elif any(match["metadata"].get("retrieval_mode") == "overview" for match in state.get("matches", [])):
        state["answer"] = _synthesize_overview(citations, web_sources, state.get("web_error"))
    elif _is_innovation_question(state["question"]):
        state["answer"] = _synthesize_innovation_answer(citations, web_sources, state.get("web_error"))
    else:
        state["answer"] = _synthesize_research_answer(citations, web_sources, state.get("web_error"))
    return state


def _synthesize_with_llm(
    question: str,
    citations: list[Citation],
    web_sources: list[WebSource],
) -> str | None:
    answer = complete(_build_rag_prompt(question, citations, web_sources))
    if not answer:
        return None
    return answer.strip()


def _build_rag_prompt(
    question: str,
    citations: list[Citation],
    web_sources: list[WebSource],
) -> str:
    evidence = "\n".join(
        (
            f"[{index}] 标题：{citation.title}\n"
            f"页码：{citation.page_number}\n"
            f"相关度：{citation.score}\n"
            f"内容：{_shorten(citation.content, 700)}"
        )
        for index, citation in enumerate(citations[:5], start=1)
    ) or "无。"
    web_context = "\n".join(
        f"- {source.title}（{source.published}）：{_shorten(source.summary, 300)}"
        for source in web_sources[:3]
    ) or "无。"
    return (
        "你是一个严谨的中文论文 RAG 问答助手。请只根据给定证据回答用户问题。\n"
        "要求：\n"
        "1. 先直接回答问题，不要复述无关片段。\n"
        "2. 如果证据不足，明确说“当前知识库证据不足”，不要编造。\n"
        "3. 不要在回答正文中单独列出依据来源、来源列表、页码列表或相关度分数。\n"
        "4. 可以自然提到论文名，但不要输出“依据来源：”“来源：”“相关度：”这类调试信息。\n"
        "5. 忽略作者简介、参考文献、页眉页脚、纯数字表格等无关噪声。\n\n"
        f"用户问题：{question}\n\n"
        f"本地知识库证据：\n{evidence}\n\n"
        f"联网检索补充：\n{web_context}\n\n"
        "请用中文给出简洁、贴合问题的回答。"
    )


def _is_innovation_question(question: str) -> bool:
    lowered = question.lower()
    markers = {
        "innovation",
        "innovations",
        "novel",
        "novelty",
        "contribution",
        "contributions",
        "\u521b\u65b0",
        "\u521b\u65b0\u70b9",
        "\u8d21\u732e",
    }
    return any(marker in lowered for marker in markers)


def _synthesize_innovation_answer(
    citations: list[Citation],
    web_sources: list[WebSource],
    web_error: str | None,
) -> str:
    selected = _select_relevant_sentences(
        citations,
        markers=(
            "propose",
            "proposed",
            "present",
            "novel",
            "contribution",
            "to handle",
            "address",
            "optimal transport",
            "bootstrap",
            "stability",
            "\u63d0\u51fa",
            "\u521b\u65b0",
            "\u89e3\u51b3",
        ),
        limit=4,
    )
    if not selected:
        selected = _select_relevant_sentences(citations, markers=(), limit=3)

    points = "\n".join(
        f"- {sentence}\uff08{citation.title}\uff0c第 {citation.page_number} 页\uff09"
        for citation, sentence in selected
    )
    if not points:
        points = "- 当前检索证据不足以稳定抽取创新点，建议查看 Introduction/Contribution/Method 页面。"
    return (
        "\u521b\u65b0\u70b9\u6982\u62ec\uff1a\n"
        f"{points}\n\n"
        f"{_format_web_section(web_sources, web_error)}"
    )


def _select_relevant_sentences(
    citations: list[Citation],
    *,
    markers: tuple[str, ...],
    limit: int,
) -> list[tuple[Citation, str]]:
    selected: list[tuple[Citation, str]] = []
    lowered_markers = tuple(marker.lower() for marker in markers)
    for citation in citations:
        for sentence in _split_sentences(citation.content):
            if _looks_like_noise(sentence):
                continue
            lowered = sentence.lower()
            if lowered_markers and not any(marker in lowered for marker in lowered_markers):
                continue
            selected.append((citation, _shorten(sentence, 220)))
            if len(selected) >= limit:
                return selected
    return selected


def _split_sentences(text: str) -> list[str]:
    compact = " ".join(text.split())
    if not compact:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", compact) if len(part.strip()) > 20]


def _looks_like_noise(sentence: str) -> bool:
    lowered = sentence.lower()
    if "journal of latex class files" in lowered:
        return True
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", sentence)
    if not tokens:
        return True
    numeric_count = sum(1 for token in tokens if re.fullmatch(r"\d+(?:\.\d+)?", token))
    return numeric_count / len(tokens) > 0.45


def _synthesize_research_answer(
    citations: list[Citation],
    web_sources: list[WebSource],
    web_error: str | None,
) -> str:
    local_evidence = "\n".join(
        f"- {citation.title}\uff08第 {citation.page_number} 页\uff09：{_shorten(citation.content, 220)}"
        for citation in citations[:3]
    )
    web_section = _format_web_section(web_sources, web_error)
    conclusion = _shorten(citations[0].content, 260)
    return (
        "\u672c\u5730\u77e5\u8bc6\u5e93\u8bc1\u636e\uff1a\n"
        f"{local_evidence}\n\n"
        f"{web_section}\n\n"
        "\u7efc\u5408\u56de\u7b54\uff1a\n"
        f"\u6839\u636e\u672c\u5730\u8bba\u6587\u7247\u6bb5\uff0c{conclusion}"
    )


def _synthesize_no_local_answer(web_sources: list[WebSource], web_error: str | None) -> str:
    if web_sources:
        return (
            "\u672c\u5730\u77e5\u8bc6\u5e93\u6682\u672a\u627e\u5230\u76f4\u63a5\u8bc1\u636e\u3002\n\n"
            f"{_format_web_section(web_sources, web_error)}\n\n"
            "\u7efc\u5408\u56de\u7b54\uff1a\n"
            "\u53ef\u4ee5\u5148\u53c2\u8003\u4e0a\u9762\u7684 arXiv \u68c0\u7d22\u7ed3\u679c\uff0c"
            "\u518d\u8865\u5145\u76f8\u5173\u8bba\u6587\u5230\u672c\u5730\u77e5\u8bc6\u5e93\u540e\u8fdb\u884c\u66f4\u7ec6\u7684 RAG \u95ee\u7b54\u3002"
        )
    return (
        "\u672a\u627e\u5230\u8db3\u591f\u76f8\u5173\u7684\u672c\u5730\u8bc1\u636e\u3002\n\n"
        f"{_format_web_section(web_sources, web_error)}"
    )


def _synthesize_overview(
    citations: list[Citation],
    web_sources: list[WebSource],
    web_error: str | None,
) -> str:
    topic_rules = [
        ("spectral clustering", "\u8c31\u805a\u7c7b"),
        ("deep clustering", "\u6df1\u5ea6\u805a\u7c7b"),
        ("subspace clustering", "\u5b50\u7a7a\u95f4\u805a\u7c7b"),
        ("optimal transport", "\u6700\u4f18\u4f20\u8f93"),
        ("kolmogorov", "Kolmogorov-Arnold/KAN \u7f51\u7edc"),
        ("self-expressive", "\u81ea\u8868\u8fbe\u5b66\u4e60"),
        ("sparse", "\u7a00\u758f\u8868\u793a\u4e0e\u7a00\u758f\u7ed3\u6784"),
        ("graph", "\u56fe\u7ed3\u6784\u4e0e\u76f8\u4f3c\u6027\u5efa\u6a21"),
        ("neural network", "\u795e\u7ecf\u7f51\u7edc\u8868\u793a\u5b66\u4e60"),
    ]
    topic_counts: dict[str, int] = {}
    for citation in citations:
        text = f"{citation.title} {citation.content}".lower()
        for needle, label in topic_rules:
            if needle in text:
                topic_counts[label] = topic_counts.get(label, 0) + 1

    ranked_topics = sorted(topic_counts.items(), key=lambda item: item[1], reverse=True)
    if ranked_topics:
        topic_text = "\n".join(f"- {topic}" for topic, _ in ranked_topics[:5])
    else:
        topic_text = "- \u5df2\u4e0a\u4f20\u8bba\u6587\u7684\u65b9\u6cd5\u3001\u4efb\u52a1\u548c\u5b9e\u9a8c\u4e3b\u9898"

    return (
        "\u4ece\u5f53\u524d\u77e5\u8bc6\u5e93\u5df2\u89e3\u6790\u7684\u8bba\u6587\u770b\uff0c"
        "\u4e3b\u8981\u96c6\u4e2d\u5728\u4ee5\u4e0b\u65b9\u5411\uff1a\n"
        f"{topic_text}\n\n"
        f"{_format_web_section(web_sources, web_error)}"
    )


def _format_web_section(web_sources: list[WebSource], web_error: str | None) -> str:
    if web_sources:
        sources = "\n".join(
            f"- {source.title}\uff08{source.published}\uff09：{_shorten(source.summary, 180)}"
            for source in web_sources[:3]
        )
        return "\u8054\u7f51\u68c0\u7d22\u8865\u5145\uff08arXiv\uff09\uff1a\n" + sources
    if web_error:
        return "\u8054\u7f51\u68c0\u7d22\u6682\u65f6\u4e0d\u53ef\u7528\uff0c\u5df2\u5148\u57fa\u4e8e\u672c\u5730\u77e5\u8bc6\u5e93\u56de\u7b54\u3002"
    return "\u8054\u7f51\u68c0\u7d22\u8865\u5145\uff08arXiv\uff09\uff1a\u6682\u672a\u68c0\u7d22\u5230\u9ad8\u76f8\u5173\u7ed3\u679c\u3002"


def _shorten(text: str, max_length: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 1].rstrip() + "\u2026"


def update_memory(state: ResearchState) -> ResearchState:
    state.setdefault("trace", []).append("update_memory_and_transcript")
    return state


def build_research_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("load_memory", load_memory)
    graph.add_node("handle_database_query", handle_database_query)
    graph.add_node("local_retrieve", local_retrieve)
    graph.add_node("rerank_retrieval", rerank_retrieval)
    graph.add_node("web_search", web_search)
    graph.add_node("synthesize_answer", synthesize_answer)
    graph.add_node("update_memory", update_memory)
    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "handle_database_query")
    graph.add_conditional_edges(
        "handle_database_query",
        lambda state: "direct" if state.get("direct_answer") else "rag",
        {"direct": "synthesize_answer", "rag": "local_retrieve"},
    )
    graph.add_edge("local_retrieve", "rerank_retrieval")
    graph.add_edge("rerank_retrieval", "web_search")
    graph.add_edge("web_search", "synthesize_answer")
    graph.add_edge("synthesize_answer", "update_memory")
    graph.add_edge("update_memory", END)
    return graph.compile()
