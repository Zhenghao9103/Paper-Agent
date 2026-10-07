import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import tiktoken
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import Settings, get_settings
from ..models.chat import ChatMessage, ChatSession
from ..models.context_checkpoint import ContextCheckpoint
from ..models.session_memory import SessionMemory
from ..schemas.memory import SessionMemorySnapshot, snapshot_session_memory
from .llm import ContextSummaryError, complete_context_summary

REFERENCE_MARKER_PATTERN = re.compile(
    r"\[doc:\d+\|page:\d+\|chunk:\d+\]"
)
SESSION_MEMORY_SECTION_FIELDS = {
    "## 用户目标与偏好": "goals_and_constraints",
    "## 已确认结论": "confirmed_findings",
    "## 当前研究决策": "current_decisions",
    "## 尚未解决的问题": "open_questions",
    "## 下一步": "next_actions",
}


def parse_session_memory_sections(summary: str) -> dict[str, str]:
    sections = {field: [] for field in SESSION_MEMORY_SECTION_FIELDS.values()}
    current_field: str | None = None
    seen_fields: set[str] = set()
    for line in summary.splitlines():
        stripped = line.strip()
        if stripped in SESSION_MEMORY_SECTION_FIELDS:
            current_field = SESSION_MEMORY_SECTION_FIELDS[stripped]
            seen_fields.add(current_field)
            continue
        if current_field is not None:
            sections[current_field].append(line.rstrip())

    if seen_fields != set(sections):
        raise ValueError("Context summary is missing session memory sections")
    return {
        field: "\n".join(lines).strip()
        for field, lines in sections.items()
    }


class ContextCompactionUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointOutcome:
    status: Literal["created", "not_needed", "failed"]
    checkpoint_id: int | None
    tokens_before: int
    first_kept_message_id: int | None
    error: str | None = None
    previous_session_memory: SessionMemorySnapshot | None = None
    current_session_memory: SessionMemorySnapshot | None = None

    def as_payload(self) -> dict[str, int | str | None]:
        return {
            "status": self.status,
            "checkpoint_id": self.checkpoint_id,
            "tokens_before": self.tokens_before,
            "first_kept_message_id": self.first_kept_message_id,
        }


def is_hard_overflow(
    outcome: CheckpointOutcome,
    settings: Settings,
) -> bool:
    return outcome.tokens_before >= settings.context_window_tokens


@lru_cache(maxsize=16)
def _encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_text_tokens(text: str, model: str) -> int:
    return len(_encoding_for_model(model).encode(text or ""))


def count_message_tokens(message: ChatMessage, model: str) -> int:
    serialized = f"{message.role}\n{message.content}"
    if message.citations:
        serialized += f"\n{message.citations}"
    return count_text_tokens(serialized, model)


def select_first_kept_index(
    messages: list[ChatMessage],
    *,
    keep_recent_tokens: int,
    model: str,
) -> int | None:
    accumulated = 0
    crossing_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        accumulated += count_message_tokens(messages[index], model)
        if accumulated >= keep_recent_tokens:
            crossing_index = index
            break
    if crossing_index is None:
        return None

    for index in range(crossing_index, len(messages)):
        if messages[index].role == "user" and index > 0:
            return index
    return None


def get_latest_checkpoint(
    db: Session,
    session_id: int,
) -> ContextCheckpoint | None:
    return db.scalar(
        select(ContextCheckpoint)
        .where(ContextCheckpoint.session_id == session_id)
        .order_by(
            ContextCheckpoint.created_at.desc(),
            ContextCheckpoint.id.desc(),
        )
        .limit(1)
    )


def _active_messages(
    db: Session,
    session_id: int,
    checkpoint: ContextCheckpoint | None,
) -> list[ChatMessage]:
    statement = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at, ChatMessage.id)
    )
    if checkpoint is not None:
        statement = statement.where(
            ChatMessage.id >= checkpoint.first_kept_message_id
        )
    return list(db.scalars(statement))


def load_active_context(db: Session, session_id: int) -> dict:
    session = db.get(ChatSession, session_id)
    if session is None:
        return {
            "session_memory": None,
            "session_summary": None,
            "recent_messages": [],
        }

    checkpoint = get_latest_checkpoint(db, session_id)
    session_memory = db.get(SessionMemory, session_id)
    messages = _active_messages(db, session_id, checkpoint)
    return {
        "session_memory": (
            {
                "goals_and_constraints": session_memory.goals_and_constraints,
                "confirmed_findings": session_memory.confirmed_findings,
                "current_decisions": session_memory.current_decisions,
                "open_questions": session_memory.open_questions,
                "next_actions": session_memory.next_actions,
                "source_checkpoint_id": session_memory.source_checkpoint_id,
                "version": session_memory.version,
            }
            if session_memory is not None
            else None
        ),
        "session_summary": (
            None
            if session_memory is not None
            else (
                checkpoint.summary
                if checkpoint is not None
                else session.session_summary
            )
        ),
        "recent_messages": [
            {"role": message.role, "content": message.content}
            for message in messages
        ],
    }


def _document_refs(
    messages: list[ChatMessage],
) -> list[dict[str, int | str]]:
    refs: dict[tuple[int, int, int], dict[str, int | str]] = {}
    for message in messages:
        if not message.citations:
            continue
        try:
            payload = json.loads(message.citations)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        local_citations = payload.get("local", [])
        if not isinstance(local_citations, list):
            continue
        for citation in local_citations:
            ref = _normalize_document_ref(citation)
            if ref is None:
                continue
            key = (
                int(ref["document_id"]),
                int(ref["page_number"]),
                int(ref["chunk_index"]),
            )
            refs[key] = ref
    return list(refs.values())


def _normalize_document_ref(
    value: object,
) -> dict[str, int | str] | None:
    if not isinstance(value, dict):
        return None
    try:
        return {
            "document_id": int(value["document_id"]),
            "title": str(value["title"]),
            "page_number": int(value["page_number"]),
            "chunk_index": int(value["chunk_index"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _checkpoint_document_refs(
    checkpoint: ContextCheckpoint | None,
) -> list[dict[str, int | str]]:
    if checkpoint is None:
        return []
    try:
        details = json.loads(checkpoint.details)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(details, dict):
        return []
    raw_refs = details.get("document_refs", [])
    if not isinstance(raw_refs, list):
        return []
    return [
        ref
        for value in raw_refs
        if (ref := _normalize_document_ref(value)) is not None
    ]


def _merge_document_refs(
    *groups: list[dict[str, int | str]],
) -> list[dict[str, int | str]]:
    merged: dict[tuple[int, int, int], dict[str, int | str]] = {}
    for group in groups:
        for ref in group:
            key = (
                int(ref["document_id"]),
                int(ref["page_number"]),
                int(ref["chunk_index"]),
            )
            merged[key] = ref
    return list(merged.values())


def _document_ref_marker(ref: dict[str, int | str]) -> str:
    return (
        f"[doc:{ref['document_id']}"
        f"|page:{ref['page_number']}"
        f"|chunk:{ref['chunk_index']}]"
    )


def _summary_prompt(
    previous_summary: str | None,
    messages: list[ChatMessage],
    document_refs: list[dict[str, int | str]],
    reference_markers: tuple[str, ...],
) -> str:
    conversation = json.dumps(
        [
            {
                "role": message.role,
                "content": message.content,
                "citations": message.citations,
            }
            for message in messages
        ],
        ensure_ascii=False,
        indent=2,
    )
    refs_by_marker = {
        _document_ref_marker(ref): {
            "marker": _document_ref_marker(ref),
            **ref,
        }
        for ref in document_refs
    }
    required_references = json.dumps(
        [
            refs_by_marker.get(
                marker,
                {"marker": marker, "source": "previous-summary"},
            )
            for marker in reference_markers
        ],
        ensure_ascii=False,
        indent=2,
    )
    return (
        "请把以下学术问答历史压缩成上下文检查点。\n"
        "必须严格使用以下 Markdown 标题：\n"
        "## 用户目标与偏好\n"
        "## 已确认结论\n"
        "## 当前研究决策\n"
        "## 尚未解决的问题\n"
        "## 下一步\n"
        "保留数字、模型名称、数据集名称、论文标题、文档 ID、页码和 chunk。"
        "不得把 web 来源或用户长期兴趣伪装成本地论文证据。\n\n"
        "如果存在本地文档依据，必须在“已确认结论”中原样保留至少一个"
        "<required-reference-markers> 中的 marker；不得自行编造 marker。\n\n"
        f"<required-reference-markers>\n{required_references}\n"
        "</required-reference-markers>\n\n"
        f"<previous-summary>\n{previous_summary or '无'}\n</previous-summary>\n\n"
        f"<conversation>\n{conversation}\n</conversation>"
    )


def compact_session_context(
    db: Session,
    session_id: int,
    *,
    settings: Settings | None = None,
    force: bool = False,
) -> CheckpointOutcome:
    active_settings = settings or get_settings()
    checkpoint = get_latest_checkpoint(db, session_id)
    active_messages = _active_messages(db, session_id, checkpoint)
    if checkpoint is not None:
        previous_summary = checkpoint.summary
    else:
        session = db.get(ChatSession, session_id)
        previous_summary = session.session_summary if session is not None else None

    try:
        tokens_before = count_text_tokens(
            previous_summary or "",
            active_settings.openai_chat_model,
        ) + sum(
            count_message_tokens(message, active_settings.openai_chat_model)
            for message in active_messages
        )
    except Exception as exc:
        return CheckpointOutcome(
            "failed",
            None,
            0,
            None,
            f"Token counting failed: {type(exc).__name__}",
        )
    threshold = (
        active_settings.context_window_tokens
        - active_settings.context_reserve_tokens
    )
    if not active_settings.context_compaction_enabled:
        return CheckpointOutcome("not_needed", None, tokens_before, None)
    if not force and tokens_before <= threshold:
        return CheckpointOutcome("not_needed", None, tokens_before, None)

    try:
        first_kept_index = select_first_kept_index(
            active_messages,
            keep_recent_tokens=active_settings.context_keep_recent_tokens,
            model=active_settings.openai_chat_model,
        )
    except Exception as exc:
        return CheckpointOutcome(
            "failed",
            None,
            tokens_before,
            None,
            f"Token counting failed: {type(exc).__name__}",
        )
    if first_kept_index is None:
        return CheckpointOutcome(
            "failed",
            None,
            tokens_before,
            None,
            "No complete turn is available for compaction",
        )

    messages_to_summarize = active_messages[:first_kept_index]
    first_kept = active_messages[first_kept_index]
    document_refs = _merge_document_refs(
        _checkpoint_document_refs(checkpoint),
        _document_refs(messages_to_summarize),
    )
    reference_markers = tuple(
        dict.fromkeys(
            [
                *REFERENCE_MARKER_PATTERN.findall(previous_summary or ""),
                *(
                    _document_ref_marker(ref)
                    for ref in document_refs
                ),
            ]
        )
    )
    prompt = _summary_prompt(
        previous_summary,
        messages_to_summarize,
        document_refs,
        reference_markers,
    )
    try:
        summary_input_tokens = count_text_tokens(
            prompt,
            active_settings.openai_chat_model,
        )
    except Exception as exc:
        return CheckpointOutcome(
            "failed",
            None,
            tokens_before,
            first_kept.id,
            f"Token counting failed: {type(exc).__name__}",
        )
    try:
        summary = complete_context_summary(
            prompt,
            max_tokens=active_settings.context_summary_max_tokens,
            required_reference_markers=reference_markers,
        )
    except ContextSummaryError as exc:
        return CheckpointOutcome(
            "failed",
            None,
            tokens_before,
            first_kept.id,
            str(exc),
        )
    except Exception as exc:
        return CheckpointOutcome(
            "failed",
            None,
            tokens_before,
            first_kept.id,
            f"Context summary failed: {type(exc).__name__}",
        )

    try:
        session_memory_sections = parse_session_memory_sections(summary)
    except ValueError as exc:
        return CheckpointOutcome(
            "failed",
            None,
            tokens_before,
            first_kept.id,
            str(exc),
        )

    session_memory = db.get(SessionMemory, session_id)
    previous_session_memory = snapshot_session_memory(session_memory)
    current_session_memory = snapshot_session_memory(session_memory_sections)
    details = {
        "previous_session_memory": previous_session_memory.model_dump(),
        "current_session_memory": current_session_memory.model_dump(),
        "summarized_from_message_id": messages_to_summarize[0].id,
        "summarized_through_message_id": messages_to_summarize[-1].id,
        "document_refs": document_refs,
        "kept_message_count": len(active_messages[first_kept_index:]),
        "summary_input_tokens": summary_input_tokens,
    }
    new_checkpoint = ContextCheckpoint(
        session_id=session_id,
        summary=summary,
        first_kept_message_id=first_kept.id,
        tokens_before=tokens_before,
        model=active_settings.openai_chat_model,
        version=1,
        details=json.dumps(details, ensure_ascii=False),
    )
    db.add(new_checkpoint)
    db.flush()
    if session_memory is None:
        session_memory = SessionMemory(
            session_id=session_id,
            **session_memory_sections,
            source_checkpoint_id=new_checkpoint.id,
            version=1,
        )
        db.add(session_memory)
    else:
        for field, value in session_memory_sections.items():
            setattr(session_memory, field, value)
        session_memory.source_checkpoint_id = new_checkpoint.id
        session_memory.version += 1
    db.flush()
    return CheckpointOutcome(
        "created",
        new_checkpoint.id,
        tokens_before,
        first_kept.id,
        previous_session_memory=previous_session_memory,
        current_session_memory=current_session_memory,
    )
