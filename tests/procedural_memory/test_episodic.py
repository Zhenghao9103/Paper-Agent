import json

import pytest
from sqlalchemy import select

from backend.app.models.chat import ChatMessage, TranscriptEvent
from backend.app.models.memory import Memory, MemorySyncOperation, ProceduralFeedback, ProceduralSkill
from backend.app.services import answering, chat, memory_extractor, research
from backend.app.services.intent_router import RoutingOutcome


def test_interest_is_used_in_a_new_general_chat(db_session, semantic_index, monkeypatch):
    db = db_session
    prompts = []
    monkeypatch.setattr(research, "route_question", lambda *_: RoutingOutcome(intent="direct"))
    monkeypatch.setattr(memory_extractor, "agent_json", lambda *_a, **_k: {"candidates": [{
        "memory_type": "research_interest", "content": "用户主要研究多视图聚类", "confidence": 0.98,
    }]})

    def complete(messages):
        prompts.append(messages)
        return "好的"

    monkeypatch.setattr(answering, "complex_text_completion", complete)

    def ask(question):
        return chat.answer_question(db, question=question, document_id=None, session_id=None)

    ask("你好")
    assert not db.scalars(select(Memory)).all()
    assert not db.scalars(select(ProceduralFeedback)).all()
    assert not db.scalars(select(TranscriptEvent).where(
        TranscriptEvent.event_type == "memory_source")).all()
    first = ask("记住，我主要研究多视图聚类")
    row, = json.loads(db.scalars(select(MemorySyncOperation)).one().payload)
    source = db.get(TranscriptEvent, row["source_transcript_id"])
    message = db.get(ChatMessage, json.loads(source.payload)["user_message_id"])
    assert message.session_id == source.session_id == first[0]
    assert message.role == "user" and message.content == "记住，我主要研究多视图聚类"
    second = ask("按照我的研究兴趣，给我一个阅读建议")
    assert first[0] != second[0]
    assert any(hit.content == "用户主要研究多视图聚类" for hit in second[5])
    assert "用户主要研究多视图聚类" in json.dumps(prompts[-1], ensure_ascii=False)
    assert not db.scalars(select(ProceduralSkill)).all()


def test_accepted_source_survives_interrupted_answer(db_session, semantic_index, monkeypatch):
    question = "记住，我主要研究多视图聚类，也偏好中文回答"
    monkeypatch.setattr(memory_extractor, "agent_json", lambda *_a, **_kw: {"candidates": [
        {"memory_type": "research_interest", "content": "研究多视图聚类", "confidence": 0.98},
        {"memory_type": "user_preference", "content": "偏好中文回答", "confidence": 0.98},
    ]})

    def interrupt(*_args, **_kwargs):
        raise RuntimeError("generation interrupted")

    monkeypatch.setattr(chat, "run_research", interrupt)
    with pytest.raises(RuntimeError, match="generation interrupted"):
        chat.answer_question(db_session, question=question, document_id=None, session_id=None)

    db_session.rollback()
    source = db_session.scalars(select(TranscriptEvent).where(
        TranscriptEvent.event_type == "memory_source")).one()
    assert json.loads(source.payload) == {"content": question}
    assert not db_session.scalars(select(ChatMessage)).all()
    operations = db_session.scalars(select(MemorySyncOperation)).all()
    assert len(operations) == 2
    assert all(json.loads(op.payload)[0]["source_transcript_id"] == source.id for op in operations)


def test_checkpoint_memory_reuses_existing_source(db_session, semantic_index, monkeypatch):
    from backend.app.models.context_checkpoint import ContextCheckpoint
    from backend.app.services import memory_harness

    checkpoint = ContextCheckpoint(session_id=1, summary="研究多视图聚类", first_kept_message_id=1,
                                   tokens_before=120, model="test", version=1, details="{}")
    db_session.add(checkpoint)
    db_session.commit()
    monkeypatch.setattr(memory_extractor, "agent_json", lambda *_a, **_kw: {"candidates": [{
        "memory_type": "research_interest", "content": "研究多视图聚类", "confidence": 0.98,
    }]})
    memory_harness.write_memories(db_session, source_session_id=checkpoint.session_id,
                                  source_checkpoint_id=checkpoint.id,
                                  current_session_memory={"goals_and_constraints": checkpoint.summary})
    operation = db_session.scalars(select(MemorySyncOperation)).one()
    assert operation.source_checkpoint_id == checkpoint.id and operation.status == "done"
    assert not db_session.scalars(select(TranscriptEvent).where(
        TranscriptEvent.event_type == "memory_source")).all()
