"""Local write -> new ordinary chat -> prompt -> revocation, with offline providers."""

from sqlalchemy import select

from backend.app.core.config import get_settings
from backend.app.models.memory import ProceduralSkill
from backend.app.services import (
    answering,
    chat,
    procedural_retrieval,
    procedural_writes,
    research,
)
from backend.app.services.intent_router import RoutingOutcome


def test_feedback_across_real_chat_sessions(db_session, monkeypatch):
    monkeypatch.setattr(get_settings(), "procedural_memory_mode", "rsi")
    monkeypatch.setattr(
        research, "route_question", lambda *_: RoutingOutcome(intent="direct")
    )
    monkeypatch.setattr(research, "safe_query_relevant_memories", lambda *_a, **_k: [])
    monkeypatch.setattr(chat, "write_memories", lambda *_a, **_k: [])
    monkeypatch.setattr(
        procedural_retrieval, "embed_text", lambda *_a, **_k: [1.0, 0.0]
    )
    advice = {
        "condition": "math explanations",
        "steps": ["start with an example"],
        "constraints": [],
    }
    monkeypatch.setattr(procedural_writes, "reflect", lambda _: advice)

    def judge(source, proposal, skills):
        if "stop" in source["content"]:
            return {"action": "deactivate", "target_id": skills[0]["id"]}
        return {"action": "add"}

    monkeypatch.setattr(procedural_writes, "judge_advice", judge)

    def answer(messages):
        if any("start with an example" in v["content"] for v in messages[:-1]):
            return "Example: 2 and 4. Definition: the arithmetic mean is their sum divided by two."
        return "A definition."

    monkeypatch.setattr(answering, "complex_text_completion", answer)

    def ask(question):
        return chat.answer_question(
            db_session, question=question, document_id=None, session_id=None
        )

    first = ask("From now on give an example before math definitions.")
    second = ask("Explain the arithmetic mean.")
    assert first[0] != second[0] and second[1].startswith("Example:")
    skill = db_session.scalars(select(ProceduralSkill)).one()
    assert next(e for e in second[-1] if e["type"] == "procedural_memory")[
        "used_skills"
    ] == [{"id": skill.id, "version": 1}]
    ask("From now on stop using that strategy.")
    assert ask("Explain the median.")[1] == "A definition."
