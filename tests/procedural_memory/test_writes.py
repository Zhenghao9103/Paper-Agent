import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.app.db.base import Base
from backend.app.models import ChatMessage


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def message(db, content, role="user"):
    row = ChatMessage(session_id=1, role=role, content=content)
    db.add(row)
    db.commit()
    return row.id


def test_feedback_is_idempotent_versioned_and_recoverable(db):
    from backend.app.services import procedural_writes as writes
    from backend.app.models.memory import (
        ProceduralFeedback,
        ProceduralSkill,
        ProceduralSync,
    )

    proposal = dict(
        condition="explain concepts", steps=["give an example"], constraints=[]
    )

    def reflector(source):
        return proposal

    def judge(source, proposed, skills):
        return dict(proposed, action="add")

    event = message(db, "From now on, explain concepts with examples first.")
    assert (
        writes.process_event(db, event, reflector=reflector, judge=judge) == "applied"
    )
    assert (
        writes.process_event(db, event, reflector=pytest.fail, judge=pytest.fail)
        == "applied"
    )
    skill = db.scalars(select(ProceduralSkill)).one()
    assert (skill.version, skill.source_event_id, skill.steps) == (
        1,
        event,
        proposal["steps"],
    )
    for action in ("update", "deactivate"):
        correction = message(db, "以后讲解时请调整举例顺序")

        def decide(source, proposed, skills):
            return dict(proposed, action=action, target_id=skill.id)

        assert (
            writes.process_event(db, correction, reflector=reflector, judge=decide)
            == "applied"
        )
    db.refresh(skill)
    assert (skill.version, skill.active) == (3, False)
    assert len(db.scalars(select(ProceduralFeedback)).all()) == 3
    assert len(db.scalars(select(ProceduralSync)).all()) == 3

    def outage(db, skill_id):
        raise RuntimeError("unavailable")

    writes.process_sync(db, sync=outage)
    assert all(
        op.status == "pending" and op.attempts == 1
        for op in db.scalars(select(ProceduralSync))
    )
    writes.process_sync(db, sync=lambda db, skill_id: None)
    assert all(op.status == "done" for op in db.scalars(select(ProceduralSync)))


def test_trigger_and_validation(db):
    from backend.app.services import procedural_writes as writes
    from backend.app.models.memory import ProceduralSkill

    for text, role in [
        ("hello", "user"),
        ("I solved it.", "assistant"),
        ("仅这一次，请先举例", "user"),
        ("Water is wet.", "user"),
    ]:
        assert (
            writes.process_event(db, message(db, text, role), reflector=pytest.fail)
            == "skipped"
        )
    assert writes.process_event(db, 999, reflector=pytest.fail) == "rejected"
    event = message(db, "今后请先举例")
    bad = dict(condition="math", steps=[], constraints=[])
    assert (
        writes.process_event(
            db,
            event,
            reflector=lambda source: bad,
            judge=lambda *args: dict(bad, action="add"),
        )
        == "rejected"
    )
    assert db.scalars(select(ProceduralSkill)).first() is None


def test_judge_cannot_overwrite_unrelated_strategy(db, monkeypatch):
    from backend.app.services import procedural_writes as writes
    from backend.app.models.memory import ProceduralSkill
    from backend.app.rag import vector_store

    topics = {"cook pasta": [1.0, 0.0], "debug software": [0.0, 1.0]}
    monkeypatch.setattr(vector_store, "embed_text", lambda text: topics[text])

    def reflect(source):
        condition = "cook pasta" if "food" in source["content"] else "debug software"
        return {
            "condition": condition,
            "steps": ["list prerequisites"],
            "constraints": [],
        }

    assert (
        writes.process_event(
            db,
            message(db, "From now on list food ingredients"),
            reflector=reflect,
            judge=lambda *_: {"action": "add"},
        )
        == "applied"
    )
    skill = db.scalars(select(ProceduralSkill)).one()

    def decision(*_):
        return {"action": "update", "target_id": skill.id}

    assert (
        writes.process_event(
            db,
            message(db, "From now on list software checks"),
            reflector=reflect,
            judge=decision,
        )
        == "rejected"
    )
    db.refresh(skill)
    assert (skill.condition, skill.version) == ("cook pasta", 1)


@pytest.mark.parametrize(
    "correction",
    [
        "这样讲我没看懂，请先举例，再解释公式。",
        "That explanation was confusing. Please show a worked example first.",
    ],
)
def test_correction_without_permanent_preference_keyword(db, correction):
    from backend.app.services import procedural_writes as writes

    message(db, "Here is a formula.", role="assistant")
    event = message(db, correction)

    def reflect(source):
        assert source["previous_answer"] == "Here is a formula."
        return {
            "condition": "explain mathematics",
            "steps": ["give an example"],
            "constraints": [],
        }

    assert (
        writes.process_event(
            db, event, reflector=reflect, judge=lambda *_: {"action": "add"}
        )
        == "applied"
    )
