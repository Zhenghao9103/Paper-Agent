import pytest
from sqlalchemy import delete, select, text

from backend.app.services import procedural_retrieval as retrieval
from backend.app.models.memory import ProceduralSkill
from backend.app.models.procedural_index import SkillCluster, SkillMember


@pytest.fixture(autouse=True)
def vectors(monkeypatch):
    values = {
        "teach": [1.0, 0.0, 0.0, 0.0],
        "example": [0.8, 0.6, 0.0, 0.0],
        "compare": [0.0, 1.0, 0.0, 0.0],
        "debug": [0.0, 0.0, 1.0, 0.0],
        "schedule": [0.0, 0.0, 0.0, 1.0],
        "all": [0.5, 0.5, 0.5, 0.5],
        "irrelevant": [-1.0, 0.0, 0.0, 0.0],
    }
    monkeypatch.setattr(retrieval, "embed_text", lambda text, input_type: values[text])


def add(db, key, condition):
    skill = ProceduralSkill(
        id=key,
        version=1,
        active=True,
        condition=condition,
        steps=["do it"],
        constraints=[],
        source_event_id=1,
        source_session_id=1,
    )
    db.add(skill)
    db.flush()
    retrieval.sync_skill(db, key)
    return skill


def ids(db, query, mode="rsi"):
    results, degraded = retrieval.retrieve_skills(db, query, mode)
    assert not degraded
    return [skill.id for skill in results]


def test_top_three_centers_then_top_four_unique_skills(db_session):
    db = db_session
    assert ids(db, "teach") == []
    for key in "edcba":
        add(db, key, "teach")
    for key in ["compare", "debug", "schedule"]:
        add(db, key, key)
    assert ids(db, "teach") == list("abcd")
    assert ids(db, "irrelevant") == []
    # Four orthogonal, equally relevant centers: top three exclude the fourth.
    for key in "bcde":
        db.get(ProceduralSkill, key).active = False
        db.flush()
        retrieval.sync_skill(db, key)
    assert ids(db, "all") == ["a", "compare", "debug"]
    assert ids(db, "all", "flat") == ["a", "compare", "debug", "schedule"]


def test_online_moves_and_retracts_only_live_versions(db_session):
    db = db_session
    first = add(db, "a", "teach")
    second = add(db, "b", "example")
    cluster = db.scalars(select(SkillCluster)).one()
    assert cluster.size == 2
    assert cluster.vector_sum == pytest.approx([1.8, 0.6, 0, 0])
    second.condition, second.version = "debug", 2
    db.flush()
    assert ids(db, "teach") == ["a"]
    retrieval.sync_skill(db, "b")
    assert cluster.size == 1
    assert cluster.vector_sum == pytest.approx([1, 0, 0, 0])
    assert ids(db, "debug") == ["b"]
    second.active = False
    db.flush()
    assert ids(db, "debug") == []
    retrieval.sync_skill(db, "b")
    db.delete(first)
    db.flush()
    assert ids(db, "teach") == []
    retrieval.sync_skill(db, "a")
    assert db.scalars(select(SkillCluster)).all() == []


def test_rebuild_and_flat_fallback_on_index_failure(db_session, monkeypatch):
    db = db_session
    add(db, "b", "example")
    add(db, "a", "teach")
    db.commit()
    db.execute(delete(SkillMember))
    db.execute(delete(SkillCluster))
    db.commit()
    with monkeypatch.context() as patch:

        def broken(session):
            raise RuntimeError("index unavailable")

        patch.setattr(retrieval, "rebuild_index", broken)
        results, degraded = retrieval.retrieve_skills(db, "teach", "rsi")
        assert degraded and [row.id for row in results] == ["a", "b"]
    assert ids(db, "teach") == ["a", "b"]
    assert db.scalars(select(SkillCluster)).one().size == 2
    retrieval.rebuild_index(db)
    assert db.scalars(select(SkillCluster)).one().size == 2
    # The flat recovery path can run using only the source relation.
    db.execute(text("DROP TABLE procedural_skill_members"))
    results, degraded = retrieval.retrieve_skills(db, "teach", "rsi")
    assert degraded and [row.id for row in results] == ["a", "b"]
