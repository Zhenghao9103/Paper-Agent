import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.app.db.base import Base
from backend.app.models import ChatMessage, TranscriptEvent
from backend.app.models.memory import ProceduralFeedback, ProceduralSkill
from backend.app.services import procedural_writes as writes


def test_verified_repair_requires_matching_committed_turn(db_session):
    db = db_session
    msg = ChatMessage(session_id=1, role="user", content="Explain the paper")
    db.add(msg)
    db.flush()
    payload = {
        "user_message_id": msg.id,
        "answer": "grounded answer",
        "trace_events": [
            {
                "type": "answer_verification_result",
                "status": "passed_after_repair",
                "passed": True,
                "citation_identity_valid": True,
            }
        ],
    }
    trace = TranscriptEvent(session_id=1, event_type="qa", payload=json.dumps(payload))
    db.add(trace)
    db.flush()
    writes.enqueue(db, msg.id, trace_id=trace.id)
    db.commit()
    proposal = {
        "condition": "citation repair",
        "steps": ["align claim with citation"],
        "constraints": [],
    }

    def reflect(source):
        assert source["verified_repair"]["trace_id"] == trace.id
        return proposal

    assert (
        writes.process_event(
            db, msg.id, reflector=reflect, judge=lambda *_: {"action": "add"}
        )
        == "applied"
    )
    other = ChatMessage(session_id=2, role="user", content="Hello")
    db.add(other)
    db.flush()
    writes.enqueue(db, other.id, trace_id=trace.id)
    db.commit()
    assert writes.process_event(db, other.id, reflector=reflect) == "rejected"


def test_concurrent_refinements_are_rejudged(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "race.db"))
    Base.metadata.create_all(engine)
    base = {"condition": "teach", "steps": ["example first"], "constraints": []}

    def new(db):
        msg = ChatMessage(session_id=1, role="user", content="From now on use examples")
        db.add(msg)
        db.commit()
        return msg.id

    with Session(engine) as db:
        writes.process_event(
            db, new(db), reflector=lambda _: base, judge=lambda *_: {"action": "add"}
        )
        events = [new(db), new(db)]
    gate = Barrier(2)

    def apply(event):
        observed = []

        def judge(source, proposal, skills):
            observed.append(skills[0]["version"])
            if len(observed) == 1:
                gate.wait(timeout=10)
            return {"action": "update", "target_id": skills[0]["id"]}

        with Session(engine) as db:
            return writes.process_event(
                db, event, reflector=lambda _: base, judge=judge
            ), observed

    with ThreadPoolExecutor(max_workers=2) as threads:
        results = list(threads.map(apply, events))
    assert sorted(v for _, versions in results for v in versions) == [1, 1, 2]
    assert all(status == "applied" for status, _ in results)
    with Session(engine) as db:
        assert db.scalars(select(ProceduralSkill)).one().version == 3
        assert {f.version for f in db.scalars(select(ProceduralFeedback))} == {1, 2, 3}
    engine.dispose()
