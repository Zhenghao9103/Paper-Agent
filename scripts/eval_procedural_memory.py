"""Paired ordinary-dialogue experiment. Run from the repo root as a Python module."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from statistics import mean
from time import perf_counter
from unittest.mock import patch

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.db.base import Base
from backend.app.models.chat import ChatMessage, ChatSession
from backend.app.models.memory import ProceduralFeedback, ProceduralSkill
from backend.app.rag.vector_store import BGE_M3_MODEL_NAME
from backend.app.services import research
from backend.app.services.intent_router import RoutingOutcome
from backend.app.services.procedural_writes import process_event, process_sync


def mark(answer, case):
    normalized = answer.casefold()
    first, second = (normalized.find(label.casefold()) for label in case["markers"])
    return 0 <= first < second and any(term in normalized for term in case["topic"])


def run(cases, db, output):
    settings = get_settings()
    result = {
        "config": {
            "agent": settings.resolved_agent_model,
            "embedding": BGE_M3_MODEL_NAME,
            "cluster_threshold": settings.procedural_cluster_threshold,
            "recall_threshold": settings.procedural_recall_threshold,
            "context_tokens": settings.procedural_memory_context_tokens,
            "model_description": os.environ.get(
                "EVAL_MODEL_DESCRIPTION", settings.resolved_agent_model
            ),
            "seed": os.environ.get("EVAL_SERVER_SEED", "provider default"),
            "routing": "fixed direct, empty semantic history",
            "scoring": "both requested headings in order and at least one topic term",
        },
        "learning": [],
        "snapshot": [],
        "evaluation": [],
        "summary": {},
    }

    def save():
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
        )

    expected = {}
    for case in cases["learning"]:
        session = ChatSession()
        db.add(session)
        db.flush()
        db.add(
            ChatMessage(
                role="assistant", session_id=session.id, content=case["previous_answer"]
            )
        )
        db.flush()
        feedback = ChatMessage(
            role="user", session_id=session.id, content=case["feedback"]
        )
        db.add(feedback)
        db.commit()
        started = perf_counter()
        status = process_event(db, feedback.id)
        entry = db.get(ProceduralFeedback, feedback.id)
        skill_id = entry.skill_id if entry else None
        expected[case["key"]] = (
            {"id": skill_id, "version": entry.version} if skill_id else None
        )
        result["learning"].append(
            {
                "key": case["key"],
                "status": status,
                "skill_id": skill_id,
                "action": entry.action if entry else None,
                "reason": entry.reason if entry else None,
                "latency_ms": round((perf_counter() - started) * 1000, 2),
            }
        )
        save()
    for skill in db.scalars(
        select(ProceduralSkill).order_by(ProceduralSkill.source_event_id)
    ):
        result["snapshot"].append(
            {
                "id": skill.id,
                "version": skill.version,
                "condition": skill.condition,
                "steps": skill.steps,
                "constraints": skill.constraints,
                "active": skill.active,
            }
        )
    started = perf_counter()
    process_sync(db)
    result["index_build_ms"] = round((perf_counter() - started) * 1000, 2)
    save()
    # Freeze the training history. V2 and write-only never read the frozen skills.
    for mode in ("off", "write", "flat", "rsi"):
        for case in cases["evaluation"]:
            session = ChatSession()
            db.add(session)
            db.commit()
            started = perf_counter()
            with (
                patch.object(settings, "procedural_memory_mode", mode),
                patch.object(
                    research,
                    "route_question",
                    return_value=RoutingOutcome(intent="direct"),
                ),
                patch.object(research, "safe_query_relevant_memories", return_value=[]),
            ):
                answer = research.run_research(
                    db, session_id=session.id, question=case["question"]
                )
            event = next(
                (e for e in answer.trace_events if e["type"] == "procedural_memory"), {}
            )
            used = event.get("used_skills", [])
            row = {
                "mode": mode,
                "id": case["id"],
                "question": case["question"],
                "answer": answer.answer,
                "expected": expected[case["key"]],
                "used": used,
                "hit": expected[case["key"]] in used,
                "success": mark(answer.answer, case),
                "tokens": event.get("token_count", 0),
                "degraded": event.get("degraded", False),
                "query_ms": event.get("timing_ms", 0),
                "elapsed_ms": round((perf_counter() - started) * 1000, 2),
            }
            result["evaluation"].append(row)
            save()
        rows = [r for r in result["evaluation"] if r["mode"] == mode]
        result["summary"][mode] = {
            k: round(mean(r[k] for r in rows), 3)
            for k in ("hit", "success", "tokens", "degraded", "query_ms", "elapsed_ms")
        }
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=Path("eval/memory/general_dialogue.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        engine = create_engine("sqlite:///" + str(Path(scratch) / "experiment.db"))
        Base.metadata.create_all(engine)
        try:
            with Session(engine) as db:
                run(json.loads(args.cases.read_text(encoding="utf-8")), db, args.output)
        finally:
            engine.dispose()
