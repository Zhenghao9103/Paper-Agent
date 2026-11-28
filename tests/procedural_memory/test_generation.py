"""Ordinary generation consumes only complete, relevant procedural guidance."""

import json
from types import SimpleNamespace

import pytest

from backend.app.services import answering, research
from backend.app.services.intent_router import RoutingOutcome


def skill(skill_id, steps):
    return SimpleNamespace(
        id=skill_id,
        version=2,
        condition="Teaching concepts",
        steps=steps,
        constraints=["Respect the requested format"],
        source_event_id=123,
        feedback="private feedback",
        source_session_id=30,
    )


@pytest.fixture()
def direct_model(monkeypatch):
    requests = []
    monkeypatch.setattr(
        research, "route_question", lambda *args: RoutingOutcome(intent="direct")
    )
    monkeypatch.setattr(
        research, "safe_query_relevant_memories", lambda *args, **kwargs: []
    )

    def complete(messages):
        requests.append(messages)
        if "formula only" in messages[-1]["content"]:
            return "F=ma"
        if any("Start with an example" in item["content"] for item in messages):
            return "Example first. Definition second."
        return "Definition only."

    monkeypatch.setattr(answering, "complex_text_completion", complete)
    return requests


def configure(monkeypatch, mode, skills, budget=220, degraded=False):
    monkeypatch.setattr(
        research,
        "get_settings",
        lambda: SimpleNamespace(
            procedural_memory_mode=mode,
            procedural_memory_context_tokens=budget,
            resolved_agent_model="gpt-4o",
        ),
    )
    monkeypatch.setattr(
        research,
        "retrieve_skills",
        lambda db, question, mode: (skills, degraded),
        raising=False,
    )


@pytest.mark.parametrize("mode", ["flat", "rsi"])
def test_normal_generation_tracks_only_complete_skills_sent_to_model(
    db_session,
    direct_model,
    monkeypatch,
    mode,
):
    configure(
        monkeypatch,
        mode,
        [
            skill("oversized", ["Unabridged " * 500]),
            skill(
                "teaching",
                ["Start with an example", "Define the term", "token=private-key"],
            ),
        ],
        degraded=(mode == "rsi"),
    )
    result = research.run_research(
        db_session, session_id=42, question="Explain acceleration"
    )
    assert result.answer == "Example first. Definition second."
    messages = direct_model[0]
    section = messages[-2]["content"]
    assert messages[-2]["role"] == "user"
    assert "current user" in messages[0]["content"].lower()
    assert (
        "tool" in messages[0]["content"].lower()
        and "evidence" in messages[0]["content"].lower()
    )
    assert "Explain acceleration" in messages[-1]["content"]
    assert all(
        text not in section
        for text in ("Unabridged", "private", "teaching", "source_event_id")
    )
    assert "Define the term" in section
    event = next(e for e in result.trace_events if e["type"] == "procedural_memory")
    assert [(s["id"], s["version"]) for s in event["used_skills"]] == [("teaching", 2)]
    assert 0 < event["token_count"] <= 220
    assert "Start with" not in json.dumps(event)
    assert event["degraded"] == (mode == "rsi")
    assert result.citations == []
    conflict = research.run_research(
        db_session, session_id=43, question="Give the formula only"
    )
    assert conflict.answer == "F=ma"


@pytest.mark.parametrize("mode,budget", [("off", 220), ("write", 220), ("flat", 1)])
def test_disabled_or_unaffordable_guidance_preserves_ordinary_request(
    db_session,
    direct_model,
    monkeypatch,
    mode,
    budget,
):
    configure(monkeypatch, mode, [skill("teaching", ["Start with an example"])], budget)
    if mode in {"off", "write"}:

        def unexpected_read(*args):
            pytest.fail("read attempted with procedural retrieval disabled")

        monkeypatch.setattr(research, "retrieve_skills", unexpected_read, raising=False)
    result = research.run_research(db_session, question="Explain acceleration")
    assert result.answer == "Definition only."
    assert len(direct_model[0]) == 2
    events = [e for e in result.trace_events if e["type"] == "procedural_memory"]
    if mode == "flat":
        assert (
            len(events) == 1
            and events[0]["used_skills"] == []
            and events[0]["token_count"] == 0
        )
    else:
        assert not events
