from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

Intent = Literal["direct", "simple_rag", "agentic_rag"]
Language = Literal["zh", "en"]
Difficulty = Literal["easy", "medium", "hard"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RouterMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=900)


class RouterInput(StrictModel):
    session_context: list[RouterMessage] = Field(max_length=4)
    current_question: str = Field(min_length=1, max_length=1000)


class RouterExpected(StrictModel):
    intent: Intent


class RouterMetadata(StrictModel):
    language: Language
    difficulty: Difficulty
    scenario: str = Field(min_length=1)
    pair_id: str | None = None
    rationale: str = Field(min_length=1)


class RouterCase(StrictModel):
    id: str = Field(pattern=r"^router-[a-z0-9-]+$")
    input: RouterInput
    expected: RouterExpected
    metadata: RouterMetadata


class RouterOutput(StrictModel):
    intent: Intent


class RouterProtocolError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def parse_router_output(raw: str) -> RouterOutput:
    if not raw.strip():
        raise RouterProtocolError("empty_output")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RouterProtocolError("invalid_json") from exc
    if not isinstance(payload, dict):
        raise RouterProtocolError("not_json_object")
    if "intent" not in payload:
        raise RouterProtocolError("missing_intent")
    if set(payload) != {"intent"}:
        raise RouterProtocolError("extra_fields")
    try:
        return RouterOutput.model_validate(payload)
    except ValidationError as exc:
        raise RouterProtocolError("invalid_intent") from exc


def load_router_cases(path: Path) -> list[RouterCase]:
    cases: list[RouterCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = RouterCase.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(f"invalid Router case at line {line_number}: {exc}") from exc
        if case.id in seen:
            raise ValueError(f"duplicate Router case id: {case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError("Router dataset is empty")
    return cases
