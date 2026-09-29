from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from eval.router.router_m0.contract import Intent, RouterInput
from eval.router.router_m0.prompt import ROUTER_M0_SYSTEM_PROMPT

FixtureLength = Literal["short", "medium", "long"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _FixtureRecord(_StrictModel):
    id: str = Field(pattern=r"^perf-[a-z0-9-]+$")
    length: FixtureLength
    expected_intent: Intent
    input: RouterInput


@dataclass(frozen=True)
class PerfFixture:
    id: str
    length: FixtureLength
    expected_intent: Intent
    runtime_input: RouterInput

    @classmethod
    def from_record(cls, record: dict[str, object]) -> PerfFixture:
        serialized_input = json.dumps(record.get("input", {}), ensure_ascii=False)
        if "expected_intent" in serialized_input:
            raise ValueError("expected intent must stay outside prompt")
        try:
            parsed = _FixtureRecord.model_validate(record)
        except ValidationError as exc:
            raise ValueError(f"invalid performance fixture: {exc}") from exc
        return cls(
            id=parsed.id,
            length=parsed.length,
            expected_intent=parsed.expected_intent,
            runtime_input=parsed.input,
        )


_CACHE_TYPES = {
    "f32",
    "f16",
    "bf16",
    "q8_0",
    "q4_0",
    "q4_1",
    "iq4_nl",
    "q5_0",
    "q5_1",
}


@dataclass(frozen=True)
class ServerConfig:
    threads: int
    threads_batch: int
    cache_prompt: bool
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    ctx_size: int = 4096
    parallel: int = 1

    def __post_init__(self) -> None:
        if self.threads < 1 or self.threads_batch < 1:
            raise ValueError("threads must be positive")
        if self.cache_type_k not in _CACHE_TYPES or self.cache_type_v not in _CACHE_TYPES:
            raise ValueError("unsupported cache type")
        if self.ctx_size < 1 or self.parallel != 1:
            raise ValueError("ctx_size must be positive and parallel must equal 1")


@dataclass(frozen=True)
class PerfMeasurement:
    fixture_id: str
    scenario: str
    repeat: int
    raw_output: str
    predicted_intent: str | None
    protocol_pass: bool
    intent_correct: bool
    ttft_ms: float | None
    total_ms: float
    prompt_tokens: int | None
    prefill_tokens: int | None
    prefill_ms: float | None
    prefill_tps: float | None
    decode_tokens: int | None
    decode_ms: float | None
    decode_tps: float | None
    cache_reused_tokens_estimate: int | None
    cache_hit_ratio_estimate: float | None
    rss_peak_mb: float | None
    error_code: str | None = None


def load_perf_fixtures(path: Path) -> list[PerfFixture]:
    fixtures: list[PerfFixture] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            fixture = PerfFixture.from_record(json.loads(line))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid performance fixture at line {line_number}: {exc}") from exc
        if fixture.id in seen:
            raise ValueError(f"duplicate performance fixture id: {fixture.id}")
        seen.add(fixture.id)
        fixtures.append(fixture)
    if not fixtures:
        raise ValueError("performance fixture dataset is empty")
    return fixtures


def _runtime_input_bytes(fixture: PerfFixture) -> bytes:
    normalized = json.dumps(
        fixture.runtime_input.model_dump(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return normalized.encode("utf-8")


def fixture_input_hashes(fixtures: list[PerfFixture]) -> set[str]:
    return {
        hashlib.sha256(_runtime_input_bytes(fixture)).hexdigest()
        for fixture in fixtures
    }


def build_perf_messages(fixture: PerfFixture) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ROUTER_M0_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                fixture.runtime_input.model_dump(), ensure_ascii=False
            ),
        },
    ]
