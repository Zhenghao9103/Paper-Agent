from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import httpx

from eval.router.router_m0.contract import (
    RouterProtocolError,
    parse_router_output,
)
from eval.router.router_perf.contract import (
    PerfFixture,
    PerfMeasurement,
    build_perf_messages,
)


class StreamProtocolError(ValueError):
    """Raised when a llama.cpp SSE event violates the expected contract."""


@dataclass(frozen=True)
class TimingBreakdown:
    prompt_tokens: int | None
    prefill_tokens: int | None
    prefill_ms: float | None
    prefill_tps: float | None
    decode_tokens: int | None
    decode_ms: float | None
    decode_tps: float | None
    cache_reused_tokens_estimate: int | None
    cache_hit_ratio_estimate: float | None


@dataclass(frozen=True)
class StreamCompletion:
    raw_output: str
    ttft_ms: float | None
    total_ms: float
    timings: TimingBreakdown
    terminal_payload: dict[str, object]


def _number(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _tokens_per_second(tokens: int | None, elapsed_ms: float | None) -> float | None:
    if tokens is None or elapsed_ms is None or elapsed_ms <= 0:
        return None
    return tokens * 1000 / elapsed_ms


def estimate_cache_reuse(
    prompt_tokens: int | None, prefill_tokens: int | None
) -> tuple[int | None, float | None]:
    if prompt_tokens is None or prefill_tokens is None:
        return None, None
    reused = max(0, prompt_tokens - prefill_tokens)
    ratio = reused / prompt_tokens if prompt_tokens > 0 else None
    return reused, ratio


def parse_timing_payload(payload: dict[str, object]) -> TimingBreakdown:
    raw_usage = payload.get("usage")
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    raw_timings = payload.get("timings")
    timings = raw_timings if isinstance(raw_timings, dict) else {}

    prompt_tokens = _integer(usage, "prompt_tokens")
    prefill_tokens = _integer(timings, "prompt_n")
    prefill_ms = _number(timings, "prompt_ms")
    decode_tokens = _integer(timings, "predicted_n")
    if decode_tokens is None:
        decode_tokens = _integer(usage, "completion_tokens")
    decode_ms = _number(timings, "predicted_ms")
    reused, hit_ratio = estimate_cache_reuse(prompt_tokens, prefill_tokens)
    return TimingBreakdown(
        prompt_tokens=prompt_tokens,
        prefill_tokens=prefill_tokens,
        prefill_ms=prefill_ms,
        prefill_tps=_tokens_per_second(prefill_tokens, prefill_ms),
        decode_tokens=decode_tokens,
        decode_ms=decode_ms,
        decode_tps=_tokens_per_second(decode_tokens, decode_ms),
        cache_reused_tokens_estimate=reused,
        cache_hit_ratio_estimate=hit_ratio,
    )


def consume_sse_events(
    lines: Iterable[str],
    *,
    started: float,
    clock: Callable[[], float] = time.perf_counter,
) -> StreamCompletion:
    fragments: list[str] = []
    ttft_ms: float | None = None
    last_elapsed_ms = 0.0
    terminal_payload: dict[str, object] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        last_elapsed_ms = (clock() - started) * 1000
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise StreamProtocolError("invalid SSE JSON") from exc
        if not isinstance(payload, dict):
            raise StreamProtocolError("SSE payload must be an object")
        terminal_payload = payload
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        first = choices[0]
        if not isinstance(first, dict):
            continue
        delta = first.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if not isinstance(content, str) or not content:
            continue
        if ttft_ms is None:
            ttft_ms = last_elapsed_ms
        fragments.append(content)

    return StreamCompletion(
        raw_output="".join(fragments),
        ttft_ms=ttft_ms,
        total_ms=last_elapsed_ms,
        timings=parse_timing_payload(terminal_payload),
        terminal_payload=terminal_payload,
    )


class StreamingRouterClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        client: Any | None = None,
        clock: Callable[[], float] = time.perf_counter,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._clock = clock
        self._client = client or httpx.Client(
            timeout=timeout_s,
            trust_env=False,
            headers={"Authorization": "Bearer local-no-key"},
        )

    def generate(
        self,
        fixture: PerfFixture,
        *,
        scenario: str,
        repeat: int,
        rss_peak_mb: float | None,
    ) -> PerfMeasurement:
        request = {
            "model": self.model,
            "messages": build_perf_messages(fixture),
            "temperature": 0,
            "max_tokens": 64,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = self._clock()
        with self._client.stream(
            "POST", f"{self._base_url}/chat/completions", json=request
        ) as response:
            response.raise_for_status()
            completion = consume_sse_events(
                response.iter_lines(), started=started, clock=self._clock
            )
        try:
            predicted_intent = parse_router_output(completion.raw_output).intent
            protocol_pass = True
        except RouterProtocolError:
            predicted_intent = None
            protocol_pass = False
        timing = completion.timings
        return PerfMeasurement(
            fixture_id=fixture.id,
            scenario=scenario,
            repeat=repeat,
            raw_output=completion.raw_output,
            predicted_intent=predicted_intent,
            protocol_pass=protocol_pass,
            intent_correct=predicted_intent == fixture.expected_intent,
            ttft_ms=completion.ttft_ms,
            total_ms=completion.total_ms,
            prompt_tokens=timing.prompt_tokens,
            prefill_tokens=timing.prefill_tokens,
            prefill_ms=timing.prefill_ms,
            prefill_tps=timing.prefill_tps,
            decode_tokens=timing.decode_tokens,
            decode_ms=timing.decode_ms,
            decode_tps=timing.decode_tps,
            cache_reused_tokens_estimate=timing.cache_reused_tokens_estimate,
            cache_hit_ratio_estimate=timing.cache_hit_ratio_estimate,
            rss_peak_mb=rss_peak_mb,
        )
