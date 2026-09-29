from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


@dataclass(frozen=True)
class RouterGeneration:
    raw_output: str
    total_ms: float
    first_token_ms: float | None = None
    output_tokens: int | None = None
    tokens_per_s: float | None = None
    error_code: str | None = None


class RouterBackend(Protocol):
    model: str

    def generate(self, messages: list[dict[str, str]]) -> RouterGeneration:
        raise NotImplementedError


class MockRouterBackend:
    def __init__(self, responses: dict[str, str]) -> None:
        self.model = "mock-router"
        self._responses = responses

    def generate_for_case(
        self, case_id: str, messages: list[dict[str, str]]
    ) -> RouterGeneration:
        del messages
        return RouterGeneration(self._responses[case_id], total_ms=1.0)


class LlamaRouterBackend:
    def __init__(
        self,
        model: str,
        base_url: str,
        client: Any | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s, trust_env=False)

    def generate(self, messages: list[dict[str, str]]) -> RouterGeneration:
        started = time.perf_counter()
        try:
            response = self._client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": "Bearer local-no-key"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 64,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            response.raise_for_status()
            payload = response.json()
            raw = payload["choices"][0]["message"]["content"]
        except httpx.TimeoutException:
            return RouterGeneration(
                "",
                (time.perf_counter() - started) * 1000,
                error_code="timeout",
            )
        except Exception:
            return RouterGeneration(
                "",
                (time.perf_counter() - started) * 1000,
                error_code="server_error",
            )
        total_ms = (time.perf_counter() - started) * 1000
        usage = payload.get("usage", {})
        tokens = usage.get("completion_tokens")
        return RouterGeneration(
            raw_output=str(raw),
            total_ms=total_ms,
            first_token_ms=None,
            output_tokens=tokens,
            tokens_per_s=(tokens * 1000 / total_ms) if tokens else None,
        )
