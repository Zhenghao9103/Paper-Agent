"""Small, isolated clients for the OpenAI-compatible routing model."""

import json
from functools import lru_cache
from typing import Any

from openai import OpenAI

from ..core.config import get_settings
from .llm import ContextWindowExceededError, is_context_window_error


class ModelError(RuntimeError):
    """Base error for model configuration and provider failures."""


class ModelConfigurationError(ModelError):
    """Raised when the independent router model is not configured."""


class ModelClientError(ModelError):
    """Raised when the router provider returns an unusable response."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "provider_error",
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.attempts = attempts


@lru_cache(maxsize=1)
def _agent_client() -> OpenAI:
    """Build the independent answer/agent transport client.

    The cache is deliberately limited to the transport object. Responses are
    never cached because they are request-scoped and may contain private paper
    evidence.
    """

    settings = get_settings()
    missing = [
        name
        for name, value in (
            ("AGENT_API_KEY", settings.resolved_agent_api_key),
            ("AGENT_BASE_URL", settings.resolved_agent_base_url),
            ("AGENT_MODEL", settings.resolved_agent_model),
        )
        if not value
    ]
    if missing:
        raise ModelConfigurationError(
            "Agent model is not configured: " + ", ".join(missing) + "."
        )

    try:
        return OpenAI(
            api_key=settings.resolved_agent_api_key,
            base_url=settings.resolved_agent_base_url,
            timeout=settings.agent_timeout_seconds,
            max_retries=0,
        )
    except Exception as exc:
        raise ModelClientError(_safe_model_error(exc)) from exc


@lru_cache(maxsize=1)
def _router_client() -> OpenAI:
    """Build and cache the transport client, never a model response."""

    settings = get_settings()
    missing = [
        name
        for name, value in (
            ("ROUTER_API_KEY", settings.resolved_router_api_key),
            ("ROUTER_BASE_URL", settings.resolved_router_base_url),
            ("ROUTER_MODEL", settings.resolved_router_model),
        )
        if not value
    ]
    if missing:
        raise ModelConfigurationError(
            "Router model is not configured: " + ", ".join(missing) + "."
        )

    try:
        return OpenAI(
            api_key=settings.resolved_router_api_key,
            base_url=settings.resolved_router_base_url,
            timeout=settings.router_timeout_seconds,
            max_retries=0,
        )
    except Exception as exc:
        raise ModelClientError(_safe_model_error(exc)) from exc


def router_json(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Call the router and decode its JSON object without caching the result.

    The generation parameters mirror the frozen M0/Phase 6 protocol exactly:
    no grammar forcing (``response_format``), a 64-token output bound, and
    Qwen3 thinking mode disabled. Grammar-constrained decoding would suppress
    the trained stop behaviour and degenerate into ~500-token generations.
    """

    settings = get_settings()
    try:
        response = _router_client().chat.completions.create(
            model=settings.resolved_router_model,
            messages=messages,
            temperature=0,
            max_tokens=64,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except ModelError:
        raise
    except Exception as exc:
        raise ModelClientError(_safe_model_error(exc)) from exc

    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    content = getattr(message, "content", None) or ""
    if not isinstance(content, str) or not content.strip():
        raise ModelClientError("Router returned an empty response.")
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ModelClientError("Router returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ModelClientError("Router response must be a JSON object.")
    return payload


def answer_json(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Call the main agent model and decode an answer JSON object.

    Provider failures are intentionally converted to ``None`` so the caller can
    use the deterministic evidence-only template. Context-window failures are
    different: they must propagate to the API's existing compaction handling.
    """

    settings = get_settings()
    effective_model = model or settings.resolved_agent_model
    request: dict[str, Any] = {
        "model": effective_model,
        "messages": messages,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    if effective_model == "glm-5.2":
        request["extra_body"] = {"enable_thinking": False}
    try:
        response = _agent_client().chat.completions.create(**request)
    except ContextWindowExceededError:
        raise
    except ModelError:
        return None
    except Exception as exc:
        if is_context_window_error(exc):
            raise ContextWindowExceededError(_safe_model_error(exc)) from exc
        return None

    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    content = getattr(message, "content", None) or ""
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def agent_json(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    """Call the Agent model for a required, bounded JSON object.

    Unlike the best-effort answer helper, structured control-plane calls fail
    explicitly so the orchestrator can distinguish provider and schema errors.
    """

    settings = get_settings()
    request: dict[str, Any] = {
        "model": settings.resolved_agent_model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    response = None
    for attempt in range(1, 3):
        try:
            response = _agent_client().chat.completions.create(**request)
            break
        except ContextWindowExceededError:
            raise
        except ModelError:
            raise
        except Exception as exc:
            if is_context_window_error(exc):
                raise ContextWindowExceededError(_safe_model_error(exc)) from exc
            category = _provider_error_category(exc)
            if attempt == 1 and category in {
                "connection_error",
                "rate_limit",
                "server_error",
                "timeout",
            }:
                continue
            raise ModelClientError(
                _safe_model_error(exc),
                category=category,
                attempts=attempt,
            ) from exc

    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    content = getattr(message, "content", None) or ""
    if not isinstance(content, str) or not content.strip():
        raise ModelClientError(
            "Agent returned an empty response.",
            category="empty_response",
        )
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ModelClientError(
            "Agent returned invalid JSON.",
            category="invalid_json",
        ) from exc
    if not isinstance(payload, dict):
        raise ModelClientError(
            "Agent response must be a JSON object.",
            category="invalid_json",
        )
    return payload


# Explicit name used by the research flow; ``answer_json`` remains as a
# backwards-compatible alias for existing callers and tests.
complex_answer_json = answer_json


def complex_text_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str | None:
    """Run a plain completion on the independent complex model."""

    return _text_completion(
        _agent_client,
        get_settings().resolved_agent_model,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _text_completion(
    client_factory: Any,
    model: str | None,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int | None,
) -> str | None:
    if not model:
        return None
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    try:
        response = client_factory().chat.completions.create(**request)
    except ContextWindowExceededError:
        raise
    except ModelError:
        return None
    except Exception as exc:
        if is_context_window_error(exc):
            raise ContextWindowExceededError(_safe_model_error(exc)) from exc
        return None
    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    content = getattr(message, "content", None) if message else None
    return content.strip() if isinstance(content, str) and content.strip() else None


def agent_chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    tool_choice: str | dict[str, Any] = "auto",
) -> Any:
    """Call the independent OpenAI-compatible agent model with tools.

    The returned SDK message is intentionally kept intact so providers such as
    DeepSeek can carry ``reasoning_content`` across the next turn.  Tool
    arguments are validated by the orchestrator locally; this request does not
    opt into provider-specific beta/strict tool schemas.
    """

    settings = get_settings()
    request = {
        "model": settings.resolved_agent_model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
    }
    response = None
    for attempt in range(1, 3):
        try:
            response = _agent_client().chat.completions.create(**request)
            break
        except ContextWindowExceededError:
            raise
        except ModelError:
            raise
        except Exception as exc:
            if is_context_window_error(exc):
                raise ContextWindowExceededError(_safe_model_error(exc)) from exc
            category = _provider_error_category(exc)
            if attempt == 1 and category in {
                "connection_error",
                "rate_limit",
                "server_error",
                "timeout",
            }:
                continue
            raise ModelClientError(
                _safe_model_error(exc),
                category=category,
                attempts=attempt,
            ) from exc

    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    if message is None:
        return type("AgentMessage", (), {"role": "assistant", "content": "", "tool_calls": []})()
    return message


def _safe_model_error(exc: Exception) -> str:
    del exc
    return "Model provider request failed."


def _provider_error_category(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 504}:
        return "timeout"
    if status_code == 429:
        return "rate_limit"
    if isinstance(status_code, int) and status_code >= 500:
        return "server_error"
    if isinstance(status_code, int) and 400 <= status_code < 500:
        return "client_error"

    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout"
    if "ratelimit" in name or "rate_limit" in name:
        return "rate_limit"
    if "connection" in name or "connect" in name:
        return "connection_error"
    if "server" in name or "serviceunavailable" in name:
        return "server_error"
    return "provider_error"
