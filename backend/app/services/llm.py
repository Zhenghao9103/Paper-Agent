import base64
import re
from pathlib import Path

from openai import OpenAI, OpenAIError

from ..core.config import get_settings


class ContextWindowExceededError(RuntimeError):
    pass


class ContextSummaryError(RuntimeError):
    pass


class VisionDescriptionError(RuntimeError):
    pass


CONTEXT_SUMMARY_HEADINGS = (
    "## 用户目标与偏好",
    "## 已确认结论",
    "## 当前研究决策",
    "## 尚未解决的问题",
    "## 下一步",
)


def is_context_window_error(exc: Exception) -> bool:
    code = str(getattr(exc, "code", "") or "").lower()
    if code == "context_length_exceeded":
        return True
    if code in {"rate_limit_exceeded", "rate_limit_error", "insufficient_quota"}:
        return False
    if getattr(exc, "status_code", None) == 429:
        return False

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "context_length_exceeded",
            "maximum context length",
            "context window",
            "too many tokens",
        )
    )


def _tier_status(
    api_key: str | None, base_url: str | None, model: str | None
) -> dict[str, str | bool | None]:
    return {
        "key_configured": bool(api_key),
        "base_url": base_url,
        "model": model,
    }


def llm_config_status() -> dict[str, object]:
    """Expose the Router and Agent tiers without revealing credentials.

    ``router_model``/``agent_model`` are the production field names; the old
    ``small_model``/``complex_model`` keys stay temporarily as aliases for
    clients that only know the legacy shape.
    """

    settings = get_settings()
    router = _tier_status(
        settings.resolved_router_api_key,
        settings.resolved_router_base_url,
        settings.resolved_router_model,
    )
    agent = _tier_status(
        settings.resolved_agent_api_key,
        settings.resolved_agent_base_url,
        settings.resolved_agent_model,
    )
    return {
        "key_configured": bool(router["key_configured"] and agent["key_configured"]),
        "base_url": "separate",
        "model": f"{router['model'] or 'unconfigured'} / {agent['model'] or 'unconfigured'}",
        "router_model": router,
        "agent_model": agent,
        "small_model": router,
        "complex_model": agent,
    }


_PING_TARGETS = {
    "router": "router_model",
    "agent": "agent_model",
    "small": "router_model",
    "complex": "agent_model",
}


def ping_llm(target: str | None = None) -> dict[str, object]:
    """Ping the Router, Agent, or both configured model tiers."""

    status = llm_config_status()
    names = [target] if target in _PING_TARGETS else ["router", "agent"]
    results: dict[str, object] = {}
    for name in names:
        model_status = status[_PING_TARGETS[name]]
        if not isinstance(model_status, dict) or not model_status.get("key_configured"):
            results[name] = {
                **(model_status if isinstance(model_status, dict) else {}),
                "ok": False,
                "error": f"{name} model is not configured.",
            }
            continue
        try:
            from .model_clients import _agent_client, _router_client

            client = _router_client() if name in {"router", "small"} else _agent_client()
            response = client.chat.completions.create(
                model=model_status.get("model"),
                messages=[
                    {"role": "system", "content": "Reply with OK only."},
                    {"role": "user", "content": "ping"},
                ],
                temperature=0,
                max_tokens=8,
            )
            content = getattr(getattr(response.choices[0], "message", None), "content", "") or ""
            results[name] = {**model_status, "ok": True, "reply": content.strip()}
        except Exception as exc:
            results[name] = {
                **model_status,
                "ok": False,
                "error": f"{type(exc).__name__}: {_safe_error(exc)}",
            }
    ok_values = [bool(value.get("ok")) for value in results.values() if isinstance(value, dict)]
    return {**status, **results, "ok": bool(ok_values) and all(ok_values)}


def complete(prompt: str) -> str | None:
    settings = get_settings()
    if hasattr(settings, "agent_api_key"):
        from .model_clients import complex_text_completion

        return complex_text_completion(
            [
                {
                    "role": "system",
                    "content": "You are a concise Chinese academic paper assistant.",
                },
                {"role": "user", "content": prompt},
            ]
        )
    if not settings.openai_api_key:
        return None

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    try:
        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise Chinese academic paper assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
    except OpenAIError:
        return None
    return response.choices[0].message.content


def describe_figure(
    image_path: Path,
    *,
    title: str,
    section: str | None,
    caption: str | None,
) -> str | None:
    """Return a bounded visual description, or ``None`` when vision is disabled.

    The prompt explicitly asks for visible structure rather than OCR or invented
    results.  Failures are best-effort so a saved crop remains usable on its own.
    """

    settings = get_settings()
    if not settings.openai_api_key or not settings.openai_vision_model:
        return None
    try:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    except OSError:
        return None
    prompt = (
        "Describe only the visible modules, relationships, axes, or visual evidence "
        "in this academic figure. Be concise. Do not invent results and do not "
        "transcribe it as OCR.\n"
        f"Paper: {title}\nSection: {section or 'Unknown'}\n"
        f"Caption: {caption or 'None'}"
    )
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    try:
        response = client.chat.completions.create(
            model=settings.openai_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                }
            ],
            temperature=0,
            max_tokens=256,
        )
    except OpenAIError as exc:
        raise VisionDescriptionError(_safe_error(exc)) from exc
    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    content = getattr(message, "content", None) if message else None
    return content.strip() if content and content.strip() else None


def complete_research(prompt: str) -> str | None:
    settings = get_settings()
    if hasattr(settings, "agent_api_key"):
        from .model_clients import complex_text_completion

        return complex_text_completion(
            [
                {
                    "role": "system",
                    "content": "You are a concise Chinese academic paper assistant.",
                },
                {"role": "user", "content": prompt},
            ]
        )
    if not settings.openai_api_key:
        return None

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    try:
        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise Chinese academic paper assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
    except OpenAIError as exc:
        if is_context_window_error(exc):
            raise ContextWindowExceededError(_safe_error(exc)) from exc
        return None
    return response.choices[0].message.content


def complete_context_summary(
    prompt: str,
    *,
    max_tokens: int,
    required_reference_markers: tuple[str, ...] = (),
) -> str:
    """Generate a context checkpoint with the Agent model.

    Context compression is an Agent-model responsibility: the local Router is
    a three-class classifier and never receives summarization egress.
    """

    settings = get_settings()
    if hasattr(settings, "agent_api_key"):
        if not (settings.agent_api_key and settings.agent_base_url and settings.agent_model):
            raise ContextSummaryError("AGENT model is not configured.")
        from .model_clients import complex_text_completion

        content = complex_text_completion(
            [{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        if not content:
            raise ContextSummaryError("Agent model returned an empty context summary.")
        return _validate_context_summary(
            content,
            required_reference_markers=required_reference_markers,
        )
    if not settings.openai_api_key:
        raise ContextSummaryError("OPENAI_API_KEY is not configured.")

    try:
        client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "请生成精确的中文学术上下文检查点，并保留所有引用标识。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
        )
        choices = getattr(response, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        content = (getattr(message, "content", None) or "").strip()
        if not content:
            raise ContextSummaryError("LLM returned an empty context summary.")
        return _validate_context_summary(
            content,
            required_reference_markers=required_reference_markers,
        )
    except ContextSummaryError:
        raise
    except Exception as exc:
        raise ContextSummaryError(_safe_error(exc)) from exc


def _validate_context_summary(
    content: str,
    *,
    required_reference_markers: tuple[str, ...],
) -> str:
    lines = [line.strip() for line in content.splitlines()]
    next_index = 0
    for heading in CONTEXT_SUMMARY_HEADINGS:
        try:
            next_index = lines.index(heading, next_index) + 1
        except ValueError as exc:
            raise ContextSummaryError(
                "LLM context summary is missing required headings or order."
            ) from exc

    if required_reference_markers and not any(
        marker in content
        for marker in required_reference_markers
    ):
        raise ContextSummaryError(
            "LLM context summary omitted every required local citation marker."
        )
    return content


def _safe_error(exc: Exception) -> str:
    message = str(exc)
    message = re.sub(r"sk-[A-Za-z0-9_*.-]+", "sk-***", message)
    message = re.sub(
        r"(?i)(api[_ -]?key|authorization|token)(\s*[:=]\s*)([^\s,;]+)",
        r"\1\2***",
        message,
    )
    return message[:500]
