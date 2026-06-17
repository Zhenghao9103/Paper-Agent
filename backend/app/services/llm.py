import re

from openai import OpenAI, OpenAIError

from ..core.config import get_settings


def llm_config_status() -> dict[str, str | bool]:
    settings = get_settings()
    return {
        "key_configured": bool(settings.openai_api_key),
        "base_url": settings.openai_base_url,
        "model": settings.openai_chat_model,
    }


def ping_llm() -> dict[str, str | bool]:
    settings = get_settings()
    status = llm_config_status()
    if not settings.openai_api_key:
        return {**status, "ok": False, "error": "OPENAI_API_KEY is not configured."}

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    try:
        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": "Reply with OK only."},
                {"role": "user", "content": "ping"},
            ],
            temperature=0,
            max_tokens=8,
        )
    except OpenAIError as exc:
        return {**status, "ok": False, "error": f"{type(exc).__name__}: {_safe_error(exc)}"}

    content = response.choices[0].message.content or ""
    return {**status, "ok": True, "reply": content.strip()}


def complete(prompt: str) -> str | None:
    settings = get_settings()
    if not settings.openai_api_key:
        return None

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    try:
        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": "You are a concise Chinese academic paper assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
    except OpenAIError:
        return None
    return response.choices[0].message.content


def _safe_error(exc: Exception) -> str:
    return re.sub(r"sk-[A-Za-z0-9_*.-]+", "sk-***", str(exc))
