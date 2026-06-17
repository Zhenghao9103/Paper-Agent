from fastapi import APIRouter

from ...core.config import get_settings
from ...schemas.health import HealthResponse
from ...services.llm import llm_config_status, ping_llm

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service="papermind-api",
        version=settings.app_version,
    )


@router.get("/health/llm")
def llm_health(ping: bool = False) -> dict[str, str | bool]:
    if ping:
        return ping_llm()
    return llm_config_status()
