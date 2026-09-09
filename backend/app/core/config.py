import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
LOCAL_ENV_OVERRIDES = {
    "OPENAI_API_KEY": "openai_api_key",
    "OPENAI_BASE_URL": "openai_base_url",
    "OPENAI_CHAT_MODEL": "openai_chat_model",
    "OPENAI_VISION_MODEL": "openai_vision_model",
    "ROUTER_API_KEY": "router_api_key",
    "ROUTER_BASE_URL": "router_base_url",
    "ROUTER_MODEL": "router_model",
    "ROUTER_TIMEOUT_SECONDS": "router_timeout_seconds",
    "ROUTER_MANAGED": "router_managed",
    "ROUTER_SERVER_PATH": "router_server_path",
    "ROUTER_GGUF_PATH": "router_gguf_path",
    "ROUTER_HOST": "router_host",
    "ROUTER_PORT": "router_port",
    "ROUTER_READY_TIMEOUT_SECONDS": "router_ready_timeout_seconds",
    "AGENT_API_KEY": "agent_api_key",
    "AGENT_BASE_URL": "agent_base_url",
    "AGENT_MODEL": "agent_model",
    "AGENT_TIMEOUT_SECONDS": "agent_timeout_seconds",
    "AGENT_TOTAL_TIMEOUT_SECONDS": "agent_total_timeout_seconds",
    "AGENT_RETRIEVAL_CANDIDATE_K": "agent_retrieval_candidate_k",
    "AGENT_JUDGE_CONTEXT_TOKENS": "agent_judge_context_tokens",
    "AGENT_GENERATION_EVIDENCE_TOKENS": "agent_generation_evidence_tokens",
    "AGENT_MAX_ROUNDS": "agent_max_rounds",
    "AGENT_MAX_TOOL_CALLS": "agent_max_tool_calls",
    "JUDGE_API_KEY": "judge_api_key",
    "JUDGE_BASE_URL": "judge_base_url",
    "JUDGE_MODEL": "judge_model",
    "JUDGE_TIMEOUT_SECONDS": "judge_timeout_seconds",
}
DEFAULT_HUGGINGFACE_CACHE_DIR = PROJECT_ROOT / ".hf-cache"
HUGGINGFACE_CACHE_ENV_VARS = (
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
)


def configure_huggingface_cache(default_root: Path = DEFAULT_HUGGINGFACE_CACHE_DIR) -> None:
    root = Path(os.environ.get("HF_HOME", str(default_root)))
    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(root / "transformers"))
    os.environ.setdefault(
        "TIKTOKEN_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "tiktoken")
    )


configure_huggingface_cache()


class Settings(BaseSettings):
    app_name: str = "PaperMind Agent API"
    app_version: str = "0.1.0"
    database_url: str = "sqlite:///./data/papermind.db"
    storage_dir: str = "../storage"
    chroma_dir: str = "../chroma"
    backend_cors_origins: str = Field(default="http://localhost:5173")
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4.1-mini"
    openai_vision_model: str | None = None
    router_api_key: str | None = None
    router_base_url: str | None = None
    router_model: str | None = None
    router_timeout_seconds: float = Field(default=15.0, gt=0)
    router_managed: bool = False
    router_server_path: str | None = None
    router_gguf_path: str | None = None
    router_host: str = "127.0.0.1"
    router_port: int = Field(default=8089, gt=0, le=65535)
    router_ready_timeout_seconds: float = Field(default=120.0, gt=0)
    agent_api_key: str | None = None
    agent_base_url: str | None = None
    agent_model: str | None = None
    agent_timeout_seconds: float = Field(default=90.0, gt=0)
    agent_total_timeout_seconds: float = Field(default=180.0, gt=0)
    agent_retrieval_candidate_k: int = Field(default=20, gt=0, le=100)
    agent_judge_context_tokens: int = Field(default=24000, gt=0)
    agent_generation_evidence_tokens: int = Field(default=12000, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    bm25_candidates: int = Field(default=20, gt=0)
    vector_candidates_per_query: int = Field(default=15, gt=0)
    fusion_candidates: int = Field(default=20, gt=0)
    simple_evidence_limit: int = Field(default=5, gt=0)
    agent_max_rounds: int = Field(default=3, gt=0)
    agent_max_tool_calls: int = Field(default=5, gt=0)
    judge_api_key: str | None = None
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_timeout_seconds: float = Field(default=60.0, gt=0)
    context_compaction_enabled: bool = True
    context_window_tokens: int = Field(default=128000, gt=0)
    context_reserve_tokens: int = Field(default=16384, gt=0)
    context_keep_recent_tokens: int = Field(default=20000, gt=0)
    context_summary_max_tokens: int = Field(default=2048, gt=0)
    mineru_root: str = str(PROJECT_ROOT / ".mineru")
    mineru_formula_enabled: bool = True
    mineru_formula_timeout_seconds: float = Field(default=30.0, gt=0)
    mineru_figure_enabled: bool = False
    mineru_figure_service_url: str = "http://127.0.0.1:8002"
    mineru_figure_startup_timeout_seconds: float = Field(default=180.0, gt=0)
    mineru_figure_request_timeout_seconds: float = Field(default=120.0, gt=0)
    mineru_figure_idle_timeout_seconds: float = Field(default=15.0, ge=0)
    mineru_figure_model_id: str = "OpenDataLab/MinerU2.5-Pro-2605-1.2B"
    mineru_figure_revision: str | None = None
    mineru_figure_python: str = str(
        PROJECT_ROOT / ".mineru" / "figure-env" / "Scripts" / "python.exe"
    )
    mineru_figure_manifest: str = str(PROJECT_ROOT / ".mineru" / "figure-model.json")

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]

    @property
    def resolved_agent_api_key(self) -> str | None:
        return self.agent_api_key

    @property
    def resolved_agent_base_url(self) -> str | None:
        return self.agent_base_url

    @property
    def resolved_agent_model(self) -> str | None:
        return self.agent_model

    @property
    def resolved_router_api_key(self) -> str | None:
        if self.router_api_key:
            return self.router_api_key
        if self.router_managed:
            # llama-server ignores credentials; the OpenAI client still needs a
            # non-empty placeholder key for the local transport.
            return "local-managed-router"
        return None

    @property
    def resolved_router_base_url(self) -> str | None:
        if self.router_base_url:
            return self.router_base_url
        if self.router_managed:
            return f"http://{self.router_host}:{self.router_port}/v1"
        return None

    @property
    def resolved_router_model(self) -> str | None:
        return self.router_model

    @model_validator(mode="after")
    def validate_context_token_budget(self) -> "Settings":
        if self.agent_total_timeout_seconds < self.agent_timeout_seconds:
            raise ValueError(
                "agent_total_timeout_seconds must be greater than or equal to "
                "agent_timeout_seconds"
            )
        if self.router_managed and not (self.router_server_path and self.router_gguf_path):
            raise ValueError(
                "router_managed requires router_server_path and router_gguf_path"
            )
        threshold = self.context_window_tokens - self.context_reserve_tokens
        if threshold <= 0:
            raise ValueError(
                "context_window_tokens must be greater than context_reserve_tokens"
            )
        if self.context_keep_recent_tokens >= threshold:
            raise ValueError(
                "context_keep_recent_tokens must be less than the compaction threshold"
            )
        for field_name in (
            "agent_judge_context_tokens",
            "agent_generation_evidence_tokens",
        ):
            if getattr(self, field_name) >= threshold:
                raise ValueError(
                    f"{field_name} must be less than the available context"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings(**_local_settings_overrides())


def _local_settings_overrides() -> dict[str, str]:
    values: dict[str, str] = {}
    for env_file in (PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"):
        values.update(_parse_env_file(env_file))
    return {
        field_name: values[env_name]
        for env_name, field_name in LOCAL_ENV_OVERRIDES.items()
        if values.get(env_name)
    }


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values
