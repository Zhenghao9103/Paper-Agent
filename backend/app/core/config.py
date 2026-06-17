from functools import lru_cache
import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
OPENAI_ENV_OVERRIDES = {
    "OPENAI_API_KEY": "openai_api_key",
    "OPENAI_BASE_URL": "openai_base_url",
    "OPENAI_CHAT_MODEL": "openai_chat_model",
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

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for field_name, value in _local_openai_overrides().items():
        setattr(settings, field_name, value)
    return settings


def _local_openai_overrides() -> dict[str, str]:
    values: dict[str, str] = {}
    for env_file in (PROJECT_ROOT / ".env", BACKEND_ROOT / ".env"):
        values.update(_parse_env_file(env_file))
    return {
        field_name: values[env_name]
        for env_name, field_name in OPENAI_ENV_OVERRIDES.items()
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
