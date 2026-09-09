from pathlib import Path

import pytest
from backend.app.core import config
from pydantic import ValidationError

AGENTIC_RAG_ENV_NAMES = (
    "ROUTER_API_KEY",
    "ROUTER_BASE_URL",
    "ROUTER_MODEL",
    "ROUTER_TIMEOUT_SECONDS",
    "ROUTER_MANAGED",
    "ROUTER_SERVER_PATH",
    "ROUTER_GGUF_PATH",
    "ROUTER_HOST",
    "ROUTER_PORT",
    "ROUTER_READY_TIMEOUT_SECONDS",
    "AGENT_API_KEY",
    "AGENT_BASE_URL",
    "AGENT_MODEL",
    "AGENT_TIMEOUT_SECONDS",
    "AGENT_TOTAL_TIMEOUT_SECONDS",
    "AGENT_RETRIEVAL_CANDIDATE_K",
    "AGENT_JUDGE_CONTEXT_TOKENS",
    "AGENT_GENERATION_EVIDENCE_TOKENS",
    "RRF_K",
    "BM25_CANDIDATES",
    "VECTOR_CANDIDATES_PER_QUERY",
    "FUSION_CANDIDATES",
    "SIMPLE_EVIDENCE_LIMIT",
    "AGENT_MAX_ROUNDS",
    "AGENT_MAX_TOOL_CALLS",
)


@pytest.fixture(autouse=True)
def isolate_agentic_rag_settings(monkeypatch):
    for name in AGENTIC_RAG_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_backend_env_openai_values_override_process_placeholders(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend_root = tmp_path / "backend"
    backend_root.mkdir()
    (backend_root / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-real-test-key",
                "OPENAI_BASE_URL=https://api.deepseek.com",
                "OPENAI_CHAT_MODEL=deepseek-chat",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "BACKEND_ROOT", backend_root)
    monkeypatch.setenv("OPENAI_API_KEY", "local-placeholder")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "placeholder-model")
    config.get_settings.cache_clear()

    settings = config.get_settings()

    assert settings.openai_api_key == "sk-real-test-key"
    assert settings.openai_base_url == "https://api.deepseek.com"
    assert settings.openai_chat_model == "deepseek-chat"


def test_agent_and_router_settings_do_not_fall_back_to_legacy_openai(monkeypatch) -> None:
    for name in (
        "AGENT_API_KEY",
        "AGENT_BASE_URL",
        "AGENT_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_CHAT_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = config.Settings(
        _env_file=None,
        openai_api_key="sk-agent-fallback",
        openai_base_url="https://api.deepseek.com",
        openai_chat_model="deepseek-chat",
    )

    assert settings.resolved_agent_api_key is None
    assert settings.resolved_agent_base_url is None
    assert settings.resolved_agent_model is None
    assert settings.resolved_router_api_key is None
    assert settings.resolved_router_base_url is None
    assert settings.resolved_router_model is None


def test_agentic_rag_settings_defaults() -> None:
    settings = config.Settings(_env_file=None)

    assert settings.router_api_key is None
    assert settings.router_base_url is None
    assert settings.router_model is None
    assert settings.agent_api_key is None
    assert settings.agent_base_url is None
    assert settings.agent_model is None
    assert settings.router_timeout_seconds == 15.0
    assert settings.router_managed is False
    assert settings.router_server_path is None
    assert settings.router_gguf_path is None
    assert settings.router_host == "127.0.0.1"
    assert settings.router_port == 8089
    assert settings.router_ready_timeout_seconds == 120.0
    assert settings.agent_timeout_seconds == 90.0
    assert settings.agent_total_timeout_seconds == 180.0
    assert settings.agent_retrieval_candidate_k == 20
    assert settings.agent_judge_context_tokens == 24000
    assert settings.agent_generation_evidence_tokens == 12000
    assert settings.rrf_k == 60
    assert settings.bm25_candidates == 20
    assert settings.vector_candidates_per_query == 15
    assert settings.fusion_candidates == 20
    assert settings.simple_evidence_limit == 5
    assert settings.agent_max_rounds == 3
    assert settings.agent_max_tool_calls == 5


def test_explicit_agent_and_router_settings_remain_independent(
    monkeypatch,
) -> None:
    monkeypatch.setenv("AGENT_API_KEY", "agent-key")
    monkeypatch.setenv("AGENT_BASE_URL", "https://agent.example/v1")
    monkeypatch.setenv("AGENT_MODEL", "agent-model")
    monkeypatch.setenv("ROUTER_API_KEY", "router-key")
    monkeypatch.setenv("ROUTER_BASE_URL", "https://router.example/v1")
    monkeypatch.setenv("ROUTER_MODEL", "router-model")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.example/v1")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "legacy-model")

    settings = config.Settings(_env_file=None)

    assert settings.resolved_agent_api_key == "agent-key"
    assert settings.resolved_agent_base_url == "https://agent.example/v1"
    assert settings.resolved_agent_model == "agent-model"
    assert settings.resolved_router_api_key == "router-key"
    assert settings.resolved_router_base_url == "https://router.example/v1"
    assert settings.resolved_router_model == "router-model"


@pytest.mark.parametrize(
    "field_name",
    [
        "router_timeout_seconds",
        "agent_timeout_seconds",
        "agent_total_timeout_seconds",
        "agent_retrieval_candidate_k",
        "agent_judge_context_tokens",
        "agent_generation_evidence_tokens",
        "rrf_k",
        "bm25_candidates",
        "vector_candidates_per_query",
        "fusion_candidates",
        "simple_evidence_limit",
        "agent_max_rounds",
        "agent_max_tool_calls",
    ],
)
def test_agentic_rag_positive_settings_reject_zero(field_name: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        config.Settings(_env_file=None, **{field_name: 0})

    assert any(error["loc"] == (field_name,) for error in exc_info.value.errors())


def test_agentic_research_safety_budgets_remain_bounded() -> None:
    settings = config.Settings(_env_file=None)

    assert settings.agent_max_rounds == 3
    assert settings.agent_max_tool_calls == 5
    assert settings.agent_total_timeout_seconds == 180.0


def test_agentic_retrieval_candidate_k_rejects_more_than_one_hundred() -> None:
    with pytest.raises(ValidationError) as exc_info:
        config.Settings(_env_file=None, agent_retrieval_candidate_k=101)

    assert any(
        error["loc"] == ("agent_retrieval_candidate_k",)
        for error in exc_info.value.errors()
    )


@pytest.mark.parametrize(
    "field_name",
    ["agent_judge_context_tokens", "agent_generation_evidence_tokens"],
)
def test_agentic_evidence_token_budgets_fit_available_context(field_name: str) -> None:
    with pytest.raises(ValidationError, match="must be less than the available context"):
        config.Settings(
            _env_file=None,
            context_window_tokens=32000,
            context_reserve_tokens=8000,
            context_keep_recent_tokens=12000,
            **{field_name: 24000},
        )


def test_managed_router_defaults_derive_local_transport() -> None:
    settings = config.Settings(
        _env_file=None,
        router_managed=True,
        router_server_path=".tools/llama.cpp/llama-server.exe",
        router_gguf_path="models/router/router.gguf",
    )

    assert settings.resolved_router_base_url == "http://127.0.0.1:8089/v1"
    assert settings.resolved_router_api_key == "local-managed-router"


def test_managed_router_requires_server_and_model_paths() -> None:
    with pytest.raises(ValidationError):
        config.Settings(_env_file=None, router_managed=True, router_server_path=None)


@pytest.mark.parametrize("port", [0, 65536, 70000])
def test_router_port_rejects_out_of_range_values(port: int) -> None:
    with pytest.raises(ValidationError):
        config.Settings(_env_file=None, router_port=port)


def test_local_env_overrides_include_router_and_agent_settings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend_root = tmp_path / "backend"
    backend_root.mkdir()
    (backend_root / ".env").write_text(
        "\n".join(
            [
                "ROUTER_API_KEY=router-key",
                "ROUTER_BASE_URL=https://router.example/v1",
                "ROUTER_MODEL=router-model",
                "ROUTER_TIMEOUT_SECONDS=12.5",
                "ROUTER_MANAGED=false",
                "ROUTER_PORT=8091",
                "AGENT_API_KEY=agent-key",
                "AGENT_BASE_URL=https://agent.example/v1",
                "AGENT_MODEL=agent-model",
                "AGENT_TIMEOUT_SECONDS=30",
                "AGENT_TOTAL_TIMEOUT_SECONDS=60",
                "AGENT_RETRIEVAL_CANDIDATE_K=24",
                "AGENT_JUDGE_CONTEXT_TOKENS=22000",
                "AGENT_GENERATION_EVIDENCE_TOKENS=11000",
                "AGENT_MAX_ROUNDS=4",
                "AGENT_MAX_TOOL_CALLS=7",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "BACKEND_ROOT", backend_root)

    assert config._local_settings_overrides() == {
        "router_api_key": "router-key",
        "router_base_url": "https://router.example/v1",
        "router_model": "router-model",
        "router_timeout_seconds": "12.5",
        "router_managed": "false",
        "router_port": "8091",
        "agent_api_key": "agent-key",
        "agent_base_url": "https://agent.example/v1",
        "agent_model": "agent-model",
        "agent_timeout_seconds": "30",
        "agent_total_timeout_seconds": "60",
        "agent_retrieval_candidate_k": "24",
        "agent_judge_context_tokens": "22000",
        "agent_generation_evidence_tokens": "11000",
        "agent_max_rounds": "4",
        "agent_max_tool_calls": "7",
    }


def test_get_settings_validates_and_coerces_local_numeric_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend_root = tmp_path / "backend"
    backend_root.mkdir()
    (backend_root / ".env").write_text(
        "\n".join(
            [
                "ROUTER_TIMEOUT_SECONDS=12.5",
                "ROUTER_PORT=8091",
                "AGENT_TIMEOUT_SECONDS=30",
                "AGENT_TOTAL_TIMEOUT_SECONDS=60",
                "AGENT_RETRIEVAL_CANDIDATE_K=24",
                "AGENT_JUDGE_CONTEXT_TOKENS=22000",
                "AGENT_GENERATION_EVIDENCE_TOKENS=11000",
                "AGENT_MAX_ROUNDS=4",
                "AGENT_MAX_TOOL_CALLS=7",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "BACKEND_ROOT", backend_root)
    monkeypatch.setenv("ROUTER_TIMEOUT_SECONDS", "99")
    monkeypatch.setenv("ROUTER_PORT", "8092")
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "80")
    monkeypatch.setenv("AGENT_TOTAL_TIMEOUT_SECONDS", "160")
    monkeypatch.setenv("AGENT_RETRIEVAL_CANDIDATE_K", "30")
    monkeypatch.setenv("AGENT_JUDGE_CONTEXT_TOKENS", "26000")
    monkeypatch.setenv("AGENT_GENERATION_EVIDENCE_TOKENS", "13000")
    monkeypatch.setenv("AGENT_MAX_ROUNDS", "8")
    monkeypatch.setenv("AGENT_MAX_TOOL_CALLS", "10")

    settings = config.get_settings()

    assert settings.router_timeout_seconds == 12.5
    assert settings.router_port == 8091
    assert settings.agent_timeout_seconds == 30.0
    assert settings.agent_total_timeout_seconds == 60.0
    assert settings.agent_retrieval_candidate_k == 24
    assert settings.agent_judge_context_tokens == 22000
    assert settings.agent_generation_evidence_tokens == 11000
    assert settings.agent_max_rounds == 4
    assert settings.agent_max_tool_calls == 7


def test_get_settings_rejects_invalid_local_cross_field_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend_root = tmp_path / "backend"
    backend_root.mkdir()
    (backend_root / ".env").write_text(
        "\n".join(
            [
                "AGENT_TIMEOUT_SECONDS=91",
                "AGENT_TOTAL_TIMEOUT_SECONDS=90",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "BACKEND_ROOT", backend_root)
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("AGENT_TOTAL_TIMEOUT_SECONDS", "40")

    with pytest.raises(ValidationError):
        config.get_settings()


@pytest.mark.parametrize(
    "overrides",
    [
        {"agent_timeout_seconds": 91.0, "agent_total_timeout_seconds": 90.0},
    ],
)
def test_agentic_rag_settings_reject_inconsistent_limits(overrides: dict) -> None:
    valid_settings = config.Settings(
        _env_file=None,
        agent_timeout_seconds=90.0,
        agent_total_timeout_seconds=90.0,
        simple_evidence_limit=5,
    )
    assert valid_settings.agent_total_timeout_seconds == 90.0

    with pytest.raises(ValidationError):
        config.Settings(_env_file=None, **overrides)


def test_config_sets_default_huggingface_cache_in_project_root(monkeypatch) -> None:
    for key in config.HUGGINGFACE_CACHE_ENV_VARS:
        monkeypatch.delenv(key, raising=False)

    config.configure_huggingface_cache()

    assert config.os.environ["HF_HOME"] == str(config.PROJECT_ROOT / ".hf-cache")
    assert config.os.environ["HUGGINGFACE_HUB_CACHE"] == str(
        config.PROJECT_ROOT / ".hf-cache" / "hub"
    )
    assert config.os.environ["TRANSFORMERS_CACHE"] == str(
        config.PROJECT_ROOT / ".hf-cache" / "transformers"
    )


def test_config_keeps_existing_huggingface_cache_env(monkeypatch) -> None:
    monkeypatch.setenv("HF_HOME", "E:\\custom-hf")
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)

    config.configure_huggingface_cache()

    assert config.os.environ["HF_HOME"] == "E:\\custom-hf"
    assert config.os.environ["HUGGINGFACE_HUB_CACHE"] == "E:\\custom-hf\\hub"
    assert config.os.environ["TRANSFORMERS_CACHE"] == "E:\\custom-hf\\transformers"


def test_context_compaction_settings_defaults(monkeypatch) -> None:
    for name in (
        "CONTEXT_COMPACTION_ENABLED",
        "CONTEXT_WINDOW_TOKENS",
        "CONTEXT_RESERVE_TOKENS",
        "CONTEXT_KEEP_RECENT_TOKENS",
        "CONTEXT_SUMMARY_MAX_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = config.Settings(_env_file=None)

    assert settings.context_compaction_enabled is True
    assert settings.context_window_tokens == 128000
    assert settings.context_reserve_tokens == 16384
    assert settings.context_keep_recent_tokens == 20000
    assert settings.context_summary_max_tokens == 2048


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "context_window_tokens": 1000,
            "context_reserve_tokens": 1000,
            "context_keep_recent_tokens": 1,
        },
        {
            "context_window_tokens": 1000,
            "context_reserve_tokens": 900,
            "context_keep_recent_tokens": 200,
        },
    ],
)
def test_context_token_settings_reject_invalid_budgets(
    overrides: dict[str, int],
) -> None:
    valid_settings = config.Settings(
        _env_file=None,
        context_window_tokens=1000,
        context_reserve_tokens=900,
        context_keep_recent_tokens=99,
        context_summary_max_tokens=10,
        agent_judge_context_tokens=20,
        agent_generation_evidence_tokens=20,
    )
    assert valid_settings.context_keep_recent_tokens == 99

    with pytest.raises(ValidationError):
        config.Settings(
            _env_file=None,
            agent_judge_context_tokens=20,
            agent_generation_evidence_tokens=20,
            **overrides,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "context_window_tokens",
        "context_reserve_tokens",
        "context_keep_recent_tokens",
        "context_summary_max_tokens",
    ],
)
def test_context_token_settings_require_positive_values(field_name: str) -> None:
    values = {
        "context_window_tokens": 128000,
        "context_reserve_tokens": 16384,
        "context_keep_recent_tokens": 20000,
        "context_summary_max_tokens": 2048,
    }
    values[field_name] = 0

    with pytest.raises(ValidationError) as exc_info:
        config.Settings(_env_file=None, **values)

    assert any(
        error["loc"] == (field_name,) and error["type"] == "greater_than"
        for error in exc_info.value.errors()
    )
