import pytest
from pydantic import ValidationError

from backend.app.core import config


def test_procedural_memory_settings(monkeypatch):
    monkeypatch.delenv("PROCEDURAL_MEMORY_MODE", raising=False)
    settings = config.Settings(_env_file=None)
    assert settings.procedural_memory_mode == "rsi"
    assert settings.app_version == "3.0.0"
    assert settings.procedural_memory_context_tokens > 0
    for mode in ("off", "write", "flat", "rsi"):
        monkeypatch.setenv("PROCEDURAL_MEMORY_MODE", mode)
        assert config.Settings(_env_file=None).procedural_memory_mode == mode
    monkeypatch.setenv("PROCEDURAL_MEMORY_MODE", "unknown")
    with pytest.raises(ValidationError):
        config.Settings(_env_file=None)
