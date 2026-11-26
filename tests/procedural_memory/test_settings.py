import pytest
from pydantic import ValidationError

from backend.app.core import config


def test_procedural_memory_settings(monkeypatch):
    monkeypatch.delenv("PROCEDURAL_MEMORY_MODE", raising=False)
    settings = config.Settings(_env_file=None)
    assert settings.procedural_memory_mode == "off"
    assert settings.procedural_memory_context_tokens > 0
    monkeypatch.setenv("PROCEDURAL_MEMORY_MODE", "write")
    assert config.Settings(_env_file=None).procedural_memory_mode == "write"
    monkeypatch.setenv("PROCEDURAL_MEMORY_MODE", "unknown")
    with pytest.raises(ValidationError):
        config.Settings(_env_file=None)
