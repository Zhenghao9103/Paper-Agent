from backend.app.services.llm import _safe_error
from fastapi.testclient import TestClient


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "papermind-api",
        "version": "0.1.0",
    }


def test_llm_health_reports_safe_configuration_status(client: TestClient) -> None:
    response = client.get("/api/health/llm")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["key_configured"], bool)
    assert payload["base_url"]
    assert payload["model"]
    assert set(payload["router_model"]) >= {"key_configured", "base_url", "model"}
    assert set(payload["agent_model"]) >= {"key_configured", "base_url", "model"}
    # Compatibility aliases remain for legacy clients.
    assert payload["small_model"] == payload["router_model"]
    assert payload["complex_model"] == payload["agent_model"]
    assert "openai_api_key" not in payload


def test_llm_error_sanitizer_masks_api_key_fragments() -> None:
    error = Exception("Incorrect API key provided: sk-12345********abcdef.")

    assert "sk-12345" not in _safe_error(error)
    assert "sk-***" in _safe_error(error)
