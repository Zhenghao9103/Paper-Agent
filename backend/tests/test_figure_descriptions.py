from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from backend.app.ingestion.figure_descriptions import (
    FigureDescriptionError,
    MinerUFigureClient,
    normalize_figure_description,
    project_figure_description,
)


def test_normalizer_returns_exact_bounded_six_field_shape() -> None:
    result = normalize_figure_description(
        {
            "figure_type": "architecture",
            "caption": "  Model overview  ",
            "visible_text": [" Encoder ", "Encoder", "Decoder"],
            "components": ["Input", "Encoder", "Decoder"],
            "relationships": ["Input flows into Encoder"],
            "summary": "  An encoder-decoder architecture.  ",
            "invented": "discard me",
        }
    )

    assert result.model_dump() == {
        "figure_type": "architecture",
        "caption": "Model overview",
        "visible_text": ["Encoder", "Decoder"],
        "components": ["Input", "Encoder", "Decoder"],
        "relationships": ["Input flows into Encoder"],
        "summary": "An encoder-decoder architecture.",
    }


def test_normalizer_defaults_missing_fields_and_unknown_type() -> None:
    result = normalize_figure_description({"figure_type": "heatmap", "summary": "Visible."})

    assert result.figure_type == "other"
    assert result.caption == ""
    assert result.visible_text == []
    assert result.components == []
    assert result.relationships == []


@pytest.mark.parametrize(
    "payload",
    [
        {"visible_text": "not-a-list"},
        {"components": ["ok", 3]},
        {"summary": ["not-a-string"]},
    ],
)
def test_normalizer_rejects_wrong_field_types(payload: dict) -> None:
    with pytest.raises(FigureDescriptionError) as exc:
        normalize_figure_description(payload)
    assert exc.value.code == "figure_description_invalid_json"


def test_normalizer_rejects_markdown_fenced_json() -> None:
    with pytest.raises(FigureDescriptionError):
        normalize_figure_description('```json\n{"summary":"x"}\n```')


def test_projection_is_deterministic_and_source_ordered() -> None:
    description = normalize_figure_description(
        {
            "figure_type": "flowchart",
            "caption": "Figure 2. Pipeline.",
            "components": ["Input", "Encoder"],
            "relationships": ["Input enters Encoder"],
            "summary": "A two-stage pipeline.",
            "visible_text": ["Stage A"],
        }
    )

    assert project_figure_description(description).splitlines() == [
        "Figure 2. Pipeline.",
        "Figure type: flowchart",
        "Components: Input; Encoder",
        "Relationships: Input enters Encoder",
        "Summary: A two-stage pipeline.",
        "Visible text: Stage A",
    ]


def test_http_client_normalizes_json_response(tmp_path: Path) -> None:
    image = tmp_path / "figure.png"
    image.write_bytes(b"png")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/figure-descriptions"
        return httpx.Response(
            200,
            json={
                "description": {
                    "figure_type": "diagram",
                    "summary": "A graph diagram.",
                },
                "model": "mineru-figure",
                "latency_ms": 12,
            },
        )

    client = MinerUFigureClient(
        "http://127.0.0.1:8002",
        transport=httpx.MockTransport(handler),
    )
    result = client.describe(image, title="Paper", section="Method", caption=None)

    assert result.description.summary == "A graph diagram."
    assert result.model == "mineru-figure"
    assert result.latency_ms == 12


def test_http_client_maps_timeout_and_invalid_json(tmp_path: Path) -> None:
    image = tmp_path / "figure.png"
    image.write_bytes(b"png")

    def timeout(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("private details")

    with pytest.raises(FigureDescriptionError) as exc:
        MinerUFigureClient(
            "http://127.0.0.1:8002", transport=httpx.MockTransport(timeout)
        ).describe(image, title="", section=None, caption=None)
    assert exc.value.code == "figure_description_timeout"

    def invalid(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"description": "bad"}).encode())

    with pytest.raises(FigureDescriptionError) as exc:
        MinerUFigureClient(
            "http://127.0.0.1:8002", transport=httpx.MockTransport(invalid)
        ).describe(image, title="", section=None, caption=None)
    assert exc.value.code == "figure_description_invalid_json"
