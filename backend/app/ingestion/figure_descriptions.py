from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

FIGURE_TYPES = {"architecture", "flowchart", "plot", "diagram", "photo", "other"}


class FigureDescriptionError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class FigureDescription(BaseModel):
    figure_type: Literal[
        "architecture", "flowchart", "plot", "diagram", "photo", "other"
    ] = "other"
    caption: str = Field(default="", max_length=1000)
    visible_text: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    summary: str = Field(default="", max_length=1500)

    model_config = ConfigDict(extra="ignore")

    @field_validator("caption", "summary", mode="before")
    @classmethod
    def _string_field(cls, value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("must be a string")
        return value.strip()

    @field_validator("visible_text", "components", "relationships", mode="before")
    @classmethod
    def _string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError("must be a list of strings")
        result: list[str] = []
        seen: set[str] = set()
        for raw_item in value:
            item = raw_item.strip()[:300]
            if item and item not in seen:
                result.append(item)
                seen.add(item)
            if len(result) >= 20:
                break
        return result


@dataclass(frozen=True)
class FigureDescriptionResult:
    description: FigureDescription
    model: str
    latency_ms: int
    status: str = "success"
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description.model_dump(),
            "model": self.model,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "warnings": list(self.warnings),
        }


def normalize_figure_description(payload: Any) -> FigureDescription:
    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped.startswith("```"):
            raise FigureDescriptionError("figure_description_invalid_json")
        try:
            payload = json.loads(stripped)
        except (TypeError, json.JSONDecodeError) as exc:
            raise FigureDescriptionError("figure_description_invalid_json") from exc
    if not isinstance(payload, dict):
        raise FigureDescriptionError("figure_description_invalid_json")
    normalized = dict(payload)
    if normalized.get("figure_type") not in FIGURE_TYPES:
        normalized["figure_type"] = "other"
    try:
        return FigureDescription.model_validate(normalized)
    except ValidationError as exc:
        raise FigureDescriptionError("figure_description_invalid_json") from exc


def project_figure_description(description: FigureDescription) -> str:
    lines: list[str] = []
    if description.caption:
        lines.append(description.caption)
    lines.append(f"Figure type: {description.figure_type}")
    for label, values in (
        ("Components", description.components),
        ("Relationships", description.relationships),
    ):
        if values:
            lines.append(f"{label}: {'; '.join(values)}")
    if description.summary:
        lines.append(f"Summary: {description.summary}")
    if description.visible_text:
        lines.append(f"Visible text: {'; '.join(description.visible_text)}")
    return "\n".join(lines)


class MinerUFigureClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def describe(
        self,
        image_path: Path,
        *,
        title: str,
        section: str | None,
        caption: str | None,
    ) -> FigureDescriptionResult:
        try:
            with image_path.open("rb") as image_file, httpx.Client(
                timeout=self.timeout_seconds, transport=self.transport
            ) as client:
                response = client.post(
                    f"{self.base_url}/v1/figure-descriptions",
                    files={"image": (image_path.name, image_file, "image/png")},
                    data={
                        "title": title[:500],
                        "section": (section or "")[:500],
                        "caption": (caption or "")[:1000],
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.TimeoutException as exc:
            raise FigureDescriptionError("figure_description_timeout") from exc
        except (OSError, httpx.HTTPError) as exc:
            raise FigureDescriptionError("mineru_vlm_unavailable") from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise FigureDescriptionError("figure_description_invalid_json") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("description"), dict):
            raise FigureDescriptionError("figure_description_invalid_json")
        description = normalize_figure_description(payload["description"])
        return FigureDescriptionResult(
            description=description,
            model=str(payload.get("model") or "mineru-figure"),
            latency_ms=max(0, int(payload.get("latency_ms") or 0)),
        )
