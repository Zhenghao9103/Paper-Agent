from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .ingestion.figure_descriptions import (
    FigureDescriptionError,
    normalize_figure_description,
)
from .services.mineru_figure_runtime import load_figure_model_manifest


class FigureInferenceBackend(Protocol):
    model_name: str

    def ready(self) -> bool: ...

    def describe(self, image: bytes, **metadata: str) -> Any: ...


class TransformersFigureBackend:
    """Lazy, local-only MinerU2.5 backend used only by the standalone process."""

    def __init__(self, manifest_path: Path, mineru_root: Path) -> None:
        self.manifest = load_figure_model_manifest(manifest_path, mineru_root)
        self.model_name = self.manifest.model_id
        self._runtime: tuple[Any, Any] | None = None
        self._lock = threading.Lock()

    def ready(self) -> bool:
        return self.manifest.local_path.is_dir()

    def _load(self) -> tuple[Any, Any]:
        if self._runtime is not None:
            return self._runtime
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(self.manifest.local_path),
            dtype="auto",
            device_map="auto",
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(
            str(self.manifest.local_path), use_fast=True, local_files_only=True
        )
        self._runtime = model, processor
        return self._runtime

    def describe(self, image: bytes, **metadata: str) -> Any:
        from io import BytesIO

        from PIL import Image

        prompt = (
            "Return exactly one JSON object with keys figure_type, caption, visible_text, "
            "components, relationships, summary. figure_type must be architecture, "
            "flowchart, plot, diagram, photo, or other. Arrays contain short visible facts. "
            "Do not invent results. No Markdown fences.\n"
            f"Paper: {metadata.get('title', '')}\nSection: {metadata.get('section', '')}\n"
            f"Source caption: {metadata.get('caption', '')}"
        )
        with self._lock:
            model, processor = self._load()
            source_image = Image.open(BytesIO(image)).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": source_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            rendered = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[rendered], images=[source_image], padding=True, return_tensors="pt"
            )
            inputs = inputs.to(model.device)
            generated = model.generate(**inputs, max_new_tokens=900, do_sample=False)
            trimmed = [
                output[len(input_ids) :]
                for input_ids, output in zip(inputs.input_ids, generated, strict=True)
            ]
            return processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]


def create_app(backend: FigureInferenceBackend) -> FastAPI:
    app = FastAPI(title="PaperMind MinerU Figure Service")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ready" if backend.ready() else "unavailable",
            "model": backend.model_name,
        }

    @app.post("/v1/figure-descriptions")
    async def describe(
        image: UploadFile = File(...),
        title: str = Form(default=""),
        section: str = Form(default=""),
        caption: str = Form(default=""),
    ) -> dict[str, Any]:
        started = perf_counter()
        try:
            raw = backend.describe(
                await image.read(),
                title=title[:500],
                section=section[:500],
                caption=caption[:1000],
            )
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise FigureDescriptionError("figure_description_invalid_json") from exc
            description = normalize_figure_description(raw)
        except FigureDescriptionError as exc:
            raise HTTPException(status_code=502, detail=exc.code) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="figure_description_failed") from exc
        return {
            "description": description.model_dump(),
            "model": backend.model_name,
            "latency_ms": max(0, round((perf_counter() - started) * 1000)),
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    root = Path(os.environ["MODELSCOPE_CACHE"]).resolve().parent
    manifest = Path(os.environ["MINERU_FIGURE_MANIFEST"])
    backend = TransformersFigureBackend(manifest, root)
    import uvicorn

    uvicorn.run(create_app(backend), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
