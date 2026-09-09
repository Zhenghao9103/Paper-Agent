from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

MODEL_NAME = "unimernet_hf_small_2503"
MODEL_RELATIVE_PATH = Path("models") / "MFR" / MODEL_NAME


@dataclass(frozen=True)
class FormulaRecognition:
    latex: str | None
    model: str
    latency_ms: int
    status: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "latex": self.latex,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "warnings": list(self.warnings),
        }


class MinerUFormulaRecognizer:
    """Recognize already-cropped equations with MinerU's local MFR model."""

    def __init__(
        self,
        pipeline_model_root: Path | str,
        *,
        model_factory: Callable[[str, str], Any] | None = None,
        device: str = "cpu",
    ) -> None:
        self.pipeline_model_root = Path(pipeline_model_root)
        self.model_path = self.pipeline_model_root / MODEL_RELATIVE_PATH
        self.model_factory = model_factory or self._default_model_factory
        self.device = device
        self._model: Any | None = None

    @staticmethod
    def _default_model_factory(model_path: str, device: str) -> Any:
        from mineru.backend.pipeline.model_init import mfr_model_init

        return mfr_model_init(model_path, device)

    def recognize(self, image_path: Path | str) -> FormulaRecognition:
        started = perf_counter()
        if not self.model_path.is_dir():
            return self._partial(started, "mineru_formula_model_unavailable")
        try:
            image = np.asarray(Image.open(image_path).convert("RGB"))[:, :, ::-1].copy()
            height, width = image.shape[:2]
            if self._model is None:
                self._model = self.model_factory(str(self.model_path), self.device)
            result = self._model.predict(
                [{"bbox": [0, 0, width, height], "label": "display_formula"}],
                image,
                batch_size=1,
                interline_enable=True,
            )
        except Exception:
            return self._partial(started, "formula_recognition_failed")
        latex = ""
        if result and isinstance(result[0], dict):
            latex = str(result[0].get("latex") or "").strip()
        if not latex:
            return self._partial(started, "formula_latex_empty")
        return FormulaRecognition(
            latex=latex,
            model=MODEL_NAME,
            latency_ms=self._elapsed_ms(started),
            status="success",
        )

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, round((perf_counter() - started) * 1000))

    def _partial(self, started: float, warning: str) -> FormulaRecognition:
        return FormulaRecognition(
            latex=None,
            model=MODEL_NAME,
            latency_ms=self._elapsed_ms(started),
            status="partial",
            warnings=(warning,),
        )
