from __future__ import annotations

from pathlib import Path

import numpy as np
from backend.app.ingestion.mineru_formula import MinerUFormulaRecognizer
from PIL import Image


class _FakeModel:
    def __init__(self, latex: str = r"x^2+y^2") -> None:
        self.latex = latex
        self.calls = 0

    def predict(self, boxes, image, **_kwargs):
        self.calls += 1
        assert boxes == [
            {"bbox": [0, 0, image.shape[1], image.shape[0]], "label": "display_formula"}
        ]
        return [{**boxes[0], "latex": self.latex}]


def _weights(tmp_path: Path) -> Path:
    root = tmp_path / "模型" / "pipeline"
    (root / "models" / "MFR" / "unimernet_hf_small_2503").mkdir(parents=True)
    return root


def _image(tmp_path: Path) -> Path:
    path = tmp_path / "公式图片.png"
    Image.fromarray(np.full((24, 80, 3), 255, dtype=np.uint8)).save(path)
    return path


def test_recognizer_loads_unicode_path_and_reuses_lazy_model(tmp_path: Path) -> None:
    model = _FakeModel()
    factory_calls: list[tuple[str, str]] = []

    def factory(path: str, device: str):
        factory_calls.append((path, device))
        return model

    recognizer = MinerUFormulaRecognizer(_weights(tmp_path), model_factory=factory)
    first = recognizer.recognize(_image(tmp_path))
    second = recognizer.recognize(_image(tmp_path))

    assert first.latex == r"x^2+y^2"
    assert first.status == "success"
    assert first.model == "unimernet_hf_small_2503"
    assert first.latency_ms >= 0
    assert second.latex == first.latex
    assert len(factory_calls) == 1
    assert model.calls == 2


def test_recognizer_reports_missing_local_model_without_factory(tmp_path: Path) -> None:
    called = False

    def factory(_path: str, _device: str):
        nonlocal called
        called = True

    result = MinerUFormulaRecognizer(tmp_path / "missing", model_factory=factory).recognize(
        _image(tmp_path)
    )

    assert result.status == "partial"
    assert result.latex is None
    assert result.warnings == ("mineru_formula_model_unavailable",)
    assert called is False


def test_recognizer_reports_empty_latex(tmp_path: Path) -> None:
    result = MinerUFormulaRecognizer(
        _weights(tmp_path), model_factory=lambda *_args: _FakeModel("  ")
    ).recognize(_image(tmp_path))

    assert result.status == "partial"
    assert result.latex is None
    assert result.warnings == ("formula_latex_empty",)


def test_recognizer_converts_model_failure_to_stable_warning(tmp_path: Path) -> None:
    def fail(*_args):
        raise RuntimeError("sensitive provider details")

    result = MinerUFormulaRecognizer(_weights(tmp_path), model_factory=fail).recognize(
        _image(tmp_path)
    )

    assert result.status == "partial"
    assert result.latex is None
    assert result.warnings == ("formula_recognition_failed",)
