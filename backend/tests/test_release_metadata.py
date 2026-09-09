from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = (
    "models/router/qwen3-1.7b-router-sft-v3/"
    "Qwen3-1.7B-Router-SFT-V3-Q4_K_M.gguf"
)


def test_gitattributes_tracks_gguf_with_lfs() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.gguf filter=lfs diff=lfs merge=lfs -text" in attributes


def test_readme_uses_full_setup_as_primary_path() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert ".\\setup.ps1" in readme
    assert "20 GB" in readme
    assert "Git LFS" in readme
    assert "AGENT_API_KEY" in readme
    assert "backend.app.evaluation" not in readme


def test_release_uses_mit_license() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("MIT License\n")
    assert "Zhenghao9103" in license_text


def test_release_excludes_runtime_data_and_all_other_models() -> None:
    paths = (
        "docs/private.md",
        "data/sample.json",
        "reports/result.json",
        ".mineru/mineru.json",
        ".tools/llama.cpp/llama-server.exe",
        ".hf-cache/hub/model.bin",
        "models/router/other.gguf",
    )
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *paths],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert set(result.stdout.splitlines()) == set(paths)
    selected = subprocess.run(
        ["git", "check-ignore", "--no-index", MODEL],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert selected.returncode == 1


def test_evaluation_and_post_training_are_absent_from_release() -> None:
    assert not (ROOT / "backend" / "app" / "evaluation").exists()
    assert not (ROOT / "backend" / "app" / "post_training").exists()
