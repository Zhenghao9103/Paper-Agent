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


def test_setup_checks_every_native_command_exit_code() -> None:
    setup = (ROOT / "setup.ps1").read_text(encoding="utf-8")
    assert "function Invoke-Native" in setup
    assert "if ($LASTEXITCODE -ne 0)" in setup
    assert "& $venvPython" not in setup
    assert "& $figurePython" not in setup
    assert "& $git.Source" not in setup


def test_invoke_native_preserves_stdout_and_throws_on_nonzero(tmp_path: Path) -> None:
    setup = (ROOT / "setup.ps1").read_text(encoding="utf-8")
    function_only = setup.split("$repoRoot =", 1)[0]
    success = tmp_path / "success.ps1"
    success.write_text(
        function_only
        + "$value = Invoke-Native -FilePath $PSHOME\\powershell.exe "
        + "-ArgumentList @('-NoProfile', '-Command', 'Write-Output version-ok; exit 0')\n"
        + "if ($value -ne 'version-ok') { throw 'stdout was not preserved' }\n",
        encoding="utf-8",
    )
    failure = tmp_path / "failure.ps1"
    failure.write_text(
        function_only
        + "Invoke-Native -FilePath $PSHOME\\powershell.exe "
        + "-ArgumentList @('-NoProfile', '-Command', 'exit 9')\n",
        encoding="utf-8",
    )

    assert subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", success],
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0
    failed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", failure],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "exit code 9" in (failed.stdout + failed.stderr)
