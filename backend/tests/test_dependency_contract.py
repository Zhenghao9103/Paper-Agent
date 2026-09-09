from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent


def _active_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _requirement_name(requirement: str) -> str:
    match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
    assert match is not None, f"invalid requirement: {requirement}"
    return match.group(1).lower().replace("_", "-")


def test_root_requirements_delegates_to_backend_runtime_contract() -> None:
    assert _active_lines(PROJECT_ROOT / "requirements.txt") == [
        "-r backend/requirements.txt"
    ]


def test_runtime_contract_covers_release_imports_and_required_pins() -> None:
    requirements = _active_lines(BACKEND_ROOT / "requirements.txt")
    declared = {_requirement_name(requirement) for requirement in requirements}

    required_pins = {
        "mineru==3.4.4",
        "modelscope==1.39.1",
        "huggingface-hub==0.36.2",
        "sentence-transformers==3.3.1",
        "FlagEmbedding==1.4.0",
    }
    assert required_pins <= set(requirements)

    import_to_distribution = {
        "FlagEmbedding": "flagembedding",
        "PIL": "pillow",
        "alembic": "alembic",
        "arxiv": "arxiv",
        "bs4": "beautifulsoup4",
        "chromadb": "chromadb",
        "fastapi": "fastapi",
        "fitz": "pymupdf",
        "httpx": "httpx",
        "huggingface_hub": "huggingface-hub",
        "jieba": "jieba",
        "mineru": "mineru",
        "modelscope": "modelscope",
        "numpy": "numpy",
        "openai": "openai",
        "pydantic": "pydantic",
        "pydantic_settings": "pydantic-settings",
        "sentence_transformers": "sentence-transformers",
        "snowballstemmer": "snowballstemmer",
        "sqlalchemy": "sqlalchemy",
        "sse_starlette": "sse-starlette",
        "tiktoken": "tiktoken",
        "torch": "torch",
        "transformers": "transformers",
        "uvicorn": "uvicorn",
    }
    imported: set[str] = set()
    for path in (BACKEND_ROOT / "app").rglob("*.py"):
        relative = path.relative_to(BACKEND_ROOT / "app")
        if relative.parts[0] in {"evaluation", "post_training"}:
            continue
        if relative.as_posix() in {"agents/research.py", "ingestion/parsers.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".", 1)[0])

    missing_mappings = (
        imported
        - import_to_distribution.keys()
        - sys.stdlib_module_names
        - {"backend", "starlette"}
    )
    assert not missing_mappings, f"unclassified runtime imports: {sorted(missing_mappings)}"
    missing_requirements = {
        import_to_distribution[name]
        for name in imported & import_to_distribution.keys()
        if import_to_distribution[name] not in declared
    }
    assert not missing_requirements, f"undeclared runtime dependencies: {missing_requirements}"
    assert {"langgraph", "python-pptx"}.isdisjoint(declared)


def test_development_contract_extends_runtime_with_tools_only() -> None:
    requirements = _active_lines(BACKEND_ROOT / "requirements-dev.txt")
    assert requirements[0] == "-r requirements.txt"
    assert {_requirement_name(requirement) for requirement in requirements[1:]} == {
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "ruff",
    }


def test_figure_contract_is_isolated_and_cpu_pinned() -> None:
    requirements = _active_lines(BACKEND_ROOT / "requirements-figure.txt")
    assert requirements == [
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        "torch==2.6.0+cpu",
        "torchvision==0.21.0+cpu",
        "transformers==4.57.3",
        "mineru-vl-utils[transformers]==1.0.5",
        "accelerate==1.5.1",
        "numpy==1.26.4",
        "Pillow==11.3.0",
        "fastapi==0.116.1",
        "uvicorn==0.35.0",
        "python-multipart==0.0.20",
    ]
    assert {"langgraph", "python-pptx"}.isdisjoint(
        {_requirement_name(requirement) for requirement in requirements}
    )


def test_pyproject_matches_requirement_contracts() -> None:
    pyproject = tomllib.loads((BACKEND_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = set(_active_lines(BACKEND_ROOT / "requirements.txt"))
    development = set(_active_lines(BACKEND_ROOT / "requirements-dev.txt")[1:])

    assert set(pyproject["project"]["dependencies"]) == runtime
    assert set(pyproject["project"]["optional-dependencies"]["dev"]) == development
