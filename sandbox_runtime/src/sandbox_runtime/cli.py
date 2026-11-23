import argparse
import logging
from pathlib import Path

from .backends.docker import DockerSandboxBackend
from .schemas import ExecutePythonRequest
from .service import SandboxService


def main():
    parser = argparse.ArgumentParser(description="Execute a Python file in an isolated container")
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    service = SandboxService(DockerSandboxBackend())
    try:
        result = service.execute(ExecutePythonRequest(code=args.file.read_text(encoding="utf-8")))
    finally:
        service.close()
    print(result.model_dump_json(indent=2))
    raise SystemExit(0 if result.status == "success" and result.cleanup_status != "failed" else 1)


if __name__ == "__main__":
    main()
