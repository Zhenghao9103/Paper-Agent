"""Optional MCP v2 stdio wrapper. All execution goes through SandboxService."""

import argparse
import base64
import logging
import threading
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import anyio
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, Field

from .artifacts import default_artifact_root
from .backends.docker import DockerSandboxBackend
from .policy import PolicyViolation, SandboxPolicy
from .schemas import ArtifactInfo, ExecutePythonRequest, ExecutePythonResult
from .service import SandboxService

logger = logging.getLogger(__name__)


class ArtifactData(BaseModel):
    info: ArtifactInfo
    data_base64: str


def create_server(service: SandboxService) -> MCPServer:
    @asynccontextmanager
    async def lifespan(server):
        try:
            recover = getattr(service.backend, "reap_stale", None)
            if recover:
                try:
                    await anyio.to_thread.run_sync(recover, service.policy)
                except Exception:
                    # Discovery/reads remain available; execution retries recovery fail-closed.
                    logger.warning("docker_cleanup_unavailable")
            yield service
        finally:
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(service.close)

    server = MCPServer("Sandbox Runtime", version="0.1.0", lifespan=lifespan)

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, open_world_hint=False))
    async def execute_python(
        code: Annotated[str, Field(min_length=1)],
        inputs: dict[str, Any] | None = None,
    ) -> Annotated[CallToolResult, ExecutePythonResult]:
        """Execute Python in an ephemeral offline container with server-enforced limits.

        Read JSON inputs from /workspace/inputs.json; write PNG/CSV/JSON to
        /workspace/artifacts. Resource settings and host paths are not accepted.
        """
        cancelled = threading.Event()
        request = ExecutePythonRequest(code=code, inputs=inputs or {})
        try:
            result = await anyio.to_thread.run_sync(
                partial(service.execute, request, cancelled=cancelled), abandon_on_cancel=True
            )
        except anyio.get_cancelled_exc_class():
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(service.cancel, cancelled)
            raise
        return CallToolResult(
            content=[TextContent(type="text", text=result.model_dump_json())],
            structured_content=result.model_dump(mode="json"),
            is_error=result.status != "success" or result.cleanup_status == "failed",
        )

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    def get_artifact(artifact_id: str) -> ArtifactData:
        """Read a retained artifact by opaque ID; never accept a filesystem path."""
        try:
            info, data = service.artifacts.read(artifact_id)
        except (FileNotFoundError, PolicyViolation):
            raise ToolError("artifact_missing_expired_or_invalid") from None
        return ArtifactData(info=info, data_base64=base64.b64encode(data).decode("ascii"))

    return server


def main():
    parser = argparse.ArgumentParser(description="Local Sandbox MCP server (stdio only)")
    parser.add_argument("--store", type=Path, default=default_artifact_root())
    parser.add_argument("--image", default=SandboxPolicy.image)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)  # logging's default stream is stderr.
    policy = SandboxPolicy(image=args.image, timeout_seconds=args.timeout)
    backend = DockerSandboxBackend()
    service = SandboxService(backend, policy=policy, artifact_root=args.store)
    try:
        create_server(service).run(transport="stdio")
    finally:
        service.close()


if __name__ == "__main__":
    main()
