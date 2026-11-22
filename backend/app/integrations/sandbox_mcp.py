"""Optional, bounded MCP client. Never imports Docker or invokes generated code."""

import asyncio
import base64
import os
import sys
from importlib.metadata import version

from ..schemas.compute import ComputeArtifact, ComputeCode, SandboxExecution


class SandboxUnavailable(RuntimeError):
    pass


def _server_environment():
    names = (
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "LOCALAPPDATA",
        "USERPROFILE",
        "HOME",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
    )
    return {key: os.environ[key] for key in names if key in os.environ}


class SandboxMCPAdapter:
    def __init__(self, settings):
        self.settings = settings

    async def _call(self, tool, arguments):
        if not self.settings.sandbox_enabled:
            raise SandboxUnavailable("sandbox_disabled")
        try:
            if version("mcp") != "2.3.0":
                raise SandboxUnavailable("mcp_version_mismatch")
            from mcp import Client, StdioServerParameters

            execution_timeout = min(30.0, max(0.1, self.settings.sandbox_timeout_seconds - 8.0))
            args = ["-m", "sandbox_runtime.mcp_server", "--timeout", str(execution_timeout)]
            if self.settings.sandbox_artifact_store:
                args.extend(["--store", str(self.settings.sandbox_artifact_store)])
            params = StdioServerParameters(
                command=self.settings.sandbox_python_executable or sys.executable,
                args=args,
                env=_server_environment(),
            )
            async with asyncio.timeout(self.settings.sandbox_timeout_seconds):
                async with Client(
                    params, read_timeout_seconds=self.settings.sandbox_timeout_seconds
                ) as client:
                    tools = (await client.list_tools()).tools
                    if tool not in {item.name for item in tools}:
                        raise SandboxUnavailable("sandbox_tool_missing")
                    result = await client.call_tool(tool, arguments)
                    if not isinstance(result.structured_content, dict):
                        raise SandboxUnavailable("sandbox_tool_error")
                    return result
        except SandboxUnavailable:
            raise
        except Exception:  # noqa: BLE001 - normalize transport and SDK errors
            raise SandboxUnavailable("mcp_unavailable") from None

    def execute(self, code: str, inputs: dict) -> SandboxExecution:
        validated = ComputeCode(code=code, inputs=inputs)
        if not validated.code.strip() or len(code.encode()) > 65536:
            raise SandboxUnavailable("invalid_code")
        result = asyncio.run(self._call("execute_python", {"code": code, "inputs": inputs}))
        try:
            execution = SandboxExecution.model_validate(result.structured_content)
            expected_error = execution.status != "success" or execution.cleanup_status == "failed"
            if bool(result.is_error) != expected_error:
                raise ValueError("contradictory_result")
            if execution.status == "success" and execution.exit_code != 0:
                raise ValueError("invalid_exit_code")
            return execution
        except ValueError:
            raise SandboxUnavailable("invalid_sandbox_result") from None

    def read_artifact(self, artifact_id: str) -> tuple[ComputeArtifact, bytes]:
        result = asyncio.run(self._call("get_artifact", {"artifact_id": artifact_id}))
        try:
            data = result.structured_content
            info = ComputeArtifact.model_validate(data["info"])
            encoded = data["data_base64"]
            if result.is_error or info.artifact_id != artifact_id or len(encoded) > 13981016:
                raise ValueError("invalid_artifact")
            content = base64.b64decode(encoded, validate=True)
            if len(content) != info.size_bytes:
                raise ValueError("invalid_artifact")
            return info, content
        except (ValueError, KeyError, TypeError):
            raise SandboxUnavailable("artifact_unavailable") from None
