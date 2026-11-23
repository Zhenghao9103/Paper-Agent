"""Independent real-Docker MCP demo; run with the runtime's Python environment."""

import asyncio
import sys

from mcp import Client, StdioServerParameters


async def main():
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "sandbox_runtime.mcp_server"]
    )
    async with Client(params, read_timeout_seconds=45) as client:
        print([tool.name for tool in (await client.list_tools()).tools])
        result = await client.call_tool(
            "execute_python", {"code": "import numpy as np; print(float(np.mean([1,2,3])))"}
        )
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
