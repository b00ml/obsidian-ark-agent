"""一次性冒烟测试：通过 stdio 连接 MCP server，列出工具并调用 3 个只读工具。"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(PROJECT, ".venv", "Scripts", "python.exe")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


async def main():
    params = StdioServerParameters(
        command=PY,
        args=[os.path.join(PROJECT, "obsidian_agent_brain", "mcp_server.py")],
        cwd=PROJECT,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"TOOLS={len(tools.tools)}")
            for t in tools.tools:
                print(f"  - {t.name}")
            # 调用三个只读工具验证真实 Vault
            r1 = await session.call_tool("brain_scan", {})
            print("brain_scan =>", r1.content[0].text[:200])
            r2 = await session.call_tool("vault_search", {"keyword": "RAG"})
            print("vault_search(RAG) =>", r2.content[0].text[:200])
            r3 = await session.call_tool("brain_reindex", {})
            print("brain_reindex =>", r3.content[0].text[:120])


if __name__ == "__main__":
    asyncio.run(main())
