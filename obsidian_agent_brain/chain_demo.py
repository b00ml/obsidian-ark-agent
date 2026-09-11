"""三链路验证：通过 MCP stdio 调用工具，模拟 dsh agent 的调用方式。

- 输入链路：bili_meta（B站）+ inbox_read_queue（收件箱队列）
- 使用链路：vault_read + vault_graph（主题理解）
- 产出链路：brain_scan + brain_search（全库统计/检索）
只读为主，不污染真实 Vault；记忆往返由单元测试覆盖。
"""
import asyncio
import os
import sys

# Windows GBK 终端打印 emoji/特殊字符会崩溃（项目已知教训）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(PROJECT, ".venv", "Scripts", "python.exe")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


async def call(session, name, args=None):
    r = await session.call_tool(name, args or {})
    return r.content[0].text


async def main():
    params = StdioServerParameters(
        command=PY,
        args=[os.path.join(PROJECT, "obsidian_agent_brain", "mcp_server.py")],
        cwd=PROJECT,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("===== [输入链路] bili_meta =====")
            print(await call(session, "bili_meta", {"bvid": "BV11MAUekEs2"}))

            print("\n===== [输入链路] inbox_read_queue =====")
            print(await call(session, "inbox_read_queue"))

            print("\n===== [使用链路] vault_graph =====")
            print(await call(session, "vault_graph",
                            {"note": "Inbox/rag-10-strategies-总结.md", "depth": 1}))

            print("\n===== [产出链路] brain_search('RAG') =====")
            print(await call(session, "brain_search", {"query": "RAG", "limit": 3}))

            print("\n===== [产出链路] brain_scan =====")
            print(await call(session, "brain_scan"))


if __name__ == "__main__":
    asyncio.run(main())
