"""示例/离线工具（供 P1 CLI 与单测跑通；生产走 brain_tools）。

读权限为主，避免任何副作用，便于 mock / demo。
"""
from __future__ import annotations

from agentlab.tools.base import Tool, tool


@tool(description="回声：原样返回输入（用于连通性自检）", permission="read")
def echo(text: str) -> str:
    return text


@tool(description="两数相加", permission="read")
def add(a: int, b: int) -> int:
    return a + b


@tool(description="内存里的笔记示例库查询（演示用，生产走 vault_search）", permission="read")
def memory_query(keyword: str, limit: int = 5) -> list[dict]:
    notes = {
        "cli": "agentlab CLI 提供 run/repl/tools/eval/trace 子命令。",
        "loop": "ReAct 循环是事件驱动纯函数，支持并行工具、length 截断防护与 terminate。",
        "pi": "Pi Agent 启示：内核克制、能力外置，压缩用结构化增量摘要。",
    }
    hits = [{"note": k, "snippet": v} for k, v in notes.items() if keyword in k or keyword in v]
    return hits[:limit]


DEMO_TOOLS: list[Tool] = [echo, add, memory_query]