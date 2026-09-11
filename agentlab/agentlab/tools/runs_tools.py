"""Agent 自省工具（#10①/OPT-135）：回答"今天/最近做了什么"这类 episodic 问题。

数据源复用 OPT-126 的 run 摘要 trace（serve 每请求落一条）。与语义记忆
（memory_query）互补：run 日志答"发生过什么"（按时间），记忆答"学到过什么"
（按主题）。实测暴露的缺口：用户问"今天做了什么"，agent 只能去 vault 搜
日报（最新一份还是 8-24 的）——会话级事件没有数据通路，此工具补上。
"""
from __future__ import annotations

from datetime import datetime, timedelta

from agentlab.tools.base import Tool, tool


def build_runs_tools(trace_dir: str) -> list[Tool]:
    """注册 runs_recent（read 权限）。trace_dir 不可读/无记录 → 空结果，不抛错。"""

    @tool(description="查看最近 N 天的 agent 运行记录（时间/输入摘要/结果/token/工具次数）。"
                      "适合回答「今天做了什么」「最近在忙什么」这类按时间的回顾问题；"
                      "主题性的经验回顾请改用 memory_query。")
    def runs_recent(days: int = 1, limit: int = 20) -> dict:
        from agentlab.runtime.trace import load_run_summaries

        days = max(1, min(int(days), 90))
        limit = max(1, min(int(limit), 100))
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        runs = [r for r in load_run_summaries(trace_dir, limit=limit * 3)
                if (r.get("time") or "") >= cutoff][:limit]
        return {"days": days, "total": len(runs), "runs": runs}

    return [runs_recent]
