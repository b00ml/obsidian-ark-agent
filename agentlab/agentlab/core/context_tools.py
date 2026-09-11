"""上下文自管理工具对（L10/OPT-106，借鉴 billion-context 的工具化压缩）。

- context_status：用量可观测——tokens/预算/nudge 与硬截断阈值，模型据此决策。
- compress_context：模型主动请求压缩——置 force 标记，loop 下一轮顶执行锚定压缩
  （复用 L9 的 [anchor, cut) 冻结前缀机制），不在工具线程里直接改消息列表
  （loop 持有 messages 局部变量，跨线程改必然撕裂）。

state 盒由 Runner.run 创建并持有（每 run 一个，serve 每请求新建 Runner，无共享）。
"""
from __future__ import annotations

import json

from agentlab.core.context import estimate_tokens
from agentlab.core.loop import RunConfig, Runner
from agentlab.tools.base import tool


class RunContextState:
    """单次 run 的共享状态盒：loop 持有并更新，context 工具经它读写。"""

    def __init__(self) -> None:
        self.messages: list = []
        self.cfg = None
        self.force_compact = False


def build_context_tools(state: RunContextState) -> list:
    @tool(
        name="context_status",
        description="查看当前上下文用量（tokens/预算/占比/nudge 与硬截断阈值/消息数）。"
                    "判断是否需要 compress_context、或回答关于上下文的问题时调用。",
        permission="read",
    )
    def context_status() -> str:
        msgs = state.messages
        est = sum(estimate_tokens(m) for m in msgs)
        cfg = state.cfg
        configured_budget = int(getattr(cfg, "context_budget", 0) or 0)
        effective_budget = Runner._effective_budget(cfg) if isinstance(cfg, RunConfig) else configured_budget
        nudge = float(getattr(state.cfg, "context_nudge", 0.75))
        hard = float(getattr(state.cfg, "context_hard_trim", 0.95))
        return json.dumps({
            "tokens": est,
            "budget": effective_budget,
            "configured_budget": configured_budget,
            "effective_budget": effective_budget,
            "context_window": int(getattr(cfg, "context_window", 0) or 0),
            "usage_pct": round(est * 100.0 / effective_budget, 1) if effective_budget else 0.0,
            "nudge_pct": round(nudge * 100),
            "hard_trim_pct": round(hard * 100),
            "messages": len(msgs),
            "scope": "full_runtime_estimate",
        }, ensure_ascii=False)

    @tool(
        name="compress_context",
        description="立即压缩上下文：把较旧对话区段折叠为检查点摘要（锚定前缀机制，"
                    "保留最近约一半预算与最新用户指令原文）。上下文偏高或完成阶段性"
                    "任务后想腾空间时调用；实际压缩在下一轮执行，用 context_status 复查。",
        permission="read",
    )
    async def compress_context() -> str:
        state.force_compact = True
        return json.dumps({"ok": True,
                           "message": "已请求压缩：下一轮执行，保留最近约一半预算与最新待办指令"},
                          ensure_ascii=False)

    return [context_status, compress_context]
