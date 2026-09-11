"""外部 Agent 咨询工具（P2-1/OPT-112）：把外部 ACP agent 暴露为 ReAct 可调用 Tool。

- agent_consult(agent, question)：咨询配置过的外部 ACP agent（Claude Code /
  Gemini CLI 等经各自 ACP 适配器），返回其回答文本（read 语义）。
- permission="danger"：spawn 外部 AI agent 属高危动作，默认走 HITL 确认，
  可经 danger_allowlist 放行常驻使用；外部 agent 的 fs/权限反向请求在协议层
  fail-closed 拒绝（见 acp_client），双保险对齐"AI 直写库需确认"红线。

未配置 agents.external 时不注册任何工具（配置化启用，零影响降级）。
"""
from __future__ import annotations

from agentlab.connectors.acp_agent import ExternalAgent
from agentlab.connectors.acp_client import AcpError
from agentlab.tools.base import tool


def build_acp_tools(agents_config=None, vault_root: str = "",
                    factory=None) -> list:
    """按配置构造 agent_consult；无可用外部 agent 时返回空列表。

    - agents_config：runtime.config.AgentsConfig；external 里 command 为空的跳过。
    - vault_root：外部 agent 默认工作目录（spec.cwd 为空时）。
    - factory：测试注入 (spec, default_cwd) -> ExternalAgent 替身。
    """
    specs = [a for a in (getattr(agents_config, "external", None) or [])
             if getattr(a, "command", "")]
    if not specs:
        return []
    agents: dict[str, ExternalAgent] = {}
    for spec in specs:
        default_cwd = spec.cwd or vault_root
        agents[spec.name] = (factory(spec, default_cwd) if factory is not None
                             else ExternalAgent(
                                 spec.name, spec.command, args=spec.args,
                                 cwd=default_cwd, env=spec.env,
                                 timeout=spec.timeout))
    names = "、".join(agents)
    tool_timeout = max(a.timeout for a in specs) + 10  # 内部每请求已限时，此为兜底

    @tool(
        name="agent_consult",
        description=f"咨询外部 ACP agent（可用：{names}）。把问题原文交给对应"
                    "外部 agent 并取回其回答文本；适合需要第二意见/专长模型的"
                    "场景。agent 名必须取自可用列表。",
        permission="danger",
        execution_timeout=tool_timeout,
    )
    async def agent_consult(agent: str, question: str) -> str:
        ext = agents.get(agent)
        if ext is None:
            return f"[未配置的外部 agent] {agent}（可用：{names}）"
        try:
            return await ext.consult(question)
        except AcpError as e:
            return f"[外部 agent 不可用] {agent}: {e}"

    return [agent_consult]


ACP_TOOLS = ["agent_consult"]
