"""Agent 描述符（对齐 docs/03 §2.1 / §2.2bis）。

持有系统指令、工具、终止与插话队列；本身不跑循环（由 Runner 使用）。
"""
from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, PrivateAttr

from agentlab.core.message import Message


class Agent(BaseModel):
    name: str = "agentlab"
    instructions: str  # system prompt（已渲染后的最终文本）
    tools: list = []  # list[Tool]，运行时强转，避免与 pydantic 互相依赖
    model: str | None = None  # 覆盖默认模型（多模型路由用）
    max_steps: int = 15  # 防循环上限
    guardrails: list[Callable] = []  # 输出 guardrail 列表
    needs_plan: bool = False  # 是否先走 Plan-and-Execute
    needs_reflexion: bool = False  # 失败是否自纠

    # —— 运行时插话/追问队列（对齐 docs/04 §1.2）——
    # 私有属性须用 default_factory，避免多实例共享同一个列表。
    _steering: list[Message] = PrivateAttr(default_factory=list)
    _follow_ups: list[Message] = PrivateAttr(default_factory=list)

    def steer(self, msg: Message) -> None:
        """运行中插话：本轮 assistant 回复后、下一次 LLM 调用前注入。"""
        assert msg.role in ("user", "system"), "steer 仅接受 user/system 消息"
        self._steering.append(msg)

    def follow_up(self, msg: Message) -> None:
        """结束注入：agent 自然收尾后仍要追问时使用。"""
        assert msg.role in ("user", "system"), "follow_up 仅接受 user/system 消息"
        self._follow_ups.append(msg)

    def drain_steering(self) -> list[Message]:
        out, self._steering = self._steering, []
        return out

    def drain_follow_ups(self) -> list[Message]:
        out, self._follow_ups = self._follow_ups, []
        return out

    def has_follow_ups(self) -> bool:
        return bool(self._follow_ups)