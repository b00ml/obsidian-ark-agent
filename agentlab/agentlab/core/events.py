"""Runner 事件契约（C2 收官）：Ev 枚举 + 每事件一个 pydantic payload。

事件名与 payload 字段即契约：emit 端字段拼错/多传在构造时立刻 ValidationError
（EventPayload extra="forbid"），事件与 payload 类型不匹配在 emit 时 TypeError，
消费端拿属性访问——杜绝旧 `**kw` 裸字典下"拼错键静默取默认值"（如 cli tracer
的 step 曾因裸 dict 缺键恒取 0）。

Ev 继承 str：`runner.on("message_end", h)` 按字符串订阅仍兼容（str enum 同哈希）。
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from agentlab.core.message import TokenUsage


class Ev(str, Enum):
    AGENT_START = "agent_start"
    TURN_START = "turn_start"
    MESSAGE_START = "message_start"
    MESSAGE_UPDATE = "message_update"
    MESSAGE_END = "message_end"
    TOOL_START = "tool_execution_start"
    TOOL_END = "tool_execution_end"
    TURN_END = "turn_end"
    AGENT_END = "agent_end"


class EventPayload(BaseModel):
    """所有事件 payload 基类：extra="forbid" 让 emit 端拼错字段立刻炸，而非静默丢弃。"""

    model_config = {"extra": "forbid"}


class AgentStart(EventPayload):
    agent: str
    user_input: str


class TurnStart(EventPayload):
    pass


class MessageStart(EventPayload):
    role: str = "assistant"


class MessageUpdate(EventPayload):
    content: str = ""  # 流式增量片段


class MessageEnd(EventPayload):
    role: str
    content: str | None = None
    usage: TokenUsage | None = None
    step: int = 0  # 本回合步数（_loop 计数器实值，供 tracer/进度展示）


class ToolStart(EventPayload):
    name: str
    arguments: str = ""


class ToolEnd(EventPayload):
    name: str
    result: str | None = None


class TurnEnd(EventPayload):
    pass


class AgentEnd(EventPayload):
    stop_reason: str


# 事件 → payload 类型的唯一映射：emit 时校验"这个事件只发它的 payload"
PAYLOAD_TYPES: dict[Ev, type[EventPayload]] = {
    Ev.AGENT_START: AgentStart,
    Ev.TURN_START: TurnStart,
    Ev.MESSAGE_START: MessageStart,
    Ev.MESSAGE_UPDATE: MessageUpdate,
    Ev.MESSAGE_END: MessageEnd,
    Ev.TOOL_START: ToolStart,
    Ev.TOOL_END: ToolEnd,
    Ev.TURN_END: TurnEnd,
    Ev.AGENT_END: AgentEnd,
}
