"""核心数据模型（对齐 docs/03 §1）。

- Message 同时承载 system/user/assistant/tool 四种角色。
- assistant 消息回写真实 usage 与 stop_reason（对齐 Pi，见 docs/04 §1.6）。
- ToolResult 可带 terminate 主动宣告结束（docs/04 §2.5）。

依赖 pydantic v2。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Role = Literal["system", "user", "assistant", "tool"]

StopReason = Literal[
    "done",
    "length",
    "max_tokens",
    "stop",
    "tool_calls",
    "error",
    "aborted",
]

ResultStopReason = Literal["done", "max_steps", "guardrail", "cancelled", "terminate", "aborted"]


class TokenUsage(BaseModel):
    """单条消息的 token 用量（来自 provider 的真实 usage 回写）。"""

    model_config = ConfigDict(extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON 字符串，执行时解析


class ToolCall(BaseModel):
    id: str  # 本轮唯一，用于回填 ToolResult 配对
    type: Literal["function"] = "function"
    function: ToolCallFunction


class Message(BaseModel):
    role: Role
    content: str | None = None
    # 仅 assistant 消息可能携带 tool_calls
    tool_calls: list[ToolCall] | None = None
    # 仅 tool 消息携带 call_id（与 tool_calls 配对）
    tool_call_id: str | None = None
    name: str | None = None  # tool 消息中为工具名
    # assistant 专属：真实 token 用量回写
    usage: TokenUsage | None = None
    stop_reason: StopReason | None = None

    def to_provider_dict(self) -> dict[str, Any]:
        """转换为 OpenAI 兼容 provider 的单条消息。"""
        if self.role == "tool":
            return {"role": "tool", "tool_call_id": self.tool_call_id, "content": self.content or ""}
        d: dict[str, Any] = {"role": self.role, "content": self.content or ""}
        if self.role == "assistant" and self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return d


class ToolResult(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str  # 执行结果（可能被截断）
    # 工具可主动宣告"任务已完成"：本批全部 terminate 则结束循环（docs/04 §2.5）
    terminate: bool = False
    details: dict[str, Any] = {}  # 结构化结果（如副作用追踪的文件清单）


def tool_result(
    tool_call_id: str,
    content: str,
    *,
    terminate: bool = False,
    details: dict[str, Any] | None = None,
) -> ToolResult:
    """便捷构造 ToolResult。"""
    return ToolResult(
        tool_call_id=tool_call_id,
        content=content,
        terminate=terminate,
        details=details or {},
    )