"""agentlab：自研轻量可复用 Python Agent 框架。

Agent = LLM + Tools + Loop。本框架强调内核克制、能力外置：
循环纯函数化（只发事件不碰 UI），工具/记忆/技能全部通过钩子与注册表注入。
"""
from agentlab.core.message import (
    Message,
    Role,
    ToolCall,
    ToolCallFunction,
    ToolResult,
    TokenUsage,
)
from agentlab.core.agent import Agent
from agentlab.core.loop import AgentResult, Runner
from agentlab.core.llm import LLMProvider, LLMResponse, OpenAICompatProvider
from agentlab.tools.base import Tool, ExecutionMode, ToolPermission, tool

__all__ = [
    "Message",
    "Role",
    "ToolCall",
    "ToolCallFunction",
    "ToolResult",
    "TokenUsage",
    "Agent",
    "AgentResult",
    "Runner",
    "LLMProvider",
    "LLMResponse",
    "OpenAICompatProvider",
    "Tool",
    "ExecutionMode",
    "ToolPermission",
    "tool",
]

__version__ = "0.1.0"