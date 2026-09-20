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
from agentlab.contracts import (
    CONTRACT_VERSION,
    Artifact,
    Citation,
    CitationStatus,
    ProcessAttempt,
    ProcessStatus,
    Project,
    Provenance,
    RequestScope,
    RetrievalItem,
    RetrievalResult,
    RetrievalScope,
    RetrievalStatus,
    RetrievalStrategy,
    bind_retrieval_scope,
    current_retrieval_scope,
    reset_retrieval_scope,
    Session,
    Source,
    SourceDocument,
    StageResult,
)
from agentlab.runtime.task_state import (
    TaskState,
    TaskStateConflict,
    TaskStateError,
    TaskStateStore,
)
from agentlab.runtime.stages import StageTracker
from agentlab.core.planning import Plan, PlanBuilder, PlanExecutor, PlanStep
from agentlab.memory.invalidation import InvalidationResult, DerivedInvalidationCoordinator
from agentlab.rag.citations import CitationRegistry

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
    "CONTRACT_VERSION",
    "Project",
    "Source",
    "SourceDocument",
    "Session",
    "Artifact",
    "ProcessAttempt",
    "ProcessStatus",
    "StageResult",
    "Citation",
    "CitationStatus",
    "Provenance",
    "RequestScope",
    "RetrievalItem",
    "RetrievalResult",
    "RetrievalScope",
    "RetrievalStatus",
    "RetrievalStrategy",
    "current_retrieval_scope",
    "bind_retrieval_scope",
    "reset_retrieval_scope",
    "TaskState",
    "TaskStateError",
    "TaskStateConflict",
    "TaskStateStore",
    "StageTracker",
    "Plan",
    "PlanStep",
    "PlanBuilder",
    "PlanExecutor",
    "InvalidationResult",
    "DerivedInvalidationCoordinator",
    "CitationRegistry",
]

__version__ = "0.1.0"
