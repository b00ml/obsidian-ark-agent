"""统一错误模型（对齐 docs/03 §5）。错误码前缀 AGENT_。"""
from __future__ import annotations

from typing import Any


class AgentError(Exception):
    """agentlab 统一运行时错误。code 见 docs/03 §5 表格。"""

    def __init__(self, code: str, message: str, detail: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def __str__(self) -> str:  # 便于 stderr 输出错误码
        return f"[{self.code}] {self.message}"