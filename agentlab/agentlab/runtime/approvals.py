"""Serve 链路的人在回路审批状态机。

审批不是安全边界本身：工具注册、Bearer 鉴权和 Vault 路径守卫仍在原链路执行。
本模块只负责 risk_based 下的等待/超时/断连拒绝，以及 approval id 幂等解析。
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Approval:
    approval_id: str
    run_id: str
    session_id: str | None
    tool_name: str
    permission: str
    risk_level: str
    prompt: str
    mode: str
    expires_at: float
    status: str = "pending"
    decision: str = ""
    reason: str = ""
    future: asyncio.Future | None = None

    def public(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "permission": self.permission,
            "risk_level": self.risk_level,
            # 当前危险工具（如 ACP consult）可能没有文件目标；字段固定在契约中，
            # 后续 vault_write/diff 只需填值，不需要改变 Ark SSE 解析。
            "target_path": "",
            "diff": "",
            "summary": self.prompt,
            "approval_mode": self.mode,
            "expires_at": self.expires_at,
            "status": self.status,
            "decision": self.decision,
            "reason": self.reason,
        }


class ApprovalManager:
    """单进程 serve 审批存储；同一事件循环内访问，不需要线程锁。"""

    def __init__(self, timeout_seconds: float = 120.0) -> None:
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._items: dict[str, Approval] = {}

    def _prune(self) -> None:
        now = time.time()
        for key, item in list(self._items.items()):
            if item.status != "pending" and item.expires_at + 300 < now:
                self._items.pop(key, None)

    @staticmethod
    def _risk(permission: str) -> str:
        return "high" if permission == "danger" else "medium"

    async def confirm(self, cfg, tool, prompt: str, *, sink, run_id: str,
                      session_id: str | None, signal: asyncio.Event | None) -> bool:
        """执行策略，并在需要时等待 Ark resolve。

        risk_based 下已在 allowlist 的工具继续走策略自动批准；只有未列入 danger
        白名单的 danger 工具进入 HITL。未知 write 仍由 serve_confirm fail-closed 拒绝。
        """
        from agentlab.runtime.serve_auth import serve_confirm

        mode = str(getattr(cfg, "approval_mode", "risk_based") or "risk_based").strip().lower()
        if mode not in {"risk_based", "allow_all"}:
            # 配置错误必须沿 serve_auth 的 fail-closed 语义直接拒绝，不能降级为
            # 人工确认；否则拼写错误会意外扩大危险工具的可执行范围。
            return False
        if mode == "allow_all":
            allowed = serve_confirm(cfg, tool, prompt)
            if not allowed:
                return False
            sink.event({"type": "approval.resolved", "approval_id": "policy",
                        "run_id": run_id, "tool_name": tool.name,
                        "status": "approved", "decision": "policy-approved: allow_all"})
            return True
        if serve_confirm(cfg, tool, prompt):
            return True
        if tool.permission != "danger":
            return False

        self._prune()
        aid = uuid.uuid4().hex
        expires = time.time() + self.timeout_seconds
        item = Approval(
            approval_id=aid, run_id=run_id, session_id=session_id,
            tool_name=tool.name, permission=tool.permission,
            risk_level=self._risk(tool.permission), prompt=prompt,
            mode=mode, expires_at=expires,
            future=asyncio.get_running_loop().create_future(),
        )
        self._items[aid] = item
        sink.event({"type": "approval.requested", **item.public()})

        signal_task = asyncio.create_task(signal.wait()) if signal is not None else None
        try:
            waiters = {item.future}
            if signal_task is not None:
                waiters.add(signal_task)
            done, _ = await asyncio.wait(waiters, timeout=self.timeout_seconds,
                                         return_when=asyncio.FIRST_COMPLETED)
            if item.future in done:
                allowed = bool(item.future.result())
            elif signal_task is not None and signal_task in done:
                self._resolve(item, "deny", "disconnect")
                allowed = False
            else:
                self._resolve(item, "deny", "timeout")
                allowed = False
            sink.event({"type": "approval.resolved", **item.public()})
            return allowed
        finally:
            if signal_task is not None:
                signal_task.cancel()

    def _resolve(self, item: Approval, decision: str, reason: str = "") -> None:
        if item.status != "pending":
            return
        item.decision = decision
        item.reason = reason
        item.status = "approved" if decision == "allow" else "denied"
        if item.future is not None and not item.future.done():
            item.future.set_result(decision == "allow")

    def resolve(self, approval_id: str, decision: str) -> dict[str, Any] | None:
        """解析审批；终态重复提交原样返回，保证 approval id 幂等。"""
        item = self._items.get(approval_id)
        if item is None:
            return None
        if item.status == "pending":
            self._resolve(item, decision)
        return item.public()

    def get(self, approval_id: str) -> dict[str, Any] | None:
        item = self._items.get(approval_id)
        return item.public() if item else None
