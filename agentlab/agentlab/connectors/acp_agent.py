"""外部 ACP Agent 外观（P2-1/OPT-112）：配置驱动 + 惰性启动 + 自愈。

- 会话策略：agent 进程长驻（spawn 一次，跨轮复用），**每轮 consult 新开 ACP
  session**——外部 agent 的会话态不跨问题泄漏（consult 语义 = 无状态问答，
  上下文由主 agent 驱动）。
- 自愈：进程崩/请求超时/连接断开 → 丢弃当前 client，下次 consult 重建；
  跨事件循环（CLI repl 每轮 asyncio.run）检测旧 client 的 loop 已死 → 重建。
- 并发：**同一实例串行化**——ACP 是 session 级串行协议，同一 agent 名被同一轮
  的多个 agent_consult 命中时必须排队，否则会共用一个 session 并发 prompt。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from agentlab.connectors.acp_client import AcpClient, AcpError

_log = logging.getLogger(__name__)

_EMPTY_ANSWER = "（外部 agent 未返回文本内容）"


class ExternalAgent:
    """单个外部 ACP agent：consult(question) -> 回答文本。"""

    def __init__(self, name: str, command: str, args: list[str] | None = None,
                 cwd: str = "", env: dict[str, str] | None = None,
                 timeout: float = 300.0,
                 on_update: Callable[[dict], None] | None = None):
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.cwd = cwd
        self.env = dict(env or {})
        self.timeout = max(1.0, timeout)
        self.on_update = on_update
        self._client: AcpClient | None = None
        self._session_id: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # 同名并发防护：锁按事件循环重建，避免跨 loop 复用已绑定死循环的锁对象。
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    async def consult(self, question: str) -> str:
        """提问一轮；任何 AcpError 先丢弃现场再上抛（下次调用重建）。

        并发语义：同一外部 agent 排队串行——同轮多个 agent_consult 打同一个名字时
        不会在同一 ACP session 上并发 prompt（协议要求 session 级串行）。排队等待
        计入调用方 elapsed，单次请求超时仍由 AcpClient 的 timeout 兜底。
        """
        async with self._lock_for_current_loop():
            try:
                return await self._consult(question)
            except AcpError as e:
                _log.warning("外部 agent %s consult 失败（%s），现场已丢弃待重建",
                             self.name, e)
                await self.close()
                raise

    def _lock_for_current_loop(self) -> asyncio.Lock:
        """取当前事件循环的串行锁；换 loop（CLI 每轮 asyncio.run）时重建。"""
        current = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not current:
            self._lock = asyncio.Lock()
            self._lock_loop = current
        return self._lock

    async def _consult(self, question: str) -> str:
        client = await self._ensure_started()
        if not self._session_id:
            self._session_id = await client.new_session(cwd=self.cwd)
        r = await client.prompt(self._session_id, question)
        text = r["text"].strip()
        if not text:
            _log.info("外部 agent %s 空回答（stopReason=%s, 工具调用 %d 次）",
                      self.name, r["stopReason"], len(r["tool_calls"]))
            return _EMPTY_ANSWER
        return text

    async def _ensure_started(self) -> AcpClient:
        current = asyncio.get_running_loop()
        if (self._client is None or not self._client.is_alive
                or self._loop is not current):  # 旧 loop 的 task/pipe 已随 loop 失效
            if self._client is not None:
                try:
                    await self._client.close()
                except Exception:
                    pass
                self._session_id = None
            client = AcpClient(self.command, self.args, cwd=self.cwd,
                               env=self.env, timeout=self.timeout,
                               on_update=self.on_update)
            await client.start()
            self._client = client
            self._session_id = None
            self._loop = current
        return self._client

    async def close(self) -> None:
        """丢弃现场（下次 consult 重建）。不重置串行锁：close 可能就在持锁路径里
        （AcpError 自愈），重建锁会让等待中的并发调用与新调用各持一把锁。"""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
        self._client = None
        self._session_id = None
        self._loop = None
