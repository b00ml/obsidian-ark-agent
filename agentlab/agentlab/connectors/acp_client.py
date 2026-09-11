"""ACP（Agent Client Protocol）最小 stdio 客户端（P2-1/OPT-112，学 Zed ACP）。

协议面（newline 分帧 JSON-RPC 2.0，stdout/stdin 各一行一条消息）：
- client → agent 请求：`initialize` 握手（契约校验）、`session/new`、`session/prompt`；
- agent → client 通知：`session/update`——agent_message_chunk 收集为回答正文，
  tool_call/tool_call_update 记录并透传 on_update，其余变体忽略；
- agent → client 反向请求：`session/request_permission` 与 `fs/*` 一律
  **fail-closed 拒绝**——外部 agent 不经我方 HITL 不得触达本地文件/工具授权
  （红线：AI 直写库既有防线不因接入外部 agent 而旁路）。

零新增依赖。进程生命周期由调用方（ExternalAgent）管理；任一请求超时/断开抛
`AcpError`，由调用方降级（工具层包装为可读结果，不中断主循环）。
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

_PROTOCOL_VERSION = 1
# 读循环对 pending future 的善后宽限（秒）：进程死后 fut 统一以断开错误收场
_CLEANUP_GRACE = 5.0


class AcpError(RuntimeError):
    """ACP 协议/子进程失败（启动、超时、断开、契约不符统一抛此类型）。"""


def _framed(msg: dict) -> bytes:
    return (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")


def _raw_popen(proc: asyncio.subprocess.Process) -> subprocess.Popen | None:
    """Return the loop-independent Popen owned by an asyncio subprocess."""
    transport = getattr(proc, "_transport", None)
    if transport is None:
        return None
    getter = getattr(transport, "get_extra_info", None)
    if getter is not None:
        try:
            raw = getter("subprocess")
            if raw is not None:
                return raw
        except Exception:
            pass
    raw = getattr(transport, "_proc", None)
    return raw if isinstance(raw, subprocess.Popen) else None


def _terminate_and_wait(raw: subprocess.Popen, timeout: float) -> int | None:
    """Terminate and reap a Popen from a worker thread (safe across event loops)."""
    if raw.poll() is not None:
        return raw.returncode
    try:
        raw.terminate()
    except (OSError, ProcessLookupError):
        pass
    try:
        return raw.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            raw.kill()
        except (OSError, ProcessLookupError):
            pass
        try:
            return raw.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return raw.poll()


def _close_transport(proc: asyncio.subprocess.Process,
                     raw: subprocess.Popen | None = None, *, close_raw: bool = False) -> None:
    """Best-effort close of pipes/transport, including an already-closed loop."""
    raw = raw or _raw_popen(proc)
    # On the owner loop the asyncio transport owns these stream handles. Closing
    # them through Popen as well races its connection_lost callbacks on Windows
    # and can leave the temporary cwd locked. The raw handles are only needed
    # for cross-loop cleanup, where the owner loop can no longer run callbacks.
    if close_raw and raw is not None:
        for stream in (raw.stdin, raw.stdout, raw.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
    transport = getattr(proc, "_transport", None)
    if transport is None:
        return
    try:
        transport.close()
    except Exception:
        # A transport whose owner loop is already closed can reject call_soon;
        # marking it closed still prevents BaseSubprocessTransport.__del__ from
        # reporting an unclosed transport after the OS process was reaped.
        try:
            transport._closed = True
        except Exception:
            pass
    try:
        transport._closed = True
    except Exception:
        pass
    # A Proactor pipe's close() schedules connection_lost on its owner loop.
    # That loop is already gone in the cross-loop case, so close the underlying
    # handles and mark the transport finalized directly to silence __del__.
    for proto in (getattr(transport, "_pipes", {}) or {}).values():
        pipe = getattr(proto, "pipe", None)
        if pipe is None:
            continue
        try:
            sock = getattr(pipe, "_sock", None)
            if sock is not None:
                sock.close()
            pipe._sock = None
            pipe._closing = True
            pipe._called_connection_lost = True
            pipe._read_fut = None
            pipe._write_fut = None
        except Exception:
            pass
    try:
        transport._pipes = {}
        transport._proc = None
        transport._protocol = None
        transport._finished = True
    except Exception:
        pass


class AcpClient:
    """单个外部 agent 子进程的 ACP 会话通道。

    同一进程内的 prompt 用锁串行（session/update 流无法按并发请求分帐，
    lane 内严格串行）；跨进程并发由 ExternalAgent 多实例承担。
    """

    def __init__(self, command: str, args: list[str] | None = None, cwd: str = "",
                 env: dict[str, str] | None = None, timeout: float = 300.0,
                 on_update: Callable[[dict], None] | None = None):
        self.command = command
        self.args = list(args or [])
        self.cwd = cwd
        self.env = dict(env or {})
        self.timeout = max(1.0, timeout)
        self.on_update = on_update  # session/update 原始 update 透传（进度/观测）
        self._proc: asyncio.subprocess.Process | None = None
        self._raw_proc: subprocess.Popen | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._reader: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self.agent_capabilities: dict = {}
        self._owner_loop: asyncio.AbstractEventLoop | None = None

    # —— 生命周期 ——
    def _resolve_spawn(self) -> tuple[str, list[str]]:
        """Windows 兼容：npm 全局包只装 .cmd shim，CreateProcess 不做 PATHEXT
        解析也不认无扩展名命令——显式 which 定位；.cmd/.bat 经 `cmd /c` 起进程。
        其他平台/显式路径原样返回。"""
        if os.name == "nt" and not Path(self.command).suffix \
                and not Path(self.command).drive:
            found = shutil.which(self.command)
            if found:
                if found.lower().endswith((".cmd", ".bat")):
                    return "cmd", ["/c", found, *self.args]
                return found, self.args
        return self.command, self.args

    async def start(self) -> None:
        """spawn 子进程 + initialize 握手（响应契约不符即判启动失败）。"""
        if self.is_alive:
            return
        exe, prefix_args = self._resolve_spawn()
        try:
            self._proc = await asyncio.create_subprocess_exec(
                exe, *prefix_args,
                cwd=self.cwd or None,
                env={**os.environ, **self.env},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,  # agent 自身日志走 stderr，不混协议流
            )
            self._raw_proc = _raw_popen(self._proc)
        except (OSError, ValueError) as e:
            raise AcpError(f"ACP agent 启动失败: {self.command}: {e}") from e
        self._pending.clear()
        self._reader = asyncio.create_task(self._read_loop())
        self._owner_loop = asyncio.get_running_loop()
        try:
            resp = await self._request("initialize", {
                "protocolVersion": _PROTOCOL_VERSION, "clientCapabilities": {}})
        except AcpError:
            await self.close()  # 握手失败：现场收掉，进程不留孤儿
            raise
        if not isinstance(resp, dict):
            await self.close()
            raise AcpError("initialize 响应契约不符（非对象）")
        self.agent_capabilities = resp.get("agentCapabilities") or {}

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def close(self) -> None:
        """关 stdin → terminate → 宽限后 kill；读循环取消，pending 全部以断开收场。

        CLI repl 每轮 `asyncio.run` 会更换事件循环。进程的 asyncio wait future
        属于启动它的旧 loop，跨 loop 不能直接 await；此时改用底层 Popen 的
        OS wait，并显式关闭旧 transport，确保 Windows 不留下句柄/ResourceWarning。
        """
        proc, self._proc = self._proc, None
        owner_loop = self._owner_loop
        self._owner_loop = None
        reader = self._reader
        self._reader = None
        if reader is not None:
            reader.cancel()
            if reader.get_loop() is asyncio.get_running_loop():
                try:
                    await asyncio.gather(reader, return_exceptions=True)
                except Exception:
                    pass
            else:
                # A cancelled Task on a closed loop is never scheduled to
                # deliver CancelledError. Close its coroutine frame directly
                # so it no longer retains Process/pipe transport references.
                try:
                    reader.get_coro().close()
                    reader._log_destroy_pending = False
                except Exception:
                    pass
        if proc is not None:
            current_loop = asyncio.get_running_loop()
            cross_loop = current_loop is not owner_loop
            if proc.returncode is None:
                try:
                    if proc.stdin:
                        proc.stdin.close()
                except Exception:
                    pass
                try:
                    proc.terminate()
                except (ProcessLookupError, RuntimeError):
                    pass
                if not cross_loop:
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=_CLEANUP_GRACE)
                    except asyncio.TimeoutError:
                        try:
                            proc.kill()
                        except ProcessLookupError:
                            pass
                        try:
                            await asyncio.wait_for(proc.wait(), timeout=_CLEANUP_GRACE)
                        except (asyncio.TimeoutError, ProcessLookupError):
                            pass
                else:
                    # The asyncio Process is loop-bound. Waiting on its underlying
                    # Popen object is loop-independent and works after the old loop
                    # has already been closed.
                    raw = self._raw_proc or _raw_popen(proc)
                    if raw is not None:
                        await asyncio.to_thread(_terminate_and_wait, raw, _CLEANUP_GRACE)
                    else:
                        # Fallback for alternate event-loop implementations without
                        # a discoverable Popen object: do not block this loop.
                        try:
                            proc.kill()
                        except (ProcessLookupError, RuntimeError):
                            pass
            _close_transport(proc, self._raw_proc, close_raw=cross_loop)
            if not cross_loop:
                # Let the owner loop run the transport's pipe-close callbacks
                # before callers remove a temporary working directory.
                await asyncio.sleep(0)
        self._raw_proc = None
        self._fail_pending(AcpError("ACP 连接已关闭"))

    def _fail_pending(self, err: Exception) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(err)
        self._pending.clear()

    # —— 协议读写 ——
    async def _read_loop(self) -> None:
        proc = self._proc
        try:
            while proc is not None:
                line = await proc.stdout.readline()
                if not line:
                    break  # EOF：agent 进程退出
                await self._handle_line(line)
        except asyncio.CancelledError:
            return
        except Exception:
            pass  # 读循环异常（进程崩溃等）→ finally 统一收场
        finally:
            self._fail_pending(AcpError("ACP agent 连接断开（进程退出/崩溃）"))

    async def _handle_line(self, line: bytes) -> None:
        s = line.decode("utf-8", errors="replace").strip()
        if not s:
            return
        try:
            msg = json.loads(s)
        except ValueError:
            return  # 非 JSON 行（启动 banner 等）忽略——协议流里容错不放大
        method, rid = msg.get("method"), msg.get("id")
        if method is not None and rid is not None:  # agent→client 请求（反向）
            await self._handle_agent_request(method, msg.get("params") or {}, rid)
        elif method is not None:  # 通知
            if method == "session/update" and self.on_update:
                update = (msg.get("params") or {}).get("update")
                if update is not None:
                    try:
                        self.on_update(update)
                    except Exception:
                        pass  # 观测回调异常不影响协议流
        elif rid is not None:  # 我方请求的响应
            fut = self._pending.pop(rid, None)
            if fut is None:
                return  # 超时后迟到的响应：丢弃
            if "error" in msg:
                fut.set_exception(AcpError(f"agent 返回错误: {msg['error']}"))
            else:
                fut.set_result(msg.get("result"))

    async def _handle_agent_request(self, method: str, params: dict, rid: int) -> None:
        """反向请求 fail-closed：权限请求自动选 reject 项，fs/其他一律协议错误拒绝。"""
        if method == "session/request_permission":
            reject = next((str(o.get("optionId")) for o in (params.get("options") or [])
                           if str(o.get("kind", "")).startswith("reject")), None)
            if reject is not None:
                await self._respond(rid, {"outcome": {"outcome": "selected",
                                                      "optionId": reject}})
                return
        await self._respond_error(rid, -32601,
                                  f"client fail-closed: {method} 被拒（外部 agent "
                                  f"不经 HITL 不得触达本地文件/授权）")

    async def _respond(self, rid: int, result: dict) -> None:
        await self._send({"jsonrpc": "2.0", "id": rid, "result": result})

    async def _respond_error(self, rid: int, code: int, message: str) -> None:
        await self._send({"jsonrpc": "2.0", "id": rid,
                          "error": {"code": code, "message": message}})

    async def _send(self, msg: dict) -> None:
        proc = self._proc
        if proc is None or proc.returncode is not None or proc.stdin is None:
            raise AcpError("ACP agent 未启动或已退出")
        try:
            proc.stdin.write(_framed(msg))
            await proc.stdin.drain()
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            raise AcpError(f"ACP 写入失败（进程可能已退出）: {e}") from e

    async def _request(self, method: str, params: dict) -> object:
        self._next_id += 1
        rid = self._next_id
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[rid] = fut
        try:
            await self._send({"jsonrpc": "2.0", "id": rid,
                              "method": method, "params": params})
            return await asyncio.wait_for(fut, timeout=self.timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise AcpError(f"ACP 请求超时({self.timeout:g}s): {method}") from None
        except AcpError:
            self._pending.pop(rid, None)
            raise

    # —— 会话面 ——
    async def new_session(self, cwd: str = "") -> str:
        resp = await self._request("session/new", {
            "cwd": cwd or self.cwd or os.getcwd(), "mcpServers": []})
        sid = resp.get("sessionId") if isinstance(resp, dict) else None
        if not sid:
            raise AcpError("session/new 响应缺 sessionId（契约不符）")
        return str(sid)

    async def prompt(self, session_id: str, text: str) -> dict:
        """发送一轮提问；收集 agent_message_chunk 为回答正文。锁内串行。

        返回 {stopReason, text, tool_calls}；空正文不视为错误（调用方定夺）。
        """
        collected: dict = {"text": [], "tool_calls": []}
        user_cb = self.on_update

        def _sink(update: dict) -> None:
            kind = str(update.get("sessionUpdate", ""))
            if kind == "agent_message_chunk":
                c = update.get("content") or {}
                if str(c.get("type", "text")) == "text":
                    collected["text"].append(str(c.get("text", "")))
            elif kind in ("tool_call", "tool_call_update"):
                collected["tool_calls"].append(update)
            if user_cb is not None:
                try:
                    user_cb(update)
                except Exception:
                    pass

        async with self._lock:
            prev, self.on_update = self.on_update, _sink
            try:
                result = await self._request("session/prompt", {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": text}]})
            finally:
                self.on_update = prev
        stop = result.get("stopReason", "") if isinstance(result, dict) else ""
        return {"stopReason": str(stop), "text": "".join(collected["text"]),
                "tool_calls": collected["tool_calls"]}
