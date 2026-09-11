"""工具注册表（对齐 docs/03 §3.2、04 §2.1-2.4）。

- register/get/unregister；active(names) 实现 defer 注入（控 token）。
- execute：权限判定 + 参数解析 + 执行 + 异常包装为结果字符串。
- 执行模型：同步函数一律丢线程池（to_thread）避免阻塞事件循环；支持 per-tool 超时
  （超时不杀线程，仅调用方不再等待并回报"[工具执行超时]"）与进度回调（长任务可见）。
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import time
from typing import Awaitable, Callable

from agentlab.core.errors import AgentError
from agentlab.core.message import Message, ToolResult, tool_result
from agentlab.tools.base import Tool

ConfirmFn = Callable[[Tool, str], bool | Awaitable[bool]]
ProgressFn = Callable[[str, float], None]  # (tool_name, elapsed_seconds)

HEARTBEAT_INTERVAL = 2.0  # 心跳间隔（秒）；测试可覆盖


async def _heartbeat(name: str, fut: asyncio.Future, progress: ProgressFn,
                     interval: float = HEARTBEAT_INTERVAL) -> None:
    """长任务心跳：定期向 progress 上报 elapsed，直到工作 future 完成/取消。

    用于长任务（如 bili 转写可达数分钟）的"是否仍在运行"可观测性——
    即使工具内部不暴露真实阶段，CLI 也能持续看到存活心跳。
    """
    t0 = time.monotonic()
    while not fut.done():
        await asyncio.sleep(interval)
        if not fut.done():
            progress(name, time.monotonic() - t0)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, t: Tool) -> None:
        if t.name in self._tools:
            raise AgentError("AGENT_TOOL_DUP", f"工具重复注册：{t.name}")
        self._tools[t.name] = t

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        t = self._tools.get(name)
        if t is None:
            raise AgentError("AGENT_TOOL_NOT_FOUND", f"模型调用了未注册工具：{name}")
        return t

    def schemas(self, names: list[str] | None = None) -> list[dict]:
        tools = self.active(names)
        return [t.schema for t in tools if not t.disable_model_invocation]

    def active(self, names: list[str] | None = None) -> list[Tool]:
        """defer 注册：仅返回 names 子集，控制注入进 prompt 的工具定义体积。"""
        if names is None:
            return list(self._tools.values())
        return [self._tools[n] for n in names if n in self._tools]

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def _parse_args(self, t: Tool, arguments: str) -> dict:
        try:
            raw = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as e:
            raise AgentError("AGENT_TOOL_ARG", f"工具 {t.name} 参数 JSON 解析失败：{e}") from e
        if not isinstance(raw, dict):
            raise AgentError("AGENT_TOOL_ARG", f"工具 {t.name} 参数须为 JSON 对象")
        if t.prepare_arguments is not None:
            raw = t.prepare_arguments(raw)
        return raw

    async def _authorize(self, t: Tool, confirm: ConfirmFn | None) -> None:
        """非 read 权限必须经 confirm 确认，否则 AGENT_TOOL_PERMISSION（fail-closed）。"""
        if t.permission == "read":
            return
        if confirm is None:
            raise AgentError("AGENT_TOOL_PERMISSION",
                             f"无可用的确认机制，拒绝执行 {t.name}（{t.permission}）")
        r = confirm(t, f"即将执行写/危险工具 {t.name}（{t.permission}），是否继续？")
        ok = r if isinstance(r, bool) else await r
        if not ok:
            raise AgentError("AGENT_TOOL_PERMISSION", f"用户拒绝执行 {t.name}")

    @staticmethod
    def _dispatch(t: Tool, args: dict) -> asyncio.Future:
        """生成工作 future：async 函数原样跑（可被取消）；同步函数丢线程池（不可中途取消）。"""
        if inspect.iscoroutinefunction(t.fn):
            return asyncio.ensure_future(t.fn(**args))
        return asyncio.ensure_future(asyncio.to_thread(t.fn, **args))

    async def _await_result(self, t: Tool, fut: asyncio.Future,
                            progress: ProgressFn | None,
                            signal: asyncio.Event | None = None) -> tuple[str, bool, bool]:
        """等待结果：心跳/取消过程透明 + 超时兜底；返回 (content, timed_out, terminate)。

        超时**不杀后台线程**：仅调用方不再等待并回报"[仍在后台运行]"，
        避免长任务（如 bili 转写数分钟）被误杀而半途重复执行。

        取消信号只停止 Agent 编排，不承诺强杀同步线程。asyncio 工具会收到取消，
        `asyncio.to_thread` 包装的同步工具则放弃等待并在后台完成；这保证停止后
        不会进入下一轮 LLM/tool 调用，同时不伪造线程已被杀死。
        """
        hb = None
        cancel_wait = None
        if progress is not None:
            hb = asyncio.create_task(
                _heartbeat(t.name, fut, progress, interval=HEARTBEAT_INTERVAL)
            )
        if signal is not None:
            cancel_wait = asyncio.create_task(signal.wait())
        try:
            if cancel_wait is None:
                if t.execution_timeout:
                    value = await asyncio.wait_for(fut, timeout=t.execution_timeout)
                else:
                    value = await fut
            else:
                done, _ = await asyncio.wait(
                    {fut, cancel_wait},
                    timeout=t.execution_timeout or None,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if fut in done:
                    value = fut.result()
                elif cancel_wait in done:
                    # 取消 async task；to_thread 的底层同步函数仍会自然完成。
                    fut.cancel()
                    return (
                        f"[已中止] 用户打断，工具 {t.name} 未完成；"
                        "同步工具线程可能仍在后台运行，但 Agent 不会继续编排",
                        False, False,
                    )
                else:
                    fut.cancel()
                    return (
                        f"[工具执行超时({t.execution_timeout:g}s)] {t.name} 仍在后台运行，"
                        f"结果未返回（线程未终止，避免半途重复执行）",
                        True, False,
                    )
        except asyncio.TimeoutError:
            return (
                f"[工具执行超时({t.execution_timeout:g}s)] {t.name} 仍在后台运行，"
                f"结果未返回（线程未终止，避免半途重复执行）",
                True, False,
            )
        finally:
            if hb is not None:
                hb.cancel()
            if cancel_wait is not None:
                cancel_wait.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cancel_wait
        content = str(value) if value is not None else ""
        return content, False, bool(t.can_terminate and getattr(value, "terminate", False))

    @staticmethod
    def _truncate(content: str, max_result_chars: int) -> str:
        if len(content) > max_result_chars:
            return content[:max_result_chars] + f"\n…[已截断，超 {max_result_chars} 字符]"
        return content

    async def execute(
        self, call, *, confirm: ConfirmFn | None = None, max_result_chars: int = 4000,
        progress: ProgressFn | None = None, signal: asyncio.Event | None = None,
    ) -> ToolResult:
        """执行单个工具调用；一切异常包装为结果字符串，不抛出（模型可感知纠错）。

        权限判定：非 read → 必须经 confirm 确认，否则 AGENT_TOOL_PERMISSION。

        长任务（可观测 + 兜底）：
        - 同步函数一律丢线程池（asyncio.to_thread），不阻塞事件循环；
        - 存在 t.execution_timeout 时用 wait_for 兜底：超时不杀线程，仅调用方不再等待；
        - 传入 progress 时启动心跳，定期上报 (tool_name, elapsed)。
        - 传入 signal 后，取消会立即停止等待并阻止 Agent 进入下一轮；
          同步线程本身不可强杀，结果不会再回流到 Agent。
        """
        t = self.get(call.function.name)
        if signal is not None and signal.is_set():
            return tool_result(
                call.id,
                f"[已中止] 用户打断，工具 {t.name} 未执行",
            )
        await self._authorize(t, confirm)
        if signal is not None and signal.is_set():
            return tool_result(
                call.id,
                f"[已中止] 用户打断，工具 {t.name} 未执行",
            )
        content = ""
        terminate = False
        try:
            args = self._parse_args(t, call.function.arguments)
            content, _timed_out, terminate = await self._await_result(
                t, self._dispatch(t, args), progress, signal
            )
        except AgentError:
            raise
        except Exception as e:  # 业务异常 → 包装为结果让模型纠错
            content = f"[工具执行失败] {t.name}: {type(e).__name__}: {e}"
        return tool_result(call.id, self._truncate(content, max_result_chars), terminate=terminate)


def fail_tool_result(call, reason: str) -> ToolResult:
    """工具整批失败用的占位结果（对齐 length 截断防护，docs/04 §1.1）。"""
    return tool_result(call.id, f"[AGENT_OUTPUT_TRUNCATED] 该工具调用因输出被截断而失败，请重发：{reason}")

class RunRegistryView(ToolRegistry):
    """按 run 注入临时工具的注册表视图（L10/OPT-106）。

    serve 每请求新建 Runner 但共享注册表——直接 register 会把请求级工具泄漏给
    并发请求。本视图叠加 extra 工具于 get/schemas/active/all，execute/权限/
    截断等安全逻辑全部复用基类（execute 先 self.get 再走授权链，天然生效）。
    defer 注入（names 给定）时不叠加临时工具，保持子集语义。
    """

    def __init__(self, base: ToolRegistry, extra: list):
        super().__init__()
        self._base = base
        for t in extra:
            self._tools[t.name] = t

    def get(self, name: str):
        if name in self._tools:
            return self._tools[name]
        return self._base.get(name)

    def schemas(self, names=None):
        if names is not None:
            return self._base.schemas(names)
        return self._base.schemas() + [t.schema for t in self._tools.values()
                                       if not t.disable_model_invocation]

    def active(self, names=None):
        if names is not None:
            return self._base.active(names)
        return self._base.active() + list(self._tools.values())

    def all(self):
        return self._base.all() + list(self._tools.values())


class FilteredRegistryView(ToolRegistry):
    """只暴露白名单内工具的注册表视图（不修改 base）。

    为什么需要它（OPT-196 实测教训）：发给模型的 **schema 来自 `runner.registry`**
    （`core/loop.py` 用 `self.registry.schemas()`），只过滤 `Agent.tools` 是无效的——
    模型照样看得见并调用被排除的工具。做"评测只读"或未来的"只读模式"必须过滤注册表本身。

    execute/权限/截断仍复用基类：`get()` 对名单外工具报 AGENT_TOOL_NOT_FOUND，
    因此即使模型凭空喊出一个名字也执行不了（纵深防御，不依赖 schema 是否泄漏）。
    """

    def __init__(self, base: ToolRegistry, allow: Callable[[Tool], bool]) -> None:
        super().__init__()
        self._base = base
        self._allow = allow
        self._filtered = [t for t in base.all() if allow(t)]
        self._names = {t.name for t in self._filtered}

    def get(self, name: str):
        if name not in self._names:
            raise AgentError("AGENT_TOOL_NOT_FOUND",
                             f"工具 {name} 不在当前注册表视图内（评测/只读模式）")
        return self._base.get(name)

    def schemas(self, names=None):
        pool = self._filtered if names is None else [t for t in self._filtered if t.name in names]
        return [t.schema for t in pool if not t.disable_model_invocation]

    def active(self, names=None):
        return list(self._filtered) if names is None else [t for t in self._filtered if t.name in names]

    def all(self):
        return list(self._filtered)
