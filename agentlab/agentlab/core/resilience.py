"""熔断 + 重试（对齐 docs/03 §工程化实践）。

- CircuitBreaker：closed→open→half-open 三态；连续失败达阈值后短暂熔断，
  恢复期放一个探测请求，成功即闭合。
- ResilientLLM：包装任一 LLMProvider，对可重试错误码（429/超时/网络/5xx）
  按指数退避重试；兜底交给熔断器判断是否继续放行。
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

from agentlab.core.errors import AgentError
from agentlab.core.llm import LLMProvider

log = logging.getLogger("agentlab.resilience")

RETRYABLE_CODES = {"AGENT_LLM_RATE", "AGENT_LLM_TIMEOUT", "AGENT_LLM_AUTH"}


class CircuitBreaker:
    """失败计数熔断。open 阶段抛 AGENT_LLM_CIRCUIT_OPEN，不触网。"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max: int = 1,
        clock: Callable[[], float] | None = None,
    ):
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_timeout = recovery_timeout
        self.half_open_max = max(1, half_open_max)
        self._clock = clock or time.monotonic
        self._failures = 0
        self._state = "closed"  # closed | open | half_open
        self._open_since = 0.0
        self._half_open_sent = 0

    @property
    def state(self) -> str:
        if self._state == "open" and self._clock() - self._open_since >= self.recovery_timeout:
            self._state = "half_open"
            self._half_open_sent = 0
        return self._state

    def allow(self) -> bool:
        st = self.state
        if st == "open":
            return False
        if st == "half_open":
            if self._half_open_sent >= self.half_open_max:
                return False
            self._half_open_sent += 1
        return True

    def on_success(self) -> None:
        self._failures = 0
        self._state = "closed"

    def on_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._state = "open"
            self._open_since = self._clock()
            self._half_open_sent = 0


class ResilientLLM(LLMProvider):
    """给任意 provider 套上重试 + 熔断。"""

    def __init__(
        self,
        inner: LLMProvider,
        breaker: CircuitBreaker | None = None,
        max_retries: int = 2,
        backoff: float = 1.0,
        retryable_codes: set[str] | None = None,
    ):
        self.inner = inner
        self.breaker = breaker or CircuitBreaker()
        self.max_retries = max_retries
        self.backoff = backoff
        self.retryable = retryable_codes or RETRYABLE_CODES

    async def chat(
        self,
        messages: list,
        tools: list[dict] | None = None,
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        stream: bool = False,
        on_stream: Callable[[str], None] | None = None,
    ):
        if not self.breaker.allow():
            raise AgentError(
                "AGENT_LLM_CIRCUIT_OPEN", "LLM 熔断开启，暂时无法发起调用"
            )
        last_err: AgentError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self.inner.chat(
                    messages, tools,
                    temperature=temperature, max_tokens=max_tokens,
                    stream=stream, on_stream=on_stream,
                )
                self.breaker.on_success()
                return resp
            except AgentError as e:
                last_err = e
                if e.code not in self.retryable:
                    self.breaker.on_failure()
                    raise
                self.breaker.on_failure()
                if attempt < self.max_retries:
                    delay = self.backoff * (2 ** attempt)
                    log.warning("LLM 重试 %s/%s（%s），%ss 后重试",
                                attempt + 1, self.max_retries, e.code, delay)
                    await asyncio.sleep(delay)
            except Exception as e:  # 非 AgentError 一律视为可重试瞬时错误
                last_err = AgentError("AGENT_LLM_TIMEOUT", f"LLM 异常：{e}")
                self.breaker.on_failure()
                if attempt < self.max_retries:
                    await asyncio.sleep(self.backoff * (2 ** attempt))
        raise last_err if last_err is not None else AgentError("AGENT_LLM_TIMEOUT", "LLM 未知故障")