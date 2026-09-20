"""LLM Provider 抽象与 OpenAI 兼容实现（对齐 docs/03 §2.4）。

- LLMProvider 是循环唯一与外部模型的接口。
- OpenAICompatProvider 内部用 httpx，`stream=True` 逐段回调；超时映射 AGENT_LLM_TIMEOUT。
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable

from pydantic import BaseModel

from agentlab.core.errors import AgentError
from agentlab.core.message import Message, TokenUsage, ToolCall, ToolCallFunction


class LLMResponse(BaseModel):
    content: str | None
    tool_calls: list[ToolCall]
    usage: TokenUsage
    stop_reason: str | None = None
    # "length" 表示输出被 token 上限截断 → 工具调用参数可能残缺，
    # 循环层将整批失败重发（docs/04 §1.1 / §2.4），不执行坏参数


class LLMProvider(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,  # 已序列化的 function schema
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        stream: bool = False,
        on_stream: Callable[[str], None] | None = None,
    ) -> LLMResponse: ...


class OpenAICompatProvider(LLMProvider):
    """OpenAI 兼容 Chat Completions（覆盖 DeepSeek/Qwen/OpenAI 等）。

    通过 PATH 归一化：本地 /v1/ 若缺失自动补，容忍 base_url 结尾 / 与不带 chat/completions。
    载荷直接发给 {base_url}/chat/completions。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        max_tokens: int | None = None,
        httpx_client: Any = None,
    ):
        if not api_key:
            raise AgentError("AGENT_LLM_AUTH", "LLM API Key 缺失（配置为空且无 AGENT_LLM_API_KEY）")
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1") and "/chat/completions" not in self.base_url:
            self.base_url += "/v1"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._httpx = httpx_client
        # Lazily created so construction remains independent from an event
        # loop.  A long-lived provider (serve) can therefore reuse HTTP
        # connections across turns while tests/CLI may still inject a client.
        self._owned_httpx = None

    async def _client(self):
        """Return the injected or process-owned async client."""
        if self._httpx is not None:
            return self._httpx
        if self._owned_httpx is None:
            import httpx

            self._owned_httpx = httpx.AsyncClient(timeout=self.timeout)
        return self._owned_httpx

    async def aclose(self) -> None:
        """Close the lazily owned connection pool; injected clients stay caller-owned."""
        client, self._owned_httpx = self._owned_httpx, None
        if client is not None:
            await client.aclose()

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        stream: bool = False,
        on_stream: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        client = await self._client()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_provider_dict() for m in messages],
            "temperature": temperature,
        }
        if (max_tokens or self.max_tokens):
            payload["max_tokens"] = max_tokens or self.max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = self.base_url if self.base_url.endswith("/chat/completions") else f"{self.base_url}/chat/completions"
        try:
            resp = await client.post(url, json=payload, headers=headers)
        except asyncio.TimeoutError:
            raise AgentError("AGENT_LLM_TIMEOUT", f"LLM 调用超时（{self.timeout}s）") from None
        except Exception as e:  # 网络类故障统一包装
            raise AgentError("AGENT_LLM_TIMEOUT", f"LLM 网络错误：{e}") from None
        if resp.status_code == 429:
            raise AgentError("AGENT_LLM_RATE", "LLM 限流（HTTP 429）")
        if resp.status_code != 200:
            raise AgentError(
                "AGENT_LLM_TIMEOUT" if resp.status_code >= 500 else "AGENT_LLM_AUTH",
                f"LLM 返回 HTTP {resp.status_code}: {resp.text[:200]}",
            )
        try:
            data = resp.json()
            choice = data["choices"][0]["message"]
            usage = data.get("usage") or {}
            return self._parse(choice, usage)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise AgentError(
                "AGENT_LLM_SCHEMA", f"LLM 响应结构无效：{type(exc).__name__}"
            ) from exc

    def _parse(self, choice: dict, usage: dict) -> LLMResponse:
        content = choice.get("content")
        raw_tools = choice.get("tool_calls") or []
        tool_calls = [
            ToolCall(
                id=tc.get("id", f"call_{i}"),
                type="function",
                function=ToolCallFunction(
                    name=tc["function"]["name"],
                    arguments=tc["function"].get("arguments") or "{}",
                ),
            )
            for i, tc in enumerate(raw_tools)
        ]
        finish = choice.get("finish_reason")
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason="tool_calls" if tool_calls else (finish or "stop"),
            usage=TokenUsage(
                input_tokens=usage.get("prompt_tokens") or 0,
                output_tokens=usage.get("completion_tokens") or 0,
            ),
        )
