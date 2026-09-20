"""Small injectable OpenAI-compatible text/vision client for ingest modules.

This is intentionally synchronous: the Bili/article processors are CLI and
batch code.  The transport is injectable so tests and long-lived callers can
reuse a session without coupling those processors to ``requests`` globals.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import requests


class ProviderError(RuntimeError):
    """Normalised provider failure with a stable code and retry decision."""

    def __init__(self, code: str, message: str, *, retryable: bool = False,
                 status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status_code = status_code


@dataclass(frozen=True)
class TextModelResult:
    content: str
    model: str
    usage: dict[str, Any]
    trace_id: str
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "usage": dict(self.usage),
            "trace_id": self.trace_id,
            "latency_ms": self.latency_ms,
        }


Transport = Callable[..., Any]


class TextModelClient:
    """Injectable synchronous client for ``/chat/completions``."""

    def __init__(self, *, api_base: str, api_key: str, model: str,
                 timeout: float = 120.0, transport: Transport | None = None,
                 trace: Callable[[dict[str, Any]], None] | None = None) -> None:
        if not api_key or api_key.startswith("sk-xxx"):
            raise ProviderError("AUTH_MISSING", f"API Key 缺失 (model={model})")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = float(timeout)
        self.transport = transport
        self.trace = trace

    def chat(self, prompt: str, *, max_tokens: int,
             image_b64: str | None = None, detail: str = "high") -> TextModelResult:
        if not callable(self.transport):
            transport = requests.post
        else:
            transport = self.transport
        content: Any = prompt
        if image_b64:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}", "detail": detail,
                }},
            ]
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
        }
        trace_id = f"trace_{uuid.uuid4().hex[:16]}"
        started = time.perf_counter()
        try:
            response = transport(
                f"{self.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
        except (TimeoutError, requests.exceptions.Timeout) as exc:
            raise ProviderError("TIMEOUT", "模型请求超时", retryable=True) from exc
        except Exception as exc:
            raise ProviderError("NETWORK", f"模型网络错误: {exc}", retryable=True) from exc
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        status = getattr(response, "status_code", None)
        if not isinstance(status, int):
            # Keep compatibility with light-weight response mocks that expose
            # only ``raise_for_status`` and ``json``.
            checker = getattr(response, "raise_for_status", None)
            if callable(checker):
                try:
                    checker()
                except requests.exceptions.Timeout as exc:
                    raise ProviderError("TIMEOUT", "模型请求超时", retryable=True) from exc
                except requests.exceptions.HTTPError as exc:
                    error_response = getattr(exc, "response", None)
                    error_status = getattr(error_response, "status_code", None)
                    if isinstance(error_status, int):
                        retryable = bool(error_status == 429 or error_status >= 500)
                        code = "RATE_LIMIT" if error_status == 429 else (
                            "HTTP_5XX" if retryable else "HTTP_4XX"
                        )
                        body = str(getattr(error_response, "text", ""))[:200]
                        raise ProviderError(
                            code, f"模型 HTTP {error_status}: {body}",
                            retryable=retryable, status_code=error_status,
                        ) from exc
                    raise ProviderError("HTTP_ERROR", "模型 HTTP 响应异常") from exc
                except Exception as exc:
                    raise ProviderError("NETWORK", f"模型 HTTP 响应异常: {exc}", retryable=True) from exc
            status = 200
        if status != 200:
            retryable = bool(status == 429 or status >= 500)
            code = "RATE_LIMIT" if status == 429 else ("HTTP_5XX" if retryable else "HTTP_4XX")
            body = str(getattr(response, "text", ""))[:200]
            raise ProviderError(code, f"模型 HTTP {status}: {body}",
                                retryable=retryable, status_code=status)
        try:
            result = response.json()
            content_text = result["choices"][0]["message"]["content"]
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("MALFORMED_RESPONSE", "模型返回结构异常") from exc
        if not isinstance(content_text, str):
            raise ProviderError("MALFORMED_RESPONSE", "模型 content 不是字符串")
        usage = result.get("usage") or {}
        out = TextModelResult(content_text, self.model, dict(usage), trace_id, latency_ms)
        if self.trace is not None:
            self.trace({"trace_id": trace_id, "model": self.model,
                        "latency_ms": latency_ms, "usage": dict(usage)})
        return out


def result_dict(result: TextModelResult) -> dict[str, Any]:
    """Compatibility shape used by existing processors."""
    return result.to_dict()


__all__ = ["ProviderError", "TextModelClient", "TextModelResult", "result_dict"]
