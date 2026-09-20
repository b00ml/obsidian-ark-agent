"""Embedding 客户端（P0-1/OPT-105；OPT-264 连接复用与超时治理）。

OpenAI 兼容 `/embeddings` 端点（dashscope compatible-mode / 硅基流动等均可），
stdlib 实现，零新增依赖。单进程内复用同一个 HTTPS 连接（keep-alive），避免每条
查询重新做一次 TCP/TLS 握手；网络级瞬时错误（超时/断连/异常响应）有界重试一次，
防止一条死连接把 warm p95 拖到 20s 级。本地模型（bge 系）留同类实现位：满足
`embed(texts) -> list[list[float]]` 协议即可注入 VectorIndex。

错误一律抛 RuntimeError 由调用方降级（向量路空，不影响关键词路）。
"""
from __future__ import annotations

import http.client
import json
import threading
from typing import Protocol, Sequence
from urllib.parse import urlsplit

# 网络级瞬时错误：值得重试（重试前丢弃复用连接）
_TRANSIENT = (TimeoutError, ConnectionError, OSError, http.client.HTTPException)


class Embedder(Protocol):
    """向量生成协议：批量文本 → 等长向量列表（顺序对应）。"""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """OpenAI 兼容 /embeddings 客户端。分批请求，单批上限 batch_size。"""

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 timeout: float = 30.0, batch_size: int = 16,
                 max_retries: int = 1):
        base = urlsplit((base_url or "").rstrip("/"))
        if not base.scheme or not base.netloc:
            raise ValueError(f"非法 base_url: {base_url!r}")
        self._scheme = base.scheme.lower()
        self._host = base.hostname or ""
        self._port = base.port or (443 if self._scheme == "https" else 80)
        self._path = f"{base.path.rstrip('/')}/embeddings"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = max(1, batch_size)
        self.max_retries = max(0, int(max_retries))
        self._conn: http.client.HTTPConnection | None = None
        # HTTPConnection is stateful and not thread-safe.  Hybrid retrieval
        # may issue work from multiple threads, so serialize requests while
        # still reusing the same keep-alive connection safely.
        self._lock = threading.Lock()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [(t or " ").strip() or " " for t in texts[i:i + self.batch_size]]
            out.extend(self._request(batch))
        return out

    def _faraday(self) -> http.client.HTTPConnection:
        """建（或复用）到服务端的连接。同一实例内的连续调用共享一条 keep-alive 连接。"""
        if self._conn is None:
            if self._scheme == "https":
                self._conn = http.client.HTTPSConnection(self._host, self._port, timeout=self.timeout)
            else:
                self._conn = http.client.HTTPConnection(self._host, self._port, timeout=self.timeout)
        return self._conn

    def _drop_connection(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass

    def _request(self, batch: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": batch}
        last_error: Exception | None = None
        with self._lock:
            for attempt in range(self.max_retries + 1):
                try:
                    data = self._post(payload)
                    return self._parse(data, batch)
                except _TRANSIENT as exc:
                    last_error = exc
                    self._drop_connection()
                    if attempt >= self.max_retries:
                        break
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(f"embeddings 响应异常: {type(exc).__name__}: {exc}") from exc
        assert last_error is not None
        raise RuntimeError(f"embeddings 请求失败: {type(last_error).__name__}: {last_error}") from last_error

    def _post(self, payload: dict) -> dict:
        conn = self._faraday()
        conn.request(
            "POST",
            self._path,
            body=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )
        resp = conn.getresponse()
        body = resp.read()
        if resp.status < 200 or resp.status >= 300:
            raise RuntimeError(
                f"embeddings HTTP {resp.status}: {body.decode('utf-8', 'replace')[:200]}"
            )
        try:
            return json.loads(body.decode("utf-8"))
        except ValueError as exc:
            raise ValueError(f"embeddings 响应非 JSON: {exc}") from exc

    @staticmethod
    def _parse(data: object, batch: list[str]) -> list[list[float]]:
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list) or len(items) != len(batch):
            raise ValueError(f"期望 {len(batch)} 条向量，实际 {len(items) if isinstance(items, list) else '缺失'}")
        vecs: list[list[float]] = []
        for item in items:
            vec = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vec, list) or not vec:
                raise ValueError("embeddings 响应缺 embedding 字段")
            vecs.append([float(x) for x in vec])
        return vecs
