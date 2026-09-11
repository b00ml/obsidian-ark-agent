"""Embedding 客户端（P0-1/OPT-105）。

OpenAI 兼容 `/embeddings` 端点（dashscope compatible-mode / 硅基流动等均可），
stdlib urllib 实现，零新增依赖。本地模型（bge 系）留同类实现位：满足
`embed(texts) -> list[list[float]]` 协议即可注入 VectorIndex。

错误一律抛 RuntimeError 由调用方降级（向量路空，不影响关键词路）。
"""
from __future__ import annotations

import json
import urllib.request
from typing import Protocol, Sequence


class Embedder(Protocol):
    """向量生成协议：批量文本 → 等长向量列表（顺序对应）。"""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """OpenAI 兼容 /embeddings 客户端。分批请求，单批上限 batch_size。"""

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 timeout: float = 30.0, batch_size: int = 16):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = max(1, batch_size)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [(t or " ").strip() or " " for t in texts[i:i + self.batch_size]]
            out.extend(self._request(batch))
        return out

    def _request(self, batch: list[str]) -> list[list[float]]:
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": batch}).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"embeddings 请求失败: {type(e).__name__}: {e}") from e
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != len(batch):
            raise RuntimeError(f"embeddings 响应异常: 期望 {len(batch)} 条向量")
        vecs: list[list[float]] = []
        for item in data:
            vec = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vec, list) or not vec:
                raise RuntimeError("embeddings 响应缺 embedding 字段")
            vecs.append([float(x) for x in vec])
        return vecs
