"""契约层幂等键（serve 拆职责：S6，对照 §4.7「幂等 + 写串行化」）。

目标：客户端同一 `request_id` 的重复提交（断线重试 / 双击 / 前端重发）不重复
跑 agent、不重复烧 token、不重复写库，而是直接重放首次请求的完整 SSE 结果。

设计（有界 + 时间窗）：
- 键状态：`done`（有可重放结果）vs `pending`（首个请求仍在跑）。
- `begin` 抢锁：同一键并发第二次提交拿到 `False` → serve 回 409，杜绝 double-run。
- `commit` 落结果、`abandon` 释放（失败/返回前不留脏 pending）。
- 懒过期：TTL 内命中，超出即丢；容量到上限时淘汰最旧（有界，防内存无限增长）。

线程模型：aiohttp 单事件循环 + 无锁同步增删（中途无 await，天然原子），
无需跨进程锁——跨进程写串行化已由 session_store 的文件锁（P0）覆盖。
"""
from __future__ import annotations

import time


class _IdempotencyCache:
    """request_id → 完整 SSE 事件序列 的有界、TTL 幂等缓存。

    值格式：`rq_id -> (recorded_at_monotonic, chunks_or_None)`，`chunks None` 表示 pending。
    """

    def __init__(self, size: int = 256, ttl: float = 300.0):
        self._size = max(1, size)
        self._ttl = ttl
        self._data: dict[str, tuple[float, list[str] | None]] = {}

    def snapshot(self, rq_id: str) -> tuple[str, list[str]] | None:
        """返回该键状态：`("done", chunks)` 或 `("pending", None)`；无键/已过期 → None。"""
        item = self._data.get(rq_id)
        if item is None:
            return None
        ts, chunks = item
        if time.monotonic() - ts > self._ttl:
            self._data.pop(rq_id, None)
            return None
        return ("done", chunks) if chunks is not None else ("pending", None)

    def begin(self, rq_id: str) -> bool:
        """抢占键的跑权。已 done/已 pending（未过期）→ False（重复）；返回 True 表示本请求持键。"""
        item = self._data.get(rq_id)
        if item is not None:
            ts, _ = item
            if time.monotonic() - ts > self._ttl:
                self._data.pop(rq_id, None)
            else:
                return False
        self._evict_if_full(rq_id)
        self._data[rq_id] = (time.monotonic(), None)
        return True

    def commit(self, rq_id: str, chunks: list[str]) -> None:
        """持键请求跑完：把完整 SSE 序列存为 done。"""
        ts = self._data.get(rq_id, (time.monotonic(), None))[0]
        self._data[rq_id] = (ts, list(chunks))

    def abandon(self, rq_id: str) -> None:
        """持键请求失败/中断：释放 pending，让重试可重新跑（不缓存失败结果）。"""
        self._data.pop(rq_id, None)

    def _evict_if_full(self, rq_id: str) -> None:
        if len(self._data) < self._size or rq_id in self._data:
            return
        oldest = min(self._data, key=lambda k: self._data[k][0])
        self._data.pop(oldest, None)