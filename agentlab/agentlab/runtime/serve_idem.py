"""契约层幂等键（serve 拆职责：S6，对照 §4.7「幂等 + 写串行化」）。

目标：客户端同一 `request_id` 的重复提交（断线重试 / 双击 / 前端重发）不重复
跑 agent、不重复烧 token、不重复写库，而是直接重放首次请求的完整 SSE 结果。

设计（有界 + 时间窗 + 可选持久化）：
- 键状态：`done`（有可重放结果）vs `pending`（首个请求仍在跑）。
- `begin` 抢锁：同一键并发第二次提交拿到 `False` → serve 回 409，杜绝 double-run。
- `commit` 落结果、`abandon` 释放（失败/返回前不留脏 pending）。
- 懒过期：TTL 内命中，超出即丢；容量到上限时淘汰最旧（有界，防内存无限增长）。
- 持久化背板（P0-04，OPT-214）：给定 `store_path` 时 commit 同步写 SQLite、
  构造时加载 TTL 内的 done 记录——服务重启后同一 request_id 仍命中重放，
  不会重复执行已完成的请求（M1 验收）。pending 状态不持久化：重启后其执行
  本身已丢失，重新提交应重跑而非 409。背板读写失败静默降级为纯内存（与
  "本地 commit 不受影响"口径一致，记 stderr 警告）。

线程模型：aiohttp 单事件循环 + 无锁同步增删（中途无 await，天然原子）；
SQLite 背板同样只在事件循环线程读写，无需跨进程锁。
"""
from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import sys
import time


class _IdempotencyCache:
    """request_id → 完整 SSE 事件序列 的有界、TTL 幂等缓存（可选 SQLite 背板）。

    内存值格式：`rq_id -> (recorded_at_monotonic, chunks_or_None)`，`chunks None` 表示 pending。
    """

    def __init__(self, size: int = 256, ttl: float = 300.0, store_path: str | None = None):
        self._size = max(1, size)
        self._ttl = ttl
        self._data: dict[str, tuple[float, list[str] | None]] = {}
        self._store_path = store_path
        if store_path:
            self._load_from_store()

    # ── 持久化背板 ────────────────────────────────────────────────

    def _open_store(self):
        # contextlib.closing 显式关连接：sqlite3 的 with 只管事务，不释放文件句柄
        # （不关会导致 Windows 上 db 文件被占用，临时目录清理报 WinError 32）
        conn = sqlite3.connect(self._store_path)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS idempotency (
                   request_id TEXT PRIMARY KEY,
                   created_at REAL NOT NULL,
                   chunks TEXT NOT NULL)""")
        return contextlib.closing(conn)

    def _load_from_store(self) -> None:
        """重启恢复：把 TTL 内的 done 记录按新到旧加载进内存（至多 size 条），顺带清过期行。"""
        try:
            os.makedirs(os.path.dirname(self._store_path) or ".", exist_ok=True)
            with self._open_store() as conn:
                conn.execute("DELETE FROM idempotency WHERE ? - created_at > ?",
                             (time.time(), self._ttl))
                rows = conn.execute(
                    "SELECT request_id, created_at, chunks FROM idempotency"
                    " ORDER BY created_at DESC LIMIT ?", (self._size,)).fetchall()
                conn.commit()
        except sqlite3.Error as e:
            print(f"[IDEM] 持久背板加载失败，退化为纯内存: {e}", file=sys.stderr)
            return
        now = time.monotonic()
        loaded = 0
        for rq_id, created_at, chunks_json in rows:
            wall_age = time.time() - created_at
            if wall_age > self._ttl:
                continue  # 过期不恢复（懒过期在下次操作时清理旧行）
            try:
                chunks = json.loads(chunks_json)
            except (ValueError, TypeError):
                continue
            # created_at 是墙钟；内存时间轴用 monotonic，平移到当前起点
            self._data[rq_id] = (now, chunks)
            loaded += 1
        if loaded:
            print(f"[IDEM] 从持久背板恢复 {loaded} 条幂等记录", file=sys.stderr)

    def _persist(self, rq_id: str, chunks: list[str] | None) -> None:
        if not self._store_path or chunks is None:
            return
        try:
            with self._open_store() as conn:
                conn.execute("INSERT OR REPLACE INTO idempotency VALUES (?, ?, ?)",
                             (rq_id, time.time(), json.dumps(chunks, ensure_ascii=False)))
                conn.execute("DELETE FROM idempotency WHERE ? - created_at > ?",
                             (time.time(), self._ttl))
                conn.commit()
        except sqlite3.Error as e:
            print(f"[IDEM] 幂等记录落盘失败（重启后该键将不可重放）: {e}", file=sys.stderr)

    def _unpersist(self, rq_id: str) -> None:
        if not self._store_path:
            return
        try:
            with self._open_store() as conn:
                conn.execute("DELETE FROM idempotency WHERE request_id=?", (rq_id,))
                conn.commit()
        except sqlite3.Error:
            pass

    # ── 内存语义（与持久化前版本一致） ───────────────────────────

    def snapshot(self, rq_id: str) -> tuple[str, list[str]] | None:
        """返回该键状态：`("done", chunks)` 或 `("pending", None)`；无键/已过期 → None。"""
        item = self._data.get(rq_id)
        if item is None:
            return None
        ts, chunks = item
        if time.monotonic() - ts > self._ttl:
            self._data.pop(rq_id, None)
            self._unpersist(rq_id)
            return None
        return ("done", chunks) if chunks is not None else ("pending", None)

    def begin(self, rq_id: str) -> bool:
        """抢占键的跑权。已 done/已 pending（未过期）→ False（重复）；返回 True 表示本请求持键。"""
        item = self._data.get(rq_id)
        if item is not None:
            ts, _ = item
            if time.monotonic() - ts > self._ttl:
                self._data.pop(rq_id, None)
                self._unpersist(rq_id)
            else:
                return False
        self._evict_if_full(rq_id)
        self._data[rq_id] = (time.monotonic(), None)
        return True

    def commit(self, rq_id: str, chunks: list[str]) -> None:
        """持键请求跑完：把完整 SSE 序列存为 done（含持久背板）。"""
        ts = self._data.get(rq_id, (time.monotonic(), None))[0]
        self._data[rq_id] = (ts, list(chunks))
        self._persist(rq_id, self._data[rq_id][1])

    def abandon(self, rq_id: str) -> None:
        """持键请求失败/中断：释放 pending，让重试可重新跑（不缓存失败结果）。"""
        self._data.pop(rq_id, None)
        self._unpersist(rq_id)

    def _evict_if_full(self, rq_id: str) -> None:
        if len(self._data) < self._size or rq_id in self._data:
            return
        oldest = min(self._data, key=lambda k: self._data[k][0])
        self._data.pop(oldest, None)
        self._unpersist(oldest)
