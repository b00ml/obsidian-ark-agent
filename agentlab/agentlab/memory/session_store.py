"""会话持久化：JSONL + append-only + 每会话跨进程锁（apent-only 事件源）。

设计对齐 tau:` src/tau_agent/session/storage.py` ——
- append-only 尾部追加（O(1)，自然的 crash-安全 event-sourcing）；
- 每会话一个 `.jsonl` + `. <id>.lock`，写操作持独占跨进程锁；
- read 在锁内读完全部行，容忍末尾残留的半行（追加中途崩溃的产物）。
- `SessionStorage` Protocol + `InMemorySessionStorage` 测试替身便于注入与单测。

服务端引用：serve 端依据 `previous_response_id` 重放历史、续写新回合（OPT-074 S2）。
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from agentlab.core.message import Message


class SessionStorage(Protocol):
    """只暴露两个操作；append 是持久化事务边界（崩了整体要么可见要么不动）。"""

    def append(self, session_id: str, msg: Message) -> None: ...
    def read_all(self, session_id: str) -> list[Message]: ...


class JsonlSessionStorage:
    """每会话一个 .jsonl，append-only + 每会话跨进程锁。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sid: str) -> Path:
        return self.root / f"{sid}.jsonl"

    def _lock_path(self, sid: str) -> Path:
        return self.root / f".{sid}.lock"

    def append(self, session_id: str, msg: Message) -> None:
        """O(1) 尾部追加：锁 → 写入一行 → fsync（崩溃只丢半行，不坏整体）。"""
        with self._locked(session_id):
            with self._path(session_id).open("ab") as f:
                line = json.dumps(msg.model_dump(), ensure_ascii=False)
                f.write((line + "\n").encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())

    def read_all(self, session_id: str) -> list[Message]:
        """读全部消息；空白或非目录行跳过，末尾残留半行丢弃。"""
        with self._locked(session_id):
            path = self._path(session_id)
            if not path.exists():
                return []
            out: list[Message] = []
            raw_lines = path.read_text(encoding="utf-8").splitlines()
            for line in raw_lines:
                if not line.strip():
                    continue
                try:
                    out.append(Message.model_validate(json.loads(line)))
                except ValueError:
                    # 追加中途崩溃残留的半行：丢弃，避免一条坏行封死整会话
                    break
            return out

    def list_sessions(self) -> Iterator[str]:
        return (p.stem for p in self.root.glob("*.jsonl"))

    def delete(self, session_id: str) -> bool:
        """删除整会话（#8/OPT-127：ark 删除按钮接线）；返回会话是否存在过。"""
        path = self._path(session_id)
        if not path.exists():
            return False
        with self._locked(session_id):
            path.unlink()
        try:  # 锁文件须在锁释放后删（Windows 不允许删除打开中的文件）
            self._lock_path(session_id).unlink()
        except OSError:
            pass
        return True

    @contextmanager
    def _locked(self, session_id: str) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock_path(session_id).open("a+b") as lock_file:
            _lock_file(lock_file)
            try:
                yield
            finally:
                _unlock_file(lock_file)


class InMemorySessionStorage:
    """测试/嵌入用替身：实现同一 Protocol，行为确定。"""

    def __init__(self) -> None:
        self._data: dict[str, list[Message]] = {}

    def append(self, session_id: str, msg: Message) -> None:
        self._data.setdefault(session_id, []).append(msg)

    def read_all(self, session_id: str) -> list[Message]:
        return list(self._data.get(session_id, []))

    def delete(self, session_id: str) -> bool:
        return self._data.pop(session_id, None) is not None


# —— 平台锁隔离成两个微函数（Windows msvcrt 无共享锁，写一律独占=安全） ——
def _lock_file(file) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
        return
    import fcntl

    fcntl.flock(file.fileno(), fcntl.LOCK_EX)


def _unlock_file(file) -> None:
    if os.name == "nt":
        import msvcrt

        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(file.fileno(), fcntl.LOCK_UN)