"""Governed Vault writes: atomic replace, revision/CAS, per-path locks and audit.

The gateway is deliberately small and synchronous because the MCP server's write
tools are synchronous.  It is the single write implementation used by
``tools_vault``; callers may still use the old function names during migration.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import uuid
from pathlib import Path

from common import assert_writable, resolve_note_path, vault_root


class VaultConflictError(RuntimeError):
    """The caller attempted a compare-and-set write against an old revision."""

    code = "VAULT_CONFLICT"

    def __init__(self, path: str, expected: str | None, actual: str | None):
        self.path, self.expected_revision, self.actual_revision = path, expected, actual
        super().__init__(
            f"Vault revision 冲突: path={path}, expected={expected or '<absent>'}, "
            f"actual={actual or '<absent>'}"
        )


_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_AUDIT_LOCK = threading.Lock()


def _path_lock(path: str) -> threading.RLock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path, threading.RLock())


def _normalise_path(path: str) -> str:
    p = path.replace("\\", "/").strip().lstrip("./")
    if not p:
        raise ValueError("路径不能为空")
    if "/" not in p:
        p = f"Inbox/{p}"
    if not p.lower().endswith(".md"):
        p += ".md"
    return p


def _revision(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _read_revision(full: str) -> tuple[str | None, str]:
    if not os.path.exists(full):
        return None, ""
    with open(full, "r", encoding="utf-8") as fh:
        content = fh.read()
    return _revision(content), content


def _audit_path(config: dict) -> str:
    return os.path.join(vault_root(config), config.get("audit_path", ".agent-brain/audit/vault.jsonl"))


def _audit(config: dict, event: dict) -> bool:
    """追加审计事件；返回是否落盘成功。

    失败不抛异常（不阻断业务写入），但调用方必须把失败带进返回值——
    审计缺失的副作用视为"结果未核实"（P0-06：审计写失败不得假报成功）。
    """
    path = _audit_path(config)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        with _AUDIT_LOCK, open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        return True
    except OSError as e:
        print(f"[VAULT] 审计写失败（该副作用结果未核实）: {e}", file=sys.stderr)
        return False


def _atomic_write(full: str, content: str) -> None:
    parent = os.path.dirname(full)
    os.makedirs(parent, exist_ok=True)
    tmp = f"{full}.tmp-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, full)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


class VaultGateway:
    """Single-process Vault write boundary used by MCP and compatibility callers."""

    def __init__(self, config: dict):
        self.config = config

    def write(self, path: str, content: str, *, expected_revision: str | None = None,
              request_id: str = "") -> dict:
        assert_writable(self.config, path)
        rel = _normalise_path(path)
        full = resolve_note_path(self.config, rel)
        lock = _path_lock(os.path.abspath(full))
        with lock:
            previous, _ = _read_revision(full)
            if expected_revision is not None and expected_revision != previous:
                # CAS 冲突也是必须可追查的副作用尝试（P0-06：冲突有事件）
                _audit(self.config, {
                    "operation": "write", "path": rel, "result": "conflict",
                    "expected_revision": expected_revision, "actual_revision": previous,
                    "request_id": request_id,
                })
                raise VaultConflictError(rel, expected_revision, previous)
            _atomic_write(full, content)
            revision = _revision(content)
            ok = _audit(self.config, {
                "operation": "write", "path": rel, "result": "written",
                "previous_revision": previous,
                "revision": revision, "bytes": len(content.encode("utf-8")),
                "request_id": request_id,
            })
        out = {"path": rel, "bytes": len(content.encode("utf-8")),
               "status": "written", "revision": revision,
               "previous_revision": previous}
        if not ok:
            out["warning"] = "audit_write_failed_result_unverified"
        return out

    def patch(self, path: str, old: str, new: str, *,
              expected_revision: str | None = None, request_id: str = "") -> dict:
        assert_writable(self.config, path)
        rel = _normalise_path(path)
        full = resolve_note_path(self.config, rel)
        lock = _path_lock(os.path.abspath(full))
        with lock:
            previous, content = _read_revision(full)
            if previous is None:
                raise FileNotFoundError(f"笔记不存在: {path}")
            if expected_revision is not None and expected_revision != previous:
                _audit(self.config, {
                    "operation": "patch", "path": rel, "result": "conflict",
                    "expected_revision": expected_revision, "actual_revision": previous,
                    "request_id": request_id,
                })
                raise VaultConflictError(rel, expected_revision, previous)
            count = content.count(old)
            if count == 0:
                raise ValueError(f"未找到待替换文本（path={path}）")
            if count > 1:
                raise ValueError(f"待替换文本出现 {count} 次，不唯一，请扩大上下文")
            new_content = content.replace(old, new)
            _atomic_write(full, new_content)
            revision = _revision(new_content)
            ok = _audit(self.config, {
                "operation": "patch", "path": rel, "result": "patched",
                "previous_revision": previous,
                "revision": revision, "bytes": len(new_content.encode("utf-8")),
                "request_id": request_id,
            })
        out = {"path": rel, "replaced": 1, "status": "patched",
               "revision": revision, "previous_revision": previous}
        if not ok:
            out["warning"] = "audit_write_failed_result_unverified"
        return out
