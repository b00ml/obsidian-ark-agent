"""受控 Vault 写入助手（P0-01 Agent/自动化写入安全，OPT-213）。

背景：bili_transcript / knowledge_compiler / tools_bili 曾直接 open(path, "w")
写 Vault 笔记——绕过 VaultGateway，无路径守卫、非原子、无 revision、无审计。

本模块给"管线生产者"提供轻量受控写入（对齐 VaultGateway 的安全规则，
但不引入跨模块依赖——bili_summarizer 不得反向 import obsidian_agent_brain）：
- 路径守卫：Vault 内拒绝写 raw/、templates/、.obsidian/、.git/；
  唯一白名单 raw/screenshots/（截图产物历史布局，笔记 [[嵌入]] 依赖，
  迁移需单独 OPT）；Vault 外路径（CLI 本地输出模式）不做 Vault 守卫。
- 原子写：同目录临时文件 + os.replace，杜绝半写文件；进程内 per-path 锁
  串行化并发写。
- revision：sha256（与 VaultGateway 口径一致），返回 previous/revision。
- 最小审计：Vault 内写入追加 {vault}/.agent-brain/audit/pipeline.jsonl
  （含操作、路径、revision、字节、来源标记；不含正文）。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid

# Vault 内拒写目录（相对 Vault 根第一段，小写比较）；AGENTS.md §5 + 执行手册 4.2
DENY_DIRS = {"raw", "templates", ".obsidian", ".git"}
# raw/ 下唯一允许写入的产物子树（截图/网格图，历史布局被既有笔记嵌入引用）
RAW_ALLOW_SUBTREE = "raw/screenshots"
AUDIT_REL = os.path.join(".agent-brain", "audit", "pipeline.jsonl")

_path_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


class VaultWriteDenied(PermissionError):
    """结构化拒绝：path + 命中的守卫规则。"""

    def __init__(self, path: str, rule: str):
        self.path, self.rule = path, rule
        super().__init__(f"VAULT_WRITE_DENIED（{rule}）: {path}")


def _path_lock(abspath: str) -> threading.Lock:
    with _locks_guard:
        return _path_locks.setdefault(os.path.normcase(os.path.abspath(abspath)),
                                      threading.Lock())


def _under(child: str, parent: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(child), os.path.abspath(parent)]) \
            == os.path.abspath(parent)
    except ValueError:  # 不同盘符（Windows）
        return False


def guard_vault_path(vault_root: str | None, abs_path: str) -> None:
    """Vault 内路径守卫；vault_root 为空或目标在 Vault 外时跳过（CLI 本地输出）。

    路径穿越（目标名义在 root 下、实际逃逸）同样拒绝。
    """
    if not vault_root or not _under(abs_path, vault_root):
        return
    rel = os.path.relpath(os.path.abspath(abs_path), os.path.abspath(vault_root))
    rel_posix = rel.replace("\\", "/")
    # 不用 lstrip("./")：会把 ".obsidian/x" 误剥成 "obsidian/x" 静默绕过守卫
    while rel_posix.startswith("./"):
        rel_posix = rel_posix[2:]
    first = rel_posix.split("/", 1)[0].lower()
    if first in DENY_DIRS:
        if not (first == "raw" and rel_posix.lower().startswith(RAW_ALLOW_SUBTREE)):
            raise VaultWriteDenied(abs_path, f"{first}/ 目录只读")
    # 二段校验：逐段走上去不允许出现 ".."（relpath 已归一，双保险）
    if ".." in rel_posix.split("/"):
        raise VaultWriteDenied(abs_path, "路径穿越")


def _audit(vault_root: str | None, record: dict) -> None:
    """最小审计：正文不落审计，只记元数据；失败不影响业务结果。"""
    if not vault_root or not _under(os.path.join(vault_root, AUDIT_REL), vault_root):
        return
    try:
        os.makedirs(os.path.dirname(os.path.join(vault_root, AUDIT_REL)), exist_ok=True)
        with open(os.path.join(vault_root, AUDIT_REL), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # 审计失败不阻断业务写入；后续 P0-06 收口"结果未知"语义


def controlled_write(path: str, content: str, *, vault_root: str | None = None,
                     overwrite: bool = True, actor: str = "pipeline") -> dict:
    """文本受控写入。返回 {path, bytes, revision, previous_revision, overwritten}。

    overwrite=False 且文件已存在时不写入，overwritten=False 直接返回现有 revision。
    """
    abs_path = os.path.abspath(path)
    guard_vault_path(vault_root, abs_path)
    if not overwrite and os.path.exists(abs_path):
        with open(abs_path, "rb") as f:
            old = f.read()
        return {"path": path, "bytes": len(old), "revision": hashlib.sha256(old).hexdigest(),
                "previous_revision": hashlib.sha256(old).hexdigest(), "overwritten": False}
    prev = None
    if os.path.exists(abs_path):
        with open(abs_path, "rb") as f:
            prev = hashlib.sha256(f.read()).hexdigest()
    data = content.encode("utf-8")
    revision = hashlib.sha256(data).hexdigest()
    with _path_lock(abs_path):
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        tmp = f"{abs_path}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, abs_path)
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
    _audit(vault_root, {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "actor": actor, "operation": "write", "result": "written",
                        "path": (os.path.relpath(abs_path, os.path.abspath(vault_root))
                                 if vault_root and _under(abs_path, vault_root)
                                 else os.path.basename(abs_path)).replace("\\", "/"),
                        "previous_revision": prev, "revision": revision,
                        "bytes": len(data), "overwrite": overwrite})
    return {"path": path, "bytes": len(data), "revision": revision,
            "previous_revision": prev, "overwritten": prev is not None}
