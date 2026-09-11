"""Project 长期任务空间（P0-2/OPT-107）：项目标识 → Vault 内规则与背景上下文。

目录约定：`<vault_root>/ark/projects/<project_id>/`（OPT-107 二期：根目录由
copilot 改名 ark，与插件同名；老目录数据手工迁移一次）
- `AGENTS.md`  项目专属规则（优先级高于全局规范，注入 system 时显式标注）
- `project.md` 项目背景与长期上下文

两文件都在 Vault 里、可直接用 Obsidian 编辑；project_id 非法（路径穿越等）或
项目目录不存在 → 返回空块，全局行为不变（降级零成本，不抛错）。
"""
from __future__ import annotations

import re
from pathlib import Path

# project_id 白名单：中英文/数字/下划线/连字符，1~64 位——客户端输入直接拼路径，防穿越
_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff]{1,64}$")


def sanitize_project_id(project_id: str | None) -> str:
    """project_id 白名单校验；非法 → 空串（等价"无项目"）。"""
    pid = (project_id or "").strip()
    if not pid or pid in (".", "..") or not _SAFE_ID.match(pid):
        return ""
    return pid


def project_dir(vault_root: str | Path, project_id: str) -> Path:
    return Path(vault_root) / "ark" / "projects" / project_id


def project_context(vault_root: str | Path, project_id: str | None) -> str:
    """加载项目规则与背景，拼为可注入 system 的文本块；无项目/缺文件 → 空串。"""
    pid = sanitize_project_id(project_id)
    if not pid:
        return ""
    d = project_dir(vault_root, pid)
    parts: list[str] = []
    agents = d / "AGENTS.md"
    if agents.exists():
        body = agents.read_text(encoding="utf-8", errors="ignore").strip()
        if body:
            parts.append(f"## 当前项目规则（{pid} · 优先于全局规范）\n{body}")
    proj = d / "project.md"
    if proj.exists():
        body = proj.read_text(encoding="utf-8", errors="ignore").strip()
        if body:
            parts.append(f"## 项目背景与长期上下文（{pid}）\n{body}")
    return "\n\n".join(parts)


def apply_project_context(base_instructions: str, vault_root: str | Path,
                          project_id: str | None) -> str:
    """把项目块追加到 system 指令后；无块 → 原样返回。文件读取失败降级。"""
    try:
        block = project_context(vault_root, project_id)
    except Exception:
        block = ""
    if not block:
        return base_instructions
    return f"{base_instructions.rstrip()}\n\n{block}"
