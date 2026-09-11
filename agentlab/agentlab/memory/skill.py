"""程序记忆：按任务标签加载技能说明注入 system prompt（docs/03 §4.3）。

渐进披露：只注入 SKILL.md 的 frontmatter 描述，正文由模型按需读取。
"""
from __future__ import annotations

import re
from pathlib import Path


def _extract_frontmatter(text: str) -> tuple[dict, str]:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm = {}
            for line in parts[1].splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    fm[k.strip()] = v.strip().strip("'\"")
            return fm, parts[2]
    return {}, text


def resolve_skills_dir(skill_dir: str, repo_root: Path | None = None) -> Path:
    """技能目录解析（P1-2/OPT-114）：相对路径先按进程 cwd，落空再按仓库根兜底。

    serve 以 agentlab/ 为 cwd 启动，config 里的仓库根相对路径（如 "skills"）在
    cwd 下不存在——此前 `.trae/skills` 一直静默解析失败、技能注入形同虚设。
    兜底以本模块源码位置锚定仓库根（agentlab 源码随仓库分发，位置稳定）。
    绝对路径直通；全部落空返回原 Path（load_skills 对不存在目录静默返回空）。
    """
    p = Path(skill_dir)
    if p.is_dir():
        return p
    root = repo_root or Path(__file__).resolve().parents[3]
    cand = root / skill_dir
    if cand.is_dir():
        return cand
    return p


def load_skills(skill_dir: str, task_tags: list[str] | None = None) -> str:
    """扫技能目录下各技能的 SKILL.md，返回匹配标签的说明块。

    匹配规则（简单 OR）：frontmatter 的 name / description 命中任一 task_tags。
    """
    root = Path(skill_dir)
    if not root.is_dir():
        return ""
    tags = set(task_tags or [])
    blocks: list[str] = []
    for sk_md in sorted(root.rglob("SKILL.md")):
        raw = sk_md.read_text(encoding="utf-8", errors="replace")
        fm, body = _extract_frontmatter(raw)
        name = fm.get("name", sk_md.parent.name)
        desc = fm.get("description", "")
        if tags and not any(t.lower() in (name + " " + desc).lower() for t in tags):
            continue
        blocks.append(f"- **{name}**: {desc}\n  路径：`{sk_md.parent}`")
    return "\n".join(blocks)