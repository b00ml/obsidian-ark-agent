"""长期记忆工具：ark/memory/ 下沉淀可复用观点（Markdown 真源）。

F5-011: 记忆从 SQLite 迁移到 Markdown 文件，用户可在 Obsidian 中审阅/编辑/删除。
- 旧实现: .agent-brain/memory/sessions.sqlite（隐藏，用户不可见）
- 新实现: ark/memory/{core,context,procedures,decisions,sessions,archive}/*.md

向后兼容：MCP 工具签名不变，内部路由到 MemoryMarkdownStore。
旧 SQLite 保留为一次性迁移源（见 scripts/migrate_memory_to_markdown.py）。
"""
import datetime
import os
import re
import sqlite3
from contextlib import contextmanager
from typing import Optional

from common import brain_dir

# F5-011: Import Markdown store
try:
    import sys
    agentlab_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agentlab")
    if agentlab_path not in sys.path:
        sys.path.insert(0, agentlab_path)
    from agentlab.memory.markdown_store import MemoryMarkdownStore
    _MARKDOWN_AVAILABLE = True
except ImportError:
    _MARKDOWN_AVAILABLE = False


def _memory_home(config: dict) -> str:
    d = os.path.join(brain_dir(config), "memory")
    os.makedirs(d, exist_ok=True)
    return d


def _db(config: dict) -> str:
    return os.path.join(_memory_home(config), "sessions.sqlite")


def _get_markdown_store(config: dict) -> Optional[MemoryMarkdownStore]:
    """获取 Markdown store 实例（如果可用）。"""
    if not _MARKDOWN_AVAILABLE:
        return None

    vault_root = config.get("vault_root")
    if not vault_root:
        return None

    return MemoryMarkdownStore(vault_root)


@contextmanager
def _connect(config: dict):
    """sqlite 连接：提交事务并确保关闭（Windows 下防文件占用）"""
    conn = sqlite3.connect(_db(config))
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_db(config: dict) -> None:
    with _connect(config) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                source_session TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )"""
        )


def _ensure_profile(config: dict) -> None:
    """首次运行时创建 agent-profile.md（等价写给 agent 的偏好说明）"""
    profile = os.path.join(_memory_home(config), "agent-profile.md")
    if os.path.exists(profile):
        return
    content = (
        "# Agent Profile\n\n"
        "> 用户偏好 / 技术栈 / 项目目标（可复用观点沉淀见 sessions.sqlite）\n\n"
        "## 用户偏好\n- 沟通语言：中文\n"
        "- 产物先入 Inbox/ 审核后再入 wiki/；raw/ 只读\n"
        "## 技术栈\n- Python 3.10+ / Obsidian / MCP\n"
        "## 项目目标\n- 把非结构化输入加工为结构化 Markdown 笔记，用 [[wikilink]] 建图谱\n"
    )
    with open(profile, "w", encoding="utf-8") as f:
        f.write(content)


def _normalize_tags(tags) -> list[str]:
    """tags 归一：list[str] | 逗号/顿号分隔 str | None → 去重清洗列表（≤8 个）。

    OPT-135 实测 bug 修复：模型直接传字符串时，旧实现 or t in (tags or [])
    会对字符串逐字符迭代 → ','.join 后存成 'i,n,b,o,x,…' 污染整库 tag 检索。
    """
    if tags is None:
        return []
    if isinstance(tags, str):
        tags = re.split(r"[,，、;；\s]+", tags)
    out: list[str] = []
    for t in tags:
        t = str(t).strip()
        if t and t not in out:
            out.append(t)
    return out[:8]


def _topic_terms(topic: str) -> list[str]:
    """召回检索词：整句 + 分词 + CJK 二元组（≤12 个）。

    OPT-135：旧实现整句 LIKE（'%今天做了什么%'）要求 content 含完整连续子串，
    几乎永远零命中——自动注入（recall.py 传整句输入）实际失效。
    """
    terms: list[str] = []
    if 2 <= len(topic) <= 24:
        terms.append(topic)
    for tok in re.split(r"[\s,，。;；、/|?!！？]+", topic):
        if len(tok) >= 2 and tok not in terms:
            terms.append(tok)
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", topic):
        for i in range(len(run) - 1):
            bg = run[i:i + 2]
            if bg not in terms:
                terms.append(bg)
    return terms[:12]


def memory_commit(config: dict, content: str, tags: list[str] | None = None,
                  source_session: str = "") -> dict:
    """沉淀一条可复用记忆（content + tags）。

    F5-011: 优先路由到 Markdown store，失败时回退到 SQLite。
    """
    content = (content or "").strip()
    if not content:
        raise ValueError("记忆内容不能为空")

    tags = _normalize_tags(tags)

    # F5-011: 尝试 Markdown store
    md_store = _get_markdown_store(config)
    if md_store:
        try:
            # 从 config 推断 project_id（如果存在）
            project_id = config.get("project_id", "default")

            mem_id = md_store.commit(
                content=content,
                tags=tags,
                mem_type="context",  # 默认为 context，可根据 tags 推断
                project_id=project_id,
                source_session=source_session
            )
            return {
                "status": "committed",
                "id": mem_id,
                "tags": tags,
                "storage": "markdown",
                "created_at": datetime.datetime.now().isoformat(timespec="seconds")
            }
        except Exception as e:
            # 降级到 SQLite
            pass

    # 回退到 SQLite（保持向后兼容）
    _ensure_db(config)
    _ensure_profile(config)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with _connect(config) as conn:
        cur = conn.execute(
            "INSERT INTO memories (content, tags, source_session, created_at) "
            "VALUES (?, ?, ?, ?)",
            (content, ",".join(tags), source_session, now),
        )
        mid = cur.lastrowid
    return {"status": "committed", "id": mid, "tags": tags, "storage": "sqlite", "created_at": now}


def memory_query(config: dict, topic: str, limit: int = 10) -> dict:
    """按 topic 召回记忆（多关键词 OR 命中 + 命中数排序，OPT-135）。

    F5-011: 优先查询 Markdown store，合并 SQLite 结果（兼容期）。
    """
    topic = (topic or "").strip()
    if not topic or limit <= 0:
        return {"topic": topic, "total": 0, "results": []}

    results = []

    # F5-011: 查询 Markdown store
    md_store = _get_markdown_store(config)
    if md_store:
        try:
            project_id = config.get("project_id")
            md_results = md_store.query(topic, limit=limit, project_id=project_id)

            # 转换为兼容格式
            for mem in md_results:
                results.append({
                    "id": mem["id"],
                    "content": mem["content"],
                    "tags": mem.get("tags", []),
                    "source_session": mem.get("source_session", ""),
                    "created_at": mem.get("created_at", ""),
                    "storage": "markdown"
                })
        except Exception as e:
            pass

    # 查询 SQLite（兼容期：合并结果）
    _ensure_db(config)
    terms = _topic_terms(topic)
    if terms:
        where = " OR ".join(["tags LIKE ? OR content LIKE ?"] * len(terms))
        params = [p for t in terms for p in (f"%{t}%", f"%{t}%")]
        with _connect(config) as conn:
            rows = conn.execute(
                f"SELECT id, content, tags, source_session, created_at "
                f"FROM memories WHERE {where} ORDER BY id DESC LIMIT ?",
                (*params, min(limit * 3, 60)),
            ).fetchall()

        def _hits(r) -> int:
            text = f"{r[1]},{r[2]}".lower()
            return sum(1 for t in terms if t.lower() in text)

        rows = sorted(rows, key=lambda r: (-_hits(r), -r[0]))[:limit]
        for r in rows:
            results.append({
                "id": str(r[0]),
                "content": r[1],
                "tags": [t for t in r[2].split(",") if t],
                "source_session": r[3],
                "created_at": r[4],
                "storage": "sqlite"
            })

    # 去重并限制数量
    seen_content = set()
    unique_results = []
    for mem in results:
        content_key = mem["content"][:100]  # 用前100字符去重
        if content_key not in seen_content:
            seen_content.add(content_key)
            unique_results.append(mem)
            if len(unique_results) >= limit:
                break

    return {"topic": topic, "total": len(unique_results), "results": unique_results}


def repair_split_tags(config: dict) -> dict:
    """修复历史 char-split 污染行（OPT-135）：'i,n,b,o,x,,,收' 重组回 'inbox,收'。

    判据：按 ',' 拆分后非空片段 ≥2 且 ≥70% 为单字符（正常 tag 词几乎不可能全单字）；
    重组时空片段即原串中被拆开的 ',' 字符本身。返回 {scanned, fixed:[{id,old,new}]}。

    注：仅适用于 SQLite 旧库，Markdown store 已在 commit 时修复。
    """
    _ensure_db(config)
    fixed: list[dict] = []
    scanned = 0
    with _connect(config) as conn:
        rows = conn.execute("SELECT id, tags FROM memories WHERE tags != ''").fetchall()
        for rid, tags in rows:
            scanned += 1
            parts = tags.split(",")
            nonempty = [p for p in parts if p]
            if len(nonempty) < 2:
                continue
            singles = sum(1 for p in nonempty if len(p) == 1)
            if singles / len(nonempty) < 0.7:
                continue
            rebuilt = _normalize_tags("".join(p if p else "," for p in parts))
            newval = ",".join(rebuilt)
            if newval and newval != tags:
                conn.execute("UPDATE memories SET tags=? WHERE id=?", (newval, rid))
                fixed.append({"id": rid, "old": tags, "new": newval})
    return {"scanned": scanned, "fixed": fixed}
