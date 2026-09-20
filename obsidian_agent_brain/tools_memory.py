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
from contextvars import ContextVar
from typing import Optional

from common import brain_dir, vault_root

# F5-011: Import Markdown store
try:
    import sys
    agentlab_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agentlab")
    if agentlab_path not in sys.path:
        sys.path.insert(0, agentlab_path)
    from agentlab.memory.invalidation import DerivedInvalidationCoordinator
    from agentlab.memory.markdown_store import MemoryMarkdownStore
    from agentlab.rag.index_store import RagIndexStore
    _MARKDOWN_AVAILABLE = True
except ImportError:
    _MARKDOWN_AVAILABLE = False


_RUNTIME_DEPENDENCIES: ContextVar[dict | None] = ContextVar(
    "memory_runtime_dependencies", default=None
)


def bind_runtime_dependencies(*, range_gateway=None, task_state_store=None,
                               task_state_id: str = ""):
    """Bind derived invalidation hooks for one Agent request."""
    return _RUNTIME_DEPENDENCIES.set({
        "range_gateway": range_gateway,
        "task_state_store": task_state_store,
        "task_state_id": str(task_state_id or ""),
    })


def reset_runtime_dependencies(token) -> None:
    _RUNTIME_DEPENDENCIES.reset(token)


def _memory_home(config: dict) -> str:
    d = os.path.join(brain_dir(config), "memory")
    os.makedirs(d, exist_ok=True)
    return d


def _db(config: dict) -> str:
    return os.path.join(_memory_home(config), "sessions.sqlite")


def _get_markdown_store(config: dict) -> Optional[MemoryMarkdownStore]:
    """获取 Markdown store 实例（如果可用）。

    OPT-223 根因修复：此前直接 `config.get("vault_root")`——而 agentlab 连接器
    （load_brain_config）注入的键是 `vault_path`，键名不一致导致 Markdown store
    永远拿不到，memory_commit 全部静默落 SQLite（Vault 内 sessions.sqlite 积累
    410 条用户不可见记忆）。统一走 common.vault_root()（注意：该函数只认
    `vault_path` 键，缺失时显式报错；此处捕获后返回 None 由调用方显式失败）。
    """
    if not _MARKDOWN_AVAILABLE:
        return None

    try:
        vault = vault_root(config)
    except Exception:
        return None

    return MemoryMarkdownStore(vault)


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
                  source_session: str = "", importance: int | None = None,
                  mem_type: str = "context", bucket: bool = False,
                  confidence=None, status: str | None = None,
                  source: str = "user", source_ref: str = "", scope=None,
                  session_id: str = "", valid_from: str | None = None,
                  valid_until: str | None = None, subject: str = "",
                  review_due_at: str | None = None, correction_of: str | None = None,
                  candidate_first: bool = False,
                  explicit_confirmation: bool = False) -> dict:
    """沉淀一条可复用记忆（content + tags）。

    F5-011: Markdown 是唯一运行时写入源（用户可在 Obsidian 审阅）。
    OPT-223: 移除 SQLite 静默回退——此前 Markdown store 拿不到/写入失败时
    `except: pass` 后落进用户不可见的 sessions.sqlite 并返回 committed，
    造成"显示成功但 Vault 里没有"的静默数据错位（真实积累 410 条）。
    现在失败即显式抛错，SQLite 仅保留迁移工具与人工恢复入口。
    OPT-224: importance（1-10）由 LLM 语义提取传入；None 落默认 5。
    OPT-225: mem_type 可显式指定（默认 context）；bucket=True 时写入月/项目桶
    （sessions/2026-09.md 式候选层，Agent 主动调用一般不需要）。
    OPT-230 B7: confidence（0-1 或 "hypothesis"）落库；**hypothesis 记忆禁止
    supersedes 已确立结论**（低置信猜测不得替代已确立事实）。
    """
    content = (content or "").strip()
    if not content:
        raise ValueError("记忆内容不能为空")

    tags = _normalize_tags(tags)

    md_store = _get_markdown_store(config)
    if md_store is None:
        raise ValueError(
            "MEMORY_MARKDOWN_UNAVAILABLE: Markdown store 不可用（vault_root 未配置"
            "或 agentlab 包缺失），记忆未写入——SQLite 静默回退已移除（OPT-223）")

    try:
        # 从 config 推断 project_id（如果存在）
        project_id = config.get("project_id", "default")
        bound_project, bound_session = _bound_scope()
        if bound_project:
            project_id = bound_project
        if bound_session and not session_id:
            session_id = bound_session
        if bound_session and not source_session:
            source_session = bound_session
        policy = config.get("memory_policy") or {}
        if not candidate_first and policy.get("write_mode") == "candidate_first" \
                and str(source or "user").lower() not in {"user", "human"}:
            candidate_first = True

        if bucket:
            mem_id = md_store.commit_to_bucket(
                mem_type=mem_type,
                content=content,
                tags=tags,
                project_id=project_id,
                importance=int(importance) if importance else 3,
                source_session=source_session,
                status=status or "active",
                source=source,
                source_ref=source_ref,
                confidence=confidence if confidence is not None else 1.0,
                candidate_first=candidate_first,
            )
        else:
            mem_id = md_store.commit(
                content=content,
                tags=tags,
                mem_type=mem_type,
                project_id=project_id,
                source_session=source_session,
                importance=int(importance) if importance else 5,
                confidence=confidence if confidence is not None else 1.0,
                status=status,
                source=source,
                source_ref=source_ref,
                scope=scope,
                session_id=session_id,
                valid_from=valid_from,
                valid_until=valid_until,
                review_due_at=review_due_at,
                subject=subject,
                correction_of=correction_of,
                candidate_first=candidate_first,
                explicit_confirmation=explicit_confirmation,
            )
    except Exception as e:
        raise RuntimeError(
            f"MEMORY_MARKDOWN_WRITE_FAILED: 记忆未写入（{e}）——"
            "请检查 Vault 可写性；不回退 SQLite（OPT-223）") from e

    return {
        "status": "committed",
        "id": mem_id,
        "tags": tags,
        "storage": "markdown",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds")
    }


def memory_query(config: dict, topic: str, limit: int = 10,
                 session_id: str | None = None,
                 statuses: list[str] | None = None,
                 include_archive: bool = False,
                 min_confidence: float | None = None) -> dict:
    """按 topic 召回记忆（多关键词 OR 命中 + 命中数排序，OPT-135）。
    
    F5-011: 优先查询 Markdown store，合并 SQLite 结果（兼容期）。
    """
    topic = (topic or "").strip()
    if not topic or limit <= 0:
        return {"topic": topic, "total": 0, "results": []}

    # OPT-223: 只查 Markdown 真源——SQLite 合并已移除（禁止默认静默双源召回）；
    # 历史数据用 scripts/migrate_memory_to_markdown.py 迁移，SQLite 文件保留为
    # 人工恢复入口（_connect/_ensure_db 不再被运行时调用）。
    md_store = _get_markdown_store(config)
    if md_store is None:
        raise ValueError(
            "MEMORY_MARKDOWN_UNAVAILABLE: Markdown store 不可用（vault_root 未配置"
            "或 agentlab 包缺失），记忆查询终止——SQLite 合并查询已移除（OPT-223）")
    try:
        project_id = config.get("project_id")
        bound_project, bound_session = _bound_scope()
        if bound_project:
            project_id = bound_project
        if bound_session and not session_id:
            session_id = bound_session
        policy = config.get("memory_policy") or {}
        md_results = md_store.query(
            topic, limit=limit, project_id=project_id,
            session_id=session_id or config.get("session_id"),
            statuses=statuses,
            include_archive=include_archive,
            min_confidence=(policy.get("recall_min_confidence", 0.0)
                            if min_confidence is None else min_confidence),
            allow_default_shared=bool(policy.get("allow_default_shared", True)),
        )
    except Exception as e:
        raise RuntimeError(
            f"MEMORY_MARKDOWN_QUERY_FAILED: 记忆查询失败（{e}）——"
            "Markdown 真源读取异常，不回退 SQLite（OPT-223）") from e

    results = []
    for mem in md_results:
        results.append({
            "id": mem["id"],
            "content": mem["content"],
            "tags": mem.get("tags", []),
            "status": mem.get("status", "active"),
            "bucket": bool(mem.get("bucket", False)),
            "project_id": mem.get("project_id", "default"),
            "scope": mem.get("scope", {}),
            "confidence": mem.get("confidence", 1.0),
            "subject": mem.get("subject", ""),
            "valid_from": mem.get("valid_from", ""),
            "valid_until": mem.get("valid_until", ""),
            "source": mem.get("source", "legacy"),
            "source_ref": mem.get("source_ref", ""),
            "source_session": mem.get("source_session", ""),
            "created_at": mem.get("created_at", ""),
            "storage": "markdown"
        })

    audit = getattr(md_store, "last_query_audit", None)
    payload = {"topic": topic, "total": len(results), "results": results}
    if audit is not None and getattr(audit, "filtered_reasons", None):
        payload["filtered_reasons"] = dict(audit.filtered_reasons)
    return payload


def memory_conflicts(config: dict, limit: int = 100,
                     session_id: str | None = None) -> dict:
    """List explicit unresolved memory conflicts for human review only."""
    md_store = _get_markdown_store(config)
    if md_store is None:
        raise ValueError("MEMORY_MARKDOWN_UNAVAILABLE: Markdown store 不可用")
    project_id = config.get("project_id")
    bound_project, bound_session = _bound_scope()
    if bound_project:
        project_id = bound_project
    if bound_session and not session_id:
        session_id = bound_session
    rows = md_store.list_conflicts(project_id=project_id,
                                   session_id=session_id or config.get("session_id"),
                                   limit=limit)
    return {"status": "ok", "total": len(rows), "items": rows}


def _memory_store_action(config: dict, action: str, mem_id: str, **kwargs) -> dict:
    """Shared wrapper for user correction/revocation/deletion/restore."""
    md_store = _get_markdown_store(config)
    if md_store is None:
        raise ValueError("MEMORY_MARKDOWN_UNAVAILABLE: Markdown store 不可用")
    try:
        if action in {"correct", "revoke", "delete"}:
            index_path = md_store.vault_root / ".agent-brain" / "rag-index-p2.sqlite"
            rag_index = RagIndexStore(index_path, None, vault_root=md_store.vault_root) \
                if index_path.exists() else None
            deps = _RUNTIME_DEPENDENCIES.get() or {}
            task_store = deps.get("task_state_store")
            task_id = str(deps.get("task_state_id") or "")

            def invalidate_task_state(*, memory_id: str, source_ref: str = "") -> bool:
                if task_store is None or not task_id:
                    return False
                return bool(task_store.invalidate_memory(
                    task_id, memory_id=memory_id, source_ref=source_ref))

            coordinator = DerivedInvalidationCoordinator(
                md_store,
                rag_index=rag_index,
                range_gateway=deps.get("range_gateway"),
                task_state_invalidator=(invalidate_task_state
                                        if task_store is not None and task_id else None),
            )
            if action == "correct":
                result = coordinator.correct(mem_id, kwargs.pop("content", ""), **kwargs)
                value = result.derived.get("successor") if result.source_updated else None
            elif action == "revoke":
                result = coordinator.revoke(mem_id, **kwargs)
                value = result.source_updated
            else:
                result = coordinator.delete(mem_id, **kwargs)
                value = result.source_updated
            return {
                "status": "ok" if result.complete else "partial",
                "action": action,
                "id": mem_id,
                "result": value,
                "invalidation": result.to_dict(),
            }
        elif action == "restore":
            value = md_store.restore(mem_id, **kwargs)
        else:
            raise ValueError(f"unknown memory action: {action}")
    except Exception as exc:
        raise RuntimeError(f"MEMORY_{action.upper()}_FAILED: {exc}") from exc
    return {"status": "ok" if value else "not_found", "action": action,
            "id": mem_id, "result": value}


def _bound_scope() -> tuple[str, str]:
    """Read trusted request scope when called from agentlab serve.

    The brain package also runs standalone in tests/MCP, where agentlab may not
    be importable; in that case an empty scope preserves the legacy behavior.
    """
    try:
        from agentlab.contracts import current_retrieval_scope
        scope = current_retrieval_scope()
        return scope.project_id, scope.session_id
    except Exception:
        return "", ""


def memory_correct(config: dict, mem_id: str, content: str,
                   tags: list[str] | None = None,
                   reason: str = "user_correction") -> dict:
    return _memory_store_action(config, "correct", mem_id, content=content,
                                tags=tags, reason=reason)


def memory_revoke(config: dict, mem_id: str,
                  reason: str = "user_revoked") -> dict:
    return _memory_store_action(config, "revoke", mem_id, reason=reason)


def memory_delete(config: dict, mem_id: str,
                  reason: str = "user_delete", hard: bool = True) -> dict:
    return _memory_store_action(config, "delete", mem_id, reason=reason, hard=hard)


def memory_restore(config: dict, mem_id: str,
                   reason: str = "user_restore") -> dict:
    return _memory_store_action(config, "restore", mem_id, reason=reason)


def memory_review(config: dict, mem_id: str, decision: str, reviewer: str,
                  expected_content_hash: str, reason: str,
                  defer_until: str = "") -> dict:
    """Explicitly confirm or defer a review-due memory.

    The content hash is an optimistic lock from the read-only review queue;
    stale queue entries fail closed instead of overwriting a newer edit.
    """
    md_store = _get_markdown_store(config)
    if md_store is None:
        raise ValueError("MEMORY_MARKDOWN_UNAVAILABLE: Markdown store 不可用")
    try:
        path = md_store._find_memory_file(mem_id)
        if path is None:
            return {"status": "not_found", "action": "review", "id": mem_id}
        memory = md_store._parse_memory_file(path)
        project_id = config.get("project_id")
        bound_project, bound_session = _bound_scope()
        project_id = bound_project or project_id
        memory_project = str(memory.get("project_id") or "default")
        if project_id and memory_project not in {"default", str(project_id)}:
            raise ValueError("memory review scope denied")
        memory_session = str(memory.get("session_id") or "")
        if bound_session and memory_session and memory_session != bound_session:
            raise ValueError("memory review session scope denied")
        result = md_store.review(
            mem_id,
            decision=decision,
            reviewer=reviewer,
            expected_content_hash=expected_content_hash,
            reason=reason,
            defer_until=defer_until or None,
        )
        return {**result, "status": "ok", "action": "review"}
    except Exception as exc:
        raise RuntimeError(f"MEMORY_REVIEW_FAILED: {exc}") from exc


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
