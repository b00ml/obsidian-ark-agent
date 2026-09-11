"""长期记忆封装（docs/03 §4.2，UC-3 跨会话召回）。

MemoryStore 包装 brain 的 memory_commit / memory_query / vault_search：
- commit：沉淀一条可复用观点（复用 brain 去重）
- query：按 topic 召回记忆（tag + 内容关键词）
- search：全库语义召回（vault_search 关键词）

复用 brain 应用层工具（同进程直调），不重复加载；brain 不可用时优雅降级为空/错误，
不抛错（对齐 build_brain_tools 的"未就绪返回空"惯例）。
"""
from __future__ import annotations

import datetime
import importlib
import re
from typing import Callable

from agentlab.tools.connectors.brain_tools import BRAIN_PKG_DIR, _add_brain_path
from agentlab.memory.capture import CAPTURE_ITEM_CHARS, govern_capture

# 去重相似度阈值（对齐 docs/07 §4.3 open-note 记忆去重）
_DUP_THRESHOLD = 0.72
# 召回时间衰减默认半衰期（天，0=关闭；open-note 借鉴点①：旧记忆降权，避免永久占用召回位）
DEFAULT_DECAY_HALF_LIFE_DAYS = 30.0


def _tokens(text: str) -> set[str]:
    """轻度切词：英文/数字整 token + 中文相邻二元组，用于相似度近似。"""
    toks: set[str] = set()
    for m in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{1,}|\d+(?:\.\d+)?", text or ""):
        toks.add(m.lower())
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", text or ""):
        for i in range(len(run) - 1):
            toks.add(run[i:i + 2])
    return toks


def _similar(a: str, b: str) -> float:
    """以新增内容视角的覆盖率：a 的 token 有多少出现在 b。"""
    mine, other = _tokens(a), _tokens(b)
    if not mine or not other:
        return 0.0
    return len(mine & other) / len(mine)


def _decay_factor(created_at: str, now: datetime.datetime, half_life_days: float) -> float:
    """置信度时间衰减：0.5 ** (age_days / half_life)。

    created_at 缺失/不可解析 → 1.0（不惩罚，向后兼容无时间戳的旧记录）；
    half_life_days <= 0 → 1.0（衰减关闭）。
    """
    if not created_at or half_life_days <= 0:
        return 1.0
    try:
        created = datetime.datetime.fromisoformat(str(created_at))
    except ValueError:
        return 1.0
    age_days = max((now - created).total_seconds(), 0.0) / 86400.0
    return 0.5 ** (age_days / half_life_days)


class MemoryStore:
    def __init__(self, brain_config: dict | None = None,
                 per_item_chars: int = CAPTURE_ITEM_CHARS,
                 decay_half_life_days: float = DEFAULT_DECAY_HALF_LIFE_DAYS,
                 now_fn: Callable[[], datetime.datetime] | None = None):
        self._brain_config = brain_config or {}
        self._per_item_chars = per_item_chars
        self._decay_half_life_days = float(decay_half_life_days)
        self._now_fn = now_fn or datetime.datetime.now  # 可注入时钟，便于单测
        _add_brain_path()  # 幂等：确保 brain 平铺模块可 flat import
        self._tm = self._import("tools_memory")
        self._tv = self._import("tools_vault")

    @staticmethod
    def _import(name: str):
        try:
            return importlib.import_module(name)
        except Exception:  # brain 未就绪（无目录/依赖缺失）→ 降级为 None
            return None

    @property
    def available(self) -> bool:
        return self._tm is not None

    def commit(self, content: str, tags: list[str] | None = None,
               source_session: str = "", dedup: bool = False) -> dict:
        """沉淀一条可复用记忆（content + tags）。

        dedup=True 时先查已有记忆做相似度去重，高度重复则返回
        `{"status":"skipped_duplicate","similar":...}`，避免重复沉淀。
        默认 False 以保持 API 语义不变，由捕获流程显式开启。
        写入前经 `govern_capture` 治理（OPT-090：Tier-1 单条截断），身份锚不丢。
        """
        if self._tm is None:
            return {"status": "unavailable"}
        content = govern_capture(content or "", self._per_item_chars)
        if dedup:
            dup = self._find_duplicate(content, tags)
            if dup is not None:
                return {"status": "skipped_duplicate", "similar": dup}
        try:
            return self._tm.memory_commit(self._brain_config, content, tags, source_session)
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _find_duplicate(self, content: str, tags: list[str] | None) -> str | None:
        """按 tags[0]（或内容前 6 字）召回已有记忆，返回相似度达阈值的那条；无则 None。"""
        try:
            topic = (tags[0] if tags else (content or "")[:6]) or "记忆"
            q = self._tm.memory_query(self._brain_config, topic, 10)
        except Exception:
            return None
        results = q.get("results") if isinstance(q, dict) else (getattr(q, "results", None) or [])
        best, best_overlap = None, 0.0
        for r in results or []:
            c = (r.get("content") if isinstance(r, dict) else getattr(r, "content", "")) or ""
            if not c.strip():
                continue
            o = _similar(content, c)
            if o > best_overlap:
                best_overlap, best = o, c
        return best if best_overlap >= _DUP_THRESHOLD else None

    def query(self, topic: str, limit: int = 10, *, apply_decay: bool = True) -> dict:
        """按 topic 召回记忆。

        apply_decay（默认开）且半衰期 > 0 时：超取 3 倍候选，按
        `brain 排名分(1/(rank+1)) × 时间衰减因子` 重排后截回 limit——
        旧记忆降权，避免"旧记忆永久占用召回位"（OPT-101）。
        """
        if self._tm is None:
            return {"status": "unavailable", "total": 0, "results": []}
        decay_on = apply_decay and self._decay_half_life_days > 0
        fetch = limit * 3 if decay_on else limit
        try:
            q = self._tm.memory_query(self._brain_config, topic, fetch)
        except Exception as e:
            return {"status": "error", "error": str(e), "total": 0, "results": []}
        results = q.get("results") if isinstance(q, dict) else (getattr(q, "results", None) or [])
        results = list(results or [])
        if decay_on and results:
            now = self._now_fn()

            def _score(idx_item):
                idx, r = idx_item
                created = (r.get("created_at") if isinstance(r, dict)
                           else getattr(r, "created_at", "")) or ""
                return (1.0 / (idx + 1)) * _decay_factor(created, now, self._decay_half_life_days)

            results = [r for _, r in sorted(enumerate(results), key=_score, reverse=True)]
        results = results[:limit]
        return {"topic": q.get("topic", topic) if isinstance(q, dict) else topic,
                "total": len(results), "results": results}

    def search(self, keyword: str, limit: int = 20) -> dict:
        """全库语义召回（vault_search）。"""
        if self._tv is None:
            return {"status": "unavailable", "total": 0, "results": []}
        try:
            return self._tv.vault_search(self._brain_config, keyword, limit=limit)
        except Exception as e:
            return {"status": "error", "error": str(e), "total": 0, "results": []}