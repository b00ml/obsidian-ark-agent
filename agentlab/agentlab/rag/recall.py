"""多路召回与融合排序（docs/04 §4.2，F9；P0-1/OPT-105 升级 RRF + 向量路）。

以"检索即工具、由 agent 决策"为原则，本模块只做编排：
- 每路召回是可注入的 `Recaller`（返回统一 {title, content, ref, source} 形状），
  默认由 brain 的 vault_search / memory_query 与框架 web_search 充当，
  可经 `RAGRecall(extra=[...])` 追加向量语义路（rag/vector_index，source="vector"）；
- 融合 `fuse()` 为纯函数（不依赖 brain/网络），**RRF（Reciprocal Rank Fusion）**
  按路内排名打分（对齐竞品规划 P0-1），替代旧"命中分+路权重"线性加权；
- 去重按 ref。

不持有全局状态；brain 未就绪时对应路降级为空，不抛错。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

# 切词：保留中文词串，英文/数字按空白切；中文整串作为低频关键词更准
_KEYWORD_RE = re.compile(
    r"([\u4e00-\u9fff]{2,})|([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*)"
)


def split_keywords(query: str) -> list[str]:
    """拆分查询为关键词：中文按连续字串、英文/标识符按 token。去重保值序。"""
    seen: set[str] = set()
    out: list[str] = []
    for zh, en in _KEYWORD_RE.findall(query or ""):
        tok = zh or en
        tok = tok.strip()
        if tok.lower() in seen:
            continue
        seen.add(tok.lower())
        out.append(tok)
    return out


_ASCII_RE = re.compile(r"^[a-zA-Z0-9_.]+$")


def retrieval_terms(query: str) -> list[str]:
    """生成适合"关键词索引"的检索词。

    brain 的 vault_search / memory_query 基于子串/关键词匹配，直接喂整句或
    过长中文串命中为 0；本函数对长中文词做 2/4 字滑动窗口补充检索词，
    英文/标识符保持原样。配合 fuse 按 ref 去重，天然并集不膨胀。
    """
    terms: list[str] = []
    for kw in split_keywords(query):
        if len(kw) > 4 and not _ASCII_RE.match(kw):
            terms.extend(kw[i:i + 4] for i in range(0, len(kw) - 3))
            terms.extend(kw[i:i + 2] for i in range(0, len(kw) - 1))
        else:
            terms.append(kw)
    # 去重、剔除 2 字停用噪声；保值序
    out: list[str] = []
    seen: set[str] = set()
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        if len(t) >= 2 or _ASCII_RE.match(t) and len(t) > 1:
            out.append(t)
    return out


def _union_recall(recall_fn, query: str, limit: int) -> list[dict]:
    """按检索词逐词召回并去重（保留首命中的内容/标题）。

    recall_fn 返回 list 或 dict（兼容 MemoryStore 的 {status, total, results}）。
    """
    rows: dict = {}

    def _rows(out):
        if isinstance(out, dict):
            return out.get("results") or []
        return out or []

    for term in retrieval_terms(query):
        for o in _rows(recall_fn(term, limit)):
            key = o.get("ref") or o.get("path") or o.get("id") or o.get("title")
            if key:
                rows.setdefault(key, o)
    return list(rows.values())


@dataclass
class RecallItem:
    title: str
    content: str = ""
    ref: str = ""            # 来源引用：文件路径 / 网络 URL / 记忆 id
    source: str = "vault"    # 路径来源：vault / memory / graph / web
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "title": self.title, "content": self.content,
            "ref": self.ref, "source": self.source, "score": round(self.score, 3),
        }


Recaller = Callable[[str], list[dict]]
"""每路召回签名：query -> list[{title, content, ref, source}]（source 可省略）。"""

# RRF 常数（对齐竞品规划 P0-1）：score(d) = Σ_路 1/(K + rank)，K=60 为业界常用值
_RRF_K = 60

# 注入治理（OPT-089 / 学 tau Tier-1/Tier-2）：单条召回正文预算 + 截断保留尾部
_PER_ITEM_CHARS = 1200   # 每条召回正文单条预算（字符）
_TAIL_CHARS = 240        # 截断时保留的尾部长度（防切断关键结论/引用）


def govern_recall(items, per_item_chars=_PER_ITEM_CHARS, max_items=None) -> list[RecallItem]:
    """把三层治理下沉到 recall 注入（OPT-089 / 学 tau Tier-1/Tier-2）。

    - Tier-1：单条 content 超预算 → head+tail 截断，留 `…[truncated N chars]…` 标记。
    - Tier-2：无论截不截，`title/ref/source`（身份锚）始终保留；项数按 max_items 封顶。
    纯函数、零 LLM，返回新列表，不污染原 items。
    """
    out: list[RecallItem] = []
    span = items if max_items is None else items[:max_items]
    for it in span:
        c = it.content or ""
        if per_item_chars > 0 and len(c) > per_item_chars:
            c = (c[:per_item_chars - _TAIL_CHARS]
                 + f"\n…[truncated {len(c) - per_item_chars} chars]…\n"
                 + c[-_TAIL_CHARS:])
        out.append(RecallItem(title=it.title, content=c, ref=it.ref,
                              source=it.source, score=it.score))
    return out


def _hit_score(kws: list[str], title: str, ref: str, content: str) -> int:
    """命中分：关键词出现在 标题/引用(+3) 或 正文(+1)。"""
    s = 0
    low_title, low_ref, low_content = title.lower(), ref.lower(), content.lower()
    for k in kws:
        low = k.lower()
        if low in low_title or (low_ref and low in low_ref):
            s += 3
        elif low in low_content:
            s += 1
    return s


def fuse(query: str, candidates: list[RecallItem], k: int = 8) -> list[RecallItem]:
    """融合（P0-1/OPT-105）：RRF 按路内排名打分 → 同 ref 去重 → top-k。纯函数。

    - 候选按 source 分组，组内顺序即该路排名：score(d) = Σ 1/(_RRF_K + rank)；
    - 同 ref 跨路去重保留先出现者（各路分数相同，先到者优先）；
    - `_hit_score` 降级为同分 tiebreak（确定性排序，测试友好）。
    """
    kws = split_keywords(query)
    # 按路分组（保持到达顺序），组内顺序即该路排名
    by_source: dict[str, list[str]] = {}
    for it in candidates:
        key = it.ref or f"{it.source}:{it.title}"
        keys = by_source.setdefault(it.source, [])
        if key not in keys:
            keys.append(key)
    scores: dict[str, float] = {}
    for keys in by_source.values():
        for rank, key in enumerate(keys, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
    by_ref: dict[str, RecallItem] = {}
    for it in candidates:
        key = it.ref or f"{it.source}:{it.title}"
        by_ref.setdefault(key, it)  # 同 ref 保留先出现者
    ranked = sorted(
        by_ref.values(),
        key=lambda x: (scores[x.ref or f"{x.source}:{x.title}"],
                       _hit_score(kws, x.title, x.ref, x.content)),
        reverse=True,
    )
    for x in ranked:
        x.score = round(scores[x.ref or f"{x.source}:{x.title}"], 4)
    return ranked[:k]


class RAGRecall:
    """多路召回编排器。默认借用 brain 工具做本地召回 + 框架 web_search 做全网。

    可注入自定义 recallers（离线测试用 mock），或注入 brain_config 走生产召回。
    """

    def __init__(self, recallers: list[Recaller] | None = None,
                 extra: list[Recaller] | None = None):
        base = list(recallers) if recallers else self._default_recallers()
        self._recallers = base + list(extra or [])  # extra：向量路等增强路（P0-1）

    @staticmethod
    def _default_recallers(brain_config: dict | None = None) -> list[Recaller]:
        """默认三路：vault_search + memory_query（brain）+ web_search（框架）。

        brain 未就绪时 MemoryStore 自动降级为空；不抛错。
        """
        lst: list[Recaller] = []

        from agentlab.memory.store import MemoryStore
        store = MemoryStore(brain_config)
        if store.available:
            def vault(q: str, store=store) -> list[dict]:
                return [
                    {"title": o.get("path", ""), "content": o.get("content", ""),
                     "ref": o.get("path", ""), "source": "vault"}
                    for o in _union_recall(store.search, q, limit=20)
                ]

            def mem(q: str, store=store) -> list[dict]:
                return [
                    {"title": o.get("content", "")[:40], "content": o.get("content", ""),
                     "ref": f"memory#{o.get('id', '')}", "source": "memory"}
                    for o in _union_recall(store.query, q, limit=10)
                ]

            lst.append(vault)
            lst.append(mem)

        from agentlab.tools.web_search import web_search

        def net(q: str) -> list[dict]:
            try:
                rows = web_search(q, limit=3)
            except Exception:
                rows = []
            return [
                {"title": o.get("title", ""), "content": o.get("snippet", ""),
                 "ref": o.get("url", ""), "source": "web"}
                for o in rows if o.get("url")
            ]

        lst.append(net)
        return lst

    def retrieve(self, query: str, limit: int = 8) -> list[RecallItem]:
        """多路并行召回 → 融合排序 → top-limit。任一路异常降级，不影响整体。"""
        candidates: list[RecallItem] = []
        for rec in self._recallers:
            try:
                rows = rec(query) or []
            except Exception:
                continue
            for row in rows:
                candidates.append(
                    RecallItem(
                        title=str(row.get("title", "")),
                        content=str(row.get("content", "")),
                        ref=str(row.get("ref", "")),
                        source=str(row.get("source", "vault")),
                    )
                )
        return fuse(query, candidates, k=limit)

    def retrieve_governed(self, query: str, limit: int = 8,
                          per_item_chars: int = _PER_ITEM_CHARS) -> list[RecallItem]:
        """召回 + 注入治理：Tier-1 单条截断、Tier-2 身份锚保留（OPT-089）。

        供注入入口（rag_retrieve）使用：在源头控住每条正文与总项数，
        不把治理责任全甩给下游 `max_tool_result_chars`。
        """
        return govern_recall(self.retrieve(query, limit=limit),
                             per_item_chars=per_item_chars, max_items=limit)