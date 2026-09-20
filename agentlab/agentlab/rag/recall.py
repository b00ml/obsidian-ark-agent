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

import hashlib
import inspect
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from agentlab.contracts import (
    RetrievalResult,
    RetrievalScope,
    RetrievalStatus,
    RetrievalStrategy,
    current_retrieval_scope,
    scoped_retrieval_result,
)
from agentlab.rag.query import QueryPlan, build_query_plan, classify_query
from agentlab.rag.rewrite import RewriteResult, rewrite_query_sync

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
_EXPANSION_QUESTION_PREFIX = re.compile(
    r"^(?:什么是|什么|如何|怎么|怎样|为什么|为何|哪个|哪些|谁|何时|何地|"
    r"请问|请告诉我|帮我|我想知道|我想了解)"
)
_EXPANSION_SPLIT_RE = re.compile(r"[,，;；、。！？!?\s]+")
_EXPANSION_STOPWORDS = frozenset({
    "的", "是", "在", "了", "和", "与", "或", "及", "要", "为", "对",
    "这", "那", "什么", "如何", "怎么", "怎样", "为什么", "为何", "哪些",
    "哪个", "哪里", "请问", "请告诉我", "帮我", "我想知道", "我想了解",
    "what", "how", "why", "when", "where", "which", "who",
})
_EXPANSION_GENERIC_TERMS = frozenset({
    # These terms occur throughout the Vault and are useful only when kept
    # inside a phrase.  Sending them as standalone expansion routes lets a
    # broad topic query outrank its named subject.
    "ai", "agent", "知识", "知识库", "笔记", "主题", "内容", "项目",
    "系统", "方法", "经验", "规模", "vault", "obsidian", "memory",
})
_EXPANSION_FUNCTION_CHARS = frozenset(
    "的是在了和与或及要为对这那什么如何怎么怎样为什么为何哪些哪个哪里"
)
_CONTROLLED_SEMANTIC_EXPANSIONS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(r"(?:处理|执行).{0,8}(?:视频|B站).{0,8}(?:前|之前).*(?:注意|核对|准备)", re.IGNORECASE),
        (
            "B站视频处理前核对 bili_meta",
            "收件箱 B站处理前先核对",
            "视频任务处理前核对元信息",
        ),
    ),
)


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


def expand_query_variants(query: str, *, max_variants: int = 5) -> list[str]:
    """Build bounded lexical variants after a low-recall result.

    This mirrors WeKnora's local expansion stage: stopword/question-word
    removal, keyword-only retrieval, and delimiter-separated segments. It is
    deliberately deterministic and never runs for identifier or negative
    queries, so it cannot broaden an unknown BV/file lookup into an answer.
    The original query is excluded; callers add it as the primary route.
    """
    original = " ".join(str(query or "").split())
    if not original or max_variants <= 0:
        return []
    query_type, _confidence, _reason = classify_query(original)
    if query_type in {"empty", "identifier", "negative"}:
        return []
    variants: list[str] = []
    seen = {original.casefold()}

    def add(value: str) -> None:
        value = " ".join(str(value or "").split()).strip(" ,，;；、。！？!?")
        if len(value) < 2 or value.casefold() in seen:
            return
        seen.add(value.casefold())
        variants.append(value)

    # A tiny, auditable synonym bridge for known lexical gaps.  Keep these
    # phrase-level and query-specific: standalone "视频"/"处理" variants
    # would broaden the corpus and increase false positives.  The original
    # query remains authoritative; callers decide shadow vs on.
    for pattern, aliases in _CONTROLLED_SEMANTIC_EXPANSIONS:
        if pattern.search(original):
            for alias in aliases:
                add(alias)
                if len(variants) >= max_variants:
                    return variants[:max_variants]

    def meaningful(token: str) -> bool:
        normalized = token.casefold().strip()
        return (
            normalized not in _EXPANSION_STOPWORDS
            and (len(token) >= 2 or bool(_ASCII_RE.fullmatch(token)))
        )

    # Keep the complete phrase before decomposing it.  This is the strongest
    # local equivalent of WeKnora's phrase expansion and prevents a common
    # noun (for example ``AI`` or ``知识``) from becoming the primary route.
    cleaned = _EXPANSION_QUESTION_PREFIX.sub("", original).strip()
    if cleaned != original:
        add(cleaned)

    segments = [part.strip() for part in _EXPANSION_SPLIT_RE.split(cleaned or original)
                if len(part.strip()) >= 3]
    for segment in sorted(set(segments), key=lambda value: (-len(value), value)):
        add(segment)
        if len(variants) >= max_variants:
            return variants[:max_variants]

    # Token-only form: use existing retrieval tokenization, but drop common
    # function words and one-character CJK noise.
    tokens = [
        token for token in split_keywords(cleaned or original)
        if meaningful(token)
    ]
    if len(tokens) >= 2:
        add(" ".join(tokens))

    # A standalone ASCII token is safe only when it is distinctive enough to
    # act as an entity.  Generic product/topic names remain embedded in the
    # phrase route above.
    for token in tokens:
        if (_ASCII_RE.fullmatch(token)
                and len(token) >= 3
                and token.casefold() not in _EXPANSION_GENERIC_TERMS):
            add(token)
            if len(variants) >= max_variants:
                return variants[:max_variants]

    # Jieba-style short Chinese terms are important for this Vault: broad
    # questions often contain the indexed noun only as a two-character span
    # (主题、规模、目录、处理).  Keep useful spans before four-character
    # windows and drop spans made of grammatical/function characters.
    han_runs = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{2,}", cleaned or original)
    short_terms: list[str] = []
    for run in han_runs:
        for index in range(len(run) - 1):
            term = run[index:index + 2]
            if (any(char in _EXPANSION_FUNCTION_CHARS for char in term)
                    or term.casefold() in _EXPANSION_GENERIC_TERMS):
                continue
            if term not in short_terms:
                short_terms.append(term)
    ordered_short_terms = list(dict.fromkeys(
        short_terms[:2] + short_terms[-2:] + short_terms[2:-2]
    ))
    for term in ordered_short_terms:
        add(term)
        if len(variants) >= max_variants:
            break

    return variants[:max_variants]


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


def _canonical_memory_ref(mem_id: str, brain_config: Mapping[str, Any] | None,
                          memory_store=None) -> str:
    """Resolve a Markdown memory id to the product-facing ``path#id`` ref.

    ``tools_memory.memory_query`` intentionally keeps its small, backwards
    compatible ``id`` payload. RAG consumers, however, need a stable source
    anchor that can be checked against Vault scope and qrels. Resolution is
    read-only; when the Vault is unavailable or the entry is bucketed, retain
    the historical ``memory#id`` fallback instead of dropping the candidate.
    """
    mem_id = str(mem_id or "").strip()
    if not mem_id:
        return ""
    fallback = f"memory#{mem_id}"
    config = brain_config if isinstance(brain_config, Mapping) else {}
    vault = str(config.get("vault_path") or config.get("vault_root") or "").strip()
    if not vault:
        return fallback
    try:
        from pathlib import Path
        from agentlab.memory.markdown_store import MemoryMarkdownStore

        store = memory_store or MemoryMarkdownStore(vault, create_dirs=False)
        file_path = store._find_memory_file(mem_id)
        if file_path is None:
            # A bucket contains several entries and deliberately cannot be
            # resolved by _find_memory_file; inspect headers without mutating
            # access metadata so read-only retrieval remains side-effect free.
            root = Path(store.memory_root)
            for candidate in root.rglob("*.md"):
                try:
                    relative = candidate.relative_to(root)
                except ValueError:
                    continue
                if relative.parts and relative.parts[0].casefold() == "archive":
                    continue
                try:
                    text = candidate.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if re.search(rf"^## {re.escape(mem_id)}\s*$", text, flags=re.M):
                    file_path = candidate
                    break
        if file_path is None:
            return fallback
        root = Path(vault).resolve()
        path = Path(file_path).resolve()
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            # Never expose an absolute local path or treat an out-of-scope
            # file as a valid citation.  Preserve the historical opaque ref
            # until the caller can resolve the id against the right Vault.
            return fallback
        return f"{rel}#{mem_id}"
    except (OSError, RuntimeError, TypeError, ValueError):
        return fallback


@dataclass
class RecallItem:
    title: str
    content: str = ""
    ref: str = ""            # 来源引用：文件路径 / 网络 URL / 记忆 id
    source: str = "vault"    # 路径来源：vault / memory / graph / web
    score: float = 0.0
    # Optional governance metadata.  Empty/default values are omitted from the
    # legacy list shape, while envelope/answer-gate callers can retain scope.
    status: str = "active"
    project_id: str = ""
    session_id: str = ""
    context_of: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {
            "title": self.title, "content": self.content,
            "ref": self.ref, "source": self.source, "score": round(self.score, 3),
        }
        if self.status != "active":
            out["status"] = self.status
        if self.project_id:
            out["project_id"] = self.project_id
        if self.session_id:
            out["session_id"] = self.session_id
        if self.context_of:
            out["context_of"] = list(self.context_of)
        return out


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
                              source=it.source, score=it.score,
                              status=it.status, project_id=it.project_id,
                              session_id=it.session_id,
                              context_of=list(it.context_of)))
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
                 extra: list[Recaller] | None = None,
                 shadow_extra: list[Recaller] | None = None,
                 exclude_sources: set[str] | None = None,
                 strategy: RetrievalStrategy | str | None = None,
                 brain_config: dict | None = None,
                 rewrite_provider: Any | None = None,
                 query_rewrite_mode: str = "off",
                 query_rewrite_max_variants: int = 1,
                 rewrite_deadline_ms: int = 250,
                 query_expansion_mode: str = "off",
                 query_expansion_max_variants: int = 5,
                 query_expansion_min_candidates: int = 8):
        # An explicit empty list means "no default routes" (used by
        # vector-only mode); None means use the normal vault/memory/web set.
        base = self._default_recallers(brain_config) if recallers is None else list(recallers)
        excluded = {str(source) for source in (exclude_sources or set())}
        if recallers is None and excluded:
            # Default routes carry a private source marker.  This lets a
            # versioned index replace only the legacy Vault route while still
            # retaining memory/web routes in the same RAG request.
            base = [route for route in base
                    if getattr(route, "_rag_source", "") not in excluded]
        self._recallers = base + list(extra or [])  # extra：向量路等增强路（P0-1）
        self._shadow_recallers = list(shadow_extra or [])
        self._strategy = strategy
        self._last_warnings: list[str] = []
        self._last_successes = 0
        self._last_query_plan: QueryPlan | None = None
        if query_rewrite_mode not in {"off", "shadow", "on"}:
            query_rewrite_mode = "off"
        self._rewrite_provider = rewrite_provider
        self._query_rewrite_mode = query_rewrite_mode
        try:
            max_variants = int(query_rewrite_max_variants)
        except (TypeError, ValueError):
            max_variants = 1
        try:
            deadline_ms = int(rewrite_deadline_ms)
        except (TypeError, ValueError):
            deadline_ms = 250
        self._query_rewrite_max_variants = max(0, min(1, max_variants))
        self._rewrite_deadline_ms = max(1, min(2000, deadline_ms))
        if query_expansion_mode not in {"off", "shadow", "on"}:
            query_expansion_mode = "off"
        try:
            expansion_max = int(query_expansion_max_variants)
        except (TypeError, ValueError):
            expansion_max = 5
        try:
            expansion_min = int(query_expansion_min_candidates)
        except (TypeError, ValueError):
            expansion_min = 8
        self._query_expansion_mode = query_expansion_mode
        self._query_expansion_max_variants = max(0, min(5, expansion_max))
        self._query_expansion_min_candidates = max(1, min(40, expansion_min))
        self._last_query_rewrite: dict[str, Any] = {
            "mode": query_rewrite_mode,
            "eligible": False,
            "applied": False,
            "candidate_retrieved": False,
            "candidate_refs": [],
        }
        self._last_query_expansion: dict[str, Any] = {
            "mode": query_expansion_mode,
            "eligible": False,
            "applied": False,
            "variants": [],
            "candidate_retrieved": False,
        }
        self._last_timings: dict[str, float] = {}
        self._last_route_timings: dict[str, float] = {}
        self._last_route_metadata: dict[str, Any] = {}

    @property
    def last_query_plan(self) -> dict[str, object] | None:
        """Shadow-only classifier output for diagnostics; never changes recall."""
        return self._last_query_plan.to_dict() if self._last_query_plan else None

    @property
    def last_query_rewrite(self) -> dict[str, Any]:
        """Redacted rewrite telemetry; query text is never returned here."""
        return dict(self._last_query_rewrite)

    @property
    def last_query_expansion(self) -> dict[str, Any]:
        """Redacted local-expansion telemetry; raw query variants are omitted."""
        return dict(self._last_query_expansion)

    @property
    def last_route_metadata(self) -> dict[str, Any]:
        """Safe index/provenance metadata from the latest route collection."""
        return dict(self._last_route_metadata)

    def _collect_query_expansion(
        self, query: str, original: Sequence[RecallItem], limit: int,
        scope: RetrievalScope,
    ) -> list[RecallItem]:
        """Run bounded local variants when the first recall is weak.

        ``shadow`` records what the extra route would return but preserves the
        original candidates. ``on`` fuses variants as separate routes. This is
        intentionally separate from LLM rewrite: no provider call and no
        semantic broadening are needed for this safety-critical fallback.
        """
        self._last_query_expansion = {
            "mode": self._query_expansion_mode,
            "eligible": False,
            "applied": False,
            "variants": [],
            "candidate_retrieved": False,
        }
        if self._query_expansion_mode == "off" or self._query_expansion_max_variants < 1:
            return []
        query_type, _confidence, _reason = classify_query(query)
        if query_type in {"empty", "identifier", "negative"}:
            self._last_query_expansion["reason"] = "unsafe_query_type"
            return []
        # ``_collect`` deliberately returns a wide candidate window (P2 uses
        # 40).  Expansion must be judged against the displayed top-k, not that
        # raw window; otherwise a pile of unrelated candidates makes a genuine
        # low-recall query look healthy and prevents the fallback from firing.
        observed = fuse(query, list(original), k=limit)
        coverage = self._estimate_lexical_coverage(query, observed)
        if len(observed) >= min(limit, self._query_expansion_min_candidates) and coverage >= 0.60:
            self._last_query_expansion["reason"] = "initial_recall_sufficient"
            return []
        variants = expand_query_variants(
            query, max_variants=self._query_expansion_max_variants,
        )
        self._last_query_expansion.update({
            "eligible": bool(variants),
            "variant_count": len(variants),
            "variant_hashes": [hashlib.sha256(item.encode("utf-8")).hexdigest()
                               for item in variants],
            "initial_candidates": len(observed),
            "initial_coverage": round(coverage, 4),
        })
        if not variants:
            self._last_query_expansion["reason"] = "no_safe_variants"
            return []
        expanded: list[RecallItem] = []
        for variant in variants:
            try:
                expanded.extend(self._collect(variant, scope))
            except Exception:
                # Local expansion is an optional recall enhancement; the
                # original route remains authoritative on variant failure.
                continue
        self._last_query_expansion.update({
            "candidate_retrieved": bool(expanded),
            "candidate_count": len(expanded),
            "applied": self._query_expansion_mode == "on" and bool(expanded),
        })
        # Raw query text is intentionally omitted from telemetry.  The
        # variant hashes and counts are enough to compare shadow/on runs.
        return expanded

    @property
    def last_timings(self) -> dict[str, float]:
        """Per-stage wall-clock timings for opt-in diagnostics."""
        return dict(self._last_timings)

    @property
    def last_route_timings(self) -> dict[str, float]:
        """Underlying route timings reported by a versioned retriever."""
        return dict(self._last_route_timings)

    def _prepare_query_rewrite(
        self, query: str, *, recent_context: str = "",
        lexical_coverage: float | None = None,
    ) -> tuple[QueryPlan, RewriteResult | None]:
        classify_started = time.perf_counter()
        plan = build_query_plan(
            query, recent_context=recent_context,
            lexical_coverage=lexical_coverage,
            rewrite_mode=self._query_rewrite_mode,
        )
        self._last_timings["query_classify_ms"] = round(
            (time.perf_counter() - classify_started) * 1000, 1,
        )
        self._last_query_plan = plan
        self._last_query_rewrite = {
            "mode": self._query_rewrite_mode,
            "query_type": plan.query_type,
            "eligible": bool(plan.should_rewrite),
            "applied": False,
            "candidate_retrieved": False,
            "candidate_refs": [],
            "lexical_coverage": lexical_coverage,
        }
        if (self._query_rewrite_mode == "off"
                or not plan.should_rewrite
                or self._query_rewrite_max_variants < 1):
            self._last_query_rewrite["reason"] = plan.reason
            return plan, None
        if self._rewrite_provider is None:
            self._last_query_rewrite["reason"] = "rewrite_provider_missing"
            return plan, None
        try:
            rewrite_started = time.perf_counter()
            result = rewrite_query_sync(
                self._rewrite_provider,
                query,
                recent_context=recent_context,
                lexical_coverage=lexical_coverage,
                mode=self._query_rewrite_mode,
                deadline_ms=self._rewrite_deadline_ms,
            )
            self._last_timings["rewrite_ms"] = round(
                (time.perf_counter() - rewrite_started) * 1000, 1,
            )
        except Exception as exc:  # optional path must never block retrieval
            self._last_timings["rewrite_ms"] = round(
                (time.perf_counter() - rewrite_started) * 1000, 1,
            )
            self._last_query_rewrite.update({
                "reason": "rewrite_runtime_error",
                "error": type(exc).__name__,
            })
            return plan, None
        candidate = result.candidate_query or ""
        if candidate and candidate != query:
            plan = replace(plan, variants=(plan.original_query, candidate))
            self._last_query_plan = plan
        self._last_query_rewrite.update({
            "reason": result.reason,
            "error": result.error,
            "applied": bool(result.applied),
            "candidate_query_hash": hashlib.sha256(
                candidate.encode("utf-8")
            ).hexdigest() if candidate else "",
        })
        return plan, result

    @staticmethod
    def _estimate_lexical_coverage(
        query: str, candidates: Sequence[RecallItem],
    ) -> float:
        """Estimate term coverage without another index/provider call."""
        terms = retrieval_terms(query)
        if not terms:
            return 1.0
        haystack = "\n".join(
            f"{item.title}\n{item.ref}\n{item.content}"
            for item in candidates
        ).casefold()
        return sum(term.casefold() in haystack for term in terms) / len(terms)

    @staticmethod
    def _fuse_query_variants(
        original_query: str,
        original: Sequence[RecallItem],
        candidate_query: str,
        candidate: Sequence[RecallItem],
        limit: int,
    ) -> list[RecallItem]:
        """Fuse original and rewrite as separate internal routes.

        The variant marker is removed before returning so product consumers see
        the normal source names and the original query remains first on ties.
        """
        prefix = "__query_rewrite__"
        tagged = list(original)
        for item in candidate:
            tagged.append(RecallItem(
                title=item.title, content=item.content, ref=item.ref,
                source=f"{prefix}{item.source}", score=item.score,
                status=item.status, project_id=item.project_id,
                session_id=item.session_id, context_of=list(item.context_of),
            ))
        rows = fuse(original_query, tagged, k=limit)
        for item in rows:
            if item.source.startswith(prefix):
                item.source = item.source[len(prefix):] or "vault"
        return rows

    @staticmethod
    def _default_recallers(brain_config: dict | None = None) -> list[Recaller]:
        """默认三路：vault_search + memory_query（brain）+ web_search（框架）。

        brain 未就绪时 MemoryStore 自动降级为空；不抛错。
        """
        lst: list[Recaller] = []

        from agentlab.memory.store import MemoryStore
        store = MemoryStore(brain_config)
        if store.available:
            # Resolve memory ids once per route instance. The underlying brain
            # API remains unchanged, while RAG emits canonical Vault anchors.
            memory_ref_store = None
            memory_ref_cache: dict[str, str] = {}
            try:
                from agentlab.memory.markdown_store import MemoryMarkdownStore
                vault_path = str((brain_config or {}).get("vault_path") or "").strip()
                if vault_path:
                    memory_ref_store = MemoryMarkdownStore(vault_path, create_dirs=False)
            except Exception:
                memory_ref_store = None

            def memory_ref(mem_id: str) -> str:
                key = str(mem_id or "").strip()
                if key not in memory_ref_cache:
                    memory_ref_cache[key] = _canonical_memory_ref(
                        key, brain_config, memory_store=memory_ref_store,
                    )
                return memory_ref_cache[key]

            def vault(q: str, scope: RetrievalScope | None = None, store=store) -> list[dict]:
                if scope is not None and scope.project_id:
                    # Legacy vault_search has no project filter.  Do not risk
                    # cross-project leakage; P2 is the scoped Vault route.
                    return []
                return [
                    {"title": o.get("path", ""), "content": o.get("content", ""),
                     "ref": o.get("path", ""), "source": "vault"}
                    for o in _union_recall(store.search, q, limit=20)
                ]
            vault._rag_source = "vault"
            vault._rag_scope_unsupported = True

            def mem(q: str, scope: RetrievalScope | None = None, store=store) -> list[dict]:
                return [
                    {"title": o.get("content", "")[:40], "content": o.get("content", ""),
                     "ref": memory_ref(o.get("id", "")), "source": "memory"}
                    for o in _union_recall(
                        lambda term, limit: store.query(
                            term, limit=limit,
                            project_id=(scope.project_id if scope is not None else None),
                        ), q, limit=10
                    )
                ]
            mem._rag_source = "memory"

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

    @staticmethod
    def _call_recaller(rec: Recaller, query: str, scope: RetrievalScope):
        """Pass scope only to routes that explicitly support it.

        Existing third-party/test recallers remain query-only; inspecting the
        signature avoids masking a TypeError raised inside a route.
        """
        try:
            params = inspect.signature(rec).parameters.values()
            accepts_scope = any(p.name == "scope" for p in params)
            accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params)
        except (TypeError, ValueError):
            accepts_scope = accepts_kwargs = False
        return rec(query, scope=scope) if accepts_scope or accepts_kwargs else rec(query)

    def _collect(self, query: str, scope: RetrievalScope) -> list[RecallItem]:
        candidates: list[RecallItem] = []
        guarded_route_candidates: list[RecallItem] | None = None
        warnings: list[str] = []
        successes = 0
        for rec in self._recallers:
            try:
                if scope.project_id and getattr(rec, "_rag_scope_unsupported", False):
                    warnings.append("recall_route_scope_unsupported")
                    continue
                rows = self._call_recaller(rec, query, scope) or []
                successes += 1
            except Exception as exc:
                warnings.append(f"recall_route_error:{type(exc).__name__}")
                continue
            route_start = len(candidates)
            route_status = getattr(rec, "_last_route_status", {})
            if isinstance(route_status, Mapping):
                for name, info in route_status.items():
                    if isinstance(info, Mapping) and str(info.get("status", "available")) != "available":
                        reason = str(info.get("reason") or "unavailable")
                        warnings.append(f"{name}_route_{reason}")
            route_record = getattr(rec, "_last_record", {})
            if isinstance(route_record, Mapping):
                record_meta = route_record.get("meta")
                if isinstance(record_meta, Mapping):
                    self._last_route_metadata = {
                        str(key): value for key, value in record_meta.items()
                        if key in {
                            "index_version", "parser_version",
                            "chunk_strategy_version", "embedding_model",
                        }
                    }
                routes = route_record.get("routes", {})
                if isinstance(routes, Mapping):
                    self._last_route_timings = {
                        str(name): float(info.get("latency_ms", 0.0) or 0.0)
                        for name, info in routes.items()
                        if isinstance(info, Mapping)
                    }
                context_stats = route_record.get("small_to_big")
                if isinstance(context_stats, Mapping) and context_stats.get("latency_ms") is not None:
                    try:
                        self._last_route_timings["context_expand"] = float(
                            context_stats.get("latency_ms") or 0.0
                        )
                    except (TypeError, ValueError):
                        pass
            for row in rows:
                if not isinstance(row, Mapping):
                    warnings.append("recall_invalid_row")
                    continue
                row_status = str(row.get("status", "active") or "active").strip().lower()
                if row_status in {"candidate", "quarantine", "superseded", "archived", "revoked", "expired", "conflict"}:
                    warnings.append(f"recall_inactive:{row_status}")
                    continue
                row_project = str(row.get("project_id", "") or "").strip()
                if scope.project_id and row_project and row_project not in {scope.project_id, "default"}:
                    warnings.append("recall_scope_denied")
                    continue
                candidates.append(
                    RecallItem(
                        title=str(row.get("title", "")),
                        content=str(row.get("content", "")),
                        ref=str(row.get("ref", "")),
                        source=str(row.get("source", "vault")),
                        status=str(row.get("status", "active") or "active"),
                        project_id=str(row.get("project_id", "") or ""),
                        session_id=str(row.get("session_id", "") or ""),
                    )
                )
            if isinstance(route_record, Mapping) and \
                    route_record.get("display_mode") in {
                        "guarded_hybrid_fallback", "memory_focused",
                    }:
                guarded_route_candidates = candidates[route_start:]
        for rec in self._shadow_recallers:
            try:
                self._call_recaller(rec, query, scope)
            except Exception:
                warnings.append("shadow_route_error")
        self._last_warnings = list(dict.fromkeys(warnings))
        self._last_successes = successes
        if guarded_route_candidates:
            # A guarded semantic fallback has already passed the exact/negative
            # safety classifier.  Keep its p2 candidates together instead of
            # letting unrelated memory/web routes crowd the vector evidence
            # during the final multi-route fuse.
            candidates = guarded_route_candidates
        return candidates

    def retrieve(self, query: str, limit: int = 8,
                 *, scope: RetrievalScope | Mapping[str, Any] | None = None,
                 recent_context: str = "") -> list[RecallItem]:
        """多路并行召回 → 融合排序 → top-limit。任一路异常降级，不影响整体。"""
        total_started = time.perf_counter()
        self._last_timings = {
            "query_classify_ms": 0.0,
            "rewrite_ms": 0.0,
            "original_recall_ms": 0.0,
            "candidate_recall_ms": 0.0,
            "fuse_ms": 0.0,
            "total_ms": 0.0,
        }
        self._last_route_timings = {}
        self._last_route_metadata = {}
        request_scope = RetrievalScope.from_value(scope) if scope is not None else current_retrieval_scope()
        recall_started = time.perf_counter()
        candidates = self._collect(query, request_scope)
        self._last_timings["original_recall_ms"] = round(
            (time.perf_counter() - recall_started) * 1000, 1,
        )
        original_warnings = list(self._last_warnings)
        original_successes = self._last_successes
        expansion_started = time.perf_counter()
        expanded_items = self._collect_query_expansion(
            query, candidates, limit, request_scope,
        )
        self._last_timings["expansion_ms"] = round(
            (time.perf_counter() - expansion_started) * 1000, 1,
        )
        # Optional variants must not replace the primary route's health state.
        self._last_warnings = original_warnings
        self._last_successes = original_successes
        expanded_candidates = (
            list(candidates) + list(expanded_items)
            if self._query_expansion_mode == "on" and expanded_items
            else list(candidates)
        )
        coverage = self._estimate_lexical_coverage(query, candidates)
        _plan, rewrite = self._prepare_query_rewrite(
            query, recent_context=recent_context, lexical_coverage=coverage,
        )
        candidate_items: list[RecallItem] = []
        candidate_query = rewrite.candidate_query if rewrite is not None else ""
        if candidate_query and candidate_query != query:
            started = time.perf_counter()
            candidate_items = self._collect(candidate_query, request_scope)
            candidate_elapsed = (time.perf_counter() - started) * 1000
            candidate_warnings = list(self._last_warnings)
            self._last_warnings = list(dict.fromkeys(original_warnings + candidate_warnings))
            self._last_successes = original_successes
            candidate_rows = fuse(candidate_query, candidate_items, k=limit)
            self._last_query_rewrite.update({
                "candidate_retrieved": True,
                "candidate_refs": [item.ref for item in candidate_rows if item.ref],
                "candidate_latency_ms": round(candidate_elapsed, 1),
            })
            self._last_timings["candidate_recall_ms"] = round(candidate_elapsed, 1)
        fuse_started = time.perf_counter()
        if self._query_rewrite_mode == "on" and candidate_items:
            rows = self._fuse_query_variants(
                query, expanded_candidates, candidate_query, candidate_items, limit,
            )
        else:
            rows = fuse(query, expanded_candidates, k=limit)
        self._last_timings["fuse_ms"] = round(
            (time.perf_counter() - fuse_started) * 1000, 1,
        )
        self._last_timings["total_ms"] = round(
            (time.perf_counter() - total_started) * 1000, 1,
        )
        return rows

    def retrieve_governed(self, query: str, limit: int = 8,
                          per_item_chars: int = _PER_ITEM_CHARS,
                          *, scope: RetrievalScope | Mapping[str, Any] | None = None,
                          recent_context: str = "") -> list[RecallItem]:
        """召回 + 注入治理：Tier-1 单条截断、Tier-2 身份锚保留（OPT-089）。

        供注入入口（rag_retrieve）使用：在源头控住每条正文与总项数，
        不把治理责任全甩给下游 `max_tool_result_chars`。
        """
        return govern_recall(self.retrieve(query, limit=limit, scope=scope,
                                           recent_context=recent_context),
                             per_item_chars=per_item_chars, max_items=limit)

    def retrieve_result(
        self,
        query: str,
        limit: int = 8,
        *,
        scope: RetrievalScope | Mapping[str, Any] | None = None,
        strategy: RetrievalStrategy | str | None = None,
        status: RetrievalStatus | str | None = None,
        per_item_chars: int = _PER_ITEM_CHARS,
        recent_context: str = "",
    ) -> RetrievalResult:
        """Return the S0 envelope while keeping ``retrieve`` list-compatible."""
        request_scope = RetrievalScope.from_value(scope) if scope is not None else current_retrieval_scope()
        items = self.retrieve_governed(
            query, limit=limit, per_item_chars=per_item_chars, scope=request_scope,
            recent_context=recent_context,
        )
        selected = strategy or self._strategy or RetrievalStrategy.HYBRID
        warnings = list(self._last_warnings)
        if status is None:
            if self._last_successes == 0 and self._recallers:
                selected_status = RetrievalStatus.UNAVAILABLE
            elif warnings:
                selected_status = RetrievalStatus.DEGRADED
            else:
                selected_status = RetrievalStatus.AVAILABLE
        else:
            selected_status = status
        payload = []
        for item in items:
            if not item.ref:
                warnings.append("item_missing_ref")
                continue
            row = item.to_dict()
            row["project_id"] = request_scope.project_id
            row["session_id"] = request_scope.session_id
            payload.append(row)
        return scoped_retrieval_result(
            payload, strategy=selected, status=selected_status,
            warnings=warnings, scope=request_scope,
        )
