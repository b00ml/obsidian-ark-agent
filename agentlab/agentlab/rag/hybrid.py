"""P4 hybrid retrieval orchestration for the P2 shadow index.

The existing ``RAGRecall`` remains the production-compatible multi-source
orchestrator.  This module is deliberately independent: it combines the P2
lexical and vector candidates, aggregates chunks to entry keys, and exposes a
shadow record without persisting sensitive document content.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable

from agentlab.rag.recall import RecallItem, govern_recall


_RRF_K = 60
_CHUNK_SUFFIX = re.compile(r":ch[0-9a-f]{8}(?:-\d+)?$", re.IGNORECASE)
_LEGACY_CHUNK_SUFFIX = re.compile(r":c\d+$", re.IGNORECASE)
_IDENTIFIER_QUERY = re.compile(r"(?i)(?:bv[a-z0-9]*\d[a-z0-9]*|[a-z]{2,}[a-z0-9]*\d[a-z0-9]*)")
_NEGATIVE_QUERY = re.compile(r"(?:没有|无|不存在|找不到|是否有|有没有|未找到|不包含|without|not found)", re.IGNORECASE)
_SEMANTIC_QUERY = re.compile(
    r"(?:什么|哪些|哪里|怎么|如何|为何|为什么|哪种|分别|适用|覆盖|存放|执行后|前要|规模|主题|经验|路径|区别|规定|算哪|解决)"
)
_PARENT_ENTRY_QUERY = re.compile(
    r"(?:主题|覆盖|知识库|记忆|条目|记录|规定|目录|存放|路径|局限|经验|"
    r"状态|检查|核对|先|之前|以前|什么|哪些|哪里|怎么|如何|为何|为什么|"
    r"哪种|分别|适用|算哪|解决|处理|失败|不完整|孤岛|补出链)",
    re.IGNORECASE,
)
_MEMORY_FOCUS_QUERY = re.compile(
    r"(?:长期记忆|历史记忆|用户偏好|用户喜欢|用户习惯|用户要求|"
    r"项目决定|之前决定|过去决定|历史经验|过往经历|"
    r"(?:用户|我|你).{0,12}(?:偏好|喜欢|习惯|要求|倾向|选择|决定|"
    r"记得|之前|上次|经验|经历)|处理.+前|任务前|开始.+前|执行.+前|"
    r"memory|preference|decision|experience)",
    re.IGNORECASE,
)
_CROSS_DOCUMENT_QUERY = re.compile(
    r"(?:跨文档|跨文件|多文档|哪些文档|分别|各自|多个来源|不同来源|"
    r"有哪些观点|有哪些判据|哪些结论|对比|比较|并列|"
    r"cross[-_ ]?doc|multi[-_ ]?document)",
    re.IGNORECASE,
)
_MEMORY_PREFERENCE_QUERY = re.compile(
    r"(?:偏好|喜欢|习惯|风格|倾向|profile|preference)", re.IGNORECASE,
)
_MEMORY_DECISION_QUERY = re.compile(
    r"(?:决定|决策|选型|判据|架构|为什么选择|取舍|decision)", re.IGNORECASE,
)
_MEMORY_PROCEDURE_QUERY = re.compile(
    r"(?:如何|怎么|怎样|流程|步骤|规范|应该|先|处理|执行|procedure|workflow|sop)",
    re.IGNORECASE,
)
_MEMORY_EPISODIC_QUERY = re.compile(
    r"(?:之前|以前|上次|历史|过去|经历|经验|任务前|处理.+前|session|episodic)",
    re.IGNORECASE,
)


def _identifier_like_query(query: str) -> bool:
    """Recognize opaque identifier queries where semantic fallback is risky.

    This deliberately requires a single compact token containing letters and
    a digit.  Natural-language queries and short symbols remain eligible for
    vector recall; an identifier with no lexical hit is treated as an unknown
    lookup and abstains from semantic candidates.
    """
    text = str(query or "").strip()
    return bool(text and not re.search(r"\s", text) and len(text) >= 8
                and _IDENTIFIER_QUERY.fullmatch(text))


def _semantic_fallback_allowed(query: str) -> bool:
    """Allow guarded vector display only for semantic natural-language asks."""
    text = " ".join(str(query or "").split())
    if not text or _identifier_like_query(text) or _NEGATIVE_QUERY.search(text):
        return False
    return bool(_SEMANTIC_QUERY.search(text))


def _parent_entry_query_allowed(query: str) -> bool:
    """Keep entry aggregation on topic/bucket-style natural-language asks.

    Exact identifiers, absence questions, and one-token lookups retain the
    established chunk lexical path.  This narrow classifier is deliberately
    local to the P2 lexical route; it is not a query rewrite or answer gate.
    """
    text = " ".join(str(query or "").split())
    if not text or _identifier_like_query(text) or _NEGATIVE_QUERY.search(text):
        return False
    return bool(_PARENT_ENTRY_QUERY.search(text))


def _memory_focus_query(query: str) -> bool:
    """Recognize atomic-memory asks that need entry-level isolation.

    Long-term memory entries are deliberately short facts, preferences and
    decisions.  Letting generic Vault chunks participate in the same small
    display window makes these entries look absent even when the memory route
    found them.  This classifier only narrows the derived P2 display set; it
    never changes the source Markdown or global production switch.
    """
    text = " ".join(str(query or "").split())
    return bool(text and not _NEGATIVE_QUERY.search(text)
                and _MEMORY_FOCUS_QUERY.search(text))


def _cross_document_query(query: str) -> bool:
    """Recognize bounded multi-source asks without requiring gold refs."""
    text = " ".join(str(query or "").split())
    return bool(text and not _NEGATIVE_QUERY.search(text)
                and _CROSS_DOCUMENT_QUERY.search(text))


def _memory_path_prefixes(query: str) -> tuple[str, ...]:
    """Return a typed memory corpus scope for an atomic-memory query.

    The prefixes are a ranking hint, not an authorization boundary.  Scope,
    status and archive filters remain enforced by the index independently.
    Broad fallbacks preserve recall when a query does not expose its memory
    type clearly.
    """
    text = " ".join(str(query or "").split())
    if _MEMORY_PREFERENCE_QUERY.search(text):
        return ("ark/memory/core/", "ark/memory/context/")
    if _MEMORY_DECISION_QUERY.search(text):
        return ("ark/memory/decisions/", "ark/memory/context/")
    if _MEMORY_PROCEDURE_QUERY.search(text):
        return ("ark/memory/procedures/", "ark/memory/decisions/",
                "ark/memory/sessions/", "ark/memory/context/")
    if _MEMORY_EPISODIC_QUERY.search(text):
        return ("ark/memory/sessions/", "ark/memory/context/")
    return ("ark/memory/",)


def _has_substantive_content(row: Mapping) -> bool:
    content = str(row.get("content") or "")
    return any(line.strip() and not line.lstrip().startswith("#")
               for line in content.splitlines())


def _prefer_richer(existing: Mapping, candidate: Mapping) -> dict:
    """Prefer body-bearing chunks over heading-only representatives."""
    if not _has_substantive_content(existing) and _has_substantive_content(candidate):
        return dict(candidate)
    return dict(existing)


def entry_key(ref: str, dedupe_by: str = "entry") -> str:
    """Return the stable aggregation key for a chunk reference."""
    text = str(ref or "").replace("\\", "/").strip()
    if dedupe_by == "ref":
        return text
    path, marker, anchor = text.partition("#")
    if dedupe_by == "file" or not marker:
        return path
    if re.fullmatch(r"c\d+", anchor, flags=re.IGNORECASE):
        return path
    # Non-bucket Markdown files use a synthetic ``#doc:ch<hash>`` anchor.
    # Their lexical representative is file-level, so both identities must
    # merge or a short memory entry receives two half-strength RRF scores.
    if re.fullmatch(r"doc(?::ch[0-9a-f]{8}(?:-\d+)?)?", anchor, flags=re.IGNORECASE):
        return path
    anchor = _CHUNK_SUFFIX.sub("", anchor)
    anchor = _LEGACY_CHUNK_SUFFIX.sub("", anchor)
    return f"{path}#{anchor}" if anchor else path


def _candidate_key(row: Mapping, dedupe_by: str) -> str:
    ref = str(row.get("ref") or "")
    if ref:
        return entry_key(ref, dedupe_by)
    return f"{row.get('source', '')}:{row.get('title', '')}"


def _route_unique(rows: Sequence[Mapping], dedupe_by: str) -> list[dict]:
    """Collapse chunks while retaining substantive evidence representatives."""
    positions: dict[str, int] = {}
    out: list[dict] = []
    for row in rows:
        item = dict(row)
        key = _candidate_key(item, dedupe_by)
        position = positions.get(key)
        if position is None:
            positions[key] = len(out)
            out.append(item)
        else:
            out[position] = _prefer_richer(out[position], item)
    return out


class ShadowLogWriter:
    """Bounded JSONL writer for optional hybrid shadow observations."""

    def __init__(self, path: str | Path, *, max_bytes: int = 5_000_000):
        self.path = Path(path)
        self.max_bytes = max(1024, int(max_bytes))

    def __call__(self, record: Mapping) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"logged_at": time.time(), **dict(record)}
            line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            if len(line.encode("utf-8")) > self.max_bytes:
                payload = {
                    key: payload.get(key)
                    for key in ("logged_at", "schema", "query_hash", "hybrid_status", "latency_ms")
                    if key in payload
                }
                line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            if len(line.encode("utf-8")) > self.max_bytes:
                return False
            if self.path.exists() and self.path.stat().st_size + len(line.encode("utf-8")) > self.max_bytes:
                rotated = self.path.with_name(self.path.name + ".1")
                try:
                    rotated.unlink(missing_ok=True)
                except OSError:
                    pass
                self.path.replace(rotated)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
            return True
        except OSError:
            # Shadow telemetry must never make retrieval fail.
            return False


def hybrid_fuse(
    routes: Mapping[str, Sequence[Mapping]],
    *,
    k: int = 8,
    dedupe_by: str = "entry",
    rrf_k: int = _RRF_K,
) -> list[dict]:
    """RRF-fuse lexical/vector routes with entry-level deduplication.

    Route-local deduplication happens before rank assignment.  Without that
    step, a monthly bucket with several chunks would gain an artificial score
    simply by occupying multiple ranks in one route.
    """
    if k <= 0:
        return []
    if dedupe_by not in {"entry", "file", "ref"}:
        raise ValueError("dedupe_by must be entry, file, or ref")
    scores: dict[str, float] = {}
    representatives: dict[str, dict] = {}
    for source, rows in routes.items():
        for rank, item in enumerate(_route_unique(rows, dedupe_by), start=1):
            key = _candidate_key(item, dedupe_by)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            representative = {**item, "source": str(item.get("source") or source)}
            if key in representatives:
                representatives[key] = _prefer_richer(representatives[key], representative)
            else:
                representatives[key] = representative
    ranked = sorted(representatives.items(), key=lambda pair: (-scores[pair[0]], pair[0]))
    result: list[dict] = []
    for key, item in ranked[:k]:
        item = dict(item)
        item["score"] = round(scores[key], 6)
        item["entry_key"] = key
        result.append(item)
    return result


def _coverage_matches(expected: str, candidate: str) -> bool:
    """Match an expected file/entry ref against a concrete chunk ref."""
    expected = entry_key(str(expected or ""), "entry")
    candidate = entry_key(str(candidate or ""), "entry")
    if not expected or not candidate:
        return False
    if "#" in expected:
        return candidate == expected
    return candidate.split("#", 1)[0] == expected


def _coverage_select(
    rows: Sequence[Mapping],
    limit: int,
    required_groups: Sequence[Sequence[str]] | None,
    *,
    diversify_documents: bool = False,
) -> list[dict]:
    """Reserve one distinct candidate for each required coverage group.

    ``hybrid_fuse`` ranks candidates globally, which is the right default for
    interactive retrieval but can let several high-scoring entries from one
    document fill a small evaluation budget.  This bounded selector keeps the
    global order while ensuring that available all/required groups get a
    representative before the remaining slots are filled.
    """
    if limit <= 0:
        return []
    ranked = [dict(row) for row in rows]
    groups = [
        [str(ref).strip() for ref in group if str(ref).strip()]
        for group in (required_groups or ())
        if group
    ]
    if not groups:
        if not diversify_documents:
            return ranked[:limit]
        selected: list[dict] = []
        selected_positions: set[int] = set()
        selected_documents: set[str] = set()
        for position, row in enumerate(ranked):
            if len(selected) >= limit:
                break
            document = _document_key(str(row.get("ref") or ""))
            if document in selected_documents:
                continue
            selected.append(row)
            selected_positions.add(position)
            selected_documents.add(document)
        if len(selected) < limit:
            for position, row in enumerate(ranked):
                if len(selected) >= limit or position in selected_positions:
                    continue
                selected.append(row)
        return selected[:limit]

    selected: list[dict] = []
    selected_keys: set[str] = set()
    selected_positions: set[int] = set()
    selected_documents: set[str] = set()
    for group in groups:
        for position, row in enumerate(ranked):
            if position in selected_positions:
                continue
            ref = str(row.get("ref") or "")
            if not any(_coverage_matches(expected, ref) for expected in group):
                continue
            # Coverage identity is always an entry, even when the configured
            # display dedupe is broader (for example ``dedupe_by=file``).
            key = entry_key(ref, "entry")
            # A single entry must not satisfy two required slots when another
            # entry is available; this is the cross-document contract.
            if key in selected_keys:
                continue
            selected.append(row)
            selected_positions.add(position)
            selected_keys.add(key)
            selected_documents.add(_document_key(ref))
            break

    for position, row in enumerate(ranked):
        if len(selected) >= limit:
            break
        if position in selected_positions:
            continue
        if diversify_documents:
            document = _document_key(str(row.get("ref") or ""))
            if document in selected_documents:
                continue
        selected.append(row)
        selected_positions.add(position)
        selected_documents.add(_document_key(str(row.get("ref") or "")))
    if diversify_documents and len(selected) < limit:
        # If fewer than ``limit`` documents exist, fill remaining slots while
        # preserving the global fused order.
        for position, row in enumerate(ranked):
            if len(selected) >= limit or position in selected_positions:
                continue
            selected.append(row)
            selected_positions.add(position)
    return selected[:limit]


def _document_key(ref: str) -> str:
    return str(ref or "").replace("\\", "/").split("#", 1)[0]


class HybridRetriever:
    """P2 lexical/vector retriever with explicit ``off|shadow|on`` modes."""

    def __init__(
        self,
        store,
        *,
        vector_mode: str = "shadow",
        lexical_mode: str = "on",
        candidate_k: int = 40,
        dedupe_by: str = "entry",
        vector_min_score: float = 0.0,
        lexical_min_coverage: float = 0.0,
        small_to_big_mode: str = "shadow",
        small_to_big_neighbors: int = 1,
        small_to_big_max_chars: int = 2400,
        vector_fallback_mode: str = "off",
        shadow_logger: Callable[[Mapping], object] | None = None,
    ):
        if vector_mode not in {"off", "shadow", "on"}:
            raise ValueError("vector_mode must be off, shadow, or on")
        if lexical_mode not in {"off", "on"}:
            raise ValueError("lexical_mode must be off or on")
        if dedupe_by not in {"entry", "file", "ref"}:
            raise ValueError("dedupe_by must be entry, file, or ref")
        if not 0.0 <= float(vector_min_score) <= 1.0:
            raise ValueError("vector_min_score must be between 0 and 1")
        if not 0.0 <= float(lexical_min_coverage) <= 1.0:
            raise ValueError("lexical_min_coverage must be between 0 and 1")
        if small_to_big_mode not in {"off", "shadow", "on"}:
            raise ValueError("small_to_big_mode must be off, shadow, or on")
        if vector_fallback_mode not in {"off", "on"}:
            raise ValueError("vector_fallback_mode must be off or on")
        self.store = store
        self.vector_mode = vector_mode
        self.lexical_mode = lexical_mode
        self.candidate_k = max(1, int(candidate_k))
        self.dedupe_by = dedupe_by
        self.vector_min_score = float(vector_min_score)
        self.lexical_min_coverage = float(lexical_min_coverage)
        self.small_to_big_mode = small_to_big_mode
        self.small_to_big_neighbors = max(0, int(small_to_big_neighbors))
        self.small_to_big_max_chars = max(0, int(small_to_big_max_chars))
        self.vector_fallback_mode = vector_fallback_mode
        self.shadow_logger = shadow_logger

    def _fused_rows(
        self,
        routes: Mapping[str, Sequence[Mapping]],
        limit: int,
        required_groups: Sequence[Sequence[str]] | None = None,
        *,
        diversify_documents: bool = False,
    ) -> list[dict]:
        """Fuse enough candidates for coverage reservation, then trim."""
        pool_size = max(
            max(0, int(limit)),
            sum(len(rows) for rows in routes.values()),
        )
        fused = hybrid_fuse(routes, k=pool_size, dedupe_by=self.dedupe_by)
        return _coverage_select(
            fused, limit, required_groups,
            diversify_documents=diversify_documents,
        )

    def _display_rows(
        self,
        query: str,
        routes: Mapping[str, Sequence[Mapping]],
        limit: int,
        required_groups: Sequence[Sequence[str]] | None = None,
        *,
        diversify_documents: bool = False,
    ) -> tuple[list[dict], str]:
        """Choose displayed candidates with a guarded semantic fallback."""
        memory_focus = _memory_focus_query(query)
        fallback = (
            self.vector_mode == "shadow"
            and self.vector_fallback_mode == "on"
            and bool(routes.get("vector"))
            and _semantic_fallback_allowed(query)
        )
        if self.vector_mode == "on" or fallback:
            display_routes = (
                {"lexical": routes.get("lexical", []), "vector": routes.get("vector", [])}
                if fallback else routes
            )
            if memory_focus:
                display_routes = {
                    name: [row for row in values
                           if str(row.get("ref") or "").replace("\\", "/").startswith("ark/memory/")]
                    for name, values in display_routes.items()
                }
            return self._fused_rows(
                display_routes, limit, required_groups,
                diversify_documents=diversify_documents,
            ), (
                "memory_focused" if memory_focus else
                ("hybrid" if self.vector_mode == "on" else "guarded_hybrid_fallback")
            )
        if memory_focus:
            memory_rows = [row for row in routes.get("lexical", [])
                           if str(row.get("ref") or "").replace("\\", "/").startswith("ark/memory/")]
            return self._fused_rows(
                {"lexical": memory_rows}, limit, required_groups,
                diversify_documents=diversify_documents,
            ), "memory_focused"
        return self._fused_rows(
            {"lexical": routes.get("lexical", [])}, limit, required_groups,
            diversify_documents=diversify_documents,
        ), "lexical"

    def _apply_context(
        self,
        rows: Sequence[Mapping],
        *,
        project_id: str | None = None,
        statuses: Sequence[str] | None = None,
        include_archive: bool = False,
    ) -> tuple[list[dict], dict]:
        original = [dict(row) for row in rows]
        if self.small_to_big_mode == "off":
            return original, {
                "mode": "off", "status": "disabled", "requested": len(original),
                "expanded": 0, "neighbors": 0, "truncated": 0,
            }
        expander = getattr(self.store, "expand_context", None)
        if not callable(expander):
            return original, {
                "mode": self.small_to_big_mode, "status": "unsupported",
                "requested": len(original), "expanded": 0, "neighbors": 0, "truncated": 0,
            }
        try:
            expanded, stats = expander(
                original,
                neighbor_chunks=self.small_to_big_neighbors,
                max_chars=self.small_to_big_max_chars,
                project_id=project_id,
                statuses=statuses,
                include_archive=include_archive,
            )
            stats = {"mode": self.small_to_big_mode, **dict(stats)}
            # Shadow computes context only for telemetry; on explicitly
            # changes injected content while preserving refs and RRF ranks.
            return (expanded if self.small_to_big_mode == "on" else original), stats
        except Exception as exc:
            return original, {
                "mode": self.small_to_big_mode, "status": "unavailable",
                "reason": type(exc).__name__, "requested": len(original),
                "expanded": 0, "neighbors": 0, "truncated": 0,
            }

    def _collect(
        self,
        query: str,
        *,
        project_id: str | None = None,
        statuses: Sequence[str] | None = None,
        include_archive: bool = False,
    ) -> tuple[
        dict[str, list[dict]],
        dict[str, float],
        dict[str, str],
        dict[str, dict[str, str]],
    ]:
        routes: dict[str, list[dict]] = {}
        timings: dict[str, float] = {}
        errors: dict[str, str] = {}
        route_statuses: dict[str, dict[str, str]] = {}
        lexical_coverage_reason = ""
        scope_kwargs = {"project_id": project_id, "statuses": statuses}
        if include_archive:
            scope_kwargs["include_archive"] = True
        memory_focus = _memory_focus_query(query)
        # The memory corpus contains intentionally short atomic entries.  Ask
        # for that corpus before ranking so generic Vault chunks cannot fill
        # the candidate window and hide a relevant preference/decision.
        lexical_scope_kwargs = dict(scope_kwargs)
        if memory_focus:
            lexical_scope_kwargs["path_prefixes"] = _memory_path_prefixes(query)
        vector_scope_kwargs = dict(scope_kwargs)
        if memory_focus:
            vector_scope_kwargs["path_prefixes"] = _memory_path_prefixes(query)

        def record_status(name: str, error: str | None = None) -> None:
            if error:
                route_statuses[name] = {"status": "unavailable", "reason": error}
                return
            stored = getattr(self.store, "last_search_status", {})
            value = stored.get(name) if isinstance(stored, Mapping) else None
            if isinstance(value, Mapping):
                status = str(value.get("status") or "available")
                reason = str(value.get("reason") or "")
                route_statuses[name] = {"status": status, "reason": reason}
                return
            if name == "vector" and not getattr(self.store, "embedding_model", ""):
                route_statuses[name] = {"status": "unavailable", "reason": "provider_missing"}
                return
            route_statuses[name] = {"status": "available", "reason": ""}

        def run_lexical() -> tuple[list[dict], float, str | None, str]:
            started = time.perf_counter()
            rows: list[dict] = []
            error: str | None = None
            reason = ""
            try:
                rows = self.store.search_lexical(
                    query, k=self.candidate_k, **lexical_scope_kwargs,
                ) or []
                # Topic/bucket queries benefit from one representative per
                # parent entry.  Keep this opt-in and prepend the bounded
                # parent list so route-local dedupe chooses it over a child
                # chunk without changing vector or answer-gate behavior.
                parent_search = getattr(self.store, "search_parent_entries", None)
                if _parent_entry_query_allowed(query) and callable(parent_search):
                    try:
                        parent_limit = max(1, min(120, self.candidate_k * 3))
                        parents = parent_search(
                            query, k=parent_limit, **lexical_scope_kwargs,
                        ) or []
                        if parents:
                            rows = list(parents) + list(rows)
                    except Exception:
                        pass
                if self.lexical_min_coverage > 0.0 and rows:
                    coverage = float(self.store.lexical_confidence(query, rows))
                    if coverage < self.lexical_min_coverage:
                        rows = []
                        reason = f"lexical_coverage_below_{self.lexical_min_coverage:g}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            return rows, time.perf_counter() - started, error, reason

        def run_vector() -> tuple[list[dict], float, str | None]:
            started = time.perf_counter()
            rows: list[dict] = []
            error: str | None = None
            try:
                rows = self.store.search_vector(
                    query, k=self.candidate_k, **vector_scope_kwargs,
                ) or []
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            return rows, time.perf_counter() - started, error

        if self.lexical_mode != "off" and self.vector_mode != "off":
            if _identifier_like_query(query):
                lexical_result = run_lexical()
                vector_result = ([], 0.0, None) if not lexical_result[0] else run_vector()
                if not lexical_result[0]:
                    route_statuses["vector"] = {
                        "status": "available", "reason": "identifier_lexical_miss_skip",
                    }
            else:
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-route") as pool:
                    lexical_future = pool.submit(run_lexical)
                    vector_future = pool.submit(run_vector)
                    lexical_result = lexical_future.result()
                    vector_result = vector_future.result()
            lexical_rows, lexical_elapsed, lexical_error, lexical_coverage_reason = lexical_result
            vector_rows, vector_elapsed, vector_error = vector_result
            routes["lexical"] = lexical_rows
            timings["lexical"] = lexical_elapsed
            errors["lexical"] = lexical_error if lexical_error else errors.get("lexical", "")
            if not errors["lexical"]:
                errors.pop("lexical", None)
            record_status("lexical", errors.get("lexical"))
            if lexical_coverage_reason:
                route_statuses["lexical"] = {"status": "available", "reason": lexical_coverage_reason}
            routes["vector"] = vector_rows
            timings["vector"] = vector_elapsed
            if vector_error:
                errors["vector"] = vector_error
            record_status("vector", errors.get("vector"))
            if (_identifier_like_query(query) and not lexical_rows
                    and not vector_error):
                route_statuses["vector"] = {
                    "status": "available", "reason": "identifier_lexical_miss_skip",
                }
            if not vector_error and self.vector_min_score > 0.0:
                routes["vector"] = [
                    row for row in routes["vector"]
                    if float(row.get("score", 0.0) or 0.0) >= self.vector_min_score
                ]
            return routes, timings, errors, route_statuses

        if self.lexical_mode != "off":
            routes["lexical"] = []
            started = time.perf_counter()
            try:
                routes["lexical"] = self.store.search_lexical(
                    query, k=self.candidate_k, **lexical_scope_kwargs,
                ) or []
                parent_search = getattr(self.store, "search_parent_entries", None)
                if _parent_entry_query_allowed(query) and callable(parent_search):
                    try:
                        parent_limit = max(1, min(120, self.candidate_k * 3))
                        parents = parent_search(
                            query, k=parent_limit, **lexical_scope_kwargs,
                        ) or []
                        if parents:
                            routes["lexical"] = list(parents) + list(routes["lexical"])
                    except Exception:
                        pass
                if self.lexical_min_coverage > 0.0 and routes["lexical"]:
                    coverage = float(self.store.lexical_confidence(query, routes["lexical"]))
                    if coverage < self.lexical_min_coverage:
                        routes["lexical"] = []
                        lexical_coverage_reason = (
                            f"lexical_coverage_below_{self.lexical_min_coverage:g}"
                        )
            except Exception as exc:
                errors["lexical"] = f"{type(exc).__name__}: {exc}"
            timings["lexical"] = time.perf_counter() - started
            record_status("lexical", errors.get("lexical"))
            if lexical_coverage_reason:
                route_statuses["lexical"] = {
                    "status": "available", "reason": lexical_coverage_reason,
                }
        if self.vector_mode != "off":
            if (self.lexical_mode != "off" and not routes.get("lexical")
                    and _identifier_like_query(query)):
                routes["vector"] = []
                timings["vector"] = 0.0
                route_statuses["vector"] = {
                    "status": "available",
                    "reason": "identifier_lexical_miss_skip",
                }
            else:
                started = time.perf_counter()
                try:
                    routes["vector"] = self.store.search_vector(
                        query, k=self.candidate_k, **vector_scope_kwargs,
                    ) or []
                except Exception as exc:
                    routes["vector"] = []
                    errors["vector"] = f"{type(exc).__name__}: {exc}"
                timings["vector"] = time.perf_counter() - started
                record_status("vector", errors.get("vector"))
                if not errors.get("vector") and self.vector_min_score > 0.0:
                    routes["vector"] = [
                        row for row in routes.get("vector", [])
                        if float(row.get("score", 0.0) or 0.0) >= self.vector_min_score
                    ]
        return routes, timings, errors, route_statuses

    @staticmethod
    def _items(rows: Sequence[Mapping], limit: int, per_item_chars: int) -> list[RecallItem]:
        items = [RecallItem(
            title=str(row.get("title", "")), content=str(row.get("content", "")),
            ref=str(row.get("ref", "")), source=str(row.get("source", "hybrid")),
            score=float(row.get("score", 0.0) or 0.0),
            context_of=[str(item) for item in row.get("context_of", []) if str(item)],
        ) for row in rows]
        return govern_recall(items, per_item_chars=per_item_chars, max_items=max(0, limit))

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 8,
        per_item_chars: int = 1200,
        project_id: str | None = None,
        statuses: Sequence[str] | None = None,
        include_archive: bool = False,
        required_groups: Sequence[Sequence[str]] | None = None,
    ) -> list[RecallItem]:
        """Return display candidates according to the configured mode.

        ``shadow`` computes the vector route but displays lexical candidates;
        callers can use :meth:`shadow_record` to inspect the comparison.
        """
        routes, _timings, _errors, _statuses = self._collect(
            query, project_id=project_id, statuses=statuses,
            include_archive=include_archive,
        )
        rows, _display_mode = self._display_rows(
            query, routes, limit, required_groups,
            diversify_documents=_cross_document_query(query),
        )
        if self.small_to_big_mode == "on":
            rows, _context_stats = self._apply_context(
                rows, project_id=project_id, statuses=statuses,
                include_archive=include_archive,
            )
        return self._items(rows, limit, per_item_chars)

    def retrieve_with_shadow_record(
        self,
        query: str,
        *,
        limit: int = 8,
        per_item_chars: int = 1200,
        project_id: str | None = None,
        session_id: str | None = None,
        statuses: Sequence[str] | None = None,
        include_archive: bool = False,
        required_groups: Sequence[Sequence[str]] | None = None,
    ) -> tuple[list[RecallItem], dict]:
        """Retrieve and build shadow telemetry from one route collection.

        Production shadow mode needs the displayed lexical rows and the
        vector comparison, but both used to trigger ``_collect`` separately.
        Keeping collection, fusion, and telemetry in one call avoids a second
        provider query and SQLite matrix scan per user request.
        """
        started = time.perf_counter()
        routes, timings, errors, route_statuses = self._collect(
            query, project_id=project_id, statuses=statuses,
            include_archive=include_archive,
        )
        document_diversity = _cross_document_query(query)
        rows, display_mode = self._display_rows(
            query, routes, limit, required_groups,
            diversify_documents=document_diversity,
        )
        display_rows, context_stats = self._apply_context(
            rows, project_id=project_id, statuses=statuses,
            include_archive=include_archive,
        )
        items = self._items(display_rows, limit, per_item_chars)
        record = self._shadow_record_from_collection(
            query, limit, project_id, session_id, routes, timings, errors, route_statuses,
            started=started, context_stats=context_stats, required_groups=required_groups,
            diversify_documents=document_diversity,
        )
        if self.shadow_logger is not None and self.vector_mode == "shadow":
            try:
                self.shadow_logger(record)
            except Exception:
                pass
        return items, record

    def _shadow_record_from_collection(
        self,
        query: str,
        limit: int,
        project_id: str | None,
        session_id: str | None,
        routes: Mapping[str, Sequence[Mapping]],
        timings: Mapping[str, float],
        errors: Mapping[str, str],
        route_statuses: Mapping[str, Mapping[str, str]],
        *,
        started: float,
        context_stats: Mapping[str, object] | None = None,
        required_groups: Sequence[Sequence[str]] | None = None,
        diversify_documents: bool = False,
    ) -> dict:
        """Build metadata from an already-collected pair of routes."""
        all_fused = self._fused_rows(
            routes, limit, required_groups,
            diversify_documents=diversify_documents,
        )
        display_rows, display_mode = self._display_rows(
            query, routes, limit, required_groups,
            diversify_documents=diversify_documents,
        )
        route_report = {}
        for name, rows in routes.items():
            unique = _route_unique(rows, self.dedupe_by)
            status_info = route_statuses.get(name, {"status": "available", "reason": ""})
            route_report[name] = {
                "status": status_info.get("status", "available"),
                "reason": status_info.get("reason", ""),
                "refs": [str(row.get("ref", "")) for row in unique],
                "scores": [round(float(row.get("score", 0.0) or 0.0), 6) for row in unique],
                "latency_ms": round(timings.get(name, 0.0) * 1000, 1),
            }
        meta = {
            "index_version": getattr(self.store, "index_version", ""),
            "parser_version": getattr(self.store, "parser_version", ""),
            "chunk_strategy_version": getattr(
                self.store, "chunk_strategy_version",
                getattr(self.store, "parser_version", ""),
            ),
            "embedding_model": getattr(self.store, "embedding_model", ""),
        }
        lexical_status = route_report.get("lexical", {}).get("status")
        vector_status = route_report.get("vector", {}).get("status")
        if self.vector_mode != "off" and vector_status == "unavailable":
            hybrid_status = (
                "degraded_vector_unavailable"
                if lexical_status == "available" else "unavailable"
            )
        elif self.lexical_mode != "off" and lexical_status == "unavailable":
            hybrid_status = (
                "degraded_lexical_unavailable"
                if vector_status == "available" else "unavailable"
            )
        elif not route_report:
            hybrid_status = "unavailable"
        else:
            hybrid_status = "available"
        return {
            "schema": "rag-hybrid-shadow-v2",
            "query_hash": hashlib.sha256(str(query).encode("utf-8")).hexdigest(),
            "vector_mode": self.vector_mode,
            "lexical_mode": self.lexical_mode,
            "dedupe_by": self.dedupe_by,
            "candidate_k": self.candidate_k,
            "vector_min_score": self.vector_min_score,
            "lexical_min_coverage": self.lexical_min_coverage,
            "small_to_big_mode": self.small_to_big_mode,
            "small_to_big_neighbors": self.small_to_big_neighbors,
            "small_to_big_max_chars": self.small_to_big_max_chars,
            "small_to_big": dict(context_stats or {
                "mode": self.small_to_big_mode, "status": "not_collected",
            }),
            "display_mode": display_mode,
            "vector_fallback": display_mode == "guarded_hybrid_fallback",
            "memory_focused": display_mode == "memory_focused",
            "document_diversity": bool(diversify_documents),
            "identifier_lexical_miss_protection": True,
            "limit": limit,
            "project_id": project_id,
            "session_id": session_id,
            "coverage_contract": {
                "required_groups": [list(group) for group in (required_groups or ())],
                "satisfied_groups": sum(
                    1 for group in (required_groups or ())
                    if any(
                        any(_coverage_matches(expected, str(row.get("ref") or ""))
                            for expected in group)
                        for row in all_fused
                    )
                ),
            },
            "routes": route_report,
            "hybrid_refs": [str(row.get("ref", "")) for row in all_fused],
            "hybrid_entry_keys": [str(row.get("entry_key", "")) for row in all_fused],
            "hybrid_status": hybrid_status,
            "display_refs": [str(row.get("ref", "")) for row in display_rows],
            "errors": dict(errors),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "meta": meta,
        }

    def shadow_record(
        self,
        query: str,
        *,
        limit: int = 8,
        project_id: str | None = None,
        session_id: str | None = None,
        statuses: Sequence[str] | None = None,
        include_archive: bool = False,
        required_groups: Sequence[Sequence[str]] | None = None,
    ) -> dict:
        """Return a safe lexical/vector comparison record.

        Only query hash, refs, scores and timings are included.  Full content
        stays in the index and is never written to a shadow log by this API.
        """
        started = time.perf_counter()
        routes, timings, errors, route_statuses = self._collect(
            query, project_id=project_id, statuses=statuses,
            include_archive=include_archive,
        )
        document_diversity = _cross_document_query(query)
        display_rows, _display_mode = self._display_rows(
            query, routes, limit, required_groups,
            diversify_documents=document_diversity,
        )
        _display_rows, context_stats = self._apply_context(
            display_rows, project_id=project_id, statuses=statuses,
            include_archive=include_archive,
        )
        record = self._shadow_record_from_collection(
            query, limit, project_id, session_id, routes, timings, errors, route_statuses,
            started=started, context_stats=context_stats, required_groups=required_groups,
            diversify_documents=document_diversity,
        )
        if self.shadow_logger is not None and self.vector_mode == "shadow":
            try:
                self.shadow_logger(record)
            except Exception:
                pass
        return record


__all__ = ["HybridRetriever", "ShadowLogWriter", "entry_key", "hybrid_fuse"]
