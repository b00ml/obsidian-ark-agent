"""RAG 工具：把可切换、可叠加的 Agentic RAG 封装成 ReAct Tool。

"检索即工具、由 agent 决策"（docs/04 §4.1）：
- rag_retrieve：多路召回（关键词 vault/memory/web + 可选向量语义路）+ RRF 融合，
  返回结构化 JSON 文本（read，无需 HITL）。
- rag_assess：充分性评估，返回 {sufficient, action, reformulated_query...}（read）；
  引擎需要 LLM，故所需的 RAGAssessor 在构造时注入闭包。
- rag_reindex：增量同步 Vault → 本地 P2 索引（write；按配置注册，配 provider 时补齐向量）。

brain/llm 未就绪时工具降级为清晰的结果提示，不抛错中断循环。
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from contextvars import ContextVar
from pathlib import Path

from agentlab.core.llm import LLMProvider
from agentlab.contracts import CitationStatus, RetrievalScope, current_retrieval_scope, strategy_from_modes
from agentlab.rag.assess import RAGAssessor
from agentlab.rag.citations import CitationRegistry
from agentlab.rag.recall import RAGRecall
from agentlab.tools.base import tool


# A model will occasionally compress a previous ``rag_retrieve`` response to
# a comma-separated list of paths before calling ``rag_assess``.  Keep the
# latest governed candidate rows in the current async request context so the
# assessor can still inspect the real evidence without sharing state across
# concurrent serve requests.  This is a compatibility fallback only; explicit
# JSON candidate payloads remain the primary contract.
_LAST_RETRIEVAL_ITEMS: ContextVar[tuple[str, list[dict]] | None] = ContextVar(
    "agentlab_last_retrieval_items", default=None
)
_LAST_CITATION_REGISTRY: ContextVar[tuple[str, list[dict], CitationRegistry] | None] = ContextVar(
    "agentlab_last_citation_registry", default=None
)


def _query_compatible(current: str, previous: str) -> bool:
    """Reject stale cross-request evidence while allowing query rewrites."""
    left = re.sub(r"\s+", "", str(current or "").strip().casefold())
    right = re.sub(r"\s+", "", str(previous or "").strip().casefold())
    return bool(left and right and (left == right or left in right or right in left))


def _redact_query_plan(plan: dict | None) -> dict | None:
    """Keep query classification telemetry free of raw user query text."""
    if not isinstance(plan, dict):
        return None
    value = dict(plan)
    original = str(value.pop("original_query", "") or "")
    variants = value.pop("variants", [])
    value["original_query_hash"] = hashlib.sha256(
        original.encode("utf-8")
    ).hexdigest() if original else ""
    value["variant_hashes"] = [
        hashlib.sha256(str(item).encode("utf-8")).hexdigest()
        for item in variants if str(item)
    ]
    return value


def _record_retrieval_evidence(query: str, rows: list[dict], scope: RetrievalScope) -> None:
    """Bind immutable retrieval rows to a run-local citation registry."""
    canonical = [dict(row) for row in rows if isinstance(row, dict)]
    registry = CitationRegistry()
    for row in canonical:
        ref = str(row.get("ref") or "").strip()
        if not ref:
            continue
        status = str(row.get("status") or "active").strip().lower()
        registry.register({
            "ref": ref,
            "source_kind": str(row.get("source") or "vault"),
            "project_id": str(row.get("project_id") or scope.project_id),
            "session_id": str(row.get("session_id") or scope.session_id),
            "status": (CitationStatus.ACTIVE.value if status in {"active", "current", "available"}
                       else CitationStatus.UNVERIFIED.value),
        }, content=str(row.get("content") or ""))
    _LAST_RETRIEVAL_ITEMS.set((query, canonical))
    _LAST_CITATION_REGISTRY.set((query, canonical, registry))


def build_vector_index(rag_config=None, vault_root: str = "", embedder=None):
    """构建向量语义路索引（P0-1）；未启用/构建失败返回 None。

    供 build_rag_tools 与 L11 区段档案（serve 侧 RangeGateway）共用同一实例
    ——vault 分块与会话区段同库不同表，避免双实例写同一 SQLite 抢锁。
    """
    if rag_config is None or not getattr(rag_config, "vector_enabled", False) \
            or getattr(rag_config, "vector_mode", "shadow") == "off" \
            or not getattr(rag_config, "embed_base_url", ""):
        return None
    try:
        from agentlab.rag.vector_index import VectorIndex
        emb = embedder
        if emb is None:
            from agentlab.rag.embed import OpenAIEmbedder
            emb = OpenAIEmbedder(rag_config.embed_base_url, rag_config.embed_model,
                                 api_key=rag_config.effective_key(),
                                 timeout=rag_config.embed_timeout)
        db = Path(vault_root or ".") / ".agent-brain" / "rag-index.sqlite"
        return VectorIndex(db, emb, chunk_chars=rag_config.chunk_chars,
                           vault_root=Path(vault_root) if vault_root else None,
                           auto_sync_limit=rag_config.auto_sync_limit)
    except Exception:
        return None  # 向量路构建失败 → 纯关键词路，不抛错


def build_p2_store(rag_config=None, vault_root: str = "", embedder=None):
    """Build the versioned P2 store when its backend is selected.

    ``auto`` only selects a current, initialized P2 SQLite file.  This keeps a
    fresh or stale deployment on the legacy path until an operator has run
    ``rag_reindex`` (or explicitly selected ``index_backend=p2``).  P2 may be
    lexical-only when no embedding endpoint is configured; vector search then
    reports provider-unavailable and HybridRetriever safely keeps lexical
    results.
    """
    if rag_config is None or not vault_root:
        return None
    backend = getattr(rag_config, "index_backend", "auto")
    vector_enabled = bool(getattr(rag_config, "vector_enabled", False))
    vector_mode = getattr(rag_config, "vector_mode", "shadow")
    configured_path = str(getattr(rag_config, "index_path", "") or "").strip()
    path = Path(configured_path) if configured_path else (
        Path(vault_root) / ".agent-brain" / "rag-index-p2.sqlite"
    )
    endpoint = str(getattr(rag_config, "embed_base_url", "") or "").strip()
    if backend == "legacy" or (backend == "auto" and not path.exists()):
        return None
    try:
        from agentlab.rag.index_store import RagIndexStore
        emb = embedder
        if emb is None and vector_enabled and vector_mode != "off" \
                and getattr(rag_config, "embed_base_url", ""):
            from agentlab.rag.embed import OpenAIEmbedder
            emb = OpenAIEmbedder(
                rag_config.embed_base_url,
                rag_config.embed_model,
                api_key=rag_config.effective_key(),
                timeout=rag_config.embed_timeout,
            )
        store = RagIndexStore(
            path, emb, vault_root=vault_root,
            embedding_model=getattr(rag_config, "embed_model", "") if emb else None,
            index_version=getattr(rag_config, "index_version", "s1-p2-v1"),
            parser_version=getattr(
                rag_config, "chunk_strategy", "markdown-structure-v1-entry-tags-v1",
            ),
            chunk_strategy_version=getattr(
                rag_config, "chunk_strategy", "markdown-structure-v1-entry-tags-v1",
            ),
            query_cache_size=getattr(rag_config, "query_embedding_cache_size", 256),
        )
        if backend == "auto":
            # File existence alone is not an activation signal: an interrupted
            # build and a stale index are both SQLite files.  Keep the legacy
            # route live until the derived P2 copy matches the Markdown source.
            status = store.index_status(vault_root)
            if (
                not status["metadata_valid"]
                or status["index_version"] != store.index_version
                or status["parser_version"] != store.parser_version
                or status["chunk_strategy_version"] != store.chunk_strategy_version
                or status["stale_files"]
            ):
                return None
        return store
    except Exception:
        return None


def build_rag_tools(brain_config: dict | None = None,
                    llm: LLMProvider | None = None,
                    item_chars: int = 1200,
                    rag_config=None,
                    vault_root: str = "",
                    embedder=None,
                    index=None,
                    p2_store=None,
                    range_gateway=None) -> list:
    """构造 rag_retrieve / rag_assess（/ rag_reindex）Tool。

    - recallers：客户端可注入自定义召回列表（测试）；默认借 brain + web_search。
    - llm：rag_assess 的充分性评估引擎；None 时 rag_assess 降级返回不可用提示。
    - item_chars：recall 注入 Tier-1 单条正文预算（OPT-089）。
    - rag_config + vault_root（P0-1/OPT-105）：无 embed_base_url 时只使用轻量关键词/
      memory/web 路；`vector_mode=on` 追加向量路并由 RRF 融合，`shadow` 只观测不展示。
      构建失败静默降级为纯关键词路。embedder 参数供测试注入 Fake。
    - index（OPT-111）：外部构建好的 VectorIndex；None 时按 rag_config 内建。
    - range_gateway（L11/OPT-111）：会话区段档案网关；注入后 rag_retrieve 多一路
      "session" 召回（当前会话窗口外内容，未绑定会话时该路返回空）。
    """
    # P2 is selected independently from the legacy VectorIndex.  This keeps
    # the old instance available for RangeGateway while the user-facing
    # vault route migrates one SQLite file at a time.
    p2 = p2_store or build_p2_store(rag_config, vault_root, embedder=embedder)
    if p2 is None and index is None:
        index = build_vector_index(rag_config, vault_root, embedder=embedder)
    extra = []
    vector_extra = []
    p2_recaller = None
    hybrid = None
    if p2 is not None:
        from agentlab.rag.hybrid import HybridRetriever, ShadowLogWriter
        shadow_path = getattr(rag_config, "shadow_log_path", "") if rag_config else ""
        hybrid = HybridRetriever(
            p2,
            vector_mode=(getattr(rag_config, "vector_mode", "shadow")
                         if getattr(rag_config, "vector_enabled", True) else "off"),
            lexical_mode=getattr(rag_config, "lexical_mode", "on"),
            candidate_k=getattr(rag_config, "hybrid_candidate_k", 40),
            dedupe_by=getattr(rag_config, "dedupe_by", "entry"),
            vector_min_score=getattr(rag_config, "vector_min_score", 0.0),
            lexical_min_coverage=getattr(rag_config, "lexical_min_coverage", 0.0),
            small_to_big_mode=getattr(rag_config, "small_to_big_mode", "shadow"),
            small_to_big_neighbors=getattr(rag_config, "small_to_big_neighbors", 1),
            small_to_big_max_chars=getattr(rag_config, "small_to_big_max_chars", 2400),
            vector_fallback_mode=getattr(rag_config, "vector_fallback_mode", "off"),
            shadow_logger=ShadowLogWriter(shadow_path) if shadow_path else None,
        )

        sync_lock = threading.Lock()
        last_sync = [0.0]

        def p2_recall(query: str, scope: RetrievalScope | None = None) -> list[dict]:
            # Ark also triggers reconcile on Vault events.  This bounded,
            # throttled check covers startup races and external edits while
            # keeping normal queries off the full scan/embedding path.
            now = time.monotonic()
            if now - last_sync[0] >= 5.0:
                with sync_lock:
                    now = time.monotonic()
                    if now - last_sync[0] >= 5.0:
                        try:
                            p2.sync_vault(
                                Path(vault_root),
                                max_files=getattr(rag_config, "auto_sync_limit", 8),
                            )
                        except Exception:
                            pass
                        last_sync[0] = now
            request_scope = scope or current_retrieval_scope()
            project_id = request_scope.project_id or None
            statuses = request_scope.statuses or None
            limit = getattr(rag_config, "hybrid_candidate_k", 40)
            if getattr(rag_config, "vector_mode", "shadow") == "shadow":
                items, _record = hybrid.retrieve_with_shadow_record(
                    query, limit=limit, project_id=project_id,
                    session_id=request_scope.session_id or None, statuses=statuses,
                    include_archive=request_scope.include_archive,
                )
                p2_recall._last_record = _record
                p2_recall._last_route_status = {
                    name: dict(value) for name, value in
                    getattr(p2, "last_search_status", {}).items()
                    if isinstance(value, dict)
                }
                return [item.to_dict() for item in items]
            items = hybrid.retrieve(
                query, limit=limit, project_id=project_id, statuses=statuses,
                include_archive=request_scope.include_archive,
            )
            p2_recall._last_route_status = {
                name: dict(value) for name, value in
                getattr(p2, "last_search_status", {}).items()
                if isinstance(value, dict)
            }
            return [item.to_dict() for item in items]

        p2_recaller = p2_recall
        p2_recaller._rag_source = "p2"
        extra.append(p2_recaller)
    elif index is not None:
        from agentlab.rag.vector_index import make_vector_recaller
        vector_recaller = make_vector_recaller(index, k=getattr(rag_config, "vector_k", 6))
        vector_recaller._rag_scope_unsupported = True
        vector_extra.append(vector_recaller)
        extra.extend(vector_extra)
    if range_gateway is not None:
        extra.append(range_gateway.recaller())

    # Keep the established keyword/memory/web route unless explicitly disabled.
    # The vector route is optional and is appended to the same candidate set,
    # so ``vector_mode=on`` is additive rather than a replacement.
    vector_mode = getattr(rag_config, "vector_mode", "shadow") if rag_config else "off"
    retrieval_strategy = strategy_from_modes(
        vector_enabled=bool(getattr(rag_config, "vector_enabled", False)) if rag_config else False,
        vector_mode=vector_mode,
        lexical_mode=getattr(rag_config, "lexical_mode", "on") if rag_config else "on",
    )
    if getattr(rag_config, "lexical_mode", "on") == "off" and extra:
        recall = RAGRecall(recallers=[], extra=extra,
                           strategy=retrieval_strategy, brain_config=brain_config,
                           rewrite_provider=llm,
                           query_rewrite_mode=getattr(
                               rag_config, "query_rewrite_mode", "off"),
            query_rewrite_max_variants=getattr(
                               rag_config, "query_rewrite_max_variants", 1),
                           rewrite_deadline_ms=getattr(
                               rag_config, "rewrite_deadline_ms", 250),
                           query_expansion_mode=getattr(
                               rag_config, "query_expansion_mode", "shadow"),
                           query_expansion_max_variants=getattr(
                               rag_config, "query_expansion_max_variants", 5),
                           query_expansion_min_candidates=getattr(
                               rag_config, "query_expansion_min_candidates", 8))
    elif p2 is None and vector_mode == "shadow" and extra:
        # Only the vector route is shadowed; session-range recall remains a
        # real user-facing route even while vector results are observed.
        recall = RAGRecall(extra=[item for item in extra if item not in vector_extra],
                           shadow_extra=vector_extra, strategy=retrieval_strategy,
                           brain_config=brain_config,
                           rewrite_provider=llm,
                           query_rewrite_mode=getattr(
                               rag_config, "query_rewrite_mode", "off"),
                           query_rewrite_max_variants=getattr(
                               rag_config, "query_rewrite_max_variants", 1),
                           rewrite_deadline_ms=getattr(
                               rag_config, "rewrite_deadline_ms", 250),
                           query_expansion_mode=getattr(
                               rag_config, "query_expansion_mode", "shadow"),
                           query_expansion_max_variants=getattr(
                               rag_config, "query_expansion_max_variants", 5),
                           query_expansion_min_candidates=getattr(
                               rag_config, "query_expansion_min_candidates", 8))
    else:
        # P2 owns the Vault lexical route.  Keep memory/web/session routes,
        # but do not run the legacy Vault search in parallel: its file refs
        # would duplicate P2 chunk refs in the fused result.
        recall = RAGRecall(
            extra=extra or None,
            exclude_sources={"vault"} if p2 is not None else None,
            strategy=retrieval_strategy,
            brain_config=brain_config,
            rewrite_provider=llm,
            query_rewrite_mode=getattr(
                rag_config, "query_rewrite_mode", "off"),
            query_rewrite_max_variants=getattr(
                rag_config, "query_rewrite_max_variants", 1),
            rewrite_deadline_ms=getattr(
                rag_config, "rewrite_deadline_ms", 250),
            query_expansion_mode=getattr(
                rag_config, "query_expansion_mode", "shadow"),
            query_expansion_max_variants=getattr(
                rag_config, "query_expansion_max_variants", 5),
            query_expansion_min_candidates=getattr(
                rag_config, "query_expansion_min_candidates", 8),
        )

    @tool(
        name="rag_retrieve",
        description="Agentic 多路检索：关键词路默认可用；配置 embedding 后可叠加本地向量语义检索，"
                    "两路融合 Vault 笔记/长期记忆/全网结果 + 当前会话历史区段，返回 top-k 命中。"
                    "需要先检索再据此回答或评估时调用；找会话早前细节也应调用。"
                    "传 envelope=true 可返回 S0 统一结果信封（默认兼容裸列表）。",
        permission="read",
    )
    def rag_retrieve(query: str, limit: int = 8, envelope: bool = False,
                     metadata: bool = False) -> str:
        request_scope = current_retrieval_scope()
        if envelope or metadata:
            result = recall.retrieve_result(
                query, limit=limit, scope=request_scope,
                per_item_chars=item_chars,
            )
            payload = result.to_dict()
            if metadata:
                # Diagnostics are opt-in so the frozen S0 envelope remains
                # stable for normal callers; query text is redacted by RAGRecall.
                payload["query_plan"] = _redact_query_plan(recall.last_query_plan)
                payload["query_rewrite"] = recall.last_query_rewrite
                payload["query_expansion"] = recall.last_query_expansion
                payload["timings_ms"] = recall.last_timings
                payload["route_timings_ms"] = recall.last_route_timings
                payload["route_metadata"] = recall.last_route_metadata
            _record_retrieval_evidence(query, [
                dict(row) for row in payload.get("items", [])
                if isinstance(row, dict)
            ], request_scope)
            return json.dumps(payload, ensure_ascii=False)
        items = recall.retrieve_governed(
            query, limit=limit, per_item_chars=item_chars, scope=request_scope,
        )  # Tier-1/2: 单条截断+身份锚保留
        rows = [it.to_dict() for it in items]
        _record_retrieval_evidence(query, rows, request_scope)
        return json.dumps(rows, ensure_ascii=False)

    tools_out = [rag_retrieve]

    reindex_target = p2 or index
    if reindex_target is not None:
        @tool(
            name="rag_reindex",
            description="增量同步 Vault → 本地向量索引（语义检索路的数据源）。"
                        "新增/修改笔记后调用；正常 reconcile 只处理 hash 变化、删除和缺失向量，"
                        "不会每日全量嵌入。首次初始化、模型或分块版本变更时才需经审批执行全量重建。"
                        "写侧工具：更新索引并可能产生 embedding 费用，调用需审批；"
                        "检索用 rag_retrieve，不要为单次问答主动重建索引。",
            permission="write",
            side_effects="index",
            idempotent=True,
        )
        def rag_reindex() -> str:
            if p2 is not None:
                # P2 owns lexical postings and vectors in one transaction.  A
                # caller can therefore initialize/migrate it through the same
                # approved write tool used by the legacy route.
                return json.dumps(p2.sync_vault(Path(vault_root)), ensure_ascii=False)
            return json.dumps(index.sync_vault(Path(vault_root)), ensure_ascii=False)

        tools_out.append(rag_reindex)

    assessor = RAGAssessor(llm) if llm is not None else None

    @tool(
        name="rag_assess",
        description="检索充分性评估：判断当前检索结果是否足以回答问题，返回 "
                    "{sufficient, action, reformulated_query, message, refs}。"
                    "action=reformulate 时可用 reformulated_query 再次检索；"
                    "items 应原样传入最近一次 rag_retrieve 返回的 JSON 候选列表，"
                    "不要只传文件名或路径摘要；若模型压缩了参数，工具会在当前请求内"
                    "回退到最近一次真实候选。生产调用会先执行引用、scope、状态和"
                    "answerability 闸门。",
        permission="read",
    )
    async def rag_assess(query: str, items: str,
                         required_refs: list[str] | None = None,
                         forbidden_refs: list[str] | None = None,
                         answerability: str = "answerable") -> str:
        # ``answerability`` is a trusted task/evaluator label, not a model
        # control plane.  A model can call this tool after seeing an earlier
        # insufficient result, but it must not be able to make that label
        # sticky (or manufacture absent/conflicting evidence) by echoing it
        # back as an argument.  Deterministic evidence/scope checks below and
        # the assessor's structured output remain authoritative.
        effective_answerability = "answerable"
        if assessor is None:
            from agentlab.rag.assess import Sufficiency
            try:
                raw_items = json.loads(items) if items else []
                if isinstance(raw_items, dict):
                    raw_items = raw_items.get("items") or []
                from agentlab.rag.assess import evaluate_answer_gate
                gate = evaluate_answer_gate(
                    query, raw_items if isinstance(raw_items, list) else [],
                    scope=current_retrieval_scope(), required_refs=required_refs,
                    forbidden_refs=forbidden_refs,
                    answerability=effective_answerability,
                )
                if not gate.allowed:
                    return json.dumps(Sufficiency(
                        sufficient=False, action="insufficient",
                        message="证据闸门未通过：" + "；".join(gate.reasons),
                    ).to_dict(), ensure_ascii=False)
            except Exception:
                pass
            return json.dumps(Sufficiency(
                sufficient=False, action="insufficient",
                message="未配置 LLM，无法评估充分性").to_dict(), ensure_ascii=False)
        try:
            parsed = json.loads(items) if items else []
            if isinstance(parsed, dict):
                parsed = parsed.get("items") or []
            cached = _LAST_RETRIEVAL_ITEMS.get()
            evidence = _LAST_CITATION_REGISTRY.get()
            cached_rows = (
                cached[1] if isinstance(cached, tuple) and len(cached) == 2
                and _query_compatible(query, cached[0]) else []
            )
            registry_rows = (
                evidence[1] if isinstance(evidence, tuple) and len(evidence) == 3
                and _query_compatible(query, evidence[0]) else []
            )
            registry = evidence[2] if registry_rows else None
            if registry is not None:
                allowed, _reasons = registry.validate(
                    [str(row.get("ref") or "") for row in registry_rows],
                    scope=current_retrieval_scope(),
                )
                allowed_refs = {citation.ref for citation in allowed}
                # The tool argument is model-controlled.  Once a current
                # registry exists, only its canonical rows may reach the LLM
                # assessor; otherwise a matching fake ref/content pair could
                # manufacture evidence within the same ReAct request.
                parsed = [row for row in registry_rows if row.get("ref") in allowed_refs]
            if registry is None and (
                not isinstance(parsed, list) or not any(isinstance(o, dict) for o in parsed)
            ):
                # Compatibility path for compressed model arguments.  Never
                # synthesize evidence from the path summary itself.
                parsed = cached_rows
            from agentlab.rag.recall import RecallItem
            objs = [RecallItem(**{k: o.get(k) for k in (
                "title", "content", "ref", "source", "status",
                "project_id", "session_id") if k in o})
                    for o in parsed if isinstance(o, dict)]
        except Exception:
            # Compressed path summaries are not JSON.  Use only the
            # request-local rows captured by ``rag_retrieve``; never treat
            # arbitrary text as evidence.
            cached = _LAST_RETRIEVAL_ITEMS.get()
            parsed = (
                cached[1] if isinstance(cached, tuple) and len(cached) == 2
                and _query_compatible(query, cached[0]) else []
            )
            objs = []
            try:
                from agentlab.rag.recall import RecallItem
                objs = [RecallItem(**{k: o.get(k) for k in (
                    "title", "content", "ref", "source", "status",
                    "project_id", "session_id") if k in o})
                        for o in parsed if isinstance(o, dict)]
            except Exception:
                objs = []
        candidate_refs = [item.ref for item in objs if item.ref]
        res = await assessor.assess(
            query, objs, refs=candidate_refs,
            scope=current_retrieval_scope(),
            required_refs=required_refs,
            forbidden_refs=forbidden_refs,
            answerability=effective_answerability,
        )
        return json.dumps(res.to_dict(), ensure_ascii=False)

    # The direct evaluator uses this read-only counter to distinguish a real
    # assessor model call from the deterministic no-provider fallback.
    rag_assess.fn.llm_call_count = lambda: int(getattr(assessor, "calls", 0)) if assessor else 0

    # These are evaluator/runtime controls, not model-facing retrieval
    # inputs.  Leaving them in the OpenAI schema invites the model to copy a
    # stale ``answerability`` or invent ``required_refs`` from prose, which
    # deterministically turns an otherwise usable candidate set into a false
    # refusal.  Keep the Python callable backward-compatible for tests and
    # trusted callers, but expose only the two actual model inputs.
    rag_assess.schema["function"]["parameters"]["properties"] = {
        "query": {"type": "string"},
        "items": {"type": "string"},
    }
    tools_out.append(rag_assess)
    return tools_out


RAG_TOOLS = ["rag_retrieve", "rag_assess", "rag_reindex"]
