"""RAG 工具：把 Agentic RAG 封装成 ReAct 里可自主调用的 Tool（F9；P0-1/OPT-105 增向量路）。

"检索即工具、由 agent 决策"（docs/04 §4.1）：
- rag_retrieve：多路召回（关键词 vault/memory/web + 向量语义路）+ RRF 融合，
  返回结构化 JSON 文本（read，无需 HITL）。
- rag_assess：充分性评估，返回 {sufficient, action, reformulated_query...}（read）；
  引擎需要 LLM，故所需的 RAGAssessor 在构造时注入闭包。
- rag_reindex：增量同步 Vault → 本地向量索引（read；配了 rag.embed_base_url 才注册）。

brain/llm 未就绪时工具降级为清晰的结果提示，不抛错中断循环。
"""
from __future__ import annotations

import json
from pathlib import Path

from agentlab.core.llm import LLMProvider
from agentlab.rag.assess import RAGAssessor
from agentlab.rag.recall import RAGRecall
from agentlab.tools.base import tool


def build_vector_index(rag_config=None, vault_root: str = "", embedder=None):
    """构建向量语义路索引（P0-1）；未启用/构建失败返回 None。

    供 build_rag_tools 与 L11 区段档案（serve 侧 RangeGateway）共用同一实例
    ——vault 分块与会话区段同库不同表，避免双实例写同一 SQLite 抢锁。
    """
    if rag_config is None or not getattr(rag_config, "vector_enabled", False) \
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


def build_rag_tools(brain_config: dict | None = None,
                    llm: LLMProvider | None = None,
                    item_chars: int = 1200,
                    rag_config=None,
                    vault_root: str = "",
                    embedder=None,
                    index=None,
                    range_gateway=None) -> list:
    """构造 rag_retrieve / rag_assess（/ rag_reindex）Tool。

    - recallers：客户端可注入自定义召回列表（测试）；默认借 brain + web_search。
    - llm：rag_assess 的充分性评估引擎；None 时 rag_assess 降级返回不可用提示。
    - item_chars：recall 注入 Tier-1 单条正文预算（OPT-089）。
    - rag_config + vault_root（P0-1/OPT-105）：配置了 embed_base_url 时启用向量
      语义路（本地 SQLite 索引 + Recaller 追加到默认路），并注册 rag_reindex；
      构建失败静默降级为纯关键词路。embedder 参数供测试注入 Fake。
    - index（OPT-111）：外部构建好的 VectorIndex；None 时按 rag_config 内建。
    - range_gateway（L11/OPT-111）：会话区段档案网关；注入后 rag_retrieve 多一路
      "session" 召回（当前会话窗口外内容，未绑定会话时该路返回空）。
    """
    if index is None:
        index = build_vector_index(rag_config, vault_root, embedder=embedder)
    extra = []
    if index is not None:
        from agentlab.rag.vector_index import make_vector_recaller
        extra.append(make_vector_recaller(index, k=getattr(rag_config, "vector_k", 6)))
    if range_gateway is not None:
        extra.append(range_gateway.recaller())

    recall = RAGRecall(extra=extra or None)

    @tool(
        name="rag_retrieve",
        description="Agentic 多路检索：融合 Vault 笔记/长期记忆/全网 + 本地向量语义检索"
                    " + 当前会话历史区段（压缩出窗的内容），返回 top-k 命中（含来源引用）。"
                    "需要先检索再据此回答或评估时调用；找会话早前细节也应调用。",
        permission="read",
    )
    def rag_retrieve(query: str, limit: int = 8) -> str:
        items = recall.retrieve_governed(query, limit=limit, per_item_chars=item_chars)  # Tier-1/2: 单条截断+身份锚保留
        return json.dumps([it.to_dict() for it in items], ensure_ascii=False)

    tools_out = [rag_retrieve]

    if index is not None:
        @tool(
            name="rag_reindex",
            description="增量同步 Vault → 本地向量索引（语义检索路的数据源）。"
                        "新增/修改笔记后调用；首次调用会全量嵌入全部笔记（耗时与费用随库规模）。",
            permission="read",
        )
        def rag_reindex() -> str:
            return json.dumps(index.sync_vault(Path(vault_root)), ensure_ascii=False)

        tools_out.append(rag_reindex)

    assessor = RAGAssessor(llm) if llm is not None else None

    @tool(
        name="rag_assess",
        description="检索充分性评估：判断当前检索结果是否足以回答问题，返回 "
                    "{sufficient, action, reformulated_query, message, refs}。"
                    "action=reformulate 时可用 reformulated_query 再次检索。",
        permission="read",
    )
    async def rag_assess(query: str, items: str) -> str:
        if assessor is None:
            from agentlab.rag.assess import Sufficiency
            return json.dumps(Sufficiency(
                sufficient=False, action="insufficient",
                message="未配置 LLM，无法评估充分性").to_dict(), ensure_ascii=False)
        try:
            parsed = json.loads(items) if items else []
            from agentlab.rag.recall import RecallItem
            objs = [RecallItem(**{k: o.get(k) for k in ("title", "content", "ref", "source")})
                    for o in parsed if isinstance(o, dict)]
        except Exception:
            objs = []
        res = await assessor.assess(query, objs)
        return json.dumps(res.to_dict(), ensure_ascii=False)

    tools_out.append(rag_assess)
    return tools_out


RAG_TOOLS = ["rag_retrieve", "rag_assess", "rag_reindex"]
