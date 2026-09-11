"""Agentic RAG（docs/04 §4，F9）：多路召回融合 + 充分性评估。

- recall.RAGRecall.retrieve()：vault/memory/graph/web 多路召回 + 融合打分排序
- assess.RAGAssessor.assess()：LLM 充分性评估（改查询重检 / 信息不足如实说明）

框架只做编排；每路召回复用 brain 工具（MemoryStore/vault_search/web_search），
不重复造检索逻辑（AGENTS.md 能力分流）。检索是 ReAct 里的工具调用，由 agent 决策。
"""
from agentlab.rag.recall import RAGRecall, RecallItem  # noqa: F401
from agentlab.rag.assess import RAGAssessor, Sufficiency  # noqa: F401