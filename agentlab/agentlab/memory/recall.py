"""长期记忆召回注入（#10①/OPT-123）：本轮输入 → memory top-k → system prompt {{memory}} 槽。

注入原则（对齐 "Connecting over MCP gives your assistant the tools. It doesn't
teach it when to reach for them"）：有 memory_query 工具 ≠ agent 会想起调用——
召回必须在每轮组装 system prompt 时自动发生。零 LLM 成本：topic 直接用用户输入，
brain LIKE 命中 + 时间衰减重排（复用 MemoryStore.query/OPT-101 口径）；
仓库不可用/空结果静默降级为占位文案，绝不阻断请求。
"""
from __future__ import annotations

from agentlab.memory.capture import govern_capture

# 注入是 Tier-0 上下文，单条预算比沉淀（2000）紧得多
RECALL_ITEM_CHARS = 400
RECALL_FALLBACK = "（本轮无相关长期记忆召回）"


def open_memory_store(cfg):
    """构造 MemoryStore；brain 不可用/构造失败 → None（测试可 patch 此工厂）。"""
    try:
        from agentlab.memory.store import MemoryStore
        from agentlab.tools.connectors.brain_tools import load_brain_config

        store = MemoryStore(load_brain_config(cfg.model_dump()))
        return store if store.available else None
    except Exception:  # noqa: BLE001 —— 召回是增强能力，失败静默降级
        return None


def build_memory_block(results: list, per_item_chars: int = RECALL_ITEM_CHARS) -> str:
    """召回结果 → 注入块（纯函数）：一条一行，带首个 tag 与日期；条目容忍 dict/属性。"""
    lines: list[str] = []
    for r in results or []:
        if isinstance(r, dict):
            get = lambda k, d=None: r.get(k, d)  # noqa: E731
        else:
            get = lambda k, d=None: getattr(r, k, d)  # noqa: E731
        content = str(get("content", "") or "").strip()
        if not content:
            continue
        tags = get("tags", None) or []
        if isinstance(tags, str):
            tags = [t for t in tags.split(",") if t]
        created = str(get("created_at", "") or "")[:10]
        prefix = f"[{tags[0]}] " if tags else ""
        suffix = f"（{created}）" if created else ""
        lines.append(f"- {prefix}{govern_capture(content, per_item_chars)}{suffix}")
    return "\n".join(lines)


def memory_block_for(cfg, topic: str, topk: int = 5) -> str:
    """cfg + 本轮用户输入 → 注入块；仓库不可用/无结果/topk=0 → 空串（调用方回退占位）。"""
    if not topic or topk <= 0:
        return ""
    store = open_memory_store(cfg)
    if store is None:
        return ""
    try:
        q = store.query(topic, limit=topk)
    except Exception:  # noqa: BLE001
        return ""
    return build_memory_block(q.get("results") if isinstance(q, dict) else [])
