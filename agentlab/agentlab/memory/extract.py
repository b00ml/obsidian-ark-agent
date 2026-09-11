"""LLM 语义记忆提取（#10①/OPT-123）：对话片段 → 结构化记忆草稿 → 沉淀。

升级原 20 触发词正则金标准：提取交 LLM（memory-extract-user.st，importance
1-10 打分、低分不出），写入仍走 MemoryStore.commit（治理/去重在仓库侧不变）。
提取器缺席（无 key/构造失败）时 loop 自动回退触发词路——降级语义与 serve 一致。
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from agentlab.core.message import Message

_log = logging.getLogger(__name__)


class MemoryExtractor:
    """LLM 提取器：entries（user/assistant 消息）→ [{content, tags, importance}]。"""

    def __init__(self, provider, per_item_chars: int = 2000, max_items: int = 6,
                 min_importance: int = 4):
        self._provider = provider
        self._per_item_chars = per_item_chars
        self._max_items = max_items
        self._min_importance = min_importance

    async def extract(self, entries: list) -> list[dict]:
        from agentlab.prompts import load_prompt

        transcript = _render_transcript(entries)
        if not transcript:
            return []
        prompt = load_prompt("memory-extract-user", transcript=transcript,
                             max_items=str(self._max_items))
        resp = await self._provider.chat([Message(role="user", content=prompt)])
        return _parse_drafts(resp.content or "", self._max_items,
                             self._min_importance, self._per_item_chars)


def _render_transcript(entries: list) -> str:
    """只取 user/assistant 正文（工具/系统消息不进提取素材）。"""
    lines: list[str] = []
    for m in entries or []:
        role = getattr(m, "role", "")
        if role not in ("user", "assistant"):
            continue
        content = (getattr(m, "content", None) or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _parse_drafts(raw: str, max_items: int, min_importance: int,
                  per_item_chars: int) -> list[dict]:
    """容错解析 LLM 输出：取首个 '[' 到末个 ']' 之间当 JSON；低分/空内容丢弃。"""
    text = (raw or "").strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "") or "").strip()
        if not content:
            continue
        try:
            importance = int(item.get("importance", 5))
        except (TypeError, ValueError):
            importance = 5
        if importance < min_importance:
            continue
        tags = item.get("tags") or []
        if not isinstance(tags, list):
            tags = [tags]
        body = content if per_item_chars <= 0 else content[:per_item_chars]
        out.append({"content": body, "tags": [str(t) for t in tags][:6],
                    "importance": importance})
        if len(out) >= max_items:
            break
    return out


async def deposit_via_extractor(depo: Any, extractor: Any, entries: list,
                                source_session: str = "") -> int:
    """提取 + 逐条落库（同步 commit 放线程池）；返回 committed 条数，失败仅留痕。"""
    try:
        drafts = await extractor.extract(entries)
    except Exception:  # noqa: BLE001 —— 提取失败不影响主循环
        _log.exception("记忆提取失败（extractor=%s）", type(extractor).__name__)
        return 0
    loop = asyncio.get_running_loop()
    committed = 0
    for d in drafts:
        try:
            r = await loop.run_in_executor(
                None, lambda c=d: depo.commit(c["content"], tags=c.get("tags"),
                                              source_session=source_session, dedup=True))
            if isinstance(r, dict) and r.get("status") == "committed":
                committed += 1
        except Exception:  # noqa: BLE001
            _log.exception("记忆沉淀失败（depository=%s）", type(depo).__name__)
    return committed
