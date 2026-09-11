"""充分性评估 / LLM 自 rerank（docs/04 §4.3，F9）。

`RAGAssessor.assess(query, items)`：判断检索结果是否足以回答问题。
- 足够 → sufficient=True，agent 直接综合回答（引用 ref）；
- 不足且可改写 → action="reformulate"，返回 reformulated_query 供再次召回；
- 严重不足 → action="insufficient"，如实"信息不足"，避免编造。

结构化输出（JSON）强制二次校验（AGENTS.md），失败抛 AGENT_GUARDRAIL。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentlab.core.errors import AgentError
from agentlab.core.guardrails import ensure_structured
from agentlab.core.llm import LLMProvider
from agentlab.core.message import Message
from agentlab.prompts import load_prompt
from agentlab.rag.recall import RecallItem


@dataclass
class Sufficiency:
    sufficient: bool
    action: str              # "answer" | "reformulate" | "insufficient"
    reformulated_query: str = ""
    message: str = ""
    refs: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "sufficient": self.sufficient, "action": self.action,
            "reformulated_query": self.reformulated_query,
            "message": self.message, "refs": self.refs or [],
        }


def _item_block(items: list[RecallItem]) -> str:
    if not items:
        return "（无检索结果）"
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(
            f"[{i}] <{it.source}/{it.ref}> {it.title}\n"
            f"    {it.content[:300]}"
        )
    return "\n".join(lines)


def _schema_ok(obj: Any) -> bool:
    """结构化输出谓词：sufficient 为 bool，action 在枚举内。"""
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("sufficient"), bool):
        return False
    return obj.get("action") in ("answer", "reformulate", "insufficient")


class RAGAssessor:
    def __init__(self, llm: LLMProvider, temperature: float = 0.2):
        self._llm = llm
        self._temperature = temperature

    async def assess(self, query: str, items: list[RecallItem],
                     refs: list[str] | None = None) -> Sufficiency:
        """执行充分性评估：渲染 rag-assess-user.st → 结构化二次校验 → Sufficiency。"""
        if not items:
            return Sufficiency(sufficient=False, action="insufficient",
                               message="暂无检索结果，无法作答", refs=refs or [])
        prompt = load_prompt(
            "rag-assess-user", query=query, items=_item_block(items), refs=", ".join(refs or [])
        )
        resp = await self._llm.chat(
            [Message(role="user", content=prompt)], tools=None, temperature=self._temperature
        )
        output = resp.content
        if not ensure_structured(output, [_schema_ok]):
            raise AgentError("AGENT_GUARDRAIL", "充分性评估输出未通过结构化校验")
        obj = obj_from(output)
        return Sufficiency(
            sufficient=bool(obj["sufficient"]),
            action=str(obj["action"]),
            reformulated_query=str(obj.get("reformulated_query") or ""),
            message=str(obj.get("message") or ""),
            refs=obj.get("refs") or refs or [],
        )


def obj_from(output: str) -> dict:
    """供测试复用的 JSON 抽取（走 guardrails.extract_json 保证剥壳/配对）。"""
    from agentlab.core.guardrails import extract_json
    return extract_json(output)