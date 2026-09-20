"""Opt-in, bounded direct query rewrite.

This module is an adapter only.  It never runs unless the caller explicitly
selects ``mode=on`` or ``mode=shadow`` and it always returns the original query
as a safe fallback.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import asdict, dataclass
from typing import Any

from agentlab.core.guardrails import extract_json
from agentlab.core.llm import LLMProvider
from agentlab.core.message import Message
from agentlab.prompts import load_prompt
from agentlab.rag.query import build_query_plan


@dataclass(frozen=True)
class RewriteResult:
    original_query: str
    query: str
    applied: bool
    preserved_entities: tuple[str, ...]
    reason: str
    error: str = ""
    # Shadow callers keep ``query`` equal to the original for safety, while
    # evaluators can inspect this bounded candidate without re-parsing traces.
    candidate_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["preserved_entities"] = list(self.preserved_entities)
        return value


def _same_entity(entity: str, query: str) -> bool:
    return entity.casefold() in query.casefold()


def _parse_output(raw: str, original: str, entities: tuple[str, ...]) -> tuple[str, str, tuple[str, ...]]:
    try:
        obj = extract_json(raw or "")
    except Exception:
        return "", "invalid_json", entities
    if not isinstance(obj, dict):
        return "", "invalid_shape", entities
    rewritten = str(obj.get("query") or "").strip()
    if not rewritten or len(rewritten) > 500:
        return "", "empty_or_oversized", entities
    declared = obj.get("preserved_entities")
    if declared is not None and not isinstance(declared, list):
        return "", "invalid_entities", entities
    output_entities = tuple(str(item).strip() for item in (declared or entities) if str(item).strip())[:24]
    if any(not any(entity.casefold() == original_entity.casefold() for original_entity in entities)
           for entity in output_entities):
        return "", "invented_entity", entities
    if any(not _same_entity(entity, rewritten) for entity in entities):
        return "", "entity_loss", entities
    if any(not _same_entity(entity, rewritten) for entity in output_entities):
        return "", "declared_entity_loss", entities
    return rewritten, str(obj.get("reason") or "direct_rewrite"), output_entities


async def rewrite_query(
    provider: LLMProvider,
    query: str,
    *,
    recent_context: str = "",
    lexical_coverage: float | None = None,
    mode: str = "off",
    deadline_ms: int = 250,
) -> RewriteResult:
    plan = build_query_plan(
        query, recent_context=recent_context,
        lexical_coverage=lexical_coverage, rewrite_mode=mode,
    )
    original = plan.original_query
    if not plan.should_rewrite or mode not in {"on", "shadow"}:
        return RewriteResult(original, original, False, plan.preserved_entities, plan.reason)
    context = " ".join(str(recent_context or "").split())[-1200:]
    prompt = load_prompt(
        "query-rewrite-user",
        query=original[:500],
        context=context,
        preserved_entities=", ".join(plan.preserved_entities),
    )
    try:
        response = await asyncio.wait_for(
            provider.chat([Message(role="user", content=prompt)], tools=None,
                          temperature=0.0, max_tokens=180),
            timeout=max(1, int(deadline_ms)) / 1000,
        )
    except asyncio.TimeoutError:
        return RewriteResult(original, original, False, plan.preserved_entities, "rewrite_timeout", "timeout")
    except Exception as exc:  # rewrite is optional; never block lexical retrieval
        return RewriteResult(original, original, False, plan.preserved_entities,
                             "rewrite_provider_error", type(exc).__name__)
    rewritten, reason, entities = _parse_output(response.content or "", original, plan.preserved_entities)
    if not rewritten:
        return RewriteResult(original, original, False, plan.preserved_entities, reason, reason)
    # Shadow records a candidate but does not change the query used by production callers.
    applied = mode == "on" and rewritten != original
    return RewriteResult(
        original,
        rewritten if applied else original,
        applied,
        entities,
        reason,
        "" if applied or mode == "shadow" else "not_applied",
        rewritten if rewritten != original else "",
    )


def rewrite_query_sync(provider: LLMProvider, query: str, **kwargs: Any) -> RewriteResult:
    """Run the bounded async adapter from synchronous retrieval code.

    Runtime tools normally execute in a worker thread, but direct callers may
    already own an event loop.  A short-lived helper thread keeps the sync
    boundary safe in both cases without changing the provider contract.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(rewrite_query(provider, query, **kwargs))

    result: list[RewriteResult] = []
    failure: list[BaseException] = []

    def run() -> None:
        try:
            result.append(asyncio.run(rewrite_query(provider, query, **kwargs)))
        except BaseException as exc:  # pragma: no cover - defensive boundary
            failure.append(exc)

    thread = threading.Thread(target=run, name="rag-query-rewrite", daemon=True)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    if not result:  # pragma: no cover - defensive boundary
        raise RuntimeError("rewrite thread returned no result")
    return result[0]
