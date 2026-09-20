"""Deterministic query classification for the RAG shadow path.

The classifier is deliberately cheap and non-generative.  It never replaces the
original query; a future rewrite can only be an additional variant after the
query type and safety conditions are recorded.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

QueryType = Literal["identifier", "natural", "followup", "multi_hop", "negative", "empty"]

_IDENTIFIER_RE = re.compile(
    r"(?:\b(?:BV[A-Za-z0-9]{6,}|av\d+|ep\d+|ss\d+|cv\d+)\b|(?:^|[\s/])[^\s/]+\.md\b|\b[A-Za-z_][A-Za-z0-9_]*\([^\n]*\)|\b[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+\b|\b[A-Z][A-Z0-9_-]{1,}\b)",
    re.IGNORECASE,
)
_STRONG_IDENTIFIER_RE = re.compile(
    r"(?:\b(?:BV[A-Za-z0-9]{6,}|av\d+|ep\d+|ss\d+|cv\d+)\b|(?:^|[\s/])[^\s/]+\.md\b|\b[A-Za-z_][A-Za-z0-9_]*\([^\n]*\)|\b[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+\b)",
    re.IGNORECASE,
)
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9_-]{1,}\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?[%年月日万千百]?")
_FOLLOWUP_RE = re.compile(r"(?:刚才|上面|上一条|前面|这个|那个|它|该问题|继续|接着|前述|如上|previous|this|that)" , re.I)
_NEGATIVE_RE = re.compile(r"(?:没有|无|不存在|找不到|是否有|有没有|哪些没有|未找到|不包含|without|doesn['’]t|not found)" , re.I)
_MULTI_HOP_RE = re.compile(r"(?:比较|对比|分别|同时|以及|并且|和|与|关联|为什么.*如何|how.*and|compare|versus|\band\b)" , re.I)


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    query_type: QueryType
    confidence: float
    variants: tuple[str, ...]
    should_rewrite: bool
    reason: str
    preserved_entities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["variants"] = list(self.variants)
        value["preserved_entities"] = list(self.preserved_entities)
        return value


def preserved_entities(query: str) -> tuple[str, ...]:
    """Return identifiers/numbers that a future rewrite must retain."""
    found = _IDENTIFIER_RE.findall(query or "") + _ACRONYM_RE.findall(query or "") + _NUMBER_RE.findall(query or "")
    seen: set[str] = set()
    result: list[str] = []
    for value in found:
        value = value.strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            result.append(value)
    return tuple(result[:24])


def classify_query(query: str, *, recent_context: str = "") -> tuple[QueryType, float, str]:
    text = " ".join(str(query or "").split())
    if not text:
        return "empty", 1.0, "query_empty"
    entities = preserved_entities(text)
    if entities and (_STRONG_IDENTIFIER_RE.search(text) or len(text) <= 80 and _NUMBER_RE.search(text)):
        return "identifier", 0.94, "identifier_or_exact_token"
    if _FOLLOWUP_RE.search(text) and (recent_context.strip() or len(text) <= 24):
        return "followup", 0.88, "followup_reference_detected"
    if _NEGATIVE_RE.search(text):
        return "negative", 0.82, "negative_or_absence_intent"
    if _MULTI_HOP_RE.search(text) or text.count("?") + text.count("？") > 1:
        return "multi_hop", 0.78, "multiple_entities_or_subquestions"
    return "natural", 0.70, "ordinary_natural_language"


def build_query_plan(
    query: str,
    *,
    recent_context: str = "",
    lexical_coverage: float | None = None,
    rewrite_mode: str = "off",
) -> QueryPlan:
    original = str(query or "").strip()
    query_type, confidence, reason = classify_query(original, recent_context=recent_context)
    coverage = None if lexical_coverage is None else max(0.0, min(1.0, float(lexical_coverage)))
    low_coverage = coverage is not None and coverage < 0.35
    eligible = query_type in {"followup", "natural"} and (query_type == "followup" or low_coverage)
    should_rewrite = rewrite_mode in {"shadow", "on"} and eligible and bool(original)
    if not should_rewrite:
        reason = reason if not eligible else f"{reason};rewrite_disabled_or_not_triggered"
    return QueryPlan(
        original_query=original,
        query_type=query_type,
        confidence=confidence,
        variants=(original,) if original else (),
        should_rewrite=should_rewrite,
        reason=reason,
        preserved_entities=preserved_entities(original),
    )
