"""Source-bound citation registry for RAG stage and answer-gate boundaries."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from agentlab.contracts import Citation, CitationStatus, RetrievalResult, RetrievalScope


def _scope(value: RetrievalScope | Mapping[str, Any] | None) -> RetrievalScope:
    return RetrievalScope.from_value(value) if value is not None else RetrievalScope()


@dataclass
class CitationRegistry:
    """Run-local registry; callers may persist ``to_dict`` as a trace artifact."""

    entries: dict[str, Citation] = field(default_factory=dict)

    def register(self, citation: Citation | Mapping[str, Any], *, content: str = "") -> Citation:
        item = citation if isinstance(citation, Citation) else Citation.model_validate(citation)
        if content and not item.source_hash:
            item = item.model_copy(update={
                "source_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            })
        self.entries[item.ref] = item
        return item

    def register_retrieval(self, result: RetrievalResult | Mapping[str, Any]) -> list[Citation]:
        envelope = result if isinstance(result, RetrievalResult) else RetrievalResult.model_validate(result)
        registered: list[Citation] = []
        for item in envelope.items:
            citation = Citation(
                ref=item.ref, source_kind=item.source,
                project_id=item.project_id or envelope.scope.project_id,
                session_id=item.session_id or envelope.scope.session_id,
                status=(CitationStatus.ACTIVE if item.status in {"active", "current", "available"}
                        else CitationStatus.UNVERIFIED),
                metadata={"strategy": envelope.strategy.value, "contract": envelope.contract},
            )
            registered.append(self.register(citation, content=item.content))
        return registered

    def revoke(self, ref: str) -> bool:
        item = self.entries.get(str(ref).strip())
        if item is None:
            return False
        self.entries[item.ref] = item.model_copy(update={"status": CitationStatus.REVOKED})
        return True

    def validate(self, refs: Iterable[str], *, scope: RetrievalScope | Mapping[str, Any] | None = None,
                 allow_unverified: bool = False) -> tuple[list[Citation], list[str]]:
        requested = _scope(scope)
        valid: list[Citation] = []
        reasons: list[str] = []
        for raw in refs:
            ref = str(raw or "").strip()
            citation = self.entries.get(ref)
            if citation is None:
                reasons.append(f"unknown:{ref}")
                continue
            if citation.status == CitationStatus.REVOKED:
                reasons.append(f"revoked:{ref}")
                continue
            if not allow_unverified and citation.status != CitationStatus.ACTIVE:
                reasons.append(f"unverified:{ref}")
                continue
            if requested.project_id and citation.project_id and citation.project_id not in {requested.project_id, "default"}:
                reasons.append(f"scope:{ref}")
                continue
            if requested.session_id and citation.session_id and citation.session_id != requested.session_id:
                reasons.append(f"session:{ref}")
                continue
            valid.append(citation)
        return valid, reasons

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [item.model_dump(mode="json") for item in self.entries.values()]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CitationRegistry":
        registry = cls()
        for item in value.get("entries", []):
            registry.register(item)
        return registry


__all__ = ["CitationRegistry"]
