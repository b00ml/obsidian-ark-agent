"""Coordinated invalidation for corrected or revoked memory facts.

Markdown remains the source of truth.  This coordinator makes derived cleanup
explicit and auditable: every configured route is invoked, failures are
reported, and callers can keep the production route fail-closed until all
required derivations have been invalidated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass
class InvalidationResult:
    memory_id: str
    operation: str
    source_updated: bool = False
    derived: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.source_updated and not self.warnings and all(
            name == "successor" or value in {"invalidated", "not_configured", "not_applicable"}
            for name, value in self.derived.items()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "operation": self.operation,
            "source_updated": self.source_updated,
            "derived": dict(self.derived),
            "warnings": list(self.warnings),
            "complete": self.complete,
        }


class DerivedInvalidationCoordinator:
    """Invalidate all configured derived routes after a source mutation."""

    def __init__(self, memory_store: Any, *, rag_index: Any | None = None,
                 range_gateway: Any | None = None,
                 session_store: Any | None = None,
                 cache_invalidators: Mapping[str, Callable[..., Any]] | None = None,
                 task_state_invalidator: Callable[..., Any] | None = None):
        self.memory_store = memory_store
        self.rag_index = rag_index
        self.range_gateway = range_gateway
        self.session_store = session_store
        self.cache_invalidators = dict(cache_invalidators or {})
        self.task_state_invalidator = task_state_invalidator

    def _source_ref(self, memory_id: str, before: Mapping[str, Any] | None) -> str:
        # ``source_ref`` is often a business provenance id (for example
        # ``correction:<memory-id>``), while RAG removal requires the actual
        # Vault-relative Markdown path.  Accept a source_ref only when it
        # resolves to a file under the configured Vault; otherwise derive the
        # canonical path from the memory id.
        candidate = str((before or {}).get("source_ref") or "").strip()
        root = Path(getattr(self.memory_store, "vault_root", ".")).resolve()
        if candidate:
            try:
                path = (root / candidate).resolve()
                if path.is_file() and root in path.parents:
                    return path.relative_to(root).as_posix()
            except (OSError, ValueError, TypeError):
                pass
        finder = getattr(self.memory_store, "_find_memory_file", None)
        if finder is None:
            return ""
        try:
            path = finder(memory_id)
            return str(Path(path).resolve().relative_to(root).as_posix()) if path else ""
        except (OSError, ValueError, TypeError):
            return ""

    def _invalidate_derived(self, result: InvalidationResult,
                            before: Mapping[str, Any] | None) -> None:
        source_ref = self._source_ref(result.memory_id, before)
        if self.rag_index is None:
            result.derived["rag_index"] = "not_configured"
        elif source_ref:
            try:
                self.rag_index.remove_document(source_ref)
                result.derived["rag_index"] = "invalidated"
            except Exception as exc:  # noqa: BLE001
                result.derived["rag_index"] = "failed"
                result.warnings.append(f"rag_index:{type(exc).__name__}")
        else:
            result.derived["rag_index"] = "not_applicable"

        session_id = str((before or {}).get("session_id") or
                         (before or {}).get("source_session") or "")
        if self.range_gateway is None or not session_id:
            result.derived["session_range"] = "not_configured" if self.range_gateway is None else "not_applicable"
        else:
            try:
                self.range_gateway.remove_session(session_id)
                result.derived["session_range"] = "invalidated"
            except Exception as exc:  # noqa: BLE001
                result.derived["session_range"] = "failed"
                result.warnings.append(f"session_range:{type(exc).__name__}")

        for name, invalidator in self.cache_invalidators.items():
            try:
                invalidator(memory_id=result.memory_id, source_ref=source_ref)
                result.derived[name] = "invalidated"
            except Exception as exc:  # noqa: BLE001
                result.derived[name] = "failed"
                result.warnings.append(f"{name}:{type(exc).__name__}")
        if self.task_state_invalidator is not None:
            try:
                self.task_state_invalidator(memory_id=result.memory_id, source_ref=source_ref)
                result.derived["task_state"] = "invalidated"
            except Exception as exc:  # noqa: BLE001
                result.derived["task_state"] = "failed"
                result.warnings.append(f"task_state:{type(exc).__name__}")

        # Session history is authoritative and is deliberately not treated as
        # a cache. Keep absent cache routes explicit in the audit.
        result.derived.setdefault("answer_cache", "not_configured")
        result.derived.setdefault("context_cache", "not_configured")

    def revoke(self, memory_id: str, *, reason: str = "user_revoked",
               expected_content_hash: str = "", reviewer: str = "",
               project_id: str = "", session_id: str = "") -> InvalidationResult:
        before = self.memory_store.get(memory_id)
        result = InvalidationResult(memory_id, "revoke")
        if expected_content_hash:
            result.source_updated = bool(self.memory_store.revoke_checked(
                memory_id, expected_content_hash=expected_content_hash,
                reviewer=reviewer, reason=reason, project_id=project_id,
                session_id=session_id,
            ))
        else:
            result.source_updated = bool(self.memory_store.revoke(memory_id, reason=reason))
        if not result.source_updated:
            result.warnings.append("source_not_found")
            return result
        self._invalidate_derived(result, before)
        return result

    def delete(self, memory_id: str, *, reason: str = "user_delete", hard: bool = False) -> InvalidationResult:
        before = self.memory_store.get(memory_id)
        result = InvalidationResult(memory_id, "delete")
        result.source_updated = bool(self.memory_store.delete(memory_id, reason=reason, hard=hard))
        if not result.source_updated:
            result.warnings.append("source_not_found")
            return result
        self._invalidate_derived(result, before)
        return result

    def correct(self, memory_id: str, content: str, *, tags: list[str] | None = None,
                reason: str = "user_correction", expected_content_hash: str = "",
                reviewer: str = "", project_id: str = "",
                session_id: str = "") -> InvalidationResult:
        before = self.memory_store.get(memory_id)
        result = InvalidationResult(memory_id, "correct")
        if expected_content_hash:
            successor = self.memory_store.correct_checked(
                memory_id, content, tags=tags,
                expected_content_hash=expected_content_hash, reviewer=reviewer,
                reason=reason, project_id=project_id, session_id=session_id,
            )
        else:
            successor = self.memory_store.correct(memory_id, content, tags=tags, reason=reason)
        result.source_updated = bool(successor)
        if not result.source_updated:
            result.warnings.append("source_not_found")
            return result
        result.derived["successor"] = str(successor)
        self._invalidate_derived(result, before)
        return result


__all__ = ["InvalidationResult", "DerivedInvalidationCoordinator"]
