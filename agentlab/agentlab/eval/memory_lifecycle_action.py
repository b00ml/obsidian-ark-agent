"""Hash-bound human lifecycle actions for Markdown long-term memory.

The lifecycle inventory is intentionally read-only.  This separate command is
the only local-CLI write path used by Ark's management modal: it requires the
rendered content hash, exact scope (when supplied), reviewer and reason, then
returns the source and derived-state evidence needed to refresh the UI.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentlab.memory.aggregate import write_memory_aggregate
from agentlab.memory.invalidation import DerivedInvalidationCoordinator
from agentlab.memory.markdown_store import MemoryMarkdownStore
from agentlab.rag.index_store import RagIndexStore


def _canonical_ref(store: MemoryMarkdownStore, memory_id: str) -> str:
    path = store._find_memory_file(memory_id)
    return path.relative_to(store.vault_root).as_posix() if path else ""


def _p2_index(store: MemoryMarkdownStore) -> RagIndexStore | None:
    path = store.vault_root / ".agent-brain" / "rag-index-p2.sqlite"
    return RagIndexStore(path, None, vault_root=store.vault_root) if path.exists() else None


def _aggregate_result(store: MemoryMarkdownStore) -> dict[str, str]:
    try:
        path = write_memory_aggregate(store)
        return {"aggregate": "rebuilt", "aggregate_ref": str(path.relative_to(store.vault_root).as_posix())}
    except (OSError, TypeError, ValueError) as exc:
        return {"aggregate": "failed", "aggregate_error": type(exc).__name__}


def apply_lifecycle_action(
    vault: str | Path,
    *,
    memory_id: str,
    action: str,
    expected_content_hash: str,
    reviewer: str,
    reason: str,
    project_id: str = "",
    session_id: str = "",
    content: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Apply one explicit lifecycle action and return auditable state."""
    root = Path(vault).resolve()
    if not root.is_dir():
        raise ValueError(f"vault does not exist: {root}")
    operation = str(action or "").strip().lower()
    if operation not in {"promote", "correct", "revoke"}:
        raise ValueError("action must be promote, correct or revoke")
    store = MemoryMarkdownStore(str(root), create_dirs=False)
    before = store.lifecycle_target(
        memory_id, expected_content_hash=expected_content_hash,
        project_id=project_id, session_id=session_id,
    )
    index = _p2_index(store)
    if operation == "promote":
        source = store.promote_checked(
            memory_id, expected_content_hash=expected_content_hash,
            reviewer=reviewer, reason=reason, project_id=project_id,
            session_id=session_id,
        )
        derived: dict[str, Any] = {
            "rag_index": "read_gate_updated",
            "session_range": "not_applicable",
            "task_state": "not_applicable",
            "answer_cache": "not_configured",
            "context_cache": "not_configured",
        }
    elif operation == "correct":
        if not str(content or "").strip():
            raise ValueError("correct action requires non-empty content")
        result = DerivedInvalidationCoordinator(store, rag_index=index).correct(
            memory_id, content, tags=tags, reason=reason,
            expected_content_hash=expected_content_hash, reviewer=reviewer,
            project_id=project_id, session_id=session_id,
        )
        if not result.source_updated:
            raise RuntimeError("memory correction did not update source")
        successor = str(result.derived.get("successor") or "")
        source = {
            "id": successor, "action": operation,
            "status": str((store.get(successor) or {}).get("status") or ""),
            "content_hash": str((store.get(successor) or {}).get("content_hash") or ""),
            "correction_of": memory_id,
        }
        derived = {
            **result.derived,
            "invalidation_complete": result.complete,
            "invalidation_warnings": list(result.warnings),
        }
    else:
        result = DerivedInvalidationCoordinator(store, rag_index=index).revoke(
            memory_id, reason=reason, expected_content_hash=expected_content_hash,
            reviewer=reviewer, project_id=project_id, session_id=session_id,
        )
        if not result.source_updated:
            raise RuntimeError("memory revocation did not update source")
        revoked = store.get(memory_id) or {}
        source = {
            "id": memory_id, "action": operation,
            "status": str(revoked.get("status") or ""),
            "content_hash": str(revoked.get("content_hash") or ""),
        }
        derived = {
            **result.derived,
            "invalidation_complete": result.complete,
            "invalidation_warnings": list(result.warnings),
        }
    derived.update(_aggregate_result(store))
    return {
        "schema": "memory-lifecycle-action-v1",
        "vault": str(root),
        "mutated": True,
        "action": operation,
        "before": {
            "id": memory_id,
            "status": str(before.get("status") or "active"),
            "content_hash": str(before.get("content_hash") or ""),
            "source_ref": _canonical_ref(store, memory_id),
        },
        "result": source,
        "derived": derived,
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _tags(value: str) -> list[str] | None:
    if not value.strip():
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("--tags-json must be a JSON string array")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply one hash-bound long-term memory lifecycle action")
    parser.add_argument("--vault", required=True)
    parser.add_argument("--apply", action="store_true", help="required acknowledgement for every write")
    parser.add_argument("--memory-id", default="")
    parser.add_argument("--action", choices=("promote", "correct", "revoke"), default="")
    parser.add_argument("--expected-content-hash", default="")
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--project-id", default="")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--content", default="")
    parser.add_argument("--tags-json", default="")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        if not args.apply:
            raise ValueError("--apply is required for lifecycle actions")
        required = {
            "--memory-id": args.memory_id,
            "--action": args.action,
            "--expected-content-hash": args.expected_content_hash,
            "--reviewer": args.reviewer,
            "--reason": args.reason,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError("--apply requires " + ", ".join(missing))
        report = apply_lifecycle_action(
            args.vault, memory_id=args.memory_id, action=args.action,
            expected_content_hash=args.expected_content_hash, reviewer=args.reviewer,
            reason=args.reason, project_id=args.project_id, session_id=args.session_id,
            content=args.content, tags=_tags(args.tags_json),
        )
    except (OSError, RuntimeError, ValueError, PermissionError, FileNotFoundError,
            json.JSONDecodeError) as exc:
        report = {"schema": "memory-lifecycle-action-v1", "passed": False, "error": str(exc)}
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
