"""Read-only operational listing for memories whose review date has passed."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentlab.memory.markdown_store import MemoryMarkdownStore


def list_review_due(vault: str | Path, *, project_id: str = "",
                    session_id: str = "", limit: int = 100) -> dict[str, Any]:
    root = Path(vault).resolve()
    if not root.is_dir():
        raise ValueError(f"vault does not exist: {root}")
    store = MemoryMarkdownStore(str(root), create_dirs=False)
    rows = store.review_due(project_id=project_id or None,
                            session_id=session_id or None, limit=limit)
    queue_payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "memory-review-due-v2",
        "vault": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_id": project_id,
        "session_id": session_id,
        "total": len(rows),
        "queue_hash": hashlib.sha256(queue_payload.encode("utf-8")).hexdigest(),
        "mutated": False,
        "items": rows,
    }


def apply_review(vault: str | Path, *, memory_id: str, decision: str,
                 reviewer: str, expected_content_hash: str, reason: str,
                 defer_until: str = "") -> dict[str, Any]:
    root = Path(vault).resolve()
    if not root.is_dir():
        raise ValueError(f"vault does not exist: {root}")
    store = MemoryMarkdownStore(str(root), create_dirs=False)
    result = store.review(
        memory_id, decision=decision, reviewer=reviewer,
        expected_content_hash=expected_content_hash, reason=reason,
        defer_until=defer_until or None,
    )
    return {
        "schema": "memory-review-action-v1",
        "vault": str(root),
        "mutated": True,
        "result": result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List or explicitly review memories due for review")
    parser.add_argument("--vault", required=True)
    parser.add_argument("--project-id", default="")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--apply", action="store_true",
                        help="apply one explicit review decision; without this flag the command is read-only")
    parser.add_argument("--memory-id", default="")
    parser.add_argument("--decision", choices=("confirm", "defer"), default="")
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--expected-content-hash", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--defer-until", default="")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        if args.apply:
            required = {
                "--memory-id": args.memory_id,
                "--decision": args.decision,
                "--reviewer": args.reviewer,
                "--expected-content-hash": args.expected_content_hash,
                "--reason": args.reason,
            }
            missing = [name for name, value in required.items() if not str(value).strip()]
            if missing:
                raise ValueError("--apply requires " + ", ".join(missing))
            report = apply_review(
                args.vault, memory_id=args.memory_id, decision=args.decision,
                reviewer=args.reviewer, expected_content_hash=args.expected_content_hash,
                reason=args.reason, defer_until=args.defer_until,
            )
        else:
            report = list_review_due(
                args.vault, project_id=args.project_id,
                session_id=args.session_id, limit=args.limit,
            )
    except (OSError, ValueError) as exc:
        report = {"schema": "memory-review-due-v2", "passed": False, "error": str(exc)}
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
