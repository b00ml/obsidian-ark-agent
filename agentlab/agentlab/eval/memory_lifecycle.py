"""Read-only lifecycle inventory for governed Markdown memories."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from agentlab.memory.markdown_store import MemoryMarkdownStore


def list_lifecycle(vault: str | Path, *, project_id: str = "", session_id: str = "", limit: int = 500) -> dict:
    root = Path(vault).resolve()
    if not root.is_dir():
        raise ValueError(f"vault does not exist: {root}")
    rows = MemoryMarkdownStore(str(root), create_dirs=False).lifecycle(
        project_id=project_id or None, session_id=session_id or None, limit=limit)
    return {"schema": "memory-lifecycle-v1", "vault": str(root),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mutated": False, "total": len(rows), "items": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List governed memory lifecycle metadata")
    parser.add_argument("--vault", required=True)
    parser.add_argument("--project-id", default="")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args(argv)
    try:
        report = list_lifecycle(args.vault, project_id=args.project_id, session_id=args.session_id, limit=args.limit)
    except (OSError, ValueError) as exc:
        report = {"schema": "memory-lifecycle-v1", "passed": False, "error": str(exc)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
