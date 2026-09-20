"""Real-Vault review_due confirmation/defer drill.

The drill uses the production Markdown store against an explicitly supplied
Vault root, creates two temporary due memories, applies hash-bound decisions,
and removes every source/audit artifact before returning.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agentlab.eval.memory_review_due import apply_review, list_review_due
from agentlab.memory.markdown_store import MemoryMarkdownStore


_RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,47}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _remove_memory(store: MemoryMarkdownStore, memory_id: str) -> bool:
    path = store._find_memory_file(memory_id)
    if path is None:
        return True
    path.unlink()
    return not path.exists()


def _remove_empty(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        path.rmdir()
    except OSError:
        return False
    return True


def run_memory_review_due_drill(vault: str | Path, *, run_id: str) -> dict:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must be 1-48 lowercase letters, digits, or hyphens")
    root = Path(vault).resolve()
    if not root.is_dir():
        raise ValueError(f"vault does not exist: {root}")
    started = time.monotonic()
    scope = f"review-drill-{run_id}"
    memory_dir = root / "ark" / "memory" / "context" / scope
    audit_path = root / ".agent-brain" / "drills" / f"review-due-{run_id}.jsonl"
    result: dict = {
        "schema": "memory-review-due-drill-v1", "run_id": run_id,
        "vault": str(root), "stages": {}, "cleanup": {}, "passed": False,
    }
    ids: list[str] = []
    try:
        store = MemoryMarkdownStore(root, create_dirs=False, audit_path=audit_path)
        due_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        for text in (f"{run_id} confirm fact", f"{run_id} defer fact"):
            ids.append(store.commit(text, project_id=scope, source="assistant",
                                    review_due_at=due_at))
        queue = list_review_due(root, project_id=scope)
        _require(queue["total"] == 2, "review queue did not contain both due memories")
        result["stages"]["queued"] = {
            "total": queue["total"], "queue_hash": queue["queue_hash"],
            "content_hashes": {item["id"]: item["content_hash"] for item in queue["items"]},
        }

        first = queue["items"][0]
        confirmed = apply_review(
            root, memory_id=first["id"], decision="confirm", reviewer=f"drill:{run_id}",
            expected_content_hash=first["content_hash"], reason="deterministic drill confirmation",
        )
        _require(confirmed["mutated"] and confirmed["result"]["status"] == "active",
                 "confirm decision did not settle")
        result["stages"]["confirmed"] = confirmed["result"]

        remaining = [item for item in queue["items"] if item["id"] != first["id"]][0]
        future = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        deferred = apply_review(
            root, memory_id=remaining["id"], decision="defer", reviewer=f"drill:{run_id}",
            expected_content_hash=remaining["content_hash"], reason="deterministic drill defer",
            defer_until=future,
        )
        _require(deferred["result"]["review_due_at"] == future,
                 "defer decision did not schedule a future review")
        after = list_review_due(root, project_id=scope)
        _require(after["total"] == 0, "settled review items remained in due queue")
        result["stages"]["deferred"] = deferred["result"]
        result["stages"]["settled_queue"] = {"total": after["total"], "mutated": after["mutated"]}
        result["passed"] = True
    except (OSError, RuntimeError, ValueError) as exc:
        result["error"] = str(exc)
    finally:
        cleanup_store = MemoryMarkdownStore(root, create_dirs=False)
        result["cleanup"]["memories_removed"] = all(
            _remove_memory(cleanup_store, memory_id) for memory_id in ids
        )
        result["cleanup"]["memory_dir_removed"] = _remove_empty(memory_dir)
        result["cleanup"]["audit_removed"] = (
            not audit_path.exists() or (audit_path.unlink() is None and not audit_path.exists())
        )
        result["cleanup"]["drill_dir_removed"] = _remove_empty(audit_path.parent)
        agent_brain = audit_path.parent.parent
        result["cleanup"]["drills_parent_removed_or_preserved"] = (
            (not agent_brain.exists()) or any(agent_brain.iterdir()) or _remove_empty(agent_brain)
        )
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["passed"] = bool(result["passed"] and all(result["cleanup"].values())
                                 and result["elapsed_seconds"] < 300)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a real-Vault review_due drill")
    parser.add_argument("--vault", help="required Vault root")
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if not args.vault:
        print(json.dumps({"error": "--vault is required"}, ensure_ascii=False))
        return 2
    try:
        report = run_memory_review_due_drill(args.vault, run_id=args.run_id)
    except (OSError, ValueError) as exc:
        report = {"passed": False, "error": str(exc)}
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
