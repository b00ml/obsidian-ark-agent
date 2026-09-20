"""Safe, repeatable lifecycle drill for the incremental P2 RAG index.

The drill writes one temporary note below ``Inbox/.agentlab-drill-<run-id>``
and uses a separate derived SQLite file.  It never reads or writes ``raw/``
and removes all artifacts before returning a report.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from agentlab.rag.index_store import RagIndexStore

_RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,47}$")


def _remove_file(path: Path) -> bool:
    """Remove only a path created by this drill; absent paths are already clean."""
    if not path.exists():
        return True
    path.unlink()
    return not path.exists()


def _remove_empty_dir(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        path.rmdir()
    except OSError:
        return False
    return True


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_incremental_drill(vault: str | Path, *, run_id: str) -> dict:
    """Exercise create, modify and delete against an isolated real Vault prefix."""
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must be 1-48 lowercase letters, digits, or hyphens")
    root = Path(vault).resolve()
    if not root.is_dir():
        raise ValueError(f"vault does not exist: {root}")

    started = time.monotonic()
    relative_dir = Path("Inbox") / f".agentlab-drill-{run_id}"
    source_dir = root / relative_dir
    source_file = source_dir / "incremental-lifecycle.md"
    index_dir = root / ".agent-brain" / "drills"
    index = index_dir / f"rag-update-drill-{run_id}.sqlite"
    result: dict = {
        "schema": "rag-incremental-drill-v1",
        "run_id": run_id,
        "vault": str(root),
        "source_prefix": relative_dir.as_posix(),
        "index": str(index),
        "embedding_enabled": False,
        "stages": {},
        "cleanup": {},
        "passed": False,
    }

    if source_dir.exists() or index.exists():
        raise ValueError(f"drill artifacts already exist for run_id={run_id}")

    initial_token = f"agentlab-drill-{run_id}-initial"
    updated_token = f"agentlab-drill-{run_id}-updated"
    try:
        source_dir.mkdir(parents=True, exist_ok=False)
        index_dir.mkdir(parents=True, exist_ok=True)
        store = RagIndexStore(
            index,
            None,
            vault_root=root,
            include_prefixes=[relative_dir.as_posix()],
        )

        source_file.write_text(f"# RAG incremental drill\n\n{initial_token}\n", encoding="utf-8")
        create_sync = store.sync_vault(root)
        _require(create_sync["updated"] == 1, "create did not update exactly one document")
        _require(bool(store.search_lexical(initial_token)), "created note is not retrievable")
        result["stages"]["create"] = {"sync": create_sync, "retrievable": True}

        source_file.write_text(f"# RAG incremental drill\n\n{updated_token}\n", encoding="utf-8")
        modify_plan = store.plan_changes(root)
        _require(modify_plan["upserts"] == 1, "modified note was not planned for replacement")
        _require(store.index_status(root)["stale_files"] == 1, "stale source was not detected")
        modify_sync = store.sync_vault(root)
        _require(modify_sync["updated"] == 1, "modified note was not replaced")
        _require(bool(store.search_lexical(updated_token)), "updated note is not retrievable")
        _require(not store.search_lexical(initial_token), "old note content remains retrievable")
        result["stages"]["modify"] = {
            "plan": modify_plan,
            "sync": modify_sync,
            "retrievable": True,
            "old_content_absent": True,
        }

        source_file.unlink()
        delete_plan = store.plan_changes(root)
        _require(delete_plan["deletes"] == 1, "deleted note was not planned for cleanup")
        delete_sync = store.sync_vault(root)
        _require(delete_sync["removed"] == 1, "deleted note was not removed from the index")
        _require(not store.search_lexical(updated_token), "deleted note remains retrievable")
        result["stages"]["delete"] = {
            "plan": delete_plan,
            "sync": delete_sync,
            "retrievable": False,
        }
        result["passed"] = True
    except (OSError, RuntimeError, ValueError) as exc:
        result["error"] = str(exc)
    finally:
        result["cleanup"]["source_removed"] = _remove_file(source_file)
        result["cleanup"]["source_directory_removed"] = _remove_empty_dir(source_dir)
        index_removed = all(_remove_file(Path(str(index) + suffix)) for suffix in ("", "-wal", "-shm"))
        result["cleanup"]["index_removed"] = index_removed
        result["cleanup"]["index_directory_empty"] = _remove_empty_dir(index_dir)
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["passed"] = bool(
            result["passed"]
            and all(result["cleanup"].values())
            and result["elapsed_seconds"] < 300
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an isolated P2 incremental-index lifecycle drill")
    parser.add_argument("--vault", help="required Vault root; no config default is used")
    parser.add_argument("--run-id", default="manual", help="lowercase identifier for isolated artifacts")
    parser.add_argument("--out", help="write the JSON report to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.vault:
        print(json.dumps({"error": "--vault is required"}, ensure_ascii=False))
        return 2
    try:
        result = run_incremental_drill(args.vault, run_id=args.run_id)
    except (OSError, ValueError) as exc:
        result = {"passed": False, "error": str(exc)}
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
