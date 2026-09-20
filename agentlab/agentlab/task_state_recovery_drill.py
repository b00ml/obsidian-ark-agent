"""Isolated TaskState recovery drill with a deterministic external ledger.

The drill creates one non-idempotent operation, records the external outcome,
injects a crash window before the local settlement, reopens the checkpoint,
and reconciles from the external ledger.  It never invokes a production tool
and removes its private artifacts before returning.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agentlab.runtime.task_state import TaskStateStore


_RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,47}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _remove_file(path: Path) -> bool:
    if not path.exists():
        return True
    path.unlink()
    return not path.exists()


def _remove_dir(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        path.rmdir()
    except OSError:
        return False
    return True


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _inject_stale_checkpoint(store: TaskStateStore, task_id: str) -> None:
    """Fault-inject a stale lease without adding a production-only API."""
    state = store.get(task_id)
    _require(state is not None, "task state missing before crash injection")
    stale = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    state.pending_tools[0]["updated_at"] = stale
    state.updated_at = stale
    payload = json.dumps(state.to_dict(), ensure_ascii=False, separators=(",", ":"))
    with store._db() as conn:  # controlled drill fault injection
        conn.execute(
            "UPDATE task_state SET state_json=?, updated_at=? WHERE task_id=?",
            (payload, stale, task_id),
        )


def run_recovery_drill(workspace: str | Path, *, run_id: str) -> dict:
    """Exercise crash recovery and external reconciliation in an isolated dir."""
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must be 1-48 lowercase letters, digits, or hyphens")
    root = Path(workspace).resolve()
    if not root.is_dir():
        raise ValueError(f"workspace does not exist: {root}")

    started = time.monotonic()
    drill_dir = root / f".agentlab-task-drill-{run_id}"
    state_path = drill_dir / "task-state.db"
    ledger_path = drill_dir / "external-ledger.json"
    task_id = f"task-{run_id}"
    operation_id = f"operation-{run_id}"
    result: dict = {
        "schema": "task-state-recovery-drill-v1",
        "run_id": run_id,
        "workspace": str(root),
        "task_id": task_id,
        "operation_id": operation_id,
        "stages": {},
        "cleanup": {},
        "passed": False,
    }

    if drill_dir.exists():
        raise ValueError(f"drill artifacts already exist for run_id={run_id}")

    try:
        drill_dir.mkdir(parents=True, exist_ok=False)
        store = TaskStateStore(state_path)
        store.ensure(task_id, session_id=f"session-{run_id}", project_id="drill")
        store.plan_tool(
            task_id,
            operation_id=operation_id,
            tool_name="external_write",
            permission="danger",
            side_effects="external",
            idempotent=False,
        )
        store.update_tool(task_id, operation_id, "running", lease_seconds=1)
        result["stages"]["planned"] = {
            "status": store.get(task_id).pending_tools[0]["status"],
            "replayed": False,
        }

        external = {
            "operation_id": operation_id,
            "status": "succeeded",
            "result_ref": f"external-artifact:{run_id}",
        }
        _write_json(ledger_path, external)
        _inject_stale_checkpoint(store, task_id)

        reopened = TaskStateStore(state_path)
        recovered = reopened.recover_pending_tools(task_id, stale_after_seconds=1)
        _require(len(recovered) == 1, "stale operation was not recovered")
        _require(recovered[0]["status"] == "unknown", "recovery did not create unknown barrier")
        _require(reopened.pending_operations(task_id)[0]["status"] == "unknown",
                 "unknown operation is not pending for verification")
        result["stages"]["recovered"] = {
            "status": "unknown",
            "replayed": False,
            "external_status": external["status"],
        }

        evidence = {
            "status": external["status"],
            "source": "isolated-external-ledger",
            "evidence_ref": f"ledger:{operation_id}",
            "result_ref": external["result_ref"],
        }
        current = reopened.get(task_id)
        _require(current is not None, "task state missing before reconciliation")
        settled = reopened.reconcile_operation(
            task_id,
            operation_id,
            external_result=evidence,
            expected_version=current.state_version,
        )
        row = next(item for item in settled.pending_tools
                   if item["operation_id"] == operation_id)
        _require(row["status"] == "succeeded", "reconciliation did not settle success")
        _require(not reopened.pending_operations(task_id), "settled operation remains pending")

        final = TaskStateStore(state_path).get(task_id)
        _require(final is not None, "final checkpoint missing after reopen")
        final_row = next(item for item in final.pending_tools
                         if item["operation_id"] == operation_id)
        _require(final_row["status"] == "succeeded", "settlement did not survive reopen")
        result["stages"]["reconciled"] = {
            "status": final_row["status"],
            "evidence_ref": final_row["reconciliation"]["evidence_ref"],
            "replayed": False,
        }
        result["passed"] = True
    except (OSError, RuntimeError, ValueError) as exc:
        result["error"] = str(exc)
    finally:
        result["cleanup"]["state_removed"] = all(
            _remove_file(state_path.with_name(state_path.name + suffix))
            for suffix in ("", "-wal", "-shm")
        )
        result["cleanup"]["ledger_removed"] = _remove_file(ledger_path)
        result["cleanup"]["directory_removed"] = _remove_dir(drill_dir)
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["passed"] = bool(
            result["passed"]
            and all(result["cleanup"].values())
            and result["elapsed_seconds"] < 300
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an isolated TaskState crash-recovery drill")
    parser.add_argument("--workspace", help="required workspace for private drill artifacts")
    parser.add_argument("--run-id", default="manual", help="lowercase isolated run id")
    parser.add_argument("--out", help="write the JSON report to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.workspace:
        print(json.dumps({"error": "--workspace is required"}, ensure_ascii=False))
        return 2
    try:
        result = run_recovery_drill(args.workspace, run_id=args.run_id)
    except (OSError, ValueError) as exc:
        result = {"passed": False, "error": str(exc)}
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
