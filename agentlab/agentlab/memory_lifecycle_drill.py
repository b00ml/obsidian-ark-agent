"""Real-Vault memory invalidation drill with isolated derived artifacts.

The drill exercises the production Markdown store and invalidation coordinator
against a real Vault root while keeping the index, ranges, task checkpoint and
audit log under a private drill directory.  It creates only one temporary
memory record and removes it, its successor, and every derived artifact before
returning a report.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agentlab.core.context_assembler import ContextAssembler
from agentlab.memory.aggregate import build_memory_aggregate, write_memory_aggregate
from agentlab.memory.invalidation import DerivedInvalidationCoordinator
from agentlab.contracts import RetrievalScope
from agentlab.memory.markdown_store import MemoryMarkdownStore
from agentlab.memory.ranges import RangeArchive, RangeGateway
from agentlab.rag.index_store import RagIndexStore
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


def _remove_empty_dir(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        path.rmdir()
    except OSError:
        return False
    return True


def _remove_memory(store: MemoryMarkdownStore, memory_id: str) -> bool:
    path = store._find_memory_file(memory_id)
    if path is None:
        return True
    path.unlink()
    return not path.exists()


class _DrillEmbedder:
    """Deterministic local vectors; no provider or network is used by drills."""

    model = "memory-lifecycle-drill-v1"

    def embed(self, texts):
        vocabulary = ("drill", "active", "expired", "future", "conflict", "corrected")
        return [[float(str(text).lower().count(word)) for word in vocabulary]
                for text in texts]


def run_memory_lifecycle_drill(vault: str | Path, *, run_id: str) -> dict:
    """Run the complete governed-memory lifecycle against a real Vault root.

    The Markdown records live briefly under a unique project scope in the
    supplied Vault.  All acceleration data, ranges, checkpoints, aggregate
    projection and audit output live under the private drill directory.
    """
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id must be 1-48 lowercase letters, digits, or hyphens")
    root = Path(vault).resolve()
    if not root.is_dir():
        raise ValueError(f"vault does not exist: {root}")

    started = time.monotonic()
    drill_dir = root / ".agent-brain" / "drills" / f"memory-lifecycle-{run_id}"
    memory_scope_dir = root / "ark" / "memory" / "context" / f"drill-{run_id}"
    index_path = drill_dir / "memory.sqlite"
    task_path = drill_dir / "task-state.db"
    ranges_root = drill_dir / "ranges"
    audit_path = drill_dir / "events.jsonl"
    session_id = f"memory-drill-{run_id}"
    project_id = f"drill-{run_id}"
    task_id = f"task-memory-{run_id}"
    token = f"memory-lifecycle-drill-{run_id}"
    result: dict = {
        "schema": "memory-lifecycle-drill-v1",
        "run_id": run_id,
        "vault": str(root),
        "index": str(index_path),
        "stages": {},
        "cleanup": {},
        "passed": False,
    }
    old_id = ""
    successor_id = ""
    corrected_id = ""
    memory_ids: list[str] = []
    aggregate_path = drill_dir / "aggregate-v1.json"
    if drill_dir.exists():
        raise ValueError(f"drill artifacts already exist for run_id={run_id}")

    try:
        drill_dir.mkdir(parents=True, exist_ok=False)
        store = MemoryMarkdownStore(root, create_dirs=False, audit_path=audit_path)
        prefix = f"ark/memory/context/{project_id}"
        index = RagIndexStore(
            index_path, _DrillEmbedder(), vault_root=root,
            include_prefixes=[prefix],
        )

        def aggregate() -> dict:
            view = build_memory_aggregate(store, project_id=project_id)
            write_memory_aggregate(store, path=aggregate_path, project_id=project_id)
            return view

        def memory_row(memory_id: str) -> dict:
            row = store.get(memory_id)
            _require(row is not None, f"memory missing: {memory_id}")
            return row

        def assert_hidden(memory_id: str, query: str) -> None:
            _require(memory_id not in {row["id"] for row in store.query(
                query, project_id=project_id, track_access=False)},
                     f"memory route leaked {memory_id}")
            projection = json.dumps(aggregate(), ensure_ascii=False)
            _require(memory_id not in projection, f"aggregate leaked {memory_id}")
            plan = ContextAssembler(budget_tokens=200).assemble(
                scope=RetrievalScope(project_id=project_id, session_id=session_id),
                memory_candidates=[memory_row(memory_id)],
            )
            _require(memory_id not in {item.id for item in plan.selected},
                     f"context assembler leaked {memory_id}")
            refs = [str(item.get("ref") or "") for item in index.search_lexical(query, k=20)]
            _require(memory_id not in "\n".join(refs), f"lexical RAG leaked {memory_id}")
            vector_refs = [str(item.get("ref") or "") for item in index.search_vector(query, k=20)]
            _require(memory_id not in "\n".join(vector_refs), f"vector RAG leaked {memory_id}")

        candidate_id = store.commit(
            f"{token} candidate preference", project_id=project_id,
            source_session=session_id, source_ref=f"drill:{run_id}:candidate",
            source="assistant", candidate_first=True,
        )
        memory_ids.append(candidate_id)
        candidate = memory_row(candidate_id)
        _require(candidate["status"] == "candidate", "candidate status was not persisted")
        assert_hidden(candidate_id, "candidate preference")
        result["stages"]["candidate"] = {"memory_id": candidate_id, "status": candidate["status"]}

        promoted = store.promote_checked(
            candidate_id, expected_content_hash=candidate["content_hash"],
            reviewer="drill", reason="lifecycle drill", project_id=project_id,
            session_id=session_id,
        )
        index.sync_vault(root)
        promoted_row = memory_row(candidate_id)
        _require(promoted_row["status"] == "active", "candidate did not promote")
        _require(index.search_lexical("candidate preference", k=20), "promoted memory not indexed")
        _require(index.search_vector("drill", k=20), "promoted memory not vector indexed")
        _require(candidate_id in json.dumps(aggregate(), ensure_ascii=False), "aggregate missed promoted memory")
        result["stages"]["promote"] = {**promoted, "indexed": True, "aggregate": "rebuilt"}

        expired_id = store.commit(
            f"{token} expired fact", project_id=project_id,
            valid_until=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            explicit_confirmation=True,
        )
        future_id = store.commit(
            f"{token} future fact", project_id=project_id,
            valid_from=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            explicit_confirmation=True,
        )
        due_id = store.commit(
            f"{token} review due fact", project_id=project_id,
            review_due_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            explicit_confirmation=True,
        )
        memory_ids.extend([expired_id, future_id, due_id])
        index.sync_vault(root)
        lifecycle_rows = {row["id"]: row for row in store.lifecycle(project_id=project_id)}
        _require(lifecycle_rows[expired_id]["status"] == "expired", "expiry status not visible")
        _require(lifecycle_rows[future_id]["status"] == "not_yet_valid", "future status not visible")
        _require(lifecycle_rows[due_id]["status"] == "review_due", "review_due status not visible")
        for memory_id, query in ((expired_id, "expired fact"), (future_id, "future fact"),
                                 (due_id, "review due fact")):
            assert_hidden(memory_id, query)
        result["stages"]["time_gates"] = {
            "expired": expired_id, "future": future_id, "review_due": due_id,
            "all_hidden": True,
        }

        conflict_base = store.commit(
            f"{token} conflict old", project_id=project_id,
            subject="drill-storage", mem_type="context", explicit_confirmation=True,
        )
        conflict_id = store.commit(
            f"{token} conflict new", project_id=project_id,
            subject="drill-storage", mem_type="context", explicit_confirmation=True,
        )
        memory_ids.extend([conflict_base, conflict_id])
        index.sync_vault(root)
        _require(memory_row(conflict_id)["status"] == "conflict", "conflict status not persisted")
        _require(conflict_id in {row["id"] for row in store.list_conflicts(project_id=project_id)},
                 "conflict missing from review list")
        assert_hidden(conflict_id, "conflict new")
        result["stages"]["conflict"] = {
            "memory_id": conflict_id, "conflicts_with": [conflict_base], "hidden": True,
        }

        old_id = store.commit(
            f"{token} old fact", project_id=project_id,
            source_session=session_id, source_ref=f"drill:{run_id}:old",
            explicit_confirmation=True,
        )
        memory_ids.append(old_id)
        old_path = store._find_memory_file(old_id)
        _require(old_path is not None, "memory source file was not created")
        source_ref = old_path.resolve().relative_to(root).as_posix()
        index.sync_vault(root)
        _require(index.search_lexical("old fact", k=20), "memory was not retrievable")

        archive = RangeArchive(ranges_root)
        gateway = RangeGateway(archive)
        from agentlab.core.message import Message
        archive.append(session_id, [Message(role="user", content=token)])
        task_store = TaskStateStore(task_path)
        task_store.ensure(task_id, session_id=session_id, project_id=project_id)
        task_store.patch(task_id, {
            "core_intent": {"memory_id": old_id, "source_ref": source_ref},
            "completed_steps": [{"evidence_ref": f"memory:{old_id}"}],
        })
        coordinator = DerivedInvalidationCoordinator(
            store, rag_index=index, range_gateway=gateway,
            task_state_invalidator=lambda **kwargs: task_store.invalidate_memory(
                task_id, memory_id=kwargs["memory_id"],
                source_ref=kwargs.get("source_ref", ""),
            ),
        )
        revoked = coordinator.revoke(old_id, reason="memory lifecycle drill")
        _require(revoked.complete, "revoke invalidation was incomplete")
        _require(not any(str(row.get("ref") or "").startswith(source_ref)
                         for row in index.search_lexical("old fact", k=20)),
                 "revoked memory remained in RAG")
        _require(not archive.path(session_id).exists(), "session range was not removed")
        checkpoint = task_store.get(task_id)
        _require(
            checkpoint is not None
            and old_id not in str({
                "core_intent": checkpoint.core_intent,
                "todo": checkpoint.todo,
                "completed_steps": checkpoint.completed_steps,
            }),
                 "task checkpoint retained revoked memory")
        result["stages"]["revoke"] = revoked.to_dict()
        assert_hidden(old_id, "old fact")

        successor_id = store.commit(
            f"{token} correction source", project_id=project_id,
            source_session=session_id, source_ref=f"drill:{run_id}:correct",
            explicit_confirmation=True,
        )
        memory_ids.append(successor_id)
        successor_path = store._find_memory_file(successor_id)
        _require(successor_path is not None, "successor source file was not created")
        successor_ref = successor_path.resolve().relative_to(root).as_posix()
        index = RagIndexStore(index_path, _DrillEmbedder(), vault_root=root,
                              include_prefixes=[prefix])
        index.sync_vault(root)
        archive.append(session_id, [Message(role="user", content=f"{token} correction range")])
        task_store.patch(task_id, {
            "core_intent": {"memory_id": successor_id, "source_ref": successor_ref},
            "completed_steps": [{"evidence_ref": f"memory:{successor_id}"}],
        })
        corrected = DerivedInvalidationCoordinator(
            store, rag_index=index, range_gateway=gateway,
            task_state_invalidator=lambda **kwargs: task_store.invalidate_memory(
                task_id, memory_id=kwargs["memory_id"], source_ref=kwargs.get("source_ref", ""),
            ),
        ).correct(
            successor_id, f"{token} corrected fact", reason="memory lifecycle drill",
        )
        _require(corrected.complete, "correct invalidation was incomplete")
        corrected_id = str(corrected.derived.get("successor") or "")
        memory_ids.append(corrected_id)
        _require(bool(corrected_id) and store.get(corrected_id) is not None,
                 "correction successor was not created")
        _require(store.get(successor_id)["status"] == "superseded", "corrected source remained active")
        _require(not archive.path(session_id).exists(), "correction did not clear session range")
        corrected_checkpoint = task_store.get(task_id)
        _require(corrected_checkpoint is not None and successor_id not in str({
            "core_intent": corrected_checkpoint.core_intent,
            "completed_steps": corrected_checkpoint.completed_steps,
        }), "correction checkpoint retained old memory")
        index.sync_vault(root)
        assert_hidden(successor_id, "correction source")
        _require(index.search_lexical("corrected fact", k=20), "corrected successor not indexed")
        _require(index.search_vector("drill", k=20), "corrected successor not vector indexed")
        result["stages"]["correct"] = corrected.to_dict()
        result["stages"]["cross_route"] = {
            "aggregate_rebuilt": aggregate_path.exists(),
            "context_and_rag_gates": True,
            "source_ref_preserved": bool(source_ref),
        }
        result["passed"] = True
    except (OSError, RuntimeError, ValueError) as exc:
        result["error"] = str(exc)
    finally:
        cleanup_store = MemoryMarkdownStore(root, create_dirs=False)
        result["cleanup"]["all_memory_records_removed"] = all(
            _remove_memory(cleanup_store, memory_id) for memory_id in dict.fromkeys(memory_ids)
        )
        result["cleanup"]["aggregate_removed"] = _remove_file(aggregate_path)
        result["cleanup"]["successor_removed"] = _remove_memory(
            cleanup_store, successor_id) if successor_id else True
        result["cleanup"]["corrected_removed"] = _remove_memory(
            cleanup_store, corrected_id) if corrected_id else True
        result["cleanup"]["memory_scope_dir_removed"] = _remove_empty_dir(memory_scope_dir)
        result["cleanup"]["index_removed"] = all(
            _remove_file(Path(str(index_path) + suffix))
            for suffix in ("", "-wal", "-shm")
        )
        result["cleanup"]["task_removed"] = all(
            _remove_file(Path(str(task_path) + suffix))
            for suffix in ("", "-wal", "-shm")
        )
        result["cleanup"]["ranges_removed"] = (
            (not ranges_root.exists() or all(_remove_file(path) for path in ranges_root.glob("*")))
            and _remove_empty_dir(ranges_root)
        )
        result["cleanup"]["audit_removed"] = _remove_file(audit_path)
        result["cleanup"]["drill_dir_removed"] = _remove_empty_dir(drill_dir)
        result["cleanup"]["drills_parent_removed"] = _remove_empty_dir(drill_dir.parent)
        agent_brain = drill_dir.parent.parent
        result["cleanup"]["agent_brain_parent_preserved_or_empty"] = (
            (not agent_brain.exists())
            or any(agent_brain.iterdir())
            or _remove_empty_dir(agent_brain)
        )
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["passed"] = bool(
            result["passed"] and all(result["cleanup"].values())
            and result["elapsed_seconds"] < 300
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a real-Vault memory invalidation drill")
    parser.add_argument("--vault", help="required Vault root")
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--out")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.vault:
        print(json.dumps({"error": "--vault is required"}, ensure_ascii=False))
        return 2
    try:
        report = run_memory_lifecycle_drill(args.vault, run_id=args.run_id)
    except (OSError, ValueError) as exc:
        report = {"passed": False, "error": str(exc)}
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
