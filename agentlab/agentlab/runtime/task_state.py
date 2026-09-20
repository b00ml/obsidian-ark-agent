"""Durable, validated task checkpoints for the P1 shadow/runtime path.

``TaskStore`` remains the delivery/lease state machine and
``JsonlSessionStorage`` remains the conversation source of truth.  This module
stores only structured intent, progress and pending side effects keyed by the
same ``task_id``; optimistic versions make stale recovery writes fail closed.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping


TASK_PHASES = frozenset({
    "IDLE", "PLANNING", "EXECUTING", "WAITING_USER", "CHECKING", "DONE", "ERROR", "CANCELLED",
})
TOOL_OPERATION_STATUSES = frozenset({
    "planned", "running", "succeeded", "failed", "cancelled", "unknown",
})
_TOOL_OPERATION_TRANSITIONS: dict[str, frozenset[str]] = {
    "planned": frozenset({"running", "failed", "cancelled", "unknown"}),
    "running": frozenset({"succeeded", "failed", "cancelled", "unknown"}),
    # Unknown is a recovery barrier.  Only reconcile_operation may settle it
    # after an external status lookup supplies durable evidence.
    "unknown": frozenset(),
    "succeeded": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}
_TRANSITIONS: dict[str, frozenset[str]] = {
    "IDLE": frozenset({"PLANNING", "CANCELLED"}),
    "PLANNING": frozenset({"EXECUTING", "WAITING_USER", "ERROR", "CANCELLED"}),
    "EXECUTING": frozenset({"WAITING_USER", "CHECKING", "DONE", "ERROR", "CANCELLED"}),
    "WAITING_USER": frozenset({"PLANNING", "EXECUTING", "CANCELLED"}),
    "CHECKING": frozenset({"DONE", "EXECUTING", "ERROR", "CANCELLED"}),
    "DONE": frozenset({"PLANNING"}),
    "ERROR": frozenset({"PLANNING", "EXECUTING", "CANCELLED"}),
    "CANCELLED": frozenset(),
}


class TaskStateError(ValueError):
    """Invalid state or optimistic-lock operation."""


class TaskStateConflict(TaskStateError):
    """Raised when a checkpoint was changed by another worker."""


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _clean_list(value: Any) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TaskStateError("task state list field must be a list")
    return list(value)


def _clean_pending_tools(value: Any) -> list[dict[str, Any]]:
    rows = _clean_list(value)
    cleaned: list[dict[str, Any]] = []
    for item in rows[-100:]:
        if not isinstance(item, dict):
            raise TaskStateError("pending_tools entries must be objects")
        operation_id = str(item.get("operation_id") or "").strip()
        status = str(item.get("status") or "planned").strip().lower()
        if not operation_id or len(operation_id) > 256:
            raise TaskStateError("pending tool operation_id must be non-empty and <=256 chars")
        if status not in TOOL_OPERATION_STATUSES:
            raise TaskStateError(f"unknown pending tool status: {status}")
        row = dict(item)
        row["operation_id"] = operation_id
        row["status"] = status
        verification = row.get("verification")
        if verification is not None:
            if not isinstance(verification, dict):
                raise TaskStateError("pending tool verification must be an object")
            # Verification data is deliberately small and non-secret.  The
            # Runner only writes tool kind, safe reference and content hashes.
            clean_verification = {}
            for key, raw in verification.items():
                if str(key) not in {"kind", "ref", "content_hash", "old_hash",
                                    "new_hash", "result_hash", "project_id", "source_ref",
                                    "decision", "defer_until", "operation_id",
                                    "remote_operation_id"}:
                    raise TaskStateError(f"unknown verification field: {key}")
                if raw is None:
                    continue
                value_text = str(raw).strip()
                if len(value_text) > 512:
                    raise TaskStateError("verification field is too long")
                clean_verification[str(key)] = value_text
            row["verification"] = clean_verification
        cleaned.append(row)
    return cleaned


@dataclass
class TaskState:
    task_id: str
    session_id: str = ""
    project_id: str = ""
    state_version: int = 0
    workflow_version: str = "1"
    phase: str = "IDLE"
    core_intent: dict[str, Any] = field(default_factory=dict)
    current_subtask: str = ""
    todo: list[dict[str, Any]] = field(default_factory=list)
    pending_tools: list[dict[str, Any]] = field(default_factory=list)
    completed_steps: list[dict[str, Any]] = field(default_factory=list)
    context_snapshot: dict[str, Any] = field(default_factory=dict)
    last_error: dict[str, Any] | None = None
    updated_at: str = ""

    def __post_init__(self) -> None:
        self.task_id = str(self.task_id or "").strip()
        if not self.task_id or len(self.task_id) > 128:
            raise TaskStateError("task_id must be a non-empty value <=128 chars")
        self.session_id = str(self.session_id or "").strip()
        self.project_id = str(self.project_id or "").strip()
        self.phase = str(self.phase or "IDLE").strip().upper()
        if self.phase not in TASK_PHASES:
            raise TaskStateError(f"unknown task phase: {self.phase}")
        if not isinstance(self.core_intent, dict):
            raise TaskStateError("core_intent must be an object")
        self.todo = _clean_list(self.todo)
        self.pending_tools = _clean_pending_tools(self.pending_tools)
        self.completed_steps = _clean_list(self.completed_steps)
        if not isinstance(self.context_snapshot, dict):
            raise TaskStateError("context_snapshot must be an object")
        if self.last_error is not None and not isinstance(self.last_error, dict):
            raise TaskStateError("last_error must be an object or null")
        self.state_version = max(0, int(self.state_version))
        self.updated_at = self.updated_at or _stamp()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "state_version": self.state_version,
            "workflow_version": self.workflow_version,
            "phase": self.phase,
            "core_intent": self.core_intent,
            "current_subtask": self.current_subtask,
            "todo": self.todo,
            "pending_tools": self.pending_tools,
            "completed_steps": self.completed_steps,
            "context_snapshot": self.context_snapshot,
            "last_error": self.last_error,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskState":
        return cls(**{key: value[key] for key in (
            "task_id", "session_id", "project_id", "state_version", "workflow_version",
            "phase", "core_intent", "current_subtask", "todo", "pending_tools",
            "completed_steps", "context_snapshot", "last_error", "updated_at",
        ) if key in value})


class TaskStateStore:
    """SQLite checkpoint store with append-only transition events."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @contextmanager
    def _db(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._db() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS task_state (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL DEFAULT '',
                    project_id TEXT NOT NULL DEFAULT '',
                    state_version INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_state_event (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    from_phase TEXT NOT NULL,
                    to_phase TEXT NOT NULL,
                    state_version INTEGER NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS task_state_event_task
                    ON task_state_event(task_id, id);
                """
            )

    def get(self, task_id: str) -> TaskState | None:
        with self._db() as conn:
            row = conn.execute("SELECT state_json FROM task_state WHERE task_id=?",
                               (str(task_id),)).fetchone()
        if row is None:
            return None
        try:
            return TaskState.from_dict(json.loads(row["state_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TaskStateError(f"invalid persisted task state: {task_id}") from exc

    def ensure(self, task_id: str, *, session_id: str = "", project_id: str = "",
               core_intent: Mapping[str, Any] | None = None) -> TaskState:
        initial = TaskState(
            task_id=str(task_id), session_id=session_id, project_id=project_id,
            core_intent=dict(core_intent or {}),
        )
        with self._db() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO task_state
                   (task_id,session_id,project_id,state_version,state_json,updated_at)
                   VALUES(?,?,?,?,?,?)""",
                (initial.task_id, initial.session_id, initial.project_id,
                 initial.state_version, _json(initial.to_dict()), initial.updated_at),
            )
            row = conn.execute("SELECT state_json FROM task_state WHERE task_id=?",
                               (initial.task_id,)).fetchone()
        existing = TaskState.from_dict(json.loads(row["state_json"]))
        # A task id is the recovery key.  Reusing it from another project or
        # session would silently mix context, so require an explicit new id.
        if (initial.project_id and existing.project_id
                and initial.project_id != existing.project_id):
            raise TaskStateConflict(
                f"task {initial.task_id} belongs to project {existing.project_id}")
        if (initial.session_id and existing.session_id
                and initial.session_id != existing.session_id):
            raise TaskStateConflict(
                f"task {initial.task_id} belongs to session {existing.session_id}")
        return existing

    def save(self, state: TaskState, *, expected_version: int | None = None,
             reason: str = "checkpoint") -> TaskState:
        if not isinstance(state, TaskState):
            state = TaskState.from_dict(state)
        now = _stamp()
        with self._db() as conn:
            row = conn.execute("SELECT state_json,state_version FROM task_state WHERE task_id=?",
                               (state.task_id,)).fetchone()
            current_version = None if row is None else int(row["state_version"])
            if row is not None:
                try:
                    current_state = TaskState.from_dict(json.loads(row["state_json"]))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise TaskStateError(
                        f"invalid persisted task state: {state.task_id}") from exc
                if (state.phase != current_state.phase
                        and state.phase not in _TRANSITIONS[current_state.phase]):
                    raise TaskStateError(
                        f"illegal transition {current_state.phase}->{state.phase}")
            if expected_version is not None and current_version != int(expected_version):
                raise TaskStateConflict(
                    f"stale task state {state.task_id}: expected {expected_version}, "
                    f"actual {current_version}")
            next_version = 0 if current_version is None else current_version + 1
            state.state_version = next_version
            state.updated_at = now
            payload = _json(state.to_dict())
            if row is None:
                conn.execute(
                    """INSERT INTO task_state
                       (task_id,session_id,project_id,state_version,state_json,updated_at)
                       VALUES(?,?,?,?,?,?)""",
                    (state.task_id, state.session_id, state.project_id,
                     next_version, payload, now),
                )
            else:
                changed = conn.execute(
                    """UPDATE task_state SET session_id=?,project_id=?,state_version=?,
                       state_json=?,updated_at=? WHERE task_id=? AND state_version=?""",
                    (state.session_id, state.project_id, next_version, payload, now,
                     state.task_id, current_version),
                ).rowcount
                if changed != 1:
                    raise TaskStateConflict(f"task state changed while saving: {state.task_id}")
        return state

    def transition(self, task_id: str, phase: str, *, expected_version: int | None = None,
                   reason: str = "") -> TaskState:
        state = self.get(task_id)
        if state is None:
            state = self.ensure(task_id)
        target = str(phase or "").strip().upper()
        if target not in TASK_PHASES:
            raise TaskStateError(f"unknown task phase: {target}")
        if target != state.phase and target not in _TRANSITIONS[state.phase]:
            raise TaskStateError(f"illegal transition {state.phase}->{target}")
        previous = state.phase
        state.phase = target
        saved = self.save(state, expected_version=(
            state.state_version if expected_version is None else expected_version), reason=reason)
        if previous != target:
            with self._db() as conn:
                conn.execute(
                    """INSERT INTO task_state_event
                       (task_id,from_phase,to_phase,state_version,reason,created_at)
                       VALUES(?,?,?,?,?,?)""",
                    (saved.task_id, previous, target, saved.state_version, str(reason)[:200], _stamp()),
                )
        return saved

    def patch(self, task_id: str, changes: Mapping[str, Any], *,
              expected_version: int | None = None, reason: str = "checkpoint") -> TaskState:
        state = self.get(task_id)
        if state is None:
            state = self.ensure(task_id)
        allowed = {
            "session_id", "project_id", "workflow_version", "phase", "core_intent",
            "current_subtask", "todo", "pending_tools", "completed_steps",
            "context_snapshot", "last_error",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise TaskStateError(f"unknown task state fields: {sorted(unknown)}")
        old_phase = state.phase
        for key, value in changes.items():
            setattr(state, key, value)
        state.__post_init__()
        if state.phase != old_phase:
            if state.phase not in _TRANSITIONS[old_phase]:
                raise TaskStateError(f"illegal transition {old_phase}->{state.phase}")
        saved = self.save(state, expected_version=(
            state.state_version if expected_version is None else expected_version), reason=reason)
        if saved.phase != old_phase:
            with self._db() as conn:
                conn.execute(
                    """INSERT INTO task_state_event
                       (task_id,from_phase,to_phase,state_version,reason,created_at)
                       VALUES(?,?,?,?,?,?)""",
                    (saved.task_id, old_phase, saved.phase, saved.state_version,
                     str(reason)[:200], _stamp()),
                )
        return saved

    def plan_tool(self, task_id: str, *, operation_id: str, tool_name: str,
                  arguments_hash: str = "", permission: str = "read",
                  side_effects: str = "", idempotent: bool | None = None,
                  lease_seconds: float = 300.0,
                  verification: Mapping[str, Any] | None = None) -> TaskState:
        """Record a tool operation before dispatching it.

        A durable ``planned`` row makes a process crash visible.  Reusing an
        existing operation id is idempotent and returns the current checkpoint;
        recovery code can then decide whether an unknown side effect needs
        confirmation instead of blindly replaying it.
        """
        state = self.get(task_id) or self.ensure(task_id)
        op_id = str(operation_id or "").strip()
        if not op_id:
            raise TaskStateError("operation_id is required")
        existing = next((row for row in state.pending_tools
                         if row.get("operation_id") == op_id), None)
        if existing is not None:
            return state
        now = _stamp()
        entry = {
            "operation_id": op_id[:256],
            "tool_name": str(tool_name or "")[:128],
            "arguments_hash": str(arguments_hash or "")[:128],
            "permission": str(permission or "read")[:32],
            "side_effects": str(side_effects or "")[:64],
            "idempotent": idempotent,
            "status": "planned",
            "created_at": now,
            "updated_at": now,
            "lease_until": "",
        }
        if verification is not None:
            # Run through the same schema validation used for persisted rows.
            checked = _clean_pending_tools([{"operation_id": op_id,
                                             "status": "planned",
                                             "verification": dict(verification)}])[0]
            entry["verification"] = checked.get("verification", {})
        state.pending_tools.append(entry)
        return self.save(state, expected_version=state.state_version, reason="tool planned")

    def update_tool(self, task_id: str, operation_id: str, status: str, *,
                    result_ref: str = "", error: str = "",
                    lease_seconds: float = 300.0) -> TaskState:
        """Advance a dispatched operation without bypassing recovery barriers.

        In particular, callers cannot use this general lifecycle method to
        settle an ``unknown`` operation.  That requires
        :meth:`reconcile_operation` and its external evidence payload.
        """
        state = self.get(task_id)
        if state is None:
            raise TaskStateError(f"unknown task state: {task_id}")
        target = str(status or "").strip().lower()
        if target not in TOOL_OPERATION_STATUSES:
            raise TaskStateError(f"unknown pending tool status: {target}")
        found = next((row for row in state.pending_tools
                      if row.get("operation_id") == str(operation_id)), None)
        if found is None:
            raise TaskStateError(f"unknown tool operation: {operation_id}")
        current = str(found.get("status") or "planned").strip().lower()
        if target not in _TOOL_OPERATION_TRANSITIONS.get(current, frozenset()):
            raise TaskStateError(
                f"illegal tool operation transition {current}->{target}; "
                "unknown operations require reconcile_operation"
            )
        now = datetime.now(timezone.utc)
        found["status"] = target
        found["updated_at"] = now.isoformat(timespec="milliseconds")
        if target == "running":
            found["lease_until"] = (now + timedelta(
                seconds=max(1.0, float(lease_seconds)))).isoformat(timespec="milliseconds")
        else:
            found["lease_until"] = ""
        if result_ref:
            found["result_ref"] = str(result_ref)[:256]
        if error:
            found["error"] = str(error)[:200]
        return self.save(state, expected_version=state.state_version, reason=f"tool {target}")

    def reconcile_operation(self, task_id: str, operation_id: str, *,
                            external_result: Mapping[str, Any],
                            expected_version: int | None = None) -> TaskState:
        """Settle one ``unknown`` operation from an external verification result.

        This is deliberately a ledger-only API: it never invokes a tool or
        schedules a retry.  ``source`` and ``evidence_ref`` make the caller's
        external lookup auditable, while the narrow result schema prevents a
        checkpoint from silently becoming a second source of truth.
        """
        if not isinstance(external_result, Mapping):
            raise TaskStateError("external_result must be an object")
        allowed = {"status", "source", "evidence_ref", "result_ref", "error_code"}
        unknown = set(external_result) - allowed
        if unknown:
            raise TaskStateError(f"unknown external result fields: {sorted(unknown)}")
        outcome = str(external_result.get("status") or "").strip().lower()
        if outcome not in {"succeeded", "failed", "cancelled"}:
            raise TaskStateError(
                "external_result.status must be succeeded, failed, or cancelled"
            )
        source = str(external_result.get("source") or "").strip()
        evidence_ref = str(external_result.get("evidence_ref") or "").strip()
        if not source or len(source) > 128:
            raise TaskStateError("external_result.source must be non-empty and <=128 chars")
        if not evidence_ref or len(evidence_ref) > 256:
            raise TaskStateError("external_result.evidence_ref must be non-empty and <=256 chars")

        state = self.get(task_id)
        if state is None:
            raise TaskStateError(f"unknown task state: {task_id}")
        found = next((row for row in state.pending_tools
                      if row.get("operation_id") == str(operation_id)), None)
        if found is None:
            raise TaskStateError(f"unknown tool operation: {operation_id}")
        if found.get("status") != "unknown":
            raise TaskStateError(
                "only unknown tool operations can be reconciled from external evidence"
            )

        now = _stamp()
        reconciliation = {
            "status": outcome,
            "source": source,
            "evidence_ref": evidence_ref,
            "reconciled_at": now,
        }
        result_ref = str(external_result.get("result_ref") or "").strip()
        error_code = str(external_result.get("error_code") or "").strip()
        if result_ref:
            reconciliation["result_ref"] = result_ref[:256]
            found["result_ref"] = result_ref[:256]
        if error_code:
            reconciliation["error_code"] = error_code[:128]
            found["error_code"] = error_code[:128]
        found["status"] = outcome
        found["lease_until"] = ""
        found["updated_at"] = now
        found["reconciliation"] = reconciliation
        return self.save(
            state,
            expected_version=(state.state_version if expected_version is None else expected_version),
            reason=f"tool reconciled {outcome}",
        )

    def pending_operations(self, task_id: str) -> list[dict[str, Any]]:
        state = self.get(task_id)
        if state is None:
            return []
        return [dict(row) for row in state.pending_tools
                if row.get("status") in {"planned", "running", "unknown"}]

    def invalidate_memory(self, task_id: str, *, memory_id: str = "",
                          source_ref: str = "") -> bool:
        """Remove stale memory references from the task checkpoint.

        Session history remains authoritative and is intentionally untouched.
        Only structured task fields are scrubbed so a later context assembly
        cannot re-inject a corrected/revoked memory ref.
        """
        state = self.get(task_id)
        if state is None:
            return False
        needles = {str(value).strip() for value in (memory_id, source_ref) if str(value).strip()}
        if not needles:
            return False

        def scrub(value: Any) -> tuple[Any, bool]:
            if isinstance(value, str):
                if any(needle in value for needle in needles):
                    return "", True
                return value, False
            if isinstance(value, list):
                out = []
                changed = False
                for item in value:
                    cleaned, item_changed = scrub(item)
                    if isinstance(cleaned, str) and not cleaned:
                        changed = True
                        continue
                    out.append(cleaned)
                    changed = changed or item_changed
                return out, changed
            if isinstance(value, dict):
                out = {}
                changed = False
                for key, item in value.items():
                    cleaned, item_changed = scrub(item)
                    if isinstance(cleaned, str) and not cleaned:
                        changed = True
                        continue
                    out[key] = cleaned
                    changed = changed or item_changed
                return out, changed
            return value, False

        changed = False
        for field_name in ("core_intent", "todo", "completed_steps", "context_snapshot"):
            cleaned, field_changed = scrub(getattr(state, field_name))
            if field_changed:
                setattr(state, field_name, cleaned)
                changed = True
        refs = list(state.context_snapshot.get("invalidated_memory_refs", []))
        for needle in sorted(needles):
            if needle not in refs:
                refs.append(needle)
        state.context_snapshot["invalidated_memory_refs"] = refs[-50:]
        return bool(self.save(state, expected_version=state.state_version,
                              reason="invalidate memory derived refs")) if changed or refs else False

    def recover_pending_tools(self, task_id: str | None = None,
                              *, stale_after_seconds: float = 300.0) -> list[dict[str, Any]]:
        """Mark stale planned/running side effects ``unknown``.

        ``unknown`` is intentionally a terminal *decision barrier*, not a
        retry state: callers must query the external system or ask for HITL
        confirmation before replaying a non-idempotent operation.
        """
        states = []
        if task_id:
            state = self.get(task_id)
            if state is not None:
                states = [state]
        else:
            with self._db() as conn:
                rows = conn.execute(
                    "SELECT state_json FROM task_state ORDER BY updated_at DESC LIMIT 100"
                ).fetchall()
            for row in rows:
                try:
                    states.append(TaskState.from_dict(json.loads(row["state_json"])))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
        cutoff = datetime.now(timezone.utc).timestamp() - max(0.0, float(stale_after_seconds))
        recovered: list[dict[str, Any]] = []
        for state in states:
            changed = False
            for row in state.pending_tools:
                if row.get("status") not in {"planned", "running"}:
                    continue
                try:
                    stamp = datetime.fromisoformat(str(row.get("updated_at")).replace("Z", "+00:00"))
                    if stamp.tzinfo is None:
                        stamp = stamp.replace(tzinfo=timezone.utc)
                    stale = stamp.timestamp() <= cutoff
                except (TypeError, ValueError, OverflowError):
                    stale = True
                if not stale:
                    continue
                row["status"] = "unknown"
                row["lease_until"] = ""
                row["recovery_reason"] = "lease_expired_or_process_crash"
                row["updated_at"] = _stamp()
                recovered.append(dict(row, task_id=state.task_id))
                changed = True
            if changed:
                try:
                    self.save(state, expected_version=state.state_version,
                              reason="recover unknown side effect")
                except TaskStateConflict:
                    continue
        return recovered

    def events(self, task_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._db() as conn:
            rows = conn.execute(
                """SELECT id,task_id,from_phase,to_phase,state_version,reason,created_at
                   FROM task_state_event WHERE task_id=? ORDER BY id LIMIT ?""",
                (str(task_id), max(1, min(1000, int(limit)))),
            ).fetchall()
        return [dict(row) for row in rows]


__all__ = [
    "TASK_PHASES", "TOOL_OPERATION_STATUSES", "TaskState", "TaskStateError",
    "TaskStateConflict", "TaskStateStore",
]
