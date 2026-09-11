"""Durable local Task/Run state for long-running Agent Hub jobs.

SQLite is intentionally used instead of an external broker: Ark is a single-user
local application.  Delivery is at-least-once; a worker must claim a task lease
and make its handler idempotent.  All public methods open short-lived connections
so the store is safe to use from the MCP worker thread and after a process restart.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


TASK_STATUSES = {"pending", "running", "succeeded", "failed", "cancelled", "dead_letter"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(value: datetime | None = None) -> str:
    return (value or _now()).isoformat(timespec="milliseconds")


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return json.dumps({"repr": repr(value)}, ensure_ascii=False)


class TaskStore:
    """SQLite-backed task state machine with leases, retry and idempotency."""

    def __init__(self, path: str | os.PathLike[str], *, default_max_attempts: int = 3,
                 default_lease_seconds: float = 300.0):
        self.path = str(path)
        self.default_max_attempts = max(1, int(default_max_attempts))
        self.default_lease_seconds = max(1.0, float(default_lease_seconds))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
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
                CREATE TABLE IF NOT EXISTS task (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK (status IN
                      ('pending','running','succeeded','failed','cancelled','dead_letter')),
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    next_retry_at TEXT,
                    lease_until TEXT,
                    error_code TEXT,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS task_status_retry
                    ON task(status, next_retry_at, created_at);
                CREATE TABLE IF NOT EXISTS run (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES task(id),
                    session_id TEXT,
                    project_id TEXT,
                    status TEXT NOT NULL,
                    trace_path TEXT,
                    created_at TEXT NOT NULL,
                    finished_at TEXT,
                    error_code TEXT,
                    error_message TEXT
                );
                CREATE TABLE IF NOT EXISTS task_event (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES task(id),
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS task_event_task ON task_event(task_id, id);
                """
            )

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        item = dict(row)
        for field in ("payload_json", "result_json"):
            raw = item.pop(field, None)
            key = field.removesuffix("_json")
            if raw is None:
                item[key] = None
            else:
                try:
                    item[key] = json.loads(raw)
                except (TypeError, ValueError):
                    item[key] = raw
        return item

    def create_task(self, kind: str, payload: Any, *, dedupe_key: str | None = None,
                    max_attempts: int | None = None, task_id: str | None = None) -> dict:
        """Create or return the existing task for a dedupe key (idempotent)."""
        dedupe = dedupe_key or f"{kind}:{uuid.uuid4().hex}"
        now = _stamp()
        task_id = task_id or uuid.uuid4().hex[:16]
        with self._db() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO task
                   (id,kind,dedupe_key,status,payload_json,max_attempts,created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (task_id, str(kind), dedupe, "pending", _json(payload),
                 max(1, int(max_attempts or self.default_max_attempts)), now),
            )
            row = conn.execute("SELECT * FROM task WHERE dedupe_key=?", (dedupe,)).fetchone()
            assert row is not None
            return self._decode(row) or {}

    def get(self, task_id: str) -> dict | None:
        self.recover_expired()
        with self._db() as conn:
            return self._decode(conn.execute("SELECT * FROM task WHERE id=?", (task_id,)).fetchone())

    def list_tasks(self, *, limit: int = 100, status: str | None = None) -> list[dict]:
        self.recover_expired()
        limit = max(1, min(int(limit), 1000))
        with self._db() as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM task ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                if status not in TASK_STATUSES:
                    raise ValueError(f"未知任务状态: {status}")
                rows = conn.execute(
                    "SELECT * FROM task WHERE status=? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            return [self._decode(row) or {} for row in rows]

    def claim(self, task_id: str, *, lease_seconds: float | None = None) -> dict | None:
        """Atomically claim a pending task, incrementing its delivery attempt."""
        self.recover_expired()
        now = _now()
        lease_until = _stamp(now + timedelta(seconds=max(
            0.01, float(lease_seconds if lease_seconds is not None
                       else self.default_lease_seconds))))
        now_stamp = _stamp(now)
        with self._db() as conn:
            # The status predicate and row-count check are the claim gate. A
            # preceding SELECT is not sufficient: two worker threads can both
            # observe ``pending`` before either transaction commits.
            changed = conn.execute(
                """UPDATE task SET status='running', attempt=attempt+1,
                   started_at=COALESCE(started_at,?), lease_until=?, next_retry_at=NULL
                   WHERE id=? AND status='pending'
                     AND (next_retry_at IS NULL OR next_retry_at<=?)""",
                (now_stamp, lease_until, task_id, now_stamp),
            ).rowcount
            if changed != 1:
                return None
            row = conn.execute("SELECT * FROM task WHERE id=?", (task_id,)).fetchone()
            self._event(conn, task_id, "claimed", {"lease_until": lease_until})
            return self._decode(row)

    def is_cancelled(self, task_id: str) -> bool:
        with self._db() as conn:
            row = conn.execute("SELECT status FROM task WHERE id=?", (task_id,)).fetchone()
            return row is None or row["status"] == "cancelled"

    def complete(self, task_id: str, result: Any) -> dict | None:
        now = _stamp()
        with self._db() as conn:
            row = conn.execute("SELECT * FROM task WHERE id=?", (task_id,)).fetchone()
            if row is None:
                return None
            if row["status"] == "cancelled":
                self._event(conn, task_id, "cancelled_result_discarded", {})
                return self._decode(row)
            if row["status"] != "running":
                return self._decode(row)
            conn.execute(
                """UPDATE task SET status='succeeded',result_json=?,finished_at=?,
                   lease_until=NULL,error_code=NULL,error_message=NULL WHERE id=?""",
                (_json(result), now, task_id),
            )
            self._event(conn, task_id, "succeeded", {})
            return self._decode(conn.execute("SELECT * FROM task WHERE id=?", (task_id,)).fetchone())

    def fail(self, task_id: str, error_code: str, error_message: str, *,
             retryable: bool = True, retry_delay: float = 0.0) -> dict | None:
        with self._db() as conn:
            row = conn.execute("SELECT * FROM task WHERE id=?", (task_id,)).fetchone()
            if row is None:
                return None
            if row["status"] == "cancelled":
                self._event(conn, task_id, "cancelled_error_discarded", {"error_code": error_code})
                return self._decode(row)
            terminal = (not retryable) or row["attempt"] >= row["max_attempts"]
            status = "dead_letter" if retryable and terminal else ("failed" if not retryable else "pending")
            retry_at = None if status != "pending" else _stamp(_now() + timedelta(seconds=max(0.0, retry_delay)))
            finished = _stamp() if status in {"failed", "dead_letter"} else None
            conn.execute(
                """UPDATE task SET status=?,next_retry_at=?,finished_at=?,lease_until=NULL,
                   error_code=?,error_message=? WHERE id=?""",
                (status, retry_at, finished, str(error_code), str(error_message), task_id),
            )
            self._event(conn, task_id, status, {"error_code": error_code})
            return self._decode(conn.execute("SELECT * FROM task WHERE id=?", (task_id,)).fetchone())

    def cancel(self, task_id: str) -> dict | None:
        with self._db() as conn:
            row = conn.execute("SELECT * FROM task WHERE id=?", (task_id,)).fetchone()
            if row is None:
                return None
            if row["status"] in {"pending", "running"}:
                conn.execute(
                    "UPDATE task SET status='cancelled',finished_at=?,lease_until=NULL WHERE id=?",
                    (_stamp(), task_id),
                )
                self._event(conn, task_id, "cancelled", {})
            return self._decode(conn.execute("SELECT * FROM task WHERE id=?", (task_id,)).fetchone())

    def recover_expired(self) -> int:
        """Return expired leases to pending, or dead-letter exhausted attempts."""
        now = _stamp()
        with self._db() as conn:
            rows = conn.execute(
                "SELECT id,attempt,max_attempts FROM task WHERE status='running' AND lease_until IS NOT NULL AND lease_until<?",
                (now,),
            ).fetchall()
            for row in rows:
                if row["attempt"] >= row["max_attempts"]:
                    conn.execute(
                        "UPDATE task SET status='dead_letter',finished_at=?,lease_until=NULL,error_code='LEASE_EXPIRED',error_message='任务租约过期且已耗尽重试' WHERE id=?",
                        (now, row["id"]),
                    )
                    self._event(conn, row["id"], "dead_letter", {"error_code": "LEASE_EXPIRED"})
                else:
                    conn.execute(
                        "UPDATE task SET status='pending',next_retry_at=?,lease_until=NULL,error_code='LEASE_EXPIRED',error_message='任务租约过期，等待重试' WHERE id=?",
                        (now, row["id"]),
                    )
                    self._event(conn, row["id"], "retry", {"error_code": "LEASE_EXPIRED"})
            return len(rows)

    def create_run(self, task_id: str, *, session_id: str = "", project_id: str = "",
                   trace_path: str = "") -> dict:
        run_id = uuid.uuid4().hex[:16]
        with self._db() as conn:
            conn.execute(
                """INSERT INTO run(id,task_id,session_id,project_id,status,trace_path,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (run_id, task_id, session_id, project_id, "running", trace_path, _stamp()),
            )
            return dict(conn.execute("SELECT * FROM run WHERE id=?", (run_id,)).fetchone())

    def finish_run(self, run_id: str, status: str, *, error_code: str = "",
                   error_message: str = "") -> dict | None:
        if status not in TASK_STATUSES:
            raise ValueError(f"未知 run 状态: {status}")
        with self._db() as conn:
            conn.execute(
                "UPDATE run SET status=?,finished_at=?,error_code=?,error_message=? WHERE id=?",
                (status, _stamp(), error_code, error_message, run_id),
            )
            row = conn.execute("SELECT * FROM run WHERE id=?", (run_id,)).fetchone()
            return dict(row) if row else None

    def events(self, task_id: str, *, limit: int = 100) -> list[dict]:
        """Return the append-only task event history for diagnostics and UI."""
        limit = max(1, min(int(limit), 1000))
        with self._db() as conn:
            rows = conn.execute(
                """SELECT id,task_id,event_type,payload_json,created_at
                   FROM task_event WHERE task_id=? ORDER BY id LIMIT ?""",
                (task_id, limit),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                try:
                    item["payload"] = json.loads(item.pop("payload_json"))
                except (TypeError, ValueError):
                    item["payload"] = item.pop("payload_json")
                result.append(item)
            return result

    def _event(self, conn: sqlite3.Connection, task_id: str, event_type: str, payload: Any) -> None:
        conn.execute(
            "INSERT INTO task_event(task_id,event_type,payload_json,created_at) VALUES(?,?,?,?)",
            (task_id, event_type, _json(payload), _stamp()),
        )
