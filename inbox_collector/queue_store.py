"""SQLite-backed inbox queue with atomic task/dedupe commits."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class InboxQueueStore:
    """Persistent inbox task queue and seen-marker store."""

    def __init__(self, db_path: str | os.PathLike[str], *,
                 queue_path: str | os.PathLike[str] | None = None,
                 seen_path: str | os.PathLike[str] | None = None):
        self.path = str(db_path)
        self.queue_path = str(queue_path) if queue_path else None
        self.seen_path = str(seen_path) if seen_path else None
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS inbox_task (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    url TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL DEFAULT 'pending',
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS inbox_task_status
                    ON inbox_task(status, created_at);
                CREATE TABLE IF NOT EXISTS inbox_seen (
                    marker TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> dict[str, Any]:
        task = json.loads(row["payload_json"])
        task.update({
            "id": row["id"],
            "type": row["type"],
            "url": row["url"],
            "status": row["status"],
            "retry_count": row["retry_count"],
        })
        return task

    @staticmethod
    def _legacy_seen(path: str | None) -> set[str]:
        if not path or not os.path.exists(path):
            return set()
        with open(path, "r", encoding="utf-8") as fh:
            return {line.strip() for line in fh if line.strip()}

    def migrate_legacy(self) -> dict[str, int]:
        """Import legacy queue.jsonl and seen.txt idempotently."""
        imported_tasks = 0
        imported_seen = 0
        conn = self._connect()
        try:
            now = self._now()
            if self.queue_path and os.path.exists(self.queue_path):
                with open(self.queue_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        try:
                            task = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        dedupe = task.get("dedupe_key") or (
                            f"{task.get('type', 'unknown')}:{task.get('url', '')}")
                        cur = conn.execute(
                            """INSERT OR IGNORE INTO inbox_task
                               (id,type,url,dedupe_key,status,retry_count,
                                payload_json,created_at,updated_at)
                               VALUES (?,?,?,?,?,?,?,?,?)""",
                            (task.get("id") or uuid.uuid4().hex[:16],
                             str(task.get("type", "unknown")),
                             str(task.get("url", "")), dedupe,
                             str(task.get("status", "pending")),
                             int(task.get("retry_count", 0) or 0),
                             json.dumps(task, ensure_ascii=False,
                                        separators=(",", ":")),
                             task.get("received_at") or now, now),
                        )
                        imported_tasks += cur.rowcount
            for marker in self._legacy_seen(self.seen_path):
                cur = conn.execute(
                    "INSERT OR IGNORE INTO inbox_seen(marker,created_at) VALUES (?,?)",
                    (marker, now),
                )
                imported_seen += cur.rowcount
            conn.commit()
        finally:
            conn.close()
        return {"tasks": imported_tasks, "seen": imported_seen}

    def seen_markers(self) -> set[str]:
        if self._legacy_seen(self.seen_path):
            self.migrate_legacy()
        conn = self._connect()
        try:
            return {row[0] for row in conn.execute("SELECT marker FROM inbox_seen")}
        finally:
            conn.close()

    def enqueue(self, task: dict[str, Any], dedupe_key: str,
                markers: list[str]) -> bool:
        """Commit one task and all seen markers in a single transaction.

        A duplicate dedupe key must still persist the new message marker.
        Otherwise the next poll reads the same email again even though its URL
        task already exists.
        """
        now = self._now()
        conn = self._connect()
        try:
            cur = conn.execute(
                """INSERT INTO inbox_task
                   (id,type,url,dedupe_key,status,retry_count,payload_json,
                    created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(dedupe_key) DO NOTHING""",
                (str(task.get("id") or uuid.uuid4().hex[:16]),
                 str(task.get("type", "unknown")), str(task.get("url", "")),
                 dedupe_key, str(task.get("status", "pending")),
                 int(task.get("retry_count", 0) or 0),
                 json.dumps(task, ensure_ascii=False, separators=(",", ":")),
                 task.get("received_at") or now, now),
            )
            inserted = cur.rowcount == 1
            for marker in markers:
                conn.execute(
                    "INSERT OR IGNORE INTO inbox_seen(marker,created_at) VALUES (?,?)",
                    (marker, now),
                )
            conn.commit()
            return inserted
        finally:
            conn.close()

    def mark_seen(self, markers: list[str]) -> int:
        now = self._now()
        conn = self._connect()
        try:
            count = 0
            for marker in markers:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO inbox_seen(marker,created_at) VALUES (?,?)",
                    (marker, now),
                )
                count += cur.rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def list_tasks(self, *, status: str | None = None,
                   limit: int = 50) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            if status:
                rows = conn.execute(
                    """SELECT * FROM inbox_task WHERE status=?
                       ORDER BY created_at,id LIMIT ?""",
                    (status, max(1, int(limit))),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM inbox_task
                       ORDER BY created_at,id LIMIT ?""",
                    (max(1, int(limit)),),
                ).fetchall()
            return [self._task_from_row(row) for row in rows]
        finally:
            conn.close()

    def stats(self) -> dict[str, int]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT status,COUNT(*) AS n FROM inbox_task GROUP BY status"
            ).fetchall()
            return {row["status"]: int(row["n"]) for row in rows}
        finally:
            conn.close()
