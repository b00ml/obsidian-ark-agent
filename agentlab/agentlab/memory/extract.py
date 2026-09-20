"""LLM 语义记忆提取（#10①/OPT-123）：对话片段 → 结构化记忆草稿 → 沉淀。

升级原 20 触发词正则金标准：提取交 LLM（memory-extract-user.st，importance
1-10 打分、低分不出），写入仍走 MemoryStore.commit（治理/去重在仓库侧不变）。
提取器缺席（无 key/构造失败）时 loop 自动回退触发词路——降级语义与 serve 一致。
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta, timezone
from typing import Any

from agentlab.core.message import Message

_log = logging.getLogger(__name__)

_DEPOSIT_QUEUE_FILE = "deposit-failures.jsonl"
_DEPOSIT_QUEUE_LIMIT = 100
_DEPOSIT_ENTRY_CHARS = 2000
_DEPOSIT_LEASE_SECONDS = 300
_QUEUE_LOCK_TIMEOUT_SECONDS = 10.0
_QUEUE_LOCK_STALE_SECONDS = 900.0
_QUEUE_LOCK = threading.Lock()
_SENSITIVE_RE = re.compile(
    r"(?i)(api[_ -]?key|access[_ -]?token|password|passwd|secret|cookie|密钥|密码)"
    r"\s*[:=]\s*[^\s,;]+"
)


# OPT-225 混合粒度：LLM 打分 >= 阈值直接落正式层（context 独立文件）；
# 低于阈值进 sessions 月桶（候选层），待重复使用/用户确认后晋升。
PROMOTE_IMPORTANCE = 7


class MemoryExtractor:
    """LLM 提取器：entries（user/assistant 消息）→ [{content, tags, importance}]。"""

    def __init__(self, provider, per_item_chars: int = 2000, max_items: int = 6,
                 min_importance: int = 4):
        self._provider = provider
        self._per_item_chars = per_item_chars
        self._max_items = max_items
        self._min_importance = min_importance

    async def extract(self, entries: list) -> list[dict]:
        from agentlab.prompts import load_prompt

        transcript = _render_transcript(entries)
        if not transcript:
            return []
        prompt = load_prompt("memory-extract-user", transcript=transcript,
                             max_items=str(self._max_items))
        resp = await self._provider.chat([Message(role="user", content=prompt)])
        return _parse_drafts(resp.content or "", self._max_items,
                             self._min_importance, self._per_item_chars)


def _render_transcript(entries: list) -> str:
    """只取 user/assistant 正文（工具/系统消息不进提取素材）。"""
    lines: list[str] = []
    for m in entries or []:
        role = getattr(m, "role", "")
        if role not in ("user", "assistant"):
            continue
        content = (getattr(m, "content", None) or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _parse_drafts(raw: str, max_items: int, min_importance: int,
                  per_item_chars: int) -> list[dict]:
    """容错解析 LLM 输出：取首个 '[' 到末个 ']' 之间当 JSON；低分/空内容丢弃。"""
    text = (raw or "").strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    # OPT-230 B5：类型白名单——core 永远只能人工晋升；非法值回退 context
    allowed_types = {"context", "decisions", "procedures"}
    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "") or "").strip()
        if not content:
            continue
        try:
            importance = int(item.get("importance", 5))
        except (TypeError, ValueError):
            importance = 5
        if importance < min_importance:
            continue
        tags = item.get("tags") or []
        if not isinstance(tags, list):
            tags = [tags]
        # OPT-230 B5/B7：type 白名单（core 禁止）；confidence 透传（1.0 或 hypothesis）
        mem_type = str(item.get("type", "context") or "context")
        if mem_type not in allowed_types:
            mem_type = "context"
        confidence = item.get("confidence", 1.0)
        if confidence != "hypothesis":
            try:
                confidence = min(1.0, max(0.0, float(confidence)))
            except (TypeError, ValueError):
                confidence = 1.0
        body = content if per_item_chars <= 0 else content[:per_item_chars]
        out.append({"content": body, "tags": [str(t) for t in tags][:6],
                    "importance": importance, "type": mem_type,
                    "confidence": confidence})
        if len(out) >= max_items:
            break
    return out


def _log_deposit_failure(
    depo: Any,
    entries: list,
    error: Exception,
    *,
    source_session: str = "",
) -> None:
    """Record a replayable extraction failure with bounded, redacted metadata.

    The historical file name is retained for compatibility.  Older rows that
    only contain ``sample`` remain readable as pending audit records, while
    new rows carry an explicit state machine and a bounded entry snapshot.
    """
    _enqueue_deposit_failure(depo, entries, error, source_session=source_session)


def _queue_path(depo: Any):
    """Resolve the local failure queue path without assuming a specific store."""
    config = getattr(depo, "_brain_config", {}) or {}
    vault = config.get("vault_path") or config.get("vault_root")
    if not vault:
        return None
    import pathlib

    return pathlib.Path(vault) / ".agent-brain" / "memory" / _DEPOSIT_QUEUE_FILE


def _safe_text(value: Any, limit: int = _DEPOSIT_ENTRY_CHARS) -> str:
    text = str(value or "")
    # Queue files are operational state, not a second trace.  Redact obvious
    # credential assignments before they can be persisted for replay.
    text = _SENSITIVE_RE.sub(lambda m: f"{m.group(1)}: [REDACTED]", text)
    return text[:limit]


def _entry_snapshot(entries: list) -> list[dict[str, str]]:
    snapshot: list[dict[str, str]] = []
    for message in entries or []:
        role = str(getattr(message, "role", "") or "")
        content = _safe_text(getattr(message, "content", "") or "")
        if role in {"user", "assistant"} and content:
            snapshot.append({"role": role, "content": content})
    return snapshot[:20]


def _safe_draft(draft: Any) -> dict[str, Any] | None:
    """Keep replay payloads bounded and redact credentials from LLM output."""
    if not isinstance(draft, dict):
        return None
    content = _safe_text(draft.get("content", ""), _DEPOSIT_ENTRY_CHARS).strip()
    if not content:
        return None
    tags = draft.get("tags") or []
    if not isinstance(tags, list):
        tags = [tags]
    try:
        importance = max(1, min(10, int(draft.get("importance", 5))))
    except (TypeError, ValueError, OverflowError):
        importance = 5
    out: dict[str, Any] = {
        "content": content,
        "tags": [_safe_text(tag, 80) for tag in tags[:6]],
        "importance": importance,
        "type": _safe_text(draft.get("type", "context"), 32),
        "confidence": draft.get("confidence", 1.0),
    }
    if out["confidence"] != "hypothesis":
        try:
            confidence = float(out["confidence"])
            out["confidence"] = min(1.0, max(0.0, confidence)) \
                if math.isfinite(confidence) else 1.0
        except (TypeError, ValueError, OverflowError):
            out["confidence"] = 1.0
    return out


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(value: datetime | None = None) -> str:
    return (value or _now()).isoformat(timespec="seconds")


def _max_attempts(depo: Any) -> int:
    policy = (getattr(depo, "_brain_config", {}) or {}).get("memory_policy") or {}
    try:
        return max(1, min(10, int(policy.get("retry_max_attempts", 3))))
    except (TypeError, ValueError):
        return 3


def _read_queue(path) -> list[dict]:
    if path is None or not path.exists():
        return []
    rows: list[dict] = []
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(row, dict):
                continue
            # Compatibility for OPT-230 audit-only rows: they are visible to
            # operators but cannot be replayed without an entry snapshot.
            row.setdefault("status", "pending")
            row.setdefault("attempts", 0)
            row.setdefault("next_retry_at", row.get("ts") or _stamp())
            row.setdefault("replayable", bool(row.get("entries")))
            row.setdefault("kind", "extract")
            row.setdefault("lease_until", "")
            rows.append(row)
    except OSError:
        return []
    return rows[-_DEPOSIT_QUEUE_LIMIT:]


def _write_queue(path, rows: list[dict]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    bounded = rows[-_DEPOSIT_QUEUE_LIMIT:]
    temp = path.with_suffix(".tmp")
    temp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bounded),
        encoding="utf-8",
    )
    temp.replace(path)


@contextmanager
def _queue_guard(path):
    """Serialize queue read/modify/write across threads *and* processes.

    JSONL remains a deliberately small, human-inspectable fallback.  The lock
    file closes the lost-update window when two workers share one Vault; a
    stale lock is reclaimable after a crashed process and all data writes stay
    atomic through ``_write_queue``.
    """
    with _QUEUE_LOCK:
        if path is None:
            yield
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        deadline = time.monotonic() + _QUEUE_LOCK_TIMEOUT_SECONDS
        acquired = False
        while not acquired:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, f"pid={os.getpid()}\n".encode("ascii", "replace"))
                finally:
                    os.close(fd)
                acquired = True
            except FileExistsError:
                try:
                    stale = (time.time() - lock_path.stat().st_mtime) > _QUEUE_LOCK_STALE_SECONDS
                except OSError:
                    stale = False
                if stale:
                    with suppress(OSError):
                        lock_path.unlink()
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("memory failure queue lock timeout")
                time.sleep(0.05)
        try:
            yield
        finally:
            with suppress(OSError):
                lock_path.unlink()


def _enqueue_deposit_failure(
    depo: Any,
    entries: list,
    error: Exception,
    *,
    source_session: str = "",
    drafts: list[dict] | None = None,
) -> None:
    """Append one bounded retry/dead-letter record; failure is best effort."""
    try:
        path = _queue_path(depo)
        if path is None:
            return
        snapshot = _entry_snapshot(entries)
        safe_drafts = [_safe_draft(d) for d in (drafts or [])]
        safe_drafts = [d for d in safe_drafts if d is not None]
        row = {
            "id": f"dep-{uuid.uuid4().hex[:12]}",
            "ts": _stamp(),
            "updated_at": _stamp(),
            "status": "pending",
            "attempts": 0,
            "max_attempts": _max_attempts(depo),
            "next_retry_at": _stamp(),
            "kind": "draft_commit" if drafts else "extract",
            "replayable": bool(snapshot or safe_drafts),
            "source_session": str(source_session or ""),
            "entries": snapshot,
            "entries_count": len(entries or []),
            "sample": "; ".join(item["content"][:60] for item in snapshot[:3])
                      or "; ".join(item["content"][:60] for item in safe_drafts[:3]),
            "error": _safe_text(error, 200),
        }
        if safe_drafts:
            row["drafts"] = safe_drafts[:10]
        with _queue_guard(path):
            _write_queue(path, _read_queue(path) + [row])
    except Exception:  # noqa: BLE001 - queue failure must not block the run
        pass


def deposit_failure_status(depo: Any) -> dict:
    """Return non-sensitive retry/dead-letter counters for operator diagnostics."""
    path = _queue_path(depo)
    with _queue_guard(path):
        rows = _read_queue(path)
    counts: dict[str, int] = {}
    ready: list[str] = []
    now = _now()
    for row in rows:
        status = str(row.get("status") or "pending")
        counts[status] = counts.get(status, 0) + 1
        if status == "pending":
            try:
                due = datetime.fromisoformat(str(row.get("next_retry_at")).replace("Z", "+00:00"))
                if due.tzinfo is None:
                    due = due.replace(tzinfo=timezone.utc)
                if due <= now:
                    ready.append(str(row.get("id", "")))
            except (TypeError, ValueError):
                ready.append(str(row.get("id", "")))
    return {
        "path": str(path) if path is not None else "",
        "total": len(rows),
        "counts": counts,
        "oldest_ready": ready[0] if ready else "",
    }


def _parse_queue_time(value: Any, fallback: datetime) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return fallback


def _claim_due_rows(path, limit: int, current: datetime) -> tuple[list[dict], int]:
    """Atomically claim due rows before awaiting provider/depository calls.

    ``processing`` plus a lease prevents concurrent retry coroutines in one
    process from replaying a row twice.  Expired leases are reclaimed so a
    crashed worker cannot strand the queue permanently.
    """
    claimed: list[dict] = []
    dead_letter = 0
    with _queue_guard(path):
        rows = _read_queue(path)
        changed = False
        for row in rows:
            if len(claimed) + dead_letter >= limit:
                break
            status = str(row.get("status") or "pending")
            if status == "processing":
                lease_until = _parse_queue_time(row.get("lease_until"), current)
                if lease_until > current:
                    continue
                row["status"] = "pending"
                changed = True
            if str(row.get("status") or "pending") != "pending":
                continue
            if not bool(row.get("replayable")):
                row["status"] = "dead_letter"
                row["error"] = "legacy_record_without_replay_snapshot"
                row["updated_at"] = _stamp(current)
                row["lease_until"] = ""
                dead_letter += 1
                changed = True
                continue
            if _parse_queue_time(row.get("next_retry_at"), current) > current:
                continue
            row["status"] = "processing"
            row["lease_until"] = _stamp(current + timedelta(seconds=_DEPOSIT_LEASE_SECONDS))
            row["updated_at"] = _stamp(current)
            claimed.append(dict(row))
            changed = True
        if changed:
            _write_queue(path, rows)
    return claimed, dead_letter


def _finish_claim(path, row_id: str, current: datetime, *, success: bool,
                  error: Exception | None = None, max_attempts: int = 3) -> str:
    """Finalize one claimed row and return its resulting queue status."""
    with _queue_guard(path):
        rows = _read_queue(path)
        target = next((row for row in rows if str(row.get("id")) == str(row_id)), None)
        if target is None or str(target.get("status") or "") != "processing":
            return "lost"
        target["updated_at"] = _stamp(current)
        target["lease_until"] = ""
        if success:
            target["status"] = "succeeded"
            target["completed_at"] = _stamp(current)
            _write_queue(path, rows)
            return "succeeded"
        target["attempts"] = int(target.get("attempts", 0) or 0) + 1
        target["error"] = _safe_text(error, 200)
        if target["attempts"] >= max(1, int(max_attempts)):
            target["status"] = "dead_letter"
            _write_queue(path, rows)
            return "dead_letter"
        delay = min(3600.0, float(2 ** max(0, int(target["attempts"]) - 1)))
        target["status"] = "pending"
        target["next_retry_at"] = _stamp(current + timedelta(seconds=delay))
        _write_queue(path, rows)
        return "pending"


async def retry_deposit_failures(
    depo: Any,
    extractor: Any,
    *,
    limit: int = 10,
    now: datetime | None = None,
) -> dict:
    """Replay due memory deposits with bounded backoff and dead-lettering.

    The function is explicit by design: normal request handling does not spend
    provider calls on retries.  ``extract`` failures rerun the extractor from
    the bounded snapshot; ``draft_commit`` failures retry the already validated
    drafts without another LLM call.  Queue writes are atomic and idempotent.
    """
    path = _queue_path(depo)
    if path is None:
        return {"status": "unavailable", "processed": 0, "succeeded": 0,
                "retried": 0, "dead_letter": 0}
    current = now or _now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    try:
        max_items = max(1, min(100, int(limit)))
    except (TypeError, ValueError, OverflowError):
        max_items = 10
    claimed, dead = _claim_due_rows(path, max_items, current)
    processed = dead + len(claimed)
    succeeded = retried = 0
    for row in claimed:
        try:
            from agentlab.core.message import Message

            entries = [Message(role=item["role"], content=item.get("content", ""))
                       for item in row.get("entries", [])
                       if isinstance(item, dict) and item.get("role") in {"user", "assistant"}]
            if row.get("kind") == "draft_commit":
                drafts = [d for d in row.get("drafts", []) if isinstance(d, dict)]
            else:
                drafts = await extractor.extract(entries)
            committed, failed = await _deposit_drafts(
                depo, drafts, source_session=str(row.get("source_session") or ""),
            )
            if failed or not drafts or committed < len(drafts):
                raise RuntimeError(f"replay commit incomplete: {committed}/{len(drafts)}")
        except Exception as exc:  # noqa: BLE001 - transition to retry/dead-letter
            try:
                row_max_attempts = int(row.get("max_attempts", _max_attempts(depo)))
            except (TypeError, ValueError, OverflowError):
                row_max_attempts = _max_attempts(depo)
            outcome = _finish_claim(
                path, str(row.get("id", "")), current, success=False,
                error=exc, max_attempts=row_max_attempts,
            )
            if outcome == "dead_letter":
                dead += 1
            elif outcome == "pending":
                retried += 1
            continue
        if _finish_claim(path, str(row.get("id", "")), current, success=True) == "succeeded":
            succeeded += 1
    return {"status": "ok", "processed": processed, "succeeded": succeeded,
            "retried": retried, "dead_letter": dead}


def _commit_draft_sync(depo: Any, draft: dict, source_session: str = "") -> dict:
    """Commit one validated draft using the same candidate-first policy as live writes."""
    _imp = draft.get("importance") or 0
    if _imp >= PROMOTE_IMPORTANCE:
        mem_type, bucket = draft.get("type") or "context", False
    else:
        mem_type, bucket = "sessions", True
    kwargs = {
        "tags": draft.get("tags"), "source_session": source_session,
        "dedup": True, "importance": draft.get("importance"),
        "confidence": draft.get("confidence"), "mem_type": mem_type,
        "bucket": bucket, "source": "assistant",
        "source_ref": f"session:{source_session}" if source_session else "",
        "candidate_first": True,
    }
    import inspect

    params = inspect.signature(depo.commit).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    result = depo.commit(draft["content"], **kwargs)
    return result if isinstance(result, dict) else {"status": "committed"}


async def _deposit_drafts(depo: Any, drafts: list[dict], *, source_session: str = "") -> tuple[int, list[dict]]:
    loop = asyncio.get_running_loop()
    committed = 0
    failed: list[dict] = []
    for draft in drafts or []:
        try:
            result = await loop.run_in_executor(
                None, lambda d=draft: _commit_draft_sync(depo, d, source_session),
            )
            if result.get("status") == "committed":
                committed += 1
            else:
                failed.append(draft)
        except Exception:  # noqa: BLE001 - caller records draft-level failure
            failed.append(draft)
    return committed, failed




async def deposit_via_extractor(depo: Any, extractor: Any, entries: list,
                                source_session: str = "") -> int:
    """提取 + 逐条落库；失败进入有界 retry/dead-letter 队列。"""
    try:
        drafts = await extractor.extract(entries)
    except Exception as e:  # noqa: BLE001 —— 提取失败不影响主循环，但留痕可查（OPT-230 B6）
        _log.exception("记忆提取失败（extractor=%s）", type(extractor).__name__)
        _log_deposit_failure(depo, entries, e, source_session=source_session)
        return 0
    committed, failed = await _deposit_drafts(
        depo, drafts, source_session=source_session,
    )
    for draft in failed:
        _log.error("记忆沉淀失败（depository=%s）", type(depo).__name__)
        _enqueue_deposit_failure(
            depo, [], RuntimeError("memory draft commit failed"),
            source_session=source_session, drafts=[draft],
        )
    return committed
