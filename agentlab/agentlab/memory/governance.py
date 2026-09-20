"""Memory write/read governance primitives (S1/P0).

The Markdown files remain the source of truth, but every new record carries a
small, explicit schema and every read goes through the same status/scope/time
filters.  The helpers in this module are deliberately dependency free so they
can be used by the brain adapter, the Markdown store and offline evaluators.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


SCHEMA_VERSION = 2
VALID_STATUSES = frozenset({
    "candidate", "active", "quarantine", "superseded", "archived",
    "revoked", "expired", "conflict",
})
DEFAULT_READ_STATUSES = frozenset({"active"})

# External text must never be promoted merely because it says "remember this".
# Keep this conservative: normal user prose is allowed, while credentials and
# instruction-hijacking phrases are quarantined for review.
_UNTRUSTED_INSTRUCTION_RE = re.compile(
    r"(?:ignore\s+(?:all\s+)?(?:previous|system|安全|规则|instructions?)|"
    r"忽略(?:之前|系统|安全|所有)?(?:的)?(?:指令|规则|提示)|"
    r"(?:记住|remember)\s*(?:这个|this)?\s*(?:api[_ -]?key|token|password|密码|密钥|cookie|secret)|"
    r"(?:api[_ -]?key|access[_ -]?token|密码|密钥|cookie|secret)\s*[:=])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WriteDecision:
    status: str
    reasons: tuple[str, ...] = ()
    accepted: bool = True


@dataclass
class ReadAudit:
    """Counters exposed to trace without returning sensitive memory text."""

    filtered_reasons: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.filtered_reasons[reason] = self.filtered_reasons.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {"filtered_reasons": dict(self.filtered_reasons)}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def content_hash(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def normalise_scope(project_id: str | None = None, session_id: str | None = None,
                   scope: Mapping[str, Any] | str | None = None) -> dict[str, str]:
    """Return a stable, serialisable scope without accepting arbitrary fields."""

    project = str(project_id or "default").strip() or "default"
    session = str(session_id or "").strip()
    if isinstance(scope, str) and scope.strip():
        project = scope.strip()
    elif isinstance(scope, Mapping):
        project = str(scope.get("project_id") or project).strip() or "default"
        session = str(scope.get("session_id") or session).strip()
    return {"project_id": project, "session_id": session}


def parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_expired(valid_until: Any, *, now: datetime | None = None) -> bool:
    deadline = parse_datetime(valid_until)
    if deadline is None:
        return False
    current = now or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc) >= deadline


def is_not_yet_valid(valid_from: Any, *, now: datetime | None = None) -> bool:
    """Return whether a memory has an explicit future effective time."""
    start = parse_datetime(valid_from)
    if start is None:
        return False
    current = now or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc) < start


def normalise_subject(value: Any) -> str:
    """Normalise an explicit conflict key without inferring one from prose."""
    subject = re.sub(r"\s+", " ", str(value or "").strip()).casefold()
    return subject[:256]


def is_review_due(review_due_at: Any, *, now: datetime | None = None) -> bool:
    """Return whether an active memory has reached its explicit review date."""
    deadline = parse_datetime(review_due_at)
    if deadline is None:
        return False
    current = now or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc) >= deadline


def decide_write(*, requested_status: str | None = None, mem_type: str,
                 confidence: Any, source: str, content: str,
                 candidate_first: bool = False,
                 explicit_confirmation: bool = False,
                 quarantine_external: bool = True) -> WriteDecision:
    """Apply the common candidate/quarantine policy.

    Existing explicit user calls remain compatible (`candidate_first=False`).
    Automatic extraction passes `candidate_first=True`, so model-derived facts
    are reviewable before they become active.  Core memories additionally need
    explicit confirmation to become active.
    """

    status = str(requested_status or ("candidate" if candidate_first else "active")).strip().lower()
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid memory status: {status}")
    reasons: list[str] = []
    source_kind = str(source or "unknown").strip().lower()
    if quarantine_external and source_kind in {"web", "tool", "external"} \
            and _UNTRUSTED_INSTRUCTION_RE.search(content or ""):
        status = "quarantine"
        reasons.append("external_instruction_or_secret")
    if candidate_first and status == "active":
        status = "candidate"
        reasons.append("candidate_first")
    if (mem_type == "core" and status == "active" and not explicit_confirmation
            and source_kind not in {"user", "human"}):
        status = "candidate"
        reasons.append("core_requires_confirmation")
    if confidence == "hypothesis" and status == "active" and not explicit_confirmation:
        status = "candidate"
        reasons.append("hypothesis_requires_confirmation")
    return WriteDecision(status=status, reasons=tuple(reasons))


def memory_scope_matches(memory: Mapping[str, Any], *, project_id: str | None,
                         session_id: str | None, allow_default_shared: bool = True) -> bool:
    """Project/session isolation check used before scoring a memory."""

    requested_project = str(project_id or "").strip()
    requested_session = str(session_id or "").strip()
    scope = memory.get("scope")
    nested = scope if isinstance(scope, Mapping) else {}
    memory_project = str(memory.get("project_id") or nested.get("project_id") or "default").strip()
    memory_session = str(memory.get("session_id") or nested.get("session_id") or "").strip()
    if requested_project and memory_project != requested_project:
        if not (allow_default_shared and memory_project == "default"):
            return False
    if requested_session and memory_session and memory_session != requested_session:
        return False
    return True


def status_matches(memory: Mapping[str, Any], statuses: set[str] | frozenset[str] | None,
                   *, include_archive: bool = False,
                   now: datetime | None = None) -> tuple[bool, str | None]:
    status = str(memory.get("status") or "active").strip().lower()
    allowed = set(statuses or DEFAULT_READ_STATUSES)
    if include_archive:
        allowed.update({"archived", "superseded", "revoked", "expired"})
    if is_not_yet_valid(memory.get("valid_from"), now=now):
        return False, "not_yet_valid"
    if is_expired(memory.get("valid_until"), now=now) and status == "active":
        status = "expired"
    if status == "active" and is_review_due(memory.get("review_due_at"), now=now):
        # A due item remains in the source of truth but is not auto-injected
        # until a reviewer explicitly asks for due items or promotes it.
        if "review_due" not in allowed:
            return False, "review_due"
        status = "review_due"
    if status not in allowed:
        return False, f"status:{status}"
    return True, None


def suspicious_external_text(content: str) -> bool:
    return bool(_UNTRUSTED_INSTRUCTION_RE.search(content or ""))
