"""Run-level budget and trace contracts used by the agent loop.

The contracts are deliberately dependency-light and serialisable.  A zero
limit means "unbounded" for backwards compatibility with existing callers.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


CURRENT_RUN_BUDGET: ContextVar["RunBudget | None"] = ContextVar(
    "agentlab_current_run_budget", default=None
)
CURRENT_RUN_TRACE: ContextVar["RunTrace | None"] = ContextVar(
    "agentlab_current_run_trace", default=None
)


def current_run_budget() -> "RunBudget | None":
    return CURRENT_RUN_BUDGET.get()


def current_run_trace() -> "RunTrace | None":
    return CURRENT_RUN_TRACE.get()


def action_fingerprint(tool_name: str, arguments: Any = None, *, result: Any = None) -> str:
    """Return a stable, redacted-safe fingerprint for semantic tool actions.

    JSON objects are sorted and common volatile fields are ignored.  Including
    a bounded result hash lets callers distinguish a state-changing retrieval
    from a no-op repeated retrieval without storing tool output in trace.
    """
    volatile = {"timestamp", "started_at", "trace_id", "run_id", "nonce"}

    def normalise(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return normalise(json.loads(value))
            except (TypeError, ValueError):
                return " ".join(value.split()).strip()
        if isinstance(value, dict):
            return {str(k): normalise(v) for k, v in sorted(value.items())
                    if str(k).lower() not in volatile}
        if isinstance(value, (list, tuple)):
            return [normalise(v) for v in value]
        return value

    payload: dict[str, Any] = {"tool": str(tool_name), "arguments": normalise(arguments)}
    if result is not None:
        text = json.dumps(normalise(result), ensure_ascii=False, sort_keys=True,
                          default=str)[:2000]
        payload["result_hash"] = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


@dataclass
class RunBudget:
    """Mutable per-run limits and stop state.

    ``deadline_at`` uses ``time.monotonic``.  It may be supplied as an
    absolute monotonic timestamp or derived with :meth:`from_timeout`.
    """

    deadline_at: float | None = None
    max_llm_calls: int = 0
    max_tool_calls: int = 0
    max_react_rounds: int = 0
    max_plan_steps: int = 0
    max_retries: int = 0
    cancelled: bool = False
    stop_reason: str = ""
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    llm_calls: int = 0
    tool_calls: int = 0
    react_rounds: int = 0
    plan_steps: int = 0
    retries: int = 0

    @classmethod
    def from_timeout(cls, timeout: float | None, **limits: int) -> "RunBudget":
        deadline = None if timeout is None or timeout <= 0 else time.monotonic() + timeout
        return cls(deadline_at=deadline, **limits)

    def remaining(self) -> float | None:
        return None if self.deadline_at is None else max(0.0, self.deadline_at - time.monotonic())

    def expired(self) -> bool:
        return self.deadline_at is not None and self.remaining() <= 0

    def cancel(self, reason: str = "cancelled") -> None:
        self.cancelled = True
        self.stop_reason = reason or "cancelled"

    def _admit(self, current: int, limit: int, reason: str) -> bool:
        if self.cancelled or self.expired():
            if not self.stop_reason:
                self.stop_reason = "deadline" if self.expired() else "cancelled"
            return False
        if limit and current >= limit:
            self.stop_reason = reason
            return False
        return True

    def admit_llm(self) -> bool:
        ok = self._admit(self.llm_calls, self.max_llm_calls, "max_llm_calls")
        if ok:
            self.llm_calls += 1
        return ok

    def admit_tool(self, count: int = 1) -> bool:
        if count < 1:
            return True
        ok = self._admit(self.tool_calls, self.max_tool_calls, "max_tool_calls")
        if ok and self.max_tool_calls and self.tool_calls + count > self.max_tool_calls:
            self.stop_reason = "max_tool_calls"
            return False
        if ok:
            self.tool_calls += count
        return ok

    def admit_round(self) -> bool:
        ok = self._admit(self.react_rounds, self.max_react_rounds, "max_react_rounds")
        if ok:
            self.react_rounds += 1
        return ok

    def admit_retry(self) -> bool:
        ok = self._admit(self.retries, self.max_retries, "max_retries")
        if ok:
            self.retries += 1
        return ok

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "run_id", "deadline_at", "max_llm_calls", "max_tool_calls",
            "max_react_rounds", "max_plan_steps", "max_retries", "cancelled",
            "stop_reason", "llm_calls", "tool_calls", "react_rounds",
            "plan_steps", "retries")}


@dataclass
class RunTrace:
    """Bounded, serialisable stage observations for one run."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    started_at: float = field(default_factory=time.time)
    events: list[dict[str, Any]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)

    def count(self, name: str, amount: int = 1) -> int:
        self.counters[name] = self.counters.get(name, 0) + amount
        return self.counters[name]

    def span(self, stage: str, started_at: float, ended_at: float | None = None,
             **fields: Any) -> dict[str, Any]:
        ended = time.time() if ended_at is None else ended_at
        event = {"stage": stage, "started_at": started_at, "ended_at": ended,
                 "duration_ms": max(0, int((ended - started_at) * 1000))}
        event.update({k: v for k, v in fields.items() if v is not None})
        self.events.append(event)
        # Keep traces bounded even for long-running sessions.
        del self.events[:-500]
        self.count(stage)
        return event

    def to_dict(self, budget: "RunBudget | None" = None) -> dict[str, Any]:
        budget_data = budget.to_dict() if budget is not None else dict(self.budget)
        return {"run_id": self.run_id, "started_at": self.started_at,
                "counters": dict(self.counters), "budget": budget_data,
                "events": list(self.events)}
