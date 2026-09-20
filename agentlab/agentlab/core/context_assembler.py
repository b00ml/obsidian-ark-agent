"""Deterministic context planning for the P1 shadow path.

The existing :mod:`agentlab.core.context` object remains the compatibility
renderer.  This module adds a side-effect-free planner that can be observed
before it is allowed to change the provider prompt.  It keeps four explicit
zones (instructions, task state, dialogue/memory, external observations),
filters inactive or out-of-scope evidence, and never drops required state just
to satisfy a soft token budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from collections import Counter
from typing import Any, Iterable, Mapping

from agentlab.contracts import RetrievalScope, current_retrieval_scope
from agentlab.core.context import estimate_text_tokens
from agentlab.memory.governance import DEFAULT_READ_STATUSES, status_matches


ZONES = ("instructions", "task_state", "dialogue_memory", "external")
_INACTIVE = {"candidate", "quarantine", "superseded", "archived", "revoked", "expired", "conflict"}


def _value(item: Any, key: str, default: Any = "") -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _safe_int(value: Any, default: int = 0, *, minimum: int | None = None) -> int:
    """Coerce untrusted metadata without letting one bad row abort planning."""
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    return result


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


@dataclass(frozen=True)
class ContextCandidate:
    """One bounded piece of context before it is selected into a plan."""

    id: str
    zone: str
    text: str
    priority: int = 0
    score: float = 0.0
    required: bool = False
    source: str = ""
    ref: str = ""
    status: str = "active"
    project_id: str = ""
    session_id: str = ""
    valid_from: str = ""
    valid_until: str = ""
    review_due_at: str = ""
    token_cost: int = 0

    def __post_init__(self) -> None:
        zone = str(self.zone or "").strip()
        ident = str(self.id or "").strip()
        text = str(self.text or "").strip()
        if zone not in ZONES:
            raise ValueError(f"unknown context zone: {zone}")
        if not ident:
            raise ValueError("context candidate id must not be empty")
        if not text:
            raise ValueError("context candidate text must not be empty")
        # Context candidates may come from tool/memory adapters.  Normalize
        # their scalar metadata at the contract boundary so malformed scores
        # or token costs do not break deterministic sorting/packing.
        object.__setattr__(self, "zone", zone)
        object.__setattr__(self, "id", ident)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "priority", _safe_int(self.priority))
        object.__setattr__(self, "score", _safe_float(self.score))
        object.__setattr__(self, "required", bool(self.required))
        object.__setattr__(self, "source", str(self.source or "").strip())
        object.__setattr__(self, "ref", str(self.ref or "").strip())
        object.__setattr__(self, "status", str(self.status or "active").strip())
        object.__setattr__(self, "project_id", str(self.project_id or "").strip())
        object.__setattr__(self, "session_id", str(self.session_id or "").strip())
        object.__setattr__(self, "valid_from", str(self.valid_from or "").strip())
        object.__setattr__(self, "valid_until", str(self.valid_until or "").strip())
        object.__setattr__(self, "review_due_at", str(self.review_due_at or "").strip())
        object.__setattr__(self, "token_cost", _safe_int(self.token_cost, minimum=0))

    @property
    def tokens(self) -> int:
        return max(1, int(self.token_cost or estimate_text_tokens(self.text)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "zone": self.zone,
            "text": self.text,
            "priority": self.priority,
            "score": self.score,
            "required": self.required,
            "source": self.source,
            "ref": self.ref,
            "status": self.status,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "review_due_at": self.review_due_at,
            "tokens": self.tokens,
        }


@dataclass(frozen=True)
class ContextPlan:
    """Serializable context selection result suitable for trace/eval output."""

    mode: str
    scope: RetrievalScope
    budget_tokens: int
    reserve_output_tokens: int
    used_tokens: int
    zones: dict[str, tuple[ContextCandidate, ...]]
    omitted: tuple[dict[str, str], ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def selected(self) -> tuple[ContextCandidate, ...]:
        return tuple(item for zone in ZONES for item in self.zones.get(zone, ()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "scope": self.scope.to_dict(),
            "budget_tokens": self.budget_tokens,
            "reserve_output_tokens": self.reserve_output_tokens,
            "used_tokens": self.used_tokens,
            "zones": {zone: [item.to_dict() for item in self.zones.get(zone, ())]
                      for zone in ZONES},
            "omitted": [dict(item) for item in self.omitted],
            "warnings": list(self.warnings),
        }

    def metrics(self) -> dict[str, Any]:
        """Return bounded, non-content metrics for runtime shadow/on traces."""
        selected = self.selected
        selected_sources = Counter((item.source or "unknown")[:96] for item in selected)
        selected_zones = Counter(item.zone for item in selected)
        omitted_reasons = Counter(str(item.get("reason") or "unknown")
                                  for item in self.omitted)
        refs = [item.ref for item in selected if item.ref]
        duplicate_refs = len(refs) - len(set(refs))
        return {
            "mode": self.mode,
            "budget_tokens": self.budget_tokens,
            "reserve_output_tokens": self.reserve_output_tokens,
            "used_tokens": self.used_tokens,
            "selected": len(selected),
            "omitted": len(self.omitted),
            "selected_by_zone": dict(sorted(selected_zones.items())),
            "selected_sources": dict(sorted(selected_sources.items())),
            "omitted_reasons": dict(sorted(omitted_reasons.items())),
            "scope_denied": omitted_reasons.get("scope_denied", 0),
            "duplicate_refs": max(0, duplicate_refs),
            "warnings": list(self.warnings),
        }


def _task_candidates(task_state: Any) -> list[ContextCandidate]:
    if task_state is None:
        return []
    out: list[ContextCandidate] = []
    core = _value(task_state, "core_intent", None)
    if isinstance(core, Mapping):
        goal = str(core.get("goal") or "").strip()
        constraints = [str(v).strip() for v in core.get("constraints", []) if str(v).strip()]
        if goal:
            out.append(ContextCandidate("task:goal", "task_state", goal,
                                        priority=100, required=True, source="task_state"))
        if constraints:
            out.append(ContextCandidate(
                "task:constraints", "task_state", "约束：" + "；".join(constraints),
                priority=99, required=True, source="task_state"))
    else:
        goal = str(core or "").strip()
        if goal:
            out.append(ContextCandidate("task:goal", "task_state", goal,
                                        priority=100, required=True, source="task_state"))
    current = str(_value(task_state, "current_subtask", "") or "").strip()
    if current:
        out.append(ContextCandidate("task:subtask", "task_state", current,
                                    priority=98, required=True, source="task_state"))
    todo = _value(task_state, "todo", [])
    if isinstance(todo, Iterable) and not isinstance(todo, (str, bytes, Mapping)):
        pending = []
        for item in todo:
            status = str(_value(item, "status", "") or "")
            if status not in {"done", "completed"}:
                label = str(_value(item, "title", "") or _value(item, "id", "") or "").strip()
                if label:
                    pending.append(label)
        if pending:
            out.append(ContextCandidate("task:todo", "task_state", "待办：" + "；".join(pending[:12]),
                                        priority=97, required=True, source="task_state"))
    return out


class ContextAssembler:
    """Build a bounded, auditable plan without changing the live prompt by default."""

    def __init__(
        self,
        *,
        budget_tokens: int = 8000,
        reserve_output_tokens: int = 1000,
        zone_budgets: Mapping[str, int] | None = None,
        mode: str = "shadow",
    ) -> None:
        self.budget_tokens = max(1, int(budget_tokens))
        self.reserve_output_tokens = max(0, int(reserve_output_tokens))
        self.zone_budgets = {
            zone: max(0, int(value))
            for zone, value in (zone_budgets or {}).items()
            if zone in ZONES
        }
        self.mode = mode if mode in {"shadow", "on"} else "shadow"
        self.last_plan: ContextPlan | None = None

    def assemble(
        self,
        *,
        scope: RetrievalScope | Mapping[str, Any] | None = None,
        task_state: Any = None,
        instructions: Iterable[Any] | None = None,
        history: Iterable[Any] | None = None,
        memory_candidates: Iterable[Any] | None = None,
        retrieval_items: Iterable[Any] | None = None,
        tool_observations: Iterable[Any] | None = None,
        mode: str | None = None,
    ) -> ContextPlan:
        request_scope = RetrievalScope.from_value(scope) if scope is not None else current_retrieval_scope()
        plan_mode = mode if mode in {"shadow", "on"} else self.mode
        all_candidates: list[ContextCandidate] = []
        all_candidates.extend(self._normalise(instructions, "instructions", "instruction", required=True))
        all_candidates.extend(_task_candidates(task_state))
        all_candidates.extend(self._normalise(history, "dialogue_memory", "history"))
        all_candidates.extend(self._normalise(memory_candidates, "dialogue_memory", "memory"))
        all_candidates.extend(self._normalise(retrieval_items, "external", "rag"))
        all_candidates.extend(self._normalise(tool_observations, "external", "tool"))

        selected: dict[str, list[ContextCandidate]] = {zone: [] for zone in ZONES}
        omitted: list[dict[str, str]] = []
        warnings: list[str] = []
        available_budget = max(0, self.budget_tokens - self.reserve_output_tokens)
        used = 0

        # Required candidates are admitted first.  If they exceed the budget we
        # preserve them and emit an explicit overflow warning instead of silently
        # dropping intent or safety constraints.
        for candidate in sorted(all_candidates, key=lambda item: (-int(item.required),
                                                                   -item.priority,
                                                                   -item.score,
                                                                   item.id)):
            if not self._scope_allowed(candidate, request_scope):
                omitted.append({"id": candidate.id, "reason": "scope_denied"})
                continue
            status = candidate.status.strip().lower()
            lifecycle_allowed, lifecycle_reason = status_matches({
                "status": status,
                "valid_from": candidate.valid_from,
                "valid_until": candidate.valid_until,
                "review_due_at": candidate.review_due_at,
            }, set(DEFAULT_READ_STATUSES))
            if status in _INACTIVE or not lifecycle_allowed:
                inactive_reason = status if status in _INACTIVE else str(lifecycle_reason or status).replace("status:", "")
                omitted.append({"id": candidate.id, "reason": f"inactive:{inactive_reason}"})
                continue
            zone_limit = self.zone_budgets.get(candidate.zone)
            zone_used = sum(item.tokens for item in selected[candidate.zone])
            fits_zone = zone_limit is None or zone_used + candidate.tokens <= zone_limit
            fits_total = used + candidate.tokens <= available_budget
            if candidate.required:
                selected[candidate.zone].append(candidate)
                used += candidate.tokens
                if not fits_zone or not fits_total:
                    warnings.append("required_budget_overflow")
                continue
            if not fits_zone:
                omitted.append({"id": candidate.id, "reason": "zone_budget"})
            elif not fits_total:
                omitted.append({"id": candidate.id, "reason": "total_budget"})
            else:
                selected[candidate.zone].append(candidate)
                used += candidate.tokens

        if omitted:
            warnings.append("context_items_omitted")
        plan = ContextPlan(
            mode=plan_mode,
            scope=request_scope,
            budget_tokens=self.budget_tokens,
            reserve_output_tokens=self.reserve_output_tokens,
            used_tokens=used,
            zones={zone: tuple(selected[zone]) for zone in ZONES},
            omitted=tuple(omitted),
            warnings=tuple(dict.fromkeys(warnings)),
        )
        self.last_plan = plan
        return plan

    @staticmethod
    def _scope_allowed(candidate: ContextCandidate, scope: RetrievalScope) -> bool:
        if scope.project_id and candidate.project_id and candidate.project_id not in {
            scope.project_id, "default",
        }:
            return False
        if scope.session_id and candidate.session_id and candidate.session_id != scope.session_id:
            return False
        return True

    @staticmethod
    def _normalise(items: Iterable[Any] | None, zone: str, prefix: str,
                   *, required: bool = False) -> list[ContextCandidate]:
        out: list[ContextCandidate] = []
        try:
            iterator = iter(items or ())
        except TypeError:
            return out
        for index, item in enumerate(iterator):
            text = str(_value(item, "text", "") or _value(item, "content", "") or "").strip()
            if not text:
                continue
            ref = str(_value(item, "ref", "") or "").strip()
            ident = str(_value(item, "id", "") or ref or f"{prefix}:{index}").strip()
            try:
                out.append(ContextCandidate(
                    id=ident,
                    zone=zone,
                    text=text,
                    priority=_safe_int(_value(item, "priority", 0) or 0),
                    score=_safe_float(_value(item, "score", 0.0) or 0.0),
                    required=bool(_value(item, "required", required)),
                    source=str(_value(item, "source", prefix) or prefix),
                    ref=ref,
                    status=str(_value(item, "status", "active") or "active"),
                    project_id=str(_value(item, "project_id", "") or ""),
                    session_id=str(_value(item, "session_id", "") or ""),
                    valid_from=str(_value(item, "valid_from", "") or ""),
                    valid_until=str(_value(item, "valid_until", "") or ""),
                    review_due_at=str(_value(item, "review_due_at", "") or ""),
                    token_cost=_safe_int(_value(item, "token_cost", 0) or 0,
                                         minimum=0),
                ))
            except (TypeError, ValueError):
                # An invalid optional observation is omitted; required task
                # state is constructed by the typed path above and therefore
                # still fails loudly if it is intrinsically invalid.
                continue
        return out


__all__ = ["ZONES", "ContextCandidate", "ContextPlan", "ContextAssembler"]
