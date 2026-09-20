"""Bounded Plan-and-Execute primitives used by the shadow path.

The planner is deliberately deterministic: it creates a small, inspectable
plan from the requested capabilities and never calls an LLM.  The executor
accepts a caller-owned step runner, so production can keep the existing ReAct
runner while this path records and enforces plan boundaries.
"""
from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Sequence


PLAN_STATUSES = frozenset({"planned", "running", "succeeded", "blocked", "failed", "cancelled"})
STEP_STATUSES = frozenset({"planned", "ready", "running", "succeeded", "blocked", "failed", "cancelled", "needs_review"})


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _clean_text(value: Any, *, limit: int = 500) -> str:
    return str(value or "").strip()[:limit]


@dataclass
class PlanStep:
    step_id: str
    objective: str
    dependencies: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    input_refs: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    completion_check: str = "evidence"
    status: str = "planned"
    attempts: int = 0
    evidence_refs: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.step_id = _clean_text(self.step_id, limit=128)
        self.objective = _clean_text(self.objective)
        if not self.step_id or not self.objective:
            raise ValueError("plan step requires step_id and objective")
        self.dependencies = [str(item) for item in self.dependencies]
        self.allowed_tools = [str(item) for item in self.allowed_tools]
        self.input_refs = [str(item) for item in self.input_refs]
        self.expected_outputs = [str(item) for item in self.expected_outputs]
        self.evidence_refs = [str(item) for item in self.evidence_refs]
        if self.status not in STEP_STATUSES:
            raise ValueError(f"unknown plan step status: {self.status}")
        self.attempts = max(0, int(self.attempts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "objective": self.objective,
            "dependencies": list(self.dependencies),
            "allowed_tools": list(self.allowed_tools),
            "input_refs": list(self.input_refs),
            "expected_outputs": list(self.expected_outputs),
            "completion_check": self.completion_check,
            "status": self.status,
            "attempts": self.attempts,
            "evidence_refs": list(self.evidence_refs),
            "result": dict(self.result),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlanStep":
        return cls(**{key: value[key] for key in (
            "step_id", "objective", "dependencies", "allowed_tools", "input_refs",
            "expected_outputs", "completion_check", "status", "attempts",
            "evidence_refs", "result",
        ) if key in value})


@dataclass
class Plan:
    plan_id: str
    goal: str
    scope: dict[str, Any] = field(default_factory=dict)
    status: str = "planned"
    created_at: str = field(default_factory=_stamp)
    deadline_at: float | None = None
    steps: list[PlanStep] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.plan_id = _clean_text(self.plan_id, limit=128)
        self.goal = _clean_text(self.goal)
        if not self.plan_id or not self.goal:
            raise ValueError("plan requires plan_id and goal")
        if self.status not in PLAN_STATUSES:
            raise ValueError(f"unknown plan status: {self.status}")
        ids = {step.step_id for step in self.steps}
        if len(ids) != len(self.steps):
            raise ValueError("plan step ids must be unique")
        if any(dep not in ids for step in self.steps for dep in step.dependencies):
            raise ValueError("plan dependency references an unknown step")

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "scope": dict(self.scope),
            "status": self.status,
            "created_at": self.created_at,
            "deadline_at": self.deadline_at,
            "steps": [step.to_dict() for step in self.steps],
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Plan":
        steps = [PlanStep.from_dict(item) for item in value.get("steps", [])]
        return cls(
            plan_id=value.get("plan_id", ""), goal=value.get("goal", ""),
            scope=dict(value.get("scope", {})), status=value.get("status", "planned"),
            created_at=value.get("created_at", _stamp()), deadline_at=value.get("deadline_at"),
            steps=steps, provenance=dict(value.get("provenance", {})),
        )

    def ready_steps(self) -> list[PlanStep]:
        done = {step.step_id for step in self.steps if step.status == "succeeded"}
        return [step for step in self.steps
                if step.status in {"planned", "ready"} and all(dep in done for dep in step.dependencies)]


class PlanBuilder:
    """Create bounded plans for explicitly complex or multi-objective work."""

    def build(
        self,
        goal: str,
        scope: Mapping[str, Any] | None = None,
        available_capabilities: Sequence[str] = (),
        *,
        explicit: bool = False,
        max_steps: int = 6,
        deadline_at: float | None = None,
    ) -> Plan | None:
        text = _clean_text(goal)
        capabilities = [str(item) for item in available_capabilities]
        complexity_markers = ("并且", "同时", "然后", "研究", "整理", "compare", "and", "then")
        complex_goal = explicit or len(text) > 180 or any(marker in text.lower() for marker in complexity_markers)
        if not complex_goal:
            return None
        requested = set(capabilities)
        steps: list[PlanStep] = []
        if "retrieve" in requested or not requested:
            steps.append(PlanStep("retrieve", "收集当前 scope 内的相关资料", allowed_tools=["rag_retrieve"], expected_outputs=["candidates"]))
        if "analyze" in requested or "assess" in requested:
            deps = [steps[-1].step_id] if steps else []
            steps.append(PlanStep("analyze", "分析资料并保留可核验证据", dependencies=deps, allowed_tools=["rag_assess"], expected_outputs=["assessment"]))
        if "write" in requested:
            deps = [steps[-1].step_id] if steps else []
            steps.append(PlanStep("write", "根据已核验资料生成草稿", dependencies=deps, allowed_tools=["draft"], expected_outputs=["artifact"]))
        if not steps:
            steps = [PlanStep("research", text, expected_outputs=["evidence"])]
        steps = steps[:max(1, int(max_steps))]
        return Plan(
            plan_id=f"plan-{uuid.uuid4().hex[:16]}", goal=text, scope=dict(scope or {}),
            deadline_at=deadline_at, steps=steps,
            provenance={"mode": "shadow", "builder": "deterministic-v1"},
        )


StepRunner = Callable[[PlanStep], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]
Checkpoint = Callable[[Plan], Any]


class PlanExecutor:
    """Execute ready steps with bounded attempts and fail-closed evidence checks."""

    def __init__(self, *, max_rounds_per_step: int = 3, step_timeout_seconds: float = 30.0):
        self.max_rounds_per_step = max(1, int(max_rounds_per_step))
        self.step_timeout_seconds = max(0.01, float(step_timeout_seconds))

    @staticmethod
    def _has_evidence(result: Mapping[str, Any]) -> bool:
        evidence = result.get("evidence_refs") or result.get("evidence") or result.get("artifact_refs")
        return bool(evidence)

    async def execute(
        self,
        plan: Plan,
        runner: StepRunner,
        *,
        checkpoint: Checkpoint | None = None,
        cancel_event: asyncio.Event | None = None,
        deadline_at: float | None = None,
    ) -> Plan:
        plan.status = "running"
        effective_deadline = deadline_at or plan.deadline_at
        if checkpoint:
            checkpoint(plan)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                plan.status = "cancelled"
                for step in plan.steps:
                    if step.status in {"planned", "ready", "running"}:
                        step.status = "cancelled"
                break
            ready = plan.ready_steps()
            if not ready:
                if all(step.status == "succeeded" for step in plan.steps):
                    plan.status = "succeeded"
                elif any(step.status in {"blocked", "failed", "needs_review"} for step in plan.steps):
                    plan.status = "blocked"
                break
            step = ready[0]
            step.status = "running"
            step.attempts += 1
            if effective_deadline is not None and time.monotonic() >= effective_deadline:
                step.status = "blocked"
                step.result = {"error": "deadline"}
                plan.status = "blocked"
                break
            try:
                remaining = self.step_timeout_seconds
                if effective_deadline is not None:
                    remaining = min(remaining, max(0.01, effective_deadline - time.monotonic()))
                value = runner(step)
                if inspect.isawaitable(value):
                    value = await asyncio.wait_for(value, timeout=remaining)
                result = dict(value or {})
                step.result = result
                step.evidence_refs = [str(item) for item in (result.get("evidence_refs") or result.get("artifact_refs") or [])]
                if not self._has_evidence(result):
                    step.status = "needs_review"
                else:
                    step.status = "succeeded"
            except asyncio.TimeoutError:
                step.status = "blocked"
                step.result = {"error": "step_timeout"}
            except asyncio.CancelledError:
                step.status = "cancelled"
                raise
            except Exception as exc:  # noqa: BLE001 - step failure is persisted for recovery
                step.status = "failed"
                step.result = {"error": type(exc).__name__, "detail": str(exc)[:200]}
            if checkpoint:
                checkpoint(plan)
            if step.status != "succeeded":
                plan.status = "blocked"
                break
        if checkpoint:
            checkpoint(plan)
        return plan


__all__ = ["PLAN_STATUSES", "STEP_STATUSES", "PlanStep", "Plan", "PlanBuilder", "PlanExecutor"]
