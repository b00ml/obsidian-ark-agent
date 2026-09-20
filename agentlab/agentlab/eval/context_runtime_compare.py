"""Runner-level ContextAssembler shadow/on comparison.

Unlike :mod:`context_assembler_compare`, this module executes the fixed
Runner/provider path and records runtime outcomes.  It is an evidence
collector, not a rollout gate: callers must provide the provider they want to
measure and production configuration remains unchanged.
"""
from __future__ import annotations

import asyncio
import argparse
import datetime as dt
import hashlib
import json
import tempfile
from typing import Any, Callable, Mapping, Sequence
from pathlib import Path

from agentlab.contracts import bind_retrieval_scope, reset_retrieval_scope
from agentlab.core.agent import Agent
from agentlab.core.context_assembler import ContextAssembler
from agentlab.core.llm import LLMProvider, LLMResponse
from agentlab.core.loop import RunConfig, Runner
from agentlab.core.message import TokenUsage, ToolCall, ToolCallFunction
from agentlab.tools.registry import ToolRegistry


ProviderFactory = Callable[[Mapping[str, Any], str], Any]
RegistryFactory = Callable[[Mapping[str, Any], str], ToolRegistry]


def _stable_hash(value: Any) -> str:
    """Hash a potentially sensitive value without exporting its contents."""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()


def _provider_message_manifest(messages: Sequence[Any], tools: Any) -> dict[str, Any]:
    """Return the model-visible input shape without retaining prompt text."""
    rows = []
    for message in messages:
        calls = []
        for call in getattr(message, "tool_calls", None) or ():
            function = getattr(call, "function", None)
            calls.append({
                "id": str(getattr(call, "id", "") or ""),
                "name": str(getattr(function, "name", "") or ""),
                "arguments_hash": _stable_hash(str(getattr(function, "arguments", "") or "")),
            })
        rows.append({
            "role": str(getattr(message, "role", "") or ""),
            "content_hash": _stable_hash(str(getattr(message, "content", "") or "")),
            "tool_call_id": str(getattr(message, "tool_call_id", "") or ""),
            "tool_calls": calls,
        })
    tool_schema = tools if isinstance(tools, list) else []
    return {
        "message_hash": _stable_hash(rows),
        "messages": rows,
        "tool_schema_hash": _stable_hash(tool_schema),
    }


def _context_manifest(plan: Any) -> dict[str, Any]:
    """Return a stable, non-content explanation of the assembled prompt."""
    if plan is None:
        return {"selected": [], "omitted": [], "used_tokens": 0}
    selected = []
    for item in getattr(plan, "selected", ()):
        text = str(getattr(item, "text", "") or "")
        selected.append({
            "id": str(getattr(item, "id", "") or ""),
            "ref": str(getattr(item, "ref", "") or ""),
            "source": str(getattr(item, "source", "") or ""),
            "status": str(getattr(item, "status", "") or ""),
            "project_id": str(getattr(item, "project_id", "") or ""),
            "session_id": str(getattr(item, "session_id", "") or ""),
            "tokens": int(getattr(item, "tokens", 0) or 0),
            "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    omitted = [
        {"id": str(item.get("id") or ""), "reason": str(item.get("reason") or "")}
        for item in getattr(plan, "omitted", ()) if isinstance(item, Mapping)
    ]
    return {"selected": selected, "omitted": omitted,
            "used_tokens": int(getattr(plan, "used_tokens", 0) or 0)}


def _metrics(result, mode: str, *, plan: Any = None,
             plan_history: Sequence[Any] = ()) -> dict[str, Any]:
    trace = result.run_trace if isinstance(result.run_trace, dict) else {}
    runtime = trace.get("context_assembler") if isinstance(trace, dict) else {}
    runtime = dict(runtime) if isinstance(runtime, dict) else {}
    gate = result.answer_gate if isinstance(result.answer_gate, dict) else {}
    trace_events = trace.get("events", []) if isinstance(trace, dict) else []
    stage_durations: dict[str, list[float]] = {}
    for event in trace_events if isinstance(trace_events, list) else []:
        if not isinstance(event, Mapping):
            continue
        stage = str(event.get("stage") or "").strip()
        if not stage:
            continue
        try:
            duration = float(event.get("duration_ms"))
        except (TypeError, ValueError):
            continue
        stage_durations.setdefault(stage, []).append(round(max(0.0, duration), 3))
    runtime.update({
        "mode": mode,
        "tool_calls": int(runtime.get("tool_calls", 0) or 0),
        "citations": len(gate.get("allowed_refs", []) or []),
        "refusal": bool(gate.get("abstained", False)),
        "task_completion": result.stop_reason == "done",
        "stop_reason": result.stop_reason,
        "recovery_consistent": int(runtime.get("recovery_unknown", 0) or 0) == 0,
        "context_manifest": _context_manifest(plan),
        "context_plan_history": [_context_manifest(item) for item in plan_history],
        "stage_durations_ms": stage_durations,
        "stage_event_counts": {key: len(value) for key, value in stage_durations.items()},
    })
    return runtime


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return round(ordered[index], 3)


def _latency_summary(rows: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        metrics = row.get(mode) if isinstance(row, Mapping) else None
        timings = metrics.get("stage_durations_ms", {}) if isinstance(metrics, Mapping) else {}
        if not isinstance(timings, Mapping):
            continue
        for stage, values in timings.items():
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                grouped.setdefault(str(stage), []).extend(float(value) for value in values)
    return {
        stage: {
            "count": len(values),
            "p50_ms": _percentile(values, 0.50),
            "p95_ms": _percentile(values, 0.95),
            "total_ms": round(sum(values), 3),
        }
        for stage, values in sorted(grouped.items())
    }


async def compare_runner_cases_async(
    provider_factory: ProviderFactory,
    cases: Sequence[Mapping[str, Any]],
    *,
    registry_factory: RegistryFactory | None = None,
) -> dict[str, Any]:
    """Run each case twice with the same provider contract, shadow then on."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        per_mode: dict[str, dict[str, Any]] = {}
        for mode in ("shadow", "on"):
            registry = (registry_factory(case, mode)
                        if registry_factory is not None else ToolRegistry())
            provider = provider_factory(case, mode)
            runner = Runner(provider, registry)
            assembler = ContextAssembler(
                budget_tokens=int(case.get("budget_tokens", 8000) or 8000),
                reserve_output_tokens=int(case.get("reserve_output_tokens", 1000) or 0),
                zone_budgets=case.get("zone_budgets") or {}, mode=mode,
            )
            cfg = RunConfig(
                context_assembler=assembler,
                context_assembler_mode=mode,
                task_state=case.get("task_state"),
                answer_gate_mode=str(case.get("answer_gate_mode") or "shadow"),
                max_steps=int(case.get("max_steps", 15) or 15),
            )
            agent = Agent(
                name="context-runtime-compare",
                instructions=str(case.get("instructions_text") or ""),
                tools=registry.all(),
                max_steps=cfg.max_steps,
            )
            # The live server binds scope before it creates a Runner.  The
            # evaluator must do the same or a scope case silently becomes a
            # global-scope case and cannot prove the read gate.
            scope_token = bind_retrieval_scope(case.get("scope") or {})
            try:
                result = await runner.run(agent, str(case.get("input") or ""), cfg=cfg)
            finally:
                reset_retrieval_scope(scope_token)
            per_mode[mode] = _metrics(
                result, mode, plan=cfg.context_plan,
                plan_history=cfg.context_plan_history,
            )
        shadow, on = per_mode["shadow"], per_mode["on"]
        rows.append({
            "id": str(case.get("id") or ""),
            "shadow": shadow,
            "on": on,
            "diff": {
                "tool_calls_delta": on["tool_calls"] - shadow["tool_calls"],
                "citations_delta": on["citations"] - shadow["citations"],
                "refusal_changed": on["refusal"] != shadow["refusal"],
                "completion_changed": on["task_completion"] != shadow["task_completion"],
                "recovery_changed": on["recovery_consistent"] != shadow["recovery_consistent"],
                "selected_changed": (
                    on["context_manifest"]["selected"]
                    != shadow["context_manifest"]["selected"]
                ),
                "omitted_changed": (
                    on["context_manifest"]["omitted"]
                    != shadow["context_manifest"]["omitted"]
                ),
            },
        })
    return _report(rows)


def compare_runner_cases(provider_factory: ProviderFactory,
                         cases: Sequence[Mapping[str, Any]], *,
                         registry_factory: RegistryFactory | None = None) -> dict[str, Any]:
    """Synchronous wrapper for scripts and tests."""
    return asyncio.run(compare_runner_cases_async(
        provider_factory, cases, registry_factory=registry_factory
    ))


def _scripted_response(row: Mapping[str, Any]) -> LLMResponse:
    calls = []
    for index, raw in enumerate(row.get("tool_calls") or ()):
        if not isinstance(raw, Mapping):
            raise ValueError("replay tool call must be an object")
        arguments = raw.get("arguments", {})
        encoded = (json.dumps(arguments, ensure_ascii=False, sort_keys=True)
                   if isinstance(arguments, Mapping) else str(arguments))
        calls.append(ToolCall(
            id=str(raw.get("id") or f"replay-call-{index}"),
            function=ToolCallFunction(
                name=str(raw.get("name") or ""), arguments=encoded,
            ),
        ))
    usage = row.get("usage") if isinstance(row.get("usage"), Mapping) else {}
    return LLMResponse(
        content=str(row.get("content") or ""),
        tool_calls=calls,
        stop_reason=str(row.get("stop_reason") or ("tool_calls" if calls else "stop")),
        usage=TokenUsage(
            input_tokens=int(usage.get("input_tokens", 1) or 0),
            output_tokens=int(usage.get("output_tokens", 1) or 0),
        ),
    )


class _ReplayProvider(LLMProvider):
    """Fixed response trace with a non-content record of every model input."""

    def __init__(self, responses: Sequence[Mapping[str, Any]]) -> None:
        self._responses = [dict(row) for row in responses]
        self.requests: list[dict[str, Any]] = []
        self.after_request: Callable[[int], None] | None = None
        self.fail_on_request: int = 0

    async def chat(self, messages, tools=None, **_kwargs):
        self.requests.append(_provider_message_manifest(messages, tools))
        if self.fail_on_request and len(self.requests) == self.fail_on_request:
            raise RuntimeError("replay process crash")
        if not self._responses:
            raise RuntimeError("replay provider exhausted before runner completed")
        response = _scripted_response(self._responses.pop(0))
        if self.after_request is not None:
            self.after_request(len(self.requests))
        return response


def _scope_visible_rows(rows: Sequence[Any]) -> list[dict[str, Any]]:
    """Apply the same request scope boundary used by read-only fixture tools."""
    from agentlab.contracts import current_retrieval_scope

    scope = current_retrieval_scope()
    visible = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        project_id = str(raw.get("project_id") or "")
        session_id = str(raw.get("session_id") or "")
        if scope.project_id and project_id and project_id not in {scope.project_id, "default"}:
            continue
        if scope.session_id and session_id and session_id != scope.session_id:
            continue
        visible.append(dict(raw))
    return visible


def _checkpoint_manifest(store: Any, task_id: str) -> dict[str, Any]:
    state = store.get(task_id) if store is not None and task_id else None
    if state is None:
        return {"present": False}
    pending = []
    for row in getattr(state, "pending_tools", ()):
        if not isinstance(row, Mapping):
            continue
        pending.append({
            "operation_id": str(row.get("operation_id") or ""),
            "tool_name": str(row.get("tool_name") or ""),
            "arguments_hash": str(row.get("arguments_hash") or ""),
            "status": str(row.get("status") or ""),
        })
    return {
        "present": True,
        "task_id": str(getattr(state, "task_id", "") or ""),
        "project_id": str(getattr(state, "project_id", "") or ""),
        "session_id": str(getattr(state, "session_id", "") or ""),
        "state_version": int(getattr(state, "state_version", 0) or 0),
        "phase": str(getattr(state, "phase", "") or ""),
        "pending_tools": pending,
    }


def _replay_registry(case: Mapping[str, Any], invocations: list[dict[str, Any]]) -> ToolRegistry:
    """Build the bounded read-only registry used by deterministic drills."""
    from agentlab.tools.base import tool

    registry = ToolRegistry()
    source_rows = case.get("retrieval_items") or []
    static_output = str(case.get("tool_output") or "")

    @tool(name="rag_retrieve", description="Replay-only fixed read fixture.",
          permission="read", side_effects="none", idempotent=True)
    def rag_retrieve(query: str, limit: int = 8) -> str:
        visible = _scope_visible_rows(source_rows)
        invocations.append({
            "name": "rag_retrieve",
            "arguments_hash": _stable_hash({"query": query, "limit": limit}),
            "visible_refs": [str(row.get("ref") or "") for row in visible],
        })
        if source_rows:
            return json.dumps({"items": visible[:max(1, min(int(limit), 8))]},
                              ensure_ascii=False, sort_keys=True)
        return static_output

    registry.register(rag_retrieve)
    return registry


async def _run_replay_mode(case: Mapping[str, Any], mode: str, *,
                           switch_to_shadow_after: int = 0) -> dict[str, Any]:
    """Run a case using one fixed provider/tool trace in an isolated ledger."""
    from agentlab.runtime.task_state import TaskStateStore
    script = case.get("response_script") or []
    if not isinstance(script, Sequence) or isinstance(script, (str, bytes)):
        raise ValueError("replay case response_script must be a list")
    provider = _ReplayProvider([row for row in script if isinstance(row, Mapping)])
    invocations: list[dict[str, Any]] = []
    registry = _replay_registry(case, invocations)
    with tempfile.TemporaryDirectory(prefix="context-runtime-replay-") as directory:
        store = TaskStateStore(Path(directory) / "task-state.db")
        task_id = str(case.get("task_state_id") or f"replay-{case.get('id') or 'case'}")
        task_state = case.get("task_state") if isinstance(case.get("task_state"), Mapping) else {}
        scope = case.get("scope") if isinstance(case.get("scope"), Mapping) else {}
        store.ensure(
            task_id,
            project_id=str(scope.get("project_id") or ""),
            session_id=str(scope.get("session_id") or ""),
            core_intent=dict(task_state.get("core_intent") or {}),
        )
        assembler = ContextAssembler(
            budget_tokens=int(case.get("budget_tokens", 8000) or 8000),
            reserve_output_tokens=int(case.get("reserve_output_tokens", 1000) or 0),
            zone_budgets=case.get("zone_budgets") or {}, mode=mode,
        )
        cfg = RunConfig(
            context_assembler=assembler,
            context_assembler_mode=mode,
            task_state=dict(task_state),
            task_state_store=store,
            task_state_id=task_id,
            answer_gate_mode=str(case.get("answer_gate_mode") or "shadow"),
            max_steps=int(case.get("max_steps", 15) or 15),
        )
        if switch_to_shadow_after > 0:
            def switch_mode(request_count: int) -> None:
                if request_count == switch_to_shadow_after:
                    cfg.context_assembler_mode = "shadow"
            provider.after_request = switch_mode
        agent = Agent(
            name="context-runtime-replay",
            instructions=str(case.get("instructions_text") or ""),
            tools=registry.all(), max_steps=cfg.max_steps,
        )
        scope_token = bind_retrieval_scope(scope)
        try:
            result = await Runner(provider, registry).run(
                agent, str(case.get("input") or ""), cfg=cfg,
            )
        finally:
            reset_retrieval_scope(scope_token)
        metrics = _metrics(
            result, mode, plan=cfg.context_plan,
            plan_history=cfg.context_plan_history,
        )
        metrics.update({
            "provider_requests": provider.requests,
            "tool_invocations": invocations,
            "checkpoint": _checkpoint_manifest(store, task_id),
            "context_mode_history": [
                str(getattr(item, "mode", "") or "")
                for item in cfg.context_plan_history
            ],
        })
        return metrics


async def _run_replay_resume_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Crash after a tool settles, then resume from the same durable ledger."""
    from agentlab.runtime.task_state import TaskStateStore

    script = [row for row in (case.get("response_script") or [])
              if isinstance(row, Mapping)]
    if not any(row.get("tool_calls") for row in script):
        return {
            "tool_contract_same": True,
            "duplicate_tool_calls": False,
            "checkpoint_before": None,
            "checkpoint_after": None,
            "tool_invocations": [],
        }
    with tempfile.TemporaryDirectory(prefix="context-runtime-resume-") as directory:
        store = TaskStateStore(Path(directory) / "task-state.db")
        task_id = str(case.get("task_state_id") or f"resume-{case.get('id') or 'case'}")
        scope = case.get("scope") if isinstance(case.get("scope"), Mapping) else {}
        task_state = case.get("task_state") if isinstance(case.get("task_state"), Mapping) else {}
        store.ensure(task_id, project_id=str(scope.get("project_id") or ""),
                    session_id=str(scope.get("session_id") or ""),
                    core_intent=dict(task_state.get("core_intent") or {}))
        first_invocations: list[dict[str, Any]] = []
        first_registry = _replay_registry(case, first_invocations)
        first_provider = _ReplayProvider(script)
        first_provider.fail_on_request = 2
        first_cfg = RunConfig(
            context_assembler=ContextAssembler(mode="on"),
            context_assembler_mode="on", task_state=dict(task_state),
            task_state_store=store, task_state_id=task_id, max_steps=15,
        )
        first_agent = Agent(name="context-runtime-resume",
                            instructions=str(case.get("instructions_text") or ""),
                            tools=first_registry.all(), max_steps=15)
        token = bind_retrieval_scope(scope)
        try:
            try:
                await Runner(first_provider, first_registry).run(
                    first_agent, str(case.get("input") or ""), cfg=first_cfg,
                )
            except RuntimeError as exc:
                if "replay process crash" not in str(exc):
                    raise
        finally:
            reset_retrieval_scope(token)
        checkpoint_before = _checkpoint_manifest(store, task_id)

        resumed_invocations: list[dict[str, Any]] = []
        resumed_registry = _replay_registry(case, resumed_invocations)
        resumed_provider = _ReplayProvider([{"content": "resume complete"}])
        resumed_cfg = RunConfig(
            context_assembler=ContextAssembler(mode="shadow"),
            context_assembler_mode="shadow", task_state=dict(task_state),
            task_state_store=store, task_state_id=task_id, max_steps=15,
        )
        resumed_agent = Agent(name="context-runtime-resume",
                              instructions=str(case.get("instructions_text") or ""),
                              tools=resumed_registry.all(), max_steps=15)
        token = bind_retrieval_scope(scope)
        try:
            await Runner(resumed_provider, resumed_registry).run(
                resumed_agent, str(case.get("input") or ""), cfg=resumed_cfg,
            )
        finally:
            reset_retrieval_scope(token)
        checkpoint_after = _checkpoint_manifest(store, task_id)
        return {
            "tool_contract_same": bool(first_invocations) and not resumed_invocations,
            "duplicate_tool_calls": bool(resumed_invocations),
            "checkpoint_before": checkpoint_before,
            "checkpoint_after": checkpoint_after,
            "tool_invocations": first_invocations,
        }


async def compare_replayed_runner_cases_async(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Attribute shadow/on differences with identical scripted observations.

    This is deliberately diagnostic-only: it proves whether the Runner and
    assembler diverge for the same inputs, but never stands in for live
    provider evidence or a production rollout gate.
    """
    rows = []
    for case in cases:
        shadow = await _run_replay_mode(case, "shadow")
        on = await _run_replay_mode(case, "on")
        diff_fields = (
            "context_plan_history", "provider_requests", "tool_invocations", "checkpoint",
            "task_completion", "stop_reason", "refusal",
        )
        rows.append({
            "id": str(case.get("id") or ""),
            "shadow": shadow,
            "on": on,
            "diff": {name: shadow.get(name) != on.get(name) for name in diff_fields},
        })
    regressions = sum(any(row["diff"].values()) for row in rows)
    return {
        "schema": "context-assembler-deterministic-replay-v1",
        "evidence_status": "deterministic_replay",
        "production_evidence": False,
        "cases": len(rows),
        "summary": {
            "replay_regression_cases": regressions,
            "plan_history_changes": sum(row["diff"]["context_plan_history"] for row in rows),
            "renderer_input_changes": sum(row["diff"]["provider_requests"] for row in rows),
            "tool_contract_changes": sum(row["diff"]["tool_invocations"] for row in rows),
            "checkpoint_changes": sum(row["diff"]["checkpoint"] for row in rows),
        },
        "per_case": rows,
    }


async def run_context_fallback_drills_async(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Exercise both P2-03 rollback paths on the fixed replay contract."""
    if not cases:
        return {
            "schema": "context-assembler-fallback-drill-v1",
            "evidence_status": "deterministic_replay",
            "production_evidence": False,
            "cases": 0,
            "summary": {"passed": 0, "failed": 0},
            "per_case": [],
        }
    rows = []
    for case in cases:
        # Run once with on -> shadow after the first model response.  The
        # fixed script makes the switch deterministic and observable.
        switched = await _run_replay_mode(case, "on", switch_to_shadow_after=1)

        # A real checkpoint resume drill is meaningful only for a tool case.
        # Use one durable store across the crash and resume halves.  The first
        # provider stops before the post-tool answer; the second provider sees
        # the same operation ledger and must not dispatch that operation again.
        resume = await _run_replay_resume_case(case)
        same_tools = switched.get("tool_invocations") == resume.get("tool_invocations")
        resume_checkpoint_stable = (
            resume.get("checkpoint_before") == resume.get("checkpoint_after")
        )
        checkpoint_compatible = (
            not switched.get("tool_invocations")
            or switched.get("checkpoint") == resume.get("checkpoint_before")
        )
        no_leak = all(
            "p2-secret" not in str(item.get("visible_refs") or [])
            for item in switched.get("tool_invocations", [])
        )
        rows.append({
            "id": str(case.get("id") or ""),
            "switch_to_shadow": {
                "mode_history": switched.get("context_mode_history", []),
                "tool_contract_same": same_tools,
                "checkpoint_same": checkpoint_compatible,
                "scope_safe": no_leak,
            },
            "resume_from_checkpoint": {
                "tool_contract_same": resume.get("tool_contract_same", False),
                "checkpoint_same": resume_checkpoint_stable,
                "duplicate_tool_calls": resume.get("duplicate_tool_calls", True),
                "scope_safe": no_leak,
            },
        })
    passed = sum(
        bool(row["switch_to_shadow"]["tool_contract_same"]
             and row["switch_to_shadow"]["checkpoint_same"]
             and row["switch_to_shadow"]["scope_safe"]
             and row["resume_from_checkpoint"]["tool_contract_same"]
             and row["resume_from_checkpoint"]["checkpoint_same"]
             and not row["resume_from_checkpoint"]["duplicate_tool_calls"])
        for row in rows
    )
    return {
        "schema": "context-assembler-fallback-drill-v1",
        "evidence_status": "deterministic_replay",
        "production_evidence": False,
        "cases": len(rows),
        "summary": {"passed": passed, "failed": len(rows) - passed},
        "per_case": rows,
    }


def run_context_fallback_drills(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return asyncio.run(run_context_fallback_drills_async(cases))


def compare_replayed_runner_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Synchronous deterministic replay wrapper for tests and CLI use."""
    return asyncio.run(compare_replayed_runner_cases_async(cases))


def _report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tool_call_changes = sum(
        row["diff"]["tool_calls_delta"] != 0 for row in rows
    )
    citation_changes = sum(
        row["diff"]["citations_delta"] != 0 for row in rows
    )
    refusal_changes = sum(row["diff"]["refusal_changed"] for row in rows)
    completion_changes = sum(row["diff"]["completion_changed"] for row in rows)
    recovery_changes = sum(row["diff"]["recovery_changed"] for row in rows)
    return {
        "schema": "context-assembler-runner-shadow-on-v1",
        "evidence_status": "provider_runtime",
        "cases": len(rows),
        "summary": {
            "tool_calls_shadow": sum(row["shadow"]["tool_calls"] for row in rows),
            "tool_calls_on": sum(row["on"]["tool_calls"] for row in rows),
            "citations_shadow": sum(row["shadow"]["citations"] for row in rows),
            "citations_on": sum(row["on"]["citations"] for row in rows),
            "tool_call_change_cases": tool_call_changes,
            "tool_calls_delta": sum(row["diff"]["tool_calls_delta"] for row in rows),
            "citation_change_cases": citation_changes,
            "refusal_changes": refusal_changes,
            "completion_changes": completion_changes,
            "recovery_changes": recovery_changes,
            "runtime_regressions": (
                tool_call_changes + citation_changes + refusal_changes
                + completion_changes + recovery_changes
            ),
        },
        "runtime_metrics": {
            "tool_calls": "measured",
            "citations": "measured",
            "refusals": "measured",
            "task_completion": "measured",
            "recovery": "measured",
            "stage_durations_ms": "measured from RunTrace events",
        },
        "latency": {
            "shadow": _latency_summary(rows, "shadow"),
            "on": _latency_summary(rows, "on"),
        },
        "per_case": rows,
    }


__all__ = ["compare_runner_cases", "compare_runner_cases_async"]


DEFAULT_CASES = Path(__file__).resolve().parents[3] / ".ai" / "evals" / "context_runtime-v1.jsonl"
DEFAULT_REPLAY_CASES = (Path(__file__).resolve().parents[3] / ".ai" / "evals"
                        / "context_runtime_replay-v1.jsonl")


def load_runtime_cases(path: str | Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not str(value.get("id") or "").strip():
            raise ValueError(f"case line {line_number} must be an object with id")
        cases.append(value)
    return cases


def collect_live_context_runtime(
    cfg: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    model: str = "",
) -> dict[str, Any]:
    """Run the explicit live provider comparison with read-only fixture tools."""
    from agentlab.runtime.cli import _resilient_for_model

    selected_model = model or str(getattr(cfg.llm, "model", ""))

    def provider_factory(_case: Mapping[str, Any], _mode: str):
        return _resilient_for_model(cfg, selected_model, max_tokens=cfg.llm.max_tokens)

    def registry_factory(case: Mapping[str, Any], _mode: str) -> ToolRegistry:
        from agentlab.tools.base import tool

        registry = ToolRegistry()
        rows = case.get("retrieval_items") or []

        @tool(name="rag_retrieve", description="只读检索固定测试资料，返回 JSON 引用。",
              permission="read", side_effects="none", idempotent=True)
        def rag_retrieve(query: str, limit: int = 8) -> str:
            del query
            # Preserve the production request boundary in the fixture path;
            # otherwise a scope case would feed an out-of-scope row directly
            # into the model and falsely blame ContextAssembler for it.
            visible = _scope_visible_rows(rows)
            return json.dumps({"items": visible[:max(1, min(int(limit), 8))]},
                              ensure_ascii=False)

        registry.register(rag_retrieve)
        return registry

    report = compare_runner_cases(provider_factory, cases, registry_factory=registry_factory)
    report.update({
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": selected_model,
        "provider": str(getattr(cfg.llm, "base_url", "")),
        "vault_manifest": str(getattr(cfg, "vault_root", "")),
    })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect live ContextAssembler shadow/on runtime evidence")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--replay-cases", default=str(DEFAULT_REPLAY_CASES))
    parser.add_argument("--config")
    parser.add_argument("--model", default="")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replay", action="store_true",
                        help="run fixed deterministic replay; never calls a provider")
    parser.add_argument("--fallback-drill", action="store_true",
                        help="run rollback/resume drills; never calls a provider")
    parser.add_argument("--execute", action="store_true", help="explicitly permit provider calls")
    args = parser.parse_args(argv)
    if args.fallback_drill:
        try:
            cases = load_runtime_cases(args.replay_cases)[:max(1, args.limit)]
            report = run_context_fallback_drills(cases)
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"schema": report["schema"], "cases": report["cases"],
                              "passed": report["summary"]["passed"],
                              "failed": report["summary"]["failed"],
                              "out": str(output)}, ensure_ascii=False))
            return 0 if report["summary"]["failed"] == 0 else 1
        except Exception as exc:  # noqa: BLE001 - bounded CLI diagnostics
            print(f"[CONTEXT_RUNTIME] {type(exc).__name__}: {exc}")
            return 1
    if args.replay:
        try:
            cases = load_runtime_cases(args.replay_cases)[:max(1, args.limit)]
            report = compare_replayed_runner_cases(cases)
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"schema": report["schema"], "cases": report["cases"],
                              "replay_regression_cases": report["summary"]["replay_regression_cases"],
                              "out": str(output)}, ensure_ascii=False))
            return 0 if report["summary"]["replay_regression_cases"] == 0 else 1
        except Exception as exc:  # noqa: BLE001 - bounded CLI diagnostics
            print(f"[CONTEXT_RUNTIME] {type(exc).__name__}: {exc}")
            return 1
    if not args.execute:
        print("[CONTEXT_RUNTIME] live collection requires --execute")
        return 2
    try:
        from agentlab.runtime.config import load_config
        cfg = load_config(path=args.config)
        if not cfg.llm.effective_key():
            print("[CONTEXT_RUNTIME] no configured LLM credential")
            return 2
        cases = load_runtime_cases(args.cases)[:max(1, args.limit)]
        report = collect_live_context_runtime(cfg, cases, model=args.model)
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"schema": report["schema"], "cases": report["cases"],
                          "runtime_regressions": report["summary"]["runtime_regressions"],
                          "out": str(output)}, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - report the live collection failure
        print(f"[CONTEXT_RUNTIME] {type(exc).__name__}: {exc}")
        return 1


__all__ += [
    "DEFAULT_CASES", "DEFAULT_REPLAY_CASES", "load_runtime_cases", "collect_live_context_runtime",
    "compare_replayed_runner_cases", "compare_replayed_runner_cases_async",
    "run_context_fallback_drills", "run_context_fallback_drills_async", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
