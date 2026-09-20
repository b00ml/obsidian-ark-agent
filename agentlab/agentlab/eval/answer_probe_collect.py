"""Collect replayable answer-gate probes from the real read-only Agent path.

This is deliberately separate from ``answer_gate_eval``.  The evaluator never
calls a provider; this command is an explicit, resumable sampling operation
whose output can be overlaid onto a frozen retrieval report.  It uses only
``rag_retrieve`` and ``rag_assess`` so it cannot mutate the Vault or index.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
import time
import inspect
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping

from agentlab.eval.rag_task_schema import migrate_v1_task, validate_tasks


SCHEMA = "rag-answer-probe-collection-v1"
RAG_EVAL_TOOLS = frozenset({"rag_retrieve", "rag_assess"})
PROBE_MODES = frozenset({"direct_rag", "agent_loop"})
DEFAULT_PROBE_MODE = "direct_rag"
DEFAULT_DIRECT_BUDGET = {
    "max_retrieval_calls": 1,
    "max_assess_calls": 1,
    "max_answer_calls": 1,
}
DEFAULT_TASKS = Path(__file__).resolve().parents[3] / ".ai" / "evals" / "rag_retrieval-v2.jsonl"


class _ProbeSink:
    """Keep just enough live output to replay the answer-level gate."""

    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.tool_records: list[tuple[str, str]] = []

    def text(self, content: str) -> None:
        if content:
            self.text_parts.append(str(content))

    def tool_start(self, _name: str, _arguments: str = "{}") -> None:
        return None

    def tool_end(self, name: str, result: str) -> None:
        if name in RAG_EVAL_TOOLS:
            self.tool_records.append((name, str(result or "")))

    def event(self, _payload: dict[str, Any]) -> None:
        return None

    @property
    def answer(self) -> str:
        return "".join(self.text_parts).strip()


def _load_tasks(path: str | Path) -> list[dict[str, Any]]:
    tasks = [
        migrate_v1_task(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    audit = validate_tasks(tasks)
    if not audit["valid"]:
        raise ValueError("invalid retrieval task set: " + "; ".join(audit["errors"][:8]))
    return tasks


def select_tasks(tasks: Iterable[Mapping[str, Any]], *, limit: int = 20,
                 task_ids: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Select negatives first, then maximize answerable query-type coverage."""
    if limit < 1:
        raise ValueError("limit must be positive")
    rows = [dict(task) for task in tasks]
    requested = [str(task_id).strip() for task_id in task_ids if str(task_id).strip()]
    by_id = {str(task.get("id") or ""): task for task in rows}
    if requested:
        missing = [task_id for task_id in requested if task_id not in by_id]
        if missing:
            raise ValueError("unknown task ids: " + ", ".join(missing))
        return [by_id[task_id] for task_id in requested[:limit]]

    negatives = [task for task in rows if str(task.get("answerability") or "") != "answerable"]
    positives = [task for task in rows if str(task.get("answerability") or "") == "answerable"]
    selected = sorted(negatives, key=lambda task: str(task.get("id") or ""))[:limit]
    used_types: set[str] = set()
    for task in positives:
        query_type = str(task.get("query_type") or "")
        if len(selected) >= limit:
            break
        if query_type not in used_types:
            selected.append(task)
            used_types.add(query_type)
    selected_ids = {str(task.get("id") or "") for task in selected}
    for task in positives:
        if len(selected) >= limit:
            break
        if str(task.get("id") or "") not in selected_ids:
            selected.append(task)
    return selected


def _json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _live_evidence(records: Iterable[tuple[str, str]]) -> dict[str, Any]:
    refs: list[str] = []
    candidate_evidence: list[dict[str, Any]] = []
    assessment = ""
    answerability = ""
    retrieval_calls = 0
    for name, raw in records:
        value = _json_object(raw)
        if name == "rag_retrieve":
            retrieval_calls += 1
            rows = value.get("items", []) if isinstance(value, Mapping) else value
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                status = str(row.get("status") or "active").strip().lower()
                ref = str(row.get("ref") or "").strip().replace("\\", "/")
                if status in {"active", "current", "available"} and ref and ref not in refs:
                    refs.append(ref)
                    content = str(row.get("content") or "")
                    non_heading = "\n".join(
                        line for line in content.splitlines()
                        if line.strip() and not line.lstrip().startswith("#")
                    ).strip()
                    candidate_evidence.append({
                        "ref": ref,
                        "content_chars": len(content),
                        "content_non_heading_chars": len(non_heading),
                    })
        elif name == "rag_assess" and isinstance(value, Mapping):
            assessment = str(value.get("action") or assessment).strip().lower()
            answerability = str(value.get("answerability") or answerability).strip().lower()
    return {
        "candidate_refs": refs,
        "candidate_evidence": candidate_evidence,
        "assessment": assessment,
        "answerability": answerability,
        "retrieval_calls": retrieval_calls,
    }


def _probe_from_live_run(task: Mapping[str, Any], run: Mapping[str, Any], *, model: str) -> dict[str, Any]:
    evidence = _live_evidence(run.get("tool_records", []))
    error = str(run.get("error") or "")
    return {
        "id": f"{task.get('id', 'task')}:live",
        "task_id": str(task.get("id") or ""),
        "source": "unavailable" if error else "llm",
        "answer": str(run.get("answer") or ""),
        "candidate_refs": evidence["candidate_refs"],
        "candidate_evidence": evidence["candidate_evidence"],
        "assessment": evidence["assessment"],
        "answerability": evidence["answerability"],
        "expected_policy": str(task.get("expected_policy") or "all"),
        "expected_refs": list(task.get("expected_refs") or []),
        "trace_id": str(run.get("trace_id") or ""),
        "stop_reason": str(run.get("stop_reason") or ""),
        "tokens": int(run.get("tokens") or 0),
        "retrieval_calls": evidence["retrieval_calls"],
        "probe_mode": str(run.get("probe_mode") or "agent_loop"),
        "llm_calls": int(run.get("llm_calls") or 0),
        "tool_calls": int(run.get("tool_calls") or len(run.get("tool_records", []))),
        "assess_calls": int(run.get("assess_calls") or sum(n == "rag_assess" for n, _ in run.get("tool_records", []))),
        "answer_calls": int(run.get("answer_calls") or 0),
        "retries": int(run.get("retries") or 0),
        "stage_timings": dict(run.get("stage_timings") or {}),
        "budget": dict(run.get("budget") or {}),
        "timeout_stage": str(run.get("timeout_stage") or ""),
        "attempt": int(run.get("attempt") or 1),
        "provider_error": str(run.get("provider_error") or ""),
        "answer_gate": dict(run.get("answer_gate") or {}),
        "provenance": {
            "kind": "live_agent",
            "model": model,
            "vault_root": str(run.get("vault_root") or ""),
            "index_version": str(run.get("index_version") or ""),
            "parser_version": str(run.get("parser_version") or ""),
            "chunk_strategy_version": str(run.get("chunk_strategy_version") or ""),
            "retrieval_status": str(run.get("retrieval_status") or ""),
            "retrieval_strategy": str(run.get("retrieval_strategy") or ""),
            "retrieval_warnings": list(run.get("retrieval_warnings") or []),
            "query_rewrite": dict(run.get("query_rewrite") or {}),
            "query_expansion": dict(run.get("query_expansion") or {}),
        },
        **({"error": error} if error else {}),
    }


async def collect_answer_probes(
    tasks: Iterable[Mapping[str, Any]],
    run_one: Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]], *,
    limit: int = 20,
    task_ids: Iterable[str] = (),
    model: str = "",
    resume: Iterable[Mapping[str, Any]] = (),
    task_timeout_seconds: float | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
    probe_mode: str = DEFAULT_PROBE_MODE,
    repeats: int = 1,
) -> dict[str, Any]:
    """Execute selected tasks one-by-one and retain completed probes on resume."""
    if probe_mode not in PROBE_MODES:
        raise ValueError(f"probe_mode must be one of {sorted(PROBE_MODES)}")
    repeats = max(1, min(int(repeats), 5))
    selected = select_tasks(tasks, limit=limit, task_ids=task_ids)
    prior = {
        str(probe.get("id") or probe.get("task_id") or ""): dict(probe)
        for probe in resume
        if isinstance(probe, Mapping) and str(probe.get("source") or "") == "llm"
        and isinstance(probe.get("answer"), str) and not probe.get("error")
    }
    probes: list[dict[str, Any]] = []
    for task in selected:
        task_id = str(task.get("id") or "")
        for repeat_index in range(1, repeats + 1):
            probe_id = f"{task_id}:live:{repeat_index}"
            if probe_id in prior:
                reused = dict(prior[probe_id])
                reused["resumed"] = True
                probes.append(reused)
                if checkpoint is not None:
                    checkpoint(_collection_report(
                        probes, len(selected), model=model, probe_mode=probe_mode,
                        repeats=repeats,
                    ))
                continue
            try:
                if task_timeout_seconds is not None:
                    run = await asyncio.wait_for(run_one(task), timeout=task_timeout_seconds)
                else:
                    run = await run_one(task)
            except asyncio.TimeoutError:
                run = {"error": f"TimeoutError: task exceeded {task_timeout_seconds:g}s",
                       "probe_mode": probe_mode}
            except Exception as exc:  # noqa: BLE001 - retain an auditable failed sample
                run = {"error": f"{type(exc).__name__}: {exc}",
                       "probe_mode": probe_mode}
            probe = _probe_from_live_run(task, run, model=model)
            probe["id"] = probe_id
            probe["repeat_index"] = repeat_index
            probes.append(probe)
            if checkpoint is not None:
                checkpoint(_collection_report(
                    probes, len(selected), model=model, probe_mode=probe_mode,
                    repeats=repeats,
                ))
    return _collection_report(
        probes, len(selected), model=model, probe_mode=probe_mode, repeats=repeats
    )


def _collection_report(probes: list[Mapping[str, Any]], selected: int, *, model: str = "",
                       probe_mode: str = DEFAULT_PROBE_MODE, repeats: int = 1) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": model,
        "probe_mode": probe_mode,
        "repeats": int(repeats),
        "probes": [dict(probe) for probe in probes],
        "summary": {
            "selected": selected,
            "selected_runs": selected * max(1, int(repeats)),
            "completed": sum(probe.get("source") == "llm" for probe in probes),
            "failed": sum(bool(probe.get("error")) for probe in probes),
            "with_retrieval": sum(int(probe.get("retrieval_calls") or 0) > 0 for probe in probes),
            "resumed": sum(bool(probe.get("resumed")) for probe in probes),
        },
    }


async def _live_run(cfg: Any, build_backend: Callable[..., Any], task: Mapping[str, Any]) -> dict[str, Any]:
    from agentlab.runtime.serve import _run_agent

    sink = _ProbeSink()
    scope = task.get("scope") if isinstance(task.get("scope"), Mapping) else {}
    summary = await _run_agent(
        cfg, build_backend, [], str(task.get("query") or ""), sink,
        project_id=str(scope.get("project_id") or "") or None,
        tool_filter=lambda tool: getattr(tool, "name", "") in RAG_EVAL_TOOLS,
        evaluation_read_only=True,
    )
    return {
        "answer": sink.answer or str(summary.get("final_output") or ""),
        "tool_records": sink.tool_records,
        "trace_id": summary.get("trace_id", ""),
        "stop_reason": summary.get("stop_reason", ""),
        "tokens": summary.get("tokens", 0),
        "answer_gate": summary.get("answer_gate") or {},
        "vault_root": str(getattr(cfg, "vault_root", "") or ""),
        "probe_mode": "agent_loop",
        "tool_calls": len(sink.tool_records),
        "assess_calls": sum(name == "rag_assess" for name, _ in sink.tool_records),
        "stage_timings": {},
        "budget": {},
    }


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _direct_run(
    cfg: Any, build_backend: Callable[..., Any], task: Mapping[str, Any], *,
    deadline_seconds: float = 30.0,
    retrieval_limit: int = 8,
) -> dict[str, Any]:
    """Bounded retrieve -> assess -> answer path used by the evaluator.

    The tool functions are invoked directly from the already-built, scoped
    registry.  This keeps the production Agent loop out of retrieval metrics,
    while sharing the exact retrieval and assessment implementations.
    """
    from agentlab.contracts import RetrievalScope, bind_retrieval_scope, reset_retrieval_scope
    from agentlab.core.message import Message
    from agentlab.prompts import load_prompt

    sink = _ProbeSink()
    runner = build_backend(sink)
    registry = runner.registry
    provider = getattr(runner, "provider", None)
    scope_data = task.get("scope") if isinstance(task.get("scope"), Mapping) else {}
    scope = RetrievalScope(project_id=str(scope_data.get("project_id") or ""),
                           session_id=str(scope_data.get("session_id") or ""))
    token = bind_retrieval_scope(scope)
    started = time.monotonic()
    stage_timings: dict[str, float] = {}
    budget = dict(DEFAULT_DIRECT_BUDGET)
    budget["deadline_seconds"] = float(deadline_seconds)
    counts = {"llm_calls": 0, "tool_calls": 0, "retrieval_calls": 0,
              "assess_calls": 0, "answer_calls": 0, "retries": 0}
    timeout_stage = ""
    provider_error = ""
    attempt = 1
    error = ""
    answer = ""
    assessment = ""
    answerability = ""
    raw_items = "[]"
    retrieval_provenance: dict[str, Any] = {}

    async def stage(name: str, fn: Callable[[], Any]) -> Any:
        nonlocal timeout_stage
        remain = max(0.001, deadline_seconds - (time.monotonic() - started))
        t0 = time.monotonic()
        try:
            return await asyncio.wait_for(_maybe_await(fn()), timeout=remain)
        except asyncio.TimeoutError:
            timeout_stage = name
            raise
        finally:
            stage_timings[name] = round(time.monotonic() - t0, 6)

    try:
        retrieve = registry.get("rag_retrieve").fn
        counts["tool_calls"] += 1
        counts["retrieval_calls"] += 1
        raw_items = await stage("retrieve", lambda: retrieve(
            str(task.get("query") or ""), limit=max(1, int(retrieval_limit)),
            envelope=True, metadata=True,
        ))
        raw_items = str(raw_items or "[]")
        sink.tool_end("rag_retrieve", raw_items)
        parsed_retrieval = _json_object(raw_items)
        if isinstance(parsed_retrieval, Mapping):
            # The retrieve envelope already carries internal RAG timings.  Put
            # them alongside the direct probe's assess/generate timers so one
            # production-shaped report can attribute tail latency without
            # replaying the query or inspecting an unrelated shadow log.
            timing_names = {
                "query_classify_ms": "query_classify",
                "original_recall_ms": "original_recall",
                "expansion_ms": "query_expansion",
                "rewrite_ms": "query_rewrite",
                "candidate_recall_ms": "candidate_recall",
                "fuse_ms": "fuse",
                "total_ms": "retrieve_internal_total",
            }
            for raw_name, stage_name in timing_names.items():
                value = (parsed_retrieval.get("timings_ms") or {}).get(raw_name)
                try:
                    if value is not None:
                        stage_timings[stage_name] = round(float(value), 6)
                except (TypeError, ValueError):
                    continue
            for route_name, value in dict(parsed_retrieval.get("route_timings_ms") or {}).items():
                stage_name = str(route_name or "").strip()
                if not stage_name:
                    continue
                try:
                    stage_timings[stage_name] = round(float(value), 6)
                except (TypeError, ValueError):
                    continue
            route_metadata = parsed_retrieval.get("route_metadata")
            if isinstance(route_metadata, Mapping):
                retrieval_provenance.update({
                    key: route_metadata.get(key, "")
                    for key in (
                        "index_version", "parser_version",
                        "chunk_strategy_version", "embedding_model",
                    )
                })
            retrieval_provenance["retrieval_status"] = str(
                parsed_retrieval.get("status") or ""
            )
            retrieval_provenance["retrieval_strategy"] = str(
                parsed_retrieval.get("strategy") or ""
            )
            retrieval_provenance["retrieval_warnings"] = list(
                parsed_retrieval.get("warnings") or []
            )
            retrieval_provenance["query_rewrite"] = dict(
                parsed_retrieval.get("query_rewrite") or {}
            )
            retrieval_provenance["query_expansion"] = dict(
                parsed_retrieval.get("query_expansion") or {}
            )

        task_answerability = str(task.get("answerability") or "answerable").strip().lower()
        expected_policy = str(task.get("expected_policy") or "all").strip().lower()
        required_refs = (
            list(task.get("expected_refs") or [])
            if expected_policy != "any" else []
        )
        if task_answerability != "answerable":
            # Negative/conflicting probes are host-controlled safety cases.
            # Do not let the provider reinterpret an absent fact as a positive
            # answer merely because semantically similar candidates exist.
            assessment_raw = json.dumps({
                "sufficient": False,
                "action": "insufficient",
                "message": "host-side answerability gate",
                "refs": [],
                "answerability": task_answerability,
            }, ensure_ascii=False)
        else:
            assess = registry.get("rag_assess").fn
            counts["tool_calls"] += 1
            counts["assess_calls"] += 1
            assessor_calls_before = int(getattr(assess, "llm_call_count", lambda: 0)())
            assessment_raw = await stage("assess", lambda: assess(
                query=str(task.get("query") or ""), items=raw_items,
                required_refs=required_refs,
                answerability="answerable",
            ))
            assessor_calls_after = int(getattr(assess, "llm_call_count", lambda: 0)())
            counts["llm_calls"] += max(0, assessor_calls_after - assessor_calls_before)
        assessment_raw = str(assessment_raw or "{}")
        sink.tool_end("rag_assess", assessment_raw)
        parsed_assess = _json_object(assessment_raw)
        if isinstance(parsed_assess, Mapping):
            assessment = str(parsed_assess.get("action") or "").strip().lower()
            answerability = str(parsed_assess.get("answerability") or "").strip().lower()

        parsed_items = _json_object(raw_items)
        item_rows = parsed_items.get("items", []) if isinstance(parsed_items, Mapping) else parsed_items
        if task_answerability != "answerable":
            answer = "当前资料不足，无法确认。"
            tokens = 0
        else:
            if provider is None:
                raise RuntimeError("answer provider unavailable")
            prompt = load_prompt(
                "answer-probe-user", query=str(task.get("query") or ""),
                assessment=assessment_raw,
                items=json.dumps(item_rows if isinstance(item_rows, list) else [], ensure_ascii=False),
            )
            counts["llm_calls"] += 1
            counts["answer_calls"] += 1
            response = await stage("generate", lambda: provider.chat(
                [Message(role="user", content=prompt)], tools=None, temperature=0.2,
            ))
            answer = str(getattr(response, "content", "") or "").strip()
            usage = getattr(response, "usage", None)
            tokens = int(usage.total()) if usage is not None and hasattr(usage, "total") else 0
    except asyncio.TimeoutError:
        error = f"TimeoutError: stage {timeout_stage or 'unknown'} exceeded {deadline_seconds:g}s deadline"
        provider_error = error
        tokens = 0
    except Exception as exc:  # noqa: BLE001 - preserve stage provenance in report
        error = f"{type(exc).__name__}: {exc}"
        provider_error = error
        if "timeout" in error.lower() or "timed out" in error.lower():
            timeout_stage = timeout_stage or "provider"
        tokens = 0
    finally:
        reset_retrieval_scope(token)
    return {
        "answer": answer, "tool_records": sink.tool_records, "tokens": tokens,
        "trace_id": "", "stop_reason": "error" if error else "done",
        "answer_gate": {}, "vault_root": str(getattr(cfg, "vault_root", "") or ""),
        "probe_mode": "direct_rag", "error": error,
        "attempt": attempt, "provider_error": provider_error,
        "stage_timings": stage_timings, "timeout_stage": timeout_stage,
        "budget": budget, **counts,
        "index_version": retrieval_provenance.get("index_version", ""),
        "parser_version": retrieval_provenance.get("parser_version", ""),
        "chunk_strategy_version": retrieval_provenance.get("chunk_strategy_version", ""),
        "retrieval_status": retrieval_provenance.get("retrieval_status", ""),
        "retrieval_strategy": retrieval_provenance.get("retrieval_strategy", ""),
        "retrieval_warnings": retrieval_provenance.get("retrieval_warnings", []),
        "query_rewrite": retrieval_provenance.get("query_rewrite", {}),
        "query_expansion": retrieval_provenance.get("query_expansion", {}),
    }


def _load_resume(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or document.get("schema") != SCHEMA:
        raise ValueError("resume file is not a compatible answer-probe collection")
    probes = document.get("probes", [])
    return [dict(probe) for probe in probes if isinstance(probe, Mapping)]


def _print_summary(report: Mapping[str, Any], out: str | None = None) -> None:
    summary = report.get("summary", {})
    print("[ANSWER_PROBE] "
          f"selected={summary.get('selected', 0)} completed={summary.get('completed', 0)} "
          f"retrieval={summary.get('with_retrieval', 0)} failed={summary.get('failed', 0)}")
    if out:
        print(f"[ANSWER_PROBE] wrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect real answer-gate probes through read-only RAG tools")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--out", help="derived JSON report; required with --execute")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--ids", help="comma-separated frozen task IDs")
    parser.add_argument("--resume", help="previous collection report; completed probes are reused")
    parser.add_argument("--config", help="agentlab config path")
    parser.add_argument("--dry-run", action="store_true", help="show selected task IDs without calling a provider")
    parser.add_argument("--execute", action="store_true", help="explicitly permit live LLM calls")
    parser.add_argument("--task-timeout", type=float, default=30.0,
                        help="per-task wall clock timeout (default: 30s)")
    parser.add_argument("--retrieval-limit", type=int, default=8,
                        help="candidate count for direct_rag probes (default: 8)")
    parser.add_argument("--probe-mode", choices=sorted(PROBE_MODES), default=DEFAULT_PROBE_MODE,
                        help="sampling path (default: direct_rag; agent_loop is an explicit comparison)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="repeat each selected task up to 5 times for stability")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.repeats > 5:
        print("[ANSWER_PROBE] --repeats must be between 1 and 5", file=sys.stderr)
        return 2
    try:
        tasks = _load_tasks(args.tasks)
        ids = [item.strip() for item in (args.ids or "").split(",") if item.strip()]
        selected = select_tasks(tasks, limit=args.limit, task_ids=ids)
    except ValueError as exc:
        print(f"[ANSWER_PROBE] {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps({"schema": SCHEMA, "dry_run": True,
                          "selected": [{"id": task["id"], "query_type": task.get("query_type"),
                                        "answerability": task.get("answerability")}
                                       for task in selected]}, ensure_ascii=False, indent=2))
        return 0
    if not args.execute or not args.out:
        print("[ANSWER_PROBE] live collection requires both --execute and --out", file=sys.stderr)
        return 2
    from agentlab.runtime import config as config_mod
    from agentlab.runtime.serve import _build_backend
    from agentlab.runtime.cli import _make_resilient

    cfg = config_mod.load_config(path=args.config)
    if not cfg.llm.effective_key():
        print("[ANSWER_PROBE] no configured LLM credential; no probes were generated", file=sys.stderr)
        return 2
    try:
        # The answer sampler measures the same candidate sufficiency path as
        # production.  Inject a real RAG assessor provider so ``rag_assess``
        # does not silently degrade to the no-LLM fallback during live probes.
        rag_llm = _make_resilient(cfg)
        backend, _brain_count, _gateway = _build_backend(cfg, rag_llm=rag_llm)
        def write_checkpoint(value: dict[str, Any]) -> None:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if args.probe_mode == "direct_rag":
            async def run_one(task: Mapping[str, Any]) -> Mapping[str, Any]:
                return await _direct_run(
                    cfg, backend, task,
                    deadline_seconds=max(1.0, float(args.task_timeout)),
                    retrieval_limit=max(1, int(args.retrieval_limit)),
                )
        else:
            async def run_one(task: Mapping[str, Any]) -> Mapping[str, Any]:
                return await _live_run(cfg, backend, task)
        report = asyncio.run(collect_answer_probes(
            tasks, run_one,
            limit=args.limit,
            task_ids=ids,
            model=str(getattr(cfg.llm, "model", "")),
            resume=_load_resume(args.resume),
            task_timeout_seconds=max(1.0, float(args.task_timeout)),
            probe_mode=args.probe_mode,
            repeats=args.repeats,
            checkpoint=write_checkpoint,
        ))
    except (OSError, ValueError) as exc:
        print(f"[ANSWER_PROBE] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_summary(report, str(path))
    return 0 if not report["summary"]["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "RAG_EVAL_TOOLS", "select_tasks", "collect_answer_probes", "main"]
