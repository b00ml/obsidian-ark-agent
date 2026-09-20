"""Offline shadow/on comparison for ContextAssembler.

This evaluator compares only deterministic context plans.  It never changes
the production renderer, calls an LLM, or treats planner parity as proof that
tool calls, citations, refusals, or task completion are unchanged.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from agentlab.contracts import RetrievalScope
from agentlab.core.context_assembler import ContextAssembler, ContextPlan


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"case line {line_number} must be an object")
        if not str(value.get("id") or "").strip():
            raise ValueError(f"case line {line_number} is missing id")
        cases.append(value)
    return cases


def _plan(case: Mapping[str, Any], mode: str) -> ContextPlan:
    assembler = ContextAssembler(
        budget_tokens=int(case.get("budget_tokens", 8000)),
        reserve_output_tokens=int(case.get("reserve_output_tokens", 1000)),
        zone_budgets=case.get("zone_budgets") or {},
        mode=mode,
    )
    return assembler.assemble(
        scope=RetrievalScope.from_value(case.get("scope") or {}),
        task_state=case.get("task_state"),
        instructions=case.get("instructions") or [],
        history=case.get("history") or [],
        memory_candidates=case.get("memory_candidates") or [],
        retrieval_items=case.get("retrieval_items") or [],
        tool_observations=case.get("tool_observations") or [],
        mode=mode,
    )


def _metrics(plan: ContextPlan) -> dict[str, Any]:
    selected = list(plan.selected)
    omitted_reasons = Counter(item.get("reason", "") for item in plan.omitted)
    required = [item.id for item in selected if item.required]
    return {
        "mode": plan.mode,
        "scope": plan.scope.to_dict(),
        "selected_ids": [item.id for item in selected],
        "selected_refs": [item.ref for item in selected if item.ref],
        "required_ids": required,
        "used_tokens": plan.used_tokens,
        "omitted": len(plan.omitted),
        "omitted_reasons": dict(sorted(omitted_reasons.items())),
        "scope_denied": omitted_reasons.get("scope_denied", 0),
        "warnings": list(plan.warnings),
    }


def compare_context_plans(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        shadow = _metrics(_plan(case, "shadow"))
        on = _metrics(_plan(case, "on"))
        rows.append({
            "id": str(case.get("id") or ""),
            "shadow": shadow,
            "on": on,
            "diff": {
                "selected_same": shadow["selected_ids"] == on["selected_ids"],
                "selected_ref_same": shadow["selected_refs"] == on["selected_refs"],
                "required_same": shadow["required_ids"] == on["required_ids"],
                "scope_same": shadow["scope"] == on["scope"],
                "used_tokens_delta": on["used_tokens"] - shadow["used_tokens"],
                "omitted_delta": on["omitted"] - shadow["omitted"],
            },
        })

    def count(field: str) -> int:
        return sum(1 for row in rows if row["diff"][field])

    scope_denials = sum(row["on"]["scope_denied"] for row in rows)
    return {
        "schema": "context-assembler-shadow-on-v1",
        "cases": len(rows),
        "summary": {
            "selected_same": count("selected_same"),
            "selected_ref_same": count("selected_ref_same"),
            "required_same": count("required_same"),
            "scope_same": count("scope_same"),
            "scope_denied": scope_denials,
            "shadow_used_tokens": sum(row["shadow"]["used_tokens"] for row in rows),
            "on_used_tokens": sum(row["on"]["used_tokens"] for row in rows),
            "planner_regressions": sum(
                not row["diff"]["required_same"] or not row["diff"]["scope_same"]
                for row in rows
            ),
        },
        "runtime_metrics": {
            "tool_calls": "not_measured",
            "citations": "not_measured",
            "refusals": "not_measured",
            "task_completion": "not_measured",
            "reason": "planner-only comparison; use Runner/provider evaluation for runtime behavior",
        },
        "per_case": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare ContextAssembler shadow and on plans")
    default = Path(__file__).resolve().parents[3] / ".ai" / "evals" / "context_assembler-v1.jsonl"
    parser.add_argument("--cases", default=str(default))
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    report = compare_context_plans(load_cases(args.cases))
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["summary"]["planner_regressions"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
