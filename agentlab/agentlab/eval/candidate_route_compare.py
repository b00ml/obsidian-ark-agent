"""Compare expected-reference coverage across frozen retrieval routes.

This report is intentionally narrower than the retrieval quality evaluator:
it explains *why* an answer-level probe missed evidence by keeping every
route's candidate refs side by side.  It never calls a provider or changes a
retrieval result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from agentlab.eval.answer_gate_eval import _ref_matches
from agentlab.eval.rag_task_schema import migrate_v1_task, validate_tasks


SCHEMA = "rag-candidate-route-compare-v1"
DEFAULT_TASKS = Path(__file__).resolve().parents[3] / ".ai" / "evals" / "rag_retrieval-v2.jsonl"
DEFAULT_ROUTES = ("p2_lexical", "p2_vector", "rrf")


def _refs(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    return list(dict.fromkeys(
        str(value).strip().replace("\\", "/")
        for value in values if str(value).strip()
    ))


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    tasks = [
        migrate_v1_task(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audit = validate_tasks(tasks)
    if not audit["valid"]:
        raise ValueError("invalid retrieval task set: " + "; ".join(audit["errors"][:8]))
    return tasks


def _route_coverage(task: Mapping[str, Any], row: Mapping[str, Any] | None,
                    route: str) -> dict[str, Any]:
    value = row.get(route) if isinstance(row, Mapping) else None
    expected = _refs(task.get("expected_refs"))
    if not isinstance(value, Mapping):
        return {
            "status": "missing",
            "refs": [],
            "expected_ref_hits": [],
            "any_expected_ref_hit": None if not expected else False,
            "all_expected_refs_hit": None if not expected else False,
        }
    refs = _refs(value.get("refs"))
    status = str(value.get("status") or "available").strip().lower()
    hits = [
        any(_ref_matches(gold, candidate) or _ref_matches(candidate, gold)
            for candidate in refs)
        for gold in expected
    ]
    return {
        "status": status,
        "refs": refs,
        "expected_ref_hits": hits,
        "any_expected_ref_hit": any(hits) if hits else None,
        "all_expected_refs_hit": all(hits) if hits else None,
    }


def _empty_route_summary() -> dict[str, int]:
    return {
        "positive_tasks": 0,
        "available_tasks": 0,
        "any_expected_ref_hit_tasks": 0,
        "all_expected_refs_hit_tasks": 0,
    }


def compare_candidate_routes(
    retrieval_report: Mapping[str, Any],
    tasks: Iterable[Mapping[str, Any]], *,
    routes: Iterable[str] = DEFAULT_ROUTES,
    task_ids: Iterable[str] = (),
    query_types: Iterable[str] = (),
) -> dict[str, Any]:
    task_rows = [migrate_v1_task(dict(task)) for task in tasks]
    audit = validate_tasks(task_rows)
    if not audit["valid"]:
        raise ValueError("invalid retrieval task set: " + "; ".join(audit["errors"][:8]))
    route_names = list(dict.fromkeys(str(route).strip() for route in routes if str(route).strip()))
    if not route_names:
        raise ValueError("at least one route is required")
    wanted_ids = {str(value).strip() for value in task_ids if str(value).strip()}
    wanted_types = {str(value).strip() for value in query_types if str(value).strip()}
    rows_by_id = {
        str(row.get("id")): row
        for row in retrieval_report.get("per_query", [])
        if isinstance(row, Mapping)
    }
    cases: list[dict[str, Any]] = []
    by_query_type: dict[str, dict[str, Any]] = {}
    for task in task_rows:
        task_id = str(task.get("id") or "")
        query_type = str(task.get("query_type") or "unknown")
        if wanted_ids and task_id not in wanted_ids:
            continue
        if wanted_types and query_type not in wanted_types:
            continue
        row = rows_by_id.get(task_id)
        route_results = {
            route: _route_coverage(task, row, route) for route in route_names
        }
        case = {
            "id": task_id,
            "query_type": query_type,
            "expected_refs": _refs(task.get("expected_refs")),
            "expected_policy": str(task.get("expected_policy") or "all"),
            "routes": route_results,
        }
        cases.append(case)
        group = by_query_type.setdefault(query_type, {
            "tasks": 0,
            "positive_tasks": 0,
            "routes": {route: _empty_route_summary() for route in route_names},
        })
        group["tasks"] += 1
        positive = bool(case["expected_refs"])
        group["positive_tasks"] += int(positive)
        if not positive:
            continue
        for route, result in route_results.items():
            summary = group["routes"][route]
            summary["positive_tasks"] += 1
            if result["status"] in {"available", "mixed"}:
                summary["available_tasks"] += 1
            summary["any_expected_ref_hit_tasks"] += int(bool(result["any_expected_ref_hit"]))
            summary["all_expected_refs_hit_tasks"] += int(bool(result["all_expected_refs_hit"]))
    for group in by_query_type.values():
        for summary in group["routes"].values():
            count = summary["positive_tasks"]
            summary["any_expected_ref_hit_rate"] = round(
                summary["any_expected_ref_hit_tasks"] / count, 4
            ) if count else None
            summary["all_expected_refs_hit_rate"] = round(
                summary["all_expected_refs_hit_tasks"] / count, 4
            ) if count else None
    return {
        "schema": SCHEMA,
        "routes": route_names,
        "tasks": len(cases),
        "summary": {
            "by_query_type": dict(sorted(by_query_type.items())),
            "positive_tasks": sum(bool(case["expected_refs"]) for case in cases),
            "negative_tasks": sum(not bool(case["expected_refs"]) for case in cases),
        },
        "snapshot": {
            key: retrieval_report.get("meta", {}).get(key)
            for key in ("vault_root", "git_commit", "snapshot_at", "embedding_model")
            if isinstance(retrieval_report.get("meta"), Mapping)
            and retrieval_report.get("meta", {}).get(key) is not None
        },
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare candidate expected-ref coverage by retrieval route")
    parser.add_argument("--retrieval-report", required=True)
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--ids", help="comma-separated task IDs")
    parser.add_argument("--query-types", help="comma-separated query types")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        report = json.loads(Path(args.retrieval_report).read_text(encoding="utf-8"))
        tasks = load_tasks(args.tasks)
        result = compare_candidate_routes(
            report,
            tasks,
            routes=args.routes.split(","),
            task_ids=(args.ids or "").split(","),
            query_types=(args.query_types or "").split(","),
        )
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ROUTE_COMPARE] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "tasks": result["tasks"],
        "positive_tasks": result["summary"]["positive_tasks"],
        "out": str(args.out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "compare_candidate_routes", "load_tasks", "main"]
