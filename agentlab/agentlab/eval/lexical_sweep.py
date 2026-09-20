"""Read-only lexical/entry aggregation sweep over a frozen retrieval report.

The production lexical scorer is intentionally unchanged.  This evaluator
replays recorded lexical refs with different deduplication granularities so
we can quantify whether entry aggregation, rather than BM25 complexity, is
the current quality bottleneck.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from agentlab.eval.rag_task_schema import migrate_v1_task, validate_tasks
from agentlab.eval.run_rag_retrieval import _expectation_sets, _mrr, _recall_at, _ref_matches
from agentlab.rag.hybrid import entry_key

SCHEMA = "rag-lexical-aggregation-sweep-v1"


def _refs(row: Mapping[str, Any]) -> list[str]:
    shadow = row.get("hybrid_shadow")
    routes = shadow.get("routes") if isinstance(shadow, Mapping) else None
    route = routes.get("lexical") if isinstance(routes, Mapping) else None
    if isinstance(route, Mapping):
        return [str(ref).replace("\\", "/") for ref in route.get("refs", []) if str(ref).strip()]
    legacy = row.get("p2_lexical")
    return [str(ref).replace("\\", "/") for ref in legacy.get("refs", [])] if isinstance(legacy, Mapping) else []


def replay_refs(refs: Iterable[str], *, dedupe: str, k: int = 10) -> list[str]:
    """Replay a lexical list with ref, file, or entry aggregation."""
    seen: set[str] = set()
    out: list[str] = []
    for ref in refs:
        value = str(ref).replace("\\", "/").strip()
        if not value:
            continue
        key = value if dedupe == "ref" else entry_key(value, dedupe)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= max(1, int(k)):
            break
    return out


def sweep_lexical(report: Mapping[str, Any], tasks: Iterable[Mapping[str, Any]], *, k: int = 10) -> dict[str, Any]:
    rows = {str(row.get("id")): row for row in report.get("per_query", []) if isinstance(row, Mapping)}
    task_rows = [migrate_v1_task(dict(task)) for task in tasks]
    audit = validate_tasks(task_rows)
    if not audit["valid"]:
        raise ValueError("invalid retrieval task set: " + "; ".join(audit["errors"][:8]))
    variants = ("ref", "entry", "file")
    summary: dict[str, dict[str, dict[str, float | int | None]]] = {}
    cases: list[dict[str, Any]] = []
    for task in task_rows:
        row = rows.get(str(task.get("id")))
        if not row:
            continue
        refs = _refs(row)
        expectations = _expectation_sets(task)
        positive = bool(expectations)
        values: dict[str, Any] = {}
        qtype = str(task.get("query_type") or "unknown")
        bucket = summary.setdefault(qtype, {name: {"positive": 0, "negative": 0, "recall@5": [], "recall@10": [], "mrr": [], "clean@5": []} for name in variants})
        for name in variants:
            got = replay_refs(refs, dedupe=name, k=k)
            result = {"refs": got, "recall@5": _recall_at(expectations, got, 5), "recall@10": _recall_at(expectations, got, 10), "mrr": _mrr(expectations, got)}
            values[name] = result
            target = bucket[name]
            if positive:
                target["positive"] += 1
                for metric in ("recall@5", "recall@10", "mrr"):
                    target[metric].append(float(result[metric]))
            else:
                target["negative"] += 1
                target["clean@5"].append(1.0 if not got[:5] else 0.0)
        cases.append({"id": str(task.get("id")), "query_type": qtype, "positive": positive, "variants": values})

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None
    for group in summary.values():
        for metrics in group.values():
            for metric in ("recall@5", "recall@10", "mrr", "clean@5"):
                metrics[metric] = mean(metrics[metric])
    meta = report.get("meta")
    return {"schema": SCHEMA, "k": k, "variants": list(variants), "tasks": len(cases), "summary": {"by_query_type": summary}, "snapshot": {key: meta.get(key) for key in ("vault_root", "snapshot_at", "index", "embedding_model") if isinstance(meta, Mapping) and meta.get(key) is not None}, "notes": {"offline_only": True, "production_unchanged": True, "meaning": "ref preserves chunks; entry/file collapse duplicates before top-k; this is not a BM25 implementation"}, "cases": cases}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep lexical aggregation over a frozen retrieval report")
    parser.add_argument("--retrieval-report", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        report = json.loads(Path(args.retrieval_report).read_text(encoding="utf-8"))
        tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines() if line.strip()]
        result = sweep_lexical(report, tasks, k=max(1, args.k))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"tasks": result["tasks"], "out": args.out}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[LEXICAL_SWEEP] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["SCHEMA", "replay_refs", "sweep_lexical", "main"]
