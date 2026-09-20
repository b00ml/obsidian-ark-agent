"""Offline evaluation for the bounded Query Rewrite adapter.

The evaluator compares the original query with a rewrite candidate on the same
task snapshot.  It never enables rewrite in the production retriever and only
calls a provider supplied by the caller; the CLI uses a static JSON map so a
normal evaluation run stays local and reproducible.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agentlab.core.llm import LLMProvider, LLMResponse
from agentlab.core.message import Message, TokenUsage
from agentlab.eval.rag_task_schema import migrate_v1_task
from agentlab.rag.query import build_query_plan, classify_query
from agentlab.rag.rewrite import RewriteResult, rewrite_query


RetrieveFn = Callable[[str, Mapping[str, Any]], tuple[Sequence[str], float]]


class MappingRewriteProvider(LLMProvider):
    """Deterministic provider used by offline fixtures and CI."""

    def __init__(self, mapping: Mapping[str, Any]):
        self.mapping = dict(mapping)
        self.calls = 0

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        stream: bool = False,
        on_stream=None,
    ) -> LLMResponse:
        self.calls += 1
        query = ""
        for message in messages:
            content = str(message.content or "")
            marker = "原始 query："
            if marker not in content:
                continue
            tail = content.split(marker, 1)[1]
            # The production prompt puts the value on the following line;
            # accepting an inline value keeps small hand-written fixtures valid.
            lines = tail.splitlines()
            query = (lines[0].strip() if lines and lines[0].strip()
                     else next((line.strip() for line in lines[1:] if line.strip()), ""))
            break
        value = self.mapping.get(query)
        if isinstance(value, Mapping):
            payload = dict(value)
        elif value is None:
            payload = {"query": query, "reason": "fixture_missing"}
        else:
            payload = {"query": str(value), "reason": "fixture"}
        return LLMResponse(
            content=json.dumps(payload, ensure_ascii=False),
            tool_calls=[], usage=TokenUsage(),
        )


class _UnavailableProvider(LLMProvider):
    async def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        raise RuntimeError("rewrite provider not configured")


def _ref_matches(expected: str, got: str) -> bool:
    expected = str(expected or "").strip().replace("\\", "/")
    got = str(got or "").strip().replace("\\", "/")
    if "#" in expected:
        return got == expected or got.startswith(expected + ":ch")
    return got.split("#", 1)[0] == expected


def _expectations(task: Mapping[str, Any]) -> list[set[str]]:
    refs = [str(value).strip() for value in task.get("expected_refs", []) if str(value).strip()]
    groups = task.get("expected_groups")
    if groups:
        return [{str(value).strip() for value in group if str(value).strip()} for group in groups]
    if task.get("expected_policy", "all") == "any":
        return [set(refs)] if refs else []
    return [{value} for value in refs]


def _score(task: Mapping[str, Any], refs: Sequence[str]) -> dict[str, float | bool]:
    expectations = _expectations(task)
    top5 = list(refs)[:5]
    top10 = list(refs)[:10]
    if not expectations:
        recall5 = 1.0 if not top5 else 0.0
        recall10 = 1.0 if not top10 else 0.0
        mrr = 0.0
    else:
        recall5 = sum(
            any(_ref_matches(expected, actual) for expected in group for actual in top5)
            for group in expectations
        ) / len(expectations)
        recall10 = sum(
            any(_ref_matches(expected, actual) for expected in group for actual in top10)
            for group in expectations
        ) / len(expectations)
        mrr = 0.0
        for index, actual in enumerate(refs, 1):
            if any(_ref_matches(expected, actual) for group in expectations for expected in group):
                mrr = 1.0 / index
                break
    return {
        "recall@5": round(recall5, 4),
        "recall@10": round(recall10, 4),
        "mrr": round(mrr, 4),
        "clean@5": not bool(top5) if not expectations else True,
    }


def _p50_p95(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    ordered = sorted(float(value) for value in values)
    return (
        round(ordered[int(0.5 * (len(ordered) - 1))], 1),
        round(ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))], 1),
    )


async def evaluate_query_rewrites(
    tasks: Sequence[Mapping[str, Any]],
    retrieve: RetrieveFn,
    provider: LLMProvider | None = None,
    *,
    mode: str = "shadow",
    deadline_ms: int = 250,
) -> dict[str, Any]:
    """Compare original/candidate query metrics grouped by query type."""
    if mode not in {"off", "shadow", "on"}:
        raise ValueError("mode must be off, shadow, or on")
    provider = provider or _UnavailableProvider()
    rows: list[dict[str, Any]] = []
    for raw_task in tasks:
        task = migrate_v1_task(dict(raw_task))
        query = str(task.get("query") or "").strip()
        recent_context = str(task.get("recent_context") or "")
        coverage = task.get("lexical_coverage")
        plan = build_query_plan(
            query,
            recent_context=recent_context,
            lexical_coverage=coverage,
            rewrite_mode=mode,
        )
        original_refs, original_elapsed = retrieve(query, task)
        rewrite: RewriteResult = await rewrite_query(
            provider,
            query,
            recent_context=recent_context,
            lexical_coverage=coverage,
            mode=mode,
            deadline_ms=deadline_ms,
        )
        candidate_query = rewrite.candidate_query or query
        candidate_refs: Sequence[str] = original_refs
        candidate_elapsed = original_elapsed
        candidate_retrieved = False
        if candidate_query and candidate_query != query:
            candidate_refs, candidate_elapsed = retrieve(candidate_query, task)
            candidate_retrieved = True
        actual_refs = candidate_refs if mode == "on" and rewrite.applied else original_refs
        rows.append({
            "id": task.get("id", ""),
            "query": query,
            "query_type": task.get("query_type") or classify_query(query, recent_context=recent_context)[0],
            "eligible": plan.should_rewrite,
            "original_query": query,
            "candidate_query": candidate_query,
            "actual_query": rewrite.query,
            "applied": rewrite.applied,
            "candidate_retrieved": candidate_retrieved,
            "reason": rewrite.reason,
            "error": rewrite.error,
            "original": {**_score(task, original_refs), "refs": list(original_refs),
                          "latency_ms": round(float(original_elapsed) * 1000, 1)},
            "candidate": {**_score(task, candidate_refs), "refs": list(candidate_refs),
                          "latency_ms": round(float(candidate_elapsed) * 1000, 1)},
            "actual": {**_score(task, actual_refs), "refs": list(actual_refs)},
        })

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["query_type"])].append(row)

    def summary(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tasks": len(group),
            "eligible": sum(bool(row["eligible"]) for row in group),
            "candidate_retrieved": sum(bool(row["candidate_retrieved"]) for row in group),
            "applied": sum(bool(row["applied"]) for row in group),
        }
        for side in ("original", "candidate"):
            metrics = [row[side] for row in group]
            latencies = [float(metric["latency_ms"]) for metric in metrics]
            p50, p95 = _p50_p95(latencies)
            result[side] = {
                "recall@5": round(statistics.mean(metric["recall@5"] for metric in metrics), 4) if metrics else None,
                "recall@10": round(statistics.mean(metric["recall@10"] for metric in metrics), 4) if metrics else None,
                "mrr": round(statistics.mean(metric["mrr"] for metric in metrics), 4) if metrics else None,
                "p50_ms": p50, "p95_ms": p95,
            }
        result["delta"] = {
            key: round(result["candidate"][key] - result["original"][key], 4)
            for key in ("recall@5", "recall@10", "mrr")
            if result["candidate"][key] is not None and result["original"][key] is not None
        }
        result["regressions"] = sum(
            row["candidate"]["recall@5"] < row["original"]["recall@5"]
            or row["candidate"]["mrr"] < row["original"]["mrr"]
            for row in group if row["candidate_retrieved"]
        )
        return result

    return {
        "schema": "rag-query-rewrite-eval-v1",
        "mode": mode,
        "deadline_ms": int(deadline_ms),
        "provider_calls": getattr(provider, "calls", None),
        "summary": {name: summary(group) for name, group in sorted(groups.items())},
        "overall": summary(rows),
        "per_query": rows,
    }


def evaluate_query_rewrites_sync(*args, **kwargs) -> dict[str, Any]:
    return asyncio.run(evaluate_query_rewrites(*args, **kwargs))


def _fixture_retriever(path: str | Path) -> RetrieveFn:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("retrieval fixture must be an object keyed by query")

    def retrieve(query: str, _task: Mapping[str, Any]) -> tuple[Sequence[str], float]:
        value = data.get(query, {})
        if isinstance(value, list):
            return [str(item) for item in value], 0.0
        if not isinstance(value, Mapping):
            return [], 0.0
        return [str(item) for item in value.get("refs", [])], float(value.get("latency_ms", 0.0)) / 1000
    return retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline query rewrite evaluation")
    parser.add_argument("--tasks", required=True, help="v1/v2 retrieval task JSONL")
    parser.add_argument("--retrieval-fixture", required=True, help="JSON map query -> refs/latency")
    parser.add_argument("--rewrite-map", help="JSON map original query -> rewritten query/object")
    parser.add_argument("--mode", choices=("off", "shadow", "on"), default="shadow")
    parser.add_argument("--deadline-ms", type=int, default=250)
    parser.add_argument("--out")
    args = parser.parse_args()
    tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines() if line.strip()]
    mapping = json.loads(Path(args.rewrite_map).read_text(encoding="utf-8")) if args.rewrite_map else {}
    report = evaluate_query_rewrites_sync(
        tasks,
        _fixture_retriever(args.retrieval_fixture),
        MappingRewriteProvider(mapping),
        mode=args.mode,
        deadline_ms=args.deadline_ms,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
