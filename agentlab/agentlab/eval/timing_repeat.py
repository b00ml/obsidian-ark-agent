"""Aggregate repeated RAG timing reports without changing runtime state."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


def _number(values: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number and number not in {float("inf"), float("-inf")}:
            out.append(number)
    return out


def _distribution(values: Sequence[Any]) -> dict[str, Any]:
    numbers = sorted(_number(values))
    if not numbers:
        return {"values": [], "min": None, "max": None, "mean": None}
    return {
        "values": [round(value, 4) for value in numbers],
        "min": round(numbers[0], 4),
        "max": round(numbers[-1], 4),
        "mean": round(statistics.mean(numbers), 4),
    }


def aggregate_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate route metrics and assert all reports use one snapshot."""
    if not reports:
        raise ValueError("at least one report is required")
    keys = (
        "tasks_path", "tasks_sha256", "index_sha256", "index_version",
        "parser_version", "embedding_model",
    )
    snapshot: dict[str, Any] = {}
    mismatches: list[str] = []
    for key in keys:
        values = []
        for report in reports:
            meta = report.get("meta", {})
            index = meta.get("index", {}) if isinstance(meta, Mapping) else {}
            hashes = meta.get("input_hashes", {}) if isinstance(meta, Mapping) else {}
            chunking = meta.get("chunking", {}) if isinstance(meta, Mapping) else {}
            values.append({
                "tasks_path": meta.get("tasks_path") if isinstance(meta, Mapping) else None,
                "tasks_sha256": hashes.get("tasks_sha256") if isinstance(hashes, Mapping) else None,
                "index_sha256": hashes.get("index_sha256") if isinstance(hashes, Mapping) else None,
                "index_version": chunking.get("index_version") if isinstance(chunking, Mapping) else None,
                "parser_version": chunking.get("parser_version") if isinstance(chunking, Mapping) else None,
                "embedding_model": meta.get("embedding_model") if isinstance(meta, Mapping) else None,
            }[key])
        distinct = {str(value) for value in values}
        if len(distinct) > 1:
            mismatches.append(key)
        snapshot[key] = values[0]
    if mismatches:
        raise ValueError("snapshot mismatch: " + ", ".join(mismatches))

    metric_names = ("p50_ms", "p95_ms", "recall@5", "recall@10", "mrr", "clean@5")
    route_names = sorted({
        str(name)
        for report in reports
        for name, value in (report.get("summary", {}) or {}).items()
        if isinstance(value, Mapping) and any(metric in value for metric in metric_names)
    })
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for route in route_names:
        metrics[route] = {}
        for metric in metric_names:
            values = [
                (report.get("summary", {}) or {}).get(route, {}).get(metric)
                for report in reports
                if isinstance((report.get("summary", {}) or {}).get(route), Mapping)
            ]
            metrics[route][metric] = _distribution(values)
    return {
        "schema": "rag-timing-repeat-v1",
        "runs": len(reports),
        "snapshot": snapshot,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate repeated RAG timing reports")
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
    payload = aggregate_reports(reports)
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
