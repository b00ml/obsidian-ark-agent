"""Summarize repeated live answer probes for stability decisions.

The answer-gate report contains one case per task/probe.  This module groups
``task:live:<repeat>`` cases so retrieval, assessor, and refusal behavior can
be classified as stable or intermittent without treating failed provider
calls as evidence.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


def _base_task_id(value: Any) -> str:
    task_id = str(value or "").strip()
    if ":live:" in task_id:
        return task_id.split(":live:", 1)[0]
    if ":synthetic" in task_id:
        return task_id.split(":synthetic", 1)[0]
    return task_id


def _consistency(values: Iterable[Any]) -> dict[str, Any]:
    normalized = [value for value in values if value is not None]
    if not normalized:
        return {"observed": 0, "distinct": [], "stable": None, "majority": None}
    counts = Counter(str(value) for value in normalized)
    majority_value, majority_count = counts.most_common(1)[0]
    return {
        "observed": len(normalized),
        "distinct": sorted(counts),
        "stable": len(counts) == 1,
        "majority": majority_value,
        "majority_rate": round(majority_count / len(normalized), 4),
    }


def summarize_repeated_cases(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return per-task repeat stability and a compact overall summary."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        # A mixed answer-gate report also contains synthetic fallback cases
        # for every task.  They are not repeat observations and would make a
        # three-run stability report look like a one-run report for 133 tasks.
        case_id = str(case.get("id") or "")
        if ":live:" not in case_id:
            continue
        task_id = _base_task_id(case.get("task_id") or case.get("id"))
        if task_id:
            grouped.setdefault(task_id, []).append(case)

    rows: list[dict[str, Any]] = []
    for task_id, task_cases in sorted(grouped.items()):
        real = [case for case in task_cases if case.get("real_probe")]
        failed = [case for case in task_cases
                  if str(case.get("source") or "").lower() == "unavailable"
                  or case.get("error")]
        positive = [case for case in real if case.get("expected_decision") == "answerable"]
        rows.append({
            "task_id": task_id,
            "runs": len(task_cases),
            "real_runs": len(real),
            "failed_runs": len(failed),
            "expected_decision": str((task_cases[0].get("expected_decision") or "")),
            "abstention": _consistency(case.get("predicted_abstention") for case in real),
            "candidate_expected_ref_any": _consistency(
                case.get("candidate_expected_ref_any") for case in positive
            ),
            "candidate_expected_ref_all": _consistency(
                case.get("candidate_expected_ref_all") for case in positive
            ),
            "candidate_expected_ref_contentful": _consistency(
                case.get("candidate_expected_ref_contentful") for case in positive
            ),
            "error_attribution": _consistency(
                case.get("error_attribution") for case in positive
            ),
            "assessment": _consistency(case.get("assessment") for case in real),
            "retrieval_miss_stable": bool(positive) and all(
                case.get("error_attribution") == "retrieval_miss" for case in positive
            ),
            "assessor_false_negative_stable": bool(positive) and all(
                case.get("error_attribution") == "assessor_false_negative"
                for case in positive
            ),
            "retrieval_content_gap_stable": bool(positive) and all(
                case.get("error_attribution") == "retrieval_content_gap"
                for case in positive
            ),
        })

    positive_rows = [row for row in rows if row["expected_decision"] == "answerable"]
    negative_rows = [row for row in rows if row["expected_decision"] != "answerable"]
    return {
        "schema": "rag-answer-probe-repeat-analysis-v1",
        "tasks": len(rows),
        "positive_tasks": len(positive_rows),
        "negative_tasks": len(negative_rows),
        "stable_positive_retrieval_miss_tasks": sum(
            row["retrieval_miss_stable"] for row in positive_rows
        ),
        "stable_positive_assessor_false_negative_tasks": sum(
            row["assessor_false_negative_stable"] for row in positive_rows
        ),
        "stable_positive_retrieval_content_gap_tasks": sum(
            row["retrieval_content_gap_stable"] for row in positive_rows
        ),
        "intermittent_tasks": sum(
            bool(row["abstention"]["distinct"] and not row["abstention"]["stable"])
            or bool(row["error_attribution"]["distinct"]
                    and not row["error_attribution"]["stable"])
            for row in rows
        ),
        "rows": rows,
    }


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(document, Mapping):
        document = document.get("cases", [])
    if not isinstance(document, list):
        raise ValueError("answer-gate report must contain a cases list")
    return [dict(case) for case in document if isinstance(case, Mapping)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize repeated live answer probes")
    parser.add_argument("--answer-gate-report", required=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    report = summarize_repeated_cases(load_cases(args.answer_gate_report))
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["load_cases", "summarize_repeated_cases", "main"]
