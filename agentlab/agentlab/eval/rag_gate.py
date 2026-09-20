"""P4.5-D read-only gate for RRF shadow reports.

The gate turns a completed retrieval report into an auditable recommendation;
it never changes runtime configuration or enables the production hybrid route.
Candidate reports must contain available vector and RRF metrics before a manual
gray decision can be considered.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Mapping


SCHEMA = "rag-p4.5-gate-v1"


def _summary(report: Mapping, name: str) -> dict:
    value = report.get("summary", {}).get(name, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _status(metric: Mapping) -> str:
    return str(metric.get("status") or "unavailable")


def _negative(report: Mapping, name: str) -> dict:
    value = report.get("summary", {}).get("negative", {}).get(name, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _route_available(summary: Mapping) -> bool:
    """Whether a route has usable positive-task measurements.

    A report may contain independent fixtures that are intentionally marked
    ``not_applicable``.  They must not make an otherwise complete canonical
    route look unavailable.  Any real unavailable/error status still blocks
    the gate, while a mix of ``available`` + ``not_applicable`` is accepted.
    """
    if int(summary.get("tasks", 0) or 0) <= 0:
        return False
    status = _status(summary)
    if status == "available":
        return True
    if status != "mixed":
        return False
    counts = summary.get("status_counts", {})
    if not isinstance(counts, Mapping) or int(counts.get("available", 0) or 0) <= 0:
        return False
    return not any(
        int(value or 0) > 0 and name not in {"available", "not_applicable"}
        for name, value in counts.items()
    )


def assess(
    baseline: Mapping,
    candidate: Mapping,
    *,
    max_recall_drop: float = 0.0,
    max_p95_ms: float = 500.0,
    answer_gate: Mapping | None = None,
    min_real_answer_cases: int = 20,
    max_false_answer_rate: float = 0.05,
    max_false_refusal_rate: float = 0.20,
) -> dict:
    """Return a deterministic eligibility report without mutating inputs."""
    base_lexical = _summary(baseline, "p2_lexical")
    candidate_lexical = _summary(candidate, "p2_lexical")
    candidate_vector = _summary(candidate, "p2_vector")
    candidate_rrf = _summary(candidate, "rrf")
    answer_source = answer_gate or candidate.get("answer_gate") or candidate
    answer_summary = _summary(answer_source, "answer_gate")
    checks: dict[str, dict] = {}

    def check(name: str, passed: bool, observed=None, required=None, reason: str = "") -> None:
        checks[name] = {
            "passed": bool(passed),
            "observed": observed,
            "required": required,
            **({"reason": reason} if reason else {}),
        }

    check(
        "lexical_available",
        _route_available(candidate_lexical),
        _status(candidate_lexical),
        "available",
        "candidate lexical route must remain available as the fallback path",
    )
    check(
        "vector_available",
        _route_available(candidate_vector),
        _status(candidate_vector),
        "available",
        "candidate vector route must have scored positive tasks",
    )
    check(
        "rrf_available",
        _route_available(candidate_rrf),
        _status(candidate_rrf),
        "available",
        "candidate RRF route must have scored positive tasks",
    )

    base_recall = base_lexical.get("recall@5")
    rrf_recall = candidate_rrf.get("recall@5")
    recall_required = None if base_recall is None else float(base_recall) - max(0.0, max_recall_drop)
    check(
        "rrf_recall_at_5",
        recall_required is not None and rrf_recall is not None and float(rrf_recall) >= recall_required,
        rrf_recall,
        recall_required,
        "RRF recall@5 must not regress from the frozen lexical baseline",
    )

    rrf_p95 = candidate_rrf.get("p95_ms")
    check(
        "rrf_p95",
        rrf_p95 is not None and float(rrf_p95) < max_p95_ms,
        rrf_p95,
        f"<{max_p95_ms:g}ms",
        "warm RRF p95 exceeds the gray latency budget",
    )

    base_negative = _negative(baseline, "p2_lexical")
    candidate_negative = _negative(candidate, "rrf")
    base_clean = base_negative.get("clean@5")
    candidate_clean = candidate_negative.get("clean@5")
    check(
        "negative_clean_at_5",
        base_clean is not None and candidate_clean is not None
        and float(candidate_clean) >= float(base_clean),
        candidate_clean,
        base_clean,
        "RRF negative clean@5 must not regress from lexical baseline",
    )

    forbidden = candidate.get("summary", {}).get("forbidden", {})
    forbidden_hits = forbidden.get("hits@5") if isinstance(forbidden, Mapping) else None
    check(
        "forbidden_zero",
        forbidden_hits is not None and int(forbidden_hits) == 0,
        forbidden_hits,
        0,
        "forbidden/project/archive candidates must not leak into top 5",
    )

    failures = candidate.get("meta", {}).get("index", {}).get("failures")
    check(
        "index_failures_zero",
        failures is not None and int(failures) == 0,
        failures,
        0,
        "candidate index must have no pending/dead failures",
    )
    coverage = candidate.get("meta", {}).get("index", {}).get("coverage")
    check(
        "index_coverage",
        coverage is not None and float(coverage) >= 1.0,
        coverage,
        1.0,
        "candidate index must cover the full Vault snapshot",
    )

    # Retrieval clean@k is not answer safety.  A candidate list can be
    # non-empty for a plausible-absent question, so a production decision
    # requires a separate answer-level report with real generated answers.
    answer_status = _status(answer_summary)
    real_cases = int(answer_summary.get(
        "eligible_real_cases", answer_summary.get("real_cases", 0)
    ) or 0)
    false_answer_rate = answer_summary.get("false_answer_rate")
    false_refusal_rate = answer_summary.get("false_refusal_rate")
    check(
        "answer_gate_available",
        answer_status == "available" and int(answer_summary.get("cases", 0) or 0) > 0,
        answer_status,
        "available",
        "answer-level abstention/citation report is required",
    )
    check(
        "answer_gate_real_cases",
        real_cases >= max(0, int(min_real_answer_cases)),
        real_cases,
        f">={int(min_real_answer_cases)}",
        "synthetic probes cannot establish production answer safety",
    )
    check(
        "answer_false_rate",
        false_answer_rate is not None and float(false_answer_rate) <= max(0.0, max_false_answer_rate),
        false_answer_rate,
        f"<={max_false_answer_rate:g}",
        "answerable abstention negatives must not be answered",
    )
    check(
        "answer_false_refusal_rate",
        false_refusal_rate is not None and float(false_refusal_rate) <= max(0.0, max_false_refusal_rate),
        false_refusal_rate,
        f"<={max_false_refusal_rate:g}",
        "answerable evidence must not be rejected too often",
    )
    check(
        "answer_citation_out_of_scope_zero",
        answer_summary.get("citation_out_of_scope_rate") in (0, 0.0),
        answer_summary.get("citation_out_of_scope_rate"),
        0,
        "generated citations must stay within retrieved candidates",
    )
    check(
        "answer_forbidden_zero",
        answer_summary.get("forbidden_hit_rate") in (0, 0.0),
        answer_summary.get("forbidden_hit_rate"),
        0,
        "generated answers must not cite forbidden refs",
    )

    passed = all(item["passed"] for item in checks.values())
    return {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": "eligible_for_manual_gray" if passed else "blocked",
        "production_switch": "manual_only",
        "checks": checks,
        "baseline": {
            "path": baseline.get("meta", {}).get("input_hashes", {}).get("tasks_sha256"),
            "p2_lexical_recall@5": base_recall,
        },
        "candidate": {
            "p2_lexical_recall@5": candidate_lexical.get("recall@5"),
            "p2_vector_recall@5": candidate_vector.get("recall@5"),
            "rrf_recall@5": rrf_recall,
            "rrf_p95_ms": rrf_p95,
            "answer_gate": answer_summary,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P4.5-D RRF shadow release gate")
    parser.add_argument("--baseline", required=True, help="frozen lexical baseline report")
    parser.add_argument("--candidate", required=True, help="candidate vector/RRF report")
    parser.add_argument("--max-recall-drop", type=float, default=0.0)
    parser.add_argument("--max-p95-ms", type=float, default=500.0)
    parser.add_argument(
        "--answer-gate", help="answer-level report from agentlab.eval.answer_gate_eval",
    )
    parser.add_argument("--min-real-answer-cases", type=int, default=20)
    parser.add_argument("--max-false-answer-rate", type=float, default=0.05)
    parser.add_argument("--max-false-refusal-rate", type=float, default=0.20)
    parser.add_argument("--out", help="write the gate report to a hidden/derived path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_recall_drop < 0 or args.max_p95_ms <= 0:
        raise SystemExit("--max-recall-drop must be non-negative and --max-p95-ms positive")
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    answer_gate = (
        json.loads(Path(args.answer_gate).read_text(encoding="utf-8"))
        if args.answer_gate else None
    )
    report = assess(
        baseline,
        candidate,
        max_recall_drop=args.max_recall_drop,
        max_p95_ms=args.max_p95_ms,
        answer_gate=answer_gate,
        min_real_answer_cases=args.min_real_answer_cases,
        max_false_answer_rate=args.max_false_answer_rate,
        max_false_refusal_rate=args.max_false_refusal_rate,
    )
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if report["decision"] == "eligible_for_manual_gray" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "assess", "main"]
