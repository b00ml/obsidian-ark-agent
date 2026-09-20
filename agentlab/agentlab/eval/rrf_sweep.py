"""Offline weighted-RRF sweep over a frozen retrieval report.

The sweep is diagnostic only.  It re-ranks recorded lexical/vector refs and
does not call an embedding provider, read the Vault, or change production
configuration.  A variant can therefore be rejected using fixed quality
metrics before any runtime wiring is attempted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from agentlab.eval.run_rag_retrieval import _expectation_sets, _mrr, _recall_at, _ref_matches
from agentlab.eval.rag_task_schema import migrate_v1_task, validate_tasks
from agentlab.rag.hybrid import _semantic_fallback_allowed, entry_key


SCHEMA = "rag-rrf-sweep-v4"
DEFAULT_TASKS = Path(__file__).resolve().parents[3] / ".ai" / "evals" / "rag_retrieval-v2.jsonl"
DEFAULT_VARIANTS: dict[str, dict[str, float]] = {
    "lexical": {"lexical": 1.0, "vector": 0.0},
    "vector": {"lexical": 0.0, "vector": 1.0},
    "equal": {"lexical": 1.0, "vector": 1.0},
    "vector_1_25": {"lexical": 1.0, "vector": 1.25},
    "vector_1_5": {"lexical": 1.0, "vector": 1.5},
    "vector_2": {"lexical": 1.0, "vector": 2.0},
    "lexical_0_5": {"lexical": 0.5, "vector": 1.0},
}

# This is a diagnostic policy, not a runtime default.  Exact identifiers and
# negative/isolation cases stay lexical-first; only the query types with a
# measured semantic upside are allowed to trial a vector-boosted RRF route.
DEFAULT_QUERY_TYPE_POLICY: dict[str, str] = {
    "exact_title": "lexical",
    "bv_entity": "lexical",
    "named_entity": "lexical",
    "freshness": "lexical",
    "negative": "lexical",
    "isolation": "lexical",
    "bucket_entry": "vector_2",
    "episodic_semantic": "vector_1_25",
    "paraphrase": "vector_1_25",
    "cross_doc": "equal",
    "decision_semantic": "equal",
    "theme": "equal",
}


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


def _route_rows(row: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    shadow = row.get("hybrid_shadow")
    routes = shadow.get("routes") if isinstance(shadow, Mapping) else None
    route = routes.get(name) if isinstance(routes, Mapping) else None
    if isinstance(route, Mapping):
        return [{"ref": ref} for ref in route.get("refs", []) if str(ref).strip()]
    # Older reports only retained the top-k component refs.  Keep the sweep
    # backwards-compatible, but mark its result as lower-confidence in the
    # report metadata rather than silently reading another snapshot.
    legacy = row.get(f"p2_{name}")
    if isinstance(legacy, Mapping):
        return [{"ref": ref} for ref in legacy.get("refs", []) if str(ref).strip()]
    return []


def weighted_rrf(routes: Mapping[str, Iterable[Mapping[str, Any]]], *,
                 weights: Mapping[str, float], k: int = 10,
                 rrf_k: int = 60) -> list[str]:
    """Fuse recorded routes with route weights and entry-level deduplication."""
    if k <= 0:
        return []
    scores: dict[str, float] = {}
    representatives: dict[str, str] = {}
    for route, rows in routes.items():
        weight = float(weights.get(route, 1.0))
        if weight <= 0:
            continue
        seen: set[str] = set()
        rank = 0
        for row in rows:
            ref = str(row.get("ref") or "").replace("\\", "/").strip()
            if not ref:
                continue
            key = entry_key(ref, "entry")
            if key in seen:
                continue
            seen.add(key)
            rank += 1
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            representatives.setdefault(key, ref)
    ranked = sorted(representatives, key=lambda key: (-scores[key], key))
    return [representatives[key] for key in ranked[:k]]


def _available(row: Mapping[str, Any]) -> bool:
    """Whether the recorded RRF attempt completed, including empty results.

    An empty candidate list is the expected success result for a clean
    negative.  Treating it as unavailable silently removes both clean
    negatives and genuine positive recall misses from the comparison set.
    """
    status = row.get("rrf")
    if isinstance(status, Mapping):
        value = str(status.get("status") or "available")
        if value in {"not_applicable", "unavailable", "error"}:
            return False
        if value in {"available", "mixed"}:
            return True
    return any(
        _route_status(row, route) in {"available", "mixed"}
        for route in ("lexical", "vector")
    )


def _route_status(row: Mapping[str, Any], route: str) -> str:
    statuses = row.get("route_statuses")
    if isinstance(statuses, Mapping):
        value = statuses.get(f"p2_{route}")
        if value:
            return str(value)
    shadow = row.get("hybrid_shadow")
    routes = shadow.get("routes") if isinstance(shadow, Mapping) else None
    item = routes.get(route) if isinstance(routes, Mapping) else None
    if isinstance(item, Mapping):
        return str(item.get("status") or "available")
    return "available" if _route_rows(row, route) else "unavailable"


def _route_latency_ms(row: Mapping[str, Any], route: str) -> float | None:
    metric = row.get(f"p2_{route}")
    if isinstance(metric, Mapping) and metric.get("latency_ms") is not None:
        try:
            return float(metric["latency_ms"])
        except (TypeError, ValueError):
            pass
    shadow = row.get("hybrid_shadow")
    routes = shadow.get("routes") if isinstance(shadow, Mapping) else None
    item = routes.get(route) if isinstance(routes, Mapping) else None
    if isinstance(item, Mapping) and item.get("latency_ms") is not None:
        try:
            return float(item["latency_ms"])
        except (TypeError, ValueError):
            pass
    return None


def _variant_available(
    row: Mapping[str, Any],
    variant: str,
    variants: Mapping[str, Mapping[str, float]] = DEFAULT_VARIANTS,
) -> bool:
    weights = variants.get(variant)
    if not weights:
        return False
    required = [route for route, weight in weights.items() if float(weight) > 0]
    return bool(required) and all(
        _route_status(row, route) in {"available", "mixed"}
        for route in required
    )


def _select_policy_variant(
    query_type_policy: Mapping[str, str],
    query_type: str,
    variants: Mapping[str, Mapping[str, float]],
) -> tuple[str | None, str, bool]:
    """Resolve a policy name without making a custom sweep crash.

    A caller may intentionally pass a reduced variant set.  Keep the
    requested name for auditability, but choose a deterministic available
    fallback so the report remains comparable instead of indexing a missing
    ``equal`` variant.
    """
    requested = str(query_type_policy.get(query_type) or "equal")
    if requested in variants:
        return requested, requested, True
    for fallback in ("equal", "lexical", "vector"):
        if fallback in variants:
            return fallback, requested, False
    names = sorted(str(name) for name in variants)
    return (names[0] if names else None), requested, False


def _variant_latency_ms(
    row: Mapping[str, Any],
    variant: str,
    variants: Mapping[str, Mapping[str, float]] = DEFAULT_VARIANTS,
) -> float | None:
    weights = variants.get(variant)
    if not weights:
        return None
    routes = [route for route, weight in weights.items() if float(weight) > 0]
    values = [_route_latency_ms(row, route) for route in routes]
    values = [value for value in values if value is not None]
    if not values:
        return None
    # P2 routes run concurrently.  Offline replay therefore uses the slower
    # route as the latency estimate rather than adding route durations.
    return max(values)


def _scope_status(task: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    if task.get("vault_root") or str(task.get("corpus_scope") or "") == "fixture":
        status = row.get("route_statuses", {}).get("rrf") if isinstance(
            row.get("route_statuses"), Mapping) else None
        return "not_applicable" if status == "not_applicable" else "unknown"
    expected_scope = task.get("scope")
    shadow = row.get("hybrid_shadow")
    if not isinstance(expected_scope, Mapping) or not isinstance(shadow, Mapping):
        return "unknown"
    actual_scope = shadow.get("scope")
    actual_scope = actual_scope if isinstance(actual_scope, Mapping) else shadow
    defaults: dict[str, Any] = {
        "project_id": "default",
        "session_id": "",
        "statuses": (),
        "include_archive": False,
    }
    for field, default in defaults.items():
        expected_value = expected_scope.get(field, default)
        if field == "project_id":
            expected_value = str(expected_value or default)
        elif field == "session_id":
            expected_value = str(expected_value or default)
        elif field == "statuses":
            expected_value = tuple(sorted(str(item) for item in (expected_value or ()) if str(item)))
        else:
            expected_value = bool(expected_value)
        if field not in actual_scope or actual_scope.get(field) is None:
            # Older shadow reports only carried project/session.  Defaults are
            # still provable; a requested non-default filter is not.
            if expected_value != default:
                return "unknown"
            continue
        actual_value = actual_scope.get(field)
        if field == "project_id":
            actual_value = str(actual_value or default)
        elif field == "session_id":
            actual_value = str(actual_value or default)
        elif field == "statuses":
            actual_value = tuple(sorted(str(item) for item in (actual_value or ()) if str(item)))
        else:
            actual_value = bool(actual_value)
        if actual_value != expected_value:
            return "mismatch"
    return "consistent"


def _forbidden_hits(task: Mapping[str, Any], refs: Iterable[str]) -> int:
    forbidden = [str(ref).strip() for ref in task.get("forbidden_refs", []) if str(ref).strip()]
    return sum(
        1 for actual in list(refs)[:5]
        if any(_ref_matches(expected, actual) for expected in forbidden)
    )


def _p50_p95(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    ordered = sorted(values)
    return (
        round(ordered[int(0.5 * (len(ordered) - 1))], 1),
        round(ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))], 1),
    )


def _ref_source_types(refs: Iterable[str]) -> dict[str, int]:
    """Classify report refs without reading a Vault or inferring relevance."""
    counts: dict[str, int] = {}
    for ref in refs:
        path = str(ref).replace("\\", "/").split("#", 1)[0].lstrip("/").lower()
        if path.startswith("ark/memory/"):
            source_type = "memory"
        elif path.startswith("inbox/"):
            source_type = "inbox"
        elif path.startswith("ark/projects/"):
            source_type = "project"
        else:
            source_type = "vault"
        counts[source_type] = counts.get(source_type, 0) + 1
    return dict(sorted(counts.items()))


def _new_policy_group(*, variant: str | None = None,
                      configured: bool | None = None) -> dict[str, Any]:
    """Create an accumulator shared by grouped and whole-policy summaries."""
    return {
        **({"variant": variant} if variant is not None else {}),
        **({"configured": configured} if configured is not None else {}),
        "tasks": 0,
        "available_tasks": 0,
        "positive_tasks": 0,
        "negative_tasks": 0,
        "recall@5": [],
        "recall@10": [],
        "mrr": [],
        "clean@5": [],
        "latency_ms": [],
        "forbidden_hits@5": 0,
        "scope": {"consistent": 0, "mismatch": 0, "unknown": 0, "not_applicable": 0},
        "eval_coverage_forced_tasks": 0,
        "negative_by_kind": {},
    }


def _add_policy_case(group: dict[str, Any], case: Mapping[str, Any]) -> None:
    """Accumulate a selected-policy case while retaining absent-query causes."""
    policy = case["policy"]
    positive = bool(case["positive"])
    group["tasks"] += 1
    group["available_tasks"] += int(policy["available"])
    group["positive_tasks"] += int(positive)
    group["negative_tasks"] += int(not positive)
    if policy["available"]:
        if positive:
            group["recall@5"].append(float(policy["recall@5"]))
            group["recall@10"].append(float(policy["recall@10"]))
            group["mrr"].append(float(policy["mrr"]))
        elif policy["clean@5"] is not None:
            group["clean@5"].append(float(policy["clean@5"]))
        if policy["estimated_latency_ms"] is not None:
            group["latency_ms"].append(float(policy["estimated_latency_ms"]))
    group["forbidden_hits@5"] += int(policy["forbidden_hits@5"])
    group["eval_coverage_forced_tasks"] += int(
        bool(policy.get("eval_coverage_forced"))
    )
    scope_status = policy["scope_status"]
    group["scope"][scope_status] = group["scope"].get(scope_status, 0) + 1
    if positive:
        return
    negative_kind = str(case.get("negative_kind") or "unspecified")
    negative = group["negative_by_kind"].setdefault(negative_kind, {
        "tasks": 0,
        "available_tasks": 0,
        "clean_tasks": 0,
        "non_clean_case_ids": [],
        "non_clean_source_types": {},
    })
    negative["tasks"] += 1
    negative["available_tasks"] += int(policy["available"])
    if not policy["available"]:
        return
    if policy["clean@5"] == 1.0:
        negative["clean_tasks"] += 1
        return
    negative["non_clean_case_ids"].append(str(case["id"]))
    for source_type, count in policy["source_types"].items():
        negative["non_clean_source_types"][source_type] = (
            negative["non_clean_source_types"].get(source_type, 0) + int(count)
        )


def _finalize_policy_group(group: dict[str, Any]) -> None:
    """Replace internal samples with stable, compact report metrics."""
    latency_p50, latency_p95 = _p50_p95(group.pop("latency_ms"))
    for field in ("recall@5", "recall@10", "mrr", "clean@5"):
        values = group[field]
        group[field] = round(sum(values) / len(values), 4) if values else None
    group["latency"] = {"p50_ms": latency_p50, "p95_ms": latency_p95}
    group["production_shaped"] = group["eval_coverage_forced_tasks"] == 0
    for negative in group["negative_by_kind"].values():
        available = int(negative["available_tasks"])
        negative["clean@5"] = round(negative["clean_tasks"] / available, 4) \
            if available else None
        negative["non_clean_source_types"] = dict(
            sorted(negative["non_clean_source_types"].items())
        )


def _policy_result(
    task: Mapping[str, Any],
    row: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, float]],
    case_variants: Mapping[str, Mapping[str, Any]],
    variant: str | None,
    *,
    requested_variant: str,
    configured: bool,
) -> dict[str, Any]:
    """Build the common audit record for a selected display variant."""
    expectations = _expectation_sets(task)
    available = variant is not None and _variant_available(row, variant, variants)
    refs = list(case_variants[variant]["refs"]) if available and variant is not None else []
    latency = _variant_latency_ms(row, variant, variants) if variant is not None else None
    return {
        "requested_variant": requested_variant,
        "variant": variant,
        "configured": configured,
        "fallback": variant != requested_variant,
        "available": available,
        "refs": refs,
        "recall@5": case_variants[variant]["recall@5"] if available and variant is not None else None,
        "recall@10": case_variants[variant]["recall@10"] if available and variant is not None else None,
        "mrr": case_variants[variant]["mrr"] if available and variant is not None else None,
        "clean@5": (1.0 if not refs[:5] else 0.0) if available and not expectations else None,
        "source_types": _ref_source_types(refs[:5]),
        "forbidden_hits@5": _forbidden_hits(task, refs),
        "scope_status": _scope_status(task, row),
        "estimated_latency_ms": round(latency, 1) if latency is not None else None,
    }


def _runtime_display_result(
    task: Mapping[str, Any], row: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Read the actual display route recorded by ``HybridRetriever``.

    A guarded fallback can apply additional memory-focused filtering after RRF.
    Replaying lexical/equal weights alone therefore cannot establish what the
    runtime would have exposed to a user; consume ``display_refs`` when the
    retrieval runner recorded them.
    """
    shadow = row.get("hybrid_shadow")
    if not isinstance(shadow, Mapping) or "display_refs" not in shadow:
        return None
    status = str(shadow.get("hybrid_status") or "available")
    coverage = shadow.get("coverage_contract")
    required_groups = coverage.get("required_groups") if isinstance(coverage, Mapping) else []
    eval_coverage_forced = bool(required_groups)
    available = status in {"available", "mixed"}
    refs = [str(ref) for ref in shadow.get("display_refs", []) if str(ref).strip()]
    expectations = _expectation_sets(task)
    latency = shadow.get("latency_ms")
    try:
        latency_ms = round(float(latency), 1) if latency is not None else None
    except (TypeError, ValueError):
        latency_ms = None
    return {
        "requested_variant": str(shadow.get("display_mode") or "unknown"),
        "variant": str(shadow.get("display_mode") or "unknown"),
        "configured": True,
        "fallback": False,
        "available": available,
        "refs": refs if available else [],
        "recall@5": _recall_at(expectations, refs, 5) if available else None,
        "recall@10": _recall_at(expectations, refs, 10) if available else None,
        "mrr": _mrr(expectations, refs) if available else None,
        "clean@5": (1.0 if not refs[:5] else 0.0)
        if available and not expectations else None,
        "source_types": _ref_source_types(refs[:5]),
        "forbidden_hits@5": _forbidden_hits(task, refs),
        "scope_status": _scope_status(task, row),
        "estimated_latency_ms": latency_ms,
        "mode": str(shadow.get("display_mode") or "unknown"),
        "eval_coverage_forced": eval_coverage_forced,
    }


def sweep_rrf(
    retrieval_report: Mapping[str, Any],
    tasks: Iterable[Mapping[str, Any]], *,
    variants: Mapping[str, Mapping[str, float]] = DEFAULT_VARIANTS,
    query_type_policy: Mapping[str, str] = DEFAULT_QUERY_TYPE_POLICY,
    answer_gate_report: Mapping[str, Any] | None = None,
    runtime_guarded_fallback: bool = False,
    k: int = 10,
) -> dict[str, Any]:
    task_rows = [migrate_v1_task(dict(task)) for task in tasks]
    audit = validate_tasks(task_rows)
    if not audit["valid"]:
        raise ValueError("invalid retrieval task set: " + "; ".join(audit["errors"][:8]))
    rows_by_id = {
        str(row.get("id")): row
        for row in retrieval_report.get("per_query", [])
        if isinstance(row, Mapping)
    }
    variant_names = list(variants)
    policy_variants = dict(variants)
    cases: list[dict[str, Any]] = []
    buckets: dict[str, dict[str, dict[str, list[float]]]] = {}
    for task in task_rows:
        task_id = str(task.get("id") or "")
        row = rows_by_id.get(task_id)
        if not isinstance(row, Mapping):
            continue
        available = _available(row)
        expectations = _expectation_sets(task)
        routes = {
            "lexical": _route_rows(row, "lexical"),
            "vector": _route_rows(row, "vector"),
        }
        case_variants: dict[str, dict[str, Any]] = {}
        for name, weights in variants.items():
            refs = weighted_rrf(routes, weights=weights, k=k)
            case_variants[name] = {
                "available": _variant_available(row, name, variants),
                "refs": refs,
                "recall@5": _recall_at(expectations, refs, 5),
                "recall@10": _recall_at(expectations, refs, 10),
                "mrr": _mrr(expectations, refs),
                "any_expected_ref_hit": bool(expectations) and any(
                    _ref_matches(expected, actual)
                    for group in expectations for expected in group
                    for actual in refs
                ),
            }
        query_type = str(task.get("query_type") or "unknown")
        selected_policy, requested_policy, policy_configured = _select_policy_variant(
            query_type_policy, query_type, policy_variants,
        )
        negative_kind = str(row.get("negative_kind") or "") if not expectations else ""
        policy_result = _policy_result(
            task, row, policy_variants, case_variants, selected_policy,
            requested_variant=requested_policy,
            configured=policy_configured,
        )
        case = {
            "id": task_id,
            "query_type": query_type,
            "positive": bool(expectations),
            "negative_kind": negative_kind or None,
            "route_status": str((row.get("rrf") or {}).get("status") or "available")
            if isinstance(row.get("rrf"), Mapping) else "available",
            "available": available,
            "variants": case_variants,
            "policy": policy_result,
        }
        display = _runtime_display_result(task, row)
        if display is not None:
            case["runtime_display"] = display
        if runtime_guarded_fallback:
            enabled = _semantic_fallback_allowed(str(task.get("query") or ""))
            runtime_variant = "equal" if enabled else "lexical"
            case["runtime_guarded_fallback"] = {
                **_policy_result(
                    task, row, policy_variants, case_variants, runtime_variant,
                    requested_variant=runtime_variant,
                    configured=True,
                ),
                "enabled": enabled,
                "reason": "semantic_fallback_allowed" if enabled else "lexical_guard",
            }
        cases.append(case)
        bucket = buckets.setdefault(query_type, {
            name: {"recall@5": [], "recall@10": [], "mrr": [], "clean@5": []}
            for name in variant_names
        })
        for name in variant_names:
            result = case_variants[name]
            if expectations and result["available"]:
                bucket[name]["recall@5"].append(float(result["recall@5"]))
                bucket[name]["recall@10"].append(float(result["recall@10"]))
                bucket[name]["mrr"].append(float(result["mrr"]))
            if not expectations and result["available"]:
                bucket[name]["clean@5"].append(
                    1.0 if not result["refs"][:5] else 0.0
                )

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    summary: dict[str, Any] = {}
    for query_type, variant_values in sorted(buckets.items()):
        summary[query_type] = {}
        for name, values in variant_values.items():
            summary[query_type][name] = {
                "positive_tasks": len(values["recall@5"]),
                "negative_tasks": len(values["clean@5"]),
                "recall@5": mean(values["recall@5"]),
                "recall@10": mean(values["recall@10"]),
                "mrr": mean(values["mrr"]),
                "clean@5": mean(values["clean@5"]),
            }
    policy_groups: dict[str, dict[str, Any]] = {}
    policy_overall = _new_policy_group(variant="by_query_type", configured=True)
    for case in cases:
        policy = case["policy"]
        group = policy_groups.setdefault(
            case["query_type"],
            _new_policy_group(variant=policy["variant"], configured=bool(policy["configured"])),
        )
        _add_policy_case(group, case)
        _add_policy_case(policy_overall, case)
    for group in policy_groups.values():
        _finalize_policy_group(group)
    _finalize_policy_group(policy_overall)
    runtime_summary = None
    if runtime_guarded_fallback:
        runtime_summary = _new_policy_group(
            variant="runtime_guarded_fallback", configured=True,
        )
        for case in cases:
            runtime_case = {**case, "policy": case["runtime_guarded_fallback"]}
            _add_policy_case(runtime_summary, runtime_case)
        _finalize_policy_group(runtime_summary)
    display_summary = None
    display_by_mode: dict[str, dict[str, Any]] = {}
    display_cases = [case for case in cases if "runtime_display" in case]
    if display_cases:
        display_summary = _new_policy_group(variant="recorded_runtime_display", configured=True)
        for case in display_cases:
            display_case = {**case, "policy": case["runtime_display"]}
            _add_policy_case(display_summary, display_case)
            mode = str(case["runtime_display"]["mode"])
            by_mode = display_by_mode.setdefault(
                mode,
                _new_policy_group(variant=mode, configured=True),
            )
            _add_policy_case(by_mode, display_case)
        _finalize_policy_group(display_summary)
        for group in display_by_mode.values():
            _finalize_policy_group(group)
    answer_gate = None
    if isinstance(answer_gate_report, Mapping):
        raw_summary = answer_gate_report.get("summary")
        if isinstance(raw_summary, Mapping):
            selected_summary = raw_summary.get("answer_gate", raw_summary)
            answer_gate = dict(selected_summary) if isinstance(selected_summary, Mapping) else {
                "status": "unknown",
            }
        else:
            answer_gate = {"status": "unknown"}
        answer_gate["evidence"] = {
            "source": "external_report",
            "schema": answer_gate_report.get("schema"),
            "generated_at": answer_gate_report.get("generated_at"),
            "mode": answer_gate_report.get("mode"),
            "cases": len(answer_gate_report.get("cases", []))
            if isinstance(answer_gate_report.get("cases"), list) else None,
            "report_sha256": hashlib.sha256(
                json.dumps(
                    answer_gate_report,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
    meta = retrieval_report.get("meta")
    return {
        "schema": SCHEMA,
        "k": k,
        "variants": {name: dict(weights) for name, weights in variants.items()},
        "query_type_policy": dict(sorted(query_type_policy.items())),
        "tasks": len(cases),
        "summary": {
            "by_query_type": summary,
            "query_type_policy": dict(sorted(policy_groups.items())),
            "policy": policy_overall,
            **({"runtime_guarded_fallback": runtime_summary}
               if runtime_summary is not None else {}),
            **({"runtime_display": display_summary,
                "runtime_display_by_mode": dict(sorted(display_by_mode.items()))}
               if display_summary is not None else {}),
            "policy_safety": {
                "forbidden_hits@5": sum(int(case["policy"]["forbidden_hits@5"]) for case in cases),
                "scope": {
                    status: sum(int(case["policy"]["scope_status"] == status) for case in cases)
                    for status in ("consistent", "mismatch", "unknown", "not_applicable")
                },
            },
        },
        "answer_gate": answer_gate,
        "snapshot": {
            key: meta.get(key)
            for key in ("vault_root", "git_commit", "snapshot_at", "embedding_model")
            if isinstance(meta, Mapping) and meta.get(key) is not None
        },
        "notes": {
            "offline_only": True,
            "not_applicable": "fixture rows are retained but excluded from available metrics",
            "query_type_policy": "diagnostic only; this policy does not change runtime routing or production defaults",
            "runtime_guarded_fallback": "when requested, replay the current semantic fallback classifier; diagnostic only",
            "runtime_display": "when the input report includes display_refs, measure the route actually exposed by HybridRetriever",
            "production_shaped": "false when the recorded display route reserved expected refs through an eval coverage contract",
            "latency": "policy latency is an offline estimate from recorded route timings; no provider call is made; recorded refs are only re-ranked",
        },
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep weighted RRF on a frozen retrieval report")
    parser.add_argument("--retrieval-report", required=True)
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--answer-gate-report",
        help="optional external answer-gate report; recorded as evidence, not attributed to each policy",
    )
    parser.add_argument(
        "--query-type-policy",
        help="optional JSON object or JSON file mapping query_type to a variant name",
    )
    parser.add_argument(
        "--runtime-guarded-fallback",
        action="store_true",
        help="replay the existing semantic fallback classifier without changing runtime configuration",
    )
    args = parser.parse_args(argv)
    try:
        report = json.loads(Path(args.retrieval_report).read_text(encoding="utf-8"))
        answer_gate = None
        if args.answer_gate_report:
            answer_gate = json.loads(Path(args.answer_gate_report).read_text(encoding="utf-8"))
        policy = DEFAULT_QUERY_TYPE_POLICY
        if args.query_type_policy:
            policy_value = args.query_type_policy
            policy_path = Path(policy_value)
            if policy_path.exists():
                policy_value = policy_path.read_text(encoding="utf-8")
            loaded_policy = json.loads(policy_value)
            if not isinstance(loaded_policy, Mapping) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in loaded_policy.items()
            ):
                raise ValueError("query-type policy must be a JSON object of string to string")
            policy = dict(loaded_policy)
        result = sweep_rrf(
            report,
            load_tasks(args.tasks),
            query_type_policy=policy,
            answer_gate_report=answer_gate,
            runtime_guarded_fallback=bool(args.runtime_guarded_fallback),
            k=max(1, args.k),
        )
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[RRF_SWEEP] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"tasks": result["tasks"], "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_QUERY_TYPE_POLICY", "DEFAULT_VARIANTS", "SCHEMA", "sweep_rrf",
    "weighted_rrf", "main",
]
