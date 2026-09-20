"""Answer-level abstention and citation-gate evaluation.

The retrieval evaluator answers "did we retrieve a gold ref?".  This module
answers the separate safety question: given the candidates and a generated
answer, should the system answer, abstain, or surface a conflict?  It is
deliberately deterministic and reuses the production gates.  Task-local
``answer_probes`` may contain human/LLM answers; when absent, a conservative
synthetic probe is generated and the report is marked ``synthetic``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from agentlab.eval.rag_task_schema import migrate_v1_task, validate_tasks
from agentlab.rag.assess import (
    AnswerEvidence,
    evaluate_answer_gate,
    evaluate_generated_answer,
    _answer_citation_tokens,
    _ref_matches,
)


SCHEMA = "rag-answer-gate-eval-v1"
MIN_REAL_CASES = 20
DECISIONS = {"answerable", "insufficient", "conflicting"}

# A probe's transport label is not enough to establish real answer evidence.
# Fixture replays intentionally use the same answer-gate path as production,
# but their deterministic answer is synthetic and must never satisfy the
# production minimum-real-sample gate.  Keep the original ``source`` visible
# in each case while deriving eligibility from provenance as well.
_FIXTURE_PROVENANCE_KINDS = frozenset({
    "fixture", "fixture_replay", "deterministic", "synthetic",
})
_REAL_PROBE_SOURCES = frozenset({"human", "llm", "real"})


def _is_real_probe(probe: Mapping[str, Any]) -> bool:
    source = str(probe.get("source") or "synthetic").strip().lower()
    if source not in _REAL_PROBE_SOURCES:
        return False
    provenance = probe.get("provenance")
    if not isinstance(provenance, Mapping):
        return True
    kind = str(provenance.get("kind") or "").strip().lower()
    return kind not in _FIXTURE_PROVENANCE_KINDS


def _refs(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    return list(dict.fromkeys(str(value).strip().replace("\\", "/")
                            for value in values if str(value).strip()))


def _expected_decision(task: Mapping[str, Any], probe: Mapping[str, Any]) -> str:
    value = str(probe.get("expected_decision") or task.get("expected_decision") or "").strip().lower()
    if value in DECISIONS:
        return value
    answerability = str(probe.get("answerability") or task.get("answerability") or "").strip().lower()
    if answerability == "conflicting":
        return "conflicting"
    if answerability == "answerable" and task.get("expected_refs"):
        return "answerable"
    return "insufficient"


def _expected_abstention(task: Mapping[str, Any], probe: Mapping[str, Any], decision: str) -> bool:
    value = probe.get("expected_abstention", task.get("expected_abstention"))
    if isinstance(value, bool):
        return value
    return decision != "answerable"


def _allowed_refs(task: Mapping[str, Any], probe: Mapping[str, Any]) -> list[str]:
    values = probe.get("allowed_refs", task.get("allowed_refs"))
    refs = _refs(values)
    return refs if refs else _refs(task.get("expected_refs"))


def _forbidden_refs(task: Mapping[str, Any], probe: Mapping[str, Any]) -> list[str]:
    return _refs(probe.get("forbidden_refs", task.get("forbidden_refs")))


def _candidate_refs(row: Mapping[str, Any] | None, mode: str) -> list[str]:
    if not isinstance(row, Mapping):
        return []
    value = row.get(mode)
    if not isinstance(value, Mapping):
        return []
    return _refs(value.get("refs"))


def _effective_mode(task: Mapping[str, Any], row: Mapping[str, Any] | None,
                    requested: str) -> str:
    """Use the requested route, falling back to a task-specific available route.

    Isolation and other independent fixtures intentionally mark RRF/P2 as
    ``not_applicable``.  Forcing those rows through the global default would
    turn valid evidence into a synthetic no-candidate answer failure.
    """
    if not isinstance(row, Mapping):
        return requested
    value = row.get(requested)
    if isinstance(value, Mapping) and str(value.get("status") or "available") \
            not in {"not_applicable", "unavailable", "error"}:
        return requested
    for fallback in (
        str(task.get("route") or "").strip(),
        str(row.get("route") or "").strip(),
    ):
        fallback_value = row.get(fallback)
        if fallback and isinstance(fallback_value, Mapping) \
                and str(fallback_value.get("status") or "available") \
                not in {"not_applicable", "unavailable", "error"}:
            return fallback
    return requested


def _expected_ref_hit(task: Mapping[str, Any], candidates: Iterable[str]) -> bool | None:
    """Whether a positive task's expected evidence entered the candidate set."""
    expected = _refs(task.get("expected_refs"))
    if not expected:
        return None
    actual = _refs(candidates)
    return any(
        _ref_matches(gold, candidate)
        or _ref_matches(candidate, gold)
        for gold in expected for candidate in actual
    )


def _expected_ref_coverage(task: Mapping[str, Any], candidates: Iterable[str]) -> dict[str, Any]:
    """Return both any-hit and all-required coverage for a task.

    ``candidate_expected_ref_hit`` historically meant any-hit even for
    all-of tasks.  Keep that field for compatibility, but expose the complete
    coverage so retrieval misses are not confused with assessor failures.
    """
    expected = _refs(task.get("expected_refs"))
    actual = _refs(candidates)
    matched = [gold for gold in expected if any(
        _ref_matches(gold, candidate) or _ref_matches(candidate, gold)
        for candidate in actual
    )]
    policy = str(task.get("expected_policy") or "all").strip().lower()
    any_hit = bool(matched) if expected else None
    all_hit = bool(expected) and len(matched) == len(expected) if expected else None
    required_hit = any_hit if policy == "any" else all_hit
    return {
        "any": any_hit,
        "all": all_hit,
        "required": required_hit,
        "matched_refs": matched,
        "missing_refs": [ref for ref in expected if ref not in matched],
        "policy": policy,
    }


def _expected_ref_content_coverage(task: Mapping[str, Any], probe: Mapping[str, Any]) -> bool | None:
    """Whether matched expected refs carried substantive chunk content.

    A path/entry hit is not sufficient evidence when dedupe selects a heading-
    only chunk from a Markdown file.  Live probes record content shape without
    persisting the full candidate body; older probes have no such field and
    deliberately remain ``None`` for backwards-compatible attribution.
    """
    evidence = probe.get("candidate_evidence")
    if not isinstance(evidence, list):
        return None
    expected = _refs(task.get("expected_refs"))
    if not expected:
        return None
    matched = []
    for gold in expected:
        rows = [row for row in evidence if isinstance(row, Mapping) and
                _ref_matches(gold, str(row.get("ref") or ""))]
        if not rows:
            continue
        matched.append(any(int(row.get("content_non_heading_chars") or 0) > 0 for row in rows))
    if not matched:
        return False
    policy = str(task.get("expected_policy") or "all").strip().lower()
    return any(matched) if policy == "any" else all(matched)


def _scope(task: Mapping[str, Any]) -> dict[str, Any]:
    value = task.get("scope")
    return dict(value) if isinstance(value, Mapping) else {
        "project_id": str(task.get("project_id") or "default"),
        "session_id": "",
        "statuses": [],
        "include_archive": False,
    }


def _scope_match(task: Mapping[str, Any], probe: Mapping[str, Any]) -> bool | None:
    """Compare a live probe's corpus root with the task fixture when declared."""
    return _scope_match_with_root(task, probe, "")


def _scope_match_with_root(task: Mapping[str, Any], probe: Mapping[str, Any],
                           report_vault_root: str) -> bool | None:
    """Compare a probe root with task root, falling back to report metadata."""
    expected = str(task.get("vault_root") or report_vault_root or "").strip()
    provenance = probe.get("provenance")
    actual = str(provenance.get("vault_root") or "").strip() \
        if isinstance(provenance, Mapping) else ""
    if not expected or not actual:
        return None
    return os.path.normcase(os.path.normpath(expected.replace("\\", "/"))) == \
        os.path.normcase(os.path.normpath(actual.replace("\\", "/")))


def _report_provenance(report_meta: Mapping[str, Any] | None) -> dict[str, str]:
    """Extract the derived-index identity used by the frozen report."""
    if not isinstance(report_meta, Mapping):
        return {}
    chunking = report_meta.get("chunking") if isinstance(report_meta.get("chunking"), Mapping) else {}
    index = report_meta.get("index") if isinstance(report_meta.get("index"), Mapping) else {}
    return {
        "index_version": str(index.get("index_version") or chunking.get("index_version") or ""),
        "parser_version": str(index.get("parser_version") or chunking.get("parser_version") or ""),
        "chunk_strategy_version": str(
            index.get("chunk_strategy_version") or chunking.get("strategy") or ""
        ),
        "embedding_model": str(index.get("embedding_model") or report_meta.get("embedding_model") or ""),
    }


def _probe_provenance_match(probe: Mapping[str, Any], expected: Mapping[str, str]) -> bool | None:
    """Reject a live probe produced from a different derived index."""
    provenance = probe.get("provenance")
    if not isinstance(provenance, Mapping) or not expected:
        return None
    declared = {key: str(provenance.get(key) or "") for key in expected}
    present = {key: value for key, value in declared.items() if value}
    if not present:
        return None
    return all(not expected.get(key) or value == expected.get(key)
               for key, value in present.items())


def _items(refs: Iterable[str], scope: Mapping[str, Any]) -> list[dict[str, Any]]:
    project_id = str(scope.get("project_id") or "")
    session_id = str(scope.get("session_id") or "")
    return [{"ref": ref, "status": "active", "project_id": project_id,
             "session_id": session_id} for ref in refs]


def _required_refs(task: Mapping[str, Any], allowed: list[str], decision: str) -> list[str]:
    if decision != "answerable" or not allowed:
        return []
    # evaluate_answer_gate has an all-of required_refs contract.  For an any
    # qrel task, one allowed ref is sufficient evidence by definition.
    return allowed[:1] if task.get("expected_policy") == "any" else allowed


def _synthetic_answer(expected_abstention: bool, allowed: list[str], candidates: list[str]) -> str:
    if expected_abstention:
        return "当前资料不足，无法确认。"
    citation = allowed[0] if allowed else (candidates[0] if candidates else "")
    if citation:
        return f"依据 [[{citation}]] 可回答该问题。"
    return "根据现有资料可回答该问题。"


def _synthetic_allowed_refs(allowed: list[str], candidates: list[str]) -> list[str]:
    """Return candidate refs that are actually within the task allow-list."""
    return [candidate for candidate in candidates if any(
        _ref_matches(candidate, expected) or _ref_matches(expected, candidate)
        for expected in allowed
    )]


def _probe(task: Mapping[str, Any], row: Mapping[str, Any] | None, mode: str,
           external: Iterable[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    supplied = [dict(item) for item in external if isinstance(item, Mapping)]
    if supplied:
        # Labels belong to the frozen task, not to an execution artifact.  A
        # live collection may report the model's assessment, but cannot
        # redefine what counts as a correct answer/refusal after the fact.
        for item in supplied:
            item.pop("expected_decision", None)
            item.pop("expected_abstention", None)
        return supplied
    explicit = task.get("answer_probes")
    if isinstance(explicit, list) and explicit:
        return [dict(item) for item in explicit if isinstance(item, Mapping)]
    candidates = _candidate_refs(row, mode)
    decision = _expected_decision(task, {})
    abstention = _expected_abstention(task, {}, decision)
    allowed = _allowed_refs(task, {})
    citeable = _synthetic_allowed_refs(allowed, candidates)
    if decision == "answerable" and not citeable:
        abstention = True
    return [{
        "id": f"{task.get('id', 'task')}:synthetic",
        "answer": _synthetic_answer(abstention, citeable or allowed, candidates),
        "candidate_refs": candidates,
        "expected_decision": decision,
        "expected_abstention": abstention,
        "answerability": str(task.get("answerability") or "answerable"),
        "allowed_refs": allowed,
        "forbidden_refs": _forbidden_refs(task, {}),
        "source": "synthetic",
    }]


def _evaluate_probe(task: Mapping[str, Any], row: Mapping[str, Any] | None,
                    probe: Mapping[str, Any], mode: str,
                    report_vault_root: str = "",
                    report_provenance: Mapping[str, str] | None = None,
                    allow_bounded_partial: bool = False) -> dict[str, Any]:
    expected = _expected_decision(task, probe)
    expected_abstention = _expected_abstention(task, probe, expected)
    candidates = _refs(probe.get("candidate_refs"))
    if not candidates:
        candidates = _candidate_refs(row, mode)
    allowed = _allowed_refs(task, probe)
    forbidden = _forbidden_refs(task, probe)
    scope = _scope(task)
    answerability = str(probe.get("answerability") or task.get("answerability") or "answerable").strip().lower()
    if expected == "conflicting":
        answerability = "conflicting"
    required = _required_refs(task, allowed, expected)
    items = _items(candidates, scope)
    pre = evaluate_answer_gate(
        str(task.get("query") or ""), items, scope=scope,
        required_refs=required, forbidden_refs=forbidden,
        answerability=answerability,
    )
    answer = str(probe.get("answer") or _synthetic_answer(expected_abstention, allowed, candidates))
    require_citation = bool(probe.get("require_citation", expected == "answerable"))
    assessment = str(probe.get("assessment") or (
        "answer" if answerability == "answerable" else "insufficient"
    )).strip().lower()

    # Exercise the same evidence ledger used by Runner, then compare it with
    # the direct helper.  A mismatch is a contract regression, not a quality
    # score, and remains visible in the report.
    evidence = AnswerEvidence(
        require_citation=require_citation,
        allow_bounded_partial=allow_bounded_partial,
    )
    evidence.observe("rag_retrieve", json.dumps({"items": items}, ensure_ascii=False))
    evidence.observe("rag_assess", json.dumps({
        "action": assessment,
        "answerability": answerability,
    }, ensure_ascii=False))
    observed = evidence.check(answer)
    direct = evaluate_generated_answer(
        answer, candidates, answerability=answerability,
        assessment=assessment,
        require_citation=require_citation,
        allow_bounded_partial=allow_bounded_partial,
    )
    post = observed or direct
    mismatched = bool(observed and (
        observed.decision != direct.decision or observed.reasons != direct.reasons
    ))
    citation_tokens = _answer_citation_tokens(answer)
    forbidden_hit = any(
        _ref_matches(blocked, token) or _ref_matches(token, blocked)
        for token in citation_tokens for blocked in forbidden
    )
    predicted = "answerable" if post.allowed else (
        "conflicting" if post.decision == "conflicting" else "insufficient"
    )
    coverage = _expected_ref_coverage(task, candidates)
    content_coverage = _expected_ref_content_coverage(task, probe)
    attribution = "none"
    if expected == "answerable" and not post.allowed:
        if coverage["required"] is False and coverage["any"] is False:
            attribution = "retrieval_miss"
        elif coverage["required"] is False:
            attribution = "required_ref_missing"
        elif content_coverage is False:
            attribution = "retrieval_content_gap"
        elif assessment in {"insufficient", "reformulate"}:
            attribution = "assessor_false_negative"
        else:
            attribution = "answer_generation_error"
    scope_match = _scope_match_with_root(task, probe, report_vault_root)
    provenance_match = _probe_provenance_match(probe, report_provenance or {})
    source = str(probe.get("source") or "synthetic").strip().lower()
    return {
        "id": str(probe.get("id") or f"{task.get('id', 'task')}:probe"),
        "task_id": str(task.get("id") or ""),
        "query_type": str(task.get("query_type") or "unknown"),
        "retrieval_mode": mode,
        "scope_match": scope_match,
        "provenance_match": provenance_match,
        "excluded_from_metrics": scope_match is False or provenance_match is False,
        "source": source,
        "real_probe": _is_real_probe(probe),
        "expected_decision": expected,
        "predicted_decision": predicted,
        "expected_abstention": expected_abstention,
        "predicted_abstention": not post.allowed,
        "assessment": assessment,
        "probe_answerability": answerability,
        "pre_gate": pre.to_dict(),
        "post_gate": post.to_dict(),
        "candidate_refs": candidates,
        "candidate_expected_ref_hit": coverage["any"],
        "candidate_expected_ref_any": coverage["any"],
        "candidate_expected_ref_all": coverage["all"],
        "candidate_required_ref_hit": coverage["required"],
        "candidate_expected_ref_contentful": content_coverage,
        "matched_expected_refs": coverage["matched_refs"],
        "missing_expected_refs": coverage["missing_refs"],
        "error_attribution": attribution,
        "answer": answer,
        "citation_tokens": citation_tokens,
        "citation_missing": "citation_missing" in post.reasons,
        "citation_out_of_scope": "citation_out_of_scope" in post.reasons,
        "forbidden_hit": forbidden_hit,
        "ledger_mismatch": mismatched,
    }


def evaluate_answer_gate_report(retrieval_report: Mapping[str, Any],
                                tasks: Iterable[Mapping[str, Any]], *,
                                mode: str = "rrf",
                                external_probes: Iterable[Mapping[str, Any]] = (),
                                allow_bounded_partial: bool = False) -> dict[str, Any]:
    task_rows = [migrate_v1_task(dict(task)) for task in tasks]
    audit = validate_tasks(task_rows)
    if not audit["valid"]:
        raise ValueError("invalid answer-gate task set: " + "; ".join(audit["errors"][:8]))
    rows_by_id = {
        str(row.get("id")): row for row in retrieval_report.get("per_query", [])
        if isinstance(row, Mapping)
    }
    report_meta = retrieval_report.get("meta")
    report_vault_root = str(report_meta.get("vault_root") or "").strip() \
        if isinstance(report_meta, Mapping) else ""
    report_provenance = _report_provenance(report_meta)
    probes_by_task: dict[str, list[Mapping[str, Any]]] = {}
    for probe in external_probes:
        if not isinstance(probe, Mapping):
            raise ValueError("external answer probe must be an object")
        task_id = str(probe.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("external answer probe is missing task_id")
        if not isinstance(probe.get("answer"), str):
            raise ValueError(f"external answer probe {task_id!r} is missing a string answer")
        probes_by_task.setdefault(task_id, []).append(probe)
    cases: list[dict[str, Any]] = []
    for task in task_rows:
        row = rows_by_id.get(str(task.get("id")))
        effective_mode = _effective_mode(task, row, mode)
        for probe in _probe(
            task, row, effective_mode, probes_by_task.get(str(task.get("id")), ())
        ):
            cases.append(_evaluate_probe(
                task, row, probe, effective_mode, report_vault_root,
                report_provenance=report_provenance,
                allow_bounded_partial=allow_bounded_partial,
            ))

    total = len(cases)
    answer_sources = [case["source"] for case in cases]
    real_cases = sum(bool(case.get("real_probe")) for case in cases)
    synthetic_cases = total - real_cases
    real_metric_cases = [case for case in cases
                         if case.get("real_probe")
                         and not case.get("excluded_from_metrics")]
    scope_mismatch_cases = sum(
        bool(case.get("excluded_from_metrics")) for case in cases
    )
    provenance_mismatch_cases = sum(
        case.get("provenance_match") is False for case in cases
    )
    metric_cases = real_metric_cases
    # A partial live collection is normally overlaid on the frozen retrieval
    # set.  Synthetic fallbacks remain valuable diagnostics, but they must not
    # bias the safety decision once actual generated answers are present.
    metrics_source = "real" if metric_cases else "synthetic"
    if not metric_cases:
        metric_cases = [case for case in cases if case["source"] not in {"human", "llm", "real"}]
    expected_abstain = sum(bool(case["expected_abstention"]) for case in metric_cases)
    predicted_abstain = sum(bool(case["predicted_abstention"]) for case in metric_cases)
    true_abstain = sum(bool(case["expected_abstention"] and case["predicted_abstention"])
                       for case in metric_cases)
    false_answers = sum(bool(case["expected_abstention"] and not case["predicted_abstention"])
                        for case in metric_cases)
    answerable = len(metric_cases) - expected_abstain
    false_refusals = sum(bool(not case["expected_abstention"] and case["predicted_abstention"])
                         for case in metric_cases)
    precision = true_abstain / predicted_abstain if predicted_abstain else 1.0
    recall = true_abstain / expected_abstain if expected_abstain else 1.0
    summary = {
        "status": "available" if total else "unavailable",
        "evaluation_mode": "real" if real_cases and not synthetic_cases else (
            "mixed" if real_cases else "synthetic"
        ),
        "cases": total,
        "real_cases": real_cases,
        "real_probe_cases": real_cases,
        "eligible_real_cases": len(real_metric_cases),
        "scope_mismatch_cases": scope_mismatch_cases,
        "provenance_mismatch_cases": provenance_mismatch_cases,
        "synthetic_cases": synthetic_cases,
        "metrics_source": metrics_source,
        "metrics_cases": len(metric_cases),
        "expected_abstention_cases": expected_abstain,
        "predicted_abstention_cases": predicted_abstain,
        "abstention_precision": round(precision, 4),
        "abstention_recall": round(recall, 4),
        "false_answer_rate": round(false_answers / expected_abstain, 4) if expected_abstain else 0.0,
        "false_refusal_rate": round(false_refusals / answerable, 4) if answerable else 0.0,
        "citation_missing_rate": round(
            sum(bool(case["citation_missing"]) for case in metric_cases
                if not case["expected_abstention"])
            / answerable, 4
        ) if answerable else 0.0,
        "citation_out_of_scope_rate": round(
            sum(bool(case["citation_out_of_scope"]) for case in metric_cases) / len(metric_cases), 4
        ) if metric_cases else 0.0,
        "forbidden_hit_rate": round(
            sum(bool(case["forbidden_hit"]) for case in metric_cases) / len(metric_cases), 4
        ) if metric_cases else 0.0,
        "ledger_mismatch_cases": sum(bool(case["ledger_mismatch"]) for case in metric_cases),
        "error_attribution": {
            kind: sum(case.get("error_attribution") == kind for case in metric_cases)
            for kind in ("retrieval_miss", "required_ref_missing",
                         "retrieval_content_gap", "assessor_false_negative",
                         "answer_generation_error")
        },
        "min_real_cases": MIN_REAL_CASES,
        "bounded_partial_enabled": bool(allow_bounded_partial),
    }
    # Explain false refusals without weakening the gate: a positive whose
    # expected ref was retrieved but was rejected by assessor/answer evidence
    # is a generation/assessment problem, while a miss is a retrieval problem.
    summary["candidate_hit_but_refused_cases"] = sum(
        bool(case.get("candidate_expected_ref_hit"))
        and not case["expected_abstention"]
        and case["predicted_abstention"]
        for case in metric_cases
    )
    summary["candidate_miss_refused_cases"] = sum(
        case.get("candidate_expected_ref_hit") is False
        and not case["expected_abstention"]
        and case["predicted_abstention"]
        for case in metric_cases
    )
    summary["candidate_hit_out_of_scope_cases"] = sum(
        bool(case.get("candidate_expected_ref_hit"))
        and bool(case.get("citation_out_of_scope"))
        for case in metric_cases
    )
    summary["candidate_expected_ref_any_cases"] = sum(
        case.get("candidate_expected_ref_any") is not None for case in metric_cases
    )
    summary["candidate_expected_ref_any_hits"] = sum(
        bool(case.get("candidate_expected_ref_any")) for case in metric_cases
        if case.get("candidate_expected_ref_any") is not None
    )
    summary["candidate_expected_ref_all_cases"] = sum(
        case.get("candidate_expected_ref_all") is not None for case in metric_cases
    )
    summary["candidate_expected_ref_all_hits"] = sum(
        bool(case.get("candidate_expected_ref_all")) for case in metric_cases
        if case.get("candidate_expected_ref_all") is not None
    )
    # Overall rates can hide a route-specific recall problem.  Keep the same
    # real/synthetic selection as the top-level metrics and expose a compact
    # per-query-type breakdown for the next retrieval/answer iteration.
    by_query_type: dict[str, dict[str, Any]] = {}
    for case in metric_cases:
        group = by_query_type.setdefault(str(case.get("query_type") or "unknown"), {
            "cases": 0,
            "expected_abstention_cases": 0,
            "predicted_abstention_cases": 0,
            "true_abstentions": 0,
            "false_answers": 0,
            "answerable_cases": 0,
            "false_refusals": 0,
            "citation_missing_cases": 0,
            "citation_out_of_scope_cases": 0,
            "forbidden_hit_cases": 0,
            "candidate_expected_ref_hits": 0,
            "candidate_expected_ref_cases": 0,
            "candidate_expected_ref_all_hits": 0,
            "candidate_expected_ref_all_cases": 0,
            "error_attribution": {
                "retrieval_miss": 0,
                "required_ref_missing": 0,
                "retrieval_content_gap": 0,
                "assessor_false_negative": 0,
                "answer_generation_error": 0,
            },
        })
        expected_abstain_case = bool(case["expected_abstention"])
        predicted_abstain_case = bool(case["predicted_abstention"])
        group["cases"] += 1
        group["expected_abstention_cases"] += int(expected_abstain_case)
        group["predicted_abstention_cases"] += int(predicted_abstain_case)
        group["true_abstentions"] += int(expected_abstain_case and predicted_abstain_case)
        group["false_answers"] += int(expected_abstain_case and not predicted_abstain_case)
        group["answerable_cases"] += int(not expected_abstain_case)
        group["false_refusals"] += int(not expected_abstain_case and predicted_abstain_case)
        group["citation_missing_cases"] += int(
            bool(case["citation_missing"]) and not expected_abstain_case
        )
        group["citation_out_of_scope_cases"] += int(bool(case["citation_out_of_scope"]))
        group["forbidden_hit_cases"] += int(bool(case["forbidden_hit"]))
        if case.get("candidate_expected_ref_hit") is not None:
            group["candidate_expected_ref_cases"] += 1
            group["candidate_expected_ref_hits"] += int(bool(case["candidate_expected_ref_hit"]))
        if case.get("candidate_expected_ref_all") is not None:
            group["candidate_expected_ref_all_cases"] += 1
            group["candidate_expected_ref_all_hits"] += int(bool(case["candidate_expected_ref_all"]))
        attribution = str(case.get("error_attribution") or "none")
        if attribution in group["error_attribution"]:
            group["error_attribution"][attribution] += 1
    for group in by_query_type.values():
        expected_count = group["expected_abstention_cases"]
        predicted_count = group["predicted_abstention_cases"]
        answerable_count = group["answerable_cases"]
        candidate_count = group["candidate_expected_ref_cases"]
        group.update({
            "abstention_precision": round(
                group["true_abstentions"] / predicted_count, 4
            ) if predicted_count else 1.0,
            "abstention_recall": round(
                group["true_abstentions"] / expected_count, 4
            ) if expected_count else 1.0,
            "false_answer_rate": round(
                group["false_answers"] / expected_count, 4
            ) if expected_count else 0.0,
            "false_refusal_rate": round(
                group["false_refusals"] / answerable_count, 4
            ) if answerable_count else 0.0,
            "citation_missing_rate": round(
                group["citation_missing_cases"] / answerable_count, 4
            ) if answerable_count else 0.0,
            "citation_out_of_scope_rate": round(
                group["citation_out_of_scope_cases"] / group["cases"], 4
            ) if group["cases"] else 0.0,
            "forbidden_hit_rate": round(
                group["forbidden_hit_cases"] / group["cases"], 4
            ) if group["cases"] else 0.0,
            "candidate_expected_ref_hit_rate": round(
                group["candidate_expected_ref_hits"] / candidate_count, 4
            ) if candidate_count else None,
            "candidate_expected_ref_all_rate": round(
                group["candidate_expected_ref_all_hits"] / group["candidate_expected_ref_all_cases"], 4
            ) if group["candidate_expected_ref_all_cases"] else None,
        })
        for key in (
            "true_abstentions", "false_answers", "answerable_cases",
            "false_refusals", "citation_missing_cases",
            "citation_out_of_scope_cases", "forbidden_hit_cases",
        ):
            group.pop(key, None)
    summary["by_query_type"] = dict(sorted(by_query_type.items()))
    return {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": mode,
        "summary": {"answer_gate": summary},
        "cases": cases,
    }


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_external_probes(path: str | Path) -> list[dict[str, Any]]:
    """Read a live collection report or a line-oriented probe artifact."""
    raw = Path(path).read_text(encoding="utf-8")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        document = _load_jsonl(path)
    if isinstance(document, Mapping):
        document = document.get("probes", document.get("cases", []))
    if not isinstance(document, list):
        raise ValueError("external probes must be a list or a report with probes")
    return [dict(row) for row in document if isinstance(row, Mapping)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate answer-level abstention and citation safety")
    parser.add_argument("--retrieval-report", required=True)
    parser.add_argument("--tasks", default=".ai/evals/rag_retrieval-v2.jsonl")
    parser.add_argument("--mode", default="rrf")
    parser.add_argument("--probes", help="live-agent or human probe JSON/JSONL overlay")
    parser.add_argument(
        "--allow-bounded-partial", action="store_true",
        help="offline experiment: accept cited answers that explicitly state evidence limits",
    )
    parser.add_argument("--out")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_answer_gate_report(
        json.loads(Path(args.retrieval_report).read_text(encoding="utf-8")),
        _load_jsonl(args.tasks), mode=args.mode,
        external_probes=load_external_probes(args.probes) if args.probes else (),
        allow_bounded_partial=args.allow_bounded_partial,
    )
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    # Windows PowerShell commonly exposes a GBK stdout; reports contain Vault
    # refs and headings, so keep the CLI deterministic and UTF-8 safe.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "MIN_REAL_CASES", "evaluate_answer_gate_report", "load_external_probes", "main"]
