"""Versioned task contract for retrieval evaluation datasets.

The v2 contract makes corpus scope and graded relevance explicit while keeping
the v1 JSONL fields readable.  It intentionally validates task metadata only;
it never reads or modifies Vault content.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


SCHEMA = "rag-retrieval-task-v2"
POLICIES = {"all", "any"}
SCOPES = {"full_vault", "knowledge", "memory", "multi_route", "fixture"}
ANSWERABILITY = {"answerable", "absent", "conflicting", "unknown", "scope_denied"}
EXPECTED_DECISIONS = {"answerable", "insufficient", "conflicting"}


def _refs(task: dict) -> list[str]:
    return [str(ref).strip() for ref in task.get("expected_refs", []) if str(ref).strip()]


def infer_scope(task: dict) -> str:
    if task.get("vault_root"):
        return "fixture"
    paths = {ref.split("#", 1)[0].replace("\\", "/") for ref in _refs(task)}
    if not paths:
        return "full_vault"
    kinds = set()
    for path in paths:
        if path.startswith("wiki/"):
            kinds.add("knowledge")
        elif path.startswith("ark/memory/"):
            kinds.add("memory")
        else:
            kinds.add("other")
    if kinds == {"knowledge"}:
        return "knowledge"
    if kinds == {"memory"}:
        return "memory"
    if len(kinds) > 1:
        return "multi_route"
    return "full_vault"


def migrate_v1_task(task: dict) -> dict:
    """Add v2 fields without changing the v1 query or expected refs."""
    migrated = dict(task)
    migrated["dataset_schema"] = SCHEMA
    migrated["corpus_scope"] = str(task.get("corpus_scope") or infer_scope(task))
    migrated["route"] = str(task.get("route") or "local_combined")
    migrated["qrels"] = [
        {"ref": ref, "relevance": 1}
        for ref in _refs(task)
    ]
    # S1 answer-level fields are additive.  Existing negative rows are
    # conservatively classified as absent until an annotator supplies a more
    # specific label; retrieval metrics remain unchanged.
    migrated["graded_qrels"] = list(task.get("graded_qrels") or migrated["qrels"])
    migrated["answerability"] = str(
        task.get("answerability") or ("answerable" if _refs(task) else "absent")
    ).strip().lower()
    migrated["forbidden_refs"] = [
        str(ref).strip() for ref in (task.get("forbidden_refs") or []) if str(ref).strip()
    ]
    migrated["allowed_refs"] = [
        str(ref).strip() for ref in (task.get("allowed_refs") or []) if str(ref).strip()
    ]
    decision = str(task.get("expected_decision") or "").strip().lower()
    if decision not in EXPECTED_DECISIONS:
        decision = "answerable" if migrated["answerability"] == "answerable" and _refs(task) else "insufficient"
    migrated["expected_decision"] = decision
    expected_abstention = task.get("expected_abstention")
    migrated["expected_abstention"] = (
        bool(expected_abstention) if isinstance(expected_abstention, bool)
        else decision != "answerable"
    )
    # Optional answer probes are the only place where generated answer text is
    # stored.  Retrieval-only historical tasks remain valid and are evaluated
    # with an explicitly labelled synthetic probe by answer_gate_eval.
    if isinstance(task.get("answer_probes"), list):
        migrated["answer_probes"] = list(task["answer_probes"])
    migrated["scope"] = dict(task.get("scope") or {
        "project_id": str(task.get("project_id") or "default"),
        "session_id": str(task.get("session_id") or ""),
        "statuses": list((task.get("filters") or {}).get("statuses") or []),
        "include_archive": bool((task.get("filters") or {}).get("include_archive", False)),
    })
    migrated["migration"] = "v1_binary_refs"
    return migrated


def validate_task(task: dict) -> list[str]:
    errors: list[str] = []
    task_id = task.get("id", "<missing-id>")
    if not isinstance(task.get("id"), str) or not task["id"].strip():
        errors.append("id must be a non-empty string")
    if not isinstance(task.get("query"), str) or not task["query"].strip():
        errors.append("query must be a non-empty string")
    if not isinstance(task.get("expected_refs", []), list):
        errors.append("expected_refs must be a list")
    policy = task.get("expected_policy", "all")
    if policy not in POLICIES:
        errors.append(f"expected_policy must be one of {sorted(POLICIES)}")
    scope = task.get("corpus_scope")
    if scope not in SCOPES:
        errors.append(f"corpus_scope must be one of {sorted(SCOPES)}")
    if not isinstance(task.get("route"), str) or not task["route"].strip():
        errors.append("route must be a non-empty string")
    answerability = str(task.get("answerability") or "").strip().lower()
    if answerability not in ANSWERABILITY:
        errors.append(f"answerability must be one of {sorted(ANSWERABILITY)}")
    if not isinstance(task.get("forbidden_refs", []), list):
        errors.append("forbidden_refs must be a list")
    if not isinstance(task.get("allowed_refs", []), list):
        errors.append("allowed_refs must be a list")
    decision = str(task.get("expected_decision") or "").strip().lower()
    if decision not in EXPECTED_DECISIONS:
        errors.append(f"expected_decision must be one of {sorted(EXPECTED_DECISIONS)}")
    if not isinstance(task.get("expected_abstention"), bool):
        errors.append("expected_abstention must be a boolean")
    probes = task.get("answer_probes", [])
    if not isinstance(probes, list):
        errors.append("answer_probes must be a list")
    else:
        for index, probe in enumerate(probes):
            if not isinstance(probe, dict):
                errors.append(f"answer_probes[{index}] must be an object")
                continue
            probe_decision = str(probe.get("expected_decision") or decision).strip().lower()
            if probe_decision not in EXPECTED_DECISIONS:
                errors.append(
                    f"answer_probes[{index}].expected_decision must be one of {sorted(EXPECTED_DECISIONS)}"
                )
            if "answer" in probe and not isinstance(probe.get("answer"), str):
                errors.append(f"answer_probes[{index}].answer must be a string")
            if "expected_abstention" in probe and not isinstance(probe.get("expected_abstention"), bool):
                errors.append(f"answer_probes[{index}].expected_abstention must be a boolean")
    scope_value = task.get("scope", {})
    if not isinstance(scope_value, dict):
        errors.append("scope must be an object")
    elif (not isinstance(scope_value.get("project_id", ""), str)
          or not isinstance(scope_value.get("session_id", ""), str)
          or not isinstance(scope_value.get("statuses", []), list)
          or not isinstance(scope_value.get("include_archive", False), bool)):
        errors.append("scope contains invalid field types")
    qrels = task.get("qrels")
    if not isinstance(qrels, list):
        errors.append("qrels must be a list")
    else:
        seen: set[str] = set()
        expected = set(_refs(task))
        for index, qrel in enumerate(qrels):
            if not isinstance(qrel, dict) or not qrel.get("ref"):
                errors.append(f"qrels[{index}] must contain ref")
                continue
            ref = str(qrel["ref"]).strip()
            if ref in seen:
                errors.append(f"qrels duplicate ref: {ref}")
            seen.add(ref)
            try:
                relevance = int(qrel.get("relevance"))
            except (TypeError, ValueError):
                relevance = -1
            if relevance not in {0, 1, 2}:
                errors.append(f"qrels[{index}] relevance must be 0, 1 or 2")
        if not expected and qrels:
            errors.append("negative task must have empty qrels")
        if expected and not expected.issubset(seen):
            errors.append("every expected_ref must have a qrel")
        graded = task.get("graded_qrels", qrels)
        if not isinstance(graded, list):
            errors.append("graded_qrels must be a list")
    if task.get("corpus_scope") == "fixture" and not task.get("vault_root"):
        errors.append("fixture scope requires vault_root")
    return [f"{task_id}: {error}" for error in errors]


def validate_tasks(tasks: Iterable[dict]) -> dict:
    errors: list[str] = []
    ids: set[str] = set()
    rows = list(tasks)
    for task in rows:
        task_id = str(task.get("id", ""))
        if task_id in ids:
            errors.append(f"duplicate task id: {task_id}")
        ids.add(task_id)
        errors.extend(validate_task(task))
    counts = Counter(
        (str(task.get("corpus_scope", "missing")), str(task.get("query_type", "")))
        for task in rows
    )
    return {
        "schema": "rag-retrieval-task-audit-v1",
        "tasks": len(rows),
        "positive_tasks": sum(bool(_refs(task)) for task in rows),
        "negative_tasks": sum(not bool(_refs(task)) for task in rows),
        "qrels_tasks": sum(isinstance(task.get("qrels"), list) for task in rows),
        "by_scope_and_type": {
            f"{scope}::{query_type}": count
            for (scope, query_type), count in sorted(counts.items())
        },
        "errors": errors,
        "valid": not errors,
    }


def _load(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="migrate and validate retrieval task JSONL")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    migrated = [migrate_v1_task(task) for task in _load(args.input)]
    report = validate_tasks(migrated)
    report["input"] = str(Path(args.input))
    report["output"] = str(Path(args.output))
    if not report["valid"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in migrated),
        encoding="utf-8",
    )
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
