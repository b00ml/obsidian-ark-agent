"""Deterministic memory governance and red-team evaluation.

The evaluator deliberately uses a temporary Markdown store and no model.  It
checks the properties that must remain true even when extraction or an LLM is
wrong: candidate/quarantine isolation, scope filtering, expiry, correction,
revocation and deletion propagation at the source-of-truth layer.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agentlab.memory.markdown_store import MemoryMarkdownStore


def _load(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("memory eval rows must be objects")
        rows.append(value)
    return rows


def _contains(store: MemoryMarkdownStore, query: str, *, project_id: str = "",
              expected_id: str = "") -> bool:
    rows = store.query(query, limit=20, project_id=project_id or None, track_access=False)
    return any(not expected_id or row.get("id") == expected_id for row in rows)


def _run_case(store: MemoryMarkdownStore, case: dict[str, Any]) -> tuple[bool, str]:
    kind = str(case.get("kind") or "").strip()
    query = str(case.get("query") or case.get("content") or "").strip()
    project = str(case.get("project_id") or "default")
    if kind == "candidate_first":
        mem_id = store.commit(case["content"], project_id=project, source="assistant",
                              candidate_first=True)
        ok = store.get(mem_id).get("status") == "candidate" and not _contains(
            store, query, project_id=project)
        return ok, "candidate is isolated from active reads"
    if kind == "promotion":
        mem_id = store.commit(case["content"], project_id=project,
                              source="assistant", candidate_first=True)
        before = store.query(query, limit=20, project_id=project, track_access=False)
        promoted = store.promote(mem_id, reason="eval explicit confirmation")
        after = store.query(query, limit=20, project_id=project, track_access=False)
        audit_path = store.vault_root / ".agent-brain" / "memory" / "events.jsonl"
        events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
                  if line.strip()] if audit_path.exists() else []
        event = next((item for item in reversed(events)
                      if item.get("memory_id") == mem_id and item.get("event") == "promote"), None)
        ok = (
            not any(row.get("id") == mem_id for row in before)
            and promoted
            and store.get(mem_id).get("status") == "active"
            and any(row.get("id") == mem_id for row in after)
            and event is not None
            and event.get("old_status") == "candidate"
            and event.get("new_status") == "active"
        )
        return ok, "explicit promotion is required before active reads"
    if kind == "external_instruction":
        mem_id = store.commit(case["content"], project_id=project, source="web")
        ok = store.get(mem_id).get("status") == "quarantine" and not _contains(
            store, query, project_id=project)
        return ok, "external instruction is quarantined"
    if kind == "scope":
        own = store.commit(case["content"], project_id=project)
        other = store.commit(case["other_content"], project_id=case["other_project"])
        rows = store.query(query, limit=20, project_id=project, track_access=False)
        ids = {row.get("id") for row in rows}
        return own in ids and other not in ids, "cross-project memory is hidden"
    if kind == "default_shared":
        shared = store.commit(case["content"], project_id="default")
        rows = store.query(query, limit=20, project_id=project, track_access=False)
        return shared in {row.get("id") for row in rows}, "default shared memory is visible"
    if kind == "expired":
        mem_id = store.commit(
            case["content"], project_id=project,
            valid_until=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        )
        rows = store.query(query, limit=20, project_id=project, track_access=False)
        return mem_id not in {row.get("id") for row in rows}, "expired memory is hidden"
    if kind == "correction":
        old = store.commit(case["content"], project_id=project)
        new = store.correct(old, case["replacement"], reason="eval correction",
                            project_id=project)
        rows = store.query(case["query"], limit=20, project_id=project, track_access=False)
        ids = {row.get("id") for row in rows}
        return bool(new) and old not in ids and new in ids, "successor replaces old memory"
    if kind == "revoke":
        mem_id = store.commit(case["content"], project_id=project)
        store.revoke(mem_id, reason="eval revoke")
        return not _contains(store, query, project_id=project), "revoked memory is hidden"
    if kind == "delete":
        mem_id = store.commit(case["content"], project_id=project)
        store.delete(mem_id, reason="eval delete", hard=True)
        return store.get(mem_id) is None and not _contains(store, query, project_id=project), \
            "deleted memory and read path are empty"
    if kind == "hypothesis":
        mem_id = store.commit(case["content"], project_id=project,
                              confidence="hypothesis", source="assistant")
        return store.get(mem_id).get("status") == "candidate" and not _contains(
            store, query, project_id=project), "hypothesis cannot become active"
    return False, f"unknown eval case kind: {kind}"


def run_memory_eval(dataset: str | Path) -> dict[str, Any]:
    cases = _load(dataset)
    failures: list[dict[str, Any]] = []
    by_kind: dict[str, dict[str, int]] = {}
    with tempfile.TemporaryDirectory(prefix="agentlab-memory-eval-") as root:
        for case in cases:
            kind = str(case.get("kind") or "unknown")
            bucket = by_kind.setdefault(kind, {"total": 0, "passed": 0})
            bucket["total"] += 1
            try:
                # Cases are independent red-team probes; no fixture may pass
                # merely because an earlier case happened to create a matching
                # memory in the same temporary Vault.
                case_root = Path(root) / str(case.get("id") or kind)
                passed, detail = _run_case(MemoryMarkdownStore(case_root), case)
            except Exception as exc:  # fixture errors are reported, not hidden
                passed, detail = False, f"{type(exc).__name__}: {exc}"
            if passed:
                bucket["passed"] += 1
            else:
                failures.append({"id": case.get("id", ""), "kind": kind, "detail": detail})
    total = len(cases)
    passed = total - len(failures)
    redteam_kinds = {"external_instruction", "scope", "expired", "correction", "revoke", "delete"}
    redteam_total = sum(v["total"] for k, v in by_kind.items() if k in redteam_kinds)
    redteam_passed = sum(v["passed"] for k, v in by_kind.items() if k in redteam_kinds)
    return {
        "schema": "memory-governance-eval-v1",
        "dataset": str(dataset),
        "total": total,
        "passed": passed,
        "failed": len(failures),
        "pass_rate": round(passed / total, 4) if total else 1.0,
        "redteam": {
            "total": redteam_total,
            "passed": redteam_passed,
            "pass_rate": round(redteam_passed / redteam_total, 4) if redteam_total else 1.0,
        },
        "by_kind": by_kind,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic memory governance eval")
    default = Path(__file__).resolve().parents[3] / ".ai" / "evals" / "memory_governance-v1.jsonl"
    parser.add_argument("--dataset", default=str(default))
    args = parser.parse_args(argv)
    result = run_memory_eval(args.dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
