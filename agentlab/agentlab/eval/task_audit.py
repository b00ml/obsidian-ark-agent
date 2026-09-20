"""Audit retrieval task labels against the current Markdown corpus.

This is a dataset audit, not a retrieval evaluation.  It checks whether
expected refs exist and whether an ``absent`` label is contradicted by an
exact query phrase in the corpus.  Semantic sufficiency remains a human or
provider-level judgment and is reported separately.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from agentlab.eval.rag_task_schema import migrate_v1_task, validate_tasks


SCHEMA = "rag-task-corpus-audit-v1"
_WS_RE = re.compile(r"\s+")


def _normalise(value: Any) -> str:
    return _WS_RE.sub(" ", str(value or "")).strip().lower()


def _resolve_root(project_root: Path, task: Mapping[str, Any], default_root: Path) -> Path:
    raw = str(task.get("vault_root") or "").strip()
    if not raw:
        return default_root
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    direct = (project_root / candidate).resolve()
    if direct.exists():
        return direct
    return (project_root / ".ai" / "evals" / candidate).resolve()


def _ref_path(ref: str) -> str:
    return str(ref or "").split("#", 1)[0].replace("\\", "/").lstrip("./")


def _read_markdown(root: Path) -> tuple[list[Path], dict[str, str]]:
    files: list[Path] = []
    contents: dict[str, str] = {}
    if not root.exists():
        return files, contents
    for path in root.rglob("*.md"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        files.append(path)
        contents[str(path)] = _normalise(text)
    return files, contents


def audit_tasks(
    tasks: Iterable[Mapping[str, Any]],
    *,
    vault_root: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return deterministic corpus/label findings for a task set."""
    project = Path(project_root or Path.cwd()).resolve()
    default_root = Path(vault_root).resolve()
    rows = [migrate_v1_task(dict(task)) for task in tasks]
    contract = validate_tasks(rows)
    findings: list[dict[str, Any]] = []
    refs_checked = 0
    roots: dict[str, dict[str, Any]] = {}
    for task in rows:
        task_id = str(task.get("id") or "")
        root = _resolve_root(project, task, default_root)
        root_key = str(root)
        if root_key not in roots:
            files, contents = _read_markdown(root)
            roots[root_key] = {"files": files, "contents": contents}
        contents = roots[root_key]["contents"]
        expected_refs = [str(ref) for ref in task.get("expected_refs") or [] if str(ref).strip()]
        for ref in expected_refs:
            refs_checked += 1
            rel = _ref_path(ref)
            target = (root / rel).resolve()
            if root not in target.parents and target != root:
                findings.append({"severity": "error", "kind": "ref_escape",
                                 "task_id": task_id, "ref": ref})
                continue
            if not target.is_file():
                findings.append({"severity": "error", "kind": "missing_expected_ref",
                                 "task_id": task_id, "ref": ref,
                                 "path": rel})
                continue
            try:
                text = target.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                text = ""
            if not text.strip():
                findings.append({"severity": "error", "kind": "empty_expected_ref",
                                 "task_id": task_id, "ref": ref,
                                 "path": rel})

        answerability = str(task.get("answerability") or "").strip().lower()
        if answerability == "answerable" and not expected_refs:
            findings.append({"severity": "error", "kind": "answerable_without_refs",
                             "task_id": task_id})
        if answerability != "answerable" and not expected_refs:
            query = _normalise(task.get("query"))
            if len(query) >= 6 and any(query in text for text in contents.values()):
                findings.append({
                    "severity": "warning", "kind": "negative_label_conflict",
                    "task_id": task_id, "query": str(task.get("query") or ""),
                    "message": "exact normalized query phrase exists in corpus",
                })

    errors = [item for item in findings if item["severity"] == "error"]
    warnings = [item for item in findings if item["severity"] == "warning"]
    return {
        "schema": SCHEMA,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contract": contract,
        "valid": bool(contract.get("valid")) and not errors,
        "tasks": len(rows),
        "roots": sorted(roots),
        "refs_checked": refs_checked,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "missing_expected_refs": sum(i["kind"] == "missing_expected_ref" for i in findings),
            "empty_expected_refs": sum(i["kind"] == "empty_expected_ref" for i in findings),
            "answerable_without_refs": sum(i["kind"] == "answerable_without_refs" for i in findings),
            "negative_label_conflicts": len(warnings),
        },
    }


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit RAG task labels against a Markdown corpus")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--vault-root", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out")
    parser.add_argument("--strict", action="store_true", help="treat label conflicts as errors")
    args = parser.parse_args(argv)
    report = audit_tasks(load_tasks(args.tasks), vault_root=args.vault_root,
                         project_root=args.project_root)
    if args.strict and report["warnings"]:
        report["valid"] = False
        report["strict_failed"] = True
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if report["valid"] else 2


__all__ = ["SCHEMA", "audit_tasks", "load_tasks", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
