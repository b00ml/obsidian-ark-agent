"""Audit fixed retrieval tasks against a concrete index scope.

The full-vault task set intentionally contains targets from several runtime
routes.  A vector index with an explicit allow-list must not be scored as if it
contained those other routes.  This module therefore provides both a scope
audit and a deterministic scoped-task projection; the original task file is
never modified.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


def _norm_path(value: str) -> str:
    return str(value or "").strip().replace("\\", "/").lstrip("./")


def _load_indexed_paths(index_path: str | Path) -> set[str]:
    conn = sqlite3.connect(index_path)
    try:
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "files" not in tables:
            raise ValueError("index does not contain the required files table")
        return {
            _norm_path(str(row[0]))
            for row in conn.execute("SELECT path FROM files")
            if row[0]
        }
    except sqlite3.Error as exc:
        raise ValueError(f"cannot read index scope: {exc}") from exc
    finally:
        conn.close()


def _load_tasks(tasks_path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(tasks_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def classify_task(task: dict, indexed: set[str]) -> dict:
    """Return a machine-readable scope classification for one task."""
    refs = [str(ref) for ref in task.get("expected_refs", []) if ref]
    if not refs:
        category = "negative"
        inside, outside = [], []
    elif task.get("vault_root"):
        # Independent fixture roots are evaluated against their own index, not
        # against the canonical Vault index supplied to this audit.
        category = "not_applicable"
        inside, outside = [], refs
    else:
        inside = [
            ref for ref in refs
            if _norm_path(ref.split("#", 1)[0]) in indexed
        ]
        outside = [ref for ref in refs if ref not in inside]
        category = (
            "in_scope" if not outside else
            "out_of_scope" if not inside else
            "mixed_scope"
        )
    return {
        "id": task.get("id"),
        "query_type": task.get("query_type"),
        "category": category,
        "expected": len(refs),
        "in_scope": len(inside),
        "out_of_scope": len(outside),
        "in_scope_refs": inside,
        "out_of_scope_refs": outside,
    }


def scoped_tasks(tasks: list[dict], indexed: set[str], *, include_negative: bool = True) -> tuple[list[dict], dict]:
    """Project canonical tasks onto a concrete index without changing qrels.

    Fully in-scope positive tasks are retained.  Mixed and out-of-scope
    positives are excluded instead of silently dropping expected refs, since
    doing so would change the question's meaning.  Negative tasks remain
    useful for the scoped candidate unless they use an independent fixture.
    """
    selected: list[dict] = []
    classifications = []
    for task in tasks:
        row = classify_task(task, indexed)
        classifications.append(row)
        if row["category"] == "in_scope":
            copy = dict(task)
            copy["evaluation_scope"] = "index_scope"
            copy["scope_source_task"] = task.get("id")
            selected.append(copy)
        elif include_negative and row["category"] == "negative":
            copy = dict(task)
            copy["evaluation_scope"] = "index_scope_negative"
            copy["scope_source_task"] = task.get("id")
            selected.append(copy)
    return selected, {
        "schema": "rag-scope-manifest-v1",
        "selected_tasks": len(selected),
        "excluded_positive_tasks": sum(
            row["category"] in {"out_of_scope", "mixed_scope"}
            for row in classifications
        ),
        "not_applicable_tasks": sum(
            row["category"] == "not_applicable" for row in classifications
        ),
        "classifications": classifications,
    }


def audit(tasks_path: str | Path, index_path: str | Path) -> dict:
    tasks = _load_tasks(tasks_path)
    indexed = _load_indexed_paths(index_path)
    groups = Counter()
    rows = []
    for task in tasks:
        row = classify_task(task, indexed)
        groups[(str(row.get("query_type", "")), row["category"])] += 1
        rows.append(row)
    positive = [row for row in rows if row["expected"]]
    return {"schema": "rag-scope-audit-v1", "indexed_files": len(indexed),
            "tasks": len(tasks), "positive_tasks": len(positive),
            "in_scope_positive_tasks": sum(row["category"] == "in_scope" for row in positive),
            "out_of_scope_positive_tasks": sum(row["category"] == "out_of_scope" for row in positive),
            "mixed_scope_positive_tasks": sum(row["category"] == "mixed_scope" for row in positive),
            "not_applicable_tasks": sum(row["category"] == "not_applicable" for row in rows),
            "by_type": {f"{key[0]}::{key[1]}": value for key, value in sorted(groups.items())},
            "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit retrieval tasks against index scope")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--out")
    parser.add_argument(
        "--write-scoped",
        help="write a deterministic JSONL projection containing only fully in-scope positives and canonical negatives",
    )
    args = parser.parse_args()
    report = audit(args.tasks, args.index)
    if args.write_scoped:
        indexed = _load_indexed_paths(args.index)
        selected, manifest = scoped_tasks(_load_tasks(args.tasks), indexed)
        out_path = Path(args.write_scoped)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in selected),
            encoding="utf-8",
        )
        report["scoped_manifest"] = {
            "path": str(out_path),
            "schema": manifest["schema"],
            "selected_tasks": manifest["selected_tasks"],
            "excluded_positive_tasks": manifest["excluded_positive_tasks"],
            "not_applicable_tasks": manifest["not_applicable_tasks"],
        }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
