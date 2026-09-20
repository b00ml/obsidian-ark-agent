"""P4.5-B shadow runner for Markdown structure-v2 candidates.

Each candidate gets a fresh SQLite derivation index and the same fixed
evaluation set.  The runner never changes the canonical v1 index, production
``rag_retrieve`` wiring, Vault Markdown, or calls a real embedding provider.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from agentlab.eval.run_rag_retrieval import _load_tasks, evaluate
from agentlab.rag.chunker import (
    V2_MIN_CHARS,
    build_v2_shadow_report,
    make_chunker_v2,
)
from agentlab.rag.hybrid import HybridRetriever
from agentlab.rag.index_store import RagIndexStore


def _variant(value: str) -> tuple[str, int]:
    text = str(value).strip().lower()
    if text in {"heading", "heading-only", "none", "0"}:
        return "heading-only", 0
    minimum = int(text)
    if minimum not in V2_MIN_CHARS:
        raise ValueError(f"min_chars must be one of {V2_MIN_CHARS} or heading")
    return f"min-{minimum}", minimum


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")


def _fresh_db(path: Path) -> None:
    """Remove only this runner's derived index files before a clean rebuild."""
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        candidate.unlink(missing_ok=True)


def run_variant(
    vault_root: str | Path,
    tasks_path: str | Path,
    work_dir: str | Path,
    variant: str,
    *,
    task_limit: int | None = None,
    candidate_k: int = 40,
) -> dict:
    label, min_chars = _variant(variant)
    root = Path(vault_root).resolve()
    tasks_file = Path(tasks_path).resolve()
    output_dir = Path(work_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / f"rag-index-p4.5b-{_slug(label)}.sqlite"
    _fresh_db(index_path)

    strategy = f"markdown-structure-v2-{label}"
    index_version = f"s1-p4.5b-{_slug(label)}"
    store = RagIndexStore(
        index_path,
        None,
        vault_root=root,
        index_version=index_version,
        parser_version=strategy,
        chunker=make_chunker_v2(min_chars),
        chunk_strategy_version=strategy,
    )
    sync = store.sync_vault(root)
    status = store.index_status(root)
    fragment = build_v2_shadow_report(root, min_chars=min_chars)
    tasks = _load_tasks(str(tasks_file), task_limit)
    cfg = {
        "vault_path": str(root),
        "_tasks_root": str(tasks_file.parent),
        "_p2_store": store,
        "_p2_index_path": str(index_path),
        "_hybrid_retriever": HybridRetriever(
            store,
            vector_mode="off",
            lexical_mode="on",
            candidate_k=candidate_k,
            dedupe_by="entry",
        ),
    }
    evaluation = evaluate(cfg, tasks)
    return {
        "variant": label,
        "min_chars": min_chars,
        "strategy": strategy,
        "index_version": index_version,
        "index_path": str(index_path),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sync": sync,
        "index": status,
        "fragments": {
            "files": fragment["files"],
            "legacy_total_chunks": fragment["legacy_total_chunks"],
            "v1_total_chunks": fragment["v1_total_chunks"],
            "v2_total_chunks": fragment["v2_total_chunks"],
            "short_fragments": fragment["short_fragments"],
            "max_chars": fragment["max_chars"],
            "over_hard": fragment["over_hard"],
            "coverage_avg": fragment["coverage_avg"],
            "coverage_min": fragment["coverage_min"],
            "coverage_below_95": fragment["coverage_below_95"],
            "heading_only": fragment["heading_only"],
            "merged": fragment["merged"],
            "diagnostics": fragment["diagnostics"],
        },
        "evaluation": evaluation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P4.5-B markdown-structure-v2 shadow runner")
    parser.add_argument("--vault-root", required=True)
    parser.add_argument("--tasks", default=".ai/evals/rag_retrieval.jsonl")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--variants",
        default="heading,32,64,80",
        help="comma-separated candidates: heading,32,64,80",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--candidate-k", type=int, default=40)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    results = [
        run_variant(
            args.vault_root,
            args.tasks,
            args.work_dir,
            variant,
            task_limit=args.limit,
            candidate_k=args.candidate_k,
        )
        for variant in variants
    ]
    report = {
        "schema": "rag-chunk-shadow-v2",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "vault_root": str(Path(args.vault_root).resolve()),
        "tasks": str(Path(args.tasks).resolve()),
        "variants": results,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": report["schema"],
        "variants": [
            {
                "variant": item["variant"],
                "chunks": item["index"]["chunks"],
                "coverage": item["index"]["coverage"],
                "recall@5": item["evaluation"]["summary"]["p2_lexical"].get("recall@5"),
                "p95_ms": item["evaluation"]["summary"]["p2_lexical"].get("p95_ms"),
                "short<=20": item["fragments"]["short_fragments"]["le20_ratio"],
                "short<=50": item["fragments"]["short_fragments"]["le50_ratio"],
            }
            for item in results
        ],
        "out": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
