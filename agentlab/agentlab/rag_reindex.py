"""Operational CLI for the P2 incremental RAG index.

This command owns the versioned P2 index used by ``rag_retrieve`` when the
configured backend selects it.  The legacy index remains an explicit rollback;
an operator still chooses when to scan and ingest the Vault.

Examples::

    python -m agentlab.rag_reindex --full
    python -m agentlab.rag_reindex --since 2026-09-13T00:00:00+08:00
    python -m agentlab.rag_reindex --status
    python -m agentlab.rag_reindex --retry-failed --limit 20
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from agentlab.rag.index_store import RagIndexStore
from agentlab.runtime.config import load_config


def _timestamp(value: str) -> float:
    """Parse Unix seconds or an ISO-8601 timestamp for ``--since``."""
    try:
        return float(value)
    except ValueError:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()


def _store(args: argparse.Namespace) -> tuple[RagIndexStore, Path, Path]:
    cfg = load_config(path=args.config)
    vault = Path(args.vault or cfg.vault_root)
    index = Path(args.index or (vault / ".agent-brain" / "rag-index-p2.sqlite"))
    embedder = None
    if (getattr(cfg.rag, "vector_enabled", False)
            and getattr(cfg.rag, "vector_mode", "shadow") != "off"
            and getattr(cfg.rag, "embed_base_url", "")):
        from agentlab.rag.embed import OpenAIEmbedder

        embedder = OpenAIEmbedder(
            cfg.rag.embed_base_url,
            cfg.rag.embed_model,
            api_key=cfg.rag.effective_key(),
            timeout=cfg.rag.embed_timeout,
        )
    stored_meta: dict[str, str] = {}
    if index.exists():
        conn = None
        try:
            conn = sqlite3.connect(index)
            stored_meta = {
                str(key): str(value)
                for key, value in conn.execute("SELECT key,value FROM index_meta")
            }
        except (OSError, sqlite3.Error):
            stored_meta = {}
        finally:
            if conn is not None:
                conn.close()
    strategy = getattr(args, "chunk_strategy", None) or stored_meta.get(
        "chunk_strategy_version"
    ) or getattr(cfg.rag, "chunk_strategy", "markdown-structure-v1")
    chunker = None
    match = re.search(r"markdown-structure-v2-min-?(32|64|80)$", strategy)
    if match:
        from agentlab.rag.chunker import make_chunker_v2
        chunker = make_chunker_v2(int(match.group(1)))
    store = RagIndexStore(
        index,
        embedder,
        vault_root=vault,
        embedding_model=getattr(cfg.rag, "embed_model", "") if embedder else None,
        index_version=getattr(args, "index_version", None) or stored_meta.get(
            "index_version"
        ) or getattr(cfg.rag, "index_version", "s1-p2-v1"),
        parser_version=strategy,
        chunker=chunker,
        chunk_strategy_version=strategy,
    )
    return store, vault, index


def _run_ingest(store: RagIndexStore, vault: Path, *, mode: str,
                since: float | None, force: bool, dry_run: bool,
                limit: int | None) -> dict:
    plan = store.plan_changes(vault, since=since, force=force)
    if dry_run:
        return {
            "mode": mode, "dry_run": True, "vault": str(vault),
            "total_files": plan.get("total_files", 0),
            "upserts": plan.get("upserts", 0), "deletes": plan.get("deletes", 0),
            "plan": plan,
        }
    queued = store.enqueue_changes(vault, since=since, force=force)
    run = store.process_queue(vault, limit=limit)
    plan_summary = {key: value for key, value in plan.items() if key != "changes"}
    return {
        "mode": mode, "dry_run": False, "vault": str(vault),
        "total_files": plan.get("total_files", 0),
        "upserts": plan.get("upserts", 0), "deletes": plan.get("deletes", 0),
        "enqueued": queued.get("enqueued", 0), "skipped": queued.get("skipped", 0),
        "files": run.get("updated", 0), "chunks": run.get("chunks", 0),
        "success": run.get("succeeded", 0), "failed": run.get("failed", 0),
        "vector_failed": run.get("vector_failed", 0),
        "dead": run.get("dead", 0), "deleted": run.get("deleted", 0),
        "elapsed_seconds": run.get("elapsed_seconds", 0.0),
        "embedding_model": store._embedding_model(),
        "parser_version": store.parser_version,
        "index_version": store.index_version,
        "plan": plan_summary,
        "queue": run,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P3 增量 RAG 索引运维入口")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--full", action="store_true", help="强制全量扫描并重建（会重新处理全部文件）")
    modes.add_argument("--reconcile", action="store_true", help="扫描并仅处理新增/变更/删除文件")
    modes.add_argument("--since", metavar="TIMESTAMP", help="只处理 mtime 晚于时间戳的文件")
    modes.add_argument("--status", action="store_true", help="查看索引、队列和 checkpoint 状态")
    modes.add_argument("--retry-failed", action="store_true", help="重置失败/死信项目并重新处理")
    parser.add_argument("--vault", help="Vault 根目录，默认读取 config")
    parser.add_argument("--index", help="SQLite 索引路径，默认 <vault>/.agent-brain/rag-index-p2.sqlite")
    parser.add_argument("--config", help="agentlab config.json 路径")
    parser.add_argument("--index-version", help="覆盖索引版本（维护 shadow/v2 索引时使用）")
    parser.add_argument("--chunk-strategy", help="覆盖 chunk 策略版本，如 markdown-structure-v2-min64")
    parser.add_argument("--dry-run", action="store_true", help="只输出扫描计划，不写队列或调用 embedding")
    parser.add_argument("--limit", type=int, default=0, help="最多处理项目数，0 表示不限制")
    parser.add_argument("--out", help="将本次运维结果写入隐藏 JSON 状态快照")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store, vault, index = _store(args)
    if args.status:
        result = store.queue_status(vault)
        result["dry_run"] = bool(args.dry_run)
    elif args.retry_failed:
        if args.dry_run:
            result = {
                "mode": "retry-failed", "dry_run": True,
                "failed_items": store.list_queue_failures(args.limit or 20),
            }
        else:
            queue_result = store.retry_queue(vault, limit=args.limit or 20, process=True)
            # Lexical-first updates record provider failures separately from
            # ingest queue failures. Retry both classes from one operator
            # command, while keeping their counters and evidence distinct.
            vector_result = store.retry_failures(vault, limit=args.limit or 20)
            result = {
                **queue_result,
                "queue": queue_result,
                "vector": vector_result,
                "vector_retried": vector_result.get("retried", 0),
                "vector_succeeded": vector_result.get("succeeded", 0),
                "vector_remaining": vector_result.get("remaining", 0),
            }
            result.update({"mode": "retry-failed", "dry_run": False})
        result.update({
            "vault": str(vault), "index": str(index),
            "embedding_model": store._embedding_model(),
            "index_version": store.index_version,
        })
    else:
        since = _timestamp(args.since) if args.since is not None else None
        mode = "full" if args.full else "reconcile" if args.reconcile else "since"
        result = _run_ingest(
            store, vault, mode=mode,
            since=since, force=bool(args.full), dry_run=args.dry_run,
            limit=args.limit or None,
        )
        result["index"] = str(index)
    if args.out:
        snapshot = {
            "schema": "rag-ops-status-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "status" if args.status else "retry-failed" if args.retry_failed else "ingest",
            "index": str(index),
            "vault": str(vault),
            "result": result,
        }
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["snapshot_path"] = str(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
