"""P0/S1-0 检索规模盘点（只读，不改线上召回）。

扫描口径与 `VectorIndex.sync_vault()` 一致（排除目录 + 记忆归档区），
输出 JSON：文件总数、按顶层目录/文档类型分布、字节数、段落数、
按当前 legacy-600 策略的预计 chunk 数、长度分位数、最近 7/30 天修改量、
超长文件清单。用于《记忆系统机制与方法》§9.1 的 S0/S1 规模判定。

用法：python -m agentlab.scripts.rag_inventory --vault-root C:/path/to/your/obsidian-vault [--out PATH]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

# 与 rag/vector_index.py 的排除口径保持一致（含 OPT-227 的记忆归档区）
_EXCLUDE_DIRS = {".obsidian", ".agent-brain", ".trash", ".tmp", "node_modules"}
_MEMORY_ARCHIVE_REL = "ark/memory/archive/"

# legacy-600 分块近似：与 vector_index 的 chunk_text 相同的空行分段 + 600 字符贪心合并
_CHUNK_CHARS = 600


def _legacy_chunk_count(text: str) -> int:
    """按 legacy 策略估算 chunk 数（近似：空行分段后 600 字符贪心合并）。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = 0, 0
    for para in paragraphs:
        for piece in [para[i:i + _CHUNK_CHARS]
                      for i in range(0, len(para), _CHUNK_CHARS)] or [""]:
            if buf and buf + len(piece) > _CHUNK_CHARS:
                chunks += 1
                buf = len(piece)
            else:
                buf += len(piece)
    if buf:
        chunks += 1
    return max(chunks, 1 if text.strip() else 0)


def inventory(vault_root: str) -> dict:
    root = Path(vault_root)
    files: list[dict] = []
    now_ts = max((root / "Inbox").stat().st_mtime, 1) if (root / "Inbox").exists() else 0
    import time
    now_ts = time.time()

    for p in root.rglob("*.md"):
        rel = p.relative_to(root)
        if any(part in _EXCLUDE_DIRS for part in rel.parts):
            continue
        if rel.as_posix().startswith(_MEMORY_ARCHIVE_REL):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            stat = p.stat()
        except OSError:
            continue
        paragraphs = len([x for x in re.split(r"\n\s*\n", text) if x.strip()])
        files.append({
            "path": rel.as_posix(),
            "top": rel.parts[0] if len(rel.parts) > 1 else "(root)",
            "bytes": stat.st_size,
            "chars": len(text),
            "paragraphs": paragraphs,
            "est_chunks": _legacy_chunk_count(text),
            "mtime": stat.st_mtime,
        })

    total_files = len(files)
    total_chars = sum(f["chars"] for f in files)
    total_chunks = sum(f["est_chunks"] for f in files)
    char_list = sorted(f["chars"] for f in files)

    def _pct(seq, p):
        if not seq:
            return 0
        k = max(0, min(len(seq) - 1, int(round(p / 100 * (len(seq) - 1)))))
        return seq[k]

    by_top = Counter()
    by_top_bytes = Counter()
    by_top_chunks = Counter()
    for f in files:
        by_top[f["top"]] += 1
        by_top_bytes[f["top"]] += f["bytes"]
        by_top_chunks[f["top"]] += f["est_chunks"]

    week = now_ts - 7 * 86400
    month = now_ts - 30 * 86400
    recent_7d = sum(1 for f in files if f["mtime"] > week)
    recent_30d = sum(1 for f in files if f["mtime"] > month)

    oversized = sorted((f for f in files if f["chars"] > 6000),
                       key=lambda f: -f["chars"])[:15]

    return {
        "schema": "rag-inventory-v0",
        "scanner_version": "p4-provenance-v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "vault_root": str(root),
        "total_files": total_files,
        "total_bytes": sum(f["bytes"] for f in files),
        "total_chars": total_chars,
        "total_paragraphs": sum(f["paragraphs"] for f in files),
        "est_total_chunks": total_chunks,
        "chars_p50": _pct(char_list, 50),
        "chars_p95": _pct(char_list, 95),
        "chars_p99": _pct(char_list, 99),
        "avg_chars_per_chunk": round(total_chars / total_chunks, 1) if total_chunks else 0,
        "recent_modified_7d": recent_7d,
        "recent_modified_30d": recent_30d,
        "by_top_dir": {k: {"files": v, "bytes": by_top_bytes[k], "est_chunks": by_top_chunks[k]}
                       for k, v in sorted(by_top.items(), key=lambda kv: -kv[1])},
        "oversized_files": [{"path": f["path"], "chars": f["chars"],
                             "est_chunks": f["est_chunks"]} for f in oversized],
    }


def main():
    parser = argparse.ArgumentParser(description="S1-0 rag inventory (read-only)")
    parser.add_argument("--vault-root", required=True)
    parser.add_argument("--out", help="write JSON report to this path")
    args = parser.parse_args()
    report = inventory(args.vault_root)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"[INVENTORY] 已写入 {args.out}")
    print(text)


if __name__ == "__main__":
    main()
