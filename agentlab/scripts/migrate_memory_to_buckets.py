"""OPT-225 混合粒度迁移：存量单文件记忆 → 桶化布局。

规则（与《记忆系统机制与方法》§6 一致）：
- sessions/{YYYY-MM}/mem-*.md  → 聚合进 sessions/{YYYY-MM}.md 月桶（按原 created_at 月份）
- context/**/mem-*.md          → 聚合进 context/{project_id|default}.md 桶（自动候选层）
- core|decisions|procedures/   → 保留独立文件，重命名为语义文件名（{标题slug}-{id6}.md）

迁移前请自行备份 ark/memory（脚本不删除任何内容本体，只移动/重命名/聚合）。
用法：
  python -m agentlab.scripts.migrate_memory_to_buckets --vault-root C:/path/to/your/obsidian-vault [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentlab.memory.markdown_store import MemoryMarkdownStore  # noqa: E402

SLUG_RE = re.compile(r'[\\/:*?"<>|#\[\]]')


def _slug(content: str, mem_id: str) -> str:
    first = content.strip().splitlines()[0].lstrip("# ") if content.strip() else ""
    slug = SLUG_RE.sub("", first).strip()
    slug = re.sub(r"\s+", "-", slug)[:24].strip("-")
    return f"{slug}-{mem_id[-6:]}.md" if slug else f"{mem_id}.md"


def _bucket_rel(mem_type: str, project_id: str, month: str) -> str:
    if mem_type == "sessions":
        return f"sessions/{month}.md"
    return f"{mem_type}/{project_id or 'default'}.md"


def migrate(vault_root: str, dry_run: bool = False) -> dict:
    store = MemoryMarkdownStore(vault_root)
    root = store.memory_root
    buckets: dict[str, list[tuple[str, str]]] = {}  # rel桶路径 -> [(区块, 原文件相对路径)]
    renames: list[tuple[Path, Path]] = []
    removed: list[str] = []

    # 1) sessions 单文件 → 月桶
    sessions_dir = root / "sessions"
    if sessions_dir.exists():
        for md in sorted(sessions_dir.rglob("*.md")):
            if re.match(r"\d{4}-\d{2}\.md$", md.name):
                continue  # 已是桶
            text = md.read_text(encoding="utf-8")
            created = (re.search(r"created_at: '?(\d{4}-\d{2})", text) or [None, "1970-01"])[1]
            mem_id = (re.search(r"^id: (mem-[0-9a-f]+)", text, re.M) or [None, "mem-unknown"])[1]
            rel_bucket = _bucket_rel("sessions", "default", created)
            body = text.split("---", 2)[-1].strip()
            meta_src = (re.search(r"source_session: (.+)", text) or [None, ""])[1].strip()
            tags_m = re.findall(r"- (.+)", (re.search(r"tags:\n((?:- .+\n?)+)", text) or [None, ""])[1] if "tags:" in text else "")
            meta = (f"> importance=5 | tags={', '.join(t.strip() for t in tags_m)} | "
                    f"created={created} | project=default"
                    + (f" | source={meta_src}" if meta_src else ""))
            block = f"\n## {mem_id}\n{meta}\n\n{body}\n"
            buckets.setdefault(rel_bucket, []).append((block, md.relative_to(root).as_posix()))
            removed.append(md.relative_to(root).as_posix())

    # 2) context 单文件 → 项目桶
    ctx_dir = root / "context"
    if ctx_dir.exists():
        for md in sorted(ctx_dir.rglob("*.md")):
            if not md.name.startswith("mem-"):
                continue  # 已是桶（default.md / {pid}.md）或非记忆文件
            text = md.read_text(encoding="utf-8")
            pid = (re.search(r"project_id: (.+)", text) or [None, "default"])[1].strip() or "default"
            mem_id = (re.search(r"^id: (mem-[0-9a-f]+)", text, re.M) or [None, "mem-unknown"])[1]
            created = (re.search(r"created_at: '?(\d{4}-\d{2})", text) or [None, "1970-01"])[1]
            rel_bucket = _bucket_rel("context", pid, created)
            body = text.split("---", 2)[-1].strip()
            meta_src = (re.search(r"source_session: (.+)", text) or [None, ""])[1].strip()
            tags_m = re.findall(r"- (.+)", (re.search(r"tags:\n((?:- .+\n?)+)", text) or [None, ""])[1] if "tags:" in text else "")
            meta = (f"> importance=5 | tags={', '.join(t.strip() for t in tags_m)} | "
                    f"created={created} | project={pid}"
                    + (f" | source={meta_src}" if meta_src else ""))
            block = f"\n## {mem_id}\n{meta}\n\n{body}\n"
            buckets.setdefault(rel_bucket, []).append((block, md.relative_to(root).as_posix()))
            removed.append(md.relative_to(root).as_posix())

    # 3) core/decisions/procedures 语义重命名
    for mem_type in ("core", "decisions", "procedures"):
        tdir = root / mem_type
        if not tdir.exists():
            continue
        for md in sorted(tdir.glob("*.md")):
            text = md.read_text(encoding="utf-8")
            mem_id = (re.search(r"^id: (mem-[0-9a-f]+)", text, re.M) or [None, ""])[1]
            body = text.split("---", 2)[-1].strip()
            new_name = _slug(body, mem_id)
            if new_name != md.name:
                renames.append((md, tdir / new_name))

    report = {"buckets": {k: len(v) for k, v in buckets.items()},
              "renames": len(renames), "removed": len(removed), "dry_run": dry_run}

    if dry_run:
        report["bucket_files_preview"] = {k: v[0][1] for k, v in buckets.items()}
        return report

    # 执行：写桶 → 删原文件 → 重命名
    for rel, blocks in buckets.items():
        bucket = root / rel
        bucket.parent.mkdir(parents=True, exist_ok=True)
        if bucket.exists():
            text = bucket.read_text(encoding="utf-8").rstrip("\n")
        else:
            mt = rel.split("/")[0]
            month = re.match(r"sessions/(\d{4}-\d{2})", rel)
            head = (f"---\nbucket: true\ntype: {mt}\nproject_id: default\n"
                    f"month: {month.group(1) if month else 'default'}\n---\n"
                    f"# {mt} 候选记忆\n")
            text = head.rstrip("\n")
        for block, _src in blocks:
            text += "\n" + block
        tmp = bucket.with_suffix(".tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(bucket)
    for rel in removed:
        (root / rel).unlink()
    for old, new in renames:
        old.rename(new)

    report["index"] = str(store.build_index().relative_to(root))
    return report


def main():
    parser = argparse.ArgumentParser(description="OPT-225: bucket-ize existing memory files")
    parser.add_argument("--vault-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = migrate(args.vault_root, dry_run=args.dry_run)
    import json
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
