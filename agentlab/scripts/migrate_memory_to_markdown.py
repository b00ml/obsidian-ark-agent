"""
一次性迁移工具：SQLite 记忆 → Markdown 文件

F5-011 Phase 2: 将现有 .agent-brain/memory/sessions.sqlite 记忆迁移到
ark/memory/ Markdown 文件，保持向后兼容。

使用方式：
    python -m agentlab.scripts.migrate_memory_to_markdown --vault-root C:/path/to/your/obsidian-vault --dry-run
    python -m agentlab.scripts.migrate_memory_to_markdown --vault-root C:/path/to/your/obsidian-vault

迁移逻辑：
1. 读取 SQLite 所有记忆（id/content/tags/source_session/created_at）
2. 根据 tags/content 推断 memory type（core/context/procedures/decisions/session）
3. 生成 Markdown 文件到 ark/memory/{type}/
4. 备份原 SQLite 为 sessions.sqlite.pre-f5011
5. 验证：新旧查询结果一致性
"""

import argparse
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# 添加 agentlab 到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentlab.memory.markdown_store import MemoryMarkdownStore


def infer_memory_type(content: str, tags: list[str]) -> str:
    """
    根据内容和标签推断记忆类型。

    启发式规则：
    - tags 含 user-preference/profile/coding-style → core
    - tags 含 framework/stack/project → context
    - tags 含 workflow/procedure/steps → procedures
    - tags 含 decision/trade-off/choice → decisions
    - 默认 → sessions
    """
    tags_lower = [t.lower() for t in tags]
    content_lower = content.lower()

    # Core: 用户偏好、画像
    core_keywords = ["用户偏好", "喜欢", "不喜欢", "user prefer", "profile", "coding style", "风格"]
    core_tags = ["user-preference", "profile", "coding-style", "preference"]
    if any(kw in content_lower for kw in core_keywords) or any(t in tags_lower for t in core_tags):
        return "core"

    # Procedures: 工作流、步骤
    procedure_keywords = ["工作流", "步骤", "流程", "操作", "workflow", "procedure", "step", "how to"]
    procedure_tags = ["workflow", "procedure", "steps", "操作"]
    if any(kw in content_lower for kw in procedure_keywords) or any(t in tags_lower for t in procedure_tags):
        return "procedures"

    # Decisions: 决策、权衡
    decision_keywords = ["决策", "决定", "权衡", "选择", "decision", "trade-off", "choice", "选用"]
    decision_tags = ["decision", "trade-off", "choice", "决策"]
    if any(kw in content_lower for kw in decision_keywords) or any(t in tags_lower for t in decision_tags):
        return "decisions"

    # Context: 项目背景、技术栈、框架
    context_keywords = ["项目", "框架", "技术栈", "使用", "project", "framework", "stack", "use"]
    context_tags = ["framework", "stack", "project", "context", "技术栈"]
    if any(kw in content_lower for kw in context_keywords) or any(t in tags_lower for t in context_tags):
        return "context"

    # 默认为 sessions（会话记忆）
    return "sessions"


def migrate_sqlite_to_markdown(
    vault_root: str,
    sqlite_path: str,
    dry_run: bool = False,
    project_id: str = "default"
) -> dict:
    """
    迁移 SQLite 记忆到 Markdown。

    Args:
        vault_root: Vault 根目录
        sqlite_path: SQLite 数据库路径
        dry_run: 仅预览，不实际写入
        project_id: 默认 project_id

    Returns:
        迁移统计信息
    """
    if not os.path.exists(sqlite_path):
        return {
            "status": "error",
            "message": f"SQLite not found: {sqlite_path}"
        }

    # 初始化 Markdown store
    md_store = MemoryMarkdownStore(vault_root)

    # 连接 SQLite
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 读取所有记忆
    cursor.execute("SELECT id, content, tags, source_session, created_at FROM memories ORDER BY id")
    rows = cursor.fetchall()

    stats = {
        "total": len(rows),
        "migrated": 0,
        "skipped": 0,
        "errors": [],
        "by_type": {}
    }

    migrated_ids = []

    for row in rows:
        try:
            sqlite_id = row["id"]
            content = row["content"].strip()
            tags_str = row["tags"] or ""
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]
            source_session = row["source_session"] or ""
            created_at = row["created_at"] or datetime.now(timezone.utc).isoformat()

            if not content:
                stats["skipped"] += 1
                continue

            # 推断记忆类型
            mem_type = infer_memory_type(content, tags)

            # 默认 importance（可根据历史 access 推断，这里简化为中等）
            importance = 5

            # 尝试从 created_at 推断重要性（早期记忆可能更重要）
            try:
                created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                days_old = (datetime.now(timezone.utc) - created_dt).days
                if days_old < 7:
                    importance = 7  # 近期记忆较重要
                elif days_old > 180:
                    importance = 4  # 久远记忆降权
            except Exception:
                pass

            if dry_run:
                print(f"[DRY-RUN] Would migrate: id={sqlite_id}, type={mem_type}, tags={tags[:3]}, "
                      f"content={content[:50]}...")
                stats["migrated"] += 1
            else:
                # 实际迁移
                mem_id = md_store.commit(
                    content=content,
                    tags=tags,
                    mem_type=mem_type,
                    project_id=project_id,
                    importance=importance,
                    source_session=source_session or f"sqlite-migration-{sqlite_id}"
                )

                migrated_ids.append((sqlite_id, mem_id))
                stats["migrated"] += 1

                print(f"Migrated: SQLite id={sqlite_id} → Markdown id={mem_id} (type={mem_type})")

            # 统计类型分布
            stats["by_type"][mem_type] = stats["by_type"].get(mem_type, 0) + 1

        except Exception as e:
            stats["errors"].append({
                "sqlite_id": row["id"],
                "error": str(e)
            })

    conn.close()

    # 备份原 SQLite（仅在非 dry-run 模式）
    if not dry_run and stats["migrated"] > 0:
        backup_path = f"{sqlite_path}.pre-f5011"
        if not os.path.exists(backup_path):
            shutil.copy2(sqlite_path, backup_path)
            stats["backup"] = backup_path
            print(f"\nBackup created: {backup_path}")

    stats["status"] = "success"
    return stats


def verify_migration(vault_root: str, sqlite_path: str, sample_size: int = 10) -> dict:
    """
    验证迁移结果：对比 SQLite 和 Markdown 查询结果。

    Args:
        vault_root: Vault 根目录
        sqlite_path: SQLite 数据库路径
        sample_size: 采样验证数量

    Returns:
        验证结果
    """
    md_store = MemoryMarkdownStore(vault_root)

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 随机采样一些记忆
    cursor.execute(f"SELECT content FROM memories ORDER BY RANDOM() LIMIT {sample_size}")
    samples = cursor.fetchall()

    results = {
        "total_samples": len(samples),
        "found_in_markdown": 0,
        "missing": [],
        "status": "success"
    }

    for row in samples:
        content = row["content"].strip()
        query_term = content.split()[0] if content.split() else content[:20]

        # 在 Markdown 中查询
        md_results = md_store.query(query_term, limit=20)

        # 检查是否找到相似内容
        found = any(content[:100] in mem["content"] or mem["content"][:100] in content
                    for mem in md_results)

        if found:
            results["found_in_markdown"] += 1
        else:
            results["missing"].append(content[:100])

    conn.close()

    results["recall_rate"] = results["found_in_markdown"] / results["total_samples"] if results["total_samples"] > 0 else 0

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Migrate SQLite memories to Markdown files (F5-011 Phase 2)"
    )
    parser.add_argument(
        "--vault-root",
        required=True,
        help="Obsidian vault root directory (e.g., C:/path/to/your/obsidian-vault)"
    )
    parser.add_argument(
        "--sqlite-path",
        help="Path to sessions.sqlite (default: <vault>/.agent-brain/memory/sessions.sqlite)"
    )
    parser.add_argument(
        "--project-id",
        default="default",
        help="Default project_id for migrated memories"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without writing files"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify migration by comparing query results"
    )

    args = parser.parse_args()

    vault_root = args.vault_root

    if args.sqlite_path:
        sqlite_path = args.sqlite_path
    else:
        sqlite_path = os.path.join(vault_root, ".agent-brain", "memory", "sessions.sqlite")

    print(f"=== F5-011 Phase 2: SQLite → Markdown Migration ===")
    print(f"Vault root: {vault_root}")
    print(f"SQLite path: {sqlite_path}")
    print(f"Project ID: {args.project_id}")
    print(f"Dry run: {args.dry_run}")
    print()

    # 执行迁移
    stats = migrate_sqlite_to_markdown(
        vault_root=vault_root,
        sqlite_path=sqlite_path,
        dry_run=args.dry_run,
        project_id=args.project_id
    )

    # 打印统计
    print("\n=== Migration Summary ===")
    print(f"Total: {stats['total']}")
    print(f"Migrated: {stats['migrated']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Errors: {len(stats['errors'])}")
    print(f"\nBy type:")
    for mem_type, count in stats["by_type"].items():
        print(f"  {mem_type}: {count}")

    if stats["errors"]:
        print(f"\nErrors:")
        for err in stats["errors"][:5]:
            print(f"  SQLite id={err['sqlite_id']}: {err['error']}")
        if len(stats["errors"]) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")

    if "backup" in stats:
        print(f"\nBackup: {stats['backup']}")

    # 验证（可选）
    if args.verify and not args.dry_run and stats["migrated"] > 0:
        print("\n=== Verification ===")
        verify_results = verify_migration(vault_root, sqlite_path, sample_size=10)
        print(f"Samples: {verify_results['total_samples']}")
        print(f"Found in Markdown: {verify_results['found_in_markdown']}")
        print(f"Recall rate: {verify_results['recall_rate']:.1%}")

        if verify_results["missing"]:
            print(f"\nMissing content samples:")
            for content in verify_results["missing"][:3]:
                print(f"  - {content}...")

    print("\n=== Migration Complete ===")
    return 0 if stats["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
