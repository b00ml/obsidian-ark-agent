#!/usr/bin/env python3
"""队列状态摘要（供 Trae Hook 注入上下文用）

读取 inbox/queue.db，输出待处理任务摘要（旧 JSONL 自动导入一次）。Hook 调用本脚本并把输出注入
UserPromptSubmit 上下文，使用户任何对话都感知"收件箱有 N 条待处理"。

用法: python inbox_collector/queue_status.py
输出: 单行摘要文本（无待处理时输出空）
"""
import os
import sys

from queue_store import InboxQueueStore

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    store = InboxQueueStore(
        os.path.join(PROJECT_ROOT, "inbox", "queue.db"),
        queue_path=os.path.join(PROJECT_ROOT, "inbox", "queue.jsonl"),
        seen_path=os.path.join(PROJECT_ROOT, "inbox", "seen.txt"),
    )
    store.migrate_legacy()
    pending = store.list_tasks(status="pending", limit=1000)

    if not pending:
        return

    bili = sum(1 for t in pending if t.get("type") == "bili")
    wechat = sum(1 for t in pending if t.get("type") == "wechat_article")
    print(f"📬 收件箱有 {len(pending)} 条待处理（B站 {bili} 条、公众号 {wechat} 条）。"
          f"回复\"处理收件箱\"开始批量整理。")


if __name__ == "__main__":
    main()
