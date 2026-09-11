"""收件箱工具：薄封装 inbox_collector（采集 + SQLite 队列读取）。

遵守 AGENTS.md：inbox_poll 不调用任何 LLM，poll 只拉邮件写队列。
"""
import os

from common import setup_paths


def _abs(config: dict, rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    return os.path.join(config.get("project_root", ""), rel)


def inbox_collect(config: dict) -> dict:
    """执行一轮 Agent Mail 采集（拉邮件 → URL 分流 → 写 pending 队列）。

    返回新增任务数；失败抛错（含 COLLECTOR_* 错误码根因）。
    """
    setup_paths(config)
    import inbox_poll
    cfg_path = _abs(config, config.get("inbox_config_path", "inbox_collector/config.json"))
    poll_cfg = inbox_poll.load_config(cfg_path)
    new_tasks = inbox_poll.poll(poll_cfg, dry_run=False)
    return {"status": "ok", "new_tasks": new_tasks,
            "message": f"采集完成，新增 {new_tasks} 条 pending 任务"}


def inbox_read_queue(config: dict) -> dict:
    """读取 SQLite 队列，返回状态统计与 pending 明细；旧 JSONL 自动导入。"""
    setup_paths(config)
    qpath = _abs(config, config.get("queue_path", "inbox/queue.jsonl"))
    spath = _abs(config, config.get("seen_path", "inbox/seen.txt"))
    dbpath = _abs(config, config.get("queue_db_path", "inbox/queue.db"))
    import inbox_poll
    store = inbox_poll.InboxQueueStore(dbpath, queue_path=qpath, seen_path=spath)
    store.migrate_legacy()
    by_status = store.stats()
    pending = store.list_tasks(status="pending", limit=50)
    return {"status": "ok", "total": sum(by_status.values()), "by_status": by_status,
            "pending": pending}
