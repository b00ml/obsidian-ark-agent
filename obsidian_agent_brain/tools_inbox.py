"""收件箱工具：薄封装 inbox_collector（采集 + SQLite 队列读取）。

遵守 AGENTS.md：inbox_poll 不调用任何 LLM，poll 只拉邮件写队列。
"""
import os
import uuid

from common import setup_paths


def _stage_tracker(config: dict):
    setup_paths(config)
    from agentlab.contracts import ProcessStatus
    from agentlab.runtime.operation_context import current_operation_id
    from agentlab.runtime.stages import StageTracker
    operation_id = current_operation_id()
    run_id = operation_id or f"inbox-collect-{uuid.uuid4().hex[:16]}"
    return StageTracker(run_id, uuid.uuid4().hex[:16], operation_id=operation_id), ProcessStatus


def _abs(config: dict, rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    return os.path.join(config.get("project_root", ""), rel)


def inbox_collect(config: dict) -> dict:
    """执行一轮 Agent Mail 采集（拉邮件 → URL 分流 → 写 pending 队列）。

    返回新增任务数；失败抛错（含 COLLECTOR_* 错误码根因）。
    """
    tracker, statuses = _stage_tracker(config)
    tracker.record("accepted", "accepted", status=statuses.ACCEPTED)
    try:
        import inbox_poll
        operation_id = tracker.operation_id
        cfg_path = _abs(config, config.get("inbox_config_path", "inbox_collector/config.json"))
        poll_cfg = inbox_poll.load_config(cfg_path)
        emitted: set[str] = set()

        def record(stage_id: str) -> None:
            if stage_id in emitted:
                return
            emitted.add(stage_id)
            tracker.record(stage_id, stage_id,
                           status=getattr(statuses, stage_id.upper()))

        new_tasks = inbox_poll.poll(
            poll_cfg, dry_run=False, operation_id=operation_id, stage_callback=record)
        return {"status": "ok", "new_tasks": new_tasks,
                "operation_id": operation_id,
                "message": f"采集完成，新增 {new_tasks} 条 pending 任务",
                "stages": tracker.to_dict()}
    except Exception as exc:
        tracker.record("failed", "failed", status=statuses.FAILED,
                       error_code=type(exc).__name__)
        raise


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
