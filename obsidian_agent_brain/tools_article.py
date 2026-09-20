"""公众号文章工具：薄封装 article_summarizer.py（抓取 / 提取 / LLM 总结 / 成稿）。

处理器不直写 Vault 的约束由上层 agent 通过 vault_write 完成——本模块把成稿
渲染为笔记文本并返回，同时提供"一键入库"（内部走 vault_write 权限规则）。
"""
import os
import uuid

from common import setup_paths, vault_root
from tools_vault import vault_write


def _stage_tracker(config: dict, pipeline: str):
    setup_paths(config)
    from agentlab.contracts import ProcessStatus
    from agentlab.runtime.operation_context import current_operation_id
    from agentlab.runtime.stages import StageTracker
    operation_id = current_operation_id()
    run_id = operation_id or f"{pipeline}-{uuid.uuid4().hex[:16]}"
    return StageTracker(run_id, uuid.uuid4().hex[:16], operation_id=operation_id), ProcessStatus


def _record(tracker, stage_id: str, status, **kwargs) -> None:
    tracker.record(stage_id, stage_id, status=status, **kwargs)


def _with_stages(result: dict, tracker) -> dict:
    result["stages"] = tracker.to_dict()
    return result


def _import_article(config: dict):
    setup_paths(config)
    import article_summarizer
    return article_summarizer


def article_fetch(config: dict, url: str) -> dict:
    """抓取公众号文章并提取 {title, author, content}。"""
    tracker, statuses = _stage_tracker(config, "article-fetch")
    try:
        _record(tracker, "accepted", statuses.ACCEPTED)
        ar = _import_article(config)
        html = ar.fetch_article(url)
        _record(tracker, "fetched", statuses.FETCHED)
        article = ar.extract_article(html)
        _record(tracker, "parsed", statuses.PARSED)
        _record(tracker, "completed", statuses.COMPLETED)
        return _with_stages({"url": url, "title": article.get("title", ""),
                             "author": article.get("author", ""),
                             "content_len": len(article.get("content", "")),
                             "content": article.get("content", "")}, tracker)
    except Exception as exc:
        _record(tracker, "failed", statuses.FAILED, error_code=type(exc).__name__)
        raise


def article_summarize(config: dict, url: str, to_vault: bool = True) -> dict:
    """抓取 + LLM 总结 + 渲染 Obsidian 笔记。

    to_vault=True 时写入 Vault Inbox/（走目录权限规则），返回 note 路径；
    否则仅返回笔记文本。需要 visual_models.json 配置文本模型 API Key。
    """
    tracker, statuses = _stage_tracker(config, "article-summarize")
    try:
        _record(tracker, "accepted", statuses.ACCEPTED)
        ar = _import_article(config)
        vm_path = config.get("visual_models_path", "config/visual_models.json")
        if not os.path.isabs(vm_path):
            vm_path = os.path.join(config.get("project_root", ""), vm_path)

        from visual_analyzer import load_visual_config
        article = ar.extract_article(ar.fetch_article(url))
        _record(tracker, "fetched", statuses.FETCHED)
        _record(tracker, "parsed", statuses.PARSED)
        summary = ar.summarize_article(article, url, load_visual_config(vm_path))
        note_text = ar.build_note(article, summary, url)
        _record(tracker, "enriched", statuses.ENRICHED)

        title = summary.get("title") or article.get("title") or "article"
        filename = ar.kebab(title) + "-文章.md"
        result = {"url": url, "title": title, "summary": summary,
                  "note": note_text, "filename": filename}

        if to_vault:
            wr = vault_write(config, filename, note_text)
            result["note_path"] = wr["path"]
        _record(tracker, "completed", statuses.COMPLETED,
                artifact_refs=[str(result.get("note_path") or "")])
        return _with_stages(result, tracker)
    except Exception as exc:
        _record(tracker, "failed", statuses.FAILED, error_code=type(exc).__name__)
        raise
