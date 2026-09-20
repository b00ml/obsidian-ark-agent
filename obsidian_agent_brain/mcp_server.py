#!/usr/bin/env python3
"""obsidian-brain MCP server 入口（FastMCP / stdio）。

启动:
  .venv\\Scripts\\python.exe obsidian_agent_brain/mcp_server.py
调试:
  npx @modelcontextprotocol/inspector -- .venv\\Scripts\\python.exe obsidian_agent_brain/mcp_server.py

注册 vault / bili / article / inbox / brain / memory 七组 20 个核心规格工具
（tool_registry.BRAIN_TOOL_SPECS），另显式注册 obsidian_* 5 个与 bili 异步任务
3 个；tools/list 实测总数以运行时为准（2026-09-12 实测 28）。每个 spec 工具的
description 末尾带 [contract] 摘要（permission/side_effects/idempotent），
供客户端程序化读取契约。薄封装复用既有 bili_summarizer / inbox_collector /
官方 Obsidian CLI，零重写。
"""
import os
import sys
import threading
import json
import time

# 脚本方式运行：保证同目录模块可被 flat import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from common import load_config, setup_paths  # noqa: E402
from tool_registry import BRAIN_TOOL_SPECS, spec_contract_line  # noqa: E402
import tools_article  # noqa: E402
import tools_bili  # noqa: E402
import tools_brain  # noqa: E402
import tools_inbox  # noqa: E402
import tools_memory  # noqa: E402
import tools_obsidian  # noqa: E402
import tools_vault  # noqa: E402

CONFIG = load_config()
setup_paths(CONFIG)

# The repository keeps the Agent Hub package in ``agentlab/agentlab`` while this
# MCP server is also executable as a flat script. Add the package root explicitly
# so both ``python -m`` and ``python obsidian_agent_brain/mcp_server.py`` resolve
# the shared state-store module.
_AGENTLAB_ROOT = os.path.join(CONFIG.get("project_root", ""), "agentlab")
if os.path.isdir(_AGENTLAB_ROOT) and _AGENTLAB_ROOT not in sys.path:
    sys.path.insert(0, _AGENTLAB_ROOT)
from agentlab.runtime.state_store import TaskStore  # noqa: E402


def _task_db_path(config: dict) -> str:
    configured = config.get("task_state_path")
    if configured:
        return configured if os.path.isabs(configured) else os.path.join(
            config.get("project_root", os.getcwd()), configured)
    return os.path.join(config.get("project_root", os.getcwd()),
                        "agentlab", "runtime", "state.db")


_TASK_STORE = TaskStore(_task_db_path(CONFIG))

mcp = FastMCP("obsidian-brain")


# ============ 动态注册 brain 工具（vault/bili/article/inbox/brain/memory 六组 21 工具） ============
def _register_brain_tools(mcp_instance: FastMCP, config: dict) -> None:
    """从 tool_registry.BRAIN_TOOL_SPECS 动态注册 20 个核心规格工具到 MCP。

    每个工具通过 @mcp.tool() 装饰器注册，函数体直接调用对应 tools_*.py 模块函数；
    description 末尾附 [contract] 摘要（permission/side_effects/idempotent）。
    """
    # 模块映射：module_suffix → 实际导入的模块对象
    modules = {
        "vault": tools_vault,
        "bili": tools_bili,
        "article": tools_article,
        "inbox": tools_inbox,
        "brain": tools_brain,
        "memory": tools_memory,
    }

    for spec in BRAIN_TOOL_SPECS:
        tool_name, module_suffix = spec.name, spec.module_suffix
        mod = modules.get(module_suffix)
        if mod is None or not hasattr(mod, tool_name):
            continue  # 跳过未实现函数（预留 spec），validate_specs 不检查存在性

        # 动态构造包装函数：闭包捕获 config 并调用原始工具函数
        fn = getattr(mod, tool_name)

        def make_wrapper(original_fn, cfg):
            def wrapper(*args, **kwargs):
                return original_fn(cfg, *args, **kwargs)
            wrapper.__name__ = original_fn.__name__
            # 保留原函数签名（剔除 config 参数）
            import inspect
            sig = inspect.signature(original_fn)
            params = [p for p in sig.parameters.values() if p.name != "config"]
            wrapper.__signature__ = sig.replace(parameters=params)
            return wrapper

        wrapped = make_wrapper(fn, config)
        wrapped.__doc__ = f"{spec.description} {spec_contract_line(spec)}"
        # 注册到 MCP（FastMCP 的 @mcp.tool() 等价于 mcp.tool()(wrapped)）
        mcp_instance.tool()(wrapped)


_register_brain_tools(mcp, CONFIG)


# ============ bili 异步任务管理（持久化队列，支持重启/重试/取消） ============
def _spawn_job(fn, *args, payload: dict, dedupe_key: str):
    task = _TASK_STORE.create_task("bili_transcribe", payload, dedupe_key=dedupe_key)
    job_id = task["id"]

    def _run():
        # Delivery is at-least-once. Retryable failures are claimed again until
        # max_attempts; a cancelled running task discards a late result.
        while True:
            claimed = _TASK_STORE.claim(job_id)
            if claimed is None:
                return
            if _TASK_STORE.is_cancelled(job_id):
                return
            try:
                result = fn(*args)
                _TASK_STORE.complete(job_id, result)
                return
            except Exception as exc:  # noqa: BLE001 转写失败留在持久化状态
                # Keep retries visible in SQLite and avoid a hot loop when a
                # dependency is unavailable. A future process can resume the
                # pending task once ``next_retry_at`` is due.
                retry_delay = min(30.0, 0.5 * (2 ** max(0, int(
                    _TASK_STORE.get(job_id).get("attempt", 1) - 1))))
                state = _TASK_STORE.fail(job_id, "BILI_TRANSCRIBE_FAILED", str(exc),
                                          retryable=True, retry_delay=retry_delay)
                if not state or state["status"] != "pending":
                    return
                time.sleep(retry_delay)

    # A duplicate request reuses the same durable task. Starting an additional
    # lightweight worker for a pending task is harmless because claim() has an
    # atomic status gate; only one worker can execute the handler.
    if task["status"] == "pending":
        threading.Thread(target=_run, name=f"mcp-task-{job_id}", daemon=True).start()
    response_status = "started" if task["status"] == "pending" else "existing"
    return job_id, task["status"], response_status


@mcp.tool()
def bili_transcribe_start(bvid: str, quality: str = "fast", trust_ai: bool = False,
                           whisper_model: str | None = None,
                           whisper_device: str | None = None,
                           beam: int | None = None,
                           vad: bool | None = None) -> dict:
    """异步启动 B站字幕提取（3层降级），立即返回 job_id；用 bili_job_status 轮询结果。

    whisper_model/whisper_device/beam/vad 透传给策略3(Whisper)，None 时自动决定；
    默认走 GPU 加速（有 CUDA）+ VAD 静音过滤。
    """
    payload = {"bvid": bvid, "quality": quality, "trust_ai": trust_ai,
               "whisper_model": whisper_model, "whisper_device": whisper_device,
               "beam": beam, "vad": vad}
    dedupe_key = "bili_transcribe:" + json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                  separators=(",", ":"))
    job_id, task_status, request_status = _spawn_job(
        tools_bili.bili_transcribe, CONFIG, bvid, quality, trust_ai,
        whisper_model, whisper_device, beam, vad,
        payload=payload, dedupe_key=dedupe_key)
    return {"status": request_status, "job_id": job_id,
            "task_status": task_status, "hint": "轮询 bili_job_status(job_id) 直到 done"}


@mcp.tool()
def bili_job_status(job_id: str) -> dict:
    """查询持久化任务：pending/running/done/error/cancelled/dead_letter/unknown。"""
    job = _TASK_STORE.get(job_id)
    if job is None:
        return {"status": "unknown", "job_id": job_id}
    status = {"succeeded": "done", "failed": "error"}.get(job["status"], job["status"])
    out = {"status": status, "job_id": job_id, "task_status": job["status"],
           "attempt": job["attempt"], "max_attempts": job["max_attempts"],
           "next_retry_at": job["next_retry_at"], "lease_until": job["lease_until"]}
    if job["status"] == "succeeded":
        out["result"] = job["result"]
    if job["error_code"]:
        out["error_code"] = job["error_code"]
        out["error"] = job["error_message"]
    return out


@mcp.tool()
def bili_job_cancel(job_id: str) -> dict:
    """协作式取消 B站异步任务；第三方下载/转写不会被强杀。"""
    job = _TASK_STORE.cancel(job_id)
    if job is None:
        return {"status": "unknown", "job_id": job_id}
    return {"status": job["status"], "job_id": job_id,
            "task_status": job["status"], "cancelled": job["status"] == "cancelled"}


# ============ obsidian 组（官方 Obsidian CLI 接入，优化设计文档4.0 执行线 #3） ============
# 白名单：只读检索 + 原子属性 + move/rename；CLI 不可用时一律返回 degraded，不抛异常。
@mcp.tool()
def obsidian_links(note: str) -> dict:
    """Obsidian CLI 检索指定笔记双链：出向 links + 入向 backlinks。CLI 缺失返回 degraded。"""
    return tools_obsidian.obsidian_links(CONFIG, note)


@mcp.tool()
def obsidian_health() -> dict:
    """Obsidian CLI 全库体检：unresolved（未解析链接）/ orphans / deadends 三项 best-effort。"""
    return tools_obsidian.obsidian_health(CONFIG)


@mcp.tool()
def obsidian_rename(old_path: str, new_name: str) -> dict:
    """Obsidian CLI 重命名笔记，自动更新全库引用（保持双链）。new_name 为纯文件名。"""
    return tools_obsidian.obsidian_rename(CONFIG, old_path, new_name)


@mcp.tool()
def obsidian_move(path: str, folder: str) -> dict:
    """Obsidian CLI 移动笔记到目录，自动更新双链。folder 为 Vault 内目标目录。"""
    return tools_obsidian.obsidian_move(CONFIG, path, folder)


@mcp.tool()
def obsidian_property_set(path: str, key: str, value: str) -> dict:
    """Obsidian CLI 原子更新 frontmatter 属性（property:set），不动正文。"""
    return tools_obsidian.obsidian_property_set(CONFIG, path, key, value)


def main() -> None:
    # Windows 终端编码容错（长任务标题含特殊字符时防 GBK 崩溃）
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    n = len(mcp._tool_manager.list_tools())
    print(f"[BRAIN] MCP server 就绪 obsidian-brain tools={n}", file=sys.stderr)
    mcp.run()  # 默认 stdio 传输


if __name__ == "__main__":
    main()
