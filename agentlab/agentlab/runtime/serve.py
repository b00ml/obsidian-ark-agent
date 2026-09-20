"""Agentlab HTTP/SSE 服务：让 Obsidian ark 面板可直接对话 agentlab。

暴露的 SSE 事件与 Hermes /v1/responses 兼容，复用 ark 的 agentChat 解析：
  - response.output_item.added/done（item.type=function_call / function_call_output）→ 工具轨迹
  - response.output_text.delta → 文本增量（当前按整条 assistant 消息输出）

路由：
  GET  /health         健康检查（ark probe 读 j.version）
  POST /v1/responses    流式 agent 运行（Bearer 鉴权；body.multi 给定 → P2-2 multi-agent 并行+汇总）
  GET  /v1/tasks/{task_id}/operations  查询待核验工具账本
  POST /v1/tasks/{task_id}/operations/{operation_id}/reconcile  写入外部核验结果

HTTP 底层用 aiohttp（asyncio），消除旧版 "线程壳包 asyncio" 的矛盾：
agent 运行、SSE 写流、进度心跳全跑在同一个事件循环里，天然并发生长连接。
对外生命周期 API（start / serve_forever / shutdown）维持旧 ThreadingHTTPServer 语义，
由 `_AsyncServer` 适配器承接，调用方与测试不感知底层框架。
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import aiohttp
from aiohttp import web
from pydantic import ValidationError

from agentlab.core.loop import Ev, RunConfig
from agentlab.core.message import Message
from agentlab.memory.session_store import JsonlSessionStorage
from agentlab.runtime.serve_auth import check_bearer, serve_confirm
from agentlab.runtime.approvals import ApprovalManager
from agentlab.runtime.serve_contract import (
    CONFLICT_MESSAGE,
    CONTRACT,
    ErrorCode,
    make_json_response,
    map_exception_to_error,
    REPLAY_HEADER,
    request_id_middleware,
    ResponsesRequest,
    StandardResponse,
    _Sink,
    _sse,
)
from agentlab.runtime.serve_idem import _IdempotencyCache
from agentlab.runtime.serve_session import (
    history as _history,
    persist_delta as _persist_delta,
)


# ── 常驻服务请求/工具日志：定位"静默卡住"不再靠猜 ──
# 显式锚点：serve.py 位于 agentlab/agentlab/runtime/，日志归属项目根 logs/（避免目录层级魔法数）
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOG_PATH = _PROJECT_ROOT / "logs" / "serve.log"
_T0 = time.monotonic()
# 会话 id 白名单（与 memory.ranges._SID_RE 同口径）：防 DELETE 路径穿越
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _slog(*parts: Any) -> None:
    # 日志尽力而为：磁盘/权限失败不应影响请求主链路（C3：显式意图抑制，非裸 pass）
    with contextlib.suppress(OSError, UnicodeError):
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{time.monotonic() - _T0:8.2f}s] " + " ".join(str(p) for p in parts) + "\n")


def _load_cfg(config_path: str | None = None):
    from agentlab.runtime import config as _cfg

    return _cfg.load_config(path=config_path)


def _attach_depository(cfg, run_cfg, source_session: str = "", project_id: str = ""):
    """S6 记忆调度接线：config.memory 配置且 brain 可用 → 挂 MemoryStore 作为异步沉淀仓。

    brain 不可用（导入失败/未就绪）→ 返回原 run_cfg（depository=None，等效关闭），
    不抛错、不阻断请求（对齐记忆 store 的降级惯例）。
    OPT-224：source_session/project_id 注入记忆仓——沉淀带会话锚并可按项目隔离。
    """
    if not getattr(cfg.limits, "memorize_every", 0):
        return run_cfg
    try:
        from agentlab.memory.store import MemoryStore
        from agentlab.tools.connectors.brain_tools import load_brain_config

        brain_cfg = load_brain_config(cfg.model_dump()) or {}
        if project_id:
            brain_cfg["project_id"] = project_id  # brain memory_commit/query 读此键
        try:
            brain_cfg["memory_policy"] = cfg.memory.model_dump()
        except AttributeError:
            pass
        depo = MemoryStore(brain_cfg,
                           per_item_chars=cfg.limits.memory_item_chars,
                           decay_half_life_days=cfg.memory.decay_half_life_days,
                           source_session=source_session)
        if depo.available:
            run_cfg.depository = depo
            # #10①/OPT-123：LLM 语义提取器（importance 打分）；无 key/构造失败回退触发词路
            # 注意 _make_resilient 无 key 抛 SystemExit（BaseException），必须显式接住
            try:
                from agentlab.memory.extract import MemoryExtractor
                from agentlab.runtime.cli import _make_resilient

                run_cfg.extractor = MemoryExtractor(
                    _make_resilient(cfg), per_item_chars=cfg.limits.memory_item_chars)
            except (Exception, SystemExit) as e:  # noqa: BLE001
                _slog("MEMORY", "extractor unavailable: %s", e)
        _slog("MEMORY", "deposit_every=%s brain=%s",
              getattr(cfg.limits, "memorize_every", 0), depo.available)
    except Exception as e:  # 记忆仓初始化失败仅是能力降级，不强耦合请求主链路
        _slog("MEMORY", "depository unavailable: %s", e)
    return run_cfg


def _task_state_store(cfg):
    """Build the optional structured checkpoint store from context config."""
    context_cfg = getattr(cfg, "context", None)
    configured = str(getattr(context_cfg, "task_state_path", "") or "").strip()
    if not configured:
        return None
    try:
        from agentlab.runtime.task_state import TaskStateStore

        path = Path(configured)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        return TaskStateStore(path)
    except Exception as exc:  # noqa: BLE001 - checkpoint is an optional shadow aid
        _slog("TASK_STATE", "unavailable: %s", exc)
        return None


def _begin_task_state(cfg, task_id: str, *, session_id: str = "",
                      project_id: str = "", goal: str = ""):
    store = _task_state_store(cfg)
    if store is None or not task_id:
        return None, None
    try:
        state = store.ensure(
            task_id, session_id=session_id, project_id=project_id,
            core_intent={"goal": (goal or "")[:500]},
        )
        # A crashed worker may leave a planned/running external side effect.
        # Mark it unknown before the new run so the Runner cannot silently
        # replay a non-idempotent operation.
        store.recover_pending_tools(task_id, stale_after_seconds=300)
        # Query deterministic local artifacts before the Runner sees the task.
        # Unknown or unsupported operations remain a manual recovery barrier;
        # this never dispatches a tool or infers success from the checkpoint.
        try:
            from agentlab.runtime.operation_verifier import reconcile_unknown_operations
            reconcile_unknown_operations(
                store, task_id, vault_root=getattr(cfg, "vault_root", ""),
                project_root=_PROJECT_ROOT,
                remote_config=getattr(getattr(cfg, "remote_operation", None), "model_dump", lambda: None)(),
            )
        except Exception as exc:  # verifier is conservative and optional
            _slog("TASK_STATE", "external verification unavailable: %s", exc)
        state = store.get(task_id) or state
        if state.phase in {"DONE", "ERROR"}:
            state = store.transition(task_id, "PLANNING", reason="new request")
        elif state.phase == "WAITING_USER":
            # A user reply resumes the same task through a fresh planning
            # checkpoint before execution; finishing directly from
            # WAITING_USER is intentionally illegal in the state machine.
            state = store.transition(task_id, "PLANNING", reason="user reply")
        if state.phase == "IDLE":
            state = store.transition(task_id, "PLANNING", reason="request received")
        if state.phase == "PLANNING":
            state = store.transition(task_id, "EXECUTING", reason="agent started")
        return store, state
    except Exception as exc:  # noqa: BLE001 - state shadow must not block serving
        _slog("TASK_STATE", "begin failed: %s", exc)
        return None, None


def _finish_task_state(store, state, *, stop_reason: str = "", error: str = "",
                       context_plan=None):
    if store is None or state is None:
        return None
    try:
        # Runner tool-ledger checkpoints advance the optimistic-lock version
        # during a run.  Refresh before settling so a stale begin snapshot
        # cannot leave the task stuck in EXECUTING after a successful tool.
        latest = store.get(state.task_id) or state
        phase = "CANCELLED" if stop_reason in {"aborted", "cancelled"} else (
            "ERROR" if error else "DONE"
        )
        snapshot = {}
        if context_plan is not None:
            if callable(getattr(context_plan, "metrics", None)):
                snapshot = context_plan.metrics()
            else:
                snapshot = {
                    "mode": getattr(context_plan, "mode", "shadow"),
                    "used_tokens": int(getattr(context_plan, "used_tokens", 0) or 0),
                    "omitted": len(getattr(context_plan, "omitted", ()) or ()),
                    "warnings": list(getattr(context_plan, "warnings", ()) or ()),
                }
        changes = {"phase": phase, "context_snapshot": snapshot}
        if error:
            changes["last_error"] = {"message": str(error)[:200]}
        return store.patch(latest.task_id, changes, expected_version=latest.state_version,
                           reason="agent finished")
    except Exception as exc:  # noqa: BLE001 - state shadow must not block serving
        _slog("TASK_STATE", "finish failed: %s", exc)
        return None


def serve_config(cfg) -> dict[str, Any]:
    """serve 段配置（环境变量 AGENTLAB_SERVE_* > config.json serve 段）。

    cfg.serve 收敛为 pydantic `ServeConfig`（config.py），直接字段访问，
    不再兼容 dict/裸对象（C5：tau 式"配置收敛到 pydantic"）。**无硬编码默认
    token**——token 缺失由 start() fail-closed 拒绝启动。
    """
    from agentlab.runtime.config import ServeConfig

    sc: ServeConfig = getattr(cfg, "serve", None) or ServeConfig()
    env = os.environ.get
    return {
        "port": int(env("AGENTLAB_SERVE_PORT") or sc.port or 8643),
        "host": env("AGENTLAB_SERVE_HOST") or sc.host or "127.0.0.1",
        "token": env("AGENTLAB_SERVE_TOKEN") or sc.token or "",
        "heartbeat": int(env("AGENTLAB_SERVE_HEARTBEAT") or sc.heartbeat or 10),
        "approval_timeout_seconds": float(
            env("AGENTLAB_APPROVAL_TIMEOUT") or sc.approval_timeout_seconds or 120
        ),
    }


def _build_range_gateway(cfg, index=None):
    """L11/OPT-111：会话区段档案网关。禁用/构建失败 → None（折叠退回不可逆，不影响主流程）。

    archive 与会话 jsonl 同目录（trace_dir 同级 sessions/）；index 复用向量路实例
    （同库文件），未启用向量时 None——gateway 自动退 jsonl 关键词兜底路。
    """
    if not getattr(getattr(cfg, "rag", None), "session_ranges", True):
        return None
    try:
        from agentlab.memory.ranges import RangeArchive, RangeGateway
        sessions = Path(getattr(cfg, "trace_dir", "logs/trace")).parent / "sessions"
        return RangeGateway(RangeArchive(sessions), index=index)
    except Exception:
        return None


def _build_backend(cfg, rag_llm=None) -> tuple[Callable, int, Any]:
    """构造 (backend_factory[(builder, agent, reg)], n_brain, range_gateway)。
    backend_factory(messages, user_input, sink) 可被反复调用构建携带历史的 Runner。
    gateway 供 _run_agent 每请求 bind 当前会话（读侧）+ 注入折叠归档器（写侧）。
    """
    from agentlab.runtime.cli import _build_registry, _system_instruction, _build_runner

    from agentlab.tools.rag_tools import build_p2_store, build_vector_index
    # P2 is the user-facing Vault route when embedding is configured.  Keep
    # the legacy index only when P2 is unavailable so the old path remains a
    # rollback without doubling provider calls and SQLite scans in production.
    # Build one provider for the process.  Reconstructing it for every SSE
    # turn discarded the HTTP connection pool and reset circuit-breaker state.
    shared_llm = rag_llm
    if shared_llm is None:
        try:
            from agentlab.runtime.cli import _make_resilient

            shared_llm = _make_resilient(cfg)
        except SystemExit:
            # LLM is optional for lexical-only and test deployments.
            shared_llm = None
        except Exception:
            # LLM is optional for lexical-only and test deployments.
            shared_llm = None

    p2_store = build_p2_store(cfg.rag, cfg.vault_root)
    index = None if p2_store is not None else build_vector_index(cfg.rag, cfg.vault_root)
    gateway = _build_range_gateway(cfg, index=p2_store or index)

    reg, n_brain = _build_registry(cfg, with_brain=True, rag_llm=shared_llm,
                                   index=index, p2_store=p2_store,
                                   range_gateway=gateway)

    def build(sink) -> Any:
        # SSE 观察与 loop 内部一致：工具结果按 config.max_tool_result_chars 截断，
        # 避免此前硬编码 1000/4000 导致 6k 字转写只露开头 → agent 误判缺失反复重拉（卡顿根因）
        cap = RunConfig.from_config(cfg).max_tool_result_chars
        runner = _build_runner(cfg, registry=reg, provider=shared_llm)
        runner.on(Ev.MESSAGE_END, lambda p: sink.text(p.content or ""))
        # F5-016/OPT-182：透传真实入参（ToolStart.arguments 本就有值），
        # 前端才能显示「正在咨询 @agentX：问题…」而不是空对象。
        runner.on(Ev.TOOL_START,
                  lambda p: sink.tool_start(p.name, getattr(p, "arguments", "{}")))
        runner.on(
            Ev.TOOL_END,
            lambda p: sink.tool_end(p.name, (p.result or "")[:cap]),
        )
        return runner

    async def close() -> None:
        closer = getattr(shared_llm, "aclose", None)
        if closer is not None:
            await closer()

    # Keep the existing callable factory contract while exposing an explicit
    # lifecycle hook to Serve.aclose; no second global registry is introduced.
    build.close = close

    return build, n_brain, gateway


# ── 职责已拆至 serve_auth / serve_session ──
# 旧名 `_serve_confirm` 兼容：test_serve 早期按此导入；预置别名片以不破坏测试导入
_serve_confirm = serve_confirm


def _new_tracer(cfg):
    """#6/OPT-126：serve 侧 run 落 trace（此前仅 CLI 接 Tracer）；失败返 None 不影响主链路。"""
    try:
        from agentlab.runtime.trace import Tracer

        tracer = Tracer(cfg.trace_dir)
        tracer.new_session()
        return tracer
    except Exception:  # noqa: BLE001
        return None


# OPT-219 视觉分析按需启用：bili_visual/bili_screenshot 含截帧+视觉模型逐格分析
# （真机实测约 4 分钟 + 视觉模型费用），默认"生成笔记"走字幕/转写轻量路径即可。
# 只有用户消息明确提出视觉意图时，这两个工具才进模型工具面。
VISUAL_ONLY_TOOLS = frozenset({"bili_visual", "bili_screenshot"})
VISUAL_INTENT_KEYWORDS = ("视觉", "画面", "截图", "看图", "图", "帧", "镜头",
                          "多模态", "screenshot", "visual")


def _visual_tool_filter(user_input: str):
    """未提出视觉意图 → 返回隐藏视觉类工具的 filter；明确提出 → None（工具面原样）。"""
    text = (user_input or "").lower()
    if any(kw.lower() in text for kw in VISUAL_INTENT_KEYWORDS):
        return None
    return lambda t: getattr(t, "name", "") not in VISUAL_ONLY_TOOLS


async def _run_agent(cfg, build_backend, hist: list[Message], user_input: str, sink: _Sink,
                     store=None, session_id: str | None = None,
                     signal: asyncio.Event | None = None,
                     project_id: str | None = None,
                     range_gateway=None, approvals: ApprovalManager | None = None,
                     run_id: str = "", tool_filter=None,
                     evaluation_read_only: bool = False) -> dict:
    """跑一轮 agent。

    `tool_filter(tool) -> bool` 可选：默认 None = 不改变行为（生产链路原样）。
    质量评测用它把工具面收敛成只读（`permission == "read"`），从结构上保证
    评测过程不可能写用户 Vault——评测要能反复跑，不能有副作用。

    ``evaluation_read_only`` is stricter than a tool filter: it also disables
    automatic memory recall/deposit and TaskState writes.  Offline answer
    samples must measure retrieval/generation, not create memory candidates or
    checkpoints merely because a configured production runtime has them on.
    """
    from agentlab.core.agent import Agent
    from agentlab.core.context import Context
    from agentlab.core.loop import RunConfig, RunHooks
    from agentlab.runtime.cli import _system_instruction
    from agentlab.runtime.project import apply_project_context

    runner = build_backend(sink)
    reg = runner.registry
    # #10①/OPT-123：长期记忆自动召回——topic=本轮输入，brain LIKE+时间衰减重排（零 LLM 成本）；
    # 空结果/仓库不可用静默降级为占位文案（有工具≠会用，注入不能靠模型想起调工具）
    from agentlab.memory.recall import memory_block_for

    memory_block = ""
    if not evaluation_read_only:
        memory_block = memory_block_for(
            cfg, user_input, topk=int(getattr(cfg.limits, "memory_inject_topk", 5) or 0),
            project_id=project_id or "")
    base_instructions = _system_instruction(cfg, reg.all(), memory=memory_block)
    instructions = apply_project_context(base_instructions, cfg.vault_root, project_id)
    project_injected = instructions != base_instructions
    _slog(
        "PROJECT",
        f"session={session_id or '-'} project={project_id or '-'} injected={project_injected}",
    )
    tools = reg.all()
    if tool_filter is not None:
        # 必须换掉 runner 的注册表：发给模型的 schema 来自 runner.registry，
        # 只过滤 Agent.tools 的话模型照样看得见并调用被排除的工具（OPT-196 实测踩过）。
        from agentlab.tools.registry import FilteredRegistryView

        runner.registry = FilteredRegistryView(reg, tool_filter)
        reg = runner.registry
        tools = reg.all()
    agent = Agent(
        name="agentlab-serve",
        instructions=instructions,
        tools=tools,
        max_steps=cfg.limits.max_steps,
    )
    ctx = Context(budget=cfg.limits.context_budget)
    ctx.history = hist
    async def confirm(t, prompt):
        if approvals is not None:
            return await approvals.confirm(cfg, t, prompt, sink=sink, run_id=run_id,
                                           session_id=session_id, signal=signal)
        return _serve_confirm(cfg, t, prompt)
    hooks = RunHooks(confirm=confirm)
    run_cfg = RunConfig.from_config(cfg)
    # P0-08: build a deterministic bounded plan for complex requests, but keep
    # the existing ReAct path authoritative. The plan is persisted in the run
    # summary/checkpoint as auditable shadow data and is never treated as proof
    # of completion until PlanExecutor supplies evidence.
    try:
        from agentlab.core.planning import PlanBuilder

        plan = PlanBuilder().build(
            user_input,
            {"project_id": project_id or "", "session_id": session_id or ""},
            [getattr(tool, "name", "") for tool in tools],
            deadline_at=(time.monotonic() + run_cfg.timeout) if run_cfg.timeout else None,
        )
        run_cfg.plan = plan
    except Exception as exc:  # noqa: BLE001 - shadow planning never blocks serving
        run_cfg.plan = None
        _slog("PLAN", "shadow unavailable: %s", exc)
    # P1 ContextAssembler runs in shadow by default.  It produces an auditable
    # four-zone plan on each provider turn while leaving the compatibility
    # Context renderer and model-visible messages unchanged.
    try:
        from agentlab.core.context_assembler import ContextAssembler

        context_cfg = getattr(cfg, "context", None)
        assembler_mode = str(getattr(context_cfg, "assembler_mode", "shadow"))
        if assembler_mode in {"shadow", "on"}:
            run_cfg.context_assembler = ContextAssembler(
                budget_tokens=run_cfg.context_budget,
                reserve_output_tokens=int(getattr(
                    context_cfg, "reserve_output_tokens", 16384
                )),
                zone_budgets={
                    "dialogue_memory": int(getattr(
                        context_cfg, "history_budget_tokens", 8000
                    )) + int(getattr(context_cfg, "memory_budget_tokens", 1200)),
                    "external": int(getattr(context_cfg, "rag_budget_tokens", 4000)),
                },
                mode=assembler_mode,
            )
    except Exception as exc:  # noqa: BLE001 - planner is an optional shadow aid
        _slog("CONTEXT", "assembler unavailable: %s", exc)
    run_cfg.signal = signal  # SSE 断连 → _emit 置位此信号 → loop 中止，停止烧 token
    if not evaluation_read_only:
        run_cfg = _attach_depository(
            cfg, run_cfg, source_session=session_id or "", project_id=project_id or "",
        )
    # S0：所有检索工具读取同一请求级 scope；模型不能通过工具参数伪造项目/会话范围。
    from agentlab.contracts import RetrievalScope, bind_retrieval_scope, reset_retrieval_scope
    scope_token = bind_retrieval_scope(RetrievalScope(
        project_id=project_id or "", session_id=session_id or "",
    ))
    # L11/OPT-111：写侧注入折叠区段归档器；读侧 ContextVar 绑当前会话（rag session 路），
    # run 结束（含异常）reset，防并发请求串话。
    range_token = None
    if range_gateway is not None and session_id:
        run_cfg.range_recorder = range_gateway.recorder(
            session_id,
            chunk_chars=getattr(cfg.rag, "range_chunk_chars", 2000),
            max_chunks=getattr(cfg.rag, "range_max_chunks", 200))
        range_token = range_gateway.bind(session_id)
    if evaluation_read_only:
        task_state, task_state_snapshot = None, None
    else:
        task_state, task_state_snapshot = _begin_task_state(
            cfg, run_id or "", session_id=session_id or "", project_id=project_id or "",
            goal=user_input,
        )
    memory_runtime_token = None
    try:
        from agentlab.tools.connectors.brain_tools import _add_brain_path
        _add_brain_path()
        import tools_memory
        memory_runtime_token = tools_memory.bind_runtime_dependencies(
            range_gateway=range_gateway,
            task_state_store=task_state,
            task_state_id=str(run_id or ""),
        )
    except Exception as exc:  # noqa: BLE001 - optional invalidation wiring
        _slog("MEMORY", "runtime invalidation unavailable: %s", exc)
    # Feed the same validated checkpoint into the shadow planner.  The planner
    # only observes it by default; the compatibility prompt remains unchanged.
    run_cfg.task_state = task_state_snapshot
    run_cfg.task_state_store = task_state
    run_cfg.task_state_id = str(run_id or "")
    run_cfg.vault_root = str(getattr(cfg, "vault_root", "") or "")
    tracer = _new_tracer(cfg)  # #6/OPT-126：serve 侧 run 落 trace（失败 None 降级）
    if tracer is not None and callable(getattr(runner, "on", None)):
        # P1-02：工具 start 记参数指纹（脱敏由 tracer 负责），end 记截断结果
        runner.on(Ev.TOOL_START, lambda p: tracer.record_tool(
            p.name, "start",
            arguments_hash=hashlib.sha1((getattr(p, "arguments", "") or "").encode()
                                        ).hexdigest()[:12]))
        runner.on(Ev.TOOL_END, lambda p: tracer.record_tool(
            p.name, "end", result=(p.result or "")[:1000]))
    run_summary: dict = {
        "stop_reason": "error", "tokens": 0, "run_id": run_id,
        "session_id": session_id or "", "project_id": project_id or "",
        "model": getattr(getattr(cfg, "llm", None), "model", ""),
        "plan": run_cfg.plan.to_dict() if getattr(run_cfg, "plan", None) is not None else None,
    }
    t0 = time.monotonic()
    try:
        result = await runner.run(agent, user_input, ctx=ctx, hooks=hooks, cfg=run_cfg)
        if tracer is not None:
            result.trace_id = tracer.trace_id
        trace_data = result.run_trace if isinstance(result.run_trace, dict) else {}
        counters = trace_data.get("counters", {}) if isinstance(trace_data, dict) else {}
        run_summary.update(stop_reason=result.stop_reason,
                           tokens=result.usage.total(),
                           steps=int(counters.get("rounds", 0) or 0),
                           rounds=int(counters.get("rounds", 0) or 0),
                           llm_calls=int(counters.get("llm_calls", 0) or 0),
                           tool_calls=int(counters.get("tool_calls", 0) or 0),
                           retries=int(counters.get("retries", 0) or 0),
                           plan_steps=int(counters.get("plan_steps", 0) or 0))
        if isinstance(trace_data, dict) and isinstance(trace_data.get("context_assembler"), dict):
            run_summary["context_assembler"] = dict(trace_data["context_assembler"])
    except Exception as exc:
        run_summary["error"] = f"{type(exc).__name__}: {exc}"
        run_summary["error_code"] = getattr(exc, "code", "") or type(exc).__name__
        _finish_task_state(
            task_state, task_state_snapshot,
            stop_reason="error", error=run_summary["error"],
            context_plan=getattr(run_cfg, "context_plan", None),
        )
        raise
    finally:
        if task_state is not None and task_state_snapshot is not None and "error" not in run_summary:
            _finish_task_state(
                task_state, task_state_snapshot,
                stop_reason=str(run_summary.get("stop_reason", "")),
                context_plan=getattr(run_cfg, "context_plan", None),
            )
        reset_retrieval_scope(scope_token)
        if range_token is not None:
            range_gateway.reset(range_token)
        if memory_runtime_token is not None:
            try:
                import tools_memory
                tools_memory.reset_runtime_dependencies(memory_runtime_token)
            except Exception as exc:  # noqa: BLE001
                _slog("MEMORY", "runtime invalidation reset failed: %s", exc)
        if tracer is not None:
            with contextlib.suppress(Exception):
                run_summary["duration_ms"] = int((time.monotonic() - t0) * 1000)
                tracer.record_run(input=user_input[:300], **run_summary)
    # 会话持久化：把本轮新增消息（跳过 system 与已重放的历史）追加进会话存储
    if store is not None and session_id:
        _persist_delta(store, session_id, hist, result.messages)

    # W2/OPT-110：折叠后的最终历史快照（去 system）——前端替换本地历史，
    # 让 nudge/compress/硬截断产生的压缩**跨轮持久**（否则每轮客户端回传全量，折叠即丢失）
    # 三期修正：assistant 的 tool_calls 参数也序列化进快照（content=None 的工具调用轮
    # 曾被记成空串 → 用量条偏低 + 前端历史丢参数上下文）
    def _hist_content(m: Message) -> str:
        parts = []
        if m.content:
            parts.append(m.content)
        if m.tool_calls:
            import json as _json
            parts.append(_json.dumps(
                [{"tool": tc.function.name, "args": tc.function.arguments}
                 for tc in m.tool_calls], ensure_ascii=False))
        return "\n".join(parts)

    return {
        "final_output": result.final_output,
        "stop_reason": result.stop_reason,
        "tokens": result.usage.total(),
        "trace_id": result.trace_id,
        "answer_gate": result.answer_gate,
        "plan": run_cfg.plan.to_dict() if getattr(run_cfg, "plan", None) is not None else None,
        "project_id": project_id or "",
        "project_context_injected": project_injected,
        "history": [
            {"role": m.role, "content": _hist_content(m)}
            for m in result.messages if m.role != "system"
        ],
    }


# ── asyncio SSE 写层（替代旧 _Handler._emit 的线程式 wfile 写 + RLock）──

class _SSEWriter:
    """asyncio SSE 写封装：串行写防并发交错 + 客户端断连置位 cancel（供 loop 中止）。

    生产端（agent 回调 → queue→ drain task）与进度心跳经同一把锁写 StreamResponse，
    语义等同旧版 `_emit` 的 `threading.RLock`，但跑在同一事件循环、无线程切换。
    """

    def __init__(self, resp: web.StreamResponse, *, heartbeat: int, contract: str,
                 capture: list[str] | None = None):
        self._resp = resp
        self._ser = asyncio.Lock()
        self._hb_iv = heartbeat or 0
        self._contract = contract
        # capture：若给定，则把每条发出的 SSE 字节原样追加进去，供幂等缓存重放（§4.7）
        self._capture = capture if capture is not None else []
        self.cancel: asyncio.Event = asyncio.Event()

    async def write(self, chunk: str) -> None:
        self._capture.append(chunk)
        async with self._ser:
            try:
                # aiohttp>=3.11 StreamResponse.write 是 awaitable（drain 已弃用）：
                # 必须 await，否则字节只排队不落盘，客户端拿到空 SSE
                await self._resp.write(chunk.encode("utf-8"))
            except (ConnectionResetError, ConnectionAbortedError, aiohttp.ClientConnectionError):
                # 客户端断连：置位取消让 loop 中止；异常上抛由 on_post finally 回收
                self.cancel.set()
                _slog("CANCEL", "client disconnected -> aborting loop")
                raise


async def _run_heartbeat(writer: _SSEWriter) -> None:
    """进度心跳：长工具阻塞期间沿 SSE 流周期推 response.heartbeat。

    专用事件类型而非 output_text.delta：不累积进最终正文（未识别事件前端跳过）。
    """
    if not writer._hb_iv:
        return
    t0 = time.monotonic()
    while not writer.cancel.is_set():
        try:
            await asyncio.wait_for(writer.cancel.wait(), timeout=writer._hb_iv)
            break  # cancel 已置位，干净退出
        except asyncio.TimeoutError:
            await writer.write(_sse({
                "type": "response.heartbeat",
                "elapsed": int(time.monotonic() - t0),
                "contract": writer._contract,
            }))


async def _watch_disconnect(request: web.Request, writer: _SSEWriter,
                            interval: float = 0.1) -> None:
    """在没有 SSE 输出时探测客户端断开，及时置位 Agent 取消信号。

    仅依赖 ``transport.is_closing``，不读取请求体也不主动关闭连接；同步工具线程
    仍不可强杀，但 Runner 会在本轮工具返回前停止等待并禁止下一轮编排。
    """
    while not writer.cancel.is_set():
        transport = request.transport
        if transport is None or transport.is_closing():
            writer.cancel.set()
            _slog("CANCEL", "client transport closed -> aborting loop")
            return
        await asyncio.sleep(interval)


async def _drain(queue: "asyncio.Queue[str]", writer: _SSEWriter) -> None:
    """把 agent 同步回调推进的字节队列，逐块写到 SSE 流；断连即止。"""
    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        try:
            await writer.write(chunk)
        except (ConnectionResetError, ConnectionAbortedError, aiohttp.ClientConnectionError):
            break  # writer.cancel 已置位，交给 on_post finally 回收


# ── aiohttp 路由 ──

@web.middleware
async def _json404(request, handler):
    try:
        return await handler(request)
    except web.HTTPNotFound:
        return make_json_response(
            StandardResponse.error(ErrorCode.NOT_FOUND, "not found", request.get("request_id")),
            status=404,
            request_id=request.get("request_id")
        )


@web.middleware
async def _error_handler(request, handler):
    """Map unexpected handler exceptions to the public response contract."""
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except Exception as exc:
        code, message = map_exception_to_error(exc, request.get("request_id"))
        import logging

        logging.exception("[SERVE] Unhandled exception: %s", exc)
        return make_json_response(
            StandardResponse.error(code, message, request.get("request_id")),
            status=int(code),
            request_id=request.get("request_id"),
        )


class _ServeApp:
    """把 Serve 实例上的 cfg/build/store 绑定进 aiohttp 各路由。"""

    def __init__(self, serve: "Serve"):
        self._serve = serve

    def app(self) -> web.Application:
        app = web.Application(
            middlewares=[request_id_middleware, _json404, _error_handler]
        )
        app.router.add_get("/health", self.on_get)
        app.router.add_options("/v1/responses", self.on_options)
        app.router.add_options("/v1/approvals/{approval_id}", self.on_options)
        app.router.add_post("/v1/responses", self.on_post)
        app.router.add_post("/v1/approvals/{approval_id}", self.on_approval)
        app.router.add_post("/v1/context-status", self.on_context_status)
        app.router.add_get("/v1/runs", self.on_runs)
        app.router.add_get("/v1/runs/{trace_id}", self.on_run_detail)
        app.router.add_get("/v1/agents", self.on_agents)
        app.router.add_get("/v1/tasks/{task_id}/operations", self.on_task_operations)
        app.router.add_post(
            "/v1/tasks/{task_id}/operations/{operation_id}/reconcile",
            self.on_reconcile_operation,
        )
        app.router.add_delete("/v1/sessions/{session_id}", self.on_delete_session)
        # CORS 预检兜底（F5-021 回归修复）：带 Authorization 的跨域请求会先发 OPTIONS，
        # 此前只有 /v1/responses 与 /v1/approvals 注册了 OPTIONS，`/v1/runs`、`/v1/agents`
        # 等端点预检拿到 405 → 浏览器直接判网络失败，前端只看到 "Failed to fetch"。
        # 通配 OPTIONS 让新增端点自动获得预检能力，避免同类回归再次发生。
        app.router.add_route("OPTIONS", "/{tail:.*}", self.on_options)
        return app

    async def on_get(self, request):
        from agentlab import __version__
        return web.json_response({"ok": True, "version": __version__})

    async def on_options(self, request):
        # CORS 预检（浏览器将 `POST + Authorization + application/json` 视为非简单请求，
        # 先发 OPTIONS）；旧版 501 曾致 Obsidian fetch 报 "Failed to fetch"（回归防窜）
        return web.Response(
            status=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Authorization, Content-Type",
                "Access-Control-Max-Age": "86400",
            },
        )

    async def on_context_status(self, request):
        """W2/OPT-110：上下文用量计算端点——前端传消息数组，服务端用与 loop 完全
        相同的 estimate_tokens 口径返回 tokens/预算/阈值，供工作台用量条可视化。
        纯本地计算（零 LLM/零索引调用），鉴权同 /v1/responses。"""
        serve = self._serve
        auth_resp = self._authorize(request, serve.serve_cfg)
        if auth_resp is not None:
            return auth_resp
        try:
            body = await request.json()
        except Exception:
            return make_json_response(
                StandardResponse.error(ErrorCode.BAD_REQUEST, "invalid json", request.get("request_id")),
                status=400,
                request_id=request.get("request_id")
            )
        msgs = body.get("messages") if isinstance(body, dict) else None
        from agentlab.core.context import estimate_tokens
        from agentlab.core.message import Message
        total = 0
        count = 0
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict) and str(m.get("role") or "") in ("user", "assistant", "system", "tool"):
                    total += estimate_tokens(Message(role=str(m.get("role")), content=str(m.get("content") or "")))
                    count += 1
        from agentlab.core.loop import RunConfig, Runner
        from agentlab.runtime.project import project_context

        run_cfg = RunConfig.from_config(serve.cfg)
        configured_budget = max(1, int(run_cfg.context_budget))
        effective_budget = max(1, Runner._effective_budget(run_cfg))
        project_id = str(body.get("project_id") or "").strip() if isinstance(body, dict) else ""
        project_block = project_context(serve.cfg.vault_root, project_id)
        project_tokens = estimate_tokens(Message(role="system", content=project_block)) if project_block else 0
        return web.json_response({
            "ok": True,
            "tokens": total,
            "budget": effective_budget,
            "configured_budget": configured_budget,
            "effective_budget": effective_budget,
            "context_window": int(run_cfg.context_window),
            "usage_pct": round(total * 100.0 / effective_budget, 1),
            "nudge_pct": round(serve.cfg.limits.context_nudge * 100),
            "hard_trim_pct": round(serve.cfg.limits.context_hard_trim * 100),
            "messages": count,
            "scope": "client_history_estimate",
            "project_id": project_id,
            "project_context_loaded": bool(project_block),
            "project_context_tokens": project_tokens,
        })

    async def on_runs(self, request):
        """#6/OPT-126：最近 run 概览（只读，鉴权同 /v1/responses）。

        数据源 trace_dir/*.jsonl 的 run 摘要行（_run_agent 每请求落一条）；
        只回概览字段——输入在写入侧截断 300 字符且经 trace 脱敏，无工具结果正文。
        """
        serve = self._serve
        auth_resp = self._authorize(request, serve.serve_cfg)
        if auth_resp is not None:
            return auth_resp
        try:
            limit = min(100, max(1, int(request.query.get("limit", "20"))))
        except ValueError:
            limit = 20
        from agentlab.runtime.trace import load_run_summaries

        return web.json_response(
            {"ok": True, "runs": load_run_summaries(serve.cfg.trace_dir, limit)})

    async def on_agents(self, request):
        """P2-2/F5-019：可 @ 的答者名单（只读，鉴权同 /v1/responses）。

        Ark 的 @ 补全与"该不该走 multi"判断都以本名单为准：命中名字才进 multi，
        否则按普通文本发送。内置 agentlab 恒在；external 取 AgentsConfig.external
        中 command 非空者（与 build_multi_registry 的启用口径一致）。
        同时回传当前并发上限与单支结果限额，供前端提示"超出会被跳过"。
        """
        serve = self._serve
        auth_resp = self._authorize(request, serve.serve_cfg)
        if auth_resp is not None:
            return auth_resp
        from agentlab.runtime.multi import SELF_AGENT_NAME

        agents = [{"name": SELF_AGENT_NAME, "kind": "internal", "enabled": True,
                   "description": "内置 agentlab（全能力：工具 / 记忆 / HITL）"}]
        for spec in getattr(serve.cfg.agents, "external", None) or []:
            if not getattr(spec, "command", ""):
                continue
            agents.append({"name": spec.name, "kind": "external", "enabled": True,
                           "description": f"外部 ACP agent（{spec.command}）"})
        return web.json_response({
            "ok": True,
            "agents": agents,
            "max_parallel_consults": int(
                getattr(serve.cfg.agents, "max_parallel_consults", 3) or 3),
            "consult_result_max_chars": int(
                getattr(serve.cfg.agents, "consult_result_max_chars", 6000) or 6000),
        })

    async def on_run_detail(self, request):
        """只读 run 诊断：工具/LLM 事件元数据，不返回正文或敏感结果。"""
        serve = self._serve
        auth_resp = self._authorize(request, serve.serve_cfg)
        if auth_resp is not None:
            return auth_resp
        from agentlab.runtime.trace import load_run_detail

        detail = load_run_detail(serve.cfg.trace_dir, request.match_info.get("trace_id", ""))
        if detail is None:
            return make_json_response(
                StandardResponse.error(ErrorCode.NOT_FOUND, "run not found", request.get("request_id")),
                status=404, request_id=request.get("request_id"))
        return web.json_response({"ok": True, "detail": detail})

    @staticmethod
    def _task_path_value(request, name: str, max_length: int) -> str | None:
        value = str(request.match_info.get(name, "") or "").strip()
        if not value or len(value) > max_length or "/" in value or "\\" in value:
            return None
        return value

    @staticmethod
    def _task_state_error_response(
        request, message: str, *, status: int = 400
    ) -> web.Response:
        return make_json_response(
            StandardResponse.error(ErrorCode(status), message, request.get("request_id")),
            status=status,
            request_id=request.get("request_id"),
        )

    async def on_task_operations(self, request) -> web.Response:
        """列出恢复所需的 pending/unknown 操作；不返回参数正文，也不执行工具。"""
        serve = self._serve
        auth_resp = self._authorize(request, serve.serve_cfg)
        if auth_resp is not None:
            return auth_resp
        task_id = self._task_path_value(request, "task_id", 128)
        if task_id is None:
            return self._task_state_error_response(request, "invalid task id")
        store = _task_state_store(serve.cfg)
        if store is None:
            return self._task_state_error_response(
                request, "task state store unavailable", status=503)
        state = store.get(task_id)
        if state is None:
            return self._task_state_error_response(request, "task not found", status=404)
        return web.json_response({
            "ok": True,
            "task_id": state.task_id,
            "state_version": state.state_version,
            "phase": state.phase,
            "operations": store.pending_operations(task_id),
        })

    async def on_reconcile_operation(self, request) -> web.Response:
        """用外部证据结算一个 ``unknown`` 操作；此接口绝不重放工具。"""
        serve = self._serve
        auth_resp = self._authorize(request, serve.serve_cfg)
        if auth_resp is not None:
            return auth_resp
        task_id = self._task_path_value(request, "task_id", 128)
        operation_id = self._task_path_value(request, "operation_id", 256)
        if task_id is None or operation_id is None:
            return self._task_state_error_response(request, "invalid task or operation id")
        store = _task_state_store(serve.cfg)
        if store is None:
            return self._task_state_error_response(
                request, "task state store unavailable", status=503)
        state = store.get(task_id)
        if state is None:
            return self._task_state_error_response(request, "task not found", status=404)
        operation = next(
            (row for row in state.pending_tools if row.get("operation_id") == operation_id),
            None,
        )
        if operation is None:
            return self._task_state_error_response(request, "operation not found", status=404)
        if operation.get("status") != "unknown":
            return self._task_state_error_response(
                request, "only unknown operations can be reconciled", status=409)
        try:
            body = await request.json()
        except Exception:
            return self._task_state_error_response(request, "invalid json")
        if not isinstance(body, dict):
            return self._task_state_error_response(request, "external result must be an object")
        external_result = dict(body)
        expected_version = external_result.pop("expected_version", None)
        if expected_version is not None:
            if isinstance(expected_version, bool) or not isinstance(expected_version, int) \
                    or expected_version < 0:
                return self._task_state_error_response(request, "expected_version must be a non-negative integer")
        try:
            saved = store.reconcile_operation(
                task_id,
                operation_id,
                external_result=external_result,
                expected_version=expected_version,
            )
        except Exception as exc:
            from agentlab.runtime.task_state import TaskStateConflict, TaskStateError

            if isinstance(exc, TaskStateConflict):
                return self._task_state_error_response(request, str(exc), status=409)
            if isinstance(exc, TaskStateError):
                return self._task_state_error_response(request, str(exc), status=400)
            raise
        reconciled = next(
            row for row in saved.pending_tools if row.get("operation_id") == operation_id
        )
        _slog("TASK_RECONCILE", task_id, operation_id, reconciled.get("status"))
        return web.json_response({
            "ok": True,
            "task_id": saved.task_id,
            "state_version": saved.state_version,
            "operation": reconciled,
        })

    async def on_approval(self, request):
        """HITL resolve：allow/deny/cancel 均是幂等终态写入。"""
        serve = self._serve
        auth_resp = self._authorize(request, serve.serve_cfg)
        if auth_resp is not None:
            return auth_resp
        aid = request.match_info.get("approval_id", "")
        try:
            body = await request.json()
        except Exception:
            body = {}
        decision = str(body.get("decision") or "").strip().lower()
        if decision == "cancel":
            decision = "deny"
        if decision not in {"allow", "deny"}:
            return make_json_response(
                StandardResponse.error(ErrorCode.BAD_REQUEST, "decision must be allow or deny",
                                       request.get("request_id")),
                status=400, request_id=request.get("request_id"))
        result = serve.approvals.resolve(aid, decision)
        if result is None:
            return make_json_response(
                StandardResponse.error(ErrorCode.NOT_FOUND, "approval not found",
                                       request.get("request_id")),
                status=404, request_id=request.get("request_id"))
        _slog("APPROVAL_RESOLVE", aid, decision, result.get("status"))
        return web.json_response({"ok": True, "approval": result})

    async def on_delete_session(self, request):
        """#8/OPT-127：ark 删除会话的落点——连同 ranges 档案与向量索引行一并清理。

        sid 白名单校验防路径穿越；任一存储层缺失（测试替身/降级路径）逐项跳过，
        deleted 字段如实回报各路清理结果。
        """
        serve = self._serve
        auth_resp = self._authorize(request, serve.serve_cfg)
        if auth_resp is not None:
            return auth_resp
        sid = request.match_info.get("session_id", "")
        if not _SESSION_ID_RE.fullmatch(sid):
            return make_json_response(
                StandardResponse.error(ErrorCode.BAD_REQUEST, "invalid session id", request.get("request_id")),
                status=400,
                request_id=request.get("request_id")
            )
        deleted = {"session": False, "ranges": False, "index": False}
        delete = getattr(serve.session_store, "delete", None)
        if callable(delete):
            deleted["session"] = bool(delete(sid))
        gateway = getattr(serve, "range_gateway", None)
        if gateway is not None:
            try:
                r = gateway.remove_session(sid)
                deleted["ranges"] = bool(r.get("ranges"))
                deleted["index"] = bool(r.get("index"))
            except Exception:  # noqa: BLE001
                _slog("SESSION_DEL", "gateway cleanup failed", sid)
        _slog("SESSION_DEL", sid, json.dumps(deleted))
        return web.json_response({"ok": True, "deleted": deleted})

    async def on_post(self, request):
        serve = self._serve

        auth_resp = self._authorize(request, serve.serve_cfg)
        if auth_resp is not None:
            return auth_resp

        req, payload_resp = await self._read_payload(request)
        if payload_resp is not None:
            return payload_resp

        join = self._join_session(serve, req)
        if join is None:
            return make_json_response(
                StandardResponse.error(ErrorCode.BAD_REQUEST, "empty input", request.get("request_id")),
                status=400,
                request_id=request.get("request_id")
            )
        hist, latest, session_id = join
        _slog("REQUEST_IN", f"latest='{latest[:40]}' n_hist={len(hist)}")

        # ── 幂等（§4.7）：request_id 重复提交 → 重放首次结果；并发在跑 → 409 拒双跑 ──
        # 顺序安放：begin 前均为同步（无 await），避免两个同 id 请求在 prepare 处交错抢锁
        rq_id = req.request_id
        idem = serve.idem
        owned, replay = await self._idempotency_gate(request, idem, rq_id)
        if replay is not None:
            return replay

        resp = web.StreamResponse(status=200)
        resp.headers.update({
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*",
            "X-Agentlab-Contract": CONTRACT,
            # 每条请求独立连接（即发即关），避免 keep-alive 使客户端挂起等流结束
            "Connection": "close",
        })
        await resp.prepare(request)

        capture: list[str] = []  # 幂等：writer 会把每条 SSE 字节原样记入此处
        writer = _SSEWriter(resp, heartbeat=serve.serve_cfg.get("heartbeat") or 0,
                            contract=CONTRACT, capture=capture)
        # agent 回调是同步的 → 走队列，由 drain task 异步写流；写量与心跳共用 writer 写锁
        queue: asyncio.Queue[str] = asyncio.Queue()
        sink = _Sink(queue.put_nowait)
        write_task = asyncio.create_task(_drain(queue, writer))
        hb_task = asyncio.create_task(_run_heartbeat(writer))
        disconnect_task = asyncio.create_task(_watch_disconnect(request, writer))

        _req_t0 = time.monotonic()
        committed = False
        try:
            if req.multi:
                # P2-2/OPT-121：multi-agent 并行 + 主 agent 汇总（失败支降级不计入）
                from agentlab.runtime.multi import _run_multi

                summary = await _run_multi(
                    serve.cfg, serve.multi_registry, serve.synth_provider,
                    serve.build_backend, hist, latest, sink, multi=req.multi,
                    store=serve.session_store, session_id=session_id,
                    signal=writer.cancel, project_id=req.project_id,
                    approvals=serve.approvals,
                    range_gateway=getattr(serve, "range_gateway", None),
                    run_id=req.request_id or request.get("request_id", ""))
            else:
                summary = await _run_agent(serve.cfg, serve.build_backend, hist, latest, sink,
                                           store=serve.session_store, session_id=session_id,
                                           signal=writer.cancel,
                                           project_id=req.project_id,
                                           range_gateway=getattr(serve, "range_gateway", None),
                                           approvals=serve.approvals,
                                           tool_filter=_visual_tool_filter(latest),
                                           run_id=req.request_id or request.get("request_id", ""))
            await queue.put(None)
            await write_task
            await writer.write(_sse({"type": "response.completed",
                                     "response": {"summary": summary}}))
            await writer.write("data: [DONE]\n\n")
            if owned:  # 成功才落幂等结果；失败不缓存，允许重试重跑
                idem.commit(rq_id, capture)
                committed = True
            _slog("REQUEST_END", f"ok elapsed={time.monotonic() - _req_t0:.1f}s")
        except Exception as e:  # noqa: BLE001 —— 兜底错误也以流形式返回，避免挂死连接
            _slog("REQUEST_ERR", type(e).__name__, str(e),
                  f"elapsed={time.monotonic() - _req_t0:.1f}s")
            with contextlib.suppress(ConnectionResetError, ConnectionAbortedError,
                                     aiohttp.ClientConnectionError):
                await writer.write(_sse({"type": "response.failed", "error": str(e)}))
                await writer.write("data: [DONE]\n\n")
        finally:
            writer.cancel.set()
            for t in (disconnect_task, hb_task, write_task):
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError, ConnectionResetError,
                                         ConnectionAbortedError, aiohttp.ClientConnectionError):
                    await t
            with contextlib.suppress(Exception):
                await resp.write_eof()
            if owned and not committed:  # 失败/中断：释放 pending，让重试可重新跑
                idem.abandon(rq_id)
        return resp

    # ── on_post 拆出的前置阶段：鉴权 / 载荷解析 / 会话合并 / 幂等闸门（C6）──

    @staticmethod
    def _authorize(request, sc) -> web.Response | None:
        """Bearer 鉴权（fail-closed）；不通过返回 401 响应，通过返回 None。"""
        auth = request.headers.get("Authorization", "")
        if not check_bearer(auth, sc["token"]):
            return make_json_response(
                StandardResponse.error(ErrorCode.UNAUTHORIZED, "unauthorized", request.get("request_id")),
                status=401,
                request_id=request.get("request_id")
            )
        return None

    async def _read_payload(self, request) -> tuple[ResponsesRequest | None, web.Response | None]:
        """把请求体解析为 ResponsesRequest；非法载荷返回 (None, 400)。"""
        raw = await request.text()
        try:
            return ResponsesRequest.model_validate(json.loads(raw) if raw else {}), None
        except (ValidationError, ValueError):
            return None, web.json_response(
                {"ok": False, "error": "bad payload", "error_type": "responses_request"},
                status=400,
            )

    def _join_session(self, serve, req) -> tuple[list[Message], str, str | None] | None:
        """合并会话：携带 previous_response_id 且服务端有存量 → 重放历史，否则客户端回传。

        无新增 input（latest 为空）→ 返回 None（empty input 400）。
        """
        input_msgs = req.input or []
        session_id = req.previous_response_id or None
        store = serve.session_store
        hist, latest = _history(input_msgs)
        if session_id and store is not None:
            replay = store.read_all(session_id)
            if replay:
                hist = replay
        if not latest:
            return None
        return hist, latest, session_id

    async def _idempotency_gate(
        self, request, idem: _IdempotencyCache, rq_id: str | None
    ) -> tuple[bool, web.StreamResponse | None]:
        """幂等闸门：返回 (owned, 待返回响应)。owned=True 表示本请求抢到执行权；
        replay 非 None 时为命中结果（重放完整 SSE / 并发 409）。"""
        owned = False
        if rq_id:
            snap = idem.snapshot(rq_id)
            if snap is not None:
                state, chunks = snap
                if state == "done":
                    return False, await self._replay(request, chunks)  # 命中 → 原样重放
                return False, web.json_response(
                    {"ok": False, "error": CONFLICT_MESSAGE}, status=409)
            owned = idem.begin(rq_id)
            if not owned:  # 防御：与 snapshot 之间理论上无竞争，双保险兜底
                return False, web.json_response(
                    {"ok": False, "error": CONFLICT_MESSAGE}, status=409)
        return owned, None

    async def _replay(self, request, chunks: list[str]) -> web.StreamResponse:
        """幂等命中：不跑 agent，把首次请求捕获的完整 SSE 序列原样重放，并打 REPLAY_HEADER 标记。"""
        resp = web.StreamResponse(status=200)
        resp.headers.update({
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*",
            "X-Agentlab-Contract": CONTRACT,
            REPLAY_HEADER: "1",
            "Connection": "close",
        })
        await resp.prepare(request)
        for chunk in chunks:
            await resp.write(chunk.encode("utf-8"))
        with contextlib.suppress(Exception):
            await resp.write_eof()
        return resp


class _AsyncServer:
    """aiohttp 服务器适配：对外维持旧 ThreadingHTTPServer 的生命周期 API。

    `serve_forever` 在调用线程常驻（事件循环），`shutdown` 从任意线程线程安全地停它——
    语义与旧 server.serve_forever()/shutdown() 对齐，使 Serve/CLI/测试调用方无感底层框架。
    """

    def __init__(self, app: web.Application, host: str, port: int, serve_cfg: dict,
                 on_shutdown: Callable | None = None):
        self._app = app
        self.host = host
        self.port = port
        self.serve_cfg = serve_cfg
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        # 收尾钩子：在 loop 关闭前于同一 loop 内 await（长驻子进程需要 owner loop 收尾）
        self._on_shutdown = on_shutdown

    def serve_forever(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            self._runner = web.AppRunner(self._app)
            loop.run_until_complete(self._runner.setup())
            site = web.TCPSite(self._runner, self.host, self.port)
            loop.run_until_complete(site.start())
            loop.run_forever()
        finally:
            if self._runner is not None:
                with contextlib.suppress(Exception):
                    loop.run_until_complete(self._runner.cleanup())
            if self._on_shutdown is not None:
                with contextlib.suppress(Exception):
                    loop.run_until_complete(self._on_shutdown())
            loop.close()
            self._loop = None

    def shutdown(self) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)

    def server_close(self) -> None:
        # 资源清理已在 serve_forever 的 finally 完成；此方法仅为生命周期 API 对齐保留
        pass


class Serve:
    def __init__(self, cfg, port: int | None = None, host: str | None = None,
                 build_factory: Callable | None = None,
                 session_store=None,
                 multi_providers: dict | None = None,
                 synth_provider=None,
                 idem_store_path: str | None = None):
        from agentlab import __version__
        self.cfg = cfg
        self.port = port or 8643
        self.host = host or "127.0.0.1"
        self.version = __version__
        self._httpd = None
        # 测试注入点：替代真实 brain/LLM 后端的 Runner 构建器
        self._build_factory = build_factory
        self.build_backend = None
        # 会话存储注入点：测试传 InMemory/temp-dir；None 时 start 自动建 JSONL
        self._session_store = session_store
        # P2-2：multi-agent 答者注册表与汇总模型（None 时 build() 按配置装配；测试可直接注入）
        self._multi_providers = multi_providers
        self._synth_provider = synth_provider
        self.multi_registry: dict = {}
        self.synth_provider = None
        self.range_gateway = None  # L11：build() 时装配（_build_backend 返回三元组）
        # 契约层幂等（§4.7）：request_id 重复提交重放首次结果（有界 TTL）；
        # P0-04：默认挂 SQLite 背板（与默认 state.db 同目录），重启后已完成
        # 请求仍可重放，不重复执行。测试传 idem_store_path 指临时文件。
        if idem_store_path is None:
            idem_store_path = str(Path(__file__).resolve().parent / "idempotency.db")
        self.idem = _IdempotencyCache(store_path=idem_store_path)
        self.approvals = ApprovalManager()

    def build(self):
        if self._build_factory is not None:
            self.build_backend, self.n_brain = self._build_factory(), 0
            self.range_gateway = None  # 测试注入路径：不启用区段档案
        else:
            self.build_backend, self.n_brain, self.range_gateway = _build_backend(self.cfg)
        # P2-2/OPT-121：答者注册表（外部 ACP 长驻跨请求复用）+ 汇总模型（无 key → None 降级）
        from agentlab.runtime.multi import build_multi_registry, build_synth_provider

        self.multi_registry = self._multi_providers
        if self.multi_registry is None:
            self.multi_registry = build_multi_registry(self.cfg, self.cfg.vault_root)
        self.synth_provider = self._synth_provider
        if self.synth_provider is None and self._build_factory is None:
            self.synth_provider = build_synth_provider(self.cfg)

    def start(self) -> _AsyncServer:
        self.build()
        # P2-2/F5-021：审批策略可由 Ark 以环境变量下发（serve_manage 由插件启动时带上），
        # 避免插件去写后端 config.json（该文件含 API Key）。非法值 fail-closed 回落 risk_based。
        mode = str(os.environ.get("AGENTLAB_APPROVAL_MODE")
                   or getattr(self.cfg, "approval_mode", "") or "risk_based").strip().lower()
        if mode not in {"risk_based", "allow_all"}:
            mode = "risk_based"
        self.cfg.approval_mode = mode
        _slog("APPROVAL_MODE", mode,
              "(" + ("env" if os.environ.get("AGENTLAB_APPROVAL_MODE") else "config") + ")")
        sc = serve_config(self.cfg)
        # fail-closed：token 未配置即拒绝启动，不提供任何默认口令
        if not sc["token"]:
            raise RuntimeError(
                "serve token 未配置（fail-closed）。请在 config.json 的 serve.token 或"
                " 环境变量 AGENTLAB_SERVE_TOKEN 中显式指定一把钥匙再启动。"
            )
        if self.port == 8643 and not os.environ.get("AGENTLAB_SERVE_PORT"):
            self.port = sc["port"]
        if self.host == "127.0.0.1" and not os.environ.get("AGENTLAB_SERVE_HOST"):
            self.host = sc["host"]
        # 会话持久化：默认存到 trace_dir 同级 logs/sessions；测试可注入 temp/in-memory store
        self.session_store = self._session_store or JsonlSessionStorage(
            Path(getattr(self.cfg, "trace_dir", "logs/trace")).parent / "sessions"
        )
        self.serve_cfg = {"token": sc["token"], "heartbeat": sc["heartbeat"],
                          "approval_timeout_seconds": sc["approval_timeout_seconds"]}
        self.approvals = ApprovalManager(sc["approval_timeout_seconds"])
        self._httpd = _AsyncServer(_ServeApp(self).app(), self.host, self.port,
                                   self.serve_cfg, on_shutdown=self.aclose)
        return self._httpd

    def serve_forever(self):
        httpd = self.start()
        print(f"[SERVE] Agentlab {self.version} 监听 http://{self.host}:{self.port}"
              f" · tools={self.n_brain} brain tools 已接入"
              f"\n[SERVE] /v1/responses 需 Bearer token：{httpd.serve_cfg['token']}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.shutdown()
            httpd.server_close()

    def shutdown(self):
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()

    async def aclose(self) -> None:
        """收尾长驻外部 agent 进程（multi 答者注册表 / agent_consult 复用同一批实例）。

        由 _AsyncServer 在 owner loop 关闭前 await；单个 provider 失败不影响其余收尾。
        """
        providers = list((self.multi_registry or {}).values())
        for provider in providers:
            closer = getattr(provider, "close", None)
            if closer is None:
                continue
            try:
                await closer()
            except Exception:  # noqa: BLE001 —— 收尾失败不阻断退出
                _slog("ACLOSE_FAIL", getattr(provider, "name", "?"))
        self.multi_registry = {}
        closer = getattr(getattr(self, "build_backend", None), "close", None)
        if closer is not None:
            try:
                await closer()
            except Exception:  # noqa: BLE001 - cleanup must not mask shutdown
                _slog("LLM_ACLOSE_FAIL")


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="agentlab serve", description="启动 HTTP/SSE 服务供 Obsidian ark 对话")
    p.add_argument("-c", "--config", default=None, help="config.json 路径")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--host", default=None)
    args = p.parse_args(argv)

    cfg = _load_cfg(args.config)
    s = Serve(cfg, port=args.port, host=args.host)
    s.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
