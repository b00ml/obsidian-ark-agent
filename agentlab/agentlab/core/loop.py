"""事件驱动 ReAct 循环引擎（对齐 docs/04 §1）。

内核纯函数化：只接收 (agent, messages, emit) 并把结果折算进 messages；
不持有全局状态、不操作 UI。事件由外层订阅（CLI/日志/trace）。

含机械细节：多工具并行/串行调度、length 截断整批失败、terminate 终止、
steering/followUp 双队列、重复调用防打转、max_steps 兜底。
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

from agentlab.core.agent import Agent
from agentlab.core.context import Context, estimate_tokens
from agentlab.core.errors import AgentError
from agentlab.core.events import (
    PAYLOAD_TYPES, AgentEnd, AgentStart, Ev, EventPayload, MessageEnd, ToolEnd,
    ToolStart, TurnEnd, TurnStart,
)
from agentlab.core.guardrails import validate_output
from agentlab.core.llm import LLMProvider, LLMResponse
from agentlab.core.message import Message, ResultStopReason, ToolResult, TokenUsage, tool_result
from agentlab.memory.capture import extract_snippets, period
from agentlab.tools.registry import ConfirmFn, ToolRegistry
from agentlab.rag.assess import AnswerEvidence, sanitize_generated_answer
from agentlab.core.run_contract import (
    RunBudget, RunTrace, CURRENT_RUN_BUDGET, CURRENT_RUN_TRACE, action_fingerprint,
)
from agentlab.runtime.operation_verifier import build_verification_metadata

_log = logging.getLogger(__name__)

# 编批失败防护：整批输出被截断时放行的重发提示
_TRUNCATED_HINT = "[AGENT_OUTPUT_TRUNCATED] 该工具调用因输出被截断而失败，请重发"
# 空/纯思考回合续跑提示：模型只分析未动作时注入（学 tau：空失败不回放，改为轻 steer 拉回）
_INCOMPLETE_STEER = (
    "[AGENT_INCOMPLETE] 上一轮你只做了分析/思考，没有调用工具，也没有给出对用户问题的结论。"
    "若还需要信息，请直接调用工具；若已有结论，请直接输出最终回答。"
)
# 空/半截回答防终止（L10 语境）：续写指令——截断后保留半截并要求接着输出，禁止重复（OPT-110 观测修复）
_CONTINUE_STEER = (
    "[续写指令] 你上一条回复因输出长度被截断。请从截断处继续输出剩余内容，"
    "不要重复已输出的部分；若内容实际已完整，用一句话说明即可。"
)
# 空回合续跑的步数上限（step 只统计有工具回合，故用独立计数器防空换死循环）
_MAX_EMPTY_STEERS = 3
# 重复检测最近窗口：批次级签名入列，窗口内最旧记录按此上限淘汰
_REPETITION_WINDOW = 8
# 上下文压缩次数上限：防止摘要不收敛时空耗（max_steps 兜底收尾）
_MAX_COMPACT_ROUNDS = 3
# L10（OPT-106）：硬截断时单条工具结果保留的字符数
_HARD_TRIM_KEEP = 200
# OPT-116：打断后未执行工具的占位结果——保持 tool 对完整（API 要求每个 tool_call 有结果）
_ABORTED_TOOL_RESULT = "[已中止] 用户打断，本工具未执行"
# L10（OPT-106）：用量 ≥nudge 阈值时注入的一次性压缩提示（保留标记，_pin_directive 不钉）
_CONTEXT_NUDGE = (
    "[上下文提示] 上下文用量已超过 75% 预算。可调用 context_status 查看详情；"
    "若阶段性目标已完成，建议调用 compress_context 压缩较旧区段腾出空间。"
    "回答时避免复述此前检索到的大段原文。"
)

# 工具调用预算（成本上界）。为什么按"调用次数"而不是"步数"：`max_steps` 限制的是
# **批**数（一次响应可带多个 tool_call），实测 15 步里跑了 46 次工具、373k token，
# 步数上限完全没碰到。软阈值先给一次"收敛"提示，硬阈值再有一次"必须收尾"提示，
# 之后仍要调用就强制结束——目的是把单轮成本从"看模型心情"变成有上界。
_BUDGET_NUDGE = (
    "[预算提醒] 本轮工具调用已用 {used}/{hard} 次。请开始收敛：优先用**已有**结果作答，"
    "不要再用近义关键词反复试探；确实还缺信息时，一次把需要的调用发在同一批里。"
)
_BUDGET_STOP = (
    "[预算上限] 本轮工具调用已达 {used}/{hard} 次上限。请立刻基于已获得的信息给出结论，"
    "并明确列出哪些内容因未查完而无法确认——不要再调用工具。"
)
_BUDGET_FORCED_NOTE = (
    "已达到本轮工具调用上限（{used}/{hard} 次），自动收尾。已获得的信息见上；"
    "未查完的部分请拆成更小的任务重试。"
)

# Ev 枚举与各事件 payload 模型已迁至 core/events.py（C2 收官）；上方 import 即
# re-export，既有 `from agentlab.core.loop import Ev` 的消费端（cli/serve）无感兼容。


@dataclass
class RunHooks:
    """流式/交互回调（对齐 docs/03 §2.2）。"""

    on_text: Callable[[str], None] | None = None
    on_tool: Callable[[str, str, dict], None] | None = None  # (name, phase, payload)
    confirm: ConfirmFn | None = None


@dataclass
class RunConfig:
    """受限的运行时配置（也由 config.json 驱动）。"""

    max_steps: int = 15
    timeout: float | None = None  # 秒；None=不设超时
    context_budget: int = 8000
    max_tool_result_chars: int = 12000
    temperature: float = 0.3
    max_tokens: int | None = None
    disable_repetition_guard: bool = False
    signal: asyncio.Event | None = None  # 置位则中止（SSE 断连等，对齐 tau loop(signal)）
    memorize_every: int = 0  # 记忆沉淀周期：每 N 轮异步批量沉淀（0=关闭）
    depository: Any | None = None  # 具 .commit(content, tags, source_session, dedup) 记忆仓（对齐 store.MemoryStore）
    source_session: str = ""  # OPT-224：serve 会话锚，随沉淀写入 source_session（可追溯）
    extractor: Any | None = None  # #10①/OPT-123：LLM 语义提取器（serve 注入，具 async extract(entries)）；None=回退触发词路
    context_tools: bool = True  # 注入 context_status/compress_context（L10/OPT-106）
    context_nudge: float = 0.75  # 用量占比 ≥ 此值 → 注入一次压缩提示（L10/OPT-106）
    context_hard_trim: float = 0.95  # 用量占比 ≥ 此值 → 机械硬截断最旧工具结果（L10/OPT-106）
    context_window: int = 1048576  # 模型窗口（OPT-110 四期）：有效预算 = min(context_budget, 窗口×60%)
    compact_slice_tokens: int = 150000  # 单次折叠区段上限（token）：小步多次折叠
    max_tool_calls: int = 40  # 单轮工具调用次数上界（0/None=不限）。见 _BUDGET_NUDGE
    tool_call_nudge: float = 0.6  # 用量占比 ≥ 此值 → 注入一次"收敛"提示
    # P1 ContextAssembler shadow：可选规划器观察四区上下文契约，不改变兼容渲染器。
    context_assembler: Any | None = None
    context_assembler_mode: str = "shadow"
    context_plan: Any | None = None
    context_plan_history: list[Any] = field(default_factory=list)
    task_state: Any | None = None  # P1 structured task checkpoint for context planning
    task_state_store: Any | None = None  # optional durable operation ledger
    task_state_id: str = ""
    vault_root: str = ""  # trusted local root for pre-dispatch verification metadata
    # P0 answer-level citation/abstention gate.  ``shadow`` records the
    # decision without changing the answer; ``on`` fails closed after RAG use.
    answer_gate_mode: str = "off"
    answer_gate_require_citation: bool = True
    answer_gate_allow_bounded_partial: bool = False
    answer_evidence: Any | None = None
    answer_gate_result: Any | None = None
    range_recorder: Any = None  # L11/OPT-111：折叠区段归档器（serve 注入，具 .archive(list[Message])）；
    # 运行时按请求注入（同 signal），不进 from_config。
    # P0-06 run-level contracts.  None preserves the legacy limits above.
    budget: RunBudget | None = None
    trace: RunTrace | None = None
    max_llm_calls: int = 0
    max_react_rounds: int = 0
    max_plan_steps: int = 0
    max_retries: int = 0
    # P0-08 bounded Plan-and-Execute shadow. The compatibility ReAct loop
    # remains authoritative until an explicit caller executes this plan.
    plan: Any | None = None
    plan_mode: str = "shadow"

    @classmethod
    def from_config(cls, cfg) -> "RunConfig":
        limits = getattr(cfg, "limits", None)
        if limits is None:
            return cls()
        return cls(
            max_steps=int(getattr(limits, "max_steps", 15)),
            timeout=getattr(limits, "timeout", None),
            context_budget=int(getattr(limits, "context_budget", 8000)),
            max_tool_result_chars=int(getattr(limits, "max_tool_result_chars", 12000)),
            memorize_every=int(getattr(limits, "memorize_every", 0)),
            context_tools=bool(getattr(limits, "context_tools", True)),
            context_nudge=float(getattr(limits, "context_nudge", 0.75)),
            context_hard_trim=float(getattr(limits, "context_hard_trim", 0.95)),
            context_window=int(getattr(limits, "context_window", 1048576)),
            compact_slice_tokens=int(getattr(limits, "compact_slice_tokens", 150000)),
            max_tool_calls=int(getattr(limits, "max_tool_calls", 40)),
            tool_call_nudge=float(getattr(limits, "tool_call_nudge", 0.6)),
            max_llm_calls=int(getattr(limits, "max_llm_calls", 0) or 0),
            max_react_rounds=int(getattr(limits, "max_react_rounds", 0) or 0),
            max_plan_steps=int(getattr(limits, "max_plan_steps", 0) or 0),
            max_retries=int(getattr(getattr(cfg, "resilience", None), "max_retries", 0) or 0),
            context_assembler_mode=str(getattr(
                getattr(cfg, "context", None), "assembler_mode", "shadow"
            )),
            answer_gate_mode=str(getattr(
                getattr(cfg, "rag", None), "answer_gate_mode", "off"
            )),
            answer_gate_require_citation=bool(getattr(
                getattr(cfg, "rag", None), "answer_gate_require_citation", True
            )),
            answer_gate_allow_bounded_partial=bool(getattr(
                getattr(cfg, "rag", None), "answer_gate_allow_bounded_partial", False
            )),
        )


class AgentResult(BaseModel):
    final_output: str
    stop_reason: ResultStopReason
    messages: list[Message]
    usage: TokenUsage
    trace_id: str = ""
    answer_gate: dict[str, Any] | None = None
    run_trace: dict[str, Any] | None = None


class _RunnerEvents:
    """typed 事件总线（C2 收官）：事件名以 Ev 枚举强约束，payload 为 pydantic 模型
    （extra="forbid"）——emit 端字段拼错在构造时 ValidationError，事件与 payload
    类型不匹配 emit 时 TypeError；消费端属性访问，杜绝 **kw 裸字典静默取默认。"""

    def __init__(self) -> None:
        self._handlers: dict[Ev, list[Callable]] = {}

    def on(self, event: Ev, handler: Callable[[EventPayload], Any]) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def emit(self, event: Ev, payload: EventPayload) -> None:
        expected = PAYLOAD_TYPES.get(event)
        if expected is not None and not isinstance(payload, expected):
            raise TypeError(
                f"事件 {event.value} 的 payload 应为 {expected.__name__}，"
                f"收到 {type(payload).__name__}"
            )
        for h in self._handlers.get(event, []):
            try:
                h(payload)
            except Exception:
                # 订阅者异常不得中断主循环，但必须留痕（C3）：不是静默 pass
                _log.exception("事件订阅者处理 %s 失败", event.value)


class Runner:
    def __init__(self, provider: LLMProvider, registry: ToolRegistry | None = None,
                 memory=None):
        self.provider = provider
        self.registry = registry or ToolRegistry()
        self.memory = memory  # WorkingMemory | None；提供 summarizer 时超预算自动压缩
        self._captured_upto = 0  # 记忆沉淀游标：只消费新消息，避免重复沉淀（S6）
        self.events = _RunnerEvents()

    def on(self, event: Ev, handler: Callable) -> None:
        self.events.on(event, handler)

    async def run(
        self,
        agent: Agent,
        user_input: str,
        ctx: Context | None = None,
        hooks: RunHooks | None = None,
        cfg: RunConfig | None = None,
    ) -> AgentResult:
        cfg = cfg or RunConfig()
        hooks = hooks or RunHooks()
        if cfg.budget is None:
            cfg.budget = RunBudget.from_timeout(
                cfg.timeout,
                max_llm_calls=int(cfg.max_llm_calls or 0),
                max_tool_calls=int(cfg.max_tool_calls or 0),
                max_react_rounds=int(cfg.max_react_rounds or 0),
                max_plan_steps=int(cfg.max_plan_steps or 0),
                max_retries=int(cfg.max_retries or 0),
            )
        else:
            # Request-level overrides may tighten an injected budget, never widen it.
            for name in ("max_llm_calls", "max_tool_calls", "max_react_rounds",
                         "max_plan_steps", "max_retries"):
                requested = int(getattr(cfg, name, 0) or 0)
                existing = int(getattr(cfg.budget, name, 0) or 0)
                if requested and (not existing or requested < existing):
                    setattr(cfg.budget, name, requested)
        if cfg.trace is None:
            cfg.trace = RunTrace(run_id=cfg.budget.run_id)
        budget_token = CURRENT_RUN_BUDGET.set(cfg.budget)
        trace_token = CURRENT_RUN_TRACE.set(cfg.trace)
        if str(cfg.answer_gate_mode or "off").lower() in {"shadow", "on"}:
            if cfg.answer_evidence is None:
                cfg.answer_evidence = AnswerEvidence(
                    require_citation=bool(cfg.answer_gate_require_citation),
                    allow_bounded_partial=bool(cfg.answer_gate_allow_bounded_partial),
                )
        else:
            cfg.answer_evidence = None
        cfg.answer_gate_result = None
        messages: list[Message] = []

        # 初始上下文：ctx 拼出 system+history，再追加本次 user 输入
        if ctx is not None:
            messages = ctx.render(system_override=agent.instructions)
        else:
            messages = [Message(role="system", content=agent.instructions)]
        messages.append(Message(role="user", content=user_input))
        self.events.emit(Ev.AGENT_START, AgentStart(agent=agent.name, user_input=user_input))
        self.events.emit(Ev.TURN_START, TurnStart())

        # L10/OPT-106：context 工具按 run 注入（RunRegistryView 视图，不污染共享注册表；
        # serve 每请求新建 Runner，实例级交换并发安全），finally 恢复。
        base_registry = self.registry
        state = None
        if cfg.context_tools:
            from agentlab.core.context_tools import RunContextState, build_context_tools
            from agentlab.tools.registry import RunRegistryView
            state = RunContextState()
            state.messages = messages
            state.cfg = cfg
            self.registry = RunRegistryView(base_registry, build_context_tools(state))
        self._ctx_state = state
        try:
            if cfg.timeout:
                try:
                    result = await asyncio.wait_for(
                        self._loop(agent, messages, hooks, cfg), timeout=cfg.timeout
                    )
                except asyncio.TimeoutError:
                    total = _sum_usage(messages)
                    result = AgentResult(
                        final_output="",
                        stop_reason="cancelled",
                        messages=messages,
                        usage=total,
                    )
            else:
                result = await self._loop(agent, messages, hooks, cfg)
            if cfg.answer_gate_result is not None:
                result.answer_gate = cfg.answer_gate_result.to_dict()
            result.run_trace = cfg.trace.to_dict(cfg.budget) if cfg.trace is not None else None
            if result.run_trace is not None and getattr(cfg, "context_assembler", None) is not None:
                result.run_trace["context_assembler"] = self._context_runtime_metrics(result, cfg)
            # OPT-135：run 结束兜底沉淀——短会话（不足 memorize_every 轮）也留痕；
            # 游标保证只处理本轮未消费消息，周期沉淀过的不会重复提取。
            # 触发词/importance 过滤仍在，闲聊不会被灌入库。
            await self._maybe_deposit(cfg, messages, 0, force=True)
            return result
        finally:
            CURRENT_RUN_BUDGET.reset(budget_token)
            CURRENT_RUN_TRACE.reset(trace_token)
            self.registry = base_registry
            self._ctx_state = None

    @staticmethod
    def _context_runtime_metrics(result: AgentResult, cfg: RunConfig) -> dict[str, Any]:
        """Summarise planner evidence with runtime outcomes for shadow/on comparison."""
        events = []
        if cfg.trace is not None:
            events = [event for event in cfg.trace.events
                      if event.get("stage") == "context_plan"]
        last = events[-1] if events else {}
        gate = getattr(cfg, "answer_gate_result", None)
        pending_unknown = 0
        store = getattr(cfg, "task_state_store", None)
        task_id = str(getattr(cfg, "task_state_id", "") or "")
        if store is not None and task_id:
            try:
                pending_unknown = sum(
                    1 for row in store.pending_operations(task_id)
                    if row.get("status") == "unknown"
                )
            except Exception:
                pending_unknown = -1
        return {
            "mode": last.get("mode", getattr(cfg, "context_assembler_mode", "shadow")),
            "plan_updates": len(events),
            "selected": int(last.get("selected", 0) or 0),
            "omitted": int(last.get("omitted", 0) or 0),
            "used_tokens": int(last.get("used_tokens", 0) or 0),
            "scope_denied": int(last.get("scope_denied", 0) or 0),
            "tool_calls": int((cfg.trace.counters if cfg.trace else {}).get("tool_calls", 0) or 0),
            "citations": len(getattr(gate, "allowed_refs", ()) or ()) if gate else 0,
            "refusal": bool(getattr(gate, "abstained", False)) if gate else False,
            "task_completion": result.stop_reason == "done",
            "recovery_unknown": pending_unknown,
        }

    async def _loop(
        self, agent: Agent, messages: list[Message], hooks: RunHooks, cfg: RunConfig
    ) -> AgentResult:
        step = 0
        recent: list[str] = []
        compact_count = 0  # 连续压缩次数，防止压缩不收敛时空耗
        empty_stuck = 0  # 连续空回合数：防空换死循环（step 只计有工具回合）
        turn = 0  # 回合计数，驱动记忆沉淀周期（S6）
        # OPT-135：游标从本轮 user 输入起算（跳过 system 与重放历史）——
        # 旧值 0 会把整个重放历史每轮重复提取；消费即推进，跨 run 不重复
        self._captured_upto = max(0, len(messages) - 1)
        # 压缩锚点（L9/OPT-104）：messages[:anchor) 字节冻结、永不重折叠，
        # 已折叠检查点跨轮稳定 → provider 前缀缓存跨压缩命中；system 首消息永久冻结
        self._compact_anchor = 1 if messages and messages[0].role == "system" else 0
        state = getattr(self, "_ctx_state", None)  # L10：context 工具状态盒
        if state is not None:
            state.messages = messages
        nudged = False  # L10：压缩提示每次 run 至多一条
        tools_used = 0  # 本 run 已执行的工具调用次数（成本上界，区别于"批数" step）
        budget_nudged = False  # 收敛提示每次 run 至多一条
        eff_budget = self._effective_budget(cfg)  # OPT-110 四期：有效预算挂模型窗口

        while True:
            # 取消信号：SSE 断连等置位即中止，停止烧 token（对齐 tau loop(signal)）
            budget = cfg.budget
            if self._aborted(cfg):
                if budget is not None:
                    budget.cancel("aborted")
                return self._stop("aborted", messages, cfg=cfg)
            if budget is not None and budget.expired():
                budget.stop_reason = "deadline"
                return self._stop("cancelled", messages, cfg=cfg)
            # —— L10/OPT-106：模型主动压缩（compress_context 置的 force 标记） ——
            if state is not None and state.force_compact:
                state.force_compact = False
                messages, compact_count, compacted = await self._maybe_compact(
                    messages, cfg, compact_count, force=True)
                if compacted:
                    state.messages = messages
                    continue  # 压缩生效 → 重估（nudge/硬截断基准随之刷新）

            # 上下文预算保护（04 §1.5）：真实 usage 优先估算，超预算走"尽力压缩"而非硬
            # guardrail——先经 WorkingMemory 结构化摘要，压缩仍不收敛则继续跑（max_steps 兜底）。
            messages, compact_count, compacted = await self._maybe_compact(messages, cfg, compact_count)
            if compacted:
                if state is not None:
                    state.messages = messages
                continue  # 压缩生效 → 用压缩后历史重估

            # —— L10/OPT-106：nudge（≥75% 一次性提示）+ 硬截断（≥95% 机械兜底） ——
            if cfg.context_tools:
                est = sum(estimate_tokens(m) for m in messages)
                if est >= int(eff_budget * cfg.context_hard_trim):
                    self._hard_trim(messages, cfg, eff_budget)
                elif not nudged and est >= int(eff_budget * cfg.context_nudge):
                    messages.append(Message(role="user", content=_CONTEXT_NUDGE))
                    nudged = True
            self._update_context_plan(messages, cfg)
            if budget is not None and not budget.admit_round():
                return self._stop_with_note(messages, [],
                    "达到 ReAct 回合预算上限，已自动收尾。", reason="guardrail", cfg=cfg)
            if cfg.trace is not None:
                cfg.trace.count("rounds")
            if budget is not None and not budget.admit_llm():
                return self._stop_with_note(messages, [],
                    "达到 LLM 调用预算上限，已自动收尾。", reason="guardrail", cfg=cfg)
            llm_started = __import__("time").time()
            if cfg.trace is not None:
                cfg.trace.count("llm_calls")
            try:
                remaining = budget.remaining() if budget is not None else None
                call = self.provider.chat(
                    list(messages), tools=self.registry.schemas(),
                    temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                )
                resp = await asyncio.wait_for(call, timeout=remaining) if remaining is not None else await call
            except asyncio.TimeoutError:
                if budget is not None:
                    budget.stop_reason = "deadline"
                return self._stop("cancelled", messages, cfg=cfg)
            finally:
                if cfg.trace is not None:
                    cfg.trace.span("llm", llm_started,
                                   llm_calls=budget.llm_calls if budget else None)
            if self._aborted(cfg):
                # OPT-116：打断落在 LLM 调用期间——响应对用户不可见（未发送任何事件），
                # 直接丢弃不记录不执行；此前这里无检查，打断后仍会记录消息并把整批
                # 工具（如 B站流水线）跑完才停。半截历史由 run 收尾照常持久化。
                return self._stop("aborted", messages, cfg=cfg)
            self._record_assistant(resp, messages, hooks, step=step, cfg=cfg)
            # 记忆沉淀调度（S6）：每 memorize_every 轮对触发词命中的新消息做异步批量沉淀
            turn += 1
            await self._maybe_deposit(cfg, messages, turn)

            # —— 无工具调用 = 候选结束（guardrail / follow_up / steering / done）——
            if not resp.tool_calls:
                # B2：截断（length/max_tokens）且无工具 → 保留半截 + 续写指令。
                # （OPT-110 观测：裸重发会让模型**重新生成整篇回答** → 用户拿到内容重复的两遍；
                #  改为半截 assistant 留在历史 + 注 [续写指令]，从截断处接着输出。）
                if resp.stop_reason in ("length", "max_tokens"):
                    if step < cfg.max_steps:
                        messages.append(Message(role="user", content=_CONTINUE_STEER))
                        continue
                # B1：空/纯思考无动作 → 注入轻 steer 续跑，不再静默判 done（学 tau）
                if not (resp.content or "").strip():
                    if step < cfg.max_steps and empty_stuck < _MAX_EMPTY_STEERS:
                        empty_stuck += 1
                        self._steer_incomplete(messages, resp)
                        continue
                else:
                    empty_stuck = 0  # 模型给出文本 → 阶段性复位
                # ``_record_assistant`` may deterministically demote
                # out-of-scope wikilinks before emitting SSE.  The finish
                # gate, durable history, and returned final output must use
                # that same rendered value rather than the provider original.
                rendered_output = messages[-1].content if messages else (resp.content or "")
                result = self._finish_turn(agent, messages, rendered_output or "", cfg)
                if result is not None:
                    return result
                continue

            # —— 有工具调用 ——
            step += 1
            empty_stuck = 0  # 有动作 → 复位空回合计数
            if self._guard_truncated(resp, messages):  # 1) length 截断：整批失败让模型重发
                continue
            tool_calls = self._filter_flow_only(resp.tool_calls, messages)  # 2) 一致性
            if not tool_calls:
                continue

            # 3) 重复检测（无敌防打转）
            if not cfg.disable_repetition_guard:
                end = self._detect_repetition(tool_calls, recent, messages, cfg=cfg)
                if end is not None:
                    return end

            # 3.5) 工具调用预算（成本上界）：已经提醒过收尾还在调 → 强制结束，
            #      避免"一次响应塞一大堆调用"绕过 max_steps 把单轮成本拖到几十万 token。
            hard = int(cfg.max_tool_calls or 0)
            if (budget is not None and not budget.admit_tool(len(tool_calls))) or (hard and tools_used >= hard):
                return self._stop_with_note(
                    messages, tool_calls,
                    _BUDGET_FORCED_NOTE.format(used=tools_used, hard=hard),
                    reason="guardrail", cfg=cfg)

            # 4) 执行批次（并行/串行）
            tool_started = __import__("time").time()
            executed = await self._execute_tool_batch(tool_calls, hooks, cfg)
            if cfg.trace is not None:
                cfg.trace.span("tool_batch", tool_started,
                               tool_calls=len(tool_calls))
            messages.extend(
                Message(role="tool", tool_call_id=r.tool_call_id, content=r.content)
                for r in executed
            )
            tools_used += len(tool_calls)
            if hard and tools_used >= hard:
                # 说明只能给一次：下一轮若仍要调工具，上面的检查会强制收尾
                messages.append(Message(role="user",
                                        content=_BUDGET_STOP.format(used=tools_used, hard=hard)))
            elif hard and not budget_nudged and tools_used >= max(1, int(hard * cfg.tool_call_nudge)):
                budget_nudged = True
                messages.append(Message(role="user",
                                        content=_BUDGET_NUDGE.format(used=tools_used, hard=hard)))

            # 5) terminate 结束 / 6) steering 续跑 / max_steps 兜底
            end = self._post_tool(agent, messages, cfg, step, executed)
            if end is not None:
                return end

    @staticmethod
    def _update_context_plan(messages: list[Message], cfg: RunConfig) -> None:
        """Refresh the optional ContextAssembler plan for the current turn."""
        assembler = getattr(cfg, "context_assembler", None)
        if assembler is None:
            return
        try:
            system = [messages[0]] if messages and messages[0].role == "system" else []
            history = [m for m in messages[1:] if m.role in {"user", "assistant"}]
            observations = [m for m in messages[1:] if m.role == "tool"]
            cfg.context_plan = assembler.assemble(
                task_state=getattr(cfg, "task_state", None),
                instructions=system,
                history=history,
                tool_observations=observations,
                mode=getattr(cfg, "context_assembler_mode", "shadow"),
            )
            # Keep bounded plan snapshots for shadow/on attribution.  They
            # remain request-local and the evaluator exports only hashes.
            cfg.context_plan_history.append(cfg.context_plan)
            del cfg.context_plan_history[:-20]
            if cfg.trace is not None:
                metrics = (cfg.context_plan.metrics()
                           if callable(getattr(cfg.context_plan, "metrics", None))
                           else {
                               "mode": getattr(cfg.context_plan, "mode", "shadow"),
                               "used_tokens": int(getattr(cfg.context_plan, "used_tokens", 0) or 0),
                               "omitted": len(getattr(cfg.context_plan, "omitted", ()) or ()),
                           })
                cfg.trace.span("context_plan", __import__("time").time(), **metrics)
        except Exception as exc:  # noqa: BLE001 - shadow never blocks a run
            _log.warning("ContextAssembler shadow failed: %s", exc)
            cfg.context_plan = None

    # ── 取消信号：置位即中止（SSE 断连等，对齐 tau loop(signal)） ──
    def _aborted(self, cfg: RunConfig) -> bool:
        return cfg.signal is not None and cfg.signal.is_set()

    # ── 记忆沉淀调度（S6）：周期 + 触发词命中 → 异步批量写入长期记忆 ──
    async def _maybe_deposit(self, cfg: RunConfig, messages: list[Message], turn: int,
                             force: bool = False) -> None:
        depo = cfg.depository
        if depo is None or not cfg.memorize_every:
            return
        if not force and not period(turn, cfg.memorize_every):
            return
        new = messages[self._captured_upto:]
        self._captured_upto = len(messages)  # 消费即推进游标，防下次重复沉淀
        if not new:
            return
        if cfg.extractor is not None:
            # #10①/OPT-123：LLM 语义提取优先（importance 打分、低分不入库）；
            # 提取/落库失败在 deposit_via_extractor 内部兜底，不影响主循环
            from agentlab.memory.extract import deposit_via_extractor

            await deposit_via_extractor(depo, cfg.extractor, new,
                                        source_session=cfg.source_session)
            return
        snap = extract_snippets(new)
        if not snap:
            return
        loop = asyncio.get_running_loop()
        # 异步批量沉淀：同步 brain 调用放 executor，不阻塞事件循环（对齐"异步批量沉淀"）
        for seg in snap:
            await loop.run_in_executor(None, self._deposit, depo, seg,
                                       getattr(cfg, "source_session", ""))

    @staticmethod
    def _deposit(depo: Any, content: str, source_session: str = "") -> dict:
        """单条沉淀（在线程池执行）；仓库不可用等异常不影响主循环，仅留痕。"""
        try:
            kwargs = {
                "tags": None, "source_session": source_session, "dedup": True,
                "mem_type": "sessions", "bucket": True, "source": "assistant",
                "source_ref": f"session:{source_session}" if source_session else "",
                "candidate_first": True,
            }
            # Third-party/test depositories may still expose the pre-S1
            # signature.  Pass only supported keyword arguments.
            import inspect
            params = inspect.signature(depo.commit).parameters
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
                kwargs = {k: v for k, v in kwargs.items() if k in params}
            return depo.commit(content, **kwargs)
        except Exception:
            _log.exception("记忆沉淀失败（depository=%s）", type(depo).__name__)
            return {"status": "error"}

    # ── 1) 上下文压缩：超预算/模型请求(force) → 折叠旧区段；无进展原样返回 ──
    async def _maybe_compact(
        self, messages: list[Message], cfg: RunConfig, compact_count: int,
        force: bool = False,
    ) -> tuple[list[Message], int, bool]:
        eff = self._effective_budget(cfg)
        est = sum(estimate_tokens(m) for m in messages)
        over = est > eff
        if self.memory is None or not (force or over):
            return messages, 0, False
        if not force and compact_count >= _MAX_COMPACT_ROUNDS:
            return messages, 0, False
        keep = max(1, int(eff * 0.5))
        condensed = await self.memory.condense(
            messages, keep_recent_tokens=keep, anchor=self._compact_anchor,
            max_fold_tokens=cfg.compact_slice_tokens,
        )
        if condensed is not messages:  # 仅真正产出新压缩结果才替换；无进展则放行
            # 锚点前移跨过本次插入的头部（检查点+可能钉住的指令）：
            # [0, anchor+head) 成为新的字节稳定前缀，下轮起不再重折叠（L9/OPT-104）
            self._compact_anchor += getattr(self.memory, "last_condense_head", 0)
            await self._archive_range(
                cfg, list(getattr(self.memory, "last_folded", None) or []))
            return condensed, compact_count + 1, True
        return messages, 0, False  # 未压缩轮次计数归零（连续压缩才受 MAX_ROUNDS 限）

    # ── L11/OPT-111：折叠区段原文落 ranges.jsonl + 向量索引（窗口外内容走召回而非丢失）──
    @staticmethod
    async def _archive_range(cfg: RunConfig, folded: list[Message]) -> None:
        recorder = getattr(cfg, "range_recorder", None)
        if recorder is None or not folded:
            return
        try:  # 嵌入是网络调用，放线程不阻塞事件循环；await 内联保证折叠完成时档案已落（确定性）
            await asyncio.to_thread(recorder.archive, folded)
        except Exception:
            _log.exception("L11 区段归档失败")

    # ── OPT-110 四期：有效预算挂模型窗口（学 pi shouldCompact 挂窗口，换模型改 window 即可） ──
    @staticmethod
    def _effective_budget(cfg: RunConfig) -> int:
        if cfg.context_window > 0:
            return min(cfg.context_budget, int(cfg.context_window * 0.6))
        return cfg.context_budget

    # ── L10/OPT-106：95% 机械硬截断——从冻结区之后最旧的 tool 结果截起，前缀字节不动 ──
    def _hard_trim(self, messages: list[Message], cfg: RunConfig, budget: int) -> None:
        floor = int(budget * cfg.context_hard_trim)
        for m in messages[self._compact_anchor:]:
            if sum(estimate_tokens(x) for x in messages) <= floor:
                return
            if m.role == "tool" and len(m.content or "") > _HARD_TRIM_KEEP:
                m.content = ((m.content or "")[:_HARD_TRIM_KEEP]
                             + "…[已硬截断（上下文紧急），完整结果请重新检索]")

    # ── 记录 assistant 消息：usage/stop_reason 回写 + 事件 + 文本钩子 ──
    def _record_assistant(self, resp: LLMResponse, messages: list[Message],
                          hooks: RunHooks, step: int = 0,
                          cfg: RunConfig | None = None) -> None:
        content = resp.content
        if content and cfg is not None and cfg.answer_evidence is not None:
            content = sanitize_generated_answer(
                content, cfg.answer_evidence.candidate_refs,
            )
        messages.append(
            Message(
                role="assistant",
                content=content,
                tool_calls=resp.tool_calls or None,
                usage=resp.usage,
                stop_reason=resp.stop_reason,  # type: ignore[arg-type]
            )
        )
        self.events.emit(
            Ev.MESSAGE_END,
            MessageEnd(role="assistant", content=content, usage=resp.usage, step=step),
        )
        if hooks.on_text and content:
            hooks.on_text(content)

    # ── 空回合拉回：剔除空 assistant（学 tau：空失败不回放下轮），注入轻 steer 续跑 ──
    @staticmethod
    def _steer_incomplete(messages: list[Message], resp: LLMResponse) -> None:
        if (
            messages
            and messages[-1].role == "assistant"
            and not (messages[-1].content or "").strip()
            and not messages[-1].tool_calls
        ):
            messages.pop()
        messages.append(Message(role="user", content=_INCOMPLETE_STEER))

    # ── 无工具调用：返回结束 AgentResult 或 None（注入 follow_up/steering 后需续跑）──
    def _finish_turn(self, agent: Agent, messages: list[Message], output: str,
                     cfg: RunConfig) -> AgentResult | None:
        ok = all(g(output) for g in (agent.guardrails or [])) if agent.guardrails else True
        ok = ok and validate_output(output)
        if not ok:
            return self._stop("guardrail", messages)
        if cfg.answer_evidence is not None:
            gate = cfg.answer_evidence.check(output)
            cfg.answer_gate_result = gate
            if gate is not None and str(cfg.answer_gate_mode).lower() == "on" \
                    and not gate.allowed and not gate.abstained:
                if gate.decision == "conflicting":
                    safe_output = "检索到的资料存在冲突，暂时无法安全作答，请先确认采用哪一份来源。"
                else:
                    safe_output = "当前检索资料不足以支持带来源的可靠回答，请补充资料或缩小问题范围。"
                # Do not leave the rejected model claim in the replayable
                # assistant history; retain only the safe user-visible result.
                if messages and messages[-1].role == "assistant":
                    messages[-1].content = safe_output
                return self._stop("guardrail", messages, final_output=safe_output)
        self.events.emit(Ev.TURN_END, TurnEnd())
        injected = agent.drain_follow_ups() or agent.drain_steering()
        if injected:
            messages.extend(injected)
            return None
        self.events.emit(Ev.AGENT_END, AgentEnd(stop_reason="done"))
        return self._stop("done", messages, final_output=output)

    # ── 1) length/max_tokens 截断防护：整批失败，让模型重发（返回是否应 continue）──
    def _guard_truncated(self, resp: LLMResponse, messages: list[Message]) -> bool:
        if resp.stop_reason not in ("length", "max_tokens"):
            return False
        for c in resp.tool_calls or []:
            messages.append(Message(role="tool", tool_call_id=c.id, content=_TRUNCATED_HINT))
        return True

    # ── 2) 一致性：拒绝执行 disable_model_invocation 的工具，返回放行列表 ──
    def _filter_flow_only(self, tool_calls: list, messages: list[Message]) -> list:
        kept: list = []
        for c in tool_calls:
            try:
                t = self.registry.get(c.function.name)
                if t.disable_model_invocation:
                    raise AgentError("AGENT_TOOL_PERMISSION",
                                     f"工具 {c.function.name} 仅供流程调用，禁止模型触发")
                kept.append(c)
            except AgentError as e:
                messages.append(
                    Message(role="tool", tool_call_id=c.id,
                            content=f"[{e.code}] {e.message}", name=c.function.name)
                )
        return kept

    # ── 3) 重复调用防打转（OPT-117 批次级）：整批签名连续两轮完全相同 → 中断 ──
    def _detect_repetition(self, tool_calls: list, recent: list[str],
                           messages: list[Message], cfg: RunConfig | None = None) -> AgentResult | None:
        """以"一个响应批"为比较单位：批内各调用签名排序后拼接，连续两批完全一致才算打转。

        旧版按扁平列表比对最近两次调用——多工具批（collect+read_queue）之后单独
        复读 read_queue 属**合法状态复查**（collect 刚改过队列），却被误判打转静默
        中止（2026-09-07 收件箱实测）。真循环（同批反复重发）依旧两轮即拦。
        """
        sigs = sorted(action_fingerprint(c.function.name, c.function.arguments)
                      for c in tool_calls)
        recent.append("|".join(sigs))
        looping = len(recent) >= 2 and recent[-1] == recent[-2]
        del recent[:-_REPETITION_WINDOW]  # 仅保留最近窗口
        if not looping:
            return None
        return self._stop_with_note(
            messages, tool_calls,
            "连续两轮重复调用同一组工具（名称与参数完全相同），已自动中止以免空转。"
            "如需继续，请更换参数，或先执行会改变状态的操作再复核。", cfg=cfg)

    # ── OPT-117：中止时给用户可见说明 + 悬挂 tool_calls 补占位（历史合法） ──
    def _stop_with_note(self, messages: list[Message], tool_calls: list,
                        note: str, reason: ResultStopReason = "max_steps",
                        cfg: RunConfig | None = None) -> AgentResult:
        """防打转/步数上限收尾。此前静默 _stop：final_output 空、UI 无任何解释，
        且未执行批的 assistant.tool_calls 悬挂在历史里（下次请求可能被 API 拒绝）。"""
        for c in tool_calls:
            messages.append(Message(
                role="tool", tool_call_id=c.id, name=c.function.name,
                content="[repeat_guard] 因连续重复调用被中止，本工具未执行"))
        messages.append(Message(role="assistant", content=note))
        return self._stop(reason, messages, final_output=note, cfg=cfg)

    # ── 4) 执行批次：工具粒度串并行调度（S6） ──
    async def _execute_tool_batch(self, tool_calls: list, hooks: RunHooks,
                                  cfg: RunConfig) -> list[ToolResult]:
        """S6 工具粒度串并行：按 batch 内每工具 execution_mode 精确调度，而非整体降级。

        旧"任一 sequential → 整批全串行"会把相互独立的 parallel 工具一并拖成串行。
        现切成段：连续 parallel 工具合并在同一 gather 里并发，sequential 工具自成单段
        单独 await；段间串行以保序，且 sequential 工具绝不与任何其他工具并发（保留其
        独占假设）。结果按原工具调用序还原。
        """
        def _mode(c) -> str:
            return self.registry.get(c.function.name).execution_mode

        results: list[ToolResult | None] = [None] * len(tool_calls)
        i = 0
        while i < len(tool_calls):
            if self._aborted(cfg):
                # OPT-116：批内打断即停——未执行的工具补占位结果（tool 对保持完整，
                # 历史合法），已执行的真实结果保留
                for j in range(i, len(tool_calls)):
                    results[j] = tool_result(tool_calls[j].id, _ABORTED_TOOL_RESULT)
                break
            if _mode(tool_calls[i]) == "sequential":
                results[i] = await self._execute_one(tool_calls[i], hooks, cfg)
                i += 1
                continue
            j = i  # 收集一段连续 parallel 工具批
            while j < len(tool_calls) and _mode(tool_calls[j]) != "sequential":
                j += 1
            got = await asyncio.gather(*(self._execute_one(c, hooks, cfg) for c in tool_calls[i:j]))
            results[i:j] = got
            i = j
        return [r for r in results if r is not None]

    # ── 5)+6) terminate 终止 / steering 续跑 / max_steps 兜底，返回结束结果或 None ──
    def _post_tool(self, agent: Agent, messages: list[Message], cfg: RunConfig,
                   step: int, executed: list[ToolResult]) -> AgentResult | None:
        if executed and all(r.terminate for r in executed):
            return self._stop("terminate", messages)
        steering = agent.drain_steering()
        if steering:
            messages.extend(steering)
        if step >= cfg.max_steps:
            # OPT-117：步数上限收尾带用户可见说明（此前静默空终，UI 无解释）
            return self._stop_with_note(
                messages, [],
                f"已连续执行 {step} 步仍未收敛，达到步数上限自动收尾。"
                "可基于以上结果继续追问，或拆小任务重试。")
        return None

    def _stop(self, reason: ResultStopReason, messages: list[Message],
              final_output: str = "", cfg: RunConfig | None = None) -> AgentResult:
        self.events.emit(Ev.AGENT_END, AgentEnd(stop_reason=reason))
        return AgentResult(
            final_output=final_output, stop_reason=reason,
            messages=messages, usage=_sum_usage(messages),
            run_trace=cfg.trace.to_dict() if cfg is not None and cfg.trace is not None else None,
        )

    async def _execute_one(
        self, call, hooks: RunHooks, cfg: RunConfig
    ) -> ToolResult:
        started_at = __import__("time").time()
        self.events.emit(
            Ev.TOOL_START,
            ToolStart(name=call.function.name, arguments=call.function.arguments),
        )
        if hooks.on_tool:
            hooks.on_tool(call.function.name, "start", {})

        def _progress(name: str, elapsed: float) -> None:
            # 长任务心跳回流给调用方（CLI 打印"运行中…"，判断是否还活着）
            if hooks.on_tool:
                hooks.on_tool(name, "progress", {"elapsed": elapsed})

        task_store = getattr(cfg, "task_state_store", None)
        task_id = str(getattr(cfg, "task_state_id", "") or "")
        operation_id = ""
        operation_context_token = None
        tracked_tool = None
        blocked_unknown = False
        blocked_settled = False
        settled_status = ""
        if task_store is not None and task_id:
            try:
                tracked_tool = self.registry.get(call.function.name)
                action_id = action_fingerprint(call.function.name, call.function.arguments)
                operation_id = hashlib.sha1(
                    f"{task_id}:{action_id}".encode("utf-8", "replace")
                ).hexdigest()[:32]
                # Checkpoint rows created before action fingerprints used the
                # raw JSON string.  Keep an exact legacy lookup so an upgrade
                # cannot make an old unknown row invisible to recovery.
                legacy_operation_id = hashlib.sha1(
                    f"{task_id}:{call.function.name}:{call.function.arguments}".encode(
                        "utf-8", "replace"
                    )
                ).hexdigest()[:32]
                previous = task_store.get(task_id)
                if previous is not None and any(
                    row.get("operation_id") == legacy_operation_id
                    for row in previous.pending_tools
                ):
                    operation_id = legacy_operation_id
                state = task_store.plan_tool(
                    task_id,
                    operation_id=operation_id,
                    tool_name=call.function.name,
                    arguments_hash=action_id,
                    permission=getattr(tracked_tool, "permission", "read"),
                    side_effects=getattr(tracked_tool, "side_effects", "") or "",
                    idempotent=getattr(tracked_tool, "idempotent", None),
                    verification=build_verification_metadata(
                        call.function.name, call.function.arguments,
                        vault_root=getattr(cfg, "vault_root", None),
                        operation_id=operation_id,
                    ),
                )
                try:
                    from agentlab.runtime.operation_context import bind_operation_id
                    operation_context_token = bind_operation_id(operation_id)
                except (ImportError, ModuleNotFoundError):
                    operation_context_token = None
                entry = next((row for row in state.pending_tools
                              if row.get("operation_id") == operation_id), None)
                # A stale non-idempotent write must be checked externally or by
                # HITL; re-running it from a recovered checkpoint is unsafe.
                if entry and entry.get("status") == "unknown" and (
                        getattr(tracked_tool, "permission", "read") != "read"
                        or getattr(tracked_tool, "side_effects", "") not in {"", "none"}):
                    # Idempotency is not proof that the external operation did
                    # not already happen.  Every side effect must be settled
                    # by the verifier or explicit human evidence first.
                    blocked_unknown = True
                elif entry and entry.get("status") in {"succeeded", "failed", "cancelled"} and \
                        getattr(tracked_tool, "permission", "read") != "read":
                    # A checkpointed terminal side effect is evidence that
                    # this exact operation was already settled.  Do not let a
                    # later model turn turn the ledger into a replay queue.
                    blocked_settled = True
                    settled_status = str(entry.get("status"))
                elif entry and entry.get("status") == "planned":
                    task_store.update_tool(task_id, operation_id, "running")
            except Exception as exc:  # checkpoint is a shadow aid, never a hard dependency
                _log.warning("task operation ledger unavailable: %s", exc)

        try:
            if blocked_unknown:
                result = tool_result(
                    call.id,
                    "[AGENT_SIDE_EFFECT_UNKNOWN] 上一次运行可能已执行该写操作；"
                    "已暂停重发，请先查询外部状态或经用户确认后重试。",
                )
            elif blocked_settled:
                result = tool_result(
                    call.id,
                    "[AGENT_OPERATION_ALREADY_SETTLED] 该写操作已记录为 "
                    f"{settled_status}，本次不会自动重放。",
                )
            else:
                pending = self.registry.execute(
                    call, confirm=hooks.confirm, max_result_chars=cfg.max_tool_result_chars,
                    progress=_progress, signal=cfg.signal,
                )
                remaining = cfg.budget.remaining() if cfg.budget is not None else None
                result = await asyncio.wait_for(pending, timeout=remaining) if remaining is not None else await pending
        except asyncio.TimeoutError:
            if cfg.budget is not None:
                cfg.budget.stop_reason = "deadline"
            result = tool_result(call.id, "[AGENT_DEADLINE] 工具调用超过本次请求剩余截止时间")
        except AgentError as e:
            # 未注册/权限拒绝/参数错误：包装成工具结果，让模型感知并纠错，不中断循环
            result = ToolResult(
                role="tool", tool_call_id=call.id, content=f"[{e.code}] {e.message}"
            )
        finally:
            if operation_context_token is not None:
                try:
                    from agentlab.runtime.operation_context import reset_operation_id
                    reset_operation_id(operation_context_token)
                except (ImportError, ModuleNotFoundError):
                    pass
        if task_store is not None and task_id and operation_id and not (
                blocked_unknown or blocked_settled):
            try:
                content = str(result.content or "")
                if "超时" in content or "仍在后台运行" in content or "[已中止]" in content:
                    outcome = "unknown" if getattr(tracked_tool, "permission", "read") != "read" else "failed"
                elif content.startswith("[工具执行失败]") or content.startswith("[AGENT_"):
                    outcome = "failed"
                else:
                    outcome = "succeeded"
                task_store.update_tool(
                    task_id, operation_id, outcome,
                    result_ref=f"tool:{call.function.name}:{call.id}",
                    error=content if outcome in {"failed", "unknown"} else "",
                )
            except Exception as exc:
                _log.warning("task operation settlement unavailable: %s", exc)
        if cfg.answer_evidence is not None:
            cfg.answer_evidence.observe(call.function.name, result.content)
        if hooks.on_tool:
            hooks.on_tool(
                call.function.name, "end",
                {"result": result.content, "terminate": result.terminate},
            )
        self.events.emit(
            Ev.TOOL_END, ToolEnd(name=call.function.name, result=result.content)
        )
        if cfg.trace is not None:
            cfg.trace.count("tool_calls")
            cfg.trace.span(
                "tool", started_at, tool_name=call.function.name,
                tool_call_id=call.id,
                action_fingerprint=action_fingerprint(call.function.name, call.function.arguments),
                result_fingerprint=action_fingerprint(
                    call.function.name, call.function.arguments, result=result.content),
            )
        return result


def _sum_usage(messages: list[Message]) -> TokenUsage:
    total = TokenUsage()
    for m in messages:
        if m.usage is not None:
            total.input_tokens += m.usage.input_tokens
            total.output_tokens += m.usage.output_tokens
            total.cache_read_tokens += m.usage.cache_read_tokens
            total.cache_write_tokens += m.usage.cache_write_tokens
    return total
