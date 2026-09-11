"""P2-2/OPT-121 一期后端：Multi-Agent 并行作答 + 主 Agent 汇总。

- BranchProvider：答者统一抽象（OPT-112 留口在此定形）——external = 外部 ACP
  agent consult（复用 ExternalAgent 外观，惰性 spawn + 自愈原样继承）；
  internal = 本地 agentlab 全能力 ReAct（带工具/记忆/HITL，经 _run_agent）。
- run_branches：asyncio.gather 并行问答；单支失败只降级该支（失败支不计入汇总），
  外部支全败时回退本地单 agent 直答，本地支也败才向调用方上抛。
- synthesize：主 Agent 汇总——一次无工具 provider 调用整合成功支的答案
  （multi-synth-user.st），最终答复沿既有 output_text.delta 流出，老前端无感。

SSE 增量事件（contract v1 加法演进，未识别事件前端跳过）：
  response.branch.started / response.branch.done / response.branch.failed
  response.multi.summary（ok_count / failed / degraded / synth_error）

会话语义（一期边界）：外部支为无状态问答（与 agent_consult 一致）；会话存储只
持久化最终汇总答复，各分支过程不落 session jsonl（需要分支留痕时走 trace）。
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from agentlab.core.message import Message
from agentlab.runtime.serve_contract import _Sink

# 分支全文进 SSE/汇总事件的上限（与工具输出截断同口径，防单支超大答案刷爆前端）
BRANCH_TEXT_CAP = 20000

_log = logging.getLogger(__name__)

# F5-018：multi 阶段的审批门禁工具名。@ 一次外部 agent 属 danger 动作，
# 整轮**只确认一次**（不是每支一次）；这个名字也是 danger_allowlist 的放行键。
GATE_TOOL_NAME = "multi_consult"


class _GateTool:
    """审批门禁用的合成工具描述：让 multi 阶段复用既有的 danger 审批链路。

    risk_based 下未列入 danger_allowlist → 弹一次审批卡；
    allow_all 下 serve_confirm 自动放行、不弹卡（鉴权与白名单仍生效）。
    """

    name = GATE_TOOL_NAME
    permission = "danger"

SELF_AGENT_NAME = "agentlab"  # 本地主 agent 在 multi 名单中的保留名


@dataclass
class BranchOutcome:
    """单支作答结果：失败支 ok=False 且 error 记因，不参与汇总内容。"""

    name: str
    kind: str  # "internal" | "external" | "unknown"
    ok: bool = False
    text: str = ""
    error: str = ""
    elapsed: float = 0.0
    # F5-017：正文超过 consult_result_max_chars 时截断，全文归档到 session range；
    # ref 形如 session/<sid>#<seq>，主 Agent 可经 rag_retrieve 的 session 路召回。
    truncated: bool = False
    ref: str = ""

    def public(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "ok": self.ok,
            "error": self.error,
            "elapsed": round(self.elapsed, 2),
            "truncated": self.truncated,
            "ref": self.ref,
            "text": self.text[:BRANCH_TEXT_CAP],
        }


class BranchProvider(Protocol):
    """答者统一抽象：name 为 multi 名单引用键，kind 区分来源。"""

    name: str
    kind: str

    async def answer(self, question: str, signal: asyncio.Event | None = None) -> str: ...
    async def close(self) -> None: ...


class ExternalBranchProvider:
    """外部 ACP agent 分支：ExternalAgent 长驻复用（由 Serve 持有，不随请求关闭）。"""

    kind = "external"

    def __init__(self, name: str, command: str, args: list[str] | None = None,
                 cwd: str = "", env: dict[str, str] | None = None,
                 timeout: float = 300.0):
        from agentlab.connectors.acp_agent import ExternalAgent

        self.name = name
        self._agent = ExternalAgent(name, command, args=args, cwd=cwd,
                                    env=env, timeout=timeout)

    async def answer(self, question: str, signal: asyncio.Event | None = None) -> str:
        return await self._agent.consult(question)

    async def close(self) -> None:
        """关闭长驻外部进程（Serve 收尾统一调用；幂等）。"""
        await self._agent.close()


class InternalBranchProvider:
    """本地 agentlab 全能力分支：完整 Runner 跑一轮 ReAct（工具事件不进主流）。"""

    kind = "internal"

    def __init__(self, cfg, build_backend: Callable, project_id: str | None = None,
                 approvals=None, run_id: str = "", approval_sink=None):
        self.name = SELF_AGENT_NAME
        self._cfg = cfg
        self._build_backend = build_backend
        self._project_id = project_id
        self._approvals = approvals
        self._run_id = run_id
        self._approval_sink = approval_sink
        self.last_tokens = 0

    async def answer(self, question: str, signal: asyncio.Event | None = None) -> str:
        from agentlab.runtime.serve import _run_agent

        # 分支文本/工具轨迹仍对主流静默；只有 HITL 事件必须上行，否则危险工具会
        # 卡到超时而用户根本看不到审批卡。
        approval_sink = self._approval_sink
        class _BranchSink:
            def text(self, content):
                return None

            def tool_start(self, name):
                return None

            def tool_end(self, name, result):
                return None

            def event(self, payload):
                if str(payload.get("type", "")).startswith("approval.") and approval_sink:
                    approval_sink.event(payload)

        summary = await _run_agent(
            self._cfg, self._build_backend, [], question,
            _BranchSink(),  # 分支过程静默；HITL 事件例外，必须交给主工作台
            signal=signal, project_id=self._project_id,
            approvals=self._approvals, run_id=self._run_id)
        self.last_tokens = int(summary.get("tokens") or 0)
        return summary["final_output"]

    async def close(self) -> None:
        """内置分支无常驻资源：每轮 Runner 用完即弃，无需清理。"""
        return None


def build_multi_registry(cfg, vault_root: str) -> dict[str, BranchProvider]:
    """外部 ACP → 答者注册表（agents.external；command 为空 = 未启用，跳过）。

    实例由 Serve 长期持有跨请求复用（ExternalAgent spawn 一次的会话策略不受影响）。
    """
    registry: dict[str, BranchProvider] = {}
    for ext in getattr(cfg.agents, "external", None) or []:
        if not ext.command:
            continue
        registry[ext.name] = ExternalBranchProvider(
            ext.name, ext.command, args=ext.args,
            cwd=ext.cwd or vault_root, env=ext.env, timeout=ext.timeout)
    return registry


def build_synth_provider(cfg):
    """汇总模型：复用主模型 resilient provider（无工具单次调用）。

    未配置 key（_make_resilient 抛 SystemExit）/构造失败 → None（调用方降级为
    直取最优分支答案）。SystemExit 继承 BaseException，必须显式捕获。
    """
    try:
        from agentlab.runtime.cli import _make_resilient

        return _make_resilient(cfg)
    except (Exception, SystemExit):  # noqa: BLE001 —— 汇总模型缺失只是能力降级
        return None


async def run_branches(providers: dict[str, BranchProvider], names: list[str],
                       question: str, on_event: Callable[[dict], None],
                       signal: asyncio.Event | None = None,
                       result_cap: int = 0, range_gateway=None,
                       session_id: str | None = None) -> list[BranchOutcome]:
    """并行问答：名单逐支建 task 后 gather；单支异常（含未知名）降级为该支 failed。

    signal 置位（客户端断连）时取消未完成分支，尽力止损（外部进程有自身超时兜底）。
    result_cap > 0 时对成功支做"截断 + 全文归档 session range"（F5-017）。
    """
    async def _one(name: str) -> BranchOutcome:
        t0 = time.monotonic()
        provider = providers.get(name)
        if provider is None:
            out = BranchOutcome(name=name, kind="unknown", ok=False,
                                error=f"unknown agent: {name}")
        else:
            on_event({"type": "response.branch.started",
                      "agent": name, "kind": provider.kind})
            try:
                text = await provider.answer(question, signal=signal)
                out = BranchOutcome(name=name, kind=provider.kind, ok=True, text=text,
                                    elapsed=time.monotonic() - t0)
                # 先截断/归档再发事件：否则 SSE 会带未截断正文且缺 truncated/ref，
                # 前端与汇总拿到的口径不一致（F5-017 契约要求）。
                await _cap_and_archive(out, result_cap, range_gateway, session_id)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 —— 单支失败只影响该支（降级语义）
                out = BranchOutcome(name=name, kind=getattr(provider, "kind", "unknown"),
                                    ok=False, error=f"{type(e).__name__}: {e}",
                                    elapsed=time.monotonic() - t0)
        ev = {"type": "response.branch.done" if out.ok else "response.branch.failed",
              "agent": out.name, "kind": out.kind, "elapsed": round(out.elapsed, 2)}
        if out.ok:
            ev["chars"] = len(out.text)
            ev["truncated"] = out.truncated
            if out.ref:
                ev["ref"] = out.ref
            ev["text"] = out.text[:BRANCH_TEXT_CAP]
        else:
            ev["error"] = out.error
        on_event(ev)
        return out

    tasks = {name: asyncio.ensure_future(_one(name)) for name in names}
    try:
        return list(await asyncio.gather(*tasks.values()))
    finally:
        for t in tasks.values():
            if not t.done():
                t.cancel()


async def _cap_and_archive(outcome: BranchOutcome, cap: int, range_gateway,
                           session_id: str | None) -> None:
    """成功支正文超限 → 截断保留 head，全文落 session range（jsonl 事实源）。

    归档失败不得影响主流程（对齐 OPT-111「索引炸不丢档案」口径）：此时 ref 留空，
    截断照常——宁可少一个引用，也不把 2 万字灌进主上下文。
    """
    if not outcome.ok or len(outcome.text) <= cap:
        return
    full_text = outcome.text          # 先留全文：归档的是原文，不是截断后的 head
    outcome.text = full_text[:cap]
    outcome.truncated = True
    if range_gateway is None or not session_id:
        return
    recorder = range_gateway.recorder(session_id)
    try:
        rec = await asyncio.to_thread(
            recorder.archive, [Message(role="assistant", content=full_text)])
        outcome.ref = f"session/{session_id}#{rec.get('seq')}"
    except Exception as e:  # noqa: BLE001 —— 归档失败只丢 ref，不阻断汇总
        _log.warning("分支正文归档失败（agent=%s session=%s）：%s",
                     outcome.name, session_id, e)
        outcome.ref = ""


async def synthesize(provider, question: str,
                     outcomes: list[BranchOutcome]) -> str:
    """主 Agent 汇总：成功支全文入 prompt，失败支仅记名不记内容（不臆造）。"""
    from agentlab.prompts import load_prompt

    ok = [o for o in outcomes if o.ok]
    if not ok:
        raise RuntimeError("没有可汇总的成功分支")
    blocks = "\n\n".join(
        f"### 来源：{o.name}（{o.kind}）\n{o.text[:BRANCH_TEXT_CAP]}" for o in ok)
    failed = "、".join(o.name for o in outcomes if not o.ok) or "无"
    prompt = load_prompt("multi-synth-user", question=question,
                         branches=blocks, failed=failed)
    resp = await provider.chat([Message(role="user", content=prompt)])
    return (resp.content or "").strip()


async def _run_multi(cfg, registry: dict[str, BranchProvider], synth,
                     build_backend: Callable, hist: list[Message], user_input: str,
                     sink: _Sink, *, multi: list[str], store=None,
                     session_id: str | None = None, signal: asyncio.Event | None = None,
                     project_id: str | None = None, approvals=None,
                     run_id: str = "", range_gateway=None) -> dict:
    """multi 入口（serve.on_post 调用，返回与 _run_agent 同构的 summary dict）。

    路由语义：名单含 "agentlab" 且多于一支 → 本地 agent 也作为分支作答；
    只 @ agentlab → 等价单 agent（转 _run_agent）；外部支全败 → 降级本地直答。
    """
    from agentlab.runtime.serve import _run_agent
    from agentlab.runtime.serve_session import persist_delta

    names = [n for n in (multi or []) if n and n.strip()]
    run_internal = SELF_AGENT_NAME in names and len(names) > 1
    branch_names = [n for n in names if n != SELF_AGENT_NAME]

    t0 = time.monotonic()
    providers = dict(registry or {})
    if run_internal:
        providers[SELF_AGENT_NAME] = InternalBranchProvider(
            cfg, build_backend, project_id=project_id,
            approvals=approvals, run_id=run_id, approval_sink=sink)
        branch_names.append(SELF_AGENT_NAME)

    def _single() -> dict:
        # 无分支 / 外部全败：转单 agent 全能力直答（会话持久化语义与单轮一致）
        return _run_agent(cfg, build_backend, hist, user_input, sink, store=store,
                          session_id=session_id, signal=signal, project_id=project_id,
                          approvals=approvals, run_id=run_id)

    if not branch_names:
        return await _single()

    # F5-017：单轮支数上限——超出名单显式进 skipped（不静默丢弃），
    # 用"裁名单"而非"排队"限流，否则全部支都会跑完，失去成本上限意义。
    limit = _positive(getattr(getattr(cfg, "agents", None), "max_parallel_consults", 3), 3)
    skipped = branch_names[limit:]
    branch_names = branch_names[:limit]
    result_cap = _positive(
        getattr(getattr(cfg, "agents", None), "consult_result_max_chars", 6000), 6000)

    # F5-018（红线）：外部支 spawn 前先过门禁——调研稿 §4.3「@ stage 同样先确认后并行」。
    # 只对 external 支要确认（内置 agentlab 支走同一条工具权限体系，无需二次确认）；
    # 拒绝 → 不发起任何外部支，降级为单 agent 直答并在 summary 标 denied（决策 D4）。
    external_names = [n for n in branch_names
                      if getattr(providers.get(n), "kind", "") == "external"]
    if external_names and approvals is not None:
        allowed = await approvals.confirm(
            cfg, _GateTool(),
            f"将以多 Agent 并行方式咨询：{'、'.join(external_names)}"
            f"（共 {len(external_names)} 支）。问题：{user_input[:200]}",
            sink=sink, run_id=run_id, session_id=session_id, signal=signal)
        if not allowed:
            sink.event({"type": "response.multi.summary", "ok_count": 0,
                        "failed": [], "skipped": skipped, "denied": True,
                        "denied_agents": external_names})
            return await _single()

    outcomes = await run_branches(providers, branch_names, user_input,
                                  sink.event, signal,
                                  result_cap=result_cap,
                                  range_gateway=range_gateway,
                                  session_id=session_id)
    ok = [o for o in outcomes if o.ok]
    if not ok:
        if run_internal:
            raise RuntimeError("multi: 所有分支失败（含本地 agent）—— "
                               + "; ".join(f"{o.name}: {o.error}" for o in outcomes))
        sink.event({"type": "response.multi.summary", "ok_count": 0,
                    "failed": [o.name for o in outcomes],
                    "skipped": skipped, "degraded": True})
        return await _single()

    final, synthesized, synth_error = "", False, ""
    if synth is not None:
        try:
            final = await synthesize(synth, user_input, outcomes)
            synthesized = bool(final)
        except Exception as e:  # noqa: BLE001 —— 汇总失败降级为直取最优分支
            synth_error = f"{type(e).__name__}: {e}"
    if not synthesized:
        final = ok[0].text
    sink.event({"type": "response.multi.summary",
                "ok_count": len(ok),
                "failed": [o.name for o in outcomes if not o.ok],
                "skipped": skipped,
                "synthesized": synthesized,
                "truncated": [o.name for o in outcomes if o.truncated],
                **({"synth_error": synth_error} if synth_error else {})})
    sink.text(final)

    internal_tokens = sum(int(getattr(providers.get(o.name), "last_tokens", 0) or 0)
                          for o in outcomes if o.kind == "internal")
    answer_msgs = [Message(role="user", content=user_input),
                   Message(role="assistant", content=final)]
    if store is not None and session_id:
        # persist_delta 按 [1+len(hist):] 取增量 → 头部补占位 system 对齐单轮口径
        persist_delta(store, session_id, hist,
                      [Message(role="system", content=""), *hist, *answer_msgs])

    return {
        "final_output": final,
        "stop_reason": "multi-synth" if synthesized else "multi-fallback",
        "tokens": internal_tokens,
        "trace_id": "",
        "history": [{"role": m.role, "content": m.content}
                    for m in [*hist, *answer_msgs] if m.role != "system"],
        "multi": {
            "branches": [o.public() for o in outcomes],
            "ok": len(ok),
            "failed": [o.name for o in outcomes if not o.ok],
            "skipped": skipped,
            "synthesized": synthesized,
            "elapsed": round(time.monotonic() - t0, 2),
        },
    }


def _positive(value, default: int) -> int:
    """配置兜底：非法/缺失值回落默认，避免 0 或负数造成"零分支"或"无上限"。"""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return n if n >= 1 else default
