"""CLI / REPL（对齐 docs/03 §6、04 §2.5-2.6）。

- `agentlab run "<问题>"`：单命令跑一轮 ReAct。
- `agentlab repl`：交互循环。
- `agentlab tools list`：列出已注册工具（name + permission）。
- `agentlab trace show [id]`：trace 回放（P3）。
- 退出码：0 成功 / 2 参数错误 / 3 AGENT_* 运行时错误。

P2 安全：read 工具直接放行；write 工具默认 HITL 交互确认（`--yes` 跳过）；
danger 工具白名单（config.danger_allowlist）内放行、其余拒绝。
raw/ 写入由 brain_tools 连接器守卫（纵深防御）。
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

from agentlab.core.agent import Agent
from agentlab.core.llm import OpenAICompatProvider
from agentlab.core.events import MessageEnd, ToolEnd
from agentlab.core.loop import Ev, RunConfig, RunHooks, Runner
from agentlab.core.resilience import CircuitBreaker, ResilientLLM
from agentlab.runtime import config as _cfg
from agentlab.runtime.trace import Tracer
from agentlab.tools.demo import DEMO_TOOLS
from agentlab.tools.registry import ToolRegistry

_BANNER = "agentlab · 轻量 Agent 框架 · 输入问题（exit 退出）"


def _build_registry(cfg, with_brain: bool, rag_llm=None,
                    index=None, p2_store=None, range_gateway=None) -> tuple[ToolRegistry, int]:
    from agentlab.tools.web_search import WEB_SEARCH_TOOLS

    brain_tools = []
    if with_brain:
        from agentlab.tools.connectors.brain_tools import build_brain_tools
        brain_tools = build_brain_tools(cfg.vault_root, cfg.model_dump())
    brain_names = {t.name for t in brain_tools}

    reg = ToolRegistry()
    # brain 提供同名的演示占位工具（如 memory_query）让位给生产版本
    for t in DEMO_TOOLS:
        if t.name in brain_names:
            continue
        reg.register(t)
    for t in WEB_SEARCH_TOOLS:
        if t.name not in brain_names:
            reg.register(t)
    for t in brain_tools:
        reg.register(t)
    # P4 Agentic RAG：rag_retrieve / rag_assess（read）
    from agentlab.tools.connectors.brain_tools import load_brain_config
    brain_config = load_brain_config(cfg.model_dump())
    if brain_config is not None:
        brain_config["memory_policy"] = cfg.memory.model_dump()
    from agentlab.tools.rag_tools import build_rag_tools
    for t in build_rag_tools(brain_config, rag_llm,
                             item_chars=cfg.limits.recall_item_chars,
                             rag_config=cfg.rag, vault_root=cfg.vault_root,
                             index=index, p2_store=p2_store, range_gateway=range_gateway):
        if t.name not in brain_names:
            reg.register(t)
    # P2-1/OPT-112：外部 ACP agent（agents.external 配置了才注册，未配置零影响）
    from agentlab.tools.acp_tools import build_acp_tools
    for t in build_acp_tools(getattr(cfg, "agents", None), vault_root=cfg.vault_root):
        if t.name not in brain_names:
            reg.register(t)
    # #10①/OPT-135：agent 自省——"今天/最近做了什么"走 run 日志（episodic），
    # 与语义记忆（memory_query）互补
    from agentlab.tools.runs_tools import build_runs_tools
    for t in build_runs_tools(cfg.trace_dir):
        if t.name not in brain_names:
            reg.register(t)
    return reg, len(brain_tools)


def _system_instruction(cfg, tools_list, memory: str | None = "") -> str:
    from agentlab.memory.recall import RECALL_FALLBACK
    from agentlab.memory.skill import load_skills, resolve_skills_dir
    from agentlab.prompts import load_prompt

    tools_schema = "\n".join(f"- {t.name}: {t.description}" for t in tools_list)
    skills = ""
    if cfg.skills_dir:
        # 渐进披露：仅注入命中技能说明；相对路径 cwd 落空时回退仓库根（OPT-114 修复静默失效）
        skills = load_skills(str(resolve_skills_dir(cfg.skills_dir)))
    # #10①/OPT-123：{{memory}} 槽——loader 对未提供变量保留原文，空召回也必须给占位文案
    return load_prompt("system-user", tools=tools_schema, skills=skills,
                       memory=memory or RECALL_FALLBACK)


def _build_agent(cfg, instruction: str, with_brain: bool = True) -> tuple[Agent, ToolRegistry]:
    reg, n_brain = _build_registry(cfg, with_brain)
    agent = Agent(
        name="agentlab-demo",
        instructions=instruction,
        tools=reg.all(),
        max_steps=cfg.limits.max_steps,
    )
    return agent, reg


def _make_resilient(cfg) -> "LLMProvider":
    """构造 LLM provider：
    - 未配置 routes 且多模型路由 → 单个 resilient provider；
    - 配置了 routes（light/heavy）→ RouteLLM 按 tier 分发（F17），无该 tier 回退默认。
    """
    key = cfg.llm.effective_key()
    if not key:
        raise SystemExit(
            "[AGENT_LLM_AUTH] 缺少 API Key：配置 config.json 的 llm.api_key 或设环境变量 "
            "AGENT_LLM_API_KEY"
        )
    r = cfg.resilience

    def _resilient(model: str) -> "ResilientLLM":
        provider = OpenAICompatProvider(
            base_url=cfg.llm.base_url,
            api_key=key,
            model=model,
            timeout=cfg.llm.timeout,
            max_tokens=cfg.llm.max_tokens,
        )
        return ResilientLLM(
            provider,
            breaker=CircuitBreaker(
                failure_threshold=r.breaker_threshold,
                recovery_timeout=r.breaker_recovery,
            ),
            max_retries=r.max_retries,
            backoff=r.backoff,
        )

    routes = cfg.routes or {}
    if len(routes) <= 1:
        return _resilient(cfg.llm.model)

    from agentlab.core.router import RouteLLM, classify_light_route
    providers = {tier: _resilient(model) for tier, model in routes.items()}
    default = "heavy" if "heavy" in providers else next(iter(providers))
    # 快速决策前置（open-note）：闲聊/问候短消息走 light tier 直答，省一次工具编排
    return RouteLLM(providers, default_tier=default, classify=classify_light_route)


def _resilient_for_model(cfg, model: str, max_tokens: int | None = None) -> "ResilientLLM":
    """构造**固定模型**的 resilient provider（不走 routes 路由）。

    为什么需要它（OPT-197 实测教训）：`_make_resilient` 会按 prompt 特征路由 tier，
    落到 `heavy` 时用的是推理模型（如 deepseek-reasoner）——推理 token 吃满 max_tokens 后
    `message.content` 返回空。评测评审这类**要求跨运行可比**的场景必须把模型钉死，
    否则同一个"基线"在不同 prompt 长度下会用不同模型评分。
    """
    key = cfg.llm.effective_key()
    if not key:
        raise SystemExit(
            "[AGENT_LLM_AUTH] 缺少 API Key：配置 config.json 的 llm.api_key 或设环境变量 "
            "AGENT_LLM_API_KEY"
        )
    r = cfg.resilience
    provider = OpenAICompatProvider(
        base_url=cfg.llm.base_url,
        api_key=key,
        model=model,
        timeout=cfg.llm.timeout,
        max_tokens=max_tokens or cfg.llm.max_tokens,
    )
    return ResilientLLM(
        provider,
        breaker=CircuitBreaker(
            failure_threshold=r.breaker_threshold,
            recovery_timeout=r.breaker_recovery,
        ),
        max_retries=r.max_retries,
        backoff=r.backoff,
    )


def _build_runner(cfg, registry: ToolRegistry | None = None, provider=None) -> Runner:
    # A serve instance passes its long-lived provider here.  CLI callers keep
    # the previous construction behavior while still benefiting from the
    # provider's lazy connection pool for the duration of a run.
    resilient = provider or _make_resilient(cfg)
    # P3 预算管理：超预算时用 resilient LLM 作摘要器走结构化压缩，而非直接 guardrail
    from agentlab.memory.working import WorkingMemory
    wm = WorkingMemory(
        budget=cfg.limits.context_budget,
        keep_recent_tokens=max(1, int(cfg.limits.context_budget * 0.5)),
        summarizer=resilient,
    )
    return Runner(resilient, memory=wm, registry=registry)


def build_hooks(cfg, *, yes: bool = False, trace: Tracer | None = None) -> RunHooks:
    allowlist = set(cfg.danger_allowlist or [])
    interactive = sys.stdin.isatty() and not yes

    def confirm(t, prompt: str) -> bool:
        # danger 白名单放行；默认拒绝
        if t.permission == "danger":
            return t.name in allowlist
        # write：--yes 放行；否则交互确认；非 TTY 自动拒绝
        if yes:
            return True
        if not interactive:
            return False
        try:
            ans = input(f"{prompt} [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return ans in ("y", "yes")

    def on_tool(name, phase, payload):
        if phase == "progress":
            # 长任务心跳：非交互时不刷屏，只让守望者看到工具仍存活
            if interactive:
                print(f"\r  ▶ {name} 运行中… 已 {payload.get('elapsed', 0):.0f}s     ", end="", flush=True)
            return
        if phase == "end" and interactive:
            print()  # 结束心跳覆盖行
        if phase == "start" and trace is not None:
            trace.record_tool(name, "start")
        elif trace is not None:
            result = payload.get("result", "")
            trace.record_tool(name, "end", result=result[:1000],
                              terminate=payload.get("terminate"))

    return RunHooks(on_tool=on_tool, confirm=confirm)


def _attach_trace(runner: Runner, cfg) -> Tracer | None:
    """订阅 Runner 事件写入 JSONL；返回 Tracer（未启用时 None）。"""
    try:
        tracer = Tracer(cfg.trace_dir)
        tracer.new_session()
    except Exception:
        return None

    def on_message_end(p: MessageEnd) -> None:
        resp = SimpleNamespace(
            content=p.content,
            tool_calls=[],
            stop_reason="event",
            usage=p.usage,
        )
        tracer.record_step(p.step, resp)

    def on_tool_end(p: ToolEnd) -> None:
        tracer.record_tool(p.name, "end", result=str(p.result or "")[:1000])

    runner.on(Ev.MESSAGE_END, on_message_end)
    runner.on(Ev.TOOL_END, on_tool_end)
    return tracer


def _attach_depository(cfg, run_cfg: "RunConfig") -> "RunConfig":
    """A1/OPT-229：CLI 与 serve 的记忆沉淀接线对齐（复用 serve 实现，含降级语义）。"""
    try:
        from agentlab.runtime.serve import _attach_depository as _serve_attach
        return _serve_attach(cfg, run_cfg, source_session="cli")
    except Exception:
        return run_cfg


async def _run_once(cfg, question: str, agent, reg, *, yes: bool = False) -> None:
    runner = _build_runner(cfg, registry=reg)
    tracer = _attach_trace(runner, cfg)
    hooks = build_hooks(cfg, yes=yes, trace=tracer)
    agent.tools = reg.all()
    run_cfg = RunConfig.from_config(cfg)
    run_cfg = _attach_depository(cfg, run_cfg)
    result = await runner.run(agent, question, hooks=hooks, cfg=run_cfg)
    if tracer is not None and result:
        result.trace_id = getattr(tracer, "trace_id", "")
    if result.final_output:
        print("\n[ANSWER]")
        print(result.final_output)
    print(f"[DONE] stop_reason={result.stop_reason} tokens={result.usage.total()}")


def cmd_run(args) -> int:
    cfg = _cfg.load_config(path=args.config)
    rag_llm = None
    try:
        rag_llm = _make_resilient(cfg)  # 供 rag_assess 使用
    except SystemExit:
        pass
    reg, n_brain = _build_registry(cfg, with_brain=True, rag_llm=rag_llm)
    question = " ".join(args.question)
    # #10①/OPT-123：单命令路也自动召回长期记忆（topic=问题本身）
    from agentlab.memory.recall import memory_block_for

    agent = Agent(
        name="agentlab-demo",
        instructions=_system_instruction(
            cfg, reg.all(),
            memory=memory_block_for(cfg, question,
                                    topk=int(getattr(cfg.limits, "memory_inject_topk", 5) or 0))),
        tools=reg.all(),
        max_steps=cfg.limits.max_steps,
    )
    if cfg.llm.effective_key():
        print(f"[AGENT] 就绪 · tools={len(reg.all())}（brain={n_brain}）")
    else:
        print("注意：未配置 API Key，无法调用 LLM")
    asyncio.run(_run_once(cfg, question, agent, reg, yes=args.yes))
    return 0


def _refresh_repl_instructions(agent, cfg, reg, question: str) -> None:
    """A1/OPT-229：repl 每轮按当前输入重建 system instructions（含记忆注入块）。"""
    from agentlab.memory.recall import memory_block_for

    agent.instructions = _system_instruction(
        cfg, reg.all(),
        memory=memory_block_for(
            cfg, question,
            topk=int(getattr(cfg.limits, "memory_inject_topk", 5) or 0)))


def cmd_repl(args) -> int:
    cfg = _cfg.load_config(path=args.config)
    rag_llm = None
    try:
        rag_llm = _make_resilient(cfg)
    except SystemExit:
        pass
    reg, n_brain = _build_registry(cfg, with_brain=True, rag_llm=rag_llm)
    agent = Agent(
        name="agentlab-demo",
        instructions=_system_instruction(cfg, reg.all()),
        tools=reg.all(),
        max_steps=cfg.limits.max_steps,
    )
    runner = _build_runner(cfg, registry=reg)
    tracer = _attach_trace(runner, cfg)
    hooks = build_hooks(cfg, yes=args.yes, trace=tracer)

    def run_once(question: str):
        # A1/OPT-229：每轮按当前输入重建记忆注入（此前只在启动时组装一次）
        _refresh_repl_instructions(agent, cfg, reg, question)

        result = None

        async def _run():
            run_cfg = RunConfig.from_config(cfg)
            run_cfg = _attach_depository(cfg, run_cfg)
            return await runner.run(agent, question, hooks=hooks, cfg=run_cfg)
        return asyncio.run(_run())

    print(f"{_BANNER} · tools={len(reg.all())}（brain={n_brain}）")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            return 0
        result = run_once(line)
        if result.final_output:
            print(result.final_output)


def cmd_tools(args) -> int:
    cfg = _cfg.load_config(path=args.config)
    reg, n_brain = _build_registry(cfg, with_brain=True)
    print(f"# tools（brain 接入 {n_brain}）：")
    for t in reg.all():
        flags = []
        if t.permission != "read":
            flags.append(t.permission)
        if t.execution_mode == "sequential":
            flags.append("seq")
        tag = f"  [{','.join(flags)}]" if flags else ""
        print(f"{t.name}\t{t.description}{tag}")
    return 0


def cmd_trace(args) -> int:
    print("trace show：P3 实现（回放 JSONL）。")
    return 0


def cmd_serve(args) -> int:
    from agentlab.runtime.serve import Serve

    cfg = _cfg.load_config(path=args.config)
    Serve(cfg, port=args.port, host=args.host).serve_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="agentlab", description="自研轻量 Python Agent 框架")
    p.add_argument("-c", "--config", default=None, help="config.json 路径（默认探测 config/config.json）")
    sub = p.add_subparsers(dest="cmd")

    prun = sub.add_parser("run", help="单命令运行")
    prun.add_argument("question", nargs="+")
    prun.add_argument("--yes", action="store_true", help="跳过 write 工具确认")
    prun.set_defaults(fn=cmd_run)

    prepl = sub.add_parser("repl", help="交互 REPL")
    prepl.add_argument("--yes", action="store_true", help="跳过 write 工具确认")
    prepl.set_defaults(fn=cmd_repl)

    ptools = sub.add_parser("tools", help="工具管理")
    ptools.add_argument("action", choices=["list"])
    ptools.set_defaults(fn=cmd_tools)

    ptrace = sub.add_parser("trace", help="trace 回放")
    ptrace.add_argument("action", choices=["show"])
    ptrace.add_argument("id", nargs="?", default=None)
    ptrace.set_defaults(fn=cmd_trace)

    serve = sub.add_parser("serve", help="启动 HTTP/SSE 服务（供 Obsidian ark 对话）")
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--host", default=None)
    serve.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    if not getattr(args, "cmd", None):
        p.print_help()
        return 2
    try:
        rc = args.fn(args)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"[AGENT_ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return 3
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
