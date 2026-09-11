import asyncio
import json
import time
import unittest

from agentlab.core.agent import Agent
from agentlab.core.llm import LLMProvider, LLMResponse
from agentlab.core.loop import RunConfig, Runner
from agentlab.core.message import Message, TokenUsage, ToolCall, ToolCallFunction, tool_result
from agentlab.memory.working import WorkingMemory
from agentlab.tools.base import tool
from agentlab.tools.registry import ToolRegistry


def _tc(name, args, cid="c1"):
    return ToolCall(id=cid, function=ToolCallFunction(name=name, arguments=json.dumps(args)))


class FakeProvider(LLMProvider):
    """脚本化响应：元素为 ToolCall（触发工具）或 str（触发收尾）。"""

    def __init__(self, script, length_on_calls=False):
        self.script = list(script)
        self.length_on_calls = length_on_calls

    async def chat(self, messages, tools=None, **kw):
        item = self.script.pop(0)
        if isinstance(item, ToolCall):
            stop = "length" if self.length_on_calls else "tool_calls"
            return LLMResponse(
                content=None, tool_calls=[item],
                usage=TokenUsage(input_tokens=10, output_tokens=5), stop_reason=stop,
            )
        return LLMResponse(
            content=item, tool_calls=[],
            usage=TokenUsage(input_tokens=10, output_tokens=5), stop_reason="stop",
        )


def _build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    calls: list[str] = []

    @tool(description="加")
    def myadd(a: int, b: int) -> int:
        calls.append("myadd")
        return a + b

    @tool(description="串行样本", execution_mode="sequential")
    def srch(q: str) -> str:
        calls.append("srch")
        return f"搜:{q}"

    @tool(description="慢并行样本")
    async def slow(tag: str, delay: float) -> str:
        """异步慢工具：用 perf_counter 记录墙钟，供并发窗测量。"""
        import asyncio as _a
        t0 = __import__("time").perf_counter()
        await _a.sleep(delay)
        return f"slow:{tag}:{__import__('time').perf_counter() - t0:.3f}"

    @tool(description="做完就停", can_terminate=True)
    def done_marker(x: str):
        return tool_result("0", f"完成:{x}", terminate=True)

    reg.register(myadd)
    reg.register(srch)
    reg.register(slow)
    reg.register(done_marker)
    return reg


class _Depo:
    """假记忆仓：记录 commit 内容，供断言异步沉淀调度（S6）。"""

    def __init__(self):
        self.commits: list[str] = []

    def commit(self, content, tags=None, source_session="", dedup=False):
        self.commits.append(content)
        return {"status": "committed"}


class TestLoop(unittest.TestCase):
    def setUp(self):
        self.reg = _build_registry()

    async def _run(self, provider, agent=None, cfg=None):
        runner = Runner(provider, self.reg)
        return await runner.run(agent or self._agent(), "问题", cfg=cfg or RunConfig())

    def _agent(self, max_steps=15):
        return Agent(instructions="sys", tools=self.reg.all(), max_steps=max_steps)

    def test_done(self):
        res = asyncio.run(self._run(FakeProvider(["最终答案"])))
        self.assertEqual(res.stop_reason, "done")
        self.assertEqual(res.final_output, "最终答案")

    def test_tool_call_budget_caps_runaway_retrieval(self):
        """工具调用**次数**必须有上界：max_steps 只管批数，一次响应可塞很多调用。

        实测来源：全量基线 q-memory-recall 15 步内跑了 46 次工具 / 373k token，
        步数上限从未触发；live 会话"整理收件箱"45 次调用 / 596k token。
        """
        # 参数必须每次不同：否则会被"批次级重复检测"先拦下，测不到预算这条路径
        script = [_tc("myadd", {"a": i, "b": 2}) for i in range(8)] + ["答案"]
        res = asyncio.run(self._run(FakeProvider(script),
                                    cfg=RunConfig(max_steps=15, max_tool_calls=4)))
        self.assertEqual(res.stop_reason, "guardrail", "超预算应以 guardrail 收尾")
        self.assertIn("工具调用上限", res.final_output)
        # 硬上限 4：执行过的不超过 4 次（最后那批被拦下，用占位结果补齐历史）
        executed = [m for m in res.messages if m.role == "tool"
                    and not str(m.content).startswith("[repeat_guard]")]
        self.assertLessEqual(len(executed), 4, [m.content for m in executed])

    def test_tool_call_budget_nudges_once_before_hard_stop(self):
        """软阈值先给一次收敛提示——直接砍断会让模型没有机会给结论。"""
        script = [_tc("myadd", {"a": i, "b": 2}) for i in range(8)] + ["答案"]
        res = asyncio.run(self._run(FakeProvider(script),
                                    cfg=RunConfig(max_steps=15, max_tool_calls=4,
                                                  tool_call_nudge=0.5)))
        nudge = [m for m in res.messages if m.role == "user" and "预算提醒" in str(m.content)]
        self.assertEqual(len(nudge), 1, "收敛提示每次 run 至多一条")

    def test_budget_zero_means_unlimited(self):
        script = [_tc("myadd", {"a": i, "b": 2}) for i in range(6)] + ["答案"]
        res = asyncio.run(self._run(FakeProvider(script),
                                    cfg=RunConfig(max_steps=15, max_tool_calls=0)))
        self.assertEqual(res.stop_reason, "done", "0=不限，回到原行为")

    def test_memory_deposit_periodic(self):
        # S6：工具轮(cur1)不沉淀，第 2 轮命中触发词的片段异步沉淀进记忆仓
        depo = _Depo()
        provider = FakeProvider([_tc("myadd", {"a": 1, "b": 2}), "规律：A"])
        res = asyncio.run(self._run(provider, cfg=RunConfig(memorize_every=2, depository=depo)))
        self.assertEqual(res.stop_reason, "done")
        self.assertEqual(depo.commits, ["规律：A"])  # 仅在第 2 轮周期上沉淀

    def test_memory_deposit_respects_period_then_flushes_at_end(self):
        # S6：memorize_every=3 → 第 2 轮不在周期上，中途不沉淀；
        # OPT-135：run 结束兜底 flush 捕获剩余片段（短会话也留痕）
        depo = _Depo()
        provider = FakeProvider([_tc("myadd", {"a": 1, "b": 2}), "规律：A"])
        asyncio.run(self._run(provider, cfg=RunConfig(memorize_every=3, depository=depo)))
        self.assertEqual(depo.commits, ["规律：A"])

    def test_memory_flush_at_end_skips_chitchat(self):
        # OPT-135：结束 flush 仍走触发词/importance 过滤——闲聊不入库
        depo = _Depo()
        asyncio.run(self._run(FakeProvider(["普通回答"]),
                              cfg=RunConfig(memorize_every=5, depository=depo)))
        self.assertEqual(depo.commits, [])

    def test_memory_deposit_disabled_by_default(self):
        # S6：memorize_every=0 → 即使有触发词也不沉淀
        depo = _Depo()
        asyncio.run(self._run(FakeProvider(["关键：A"]), cfg=RunConfig(depository=depo)))
        self.assertEqual(depo.commits, [])

    def test_signal_preset_aborts_before_any_call(self):
        # 取消信号启动前已置位 → 第一轮就中止，不调用任何 LLM/工具
        sig = asyncio.Event()
        sig.set()
        provider = FakeProvider(["不应被消费"])
        res = asyncio.run(self._run(provider, cfg=RunConfig(signal=sig)))
        self.assertEqual(res.stop_reason, "aborted")
        self.assertEqual(provider.script, ["不应被消费"])  # script 未被消费 → 无多余调用

    def test_signal_aborts_after_tool_turn(self):
        # 工具执行期间触发取消 → 下一轮开头中止，不再多跑（SSE 断连场景）
        sig = asyncio.Event()
        calls: list[str] = []

        @tool(description="执行时置位取消")
        def set_abort(x: int) -> int:
            calls.append("set_abort")
            sig.set()  # 模拟长工具期间客户端断连
            return x + 1

        reg = ToolRegistry()
        reg.register(set_abort)
        runner = Runner(FakeProvider([_tc("set_abort", {"x": 1}), "不应到达"]), reg)
        res = asyncio.run(runner.run(Agent(instructions="sys", tools=reg.all()), "q",
                                     cfg=RunConfig(signal=sig)))
        self.assertEqual(res.stop_reason, "aborted")
        self.assertEqual(calls, ["set_abort"])  # 工具只跑了一次，未继续消费后续脚本

    def test_tool_then_done(self):
        provider = FakeProvider([_tc("myadd", {"a": 1, "b": 2}), "结果是 3"])
        res = asyncio.run(self._run(provider))
        self.assertEqual(res.stop_reason, "done")
        self.assertEqual(res.final_output, "结果是 3")

    def test_max_steps_stops(self):
        prov = FakeProvider([_tc("myadd", {"a": 1, "b": 2})] * 50)
        agent = self._agent(max_steps=3)
        res = asyncio.run(self._run(prov, agent=agent, cfg=RunConfig(max_steps=3)))
        self.assertEqual(res.stop_reason, "max_steps")

    def test_repetition_guard_batch_level_no_false_positive(self):
        """OPT-117：多工具批（collect+readq）后单独复读 readq 是合法状态复查，
        不得按扁平最近两次比对误判打转（2026-09-07 收件箱实测误伤）。"""

        class BatchProvider(LLMProvider):
            def __init__(self):
                self.n = 0

            async def chat(self, messages, tools=None, **kw):
                self.n += 1
                if self.n == 1:
                    return LLMResponse(content=None, tool_calls=[
                        _tc("collect", {}, "c1"), _tc("readq", {}, "c2")],
                        usage=TokenUsage(input_tokens=10, output_tokens=5),
                        stop_reason="tool_calls")
                if self.n == 2:
                    return LLMResponse(content=None, tool_calls=[_tc("readq", {}, "c3")],
                        usage=TokenUsage(input_tokens=10, output_tokens=5),
                        stop_reason="tool_calls")
                return LLMResponse(content="完成", tool_calls=[],
                    usage=TokenUsage(input_tokens=1, output_tokens=1), stop_reason="stop")

        calls: list[str] = []
        reg = ToolRegistry()

        @tool(description="采集")
        def collect() -> str:
            calls.append("collect")
            return "collected"

        @tool(description="读队列")
        def readq() -> str:
            calls.append("readq")
            return "queue"

        reg.register(collect)
        reg.register(readq)
        runner = Runner(BatchProvider(), reg)
        res = asyncio.run(runner.run(Agent(instructions="sys", tools=reg.all(), max_steps=10), "问题"))
        self.assertEqual(res.stop_reason, "done")
        self.assertEqual(calls, ["collect", "readq", "readq"], "合法复读应照常执行")

    def test_repetition_guard_same_batch_twice_stops_with_note(self):
        """OPT-117：同一批连续两轮完全相同 → 拦截；占位补全悬挂 tool_calls，说明入史入终。"""

        class LoopProvider(LLMProvider):
            def __init__(self):
                self.n = 0

            async def chat(self, messages, tools=None, **kw):
                self.n += 1
                if self.n <= 2:
                    return LLMResponse(content=None, tool_calls=[_tc("readq", {}, f"c{self.n}")],
                        usage=TokenUsage(input_tokens=10, output_tokens=5),
                        stop_reason="tool_calls")
                return LLMResponse(content="不应到达", tool_calls=[],
                    usage=TokenUsage(input_tokens=1, output_tokens=1), stop_reason="stop")

        calls: list[str] = []
        reg = ToolRegistry()

        @tool(description="读队列")
        def readq() -> str:
            calls.append("readq")
            return "queue"

        reg.register(readq)
        runner = Runner(LoopProvider(), reg)
        res = asyncio.run(runner.run(Agent(instructions="sys", tools=reg.all(), max_steps=10), "问题"))
        self.assertEqual(res.stop_reason, "max_steps")
        self.assertEqual(calls, ["readq"], "第二批在执行前被拦，只执行一次")
        self.assertIn("重复调用", res.final_output, "用户可见的中止说明")
        tool_msgs = [m for m in res.messages if m.role == "tool"]
        self.assertTrue(any("repeat_guard" in (m.content or "") for m in tool_msgs),
                        "悬挂 tool_calls 补占位，历史保持合法")
        self.assertTrue(any(m.role == "assistant" and "重复调用" in (m.content or "")
                            for m in res.messages), "中止说明同时入对话史")

    def test_length_truncation_fails_batch(self):
        never: list[str] = []

        @tool(description="不应被执行")
        def exploded(x: str):
            never.append(x)
            return "executed"

        self.reg.register(exploded)
        tc = _tc("exploded", "{\"x\": \"残")  # 截断导致的残缺参数
        prov = FakeProvider([tc, "已重发成功"], length_on_calls=True)
        res = asyncio.run(self._run(prov))
        self.assertEqual(res.stop_reason, "done")
        self.assertEqual(never, [], "截断批次不得执行残缺参数")
        self.assertTrue(any(m.role == "tool" and "截断" in (m.content or "") for m in res.messages))

    def test_terminate(self):
        tc = _tc("done_marker", {"x": "收尾"})
        res = asyncio.run(self._run(FakeProvider([tc])))
        self.assertEqual(res.stop_reason, "terminate")

    # ── B1：空/纯思考回合 → 注入轻 steer 续跑（学 tau），不再静默判 done ──
    def test_empty_thought_gets_steer_then_done(self):
        prov = FakeProvider([_tc("myadd", {"a": 1, "b": 2}), "", "结果是 3"])
        res = asyncio.run(self._run(prov))
        self.assertEqual(res.stop_reason, "done")
        self.assertEqual(res.final_output, "结果是 3")
        steers = [m for m in res.messages
                  if m.role == "user" and "AGENT_INCOMPLETE" in (m.content or "")]
        self.assertEqual(len(steers), 1, "只分析未动作的回合应被拉回续跑")
        # 空 assistant（无内容无工具）不应残留在最终消息里（tau：空失败不回放）
        empty = [m for m in res.messages
                 if m.role == "assistant" and not (m.content or "").strip() and not m.tool_calls]
        self.assertEqual(empty, [])

    def test_empty_thought_capped_by_guardrail(self):
        # 一直空转 → 最多 inject _MAX_EMPTY_STEERS 次 steer，随后空输出过不了 guardrail 收尾
        prov = FakeProvider([""] * 6)
        res = asyncio.run(self._run(prov))
        steers = [m for m in res.messages
                  if m.role == "user" and "AGENT_INCOMPLETE" in (m.content or "")]
        self.assertEqual(len(steers), 3)
        self.assertIn(res.stop_reason, ("guardrail", "done"))  # 有用护栏兜底，非死循环

    # ── B2：截断(length/max_tokens)且无工具 → 保留半截 + [续写指令]（防重新生成全文造成回答重复，OPT-110 观测） ──
    def test_length_no_tools_continues_from_truncation(self):
        class _LenNoTool(LLMProvider):
            def __init__(self, script):
                self.script = list(script)

            async def chat(self, messages, tools=None, **kw):
                item = self.script.pop(0)
                if item == "__LEN__":
                    return LLMResponse(content="", tool_calls=[], usage=TokenUsage(input_tokens=10, output_tokens=5),
                                       stop_reason="length")
                return LLMResponse(content=item, tool_calls=[], usage=TokenUsage(input_tokens=10, output_tokens=5),
                                   stop_reason="stop")

        res = asyncio.run(self._run(_LenNoTool(["__LEN__", "最终结论"])))
        self.assertEqual(res.stop_reason, "done")
        self.assertEqual(res.final_output, "最终结论")
        steers = [m for m in res.messages
                  if m.role == "user" and (m.content or "").startswith("[续写指令]")]
        self.assertEqual(len(steers), 1, "截断应注入一次续写指令")
        # 半截 assistant 保留在历史（模型可从截断处衔接，而非重新生成全文）
        self.assertTrue(any(m.role == "assistant" and (m.content or "") == "" for m in res.messages))

    def test_sequential_batch_executes_serially(self):
        provider = FakeProvider([_tc("srch", {"q": "a"}, "c1"), _tc("srch", {"q": "b"}, "c2"), "完成"])
        res = asyncio.run(self._run(provider))
        self.assertEqual(res.stop_reason, "done")

    def test_parallel_batch_single_turn(self):
        # 两个并行工具应一轮内完成，最终 done 且结果回填
        provider = FakeProvider([_tc("myadd", {"a": 1, "b": 2}, "c1"), _tc("myadd", {"a": 3, "b": 4}, "c2"), "5 and 7"])
        res = asyncio.run(self._run(provider))
        self.assertEqual(res.stop_reason, "done")
        tools = [m for m in res.messages if m.role == "tool"]
        self.assertEqual(len(tools), 2)

    def test_tool_granular_scheduling_mixed_batch(self):
        # S6 工具粒度串并行：一个 batch 里 [slow(a), slow(b), srch(串行)]
        # → 两个 parallel 工具应并发（墙钟≈0.3s 而非串行≈0.6s），结果序保真。
        class BatchProvider(LLMProvider):
            def __init__(self):
                self.rounds = 0

            async def chat(self, messages, tools=None, **kw):
                if self.rounds == 0:
                    self.rounds = 1
                    calls = [
                        ToolCall(id="a", function=ToolCallFunction(
                            name="slow", arguments=json.dumps({"tag": "a", "delay": 0.3}))),
                        ToolCall(id="b", function=ToolCallFunction(
                            name="slow", arguments=json.dumps({"tag": "b", "delay": 0.3}))),
                        ToolCall(id="c", function=ToolCallFunction(
                            name="srch", arguments=json.dumps({"q": "s"}))),
                    ]
                    return LLMResponse(content=None, tool_calls=calls,
                                       usage=TokenUsage(input_tokens=2, output_tokens=2),
                                       stop_reason="tool_calls")
                return LLMResponse(content="完成", tool_calls=[],
                                   usage=TokenUsage(input_tokens=2, output_tokens=2),
                                   stop_reason="stop")

        t0 = time.perf_counter()
        res = asyncio.run(self._run(BatchProvider()))
        dt = time.perf_counter() - t0
        self.assertEqual(res.stop_reason, "done")
        # 并发：slow(a) 与 slow(b) 重叠 → 墙钟远小于 0.3+0.3；串行(旧行为)需 ≈0.6
        self.assertLess(dt, 0.5, f"parallel 工具未并发？墙钟={dt:.2f}s")

    def test_steering_seeded_before_run(self):
        agent = self._agent()
        agent.steer(Message(role="user", content="中途插话"))
        prov = FakeProvider([_tc("myadd", {"a": 1, "b": 2}), "答案"])
        res = asyncio.run(self._run(prov, agent=agent))
        self.assertTrue(any(m.role == "user" and m.content == "中途插话" for m in res.messages))

    def test_budget_overflow_triggers_condense_not_guardrail(self):
        """超预算时先经 WorkingMemory 压缩，而非直接 guardrail（P3 预算管理）。"""

        class ShortProvider(LLMProvider):
            async def chat(self, messages, tools=None, **kw):
                return LLMResponse(
                    content="[检查点] Goal 已完成，可继续",
                    tool_calls=[],
                    usage=TokenUsage(input_tokens=4, output_tokens=4), stop_reason="stop",
                )

        wm = WorkingMemory(keep_recent_tokens=0, summarizer=ShortProvider())
        # 参数递增避免触发"重复调用防打转"
        script = [_tc("myadd", {"a": i, "b": i + 1}) for i in range(8)] + ["最终回答"]
        runner = Runner(FakeProvider(script), self.reg, memory=wm)
        cfg = RunConfig(context_budget=40, max_steps=15)
        res = asyncio.run(runner.run(self._agent(max_steps=15), "问题", cfg=cfg))
        # 压缩介入后应能跑完成，而非被 guardrail 终止
        self.assertEqual(res.stop_reason, "done")
        self.assertEqual(res.final_output, "最终回答")
        # 历史里应存在压缩产生的检查点摘要
        self.assertTrue(any(
            m.role == "system" and "[工作记忆检查点]" in (m.content or "") for m in res.messages
        ))

    def test_compact_anchor_advances_and_prefix_frozen(self):
        """L9/OPT-104：压缩锚点跨压缩单调前移；上次压缩产物整体成为下次调用的
        冻结前缀（逐对象一致 = 字节级稳定）→ provider 前缀缓存跨压缩命中。"""

        class RecordingMemory(WorkingMemory):
            def __init__(self, summarizer):
                super().__init__(keep_recent_tokens=0, summarizer=summarizer)
                self.calls = []    # (anchor, len(messages))
                self.inputs = []   # 每次调用的输入快照
                self.outputs = []

            async def condense(self, messages, keep_recent_tokens=None, anchor=0, **kw):
                self.calls.append((anchor, len(messages)))
                self.inputs.append(list(messages))
                out = await super().condense(
                    messages, keep_recent_tokens=keep_recent_tokens, anchor=anchor)
                self.outputs.append(out)
                return out

        class ShortProvider(LLMProvider):
            async def chat(self, messages, tools=None, **kw):
                return LLMResponse(
                    content="[检查点] Goal 推进中，继续",
                    tool_calls=[],
                    usage=TokenUsage(input_tokens=4, output_tokens=4), stop_reason="stop",
                )

        wm = RecordingMemory(ShortProvider())
        script = [_tc("myadd", {"a": i, "b": i + 1}) for i in range(8)] + ["最终回答"]
        runner = Runner(FakeProvider(script), self.reg, memory=wm)
        cfg = RunConfig(context_budget=40, max_steps=15)
        res = asyncio.run(runner.run(self._agent(max_steps=15), "问题", cfg=cfg))
        self.assertEqual(res.stop_reason, "done")
        # 至少两次压缩；锚点单调且有推进
        self.assertGreaterEqual(len(wm.calls), 2)
        anchors = [c[0] for c in wm.calls]
        self.assertEqual(anchors, sorted(anchors))
        self.assertLess(anchors[0], anchors[-1])
        # 冻结性质：第 2 次调用的输入（快照）与第 1 次压缩产物逐对象一致——
        # loop 原地 extend 会让 outputs[0] 继续增长，故以快照为基准正向比对
        for i, m in enumerate(wm.inputs[1]):
            self.assertIs(m, wm.outputs[0][i])

    def test_abort_discards_inflight_llm_step(self):
        """OPT-116：打断落在 LLM 调用期间——响应丢弃不记录、不执行工具，立即收尾。
        此前 chat 返回后无检查，打断后仍会记录消息并跑完整批工具。"""

        class AbortDuringChat(FakeProvider):
            def __init__(self, evt):
                super().__init__([_tc("myadd", {"a": 1, "b": 2})])
                self.evt = evt

            async def chat(self, messages, tools=None, **kw):
                self.evt.set()  # 模拟用户在 LLM 生成期间点了停止
                return await super().chat(messages, tools=tools, **kw)

        calls: list[str] = []
        reg = ToolRegistry()

        @tool(description="加")
        def myadd(a: int, b: int) -> int:
            calls.append("myadd")
            return a + b

        reg.register(myadd)
        evt = asyncio.Event()
        provider = AbortDuringChat(evt)
        runner = Runner(provider, reg)
        cfg = RunConfig(max_steps=5, signal=evt)
        res = asyncio.run(runner.run(Agent(instructions="sys", tools=reg.all(), max_steps=5),
                                     "问题", cfg=cfg))
        self.assertEqual(res.stop_reason, "aborted")
        self.assertEqual(calls, [], "打断后不得执行任何工具")
        self.assertFalse(any(m.role == "assistant" and m.tool_calls for m in res.messages),
                         "在途响应未对用户可见，应丢弃不记录")

    def test_abort_mid_tool_batch_fills_placeholder(self):
        """OPT-116：批内打断即停——已执行的真实结果保留，未执行的工具补占位（历史合法）。"""

        class BatchAbortProvider(LLMProvider):
            """单响应双工具（c1 sequential 置位打断信号，c2 排队未执行）→ 收尾。"""

            def __init__(self, evt):
                self.evt = evt

            async def chat(self, messages, tools=None, **kw):
                if not any(m.role == "assistant" for m in messages):
                    return LLMResponse(
                        content=None,
                        tool_calls=[_tc("setabort", {}, "c1"),
                                    _tc("myadd", {"a": 1, "b": 2}, "c2")],
                        usage=TokenUsage(input_tokens=10, output_tokens=5),
                        stop_reason="tool_calls")
                return LLMResponse(content="收尾", tool_calls=[],
                                   usage=TokenUsage(input_tokens=1, output_tokens=1),
                                   stop_reason="stop")

        calls: list[str] = []
        reg = ToolRegistry()
        evt = asyncio.Event()

        @tool(description="置位打断信号（模拟用户在工具执行中点停止）", execution_mode="sequential")
        def setabort() -> str:
            evt.set()
            return "signal set"

        @tool(description="加")
        def myadd(a: int, b: int) -> int:
            calls.append("myadd")
            return a + b

        reg.register(setabort)
        reg.register(myadd)
        runner = Runner(BatchAbortProvider(evt), reg)
        cfg = RunConfig(max_steps=5, signal=evt)
        res = asyncio.run(runner.run(Agent(instructions="sys", tools=reg.all(), max_steps=5),
                                     "问题", cfg=cfg))
        self.assertEqual(res.stop_reason, "aborted")
        self.assertEqual(calls, [], "打断后排队的工具不得执行")
        tool_msgs = [m for m in res.messages if m.role == "tool"]
        by_id = {m.tool_call_id: m.content for m in tool_msgs}
        self.assertIn("signal set", by_id.get("c1", ""), "已执行工具的真实结果保留")
        self.assertIn("[已中止]", by_id.get("c2", ""), "未执行工具补占位，tool 对保持完整")

    def test_range_recorder_archives_folded_segment(self):
        """L11/OPT-111：压缩折叠的区段原文归档进 recorder（可逆区段）；
        归档器炸掉不影响主流程；未注入 recorder 时行为不变。"""

        class ShortProvider(LLMProvider):
            async def chat(self, messages, tools=None, **kw):
                return LLMResponse(
                    content="[检查点] Goal 推进中，继续",
                    tool_calls=[],
                    usage=TokenUsage(input_tokens=4, output_tokens=4), stop_reason="stop",
                )

        class _Rec:
            def __init__(self):
                self.archived = []

            def archive(self, msgs):
                self.archived.append(list(msgs))

        class _BrokenRec:
            def archive(self, msgs):
                raise RuntimeError("档案盘炸了")

        script = [_tc("srch", {"q": f"专题{i}"}) for i in range(8)] + ["最终回答"]
        wm = WorkingMemory(keep_recent_tokens=0, summarizer=ShortProvider())
        rec = _Rec()
        runner = Runner(FakeProvider(list(script)), self.reg, memory=wm)
        cfg = RunConfig(context_budget=40, max_steps=15, range_recorder=rec)
        res = asyncio.run(runner.run(self._agent(max_steps=15), "问题", cfg=cfg))
        self.assertEqual(res.stop_reason, "done")
        self.assertTrue(rec.archived, "发生压缩时应调用归档器")
        folded_text = "\n".join(m.content or "" for m in rec.archived[0])
        self.assertIn("搜:专题0", folded_text, "归档的是被折叠区段原文")
        kept_text = "\n".join(m.content or "" for m in res.messages)
        self.assertNotIn("搜:专题0", kept_text, "原文已出窗——档案是唯一可逆入口")

        # 归档器抛错：run 不受影响照常完成
        runner2 = Runner(FakeProvider(list(script)), self.reg,
                         memory=WorkingMemory(keep_recent_tokens=0, summarizer=ShortProvider()))
        cfg2 = RunConfig(context_budget=40, max_steps=15, range_recorder=_BrokenRec())
        res2 = asyncio.run(runner2.run(self._agent(max_steps=15), "问题", cfg=cfg2))
        self.assertEqual(res2.stop_reason, "done")
        self.assertEqual(res2.final_output, "最终回答")

    def test_context_tools_injected_and_status_works(self):
        """L10/OPT-106：context 工具按 run 注入（视图不污染共享注册表），可执行。"""
        from agentlab.core.context_tools import RunContextState, build_context_tools
        from agentlab.tools.registry import RunRegistryView, ToolRegistry

        base = ToolRegistry()
        state = RunContextState()
        view = RunRegistryView(base, build_context_tools(state))
        names = [t["function"]["name"] for t in view.schemas()]
        self.assertIn("context_status", names)
        self.assertIn("compress_context", names)
        # base 工具仍可取
        from agentlab.tools.base import tool as _t
        @_t(name="base_tool", description="d", permission="read")
        def base_tool() -> str:
            return "ok"
        base.register(base_tool)
        self.assertEqual(view.get("base_tool").name, "base_tool")

    def test_context_status_executes_inside_run(self):
        """L10/OPT-106：模型调用 context_status → 返回用量 JSON。"""
        prov = FakeProvider([_tc("context_status", {}), "最终回答"])
        runner = Runner(prov, self.reg)
        res = asyncio.run(runner.run(self._agent(max_steps=15), "问题", cfg=RunConfig(context_budget=1000)))
        self.assertEqual(res.stop_reason, "done")
        tool_msg = next(m for m in res.messages if m.role == "tool" and "usage_pct" in (m.content or ""))
        data = json.loads(tool_msg.content)
        self.assertGreater(data["budget"], 0)
        self.assertIn("nudge_pct", data)

    def test_compress_context_tool_triggers_compaction(self):
        """L10/OPT-106：compress_context 置 force → 下一轮执行锚定压缩（未超预算也生效）。"""
        class ShortProvider(LLMProvider):
            async def chat(self, messages, tools=None, **kw):
                return LLMResponse(content="[检查点] 已折叠旧区段", tool_calls=[],
                                   usage=TokenUsage(input_tokens=4, output_tokens=4), stop_reason="stop")

        wm = WorkingMemory(keep_recent_tokens=0, summarizer=ShortProvider())
        prov = FakeProvider([_tc("compress_context", {}), "最终回答"])
        runner = Runner(prov, self.reg, memory=wm)
        res = asyncio.run(runner.run(self._agent(max_steps=15), "请压缩上下文", cfg=RunConfig(context_budget=8000)))
        self.assertEqual(res.stop_reason, "done")
        self.assertTrue(any(m.role == "system" and "[工作记忆检查点]" in (m.content or "")
                            for m in res.messages), "force 压缩应产出检查点")

    def test_nudge_injected_once_at_threshold(self):
        """L10/OPT-106：用量 ≥75% → 注入一条 [上下文提示]，且每次 run 只有一条。"""
        prov = FakeProvider(["最终回答"])
        runner = Runner(prov, self.reg)
        res = asyncio.run(runner.run(self._agent(max_steps=15), "问" * 320, cfg=RunConfig(context_budget=100)))
        self.assertEqual(res.stop_reason, "done")
        nudges = [m for m in res.messages if (m.content or "").startswith("[上下文提示]")]
        self.assertEqual(len(nudges), 1)

    def test_hard_trim_truncates_oldest_tool_result_protects_frozen(self):
        """L10/OPT-106：≥95% 机械硬截断从最旧 tool 结果截起；冻结区字节不动。"""
        runner = Runner(FakeProvider([]), self.reg)
        runner._compact_anchor = 2
        messages = [
            Message(role="system", content="sys"),
            Message(role="tool", content="f" * 3000, tool_call_id="a", name="t1"),
            Message(role="user", content="问题"),
            Message(role="tool", content="n" * 3000, tool_call_id="b", name="t2"),
        ]
        cfg = RunConfig(context_budget=100, context_hard_trim=0.95)
        runner._hard_trim(messages, cfg, cfg.context_budget)
        self.assertEqual(messages[1].content, "f" * 3000, "冻结区不可动")
        self.assertLessEqual(len(messages[3].content), 230)
        self.assertIn("已硬截断", messages[3].content)
        self.assertTrue(messages[3].content.startswith("nnn"))


if __name__ == "__main__":
    unittest.main()

class TestEffectiveBudget(unittest.TestCase):
    """OPT-110 四期自审：有效预算挂模型窗口（学 pi shouldCompact 挂窗口）。"""

    def setUp(self):
        self.reg = _build_registry()

    def _agent(self, max_steps: int = 15) -> Agent:
        return Agent(instructions="sys", tools=self.reg.all(), max_steps=max_steps)

    def test_effective_budget_caps_by_window(self):
        self.assertEqual(Runner._effective_budget(RunConfig(context_budget=600000, context_window=1048576)), 600000)
        self.assertEqual(Runner._effective_budget(RunConfig(context_budget=600000, context_window=100000)), 60000)
        self.assertEqual(Runner._effective_budget(RunConfig(context_budget=50000, context_window=1048576)), 50000)
        self.assertEqual(Runner._effective_budget(RunConfig(context_budget=50000, context_window=0)), 50000)

    def test_window_capped_budget_triggers_compaction(self):
        # 窗口小 → 有效预算小 → 即使 context_budget 巨大也照常折叠（换模型自动跟随）
        class ShortProvider(LLMProvider):
            async def chat(self, messages, tools=None, **kw):
                return LLMResponse(content="[检查点] ok", tool_calls=[],
                                   usage=TokenUsage(input_tokens=4, output_tokens=4), stop_reason="stop")

        wm = WorkingMemory(keep_recent_tokens=0, summarizer=ShortProvider())
        runner = Runner(FakeProvider([_tc("myadd", {"a": 1, "b": 2}), "ok"]), self.reg, memory=wm)
        cfg = RunConfig(context_budget=600000, context_window=1000, max_steps=15)
        # 首条消息 ~600 token：超过有效预算（窗口 1000×0.6）→ 尽管 context_budget 巨大也应折叠
        res = asyncio.run(runner.run(self._agent(15), "h" * 2400, cfg=cfg))
        self.assertTrue(any(m.role == "system" and "[工作记忆检查点]" in (m.content or "")
                            for m in res.messages), "窗口预算应触发折叠")
