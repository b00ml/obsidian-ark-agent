import unittest
import time
from agentlab.core.errors import AgentError
from agentlab.tools.base import tool
from agentlab.tools.registry import ToolRegistry


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry()

    def test_schema_generation(self):
        @tool(description="回声", permission="read")
        def echo(text: str, limit: int = 20) -> str:  # noqa
            return text[:limit]

        self.reg.register(echo)
        sch = self.reg.schemas()
        self.assertEqual(sch[0]["function"]["name"], "echo")
        props = sch[0]["function"]["parameters"]["properties"]
        self.assertEqual(props["text"]["type"], "string")
        self.assertEqual(sch[0]["function"]["parameters"]["required"], ["text"])

    def test_optional_type(self):
        @tool(description="可选参数")
        def f(name: str = "x"):  # noqa
            return name

        self.reg.register(f)
        props = self.reg.schemas()[0]["function"]["parameters"]["properties"]
        self.assertEqual(props["name"]["type"], "string")
        self.assertNotIn("required", self.reg.schemas()[0]["function"]["parameters"])

    def test_duplicate_registration(self):
        @tool(description="a")
        def a(b: int):  # noqa
            return b

        self.reg.register(a)
        with self.assertRaisesRegex(AgentError, "重复"):
            self.reg.register(a)

    def test_active_subset(self):
        @tool(description="a")
        def a():  # noqa
            return 1

        @tool(description="b")
        def b():  # noqa
            return 2

        self.reg.register(a)
        self.reg.register(b)
        self.assertEqual([t.name for t in self.reg.active(["a"])], ["a"])

    def test_disable_model_invocation_hidden_from_schema(self):
        @tool(description="内部", disable_model_invocation=True)
        def secret():  # noqa
            return "s"

        self.reg.register(secret)
        self.assertEqual(self.reg.schemas(), [])

    def test_missing_tool(self):
        from agentlab.core.message import ToolCall, ToolCallFunction
        c = ToolCall(id="c", function=ToolCallFunction(name="nope", arguments="{}"))
        with self.assertRaises(AgentError) as e:
            import asyncio
            asyncio.run(self.reg.execute(c))
        self.assertEqual(e.exception.code, "AGENT_TOOL_NOT_FOUND")


class TestLongTask(unittest.TestCase):
    """长任务可观测性 + 兜底：不阻塞事件循环、超时不杀线程、心跳上报。"""

    def setUp(self):
        self.reg = ToolRegistry()

    def _tool_call(self, name, args="{}"):
        from agentlab.core.message import ToolCall, ToolCallFunction
        return ToolCall(id="t1", function=ToolCallFunction(name=name, arguments=args))

    def test_threadpool_does_not_block_event_loop(self):
        import asyncio

        @tool(description="慢同步任务")
        def slow(secs: float = 0.3) -> str:
            time.sleep(secs)
            return "done"
        self.reg.register(slow)

        async def run():
            loop_alive = False
            fut_with_hb = asyncio.ensure_future(
                self.reg.execute(self._tool_call("slow"), progress=lambda n, e: None)
            )
            # 事件循环仍可响应其他协程 = 同步工具没阻塞事件循环
            await asyncio.sleep(0.05)
            loop_alive = True
            r = await asyncio.wait_for(fut_with_hb, timeout=2)
            return loop_alive, r.content

        loop_alive, content = asyncio.run(run())
        self.assertTrue(loop_alive, "同步长任务不应阻塞事件循环")
        self.assertEqual(content, "done")

    def test_timeout_reports_but_thread_keeps_running(self):
        import asyncio
        done_flag = []

        @tool(description="超时块", execution_timeout=0.2)
        def slow_job() -> str:
            time.sleep(0.4)  # 线程继续跑
            done_flag.append(True)
            return "late"

        self.reg.register(slow_job)

        async def run():
            r = await asyncio.wait_for(
                self.reg.execute(self._tool_call("slow_job")), timeout=2
            )
            return r.content

        content = asyncio.run(run())
        self.assertIn("工具执行超时", content)
        self.assertIn("仍在后台运行", content)
        # 超时不杀线程：后台线程应最终完成并写入 side effect
        time.sleep(0.5)
        self.assertEqual(done_flag, [True], "超时后后台线程应继续运行完成")

    def test_heartbeat_reports_while_pending(self):
        import asyncio

        @tool(description="心跳任务")
        def beater() -> str:
            time.sleep(0.25)
            return "beat-done"

        self.reg.register(beater)
        beats = []

        import agentlab.tools.registry as regmod
        old = regmod.HEARTBEAT_INTERVAL
        regmod.HEARTBEAT_INTERVAL = 0.05

        async def run():
            r = await self.reg.execute(
                self._tool_call("beater"),
                progress=lambda n, e: beats.append((n, e)),
            )
            return r.content

        try:
            content = asyncio.run(run())
        finally:
            regmod.HEARTBEAT_INTERVAL = old
        self.assertEqual(content, "beat-done")
        self.assertGreater(len(beats), 0, "长任务运行期间应收到心跳")
        names = {n for n, _ in beats}
        self.assertEqual(names, {"beater"})

    def test_async_tool_runs_natively(self):
        import asyncio

        @tool(description="原生 async 工具")
        async def af() -> str:
            return "async-ok"
        self.reg.register(af)
        content = asyncio.run(self.reg.execute(self._tool_call("af"))).content
        self.assertEqual(content, "async-ok")

    def test_signal_stops_waiting_for_async_tool(self):
        """停止必须结束 Agent 的工具等待；可取消 async 工具不应拖到自然完成。"""
        import asyncio

        started = asyncio.Event()
        finished = []

        @tool(description="可取消慢工具")
        async def slow_async() -> str:
            started.set()
            try:
                await asyncio.sleep(2)
            finally:
                finished.append(True)
            return "late"

        self.reg.register(slow_async)

        async def run():
            signal = asyncio.Event()
            task = asyncio.create_task(self.reg.execute(
                self._tool_call("slow_async"), signal=signal))
            await asyncio.wait_for(started.wait(), timeout=1)
            signal.set()
            t0 = time.monotonic()
            result = await asyncio.wait_for(task, timeout=0.5)
            return time.monotonic() - t0, result.content

        elapsed, content = asyncio.run(run())
        self.assertLess(elapsed, 0.5)
        self.assertIn("已中止", content)
        self.assertEqual(finished, [True], "async 工具应收到取消并执行清理")


class TestPermission(unittest.TestCase):
    """F4 修复：confirm 缺失必须 fail-closed 拒绝，而非放行。

    此前 `confirm=None` 时 `ok` 恒为 True → web 链路写/危险工具无确认直通。
    回归验证：无 confirm → 拒绝；confirm 拒绝 → 拒绝；confirm 放行 → 执行。
    """

    def setUp(self):
        self.reg = ToolRegistry()

    def _make_call(self, name, args="{}"):
        from agentlab.core.message import ToolCall, ToolCallFunction
        return ToolCall(id="t1", function=ToolCallFunction(name=name, arguments=args))

    @staticmethod
    def _register(reg, name, permission):
        @tool(name=name, description=name, permission=permission)
        def fn(v: int = 1) -> int:  # noqa
            return v
        reg.register(fn)
        return fn

    def test_write_without_confirm_raises(self):
        import asyncio
        self._register(self.reg, "do_write", "write")
        with self.assertRaises(AgentError) as e:
            asyncio.run(self.reg.execute(self._make_call("do_write")))

    def test_danger_without_confirm_raises(self):
        import asyncio
        self._register(self.reg, "do_danger", "danger")
        with self.assertRaises(AgentError) as e:
            asyncio.run(self.reg.execute(self._make_call("do_danger")))

    def test_read_without_confirm_is_allowed(self):
        import asyncio
        self._register(self.reg, "do_read", "read")
        r = asyncio.run(self.reg.execute(self._make_call("do_read")))
        self.assertEqual(r.content, "1")

    def test_write_confirm_false_rejected(self):
        import asyncio
        self._register(self.reg, "do_write", "write")
        with self.assertRaisesRegex(AgentError, "用户拒绝"):
            asyncio.run(self.reg.execute(self._make_call("do_write"), confirm=lambda t, p: False))

    def test_write_confirm_true_executes(self):
        import asyncio
        self._register(self.reg, "do_write", "write")
        r = asyncio.run(self.reg.execute(self._make_call("do_write"), confirm=lambda t, p: True))
        self.assertEqual(r.content, "1")


if __name__ == "__main__":
    unittest.main()
