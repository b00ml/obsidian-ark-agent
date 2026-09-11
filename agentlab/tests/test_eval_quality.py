"""F5-024 答案质量基线：离线可测的部分（加载 / 评审解析 / 汇总 / 报告 / 对比 / 失败隔离）。

真跑模型的部分不在单测里——那要花钱且有外部依赖；这里保证"评测机制本身"正确：
任务集校验、rubric 解析、分数归一、单条失败不毁整轮、对比的方向判断。
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from agentlab.core.llm import LLMProvider, LLMResponse
from agentlab.core.message import TokenUsage
from agentlab.eval import quality


class _FakeJudge(LLMProvider):
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0

    async def chat(self, messages, tools=None, **kw):
        self.calls += 1
        return LLMResponse(content=self.payload, tool_calls=[],
                           usage=TokenUsage(input_tokens=1, output_tokens=1))


class _FakeTool:
    def __init__(self, permission, name=""):
        self.permission = permission
        self.name = name


def _write_tasks(rows) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "tasks.jsonl"
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return tmp


class TestTaskLoading(unittest.TestCase):
    def test_default_task_set_loads_and_covers_real_workloads(self):
        tasks = quality.load_quality_tasks()
        ids = [t["id"] for t in tasks]
        self.assertEqual(len(ids), len(set(ids)), "id 必须唯一")
        self.assertGreaterEqual(len(tasks), 10, "质量基线至少 10 条真实任务")
        for must in ("q-anti-hallucination", "q-cross-note-compare", "q-inbox-triage",
                     "q-boundary-refusal"):
            self.assertIn(must, ids, f"关键场景缺失：{must}")

    def test_comments_and_blank_lines_are_skipped(self):
        p = Path(tempfile.mkdtemp()) / "t.jsonl"
        p.write_text('# 注释\n\n{"id":"a","task":"做点事"}\n', encoding="utf-8")
        self.assertEqual([t["id"] for t in quality.load_quality_tasks(p)], ["a"])

    def test_invalid_rows_raise_with_line_context(self):
        bad_json = Path(tempfile.mkdtemp()) / "t.jsonl"
        bad_json.write_text('{"id":"a","task":"x"}\n{不是 JSON\n', encoding="utf-8")
        with self.assertRaises(ValueError) as e:
            quality.load_quality_tasks(bad_json)
        self.assertIn(":2", str(e.exception), "报错要带行号，否则改起来靠猜")
        with self.assertRaises(ValueError):
            quality.load_quality_tasks(_write_tasks([{"id": "a", "task": "x"},
                                                    {"id": "a", "task": "y"}]))
        with self.assertRaises(ValueError):
            quality.load_quality_tasks(_write_tasks([{"id": "a"}]))
        empty = Path(tempfile.mkdtemp()) / "t.jsonl"
        empty.write_text("# 只有注释\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            quality.load_quality_tasks(empty)


class TestEvalToolSurface(unittest.TestCase):
    """评测工具面必须是**纯查询白名单**，不是"read 权限"。

    首轮 smoke 实测教训：`permission == "read"` 挡不住外部副作用——
    `inbox_collect` 拉了邮件并把条目写进队列库、`bili_*` 下载并写了 cookie 文件。
    """

    def test_side_effecting_read_tools_are_excluded(self):
        for name in ("inbox_collect", "brain_reindex", "rag_reindex",
                     "web_search", "article_fetch", "bili_meta", "bili_transcribe"):
            self.assertFalse(quality.eval_tool_filter(_FakeTool("read", name)),
                             f"{name} 有外部副作用，不得进入评测工具面")

    def test_write_danger_and_unknown_tools_are_excluded(self):
        self.assertFalse(quality.eval_tool_filter(_FakeTool("write", "vault_write")))
        self.assertFalse(quality.eval_tool_filter(_FakeTool("danger", "agent_consult")))
        self.assertFalse(quality.eval_tool_filter(object()), "无 name 的工具默认排除")
        self.assertFalse(quality.eval_tool_filter(_FakeTool("read", "未来新增的工具")))

    def test_pure_query_tools_survive(self):
        for name in ("vault_search", "vault_read", "rag_retrieve", "memory_query", "brain_scan"):
            self.assertTrue(quality.eval_tool_filter(_FakeTool("read", name)), name)

    def test_filtered_registry_view_is_the_real_gate(self):
        """真正的闸门是注册表视图，不是 Agent.tools。

        首轮 smoke 的教训：只过滤 agent.tools 时，模型仍从 runner.registry.schemas()
        拿到全部工具并真的调用了 inbox_collect/web_search/bili_meta。
        """
        from agentlab.tools.base import tool
        from agentlab.tools.registry import FilteredRegistryView, ToolRegistry

        @tool(name="vault_search", description="查询")
        def _q() -> str:
            return "ok"

        @tool(name="inbox_collect", description="有副作用的读取")
        def _s() -> str:
            return "ok"

        base = ToolRegistry()
        base.register(_q)
        base.register(_s)
        view = FilteredRegistryView(base, quality.eval_tool_filter)

        self.assertEqual([t.name for t in view.all()], ["vault_search"],
                         "被排除的工具不得出现在视图里")
        self.assertEqual([s["function"]["name"] for s in view.schemas()], ["vault_search"],
                         "发给模型的 schema 必须来自过滤后的视图")
        self.assertEqual(view.get("vault_search").name, "vault_search")
        with self.assertRaises(Exception):
            view.get("inbox_collect")
        self.assertEqual(len(base.all()), 2, "base 不得被改动（视图是只读包装）")


class TestEvidenceBuilder(unittest.TestCase):
    def test_evidence_keeps_head_and_tail_of_long_tool_result(self):
        """工具返回的计数/结论常在末尾，只留头部会让评审把真实数字判成幻觉。

        超过原始留存上限（6000）才截断；5,000 字这一档现在会**整条保留**（见
        `test_few_calls_let_the_whole_result_through`），不再人为制造信息缺口。
        """
        sink = quality._CollectSink()
        sink.tool_start("vault_scan")
        sink.tool_end("vault_scan", "开头信息" + "x" * 20000 + "共 135 篇笔记")
        ev = sink.evidence
        self.assertIn("开头信息", ev)
        self.assertIn("共 135 篇笔记", ev, "尾部结论必须保留")
        self.assertIn("中段省略", ev, "被省略的部分要显式标注，不能让评审以为看到了全文")

    def test_evidence_lists_tool_calls(self):
        sink = quality._CollectSink()
        for n in ("vault_search", "rag_retrieve"):
            sink.tool_start(n)
        sink.tool_end("vault_search", "结果")
        ev = sink.evidence
        self.assertIn("工具调用清单", ev)
        self.assertIn("vault_search", ev)
        self.assertIn("rag_retrieve", ev)

    def test_evidence_includes_tool_arguments(self):
        """没有参数，评审无法判断"查了什么才这么说"，事后也无法复现检索路径。"""
        sink = quality._CollectSink()
        sink.tool_start("vault_search", '{"keyword": "架构决定"}')
        sink.tool_end("vault_search", "{'total': 2}")
        self.assertIn("架构决定", sink.evidence, "调用清单必须带参数")

    def test_evidence_lists_scalar_fields_from_tool_result(self):
        """计数类结论常落在被省略的中段 —— 原样列出标量字段，评审才不会判成编造。"""
        sink = quality._CollectSink()
        sink.tool_start("vault_health")
        sink.tool_end("vault_health",
                      "{'total_notes': 165, 'broken_links_count': 135, "
                      "'orphan_count': 48, 'unlinked_count': 68}" + "x" * 900)
        ev = sink.evidence
        self.assertIn("broken_links_count = 135", ev)
        self.assertIn("unlinked_count = 68", ev)
        self.assertIn("标量字段", ev)

    def test_evidence_lists_source_paths(self):
        """可追溯性要能核：证据里得有"读过哪些文件"。"""
        sink = quality._CollectSink()
        sink.tool_start("vault_read", '{"path": "AGENTS.md"}')
        sink.tool_end("vault_read", "{'path': 'AGENTS.md', 'content': '规则...'}")
        sink.tool_start("vault_search", '{"keyword": "hermes"}')
        sink.tool_end("vault_search", "{'results': [{'path': '02-DB/回顾/2026-08-24-周报.md'}]}")
        ev = sink.evidence
        self.assertIn("本轮读到的来源", ev)
        self.assertIn("AGENTS.md", ev)
        self.assertIn("02-DB/回顾/2026-08-24-周报.md", ev)

    def test_evidence_budget_is_shared_across_all_calls(self):
        """额度按调用次数公平分，不能让前几次吃光、后面的整条丢失。

        实测来源（OPT-206）：Agent 一条任务最多调 46 次工具，旧的"额度耗尽即 break"
        会把末尾十几条证据全丢掉——而答案引用的常常正是后面读到的那几篇。
        """
        sink = quality._CollectSink()
        for i in range(40):
            sink.tool_start(f"vault_read_{i:02d}")
            sink.tool_end(f"vault_read_{i:02d}", f"笔记{i:02d}正文" + "x" * 5000)
        ev = sink.evidence
        missing = [f"vault_read_{i:02d}" for i in range(40)
                   if f"vault_read_{i:02d} 返回" not in ev]
        self.assertEqual(missing, [], f"这些调用的证据被整条丢弃：{missing}")
        self.assertLessEqual(len(ev), quality.EVIDENCE_TOTAL_CHARS + 200,
                             "证据块整体必须守住总量上限（含清单/来源/标量三段）")

    def test_few_calls_let_the_whole_result_through(self):
        """调用少时不该再给单工具设硬顶：只有 1 次调用时，硬顶会把清单从中间切开。

        实测来源（OPT-206/207）：`q-orphan-notes` 单次 `vault_health` 返回 3,168 字，
        按 1,200 硬顶截断后答案引用的路径评审看不到，又被判"编造"。
        """
        sink = quality._CollectSink()
        sink.tool_start("vault_health")
        sink.tool_end("vault_health", "开头" + "y" * 2000 + "结尾的计数 135")
        ev = sink.evidence
        self.assertNotIn("中段省略", ev, "调用少时整条结果都该进来")
        self.assertIn("结尾的计数 135", ev)

    def test_huge_single_result_is_bounded_by_raw_cap(self):
        sink = quality._CollectSink()
        sink.tool_start("vault_read")
        sink.tool_end("vault_read", "开头" + "y" * 20000 + "结尾的计数 135")
        ev = sink.evidence
        self.assertIn("结尾的计数 135", ev, "尾部结论必须保留")
        self.assertIn("中段省略", ev)
        self.assertLess(len(ev), quality.EVIDENCE_RAW_PER_TOOL_CHARS + 400,
                        f"超大单条要由原始留存上限兜底，实际 {len(ev)}")

    def test_short_string_facts_are_listed(self):
        """时间/编号这类短字符串也是事实：q-daily-review 引用的 15:54 就在 runs_recent 里。"""
        sink = quality._CollectSink()
        sink.tool_start("runs_recent", '{"days": 1}')
        sink.tool_end("runs_recent",
                      "{'runs': [{'time': '2026-09-11T15:54', 'status': 'done', "
                      "'input': '" + "很长的输入" * 200 + "'}]}")
        ev = sink.evidence
        self.assertIn("time = 2026-09-11T15:54", ev)
        self.assertIn("status = done", ev)



class TestJudgeParsing(unittest.TestCase):
    def _judge(self, payload: str, task=None):
        llm = _FakeJudge(payload)
        t = task or {"id": "a", "task": "做点事", "evidence": "无"}
        return asyncio.run(quality.judge_answer(llm, t, "答案")), llm

    def test_parses_plain_json_and_keeps_evidence_fields(self):
        s, llm = self._judge(json.dumps({
            "factuality": 4, "coverage": 3, "traceability": 2, "actionability": 5,
            "hallucinated": ["编造的数字 42"], "missing": ["未给出处"], "reason": "出处缺失",
        }))
        self.assertEqual((s["factuality"], s["coverage"], s["traceability"], s["actionability"]),
                         (4.0, 3.0, 2.0, 5.0))
        self.assertEqual(s["hallucinated"], ["编造的数字 42"])
        self.assertEqual(s["missing"], ["未给出处"])
        self.assertEqual(s["reason"], "出处缺失")
        self.assertEqual(llm.calls, 1)

    def test_tolerates_code_fence_and_clamps_out_of_range(self):
        s, _ = self._judge('```json\n{"factuality": 9, "coverage": -3, '
                           '"traceability": 2, "actionability": 3}\n```')
        self.assertEqual(s["factuality"], 5.0, "越界分数必须夹到 0-5")
        self.assertEqual(s["coverage"], 0.0)

    def test_missing_dimension_scores_zero_not_crash(self):
        s, _ = self._judge('{"factuality": 4}')
        self.assertEqual(s["coverage"], 0.0)
        self.assertEqual(s["actionability"], 0.0)

    def test_non_json_judge_output_raises_instead_of_silent_zero(self):
        """解析失败必须抛错：静默给 0 会被误读成"答案质量差"。"""
        with self.assertRaises(ValueError):
            self._judge("我觉得这个答案还行")

    def test_truncated_judge_json_salvages_scores_and_flags(self):
        """全量实测：评审会把 JSON 撑爆（hallucinated 里引用大段原文）。

        四个分数组在最前面且通常完整，因此打捞分数比整条丢弃更有用——但必须打标，
        不能装作拿到了完整评审。
        """
        truncated = ('{"factuality": 4, "coverage": 5, "traceability": 3, "actionability": 5, '
                     '"hallucinated": ["这句引用很长很长很长被截断了…')
        s, _ = self._judge(truncated)
        self.assertEqual((s["factuality"], s["coverage"], s["traceability"], s["actionability"]),
                         (4.0, 5.0, 3.0, 5.0))
        self.assertTrue(s.get("judge_truncated"), "打捞来的结果必须显式标记")

    def test_salvage_gives_up_when_no_scores_present(self):
        with self.assertRaises(ValueError):
            self._judge('{"hallucinated": ["只有这一项就被截断了')

    def test_unverifiable_is_not_counted_as_hallucination(self):
        """评审看不到依据时应用 unverifiable 表达，而不是把真实结论判成编造。"""
        s, _ = self._judge(json.dumps({
            "factuality": 4, "coverage": 4, "traceability": 4, "actionability": 4,
            "unverifiable": ["证据节选未覆盖的细节"], "hallucinated": [],
        }))
        self.assertEqual(s["hallucinated"], [])
        self.assertEqual(s["unverifiable"], ["证据节选未覆盖的细节"])

    def test_tool_evidence_reaches_the_judge_prompt(self):
        """评审必须拿到 Agent 实际读到的内容——否则真实结论会被判成幻觉。"""
        captured: dict[str, str] = {}

        class _Recorder(LLMProvider):
            async def chat(self, messages, tools=None, **kw):
                captured["prompt"] = messages[0].content
                return LLMResponse(content='{"factuality":5,"coverage":5,"traceability":5,"actionability":5}',
                                   tool_calls=[], usage=TokenUsage(input_tokens=1, output_tokens=1))

        task = {"id": "a", "task": "T", "evidence": "任务自带的说明（不该被优先使用）"}
        asyncio.run(quality.judge_answer(
            _Recorder(), task, "答案",
            evidence="### 工具 vault_search 返回\n真实检索结果：笔记 X 提到 Y"))
        self.assertIn("真实检索结果", captured["prompt"], "工具证据必须进 prompt")
        self.assertNotIn("任务自带的说明", captured["prompt"], "有真证据时不再退回任务说明")

    def test_judge_call_pins_max_tokens(self):
        """评审调用必须显式给 max_tokens：不限定会让推理模型把预算烧在推理上、content 返回空。"""
        captured: dict[str, object] = {}

        class _Recorder(LLMProvider):
            async def chat(self, messages, tools=None, **kw):
                captured.update(kw)
                return LLMResponse(content='{"factuality":5,"coverage":5,"traceability":5,"actionability":5}',
                                   tool_calls=[], usage=TokenUsage(input_tokens=1, output_tokens=1))

        asyncio.run(quality.judge_answer(_Recorder(), {"id": "a", "task": "T"}, "答案"))
        self.assertEqual(captured.get("max_tokens"), quality.JUDGE_MAX_TOKENS)
        self.assertEqual(captured.get("temperature"), 0.0, "评审必须确定性")

    def test_falls_back_to_task_evidence_when_no_tool_ran(self):
        captured: dict[str, str] = {}

        class _Recorder(LLMProvider):
            async def chat(self, messages, tools=None, **kw):
                captured["prompt"] = messages[0].content
                return LLMResponse(content='{"factuality":5,"coverage":5,"traceability":5,"actionability":5}',
                                   tool_calls=[], usage=TokenUsage(input_tokens=1, output_tokens=1))

        task = {"id": "a", "task": "T", "evidence": "无外部依据"}
        asyncio.run(quality.judge_answer(_Recorder(), task, "答案", evidence=""))
        self.assertIn("无外部依据", captured["prompt"])


class TestAggregate(unittest.TestCase):
    def _row(self, **dims):
        scores = {d: float(dims.get(d, 5)) for d in quality.QUALITY_DIMENSIONS}
        scores.update(hallucinated=dims.get("hallucinated", []), missing=[], reason="")
        return {"id": "x", "scores": scores}

    def test_averages_and_normalizes(self):
        rows = [self._row(factuality=5, coverage=5, traceability=5, actionability=5),
                self._row(factuality=0, coverage=0, traceability=0, actionability=0)]
        agg = quality.aggregate(rows)
        self.assertEqual(agg["per_dimension_0_5"]["factuality"], 2.5)
        self.assertEqual(agg["per_dimension_0_1"]["factuality"], 0.5)
        self.assertEqual(agg["overall_0_1"], 0.5)
        self.assertEqual((agg["tasks_scored"], agg["tasks_total"]), (2, 2))

    def test_hallucination_count_and_unscored_rows(self):
        rows = [self._row(hallucinated=["编造"]), {"id": "y", "scores": None, "error": "boom"}]
        agg = quality.aggregate(rows)
        self.assertEqual(agg["tasks_scored"], 1, "执行失败的任务不计入均分")
        self.assertEqual(agg["tasks_total"], 2)
        self.assertEqual(agg["hallucination_free_tasks"], 0)

    def test_empty_input_is_zero_not_crash(self):
        agg = quality.aggregate([])
        self.assertEqual(agg["overall_0_1"], 0.0)
        self.assertEqual(agg["tasks_scored"], 0)

    def test_truncated_judge_count_is_surfaced(self):
        r = self._row()
        r["scores"]["judge_truncated"] = True
        agg = quality.aggregate([r])
        self.assertEqual(agg["judge_truncated_tasks"], 1,
                         "打捞来的评审判定必须能被报告看见，否则基线看起来比实际可靠")


class TestEndToEndWithFakes(unittest.TestCase):
    """用假 backend / 假 _run_agent 验证报告结构与"单条失败不毁整轮"。"""

    def _run(self, tasks, judge_payload=None, raise_on=None, resume=None, calls=None,
             fresh_ids=None):
        import agentlab.runtime.serve as serve_mod

        async def fake_run_agent(cfg, build_backend, hist, user_input, sink, **kw):
            if calls is not None:
                calls.append(user_input)
            if raise_on and raise_on in user_input:
                raise RuntimeError("任务执行炸了")
            sink.text(f"回答：{user_input[:20]}")
            return {"final_output": "", "stop_reason": "done", "tokens": 7}

        originals = (serve_mod._build_backend, serve_mod._run_agent)
        serve_mod._build_backend = lambda cfg: (None, 0, None)
        serve_mod._run_agent = fake_run_agent
        try:
            cfg = type("C", (), {"llm": type("L", (), {"model": "test-model"})()})()
            judge = _FakeJudge(judge_payload or json.dumps(
                {"factuality": 4, "coverage": 4, "traceability": 4, "actionability": 4}))
            return quality.run_sync(quality.run_quality_baseline(
                cfg, tasks=tasks, judge_llm=judge, resume=resume, fresh_ids=fresh_ids))
        finally:
            serve_mod._build_backend, serve_mod._run_agent = originals

    def test_report_schema_and_tokens(self):
        tasks = [{"id": "a", "task": "任务 A"}, {"id": "b", "task": "任务 B"}]
        report = self._run(tasks)
        self.assertEqual(report["schema"], "f5-024.v1")
        self.assertEqual(report["agent_model"], "test-model")
        self.assertEqual(report["judge_model"], "test-model", "评审模型必须固定并写进报告")
        self.assertIn("vault_search", report["tool_surface"], "报告要记录评测工具面")
        self.assertNotIn("inbox_collect", report["tool_surface"])
        self.assertEqual(report["cloud_tokens_reported"], 14,
                         f"rows={report['tasks']}")
        self.assertEqual(report["summary"]["tasks_scored"], 2,
                         f"任务执行出错：{[t.get('error') for t in report['tasks']]}")
        self.assertEqual(report["tasks"][0]["answer"], "回答：任务 A",
                         f"rows={report['tasks']}")

    def test_report_counts_judge_tokens_separately(self):
        """总花费 = agent 侧 + 评审侧：此前只统计 agent 侧，全量一轮的真实花费被低估。"""
        report = self._run([{"id": "a", "task": "任务 A"}])
        self.assertIn("judge_tokens_reported", report)
        self.assertEqual(report["cloud_tokens_total"],
                         report["cloud_tokens_reported"] + report["judge_tokens_reported"],
                         "不能把评审开销漏掉")

    def test_one_task_failure_does_not_kill_the_run(self):
        tasks = [{"id": "a", "task": "任务 A"}, {"id": "bad", "task": "会炸的任务"},
                 {"id": "c", "task": "任务 C"}]
        report = self._run(tasks, raise_on="会炸")
        by_id = {t["id"]: t for t in report["tasks"]}
        self.assertIn("error", by_id["bad"])
        self.assertIsNone(by_id["bad"]["scores"])
        self.assertEqual(report["summary"]["tasks_scored"], 2, "另外两条照常评分")
        self.assertEqual(report["summary"]["tasks_total"], 3)

    def test_write_report_creates_missing_directories(self):
        report = self._run([{"id": "a", "task": "任务 A"}])
        out = Path(tempfile.mkdtemp()) / "sub" / "r.json"
        quality.write_report(report, out)
        self.assertTrue(out.exists(), "目录不存在时应自动创建")
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["schema"], "f5-024.v1")

    def test_resume_reuses_scored_rows_and_retries_failed_ones(self):
        """全量一轮 1–3M token，中途断了必须能续跑：已出分的别再付一次钱。

        实测来源：2026-09-11 全量跑到第 3 条撞上 HTTP 402，10/12 条作废而 345k token 已花。
        """
        tasks = [{"id": "a", "task": "任务 A"}, {"id": "b", "task": "会炸的任务"}]
        first = self._run(tasks, raise_on="会炸")
        self.assertEqual(first["summary"]["tasks_scored"], 1)
        self.assertIn("error", {t["id"]: t for t in first["tasks"]}["b"])

        calls: list[str] = []
        second = self._run(tasks, resume=first, calls=calls)
        by_id = {t["id"]: t for t in second["tasks"]}
        self.assertTrue(by_id["a"]["resumed"], "已出分的行应标记为复用")
        self.assertEqual(len(calls), 1, f"只应重跑失败那条，实际跑了 {calls}")
        self.assertIn("会炸", calls[0])
        self.assertNotIn("error", by_id["b"], "续跑应把上轮失败的任务补上")
        self.assertEqual(second["summary"]["tasks_scored"], 2)
        self.assertEqual(second["resumed_tasks"], 1)
        self.assertEqual(second["resumed_from"], first["timestamp"])
        self.assertEqual(second["summary"]["tasks_total"], 2, "报告仍是完整任务集")

    def test_resume_ignores_rows_that_only_have_errors(self):
        stale = {"timestamp": "t0", "tasks": [{"id": "a", "task": "任务 A",
                                              "scores": None, "error": "boom"}]}
        report = self._run([{"id": "a", "task": "任务 A"}], resume=stale)
        self.assertEqual(report["resumed_tasks"], 0, "失败行不算复用，必须重跑")
        self.assertEqual(report["summary"]["tasks_scored"], 1)

    def test_fresh_ids_force_rerun_even_if_resume_has_scores(self):
        """修完某个工具后要能"只补跑受影响的两条"，不能被续跑原样搬回来。"""
        tasks = [{"id": "a", "task": "任务 A"}, {"id": "b", "task": "任务 B"}]
        first = self._run(tasks)
        calls: list[str] = []
        second = self._run(tasks, resume=first, calls=calls, fresh_ids={"b"})
        self.assertEqual(len(calls), 1, f"只应重跑 b，实际 {calls}")
        self.assertIn("任务 B", calls[0])
        by_id = {t["id"]: t for t in second["tasks"]}
        self.assertTrue(by_id["a"]["resumed"], "未点名的那条照旧复用")
        self.assertFalse(by_id["b"].get("resumed"), "点名的要重跑，不能标 resumed")
        self.assertEqual(second["resumed_tasks"], 1)

    def test_partial_rerun_keeps_the_whole_report(self):
        """补跑两条时报告仍须是完整任务集，不能缩成只剩被点名的那两条。"""
        tasks = [{"id": x, "task": f"任务 {x}"} for x in ("a", "b", "c")]
        first = self._run(tasks)
        calls: list[str] = []
        # 模拟 CLI：--only b --resume first（tasks 已被 --only 过滤成只剩 b）
        second = self._run([tasks[1]], resume=first, calls=calls, fresh_ids={"b"})
        self.assertEqual([t["id"] for t in second["tasks"]], ["a", "b", "c"],
                         "输出集合以上一轮报告为准并保序")
        self.assertEqual(len(calls), 1, f"只应重跑 b，实际 {calls}")
        self.assertEqual(second["summary"]["tasks_total"], 3)
        self.assertEqual(second["resumed_tasks"], 2)


class TestClaimAudit(unittest.TestCase):
    """判词对证：把评审判成"编造"的条目拿去 Vault 找出处（纯本地，不调模型）。

    来源：全量基线里三条被核实为误判的条目（commit 6420703 / hermes 路径 / 技能库路径），
    全部在 Vault 里有逐字出处。这个审计用来说明 factuality 维度里有多少是度量噪声。
    """

    def setUp(self):
        self.vault = Path(tempfile.mkdtemp())
        (self.vault / "RAG向量增删.md").write_text(
            "OPT-105 三期、commit 6420703、3.0 backlog、91 文件 / 86 秒", encoding="utf-8")
        (self.vault / "周报.md").write_text("以 Hermes 为运行本体（~/AppData/Local/hermes）",
                                          encoding="utf-8")
        (self.vault / ".agent-brain").mkdir()
        (self.vault / ".agent-brain" / "mem.md").write_text("commit 6420703 不该被扫到",
                                                            encoding="utf-8")

    def _report(self, claims, task_id="q-x"):
        return {"timestamp": "t", "tasks": [
            {"id": task_id, "scores": {"hallucinated": claims}}]}

    def test_claims_with_vault_provenance_are_flagged_as_false_positives(self):
        report = self._report(["OPT-105三期、commit 6420703、91文件/86秒",
                               "Hermes 路径 ~/AppData/Local/hermes"])
        result = quality.audit_claims(report, self.vault)
        self.assertEqual(result["claims_found_in_vault"], 2, result["claims"])
        self.assertEqual(result["claims_not_found"], 0)
        files = {row["file"] for row in result["claims"]}
        self.assertIn("RAG向量增删.md", files)
        self.assertIn("周报.md", files)

    def test_claims_without_any_trace_stay_unresolved(self):
        report = self._report(["完全不存在的笔记 Zzzq-9911-YYY"])
        result = quality.audit_claims(report, self.vault)
        self.assertEqual(result["claims_found_in_vault"], 0)
        self.assertIsNone(result["claims"][0]["file"], "查无实据时不能硬指一个文件")

    def test_brain_dir_is_not_scanned(self):
        """`.agent-brain/` 是数据目录不是笔记，不能被当成出处。"""
        report = self._report(["Zzzq-9911-YYY 只存在于 data 目录"])
        result = quality.audit_claims(report, self.vault)
        self.assertEqual(result["files_scanned"], 2, "只应扫到两篇 md")
        self.assertEqual(result["claims_found_in_vault"], 0)


class TestCompare(unittest.TestCase):
    def _report(self, **vals):
        dims = {d: vals.get(d, 0.5) for d in quality.QUALITY_DIMENSIONS}
        return {"timestamp": "t", "summary": {"per_dimension_0_1": dims,
                "overall_0_1": sum(dims.values()) / len(dims)}}

    def test_compare_flags_regression_with_nonzero_exit(self):
        from agentlab.eval.run_quality import _compare
        better = self._report(factuality=0.8)
        worse = self._report(factuality=0.4)
        self.assertEqual(_compare(worse, better), 0, "变好应返回 0")
        self.assertEqual(_compare(better, worse), 1, "任一维变差应返回非零，供脚本卡口")


if __name__ == "__main__":
    unittest.main()
