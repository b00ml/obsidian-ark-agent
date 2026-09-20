import asyncio
import unittest
from unittest.mock import patch

from agentlab.eval.answer_probe_collect import (
    _direct_run, _live_run, collect_answer_probes, select_tasks,
)
from agentlab.eval.rag_task_schema import migrate_v1_task


def _task(task_id, query_type, *, answerability="answerable",
          expected_refs=None, expected_policy=None):
    return migrate_v1_task({
        "id": task_id,
        "query": f"query {task_id}",
        "expected_refs": (
            list(expected_refs) if expected_refs is not None
            else [f"wiki/{task_id}.md"] if answerability == "answerable" else []
        ),
        "query_type": query_type,
        "answerability": answerability,
        **({"expected_policy": expected_policy} if expected_policy else {}),
    })


class TestAnswerProbeCollection(unittest.TestCase):
    def test_direct_mode_runs_each_stage_once_and_reports_budget(self):
        calls = []

        class Provider:
            async def chat(self, messages, tools=None, **kwargs):
                calls.append("answer")
                return type("Response", (), {
                    "content": "答案（ref: wiki/live.md）",
                    "usage": type("Usage", (), {"total": lambda self: 7})(),
                })()

        class Tool:
            def __init__(self, fn):
                self.fn = fn

        class Registry:
            def get(self, name):
                if name == "rag_retrieve":
                    return Tool(lambda query, limit=8, envelope=False, metadata=False: calls.append("retrieve") or
                                '{"items":[{"ref":"wiki/live.md#h:1","status":"active","content":"fact"}],'
                                '"status":"available","strategy":"lexical-only",'
                                '"timings_ms":{"query_classify_ms":1.5,"fuse_ms":2.5},'
                                '"route_timings_ms":{"lexical":3.5,"vector":4.5,"context_expand":5.5},'
                                '"route_metadata":{"index_version":"s1-p2-v1",'
                                '"parser_version":"markdown-structure-v1",'
                                '"chunk_strategy_version":"markdown-structure-v1"}}')
                return Tool(self.assess)

            async def assess(self, query, items, **kwargs):
                calls.append("assess")
                return '{"action":"answer","answerability":"answerable","sufficient":true}'

        class Runner:
            registry = Registry()
            provider = Provider()

        task = _task("live", "exact_title")
        result = asyncio.run(_direct_run(object(), lambda _sink: Runner(), task, deadline_seconds=1))
        self.assertEqual(calls, ["retrieve", "assess", "answer"])
        self.assertEqual(result["probe_mode"], "direct_rag")
        self.assertEqual(result["retrieval_calls"], 1)
        self.assertEqual(result["assess_calls"], 1)
        self.assertEqual(result["answer_calls"], 1)
        self.assertEqual(result["tool_records"][0][0], "rag_retrieve")
        # The fake assessor does not call a provider; only answer generation
        # is a real model invocation in this test.
        self.assertEqual(result["llm_calls"], 1)
        self.assertEqual(result["budget"]["max_retrieval_calls"], 1)
        self.assertIn("retrieve", result["stage_timings"])
        self.assertEqual(result["stage_timings"]["query_classify"], 1.5)
        self.assertEqual(result["stage_timings"]["fuse"], 2.5)
        self.assertEqual(result["stage_timings"]["lexical"], 3.5)
        self.assertEqual(result["stage_timings"]["vector"], 4.5)
        self.assertEqual(result["stage_timings"]["context_expand"], 5.5)
        self.assertIn("generate", result["stage_timings"])
        self.assertEqual(result["index_version"], "s1-p2-v1")
        self.assertEqual(result["retrieval_status"], "available")
        self.assertEqual(result["retrieval_strategy"], "lexical-only")
        self.assertEqual(result["index_version"], "s1-p2-v1")
        self.assertEqual(result["retrieval_status"], "available")
        self.assertEqual(result["retrieval_strategy"], "lexical-only")

    def test_direct_mode_timeout_identifies_stage(self):
        class Registry:
            def get(self, name):
                if name == "rag_retrieve":
                    async def slow(*_args, **_kwargs):
                        await asyncio.sleep(0.05)
                    return type("Tool", (), {"fn": slow})()
                raise AssertionError("assess must not run after retrieve timeout")

        class Runner:
            registry = Registry()
            provider = object()

        result = asyncio.run(_direct_run(
            object(), lambda _sink: Runner(), _task("slow", "negative", answerability="absent"),
            deadline_seconds=0.001,
        ))
        self.assertEqual(result["timeout_stage"], "retrieve")
        self.assertIn("stage retrieve", result["error"])
        self.assertEqual(result["provider_error"], result["error"])

    def test_direct_mode_allows_explicit_candidate_limit_for_route_experiments(self):
        limits = []

        class Registry:
            def get(self, name):
                if name == "rag_retrieve":
                    return type("Tool", (), {
                        "fn": lambda self, query, limit=8, envelope=False, metadata=False:
                            limits.append(limit) or '{"items": []}'
                    })()
                return type("Tool", (), {
                    "fn": lambda self, **kwargs: '{"action":"insufficient","answerability":"absent"}'
                })()

        class Runner:
            registry = Registry()
            provider = None

        result = asyncio.run(_direct_run(
            object(), lambda _sink: Runner(),
            _task("limit", "negative", answerability="absent"),
            deadline_seconds=1, retrieval_limit=10,
        ))
        self.assertEqual(limits, [10])
        self.assertFalse(result["error"])
        self.assertEqual(result["answer"], "当前资料不足，无法确认。")
        self.assertEqual(result["answer_calls"], 0)

    def test_direct_mode_any_policy_never_requires_every_gold_ref(self):
        observed = {}

        class Provider:
            async def chat(self, *_args, **_kwargs):
                return type("Response", (), {
                    "content": "依据（ref: wiki/a.md）可回答。",
                    "usage": type("Usage", (), {"total": lambda self: 1})(),
                })()

        class Registry:
            def get(self, name):
                if name == "rag_retrieve":
                    return type("Tool", (), {
                        "fn": lambda *_args, **_kwargs:
                            '{"items":[{"ref":"wiki/a.md","content":"fact"}]}'
                    })()

                async def assess(*_args, **kwargs):
                    observed["required_refs"] = kwargs["required_refs"]
                    return '{"action":"answer","answerability":"answerable","sufficient":true}'

                return type("Tool", (), {"fn": assess, "llm_call_count": lambda: 0})()

        class Runner:
            registry = Registry()
            provider = Provider()

        task = _task("any", "theme", expected_refs=["wiki/a.md", "wiki/b.md"],
                     expected_policy="any")
        result = asyncio.run(_direct_run(object(), lambda _sink: Runner(), task, deadline_seconds=1))
        self.assertFalse(result["error"])
        self.assertEqual(observed["required_refs"], [])

    def test_selection_prioritises_negatives_then_diverse_positive_types(self):
        tasks = [
            _task("positive-a", "exact_title"),
            _task("negative-a", "negative", answerability="absent"),
            _task("positive-b", "paraphrase"),
            _task("negative-b", "negative", answerability="absent"),
            _task("positive-c", "exact_title"),
        ]
        selected = select_tasks(tasks, limit=4)
        self.assertEqual([task["id"] for task in selected[:2]], ["negative-a", "negative-b"])
        self.assertEqual({task["query_type"] for task in selected[2:]}, {"exact_title", "paraphrase"})

    def test_collection_records_live_evidence_and_resumes_completed_probe(self):
        task = _task("live", "exact_title")

        async def run_one(_task):
            return {
                "answer": "依据 [[wiki/live.md]] 可以回答。",
                "tool_records": [
                    ("rag_retrieve", '[{"ref":"wiki/live.md#h:ch1","status":"active"}]'),
                    ("rag_assess", '{"action":"answer","answerability":"answerable"}'),
                ],
                "trace_id": "trace-live",
                "stop_reason": "done",
                "tokens": 12,
            }

        report = asyncio.run(collect_answer_probes([task], run_one, limit=1, model="fake"))
        probe = report["probes"][0]
        self.assertEqual(probe["source"], "llm")
        self.assertEqual(probe["candidate_refs"], ["wiki/live.md#h:ch1"])
        self.assertEqual(probe["assessment"], "answer")
        self.assertEqual(report["summary"]["with_retrieval"], 1)

        async def should_not_run(_task):
            raise AssertionError("completed live probe must be reused")

        resumed = asyncio.run(collect_answer_probes(
            [task], should_not_run, limit=1, model="fake", resume=report["probes"],
        ))
        self.assertTrue(resumed["probes"][0]["resumed"])
        self.assertEqual(resumed["summary"]["resumed"], 1)

    def test_collection_retains_provider_failure_without_calling_it_real(self):
        task = _task("failed", "negative", answerability="absent")

        async def failed(_task):
            raise RuntimeError("provider unavailable")

        report = asyncio.run(collect_answer_probes([task], failed, limit=1))
        probe = report["probes"][0]
        self.assertEqual(probe["source"], "unavailable")
        self.assertIn("provider unavailable", probe["error"])
        self.assertEqual(report["summary"]["completed"], 0)

    def test_collection_repeats_each_task_with_stable_probe_ids(self):
        task = _task("repeat", "exact_title")
        calls = []

        async def run_one(_task):
            calls.append(1)
            return {"answer": "依据 [[wiki/repeat.md]]。", "tool_records": [],
                    "stop_reason": "done"}

        report = asyncio.run(collect_answer_probes(
            [task], run_one, limit=1, repeats=3,
        ))
        self.assertEqual(len(calls), 3)
        self.assertEqual([p["id"] for p in report["probes"]], [
            "repeat:live:1", "repeat:live:2", "repeat:live:3",
        ])
        self.assertEqual(report["summary"]["selected_runs"], 3)

    def test_collection_times_out_and_checkpoints_each_task(self):
        task = _task("slow", "negative", answerability="absent")
        checkpoints = []

        async def slow(_task):
            await asyncio.sleep(0.05)
            return {}

        report = asyncio.run(collect_answer_probes(
            [task], slow, limit=1, task_timeout_seconds=0.001,
            checkpoint=checkpoints.append,
        ))
        self.assertEqual(report["summary"]["failed"], 1)
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]["probes"][0]["source"], "unavailable")
        self.assertEqual(checkpoints[0]["probes"][0]["probe_mode"], "direct_rag")

    def test_live_run_uses_strict_read_only_runtime_flag(self):
        captured = {}

        async def fake_run_agent(*args, **kwargs):
            captured.update(kwargs)
            sink = args[4]
            sink.text("依据 [[wiki/live.md]] 可回答。")
            sink.tool_end("rag_retrieve", '[{"ref":"wiki/live.md#h:ch1"}]')
            return {"stop_reason": "done", "tokens": 1, "trace_id": "t"}

        task = _task("live", "exact_title")
        with patch("agentlab.runtime.serve._run_agent", new=fake_run_agent):
            result = asyncio.run(_live_run(object(), object(), task))
        self.assertTrue(captured["evaluation_read_only"])
        self.assertEqual(result["tool_records"][0][0], "rag_retrieve")
        self.assertIn("wiki/live.md#h:ch1", result["tool_records"][0][1])
        self.assertEqual(result["vault_root"], "")


if __name__ == "__main__":
    unittest.main()
