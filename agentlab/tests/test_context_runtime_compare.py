import unittest

from agentlab.core.llm import LLMProvider, LLMResponse
from agentlab.core.message import TokenUsage
from agentlab.eval.context_runtime_compare import (
    DEFAULT_REPLAY_CASES,
    compare_replayed_runner_cases,
    compare_runner_cases,
    load_runtime_cases,
    run_context_fallback_drills,
)


class _Provider(LLMProvider):
    def __init__(self, answer: str):
        self.answer = answer

    async def chat(self, messages, tools=None, **kwargs):
        return LLMResponse(
            content=self.answer,
            tool_calls=[],
            stop_reason="stop",
            usage=TokenUsage(input_tokens=2, output_tokens=2),
        )


class TestContextRuntimeCompare(unittest.TestCase):
    def test_runner_metrics_are_measured_for_both_modes(self):
        report = compare_runner_cases(
            lambda _case, _mode: _Provider("完成"),
            [{"id": "case-1", "input": "任务", "instructions_text": "规则"}],
        )
        self.assertEqual(report["evidence_status"], "provider_runtime")
        self.assertEqual(report["cases"], 1)
        self.assertEqual(report["summary"]["runtime_regressions"], 0)
        row = report["per_case"][0]
        self.assertTrue(row["shadow"]["task_completion"])
        self.assertTrue(row["on"]["task_completion"])
        self.assertEqual(row["shadow"]["mode"], "shadow")
        self.assertEqual(row["on"]["mode"], "on")
        manifest = row["shadow"]["context_manifest"]
        self.assertEqual(manifest["selected"][0]["id"], "instruction:0")
        self.assertIn("text_hash", manifest["selected"][0])
        self.assertNotIn("text", manifest["selected"][0])
        self.assertEqual(len(row["shadow"]["context_plan_history"]), 1)
        self.assertIn("selected_changed", row["diff"])
        self.assertIn("stage_durations_ms", row["shadow"])
        self.assertIn("latency", report)
        self.assertIn("shadow", report["latency"])

    def test_fixed_replay_keeps_renderer_tool_and_checkpoint_contracts(self):
        report = compare_replayed_runner_cases(load_runtime_cases(DEFAULT_REPLAY_CASES))
        self.assertEqual(report["evidence_status"], "deterministic_replay")
        self.assertFalse(report["production_evidence"])
        self.assertEqual(report["cases"], 3)
        self.assertEqual(report["summary"]["replay_regression_cases"], 0)
        row = next(item for item in report["per_case"] if item["id"] == "fixed-tool-checkpoint")
        self.assertEqual(row["shadow"]["provider_requests"], row["on"]["provider_requests"])
        self.assertEqual(row["shadow"]["tool_invocations"], row["on"]["tool_invocations"])
        self.assertEqual(row["shadow"]["checkpoint"], row["on"]["checkpoint"])
        self.assertEqual(row["shadow"]["checkpoint"]["state_version"], 3)
        self.assertNotIn("content", row["shadow"]["provider_requests"][0]["messages"][0])

    def test_fixed_replay_filters_out_of_scope_fixture_before_model_input(self):
        report = compare_replayed_runner_cases(load_runtime_cases(DEFAULT_REPLAY_CASES))
        row = next(item for item in report["per_case"] if item["id"] == "scope-filter")
        for mode in ("shadow", "on"):
            invocation = row[mode]["tool_invocations"][0]
            self.assertEqual(invocation["visible_refs"], ["wiki/p1.md"])
            self.assertNotIn("wiki/p2-secret.md", invocation["visible_refs"])

    def test_fallback_drills_preserve_tool_and_checkpoint_contract(self):
        report = run_context_fallback_drills(load_runtime_cases(DEFAULT_REPLAY_CASES))
        self.assertEqual(report["schema"], "context-assembler-fallback-drill-v1")
        self.assertFalse(report["production_evidence"])
        self.assertEqual(report["summary"], {"passed": 3, "failed": 0})
        row = report["per_case"][0]
        self.assertEqual(row["switch_to_shadow"]["mode_history"], ["on", "shadow"])
        self.assertTrue(row["switch_to_shadow"]["tool_contract_same"])
        self.assertTrue(row["resume_from_checkpoint"]["checkpoint_same"])


if __name__ == "__main__":
    unittest.main()
