import asyncio
import unittest
from datetime import datetime, timedelta, timezone

from agentlab.contracts import RetrievalScope
from agentlab.core.context_assembler import ContextAssembler, ContextCandidate
from agentlab.core.agent import Agent
from agentlab.core.llm import LLMProvider, LLMResponse
from agentlab.core.loop import RunConfig, Runner
from agentlab.core.message import TokenUsage


class TestContextAssembler(unittest.TestCase):
    def test_required_task_state_survives_budget_and_plan_is_serializable(self):
        assembler = ContextAssembler(budget_tokens=5, reserve_output_tokens=0)
        plan = assembler.assemble(
            scope=RetrievalScope(project_id="p1", session_id="s1"),
            task_state={
                "core_intent": {"goal": "完成报告", "constraints": ["不得修改 raw/"]},
                "current_subtask": "运行回归测试",
                "todo": [{"id": "T1", "status": "in_progress"}],
            },
            retrieval_items=[
                {"ref": "p1/a", "content": "证据 " * 20, "project_id": "p1"},
                {"ref": "p2/a", "content": "跨项目数据", "project_id": "p2"},
            ],
        )
        self.assertTrue(any(item.id == "task:goal" for item in plan.selected))
        self.assertTrue(any(item.id == "task:constraints" for item in plan.selected))
        self.assertIn("required_budget_overflow", plan.warnings)
        self.assertTrue(any(item["reason"] == "scope_denied" for item in plan.omitted))
        self.assertEqual(plan.to_dict()["scope"]["project_id"], "p1")

    def test_inactive_memory_is_not_selected_and_soft_items_are_bounded(self):
        assembler = ContextAssembler(budget_tokens=30, reserve_output_tokens=0)
        plan = assembler.assemble(
            memory_candidates=[
                {"id": "candidate", "content": "未确认", "status": "candidate"},
                {"id": "active", "content": "已确认", "status": "active", "score": 2},
                {"id": "other", "content": "x" * 500, "status": "active", "score": 1},
            ]
        )
        self.assertNotIn("candidate", {item.id for item in plan.selected})
        self.assertIn("active", {item.id for item in plan.selected})
        self.assertTrue(any(item["reason"] == "inactive:candidate" for item in plan.omitted))
        self.assertIn("context_items_omitted", plan.warnings)

    def test_conflicting_memory_is_not_selected(self):
        assembler = ContextAssembler(budget_tokens=30, reserve_output_tokens=0)
        plan = assembler.assemble(memory_candidates=[
            {"id": "conflict", "content": "相反事实", "status": "conflict"},
        ])
        self.assertNotIn("conflict", {item.id for item in plan.selected})
        self.assertTrue(any(item["reason"] == "inactive:conflict" for item in plan.omitted))

    def test_memory_effective_lifecycle_is_filtered_even_with_active_stored_status(self):
        assembler = ContextAssembler(budget_tokens=50, reserve_output_tokens=0)
        plan = assembler.assemble(memory_candidates=[
            {"id": "expired", "content": "过期事实", "status": "active",
             "valid_until": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()},
            {"id": "future", "content": "未来事实", "status": "active",
             "valid_from": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()},
            {"id": "due", "content": "待复核事实", "status": "active",
             "review_due_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()},
            {"id": "active", "content": "当前事实", "status": "active"},
        ])
        selected = {item.id for item in plan.selected}
        self.assertEqual(selected, {"active"})
        reasons = {item["id"]: item["reason"] for item in plan.omitted}
        self.assertEqual(reasons["expired"], "inactive:expired")
        self.assertEqual(reasons["future"], "inactive:not_yet_valid")
        self.assertEqual(reasons["due"], "inactive:review_due")

    def test_runner_refreshes_shadow_plan_without_changing_answer(self):
        class Provider(LLMProvider):
            async def chat(self, messages, tools=None, **kwargs):
                return LLMResponse(
                    content="答案", tool_calls=[], stop_reason="stop",
                    usage=TokenUsage(input_tokens=1, output_tokens=1),
                )

        assembler = ContextAssembler(budget_tokens=100)
        cfg = RunConfig(context_assembler=assembler, context_assembler_mode="shadow")
        result = asyncio.run(Runner(Provider()).run(
            Agent(name="test", instructions="规则", tools=[]), "问题", cfg=cfg,
        ))
        self.assertEqual(result.final_output, "答案")
        self.assertIsNotNone(cfg.context_plan)
        self.assertEqual(cfg.context_plan.mode, "shadow")

    def test_malformed_candidate_metadata_is_normalized(self):
        assembler = ContextAssembler(budget_tokens=20, reserve_output_tokens=0)
        plan = assembler.assemble(memory_candidates=[
            {"id": "nan", "content": "A", "score": "nan", "token_cost": "bad"},
            {"id": "negative", "content": "B", "score": "nope", "token_cost": -4},
        ])
        selected = {item.id: item for item in plan.selected}
        self.assertEqual(selected["nan"].score, 0.0)
        self.assertGreaterEqual(selected["nan"].tokens, 1)
        self.assertEqual(selected["negative"].score, 0.0)
        self.assertGreaterEqual(selected["negative"].tokens, 1)
        direct = ContextCandidate("direct", "external", "C", score="nan", token_cost="bad")
        self.assertEqual(direct.score, 0.0)
        self.assertEqual(direct.tokens, 1)

    def test_runner_plan_includes_structured_task_state(self):
        class Provider(LLMProvider):
            async def chat(self, messages, tools=None, **kwargs):
                return LLMResponse(
                    content="答案", tool_calls=[], stop_reason="stop",
                    usage=TokenUsage(input_tokens=1, output_tokens=1),
                )

        cfg = RunConfig(
            context_assembler=ContextAssembler(budget_tokens=100, reserve_output_tokens=0),
            context_assembler_mode="shadow",
            task_state={"core_intent": {"goal": "保留任务目标"}},
        )
        asyncio.run(Runner(Provider()).run(
            Agent(name="test", instructions="规则", tools=[]), "问题", cfg=cfg,
        ))
        self.assertIn("task:goal", {item.id for item in cfg.context_plan.selected})


if __name__ == "__main__":
    unittest.main()
