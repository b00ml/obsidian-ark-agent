import asyncio
import unittest

from agentlab.core.planning import Plan, PlanBuilder, PlanExecutor, PlanStep


class PlanningTests(unittest.TestCase):
    def test_simple_question_does_not_add_planning_call(self):
        self.assertIsNone(PlanBuilder().build("什么是 RAG？", available_capabilities=("retrieve",)))

    def test_complex_plan_has_bounded_dependencies(self):
        plan = PlanBuilder().build(
            "研究并整理 RAG 更新策略，然后生成草稿",
            {"project_id": "p1"}, ("retrieve", "assess", "write"), explicit=True,
            max_steps=4,
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual([s.step_id for s in plan.steps], ["retrieve", "analyze", "write"])
        self.assertEqual(plan.steps[1].dependencies, ["retrieve"])

    def test_executor_runs_ready_steps_and_checkpoints(self):
        plan = Plan(
            "p", "complex", steps=[
                PlanStep("a", "collect", expected_outputs=["evidence"]),
                PlanStep("b", "write", dependencies=["a"]),
            ]
        )
        calls = []
        checkpoints = []

        async def run(step):
            calls.append(step.step_id)
            return {"evidence_refs": [f"ref:{step.step_id}"]}

        result = asyncio.run(PlanExecutor().execute(plan, run, checkpoint=lambda p: checkpoints.append(p.status)))
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(calls, ["a", "b"])
        self.assertGreaterEqual(len(checkpoints), 3)

    def test_missing_evidence_is_needs_review_and_blocks(self):
        plan = Plan("p", "complex", steps=[PlanStep("a", "collect")])
        result = asyncio.run(PlanExecutor().execute(plan, lambda _: {}))
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.steps[0].status, "needs_review")

    def test_cancel_and_timeout_are_recoverable(self):
        plan = Plan("p", "complex", steps=[PlanStep("a", "slow")])
        cancel = asyncio.Event()
        cancel.set()
        result = asyncio.run(PlanExecutor().execute(plan, lambda _: {"evidence": ["x"]}, cancel_event=cancel))
        self.assertEqual(result.status, "cancelled")

        plan2 = Plan("p2", "complex", steps=[PlanStep("a", "slow")])

        async def slow(_):
            await asyncio.sleep(0.05)

        result2 = asyncio.run(PlanExecutor(step_timeout_seconds=0.01).execute(plan2, slow))
        self.assertEqual(result2.steps[0].status, "blocked")
        self.assertEqual(result2.steps[0].result["error"], "step_timeout")


if __name__ == "__main__":
    unittest.main()
