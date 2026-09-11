import tempfile
import unittest
from pathlib import Path

from agentlab.eval.behavior import load_behavior_scenarios, run_behavior_baseline


class TestBehaviorBaseline(unittest.TestCase):
    def test_scenarios_are_declared_and_unique(self):
        scenarios = load_behavior_scenarios()
        ids = [item["id"] for item in scenarios]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("memory-second-recall", ids)
        self.assertIn("serve-sse-contract", ids)

    def test_offline_baseline_passes(self):
        report = run_behavior_baseline()
        self.assertEqual(report["schema"], "f5-012.v1")
        self.assertEqual(report["summary"]["failed"], 0)
        self.assertEqual(report["summary"]["pending"], 0)
        # F5-020：在原 7 条基础上新增 3 条多 Agent 场景（降级 / 并发上限 / @ 门禁）
        self.assertEqual(report["summary"]["passed"], 10)
        for scenario in report["scenarios"]:
            self.assertIn("evidence", scenario)

    def test_report_can_be_written(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "baseline.json"
            report = run_behavior_baseline(path)
            self.assertTrue(path.exists())
            self.assertEqual(report["summary"]["failed"], 0)


if __name__ == "__main__":
    unittest.main()
