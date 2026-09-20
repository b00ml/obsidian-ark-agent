import json
import tempfile
import unittest
from pathlib import Path

from agentlab.eval.context_assembler_compare import (
    compare_context_plans,
    load_cases,
    main,
)


class TestContextAssemblerCompare(unittest.TestCase):
    def test_fixed_cases_have_no_planner_regression(self):
        path = Path(__file__).resolve().parents[2] / ".ai" / "evals" / "context_assembler-v1.jsonl"
        report = compare_context_plans(load_cases(path))
        self.assertEqual(report["cases"], 4)
        self.assertEqual(report["summary"]["planner_regressions"], 0)
        self.assertEqual(report["summary"]["selected_same"], 4)
        self.assertGreaterEqual(report["summary"]["scope_denied"], 1)
        self.assertEqual(report["runtime_metrics"]["tool_calls"], "not_measured")

    def test_cli_writes_report_and_rejects_planner_regression(self):
        cases = Path(tempfile.mkdtemp()) / "cases.jsonl"
        cases.write_text(json.dumps({"id": "one", "instructions": [{"content": "规则"}]}) + "\n",
                          encoding="utf-8")
        report = cases.with_name("report.json")
        self.assertEqual(main(["--cases", str(cases), "--out", str(report)]), 0)
        self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["cases"], 1)


if __name__ == "__main__":
    unittest.main()
