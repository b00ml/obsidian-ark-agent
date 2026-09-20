import json
import unittest
from pathlib import Path

from agentlab.eval.memory_governance import run_memory_eval


class TestMemoryGovernanceEval(unittest.TestCase):
    def test_frozen_governance_and_redteam_fixture_is_clean(self):
        dataset = Path(__file__).resolve().parents[2] / ".ai" / "evals" / "memory_governance-v1.jsonl"
        result = run_memory_eval(dataset)
        self.assertEqual(result["failed"], 0, json.dumps(result, ensure_ascii=False))
        self.assertEqual(result["pass_rate"], 1.0)
        self.assertEqual(result["redteam"]["pass_rate"], 1.0)
        self.assertGreaterEqual(result["redteam"]["total"], 5)


if __name__ == "__main__":
    unittest.main()
