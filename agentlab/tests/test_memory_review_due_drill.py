import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from agentlab.memory_review_due_drill import main, run_memory_review_due_drill


class TestMemoryReviewDueDrill(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self.tmp.name) / "vault"
        self.vault.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_vault_review_confirm_and_defer_cleanup(self):
        result = run_memory_review_due_drill(self.vault, run_id="review-01")
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["stages"]["queued"]["total"], 2)
        self.assertEqual(result["stages"]["settled_queue"]["total"], 0)
        self.assertTrue(all(result["cleanup"].values()))
        self.assertFalse((self.vault / ".agent-brain" / "drills").exists())

    def test_cli_requires_explicit_vault(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main([])
        self.assertEqual(code, 2)
        self.assertIn("--vault", json.loads(output.getvalue())["error"])


if __name__ == "__main__":
    unittest.main()
