import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from agentlab.memory_lifecycle_drill import main, run_memory_lifecycle_drill


class TestMemoryLifecycleDrill(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self.tmp.name) / "vault"
        self.vault.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_vault_invalidation_removes_all_derived_routes(self):
        result = run_memory_lifecycle_drill(self.vault, run_id="memory-01")
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["stages"]["revoke"]["complete"])
        self.assertTrue(result["stages"]["correct"]["complete"])
        self.assertEqual(result["stages"]["correct"]["derived"]["session_range"], "invalidated")
        self.assertEqual(result["stages"]["correct"]["derived"]["task_state"], "invalidated")
        self.assertEqual(result["stages"]["candidate"]["status"], "candidate")
        self.assertEqual(result["stages"]["promote"]["status"], "active")
        self.assertTrue(result["stages"]["time_gates"]["all_hidden"])
        self.assertTrue(result["stages"]["conflict"]["hidden"])
        self.assertTrue(result["stages"]["cross_route"]["context_and_rag_gates"])
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
