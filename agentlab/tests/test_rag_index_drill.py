import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agentlab.rag_index_drill import main, run_incremental_drill


class TestRagIndexDrill(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vault = Path(self.tmp.name) / "vault"
        self.vault.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_exercises_incremental_lifecycle_and_removes_artifacts(self):
        result = run_incremental_drill(self.vault, run_id="drill-01")

        self.assertTrue(result["passed"])
        self.assertLess(result["elapsed_seconds"], 300)
        self.assertEqual(result["stages"]["create"]["sync"]["updated"], 1)
        self.assertEqual(result["stages"]["modify"]["plan"]["upserts"], 1)
        self.assertEqual(result["stages"]["delete"]["plan"]["deletes"], 1)
        self.assertEqual(result["stages"]["delete"]["sync"]["removed"], 1)
        self.assertTrue(result["cleanup"]["source_removed"])
        self.assertTrue(result["cleanup"]["index_removed"])
        self.assertFalse((self.vault / "Inbox" / ".agentlab-drill-drill-01").exists())

    def test_cli_requires_explicit_vault_and_writes_report(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main([])
        self.assertEqual(code, 2)
        self.assertIn("--vault", json.loads(output.getvalue())["error"])

        report = Path(self.tmp.name) / "report.json"
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--vault", str(self.vault), "--run-id", "drill-02", "--out", str(report)])
        self.assertEqual(code, 0)
        self.assertTrue(report.exists())
        self.assertTrue(json.loads(report.read_text(encoding="utf-8"))["passed"])

    def test_failure_still_removes_controlled_source_and_index(self):
        with patch(
            "agentlab.rag_index_drill.RagIndexStore.sync_vault",
            side_effect=RuntimeError("simulated index failure"),
        ):
            result = run_incremental_drill(self.vault, run_id="drill-failure")

        self.assertFalse(result["passed"])
        self.assertTrue(result["cleanup"]["source_removed"])
        self.assertTrue(result["cleanup"]["source_directory_removed"])
        self.assertTrue(result["cleanup"]["index_removed"])


if __name__ == "__main__":
    unittest.main()
