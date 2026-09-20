import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agentlab.task_state_recovery_drill import main, run_recovery_drill


class TestTaskStateRecoveryDrill(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name) / "workspace"
        self.workspace.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_reopens_unknown_and_reconciles_external_success(self):
        result = run_recovery_drill(self.workspace, run_id="recovery-01")

        self.assertTrue(result["passed"])
        self.assertEqual(result["stages"]["recovered"]["status"], "unknown")
        self.assertEqual(result["stages"]["reconciled"]["status"], "succeeded")
        self.assertFalse(result["stages"]["recovered"]["replayed"])
        self.assertTrue(all(result["cleanup"].values()))
        self.assertFalse((self.workspace / ".agentlab-task-drill-recovery-01").exists())

    def test_cli_requires_workspace_and_writes_report(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main([])
        self.assertEqual(code, 2)
        self.assertIn("--workspace", json.loads(output.getvalue())["error"])

        report = Path(self.tmp.name) / "recovery.json"
        output = io.StringIO()
        with redirect_stdout(output):
            code = main([
                "--workspace", str(self.workspace),
                "--run-id", "recovery-02",
                "--out", str(report),
            ])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(report.read_text(encoding="utf-8"))["passed"])

    def test_failure_still_removes_private_artifacts(self):
        with patch(
            "agentlab.task_state_recovery_drill.TaskStateStore.reconcile_operation",
            side_effect=RuntimeError("simulated verifier failure"),
        ):
            result = run_recovery_drill(self.workspace, run_id="recovery-failure")

        self.assertFalse(result["passed"])
        self.assertTrue(all(result["cleanup"].values()))
        self.assertFalse((self.workspace / ".agentlab-task-drill-recovery-failure").exists())


if __name__ == "__main__":
    unittest.main()
