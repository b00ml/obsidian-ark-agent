import unittest

from agentlab.contracts import ProcessStatus
from agentlab.runtime.stages import StageTracker


class TestStageTracker(unittest.TestCase):
    def test_success_and_serialisation(self):
        tracker = StageTracker("run-1", "attempt-1")
        with tracker.stage("s1", "parse") as stage:
            stage.warnings = ["minor"]
        self.assertEqual(tracker.results[0].status, ProcessStatus.COMPLETED)
        self.assertIsNotNone(tracker.results[0].ended_at)
        self.assertEqual(tracker.to_dict()[0]["metadata"]["run_id"], "run-1")

    def test_failure_infers_retryability_and_preserves_error(self):
        tracker = StageTracker("run-1", "attempt-1")
        stage = tracker.start("s1", "fetch")
        tracker.fail(stage, RuntimeError("timeout"), error_code="TIMEOUT")
        self.assertEqual(stage.status, ProcessStatus.FAILED)
        self.assertTrue(stage.retryable)
        self.assertEqual(stage.error_code, "TIMEOUT")

    def test_cancelled_stage_is_terminal(self):
        tracker = StageTracker("run-1", "attempt-1")
        stage = tracker.start("s1", "fetch")
        tracker.cancel(stage)
        self.assertEqual(stage.status, ProcessStatus.CANCELLED)
        self.assertFalse(stage.retryable)

    def test_record_keeps_operation_identity_and_terminal_stage(self):
        tracker = StageTracker("run-1", "attempt-1", operation_id="operation-1")
        tracker.record("indexed", "indexed", status=ProcessStatus.INDEXED,
                       artifact_refs=["rag:index"])
        row = tracker.to_dict()[0]
        self.assertEqual(row["status"], "indexed")
        self.assertEqual(row["artifact_refs"], ["rag:index"])
        self.assertEqual(row["metadata"]["operation_id"], "operation-1")


if __name__ == "__main__":
    unittest.main()
