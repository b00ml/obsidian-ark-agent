import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agentlab.runtime.task_state import (
    TaskStateConflict,
    TaskStateError,
    TaskStateStore,
)


class TestTaskStateStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TaskStateStore(Path(self.tmp.name) / "state.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_checkpoint_roundtrip_and_optimistic_version(self):
        state = self.store.ensure("task-1", session_id="s1", project_id="p1",
                                  core_intent={"goal": "报告"})
        self.assertEqual(state.state_version, 0)
        state = self.store.patch("task-1", {
            "phase": "PLANNING", "current_subtask": "检索", "todo": [{"id": "T1"}],
        }, expected_version=0)
        self.assertEqual(state.state_version, 1)
        with self.assertRaises(TaskStateConflict):
            self.store.patch("task-1", {"current_subtask": "过期写入"}, expected_version=0)
        reopened = TaskStateStore(self.store.path)
        got = reopened.get("task-1")
        self.assertEqual(got.current_subtask, "检索")
        self.assertEqual(got.project_id, "p1")

    def test_legal_transitions_are_audited(self):
        self.store.ensure("task-2")
        self.store.transition("task-2", "PLANNING", reason="start")
        self.store.transition("task-2", "EXECUTING", reason="plan ready")
        self.store.transition("task-2", "DONE", reason="verified")
        events = self.store.events("task-2")
        self.assertEqual([e["to_phase"] for e in events], ["PLANNING", "EXECUTING", "DONE"])

    def test_illegal_transition_and_unknown_field_fail_closed(self):
        self.store.ensure("task-3")
        with self.assertRaises(TaskStateError):
            self.store.transition("task-3", "DONE")
        with self.assertRaises(TaskStateError):
            self.store.patch("task-3", {"not_a_state_field": True})

    def test_scope_mismatch_cannot_reuse_recovery_key(self):
        self.store.ensure("task-scope", session_id="s1", project_id="p1")
        with self.assertRaises(TaskStateConflict):
            self.store.ensure("task-scope", session_id="s1", project_id="p2")
        with self.assertRaises(TaskStateConflict):
            self.store.ensure("task-scope", session_id="s2", project_id="p1")

    def test_save_cannot_bypass_phase_transition_rules(self):
        state = self.store.ensure("task-save")
        state.phase = "DONE"
        with self.assertRaises(TaskStateError):
            self.store.save(state, expected_version=0)

    def test_waiting_user_resume_has_explicit_path(self):
        self.store.ensure("task-resume")
        self.store.transition("task-resume", "PLANNING")
        self.store.transition("task-resume", "WAITING_USER")
        resumed = self.store.transition("task-resume", "PLANNING")
        self.assertEqual(resumed.phase, "PLANNING")
        self.assertEqual(self.store.transition("task-resume", "EXECUTING").phase, "EXECUTING")

    def test_side_effect_operation_is_planned_and_settled(self):
        state = self.store.ensure("task-op", session_id="s", project_id="p")
        state = self.store.plan_tool(
            "task-op", operation_id="op-1", tool_name="vault_write",
            arguments_hash="abc", permission="write", side_effects="write",
            idempotent=True,
        )
        self.assertEqual(state.pending_tools[0]["status"], "planned")
        state = self.store.update_tool("task-op", "op-1", "running", lease_seconds=30)
        self.assertEqual(state.pending_tools[0]["status"], "running")
        state = self.store.update_tool("task-op", "op-1", "succeeded", result_ref="file.md")
        self.assertEqual(state.pending_tools[0]["status"], "succeeded")
        self.assertEqual(self.store.pending_operations("task-op"), [])

    def test_stale_side_effect_becomes_unknown_and_is_not_retried(self):
        self.store.ensure("task-unknown")
        self.store.plan_tool("task-unknown", operation_id="op-unknown",
                             tool_name="send", permission="danger",
                             side_effects="external", idempotent=False)
        state = self.store.get("task-unknown")
        state.pending_tools[0]["updated_at"] = (
            datetime.now(timezone.utc) - timedelta(hours=1)
        ).isoformat()
        self.store.save(state, expected_version=state.state_version)
        recovered = self.store.recover_pending_tools("task-unknown", stale_after_seconds=60)
        self.assertEqual([row["operation_id"] for row in recovered], ["op-unknown"])
        self.assertEqual(self.store.pending_operations("task-unknown")[0]["status"], "unknown")

    def test_unknown_operation_requires_external_reconciliation_evidence(self):
        self.store.ensure("task-reconcile")
        for outcome in ("succeeded", "failed", "cancelled"):
            operation_id = f"op-{outcome}"
            self.store.plan_tool(
                "task-reconcile", operation_id=operation_id, tool_name="external_write",
                permission="danger", side_effects="external", idempotent=False,
            )
            self.store.update_tool("task-reconcile", operation_id, "unknown")
            state = self.store.reconcile_operation(
                "task-reconcile", operation_id,
                external_result={
                    "status": outcome,
                    "source": "remote-api-status",
                    "evidence_ref": f"remote:operations/{operation_id}",
                    "result_ref": f"artifact:{operation_id}",
                },
            )
            row = next(item for item in state.pending_tools if item["operation_id"] == operation_id)
            self.assertEqual(row["status"], outcome)
            self.assertEqual(row["reconciliation"]["evidence_ref"],
                             f"remote:operations/{operation_id}")

        before = self.store.get("task-reconcile")
        with self.assertRaises(TaskStateError):
            self.store.reconcile_operation(
                "task-reconcile", "op-succeeded",
                external_result={
                    "status": "succeeded", "source": "remote-api-status",
                    "evidence_ref": "remote:operations/op-succeeded",
                },
            )
        with self.assertRaises(TaskStateError):
            self.store.reconcile_operation(
                "task-reconcile", "op-failed",
                external_result={"status": "succeeded", "source": "remote-api-status"},
            )
        self.assertEqual(self.store.get("task-reconcile").state_version, before.state_version)

    def test_update_tool_cannot_settle_unknown_operation(self):
        self.store.ensure("task-no-bypass")
        self.store.plan_tool("task-no-bypass", operation_id="op-1", tool_name="write",
                             permission="write", side_effects="write", idempotent=False)
        self.store.update_tool("task-no-bypass", "op-1", "unknown")
        with self.assertRaises(TaskStateError):
            self.store.update_tool("task-no-bypass", "op-1", "succeeded")


if __name__ == "__main__":
    unittest.main()
