import tempfile
import unittest
from pathlib import Path

from agentlab.eval.task_audit import audit_tasks


class TestTaskAudit(unittest.TestCase):
    def test_expected_ref_and_empty_negative_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "wiki").mkdir()
            (root / "wiki" / "fact.md").write_text("# Fact\n正文", encoding="utf-8")
            report = audit_tasks([
                {"id": "positive", "query": "fact", "expected_refs": ["wiki/fact.md"],
                 "answerability": "answerable"},
                {"id": "negative", "query": "不存在的事实", "expected_refs": [],
                 "answerability": "absent"},
            ], vault_root=root, project_root=root)
            self.assertTrue(report["valid"])
            self.assertEqual(report["refs_checked"], 1)

    def test_negative_label_conflict_is_warning_and_strict_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.md").write_text("用户偏好深色主题。", encoding="utf-8")
            report = audit_tasks([{
                "id": "negative", "query": "用户偏好深色主题", "expected_refs": [],
                "answerability": "absent",
            }], vault_root=root, project_root=root)
            self.assertTrue(report["valid"])
            self.assertEqual(report["summary"]["negative_label_conflicts"], 1)
            self.assertEqual(report["warnings"][0]["kind"], "negative_label_conflict")

    def test_missing_expected_ref_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = audit_tasks([{
                "id": "missing", "query": "x", "expected_refs": ["wiki/missing.md"],
                "answerability": "answerable",
            }], vault_root=tmp, project_root=tmp)
            self.assertFalse(report["valid"])
            self.assertEqual(report["errors"][0]["kind"], "missing_expected_ref")


if __name__ == "__main__":
    unittest.main()
