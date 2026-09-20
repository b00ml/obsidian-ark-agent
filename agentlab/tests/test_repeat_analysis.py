import unittest

from agentlab.eval.repeat_analysis import summarize_repeated_cases


class TestRepeatAnalysis(unittest.TestCase):
    def test_groups_live_runs_and_identifies_stable_attribution(self):
        report = summarize_repeated_cases([
            {"id": "R1:live:1", "task_id": "R1", "real_probe": True,
             "source": "llm", "expected_decision": "answerable",
             "predicted_abstention": True, "candidate_expected_ref_any": False,
             "candidate_expected_ref_all": False, "error_attribution": "retrieval_miss",
             "assessment": "insufficient"},
            {"id": "R1:live:2", "task_id": "R1", "real_probe": True,
             "source": "llm", "expected_decision": "answerable",
             "predicted_abstention": True, "candidate_expected_ref_any": False,
             "candidate_expected_ref_all": False, "error_attribution": "retrieval_miss",
             "assessment": "insufficient"},
            {"id": "N1:live:1", "task_id": "N1", "real_probe": True,
             "source": "llm", "expected_decision": "insufficient",
             "predicted_abstention": True, "assessment": "insufficient"},
        ])
        self.assertEqual(report["tasks"], 2)
        row = next(row for row in report["rows"] if row["task_id"] == "R1")
        self.assertTrue(row["retrieval_miss_stable"])
        self.assertFalse(row["retrieval_content_gap_stable"])
        self.assertTrue(row["error_attribution"]["stable"])
        self.assertEqual(row["candidate_expected_ref_any"]["majority"], "False")

    def test_failed_run_is_not_real_evidence(self):
        report = summarize_repeated_cases([
            {"id": "R1:live:1", "task_id": "R1", "real_probe": True,
             "source": "llm", "expected_decision": "answerable",
             "predicted_abstention": True, "error_attribution": "retrieval_miss"},
            {"id": "R1:live:2", "task_id": "R1", "real_probe": False,
             "source": "unavailable", "expected_decision": "answerable",
             "error": "provider error"},
        ])
        row = report["rows"][0]
        self.assertEqual(row["real_runs"], 1)
        self.assertEqual(row["failed_runs"], 1)
        self.assertTrue(row["retrieval_miss_stable"])

    def test_content_gap_is_stable_when_every_real_run_is_thin(self):
        report = summarize_repeated_cases([
            {"id": "R2:live:1", "task_id": "R2", "real_probe": True,
             "source": "llm", "expected_decision": "answerable",
             "predicted_abstention": True, "candidate_expected_ref_any": True,
             "candidate_expected_ref_all": True,
             "candidate_expected_ref_contentful": False,
             "error_attribution": "retrieval_content_gap"},
            {"id": "R2:live:2", "task_id": "R2", "real_probe": True,
             "source": "llm", "expected_decision": "answerable",
             "predicted_abstention": True, "candidate_expected_ref_any": True,
             "candidate_expected_ref_all": True,
             "candidate_expected_ref_contentful": False,
             "error_attribution": "retrieval_content_gap"},
        ])
        row = report["rows"][0]
        self.assertTrue(row["retrieval_content_gap_stable"])
        self.assertEqual(report["stable_positive_retrieval_content_gap_tasks"], 1)


if __name__ == "__main__":
    unittest.main()
