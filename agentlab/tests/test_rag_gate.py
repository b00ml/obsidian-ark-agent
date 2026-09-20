import unittest

from agentlab.eval.rag_gate import assess


def _report(*, vector_status="available", rrf_status="available", rrf_recall=0.62,
            rrf_p95=120.0, clean=0.5, failures=0, answer_gate=None):
    return {
        "summary": {
            "p2_lexical": {"status": "available", "tasks": 10, "recall@5": 0.60},
            "p2_vector": {"status": vector_status, "tasks": 10},
            "rrf": {
                "status": rrf_status, "tasks": 10,
                "recall@5": rrf_recall, "p95_ms": rrf_p95,
            },
            "negative": {
                "p2_lexical": {"clean@5": 0.5},
                "rrf": {"clean@5": clean},
            },
            "forbidden": {"hits@5": 0},
        },
        "meta": {"index": {"failures": failures, "coverage": 1.0}},
        "answer_gate": answer_gate or {"summary": {"answer_gate": {
            "status": "available", "cases": 24, "real_cases": 24,
            "false_answer_rate": 0.0, "false_refusal_rate": 0.0,
            "citation_out_of_scope_rate": 0.0, "forbidden_hit_rate": 0.0,
        }}},
    }


class TestRagGate(unittest.TestCase):
    def test_candidate_is_eligible_only_for_manual_gray(self):
        result = assess(_report(), _report())
        self.assertEqual(result["decision"], "eligible_for_manual_gray")
        self.assertEqual(result["production_switch"], "manual_only")
        self.assertTrue(all(item["passed"] for item in result["checks"].values()))

    def test_unavailable_vector_blocks_without_false_zero_score(self):
        result = assess(_report(), _report(vector_status="unavailable", rrf_status="unavailable"))
        self.assertEqual(result["decision"], "blocked")
        self.assertFalse(result["checks"]["vector_available"]["passed"])

    def test_latency_and_failures_block_gray(self):
        result = assess(_report(), _report(rrf_p95=501, failures=1))
        self.assertEqual(result["decision"], "blocked")
        self.assertFalse(result["checks"]["rrf_p95"]["passed"])
        self.assertFalse(result["checks"]["index_failures_zero"]["passed"])

    def test_lexical_fallback_and_coverage_are_hard_requirements(self):
        candidate = _report()
        candidate["summary"]["p2_lexical"]["status"] = "unavailable"
        candidate["meta"]["index"]["coverage"] = 0.99
        result = assess(_report(), candidate)
        self.assertEqual(result["decision"], "blocked")
        self.assertFalse(result["checks"]["lexical_available"]["passed"])
        self.assertFalse(result["checks"]["index_coverage"]["passed"])

    def test_not_applicable_fixture_does_not_block_canonical_routes(self):
        candidate = _report()
        for name in ("p2_lexical", "p2_vector", "rrf"):
            candidate["summary"][name]["status"] = "mixed"
            candidate["summary"][name]["status_counts"] = {
                "available": 10,
                "not_applicable": 1,
            }
        result = assess(_report(), candidate)
        self.assertEqual(result["decision"], "eligible_for_manual_gray")
        self.assertTrue(result["checks"]["rrf_available"]["passed"])

    def test_real_unavailable_route_inside_mixed_status_blocks(self):
        candidate = _report()
        candidate["summary"]["rrf"]["status"] = "mixed"
        candidate["summary"]["rrf"]["status_counts"] = {
            "available": 9,
            "unavailable": 1,
        }
        result = assess(_report(), candidate)
        self.assertEqual(result["decision"], "blocked")
        self.assertFalse(result["checks"]["rrf_available"]["passed"])

    def test_synthetic_answer_probes_cannot_unlock_gray(self):
        candidate = _report(answer_gate={"summary": {"answer_gate": {
            "status": "available", "cases": 24, "real_cases": 0,
            "synthetic_cases": 24, "false_answer_rate": 0.0,
            "false_refusal_rate": 0.0, "citation_out_of_scope_rate": 0.0,
            "forbidden_hit_rate": 0.0,
        }}})
        result = assess(_report(), candidate)
        self.assertEqual(result["decision"], "blocked")
        self.assertFalse(result["checks"]["answer_gate_real_cases"]["passed"])

    def test_answer_safety_failure_blocks_gray(self):
        candidate = _report(answer_gate={"summary": {"answer_gate": {
            "status": "available", "cases": 24, "real_cases": 24,
            "false_answer_rate": 0.1, "false_refusal_rate": 0.0,
            "citation_out_of_scope_rate": 0.0, "forbidden_hit_rate": 0.0,
        }}})
        result = assess(_report(), candidate)
        self.assertEqual(result["decision"], "blocked")
        self.assertFalse(result["checks"]["answer_false_rate"]["passed"])


if __name__ == "__main__":
    unittest.main()
