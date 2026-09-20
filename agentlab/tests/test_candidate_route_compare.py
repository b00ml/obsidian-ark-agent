import unittest

from agentlab.eval.candidate_route_compare import compare_candidate_routes


class TestCandidateRouteCompare(unittest.TestCase):
    def test_reports_any_and_all_hits_per_route(self):
        tasks = [{
            "id": "q", "query": "q", "expected_refs": ["note.md", "other.md"],
            "query_type": "episodic_semantic",
        }]
        report = compare_candidate_routes({
            "per_query": [{
                "id": "q",
                "p2_lexical": {"status": "available", "refs": ["note.md#h:ch1"]},
                "p2_vector": {"status": "available", "refs": ["note.md#h:ch1", "other.md#h:ch2"]},
                "rrf": {"status": "available", "refs": []},
            }],
        }, tasks)
        routes = report["cases"][0]["routes"]
        self.assertTrue(routes["p2_lexical"]["any_expected_ref_hit"])
        self.assertFalse(routes["p2_lexical"]["all_expected_refs_hit"])
        self.assertTrue(routes["p2_vector"]["all_expected_refs_hit"])
        self.assertFalse(routes["rrf"]["any_expected_ref_hit"])

    def test_not_applicable_route_is_visible_and_negative_has_no_hit(self):
        tasks = [{"id": "fixture", "query": "q", "expected_refs": ["a.md"],
                  "query_type": "isolation"},
                 {"id": "negative", "query": "q", "expected_refs": [],
                  "query_type": "negative"}]
        report = compare_candidate_routes({
            "per_query": [{
                "id": "fixture",
                "p2_lexical": None,
                "rrf": {"status": "not_applicable"},
            }, {"id": "negative", "rrf": {"status": "available", "refs": ["a.md"]}}],
        }, tasks, routes=("p2_lexical", "rrf"))
        fixture = report["cases"][0]["routes"]
        self.assertEqual(fixture["p2_lexical"]["status"], "missing")
        self.assertEqual(fixture["rrf"]["status"], "not_applicable")
        self.assertIsNone(report["cases"][1]["routes"]["rrf"]["any_expected_ref_hit"])
        summary = report["summary"]["by_query_type"]["isolation"]["routes"]["rrf"]
        self.assertEqual(summary["any_expected_ref_hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
