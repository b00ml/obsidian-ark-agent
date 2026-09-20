import unittest

from agentlab.eval.query_rewrite_eval import (
    MappingRewriteProvider,
    evaluate_query_rewrites_sync,
)


class TestQueryRewriteEval(unittest.TestCase):
    def test_groups_metrics_and_keeps_shadow_actual_query_original(self):
        tasks = [
            {
                "id": "follow",
                "query": "继续解释这个 RRF",
                "recent_context": "上一轮讨论了 RRF",
                "expected_refs": ["rrf.md"],
                "query_type": "followup",
            },
            {
                "id": "natural",
                "query": "如何找缓存方案",
                "lexical_coverage": 0.1,
                "expected_refs": ["cache.md"],
                "query_type": "natural",
            },
            {
                "id": "identifier",
                "query": "BV1AB1234567 如何下载",
                "expected_refs": ["video.md"],
                "query_type": "identifier",
            },
        ]
        fixture = {
            "继续解释这个 RRF": ([], 0.010),
            "解释 RRF 检索融合": (["rrf.md"], 0.020),
            "如何找缓存方案": ([], 0.011),
            "缓存方案检索": (["cache.md"], 0.021),
            "BV1AB1234567 如何下载": (["video.md"], 0.012),
        }

        def retrieve(query, _task):
            return fixture.get(query, ([], 0.0))

        provider = MappingRewriteProvider({
            "继续解释这个 RRF": {"query": "解释 RRF 检索融合", "preserved_entities": ["RRF"]},
            "如何找缓存方案": {"query": "缓存方案检索", "preserved_entities": []},
        })
        report = evaluate_query_rewrites_sync(tasks, retrieve, provider, mode="shadow")
        follow = report["per_query"][0]
        self.assertTrue(follow["candidate_retrieved"])
        self.assertEqual(follow["actual_query"], follow["original_query"])
        self.assertEqual(follow["original"]["recall@5"], 0.0)
        self.assertEqual(follow["candidate"]["recall@5"], 1.0)
        self.assertEqual(report["summary"]["followup"]["delta"]["recall@5"], 1.0)
        self.assertEqual(report["summary"]["identifier"]["candidate_retrieved"], 0)
        self.assertEqual(provider.calls, 2)

    def test_on_mode_applies_only_valid_candidate_and_reports_p95(self):
        tasks = [{
            "id": "follow",
            "query": "继续这个 RRF",
            "recent_context": "RRF",
            "expected_refs": ["rrf.md"],
            "query_type": "followup",
        }]
        calls = []

        def retrieve(query, _task):
            calls.append(query)
            return (["rrf.md"] if "检索融合" in query else []), 0.015

        provider = MappingRewriteProvider({
            "继续这个 RRF": {"query": "解释 RRF 检索融合", "preserved_entities": ["RRF"]},
        })
        report = evaluate_query_rewrites_sync(tasks, retrieve, provider, mode="on")
        row = report["per_query"][0]
        self.assertTrue(row["applied"])
        self.assertEqual(row["actual_query"], "解释 RRF 检索融合")
        self.assertEqual(row["actual"]["recall@5"], 1.0)
        self.assertEqual(calls, ["继续这个 RRF", "解释 RRF 检索融合"])
        self.assertEqual(report["overall"]["candidate"]["p95_ms"], 15.0)

    def test_missing_provider_is_a_safe_no_candidate_result(self):
        task = {
            "id": "follow",
            "query": "继续这个",
            "recent_context": "上一轮",
            "expected_refs": ["a.md"],
        }
        calls = []

        def retrieve(query, _task):
            calls.append(query)
            return [], 0.001

        report = evaluate_query_rewrites_sync([task], retrieve, provider=None, mode="shadow")
        row = report["per_query"][0]
        self.assertFalse(row["candidate_retrieved"])
        self.assertEqual(row["candidate_query"], row["original_query"])
        self.assertEqual(calls, ["继续这个"])


if __name__ == "__main__":
    unittest.main()
