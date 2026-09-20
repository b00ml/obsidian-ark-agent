import unittest

from agentlab.rag.query import build_query_plan, classify_query
from agentlab.rag.recall import RAGRecall


class TestQueryClassifier(unittest.TestCase):
    def test_empty_and_identifier_are_never_rewrite_candidates(self):
        self.assertEqual(classify_query("  ")[0], "empty")
        plan = build_query_plan("BV1AB1234567 如何下载", rewrite_mode="shadow", lexical_coverage=0.0)
        self.assertEqual(plan.query_type, "identifier")
        self.assertFalse(plan.should_rewrite)
        self.assertIn("BV1AB1234567", plan.preserved_entities)

    def test_followup_and_multi_hop_are_classified(self):
        follow = build_query_plan("继续解释这个", recent_context="上一轮讨论了 RRF", rewrite_mode="shadow")
        self.assertEqual(follow.query_type, "followup")
        self.assertTrue(follow.should_rewrite)
        multi = build_query_plan("比较关键词检索和向量检索", rewrite_mode="shadow", lexical_coverage=0.1)
        self.assertEqual(multi.query_type, "multi_hop")
        self.assertFalse(multi.should_rewrite)

    def test_negative_is_distinct_from_natural(self):
        self.assertEqual(classify_query("有没有关于 RRF 的笔记")[0], "negative")
        self.assertEqual(classify_query("解释 RRF 的延迟")[0], "natural")

    def test_recall_records_shadow_plan_without_changing_route_query(self):
        seen = []

        def rec(query):
            seen.append(query)
            return []

        recall = RAGRecall([rec], strategy="lexical-only")
        recall.retrieve("解释 RRF")
        self.assertEqual(seen, ["解释 RRF"])
        self.assertEqual(recall.last_query_plan["query_type"], "natural")
        self.assertEqual(recall.last_query_plan["variants"], ["解释 RRF"])
