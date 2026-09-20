import unittest
from agentlab.eval.lexical_sweep import replay_refs, sweep_lexical


class TestLexicalSweep(unittest.TestCase):
    def test_replay_dedupes_entry_and_file(self):
        refs = ["a.md#x:ch11111111", "a.md#x:ch22222222", "b.md#y:ch33333333"]
        self.assertEqual(len(replay_refs(refs, dedupe="ref")), 3)
        self.assertEqual(len(replay_refs(refs, dedupe="entry")), 2)
        self.assertEqual(len(replay_refs(refs, dedupe="file")), 2)

    def test_sweep_reports_grouped_metrics(self):
        report = {"per_query": [{"id": "R1", "hybrid_shadow": {"routes": {"lexical": {"refs": ["a.md#x:ch1", "a.md#x:ch2"]}}}}]}
        task = {"id": "R1", "query": "x", "expected_refs": ["a.md#x"], "query_type": "bucket_entry"}
        result = sweep_lexical(report, [task], k=5)
        self.assertEqual(result["schema"], "rag-lexical-aggregation-sweep-v1")
        self.assertEqual(result["summary"]["by_query_type"]["bucket_entry"]["entry"]["recall@5"], 1.0)


if __name__ == "__main__":
    unittest.main()
