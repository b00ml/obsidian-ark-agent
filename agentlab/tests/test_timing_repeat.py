import json
import tempfile
import unittest
from pathlib import Path

from agentlab.eval.timing_repeat import aggregate_reports


def _report(index_sha="idx"):
    return {
        "summary": {
            "rrf": {
                "p50_ms": 300,
                "p95_ms": 700,
                "recall@5": 0.7,
                "recall@10": 0.8,
                "mrr": 0.5,
                "clean@5": 0.5,
            }
        },
        "meta": {
            "tasks_path": "tasks.jsonl",
            "input_hashes": {"tasks_sha256": "tasks", "index_sha256": index_sha},
            "chunking": {"index_version": "v1", "parser_version": "p1"},
            "embedding_model": "embed-v1",
        },
    }


class TestTimingRepeat(unittest.TestCase):
    def test_aggregates_distributions_and_snapshot(self):
        report = aggregate_reports([_report(), {
            **_report(),
            "summary": {"rrf": {
                "p50_ms": 200, "p95_ms": 500, "recall@5": 0.7,
                "recall@10": 0.8, "mrr": 0.5, "clean@5": 0.5,
            }},
        }])
        metric = report["metrics"]["rrf"]["p95_ms"]
        self.assertEqual(report["runs"], 2)
        self.assertEqual(metric["min"], 500.0)
        self.assertEqual(metric["max"], 700.0)
        self.assertEqual(metric["mean"], 600.0)
        self.assertEqual(report["snapshot"]["index_sha256"], "idx")

    def test_mismatched_snapshot_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "snapshot mismatch: index_sha256"):
            aggregate_reports([_report("a"), _report("b")])

    def test_cli_shape_can_be_serialised(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(_report()), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(aggregate_reports([loaded])["runs"], 1)


if __name__ == "__main__":
    unittest.main()
