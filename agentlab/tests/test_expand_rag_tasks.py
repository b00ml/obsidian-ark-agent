import tempfile
import unittest
from pathlib import Path

from pathlib import Path
import importlib.util

_MODULE = Path(__file__).resolve().parents[1] / "scripts" / "expand_rag_tasks.py"
_SPEC = importlib.util.spec_from_file_location("expand_rag_tasks", _MODULE)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
build_rows = _MOD.build_rows


class TestExpandRagTasks(unittest.TestCase):
    def test_adds_typed_negatives_without_changing_positive_rows(self):
        rows = [{"id": "P1", "query": "fact", "expected_refs": ["wiki/fact.md"]}]
        expanded = build_rows(rows, negative_target=60)
        self.assertEqual(len(expanded), 61)
        self.assertEqual(sum(bool(row.get("expected_refs")) for row in expanded), 1)
        negatives = [row for row in expanded if not row.get("expected_refs")]
        self.assertEqual(len(negatives), 60)
        self.assertTrue({row["negative_kind"] for row in negatives} >= {
            "out_of_domain", "random_token", "unknown_identifier", "plausible_absent"
        })


if __name__ == "__main__":
    unittest.main()
