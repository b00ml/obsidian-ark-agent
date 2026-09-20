import unittest

from agentlab.contracts import RetrievalScope, RetrievalStatus, RetrievalStrategy
from agentlab.rag.citations import CitationRegistry


class CitationRegistryTests(unittest.TestCase):
    def _result(self):
        return {
            "strategy": RetrievalStrategy.LEXICAL_ONLY,
            "status": RetrievalStatus.AVAILABLE,
            "scope": {"project_id": "p1", "session_id": "s1"},
            "items": [{"title": "A", "content": "fact", "ref": "wiki/a.md",
                       "source": "vault", "project_id": "p1", "session_id": "s1"}],
        }

    def test_register_validate_and_roundtrip(self):
        registry = CitationRegistry()
        entries = registry.register_retrieval(self._result())
        self.assertTrue(entries[0].source_hash)
        valid, reasons = registry.validate(["wiki/a.md"], scope=RetrievalScope(project_id="p1", session_id="s1"))
        self.assertEqual([item.ref for item in valid], ["wiki/a.md"])
        self.assertEqual(reasons, [])
        restored = CitationRegistry.from_dict(registry.to_dict())
        self.assertIn("wiki/a.md", restored.entries)

    def test_revoked_and_cross_scope_are_rejected(self):
        registry = CitationRegistry()
        registry.register_retrieval(self._result())
        self.assertTrue(registry.revoke("wiki/a.md"))
        valid, reasons = registry.validate(["wiki/a.md"], scope={"project_id": "p1"})
        self.assertFalse(valid)
        self.assertIn("revoked:wiki/a.md", reasons)
        registry.register_retrieval(self._result())
        valid, reasons = registry.validate(["wiki/a.md"], scope={"project_id": "other"})
        self.assertFalse(valid)
        self.assertIn("scope:wiki/a.md", reasons)

    def test_unknown_reference_fails_closed(self):
        valid, reasons = CitationRegistry().validate(["missing.md"])
        self.assertEqual(valid, [])
        self.assertEqual(reasons, ["unknown:missing.md"])


if __name__ == "__main__":
    unittest.main()
