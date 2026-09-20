from __future__ import annotations

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from agentlab.contract_adapters import (
    citation_from_result,
    from_article_result,
    from_bili_result,
    from_inbox_task,
    from_memory_entry,
    from_rag_item,
)
from agentlab.contracts import CitationStatus, ProcessAttempt, ProcessStatus, StageResult


class TestContractAdapters(unittest.TestCase):
    def test_source_adapters_share_stage_semantics_and_preserve_unknowns(self):
        fixture = {
            "id": "source-1",
            "title": "same source",
            "content": "stable body",
            "status": "ok",
            "vendor_field": {"kept": True},
        }
        adapted = [
            from_bili_result({**fixture, "bvid": "BV1"}),
            from_article_result(fixture, url="https://example.test/a"),
            from_inbox_task({**fixture, "url": "https://example.test/a"}),
            from_rag_item({**fixture, "ref": "wiki/a.md"}),
            from_memory_entry(fixture),
        ]
        self.assertEqual({item.status for item in adapted}, {ProcessStatus.COMPLETED})
        self.assertTrue(all(item.content_hash for item in adapted))
        self.assertTrue(all(item.metadata["vendor_field"]["kept"] for item in adapted))
        for item in adapted:
            self.assertIsInstance(item.to_dict(), dict)

    def test_stage_result_rejects_backwards_time(self):
        now = datetime.now(timezone.utc)
        with self.assertRaises(ValidationError):
            StageResult(
                stage_id="s1", stage_name="parse", status=ProcessStatus.PARSED,
                started_at=now, ended_at=now.replace(year=now.year - 1),
            )

    def test_attempt_and_citation_are_serialisable(self):
        attempt = ProcessAttempt(
            run_id="run-1", attempt_id="attempt-1", source_id="source-1",
            status="retry", retry_count=1, error_code="TEMPORARY",
        )
        citation = citation_from_result(
            {"ref": "wiki/a.md", "source": "vault", "status": "revoked",
             "extra": object()},
            project_id="project-1", session_id="session-1",
        )
        self.assertEqual(attempt.status, ProcessStatus.RETRY)
        self.assertEqual(citation.status, CitationStatus.REVOKED)
        self.assertEqual(citation.project_id, "project-1")
        self.assertIsInstance(citation.to_dict(), dict)


if __name__ == "__main__":
    unittest.main()
