"""S0 shared product/runtime contract tests."""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from agentlab.contracts import (
    CONTRACT_VERSION,
    Artifact,
    Project,
    Provenance,
    RetrievalItem,
    RetrievalScope,
    RetrievalStatus,
    RetrievalStrategy,
    Session,
    Source,
    scoped_retrieval_result,
    strategy_from_modes,
)


class TestS0Contracts(unittest.TestCase):
    def test_strategy_mapping_preserves_existing_runtime_switches(self):
        self.assertEqual(
            strategy_from_modes(vector_enabled=False),
            RetrievalStrategy.LEXICAL_ONLY,
        )
        self.assertEqual(
            strategy_from_modes(vector_enabled=True, vector_mode="shadow"),
            RetrievalStrategy.SHADOW,
        )
        self.assertEqual(
            strategy_from_modes(vector_enabled=True, vector_mode="on", lexical_mode="on"),
            RetrievalStrategy.HYBRID,
        )
        self.assertEqual(
            strategy_from_modes(vector_enabled=True, vector_mode="on", lexical_mode="off"),
            RetrievalStrategy.VECTOR_ONLY,
        )
        # A disabled vector switch always wins over a stale lexical setting.
        self.assertEqual(
            strategy_from_modes(vector_enabled=False, vector_mode="on", lexical_mode="off"),
            RetrievalStrategy.LEXICAL_ONLY,
        )

    def test_scope_normalises_ids_statuses_and_round_trips(self):
        scope = RetrievalScope.model_validate({
            "project_id": " project-a ",
            "session_id": "s-1",
            "statuses": ["active", "active", " ", "draft"],
            "include_archive": False,
        })
        self.assertEqual(scope.project_id, "project-a")
        self.assertEqual(scope.statuses, ["active", "draft"])
        self.assertEqual(RetrievalScope.model_validate(scope.to_dict()), scope)

    def test_retrieval_item_requires_ref_and_derives_provenance(self):
        item = RetrievalItem(
            title="笔记",
            content="证据",
            ref="wiki/a.md#h",
            source="vault",
            project_id="p1",
            session_id="s1",
            score=0.42,
            context_of=["wiki/a.md#h:ch-neighbor"],
        )
        self.assertEqual(item.provenance[0].ref, "wiki/a.md#h")
        self.assertEqual(item.provenance[0].project_id, "p1")
        self.assertEqual(item.context_of, ["wiki/a.md#h:ch-neighbor"])
        with self.assertRaises(ValidationError):
            RetrievalItem(title="缺少 ref")

    def test_result_envelope_is_serialisable_and_warnings_degrade_status(self):
        result = scoped_retrieval_result(
            [{"title": "A", "ref": "a.md", "source": "vault"}],
            strategy="hybrid",
            warnings=["vector timeout", "vector timeout"],
            scope={"project_id": "p1", "statuses": ["active"]},
        )
        self.assertEqual(result.contract, CONTRACT_VERSION)
        self.assertEqual(result.status, RetrievalStatus.DEGRADED)
        self.assertEqual(result.warnings, ["vector timeout"])
        encoded = result.to_dict()
        self.assertEqual(encoded["strategy"], "hybrid")
        self.assertEqual(encoded["scope"]["project_id"], "p1")
        self.assertEqual(encoded["items"][0]["provenance"][0]["ref"], "a.md")

    def test_entities_share_scope_and_provenance_fields(self):
        now = datetime.now(timezone.utc)
        project = Project(id="p1", name="项目一", mode="research", goal="查证", created_at=now, updated_at=now)
        source = Source(ref="wiki/a.md", kind="vault", scope="p1", captured_at=now)
        session = Session(id="s1", project_id="p1", goal="查证")
        artifact = Artifact(
            id="a1", project_id="p1", kind="report", path="wiki/report.md",
            source_refs=["wiki/a.md", "wiki/a.md"], session_id="s1",
        )
        self.assertEqual(project.title, "项目一")
        self.assertEqual(project.name, "项目一")
        self.assertEqual(source.provenance[0].ref, "wiki/a.md")
        self.assertEqual(session.project_id, "p1")
        self.assertEqual(artifact.source_refs, ["wiki/a.md"])
        for value in (project, source, session, artifact):
            self.assertIsInstance(value.to_dict(), dict)

    def test_unknown_contract_fields_are_rejected(self):
        with self.assertRaises(ValidationError):
            Project(id="p1", unexpected="do-not-silently-drop")


if __name__ == "__main__":
    unittest.main()
