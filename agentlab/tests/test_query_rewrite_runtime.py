import json
import tempfile
import unittest
from pathlib import Path

from agentlab.core.llm import LLMResponse
from agentlab.core.message import TokenUsage
from agentlab.rag.recall import RAGRecall
from agentlab.runtime.config import RagConfig
from agentlab.tools.rag_tools import build_rag_tools


class _RewriteProvider:
    def __init__(self, rewritten: str):
        self.rewritten = rewritten
        self.calls = 0

    async def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        return LLMResponse(
            content=json.dumps({
                "query": self.rewritten,
                "preserved_entities": ["RRF"],
                "reason": "fixture",
            }, ensure_ascii=False),
            tool_calls=[], usage=TokenUsage(),
        )


class TestQueryRewriteRuntime(unittest.TestCase):
    def _recallers(self, seen):
        def rec(query):
            seen.append(query)
            if query == "继续解释这个 RRF":
                return [{"title": "原始", "content": "原始证据", "ref": "original.md", "source": "vault"}]
            return [{"title": "改写", "content": "改写证据", "ref": "rewritten.md", "source": "vault"}]
        return [rec]

    def test_shadow_retrieves_candidate_but_keeps_original_display(self):
        seen = []
        provider = _RewriteProvider("解释 RRF 检索融合")
        recall = RAGRecall(
            self._recallers(seen), rewrite_provider=provider,
            query_rewrite_mode="shadow",
        )
        rows = recall.retrieve("继续解释这个 RRF", limit=5)
        self.assertEqual([row.ref for row in rows], ["original.md"])
        self.assertEqual(seen, ["继续解释这个 RRF", "解释 RRF 检索融合"])
        self.assertEqual(provider.calls, 1)
        telemetry = recall.last_query_rewrite
        self.assertFalse(telemetry["applied"])
        self.assertTrue(telemetry["candidate_retrieved"])
        self.assertEqual(telemetry["candidate_refs"], ["rewritten.md"])
        self.assertEqual(len(telemetry["candidate_query_hash"]), 64)

    def test_on_fuses_candidate_without_exposing_internal_variant_source(self):
        seen = []
        provider = _RewriteProvider("解释 RRF 检索融合")
        recall = RAGRecall(
            self._recallers(seen), rewrite_provider=provider,
            query_rewrite_mode="on",
        )
        rows = recall.retrieve("继续解释这个 RRF", limit=5)
        self.assertEqual({row.ref for row in rows}, {"original.md", "rewritten.md"})
        self.assertTrue(all(not row.source.startswith("__query_rewrite__") for row in rows))
        self.assertEqual(recall.last_query_plan["variants"], ["继续解释这个 RRF", "解释 RRF 检索融合"])

    def test_missing_provider_and_off_mode_are_safe_fallbacks(self):
        seen = []
        shadow = RAGRecall(
            self._recallers(seen), query_rewrite_mode="shadow",
        )
        rows = shadow.retrieve("继续解释这个 RRF", limit=5)
        self.assertEqual([row.ref for row in rows], ["original.md"])
        self.assertEqual(seen, ["继续解释这个 RRF"])
        self.assertEqual(shadow.last_query_rewrite["reason"], "rewrite_provider_missing")

        cfg = RagConfig()
        self.assertEqual(cfg.query_rewrite_mode, "off")
        self.assertEqual(RagConfig(query_rewrite_mode="unsafe").query_rewrite_mode, "off")
        self.assertEqual(RagConfig(query_rewrite_max_variants=9).query_rewrite_max_variants, 1)
        self.assertEqual(RagConfig(rewrite_deadline_ms=0).rewrite_deadline_ms, 250)

    def test_tool_layer_exposes_opt_in_shadow_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            (vault / "rrf.md").write_text("RRF 检索融合方案。", encoding="utf-8")
            provider = _RewriteProvider("解释 RRF 检索融合")
            cfg = RagConfig(
                index_backend="p2", vector_enabled=False, vector_mode="off",
                query_rewrite_mode="shadow",
            )
            tools = build_rag_tools(rag_config=cfg, vault_root=str(vault), llm=provider)
            by_name = {tool.name: tool for tool in tools}
            by_name["rag_reindex"].fn()
            payload = json.loads(by_name["rag_retrieve"].fn(
                query="继续解释这个 RRF", limit=5, metadata=True,
            ))
            self.assertEqual(payload["strategy"], "lexical-only")
            self.assertFalse(payload["query_rewrite"]["applied"])
            self.assertTrue(payload["query_rewrite"]["candidate_retrieved"])
            self.assertNotIn("original_query", payload["query_rewrite"])
            self.assertNotIn("original_query", payload["query_plan"])
            self.assertEqual(len(payload["query_plan"]["original_query_hash"]), 64)
            self.assertIn("original_recall_ms", payload["timings_ms"])
            self.assertIn("candidate_recall_ms", payload["timings_ms"])
            self.assertIn("fuse_ms", payload["timings_ms"])
            self.assertGreaterEqual(payload["timings_ms"]["total_ms"], 0)
            self.assertEqual(provider.calls, 1)


if __name__ == "__main__":
    unittest.main()
