import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentlab.runtime.config import load_config


class TestConfigRagEnvironment(unittest.TestCase):
    def test_rag_environment_overrides_file_without_mutating_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            original = {
                "rag": {
                    "vector_enabled": False,
                    "vector_mode": "off",
                    "lexical_mode": "on",
                    "embed_base_url": "https://file.example/v1",
                    "embed_model": "file-model",
                    "embed_api_key": "file-key",
                }
            }
            path.write_text(json.dumps(original), encoding="utf-8")
            env = {
                "AGENT_RAG_VECTOR_ENABLED": "true",
                "AGENT_RAG_VECTOR_MODE": "on",
                "AGENT_RAG_VECTOR_FALLBACK_MODE": "on",
                "AGENT_RAG_LEXICAL_MODE": "on",
                "AGENT_RAG_EMBED_BASE_URL": "https://ui.example/v1",
                "AGENT_RAG_EMBED_MODEL": "ui-model",
                "AGENT_RAG_EMBED_API_KEY": "ui-key",
                "AGENT_RAG_EMBED_TIMEOUT": "12.5",
                "AGENT_RAG_VECTOR_MIN_SCORE": "0.35",
                "AGENT_RAG_LEXICAL_MIN_COVERAGE": "0.3",
                "AGENT_RAG_CHUNK_STRATEGY": "markdown-structure-v2-min64",
                "AGENT_RAG_INDEX_VERSION": "s1-p4.5c-v2-min64",
            }
            with patch.dict(os.environ, env, clear=False):
                cfg = load_config(path)
            self.assertTrue(cfg.rag.vector_enabled)
            self.assertEqual(cfg.rag.vector_mode, "on")
            self.assertEqual(cfg.rag.vector_fallback_mode, "on")
            self.assertEqual(cfg.rag.lexical_mode, "on")
            self.assertEqual(cfg.rag.embed_base_url, "https://ui.example/v1")
            self.assertEqual(cfg.rag.embed_model, "ui-model")
            self.assertEqual(cfg.rag.effective_key(), "ui-key")
            self.assertEqual(cfg.rag.embed_timeout, 12.5)
            self.assertEqual(cfg.rag.vector_min_score, 0.35)
            self.assertEqual(cfg.rag.lexical_min_coverage, 0.3)
            self.assertEqual(cfg.rag.chunk_strategy, "markdown-structure-v2-min64")
            self.assertEqual(cfg.rag.index_version, "s1-p4.5c-v2-min64")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), original)

    def test_empty_or_missing_environment_keeps_defaults(self):
        names = [
            "AGENT_RAG_VECTOR_ENABLED", "AGENT_RAG_VECTOR_MODE",
            "AGENT_RAG_VECTOR_FALLBACK_MODE",
            "AGENT_RAG_LEXICAL_MODE", "AGENT_RAG_EMBED_BASE_URL",
            "AGENT_RAG_EMBED_MODEL", "AGENT_RAG_EMBED_API_KEY",
            "AGENT_RAG_EMBED_TIMEOUT",
            "AGENT_RAG_VECTOR_MIN_SCORE", "AGENT_RAG_LEXICAL_MIN_COVERAGE",
            "AGENT_RAG_CHUNK_STRATEGY", "AGENT_RAG_INDEX_VERSION",
        ]
        with patch.dict(os.environ, {name: "" for name in names}, clear=False):
            cfg = load_config(Path("does-not-exist-config.json"))
        self.assertFalse(cfg.rag.vector_enabled)
        self.assertEqual(cfg.rag.vector_mode, "shadow")
        self.assertEqual(cfg.rag.vector_fallback_mode, "off")
        self.assertEqual(cfg.rag.lexical_mode, "on")
        self.assertEqual(cfg.rag.embed_base_url, "")
        self.assertEqual(cfg.rag.effective_key(), "")

    def test_memory_switch_disables_implicit_recall_and_deposit(self):
        with patch.dict(os.environ, {"AGENT_MEMORY_ENABLED": "false"}, clear=False):
            cfg = load_config(Path("does-not-exist-config.json"))
        self.assertEqual(cfg.limits.memory_inject_topk, 0)
        self.assertEqual(cfg.limits.memorize_every, 0)

    def test_memory_switch_enabled_preserves_configured_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"limits": {
                "memory_inject_topk": 3,
                "memorize_every": 7,
            }}), encoding="utf-8")
            with patch.dict(os.environ, {"AGENT_MEMORY_ENABLED": "true"}, clear=False):
                cfg = load_config(path)
        self.assertEqual(cfg.limits.memory_inject_topk, 3)
        self.assertEqual(cfg.limits.memorize_every, 7)

    def test_invalid_score_guards_fail_closed(self):
        with patch.dict(os.environ, {
            "AGENT_RAG_VECTOR_MIN_SCORE": "not-a-number",
            "AGENT_RAG_LEXICAL_MIN_COVERAGE": "2.0",
        }, clear=False):
            cfg = load_config(Path("does-not-exist-config.json"))
        self.assertEqual(cfg.rag.vector_min_score, 0.0)
        self.assertEqual(cfg.rag.lexical_min_coverage, 0.0)

    def test_empty_ui_key_overrides_stale_file_key(self):
        """清空设置页 Key 后不得回退到 config.json 中的旧凭据。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({
                "rag": {
                    "embed_base_url": "https://file.example/v1",
                    "embed_api_key": "stale-file-key",
                }
            }), encoding="utf-8")
            with patch.dict(os.environ, {"AGENT_RAG_EMBED_API_KEY": ""}, clear=False):
                cfg = load_config(path)
            self.assertEqual(cfg.rag.effective_key(), "")

    def test_context_shadow_defaults_and_invalid_mode_fail_closed(self):
        cfg = load_config(Path("does-not-exist-config.json"))
        self.assertEqual(cfg.context.assembler_mode, "shadow")
        self.assertEqual(cfg.context.memory_budget_tokens, 1200)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"context": {"assembler_mode": "unsafe"}}), encoding="utf-8")
            self.assertEqual(load_config(path).context.assembler_mode, "shadow")


if __name__ == "__main__":
    unittest.main()
