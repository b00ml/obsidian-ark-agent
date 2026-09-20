"""A1/OPT-229 CLI parity 测试：run/repl 接自动沉淀 + repl 每轮召回刷新。"""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agentlab.runtime.cli import _attach_depository, _refresh_repl_instructions


def _cfg(tmp):
    from agentlab.runtime.config import Config
    c = Config(vault_root=tmp)
    c.limits.memorize_every = 5
    return c


def _patched_brain(tmp):
    """把 brain config 指向 tmp Vault（否则 load_brain_config 读真实配置）。"""
    return patch("agentlab.tools.connectors.brain_tools.load_brain_config",
                 lambda cfg: {"vault_path": tmp})


class TestCliParity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "Inbox"), exist_ok=True)

    def test_attach_depository_wires_store_and_persists(self):
        cfg = _cfg(self.tmp)
        with _patched_brain(self.tmp):
            # attach 必须同在 patch 内：MemoryStore 构造期的 brain config 决定落盘位置
            run_cfg = _attach_depository(cfg, SimpleNamespace())
        self.assertIsNotNone(run_cfg.depository)
        with _patched_brain(self.tmp):
            r = run_cfg.depository.commit("CLI 沉淀的记忆", tags=["cli"], source_session="")
        self.assertEqual(r["status"], "committed")
        import glob as _g
        mds = _g.glob(self.tmp + "/ark/memory/**/*.md", recursive=True)
        self.assertTrue(any("CLI 沉淀的记忆" in open(f, encoding="utf-8").read() for f in mds))

    def test_attach_depository_disabled_without_memorize_every(self):
        cfg = _cfg(self.tmp)
        cfg.limits.memorize_every = 0
        run_cfg = _attach_depository(cfg, SimpleNamespace())
        self.assertIsNone(getattr(run_cfg, "depository", None))

    def test_refresh_repl_instructions_injects_memory(self):
        cfg = _cfg(self.tmp)
        from agentlab.memory.markdown_store import MemoryMarkdownStore
        MemoryMarkdownStore(self.tmp).commit("用户偏好深色主题的终端配色", tags=["pref"],
                                             mem_type="context")
        reg = SimpleNamespace(all=lambda: [])
        agent = SimpleNamespace(instructions="")
        with _patched_brain(self.tmp):
            _refresh_repl_instructions(agent, cfg, reg, "我的配色偏好是什么")
        self.assertIn("深色主题", agent.instructions)

    def test_refresh_repl_instructions_without_memory(self):
        cfg = _cfg(self.tmp)
        reg = SimpleNamespace(all=lambda: [])
        agent = SimpleNamespace(instructions="")
        with _patched_brain(self.tmp):
            _refresh_repl_instructions(agent, cfg, reg, "完全无关的问题")
        self.assertIn("本轮无相关长期记忆召回", agent.instructions)


if __name__ == "__main__":
    unittest.main()
