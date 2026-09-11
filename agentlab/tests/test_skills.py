"""P1-2/OPT-114 统一技能仓库单测：resolve_skills_dir 路径解析 + load_skills 注入。"""
import tempfile
import unittest
from pathlib import Path

from agentlab.memory.skill import load_skills, resolve_skills_dir

_SKILL = """---
name: "demo-skill"
description: "演示技能：处理演示任务时触发"
---

# 演示技能正文
"""


class TestResolveSkillsDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "skills" / "demo").mkdir(parents=True)
        (self.root / "skills" / "demo" / "SKILL.md").write_text(_SKILL, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_cwd_relative_hit_takes_priority(self):
        # cwd 下能解析（相对路径）→ 原样返回
        import os
        old = os.getcwd()
        os.chdir(self.root)
        try:
            self.assertEqual(resolve_skills_dir("skills"),
                             Path("skills"))
        finally:
            os.chdir(old)

    def test_cwd_miss_falls_back_to_repo_root(self):
        # cwd 下不存在 → 仓库根兜底命中（serve 以 agentlab/ 为 cwd 的真实形态）
        got = resolve_skills_dir("skills", repo_root=self.root)
        self.assertEqual(got, self.root / "skills")
        self.assertTrue(got.is_dir())

    def test_absolute_path_passthrough(self):
        got = resolve_skills_dir(str(self.root / "skills"), repo_root=self.root)
        self.assertEqual(got, self.root / "skills")

    def test_total_miss_returns_original(self):
        got = resolve_skills_dir("no-such-dir", repo_root=self.root)
        self.assertEqual(got, Path("no-such-dir"))
        self.assertEqual(load_skills(str(got)), "", "落空时 load_skills 静默空，不抛错")

    def test_load_skills_injects_frontmatter(self):
        got = load_skills(str(self.root / "skills"))
        self.assertIn("demo-skill", got)
        self.assertIn("演示技能", got)
        self.assertIn("skills\\demo", got.replace("/", "\\"), "带出技能路径供模型按需读正文")


if __name__ == "__main__":
    unittest.main()
