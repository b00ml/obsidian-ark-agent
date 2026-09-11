"""knowledge_compiler 单元测试（零网络：LLM 用 fake 注入，vault 用临时目录）

覆盖:
  - extract_knowledge: JSON/围栏/非JSON/非法kind/空name/条目上限/prompt 渲染
  - compile_note: 新建页(模板字段+双链)/增量合并不重复/冲突块/confirm 钩子两分支
  - index/log 更新与幂等（跑两遍不重复）
  - 接线点: article_summarizer 编译抛异常不影响主笔记产出；bili_transcript 编译入口

运行: python -m unittest test_knowledge_compiler -v   （在 bili_summarizer/ 目录下）
不依赖 faster_whisper。
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import knowledge_compiler as kc

try:
    import article_summarizer as art
except ImportError:  # pragma: no cover - requests 缺失等极端环境
    art = None

try:
    import bili_transcript as bt
except ImportError:  # pragma: no cover - requests 缺失等极端环境
    bt = None


def make_llm(payload):
    """构造可控 fake llm_call(prompt)->str，并记录收到的 prompt"""
    calls = []

    def llm(prompt):
        calls.append(prompt)
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False)

    llm.calls = calls
    return llm


def entries(*items):
    return [{"name": n, "kind": k, "one_line": o} for n, k, o in items]


class VaultCase(unittest.TestCase):
    """提供临时 vault 目录的基类"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.vault = self._tmp.name
        self.note_path = os.path.join(self.vault, "Inbox", "我的笔记.md")
        os.makedirs(os.path.dirname(self.note_path), exist_ok=True)
        with open(self.note_path, "w", encoding="utf-8") as f:
            f.write("# 我的笔记\n\n正文内容")

    def read(self, *parts):
        with open(os.path.join(self.vault, *parts), "r", encoding="utf-8") as f:
            return f.read()


class TestExtractKnowledge(unittest.TestCase):
    """LLM 提取 + 二次校验"""

    def test_valid_json_array(self):
        llm = make_llm(entries(("检索增强生成", "concept", "用外部资料增强生成"),
                               ("吴恩达", "entity", "本课程主讲人")))
        out = kc.extract_knowledge(llm, "标题", "正文")
        self.assertEqual(out, [{"name": "检索增强生成", "kind": "concept",
                                "one_line": "用外部资料增强生成"},
                               {"name": "吴恩达", "kind": "entity",
                                "one_line": "本课程主讲人"}])

    def test_fenced_json(self):
        llm = make_llm("```json\n" + json.dumps(
            entries(("注意力机制", "concept", "加权聚合序列信息")), ensure_ascii=False) + "\n```")
        out = kc.extract_knowledge(llm, "T", "C")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "注意力机制")

    def test_non_json_returns_empty(self):
        llm = make_llm("抱歉，我无法按要求输出。")
        self.assertEqual(kc.extract_knowledge(llm, "T", "C"), [])

    def test_non_list_returns_empty(self):
        llm = make_llm('{"name": "x"}')
        self.assertEqual(kc.extract_knowledge(llm, "T", "C"), [])

    def test_invalid_kind_fallback_concept(self):
        llm = make_llm(entries(("怪条目", "skill", "非法kind")))
        out = kc.extract_knowledge(llm, "T", "C")
        self.assertEqual(out[0]["kind"], "concept")

    def test_empty_name_dropped(self):
        llm = make_llm([{"name": "", "kind": "concept", "one_line": "x"},
                        {"name": "   ", "kind": "entity", "one_line": "y"},
                        {"one_line": "没有name"},
                        {"name": "有效", "kind": "concept", "one_line": "z"}])
        out = kc.extract_knowledge(llm, "T", "C")
        self.assertEqual([e["name"] for e in out], ["有效"])

    def test_cap_twelve(self):
        llm = make_llm(entries(*[(f"条目{i}", "concept", f"概括{i}")
                                 for i in range(15)]))
        self.assertEqual(len(kc.extract_knowledge(llm, "T", "C")), kc.MAX_ENTRIES)

    def test_prompt_rendered_no_placeholder(self):
        llm = make_llm([])
        kc.extract_knowledge(llm, "视频标题X", "正文片段Y")
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("视频标题X", llm.calls[0])
        self.assertIn("正文片段Y", llm.calls[0])
        self.assertNotIn("{{", llm.calls[0])


class TestCompileNoteCreate(VaultCase):
    """新建 concept/entity 页：模板字段 + 双链齐全"""

    def test_new_pages_with_fields_and_backlinks(self):
        llm = make_llm(entries(("检索增强生成", "concept", "用外部资料增强生成"),
                               ("吴恩达", "entity", "课程主讲人")))
        result = kc.compile_note(self.vault, self.note_path, "我的笔记", "正文", llm)

        concept = self.read("wiki", "concepts", "检索增强生成.md")
        entity = self.read("wiki", "entities", "吴恩达.md")
        for page, kind in ((concept, "concept"), (entity, "entity")):
            self.assertIn("type: " + kind, page)
            self.assertIn('title: "', page)
            self.assertIn("description: ", page)
            self.assertIn("sources:", page)
            self.assertIn('[[我的笔记]]', page)
            self.assertIn("generated: ", page)
            self.assertIn("status: draft", page)
            self.assertIn("## 一句话核心", page)
            self.assertIn("## 来源", page)
            self.assertIn("- [[我的笔记]]", page)
        self.assertIn("用外部资料增强生成", concept)

        self.assertEqual(result["created"], ["检索增强生成", "吴恩达"])
        self.assertEqual(result["updated"], [])
        self.assertEqual(result["conflicts"], [])
        self.assertEqual(result["skipped"], 0)

    def test_index_and_log_updated(self):
        llm = make_llm(entries(("检索增强生成", "concept", "概括"),
                               ("吴恩达", "entity", "人物")))
        kc.compile_note(self.vault, self.note_path, "我的笔记", "正文", llm)
        index = self.read("wiki", "index.md")
        self.assertIn("## 概念", index)
        self.assertIn("- [[检索增强生成]]", index)
        self.assertIn("## 实体", index)
        self.assertIn("- [[吴恩达]]", index)
        log = self.read("wiki", "log.md")
        self.assertIn("| 我的笔记 | 新建2/合并0/冲突0", log)


class TestCompileNoteMerge(VaultCase):
    """增量合并：不重复追加"""

    def setUp(self):
        super().setUp()
        page_dir = os.path.join(self.vault, "wiki", "concepts")
        os.makedirs(page_dir, exist_ok=True)
        self.page = os.path.join(page_dir, "已知概念.md")
        with open(self.page, "w", encoding="utf-8") as f:
            f.write("---\ntitle: \"已知概念\"\ntype: concept\n"
                    "description: \"既有描述\"\nsources:\n  - \"[[旧来源]]\"\n"
                    "generated: 2026-01-01\nstatus: draft\n---\n\n"
                    "# 已知概念\n\n## 一句话核心\n\n旧的核心表述\n\n"
                    "## 来源\n\n- [[旧来源]]\n")

    def test_merge_appends_section(self):
        # 新旧说法一致（旧核心包含新概括）→ 合并而非冲突
        llm = make_llm(entries(("已知概念", "concept", "旧的核心表述")))
        result = kc.compile_note(self.vault, self.note_path, "我的笔记", "正文", llm)
        page = self.read("wiki", "concepts", "已知概念.md")
        self.assertIn("## 来自 [[我的笔记]]", page)
        self.assertIn("旧的核心表述", page)
        self.assertNotIn("## 知识冲突", page)
        self.assertEqual(result["updated"], ["已知概念"])

    def test_merge_no_duplicate_on_rerun(self):
        llm = make_llm(entries(("已知概念", "concept", "旧的核心表述")))
        kc.compile_note(self.vault, self.note_path, "我的笔记", "正文", llm)
        result2 = kc.compile_note(self.vault, self.note_path, "我的笔记", "正文", llm)
        page = self.read("wiki", "concepts", "已知概念.md")
        self.assertEqual(page.count("## 来自 [[我的笔记]]"), 1)
        self.assertEqual(result2["skipped"], 1)
        self.assertEqual(result2["updated"], [])


class TestCompileNoteConflict(VaultCase):
    """矛盾不静默覆盖：冲突块 + confirm 钩子两分支"""

    def setUp(self):
        super().setUp()
        page_dir = os.path.join(self.vault, "wiki", "concepts")
        os.makedirs(page_dir, exist_ok=True)
        self.page = os.path.join(page_dir, "注意力机制.md")
        with open(self.page, "w", encoding="utf-8") as f:
            f.write("---\ntitle: \"注意力机制\"\ntype: concept\n"
                    "description: \"旧说\"\nsources:\n  - \"[[旧论文]]\"\n"
                    "generated: 2026-01-01\nstatus: draft\n---\n\n"
                    "# 注意力机制\n\n## 一句话核心\n\n注意力机制只用于机器翻译\n\n"
                    "## 来源\n\n- [[旧论文]]\n")
        self.old_core = "注意力机制只用于机器翻译"
        self.new_core = "注意力机制是通用的序列建模组件"

    def compile(self, confirm=None):
        llm = make_llm(entries(("注意力机制", "concept", self.new_core)))
        return kc.compile_note(self.vault, self.note_path, "我的笔记", "正文", llm,
                               confirm=confirm)

    def test_conflict_block_written_by_default(self):
        result = self.compile()
        page = self.read("wiki", "concepts", "注意力机制.md")
        self.assertIn("## 知识冲突", page)
        self.assertIn(f"- 旧说（原页来源：[[旧论文]]）：{self.old_core}", page)
        self.assertIn(f"- 新说（[[我的笔记]]）：{self.new_core}", page)
        self.assertIn(self.old_core, page)  # 两说并存，不覆盖
        self.assertIn(self.new_core, page)
        self.assertEqual(result["conflicts"], ["注意力机制"])

    def test_confirm_merge_branch(self):
        confirm = mock.Mock(return_value="merge")
        result = self.compile(confirm=confirm)
        page = self.read("wiki", "concepts", "注意力机制.md")
        self.assertNotIn("## 知识冲突", page)
        self.assertIn("## 来自 [[我的笔记]]", page)
        confirm.assert_called_once_with("注意力机制", self.old_core, self.new_core)
        self.assertEqual(result["updated"], ["注意力机制"])
        self.assertEqual(result["conflicts"], [])

    def test_confirm_none_branch_defaults_to_conflict(self):
        confirm = mock.Mock(return_value=None)
        result = self.compile(confirm=confirm)
        page = self.read("wiki", "concepts", "注意力机制.md")
        self.assertIn("## 知识冲突", page)
        self.assertEqual(result["conflicts"], ["注意力机制"])

    def test_conflict_not_duplicated_on_rerun(self):
        self.compile()
        result2 = self.compile()
        page = self.read("wiki", "concepts", "注意力机制.md")
        self.assertEqual(page.count("## 知识冲突"), 1)
        self.assertEqual(result2["skipped"], 1)


class TestIndexLogIdempotent(VaultCase):
    """index/log 更新且幂等（跑两遍不重复）"""

    def test_run_twice_no_duplicates(self):
        llm = make_llm(entries(("检索增强生成", "concept", "概括A"),
                               ("吴恩达", "entity", "人物B")))
        kc.compile_note(self.vault, self.note_path, "我的笔记", "正文", llm)
        result2 = kc.compile_note(self.vault, self.note_path, "我的笔记", "正文", llm)

        index = self.read("wiki", "index.md")
        self.assertEqual(index.count("- [[检索增强生成]]"), 1)
        self.assertEqual(index.count("- [[吴恩达]]"), 1)

        log_lines = [ln for ln in self.read("wiki", "log.md").splitlines()
                     if "我的笔记" in ln]
        self.assertEqual(len(log_lines), 1)
        self.assertIn("新建2/合并0/冲突0", log_lines[0])
        self.assertEqual(result2["skipped"], 2)
        self.assertEqual(result2["created"], [])

    def test_other_index_sections_preserved(self):
        index_path = os.path.join(self.vault, "wiki", "index.md")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("# 手工目录\n\n## 论文\n\n- [[某论文]]\n")
        llm = make_llm(entries(("检索增强生成", "concept", "概括")))
        kc.compile_note(self.vault, self.note_path, "我的笔记", "正文", llm)
        index = self.read("wiki", "index.md")
        self.assertIn("## 论文", index)
        self.assertIn("- [[某论文]]", index)
        self.assertIn("- [[检索增强生成]]", index)

    def test_empty_extraction_writes_nothing(self):
        llm = make_llm([])
        result = kc.compile_note(self.vault, self.note_path, "我的笔记", "正文", llm)
        self.assertEqual(result, {"created": [], "updated": [],
                                  "conflicts": [], "skipped": 0})
        self.assertFalse(os.path.exists(os.path.join(self.vault, "wiki", "log.md")))


@unittest.skipIf(art is None, "article_summarizer 依赖不可用")
class TestArticleWiring(unittest.TestCase):
    """article_summarizer 接线点：编译失败不影响主笔记产出"""

    HTML_SAMPLE = ("<html><head><title>t</title></head><body>"
                   "<div id='js_content'><p>正文</p></div></body></html>")
    MOCK_CONFIG = {"default": {"api_base": "https://example.invalid/v1",
                               "api_key": "sk-test", "model": "m"}}
    SUMMARY = {"title": "总结标题", "one_sentence": "核心", "points": [],
               "key_info": [], "summary": "总结"}

    def _run_main(self, tmp, compile_result):
        """compile_result: 异常实例 → 编译抛错；dict → 编译成功返回值"""
        out_path = os.path.join(tmp, "note.md")
        vault_path = os.path.join(tmp, "vault")
        os.makedirs(vault_path, exist_ok=True)
        stdout = io.StringIO()
        compile_mock = mock.Mock()
        if isinstance(compile_result, Exception):
            compile_mock.side_effect = compile_result
        else:
            compile_mock.return_value = compile_result
        with mock.patch("sys.argv", ["article_summarizer.py",
                                     "https://mp.weixin.qq.com/s/abc",
                                     "--output", out_path, "--vault", vault_path]), \
             mock.patch("article_summarizer.fetch_article",
                        return_value=self.HTML_SAMPLE), \
             mock.patch("article_summarizer.summarize_article",
                        return_value=self.SUMMARY), \
             mock.patch("article_summarizer.load_visual_config",
                        return_value=self.MOCK_CONFIG), \
             mock.patch("article_summarizer.compile_note", compile_mock), \
             contextlib.redirect_stdout(stdout):
            art.main()
        return out_path, stdout.getvalue(), compile_mock

    def test_compile_failure_still_writes_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path, out, _ = self._run_main(tmp, RuntimeError("LLM 爆炸"))
            self.assertTrue(os.path.exists(out_path), "主笔记必须照常落盘")
            with open(out_path, "r", encoding="utf-8") as f:
                self.assertIn("type: article-summary", f.read())
            self.assertIn("[COMPILE]", out)
            self.assertIn("不影响主笔记", out)

    def test_compile_success_called_with_vault(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault_path = os.path.join(tmp, "vault")
            os.makedirs(vault_path, exist_ok=True)
            out_path, out, compile_mock = self._run_main(
                tmp, {"created": ["X"], "updated": [], "conflicts": [], "skipped": 0})
            self.assertIn("[COMPILE] 知识编译完成", out)
            # compile_note 收到的 vault_root / note_title（文件名 stem，保证双链可解析）
            args, kwargs = compile_mock.call_args
            self.assertEqual(args[0], vault_path)
            self.assertEqual(args[1], out_path)
            self.assertEqual(args[2], "note")


@unittest.skipIf(bt is None, "bili_transcript 依赖不可用")
class TestBiliWiring(unittest.TestCase):
    """bili_transcript.compile_knowledge_for_note 接线（复用 va 客户端，fake 不出网）"""

    def test_wiring_compiles_wiki_from_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            note_path = os.path.join(tmp, "视频-总结.md")
            with open(note_path, "w", encoding="utf-8") as f:
                f.write("# 视频总结\n\n讲到了检索增强生成。")
            payload = json.dumps(entries(("检索增强生成", "concept", "外挂知识源")),
                                 ensure_ascii=False)
            fake_va = mock.Mock()
            fake_va._chat_completion.return_value = {"content": payload}
            with mock.patch("bili_transcript.va", fake_va), \
                    contextlib.redirect_stdout(io.StringIO()):
                bt.compile_knowledge_for_note(tmp, note_path, "视频-总结", {})
            page = os.path.join(tmp, "wiki", "concepts", "检索增强生成.md")
            self.assertTrue(os.path.exists(page))
            with open(page, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("[[视频-总结]]", content)

    def test_wiring_swallows_errors(self):
        # 客户端抛错（如 API Key 未配置）→ 只警告，不向主流程抛异常
        fake_va = mock.Mock()
        fake_va._chat_completion.side_effect = RuntimeError("未配置 API Key")
        with mock.patch("bili_transcript.va", fake_va), \
                contextlib.redirect_stdout(io.StringIO()):
            bt.compile_knowledge_for_note("X:/不存在的vault", "X:/无.md", "t", {})


if __name__ == "__main__":
    unittest.main()
