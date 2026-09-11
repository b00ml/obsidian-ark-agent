"""article_summarizer 单元测试（mock 外部依赖，不触发真实 API/网络）

运行: python -m unittest test_article_summarizer（在 bili_summarizer/ 目录下）
"""
import json
import unittest
from unittest import mock

import article_summarizer as art

HTML_SAMPLE = """
<html><head><title>测试公众号标题</title></head>
<body>
<script>var nickname = "测试公众号";</script>
<h1 class="rich_media_title">原文标题</h1>
<div id="js_content">
<p>第一段内容，讲知识管理。</p>
<p>第二段内容，讲 Obsidian 双链。</p>
</div>
</body></html>
"""

MOCK_CONFIG = {
    "default": {
        "api_base": "https://example.com/v1",
        "api_key": "sk-test-key",
        "model": "qwen-plus",
    }
}


class TestExtractArticle(unittest.TestCase):
    def test_extract_fields(self):
        result = art.extract_article(HTML_SAMPLE)
        self.assertEqual(result["title"], "原文标题")
        self.assertEqual(result["author"], "测试公众号")
        self.assertIn("第一段内容", result["content"])
        self.assertIn("Obsidian", result["content"])

    def test_title_fallback(self):
        result = art.extract_article("<html><title>只有 title</title><body><p>正文</p></body></html>")
        self.assertEqual(result["title"], "只有 title")


class TestKebab(unittest.TestCase):
    def test_chinese_and_special_chars(self):
        self.assertEqual(art.kebab(" 怎样/写*代码 "), "怎样-写-代码")

    def test_empty(self):
        self.assertEqual(art.kebab(""), "article")


class TestBuildNote(unittest.TestCase):
    def test_frontmatter_and_sections(self):
        note = art.build_note(
            {"title": "原文", "author": "作者A"},
            {"title": "总结标题", "one_sentence": "核心", "points": ["p1", "p2"],
             "key_info": [["概念", "说明"]], "summary": "总结"},
            "https://mp.weixin.qq.com/s/abc",
            created="2026-08-19",
        )
        self.assertIn("type: article-summary", note)
        self.assertIn("source: \"https://mp.weixin.qq.com/s/abc\"", note)
        self.assertIn("## 🎯 一句话核心", note)
        self.assertIn("核心", note)
        self.assertIn("## 💡 主要论据", note)
        self.assertIn("- p1", note)
        self.assertIn("## 📊 关键信息表", note)
        self.assertIn("| 概念 | 说明 |", note)


class TestSummarizeArticle(unittest.TestCase):
    def _mock_resp(self, content):
        resp = mock.Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "choices": [{"message": {"content": content}}],
            "usage": {"total_tokens": 100, "prompt_tokens": 60, "completion_tokens": 40},
        }
        return resp

    @mock.patch("article_summarizer.requests.post")
    def test_valid_json(self, mock_post):
        mock_post.return_value = self._mock_resp(json.dumps(
            {"title": "T", "one_sentence": "S", "points": ["p"],
             "key_info": [["k", "v"]], "summary": "sum"}, ensure_ascii=False))
        article = {"title": "原文", "author": "A", "content": "正文内容"}
        summary = art.summarize_article(article, "https://mp.weixin.qq.com/s/abc", MOCK_CONFIG)
        self.assertEqual(summary["title"], "T")
        self.assertEqual(summary["points"], ["p"])
        # 验证 prompt 从 .st 加载（非硬编码）：占位符应已渲染
        body = mock_post.call_args.kwargs["json"]
        self.assertNotIn("{{article_title}}", body["messages"][0]["content"])

    @mock.patch("article_summarizer.requests.post")
    def test_invalid_json_fallback(self, mock_post):
        mock_post.return_value = self._mock_resp("这不是 JSON，是一段文字总结。")
        article = {"title": "原文", "author": "A", "content": "正文"}
        summary = art.summarize_article(article, "url", MOCK_CONFIG)
        # 降级: summary 填充原文, 字段有默认值
        self.assertIn("一段文字总结", summary["summary"])
        self.assertEqual(summary["points"], [])

    def test_missing_api_key(self):
        article = {"title": "T", "author": "A", "content": "C"}
        with self.assertRaises(RuntimeError):
            art.summarize_article(article, "url", {"default": {"api_key": "sk-xxx"}})


if __name__ == "__main__":
    unittest.main()
