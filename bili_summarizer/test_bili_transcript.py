"""bili_transcript 单元测试

覆盖: BV号提取、时间戳格式化、Cookie加载、字幕下载格式化、策略降级
运行: python -m unittest test_bili_transcript -v
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bili_transcript as bt


class TestExtractBvid(unittest.TestCase):
    """BV号提取测试"""

    def test_plain_bvid(self):
        self.assertEqual(bt.extract_bvid("BV1xx411x7xx"), "BV1xx411x7xx")

    def test_full_url(self):
        self.assertEqual(
            bt.extract_bvid("https://www.bilibili.com/video/BV1xx411x7xx/"),
            "BV1xx411x7xx",
        )

    def test_url_with_query(self):
        self.assertEqual(
            bt.extract_bvid("https://www.bilibili.com/video/BV1xx411x7xx?p=2&spm_id_from=333.999"),
            "BV1xx411x7xx",
        )

    def test_short_url(self):
        self.assertEqual(
            bt.extract_bvid("https://b23.tv/BV1xx411x7xx"),
            "BV1xx411x7xx",
        )

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            bt.extract_bvid("https://example.com/not-a-bilibili-url")

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            bt.extract_bvid("")


class TestShortLinkResolve(unittest.TestCase):
    """b23.tv 不透明短码解析（OPT-115）：跟随重定向取 BV，全程 mock 不出网"""

    def setUp(self):
        self._orig_get = bt.requests.get

    def tearDown(self):
        bt.requests.get = self._orig_get

    def _fake_get(self, url="https://www.bilibili.com/video/BV1GwA3BUsxx?p=1", text="", exc=None):
        calls = []

        def fake_get(u, **kw):
            calls.append(u)
            if exc is not None:
                raise exc
            resp = type("R", (), {})
            resp.url = url
            resp.text = text
            return resp

        bt.requests.get = fake_get
        return calls

    def test_opaque_short_link_resolved_via_redirect(self):
        calls = self._fake_get()
        self.assertEqual(
            bt.extract_bvid("【DSH】挂一百个 Skill https://b23.tv/GwA3BUs 请处理"),
            "BV1GwA3BUsxx")
        self.assertEqual(calls, ["https://b23.tv/GwA3BUs"])

    def test_bili2233_domain_also_matched(self):
        self._fake_get()
        self.assertEqual(bt.extract_bvid("看这个 https://bili2233.cn/abcDEF"), "BV1GwA3BUsxx")

    def test_final_url_miss_falls_back_to_html(self):
        self._fake_get(url="https://www.bilibili.com/blackboard/html5mobileplayer.html",
                       text='<html>window.__INITIAL_STATE__={"bvid":"BV1GwA3BUsxx"}</html>')
        self.assertEqual(bt.extract_bvid("https://b23.tv/GwA3BUs"), "BV1GwA3BUsxx")

    def test_network_failure_raises_with_reason(self):
        self._fake_get(exc=bt.requests.exceptions.ConnectionError("reset"))
        with self.assertRaises(ValueError) as ctx:
            bt.extract_bvid("https://b23.tv/GwA3BUs")
        self.assertIn("短链解析失败", str(ctx.exception))

    def test_no_bv_found_raises(self):
        self._fake_get(url="https://www.bilibili.com/read/cv123456", text="专栏页没有BV")
        with self.assertRaises(ValueError) as ctx:
            bt.extract_bvid("https://b23.tv/GwA3BUs")
        self.assertIn("短链未解析出 BV 号", str(ctx.exception))

    def test_non_bili_domain_never_hits_network(self):
        calls = self._fake_get()
        with self.assertRaises(ValueError):
            bt.extract_bvid("https://example.com/not-a-bilibili-url")
        self.assertEqual(calls, [], "非 B站短域不发起解析请求")


class TestFormatTimestamp(unittest.TestCase):
    """时间戳格式化测试"""

    def test_zero(self):
        self.assertEqual(bt.format_timestamp(0), "[0:00]")

    def test_minutes(self):
        self.assertEqual(bt.format_timestamp(65), "[1:05]")

    def test_hours(self):
        self.assertEqual(bt.format_timestamp(3725), "[1:02:05]")

    def test_float_input(self):
        self.assertEqual(bt.format_timestamp(65.9), "[1:05]")

    def test_padding(self):
        self.assertEqual(bt.format_timestamp(7), "[0:07]")


class TestCookie(unittest.TestCase):
    """Cookie 加载测试"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_missing_cookie(self):
        with patch("builtins.print"):
            result = bt.load_cookie(os.path.join(self.tmpdir, "nonexist.json"))
        self.assertEqual(result, {})

    def test_valid_cookie(self):
        cookie_path = os.path.join(self.tmpdir, "cookie.json")
        with open(cookie_path, "w", encoding="utf-8") as f:
            f.write('{"SESSDATA": "abc123", "BILI_JCT": "def456"}')
        result = bt.load_cookie(cookie_path)
        self.assertEqual(result["SESSDATA"], "abc123")
        self.assertEqual(result["BILI_JCT"], "def456")

    def test_invalid_json(self):
        cookie_path = os.path.join(self.tmpdir, "bad.json")
        with open(cookie_path, "w", encoding="utf-8") as f:
            f.write("not json {{{")
        with patch("builtins.print"):
            result = bt.load_cookie(cookie_path)
        self.assertEqual(result, {})


class TestCookieToHeader(unittest.TestCase):
    """Cookie → HTTP Header 转换测试"""

    def test_normal(self):
        result = bt.cookie_to_header({"SESSDATA": "a", "BILI_JCT": "b"})
        self.assertEqual(result, "SESSDATA=a; BILI_JCT=b")

    def test_empty_values_skipped(self):
        result = bt.cookie_to_header({"SESSDATA": "a", "EMPTY": ""})
        self.assertNotIn("EMPTY", result)
        self.assertIn("SESSDATA=a", result)


class TestDownloadSubtitle(unittest.TestCase):
    """字幕下载与格式化测试"""

    def test_format_subtitle_body(self):
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "body": [
                {"from": 0, "to": 3.5, "content": "第一句"},
                {"from": 3.5, "to": 7.2, "content": "第二句"},
                {"from": 3725, "to": 3728, "content": "长视频部分"},
            ]
        }
        with patch("bili_transcript.requests.get", return_value=mock_resp):
            text = bt.download_subtitle("/subtitle/abc.json")
        self.assertIn("[0:00] 第一句", text)
        self.assertIn("[0:03] 第二句", text)
        self.assertIn("[1:02:05] 长视频部分", text)

    def test_url_prefix_added(self):
        """相对 URL 应自动补 https: 前缀"""
        mock_resp = Mock()
        mock_resp.json.return_value = {"body": [{"from": 0, "to": 1, "content": "x"}]}
        with patch("bili_transcript.requests.get", return_value=mock_resp) as mock_get:
            bt.download_subtitle("//subtitle/abc.json")
            url = mock_get.call_args[0][0]
            self.assertTrue(url.startswith("https:"))


class TestStrategyApi(unittest.TestCase):
    """策略1: API 字幕获取测试（WBI 签名优先 + AI字幕默认启用）"""

    def test_cc_subtitle_success(self):
        """人工CC字幕：直接信任，无需验证"""
        with patch("bili_transcript.get_video_info",
                   return_value={"cid": 123, "title": "测试视频", "bvid": "BV1xx411x7xx", "duration": 300}), \
             patch("bili_transcript.get_subtitle_url_wbi", return_value=("/sub/1.json", False)), \
             patch("bili_transcript.download_subtitle", return_value="[0:00] 这是一个测试视频的内容"), \
             patch("builtins.print"):
            title, transcript = bt.strategy_api("BV1xx411x7xx", {})
        self.assertEqual(title, "测试视频")
        self.assertIn("测试视频", transcript)

    def test_ai_subtitle_default_use_pass_validation(self):
        """AI字幕默认启用：下载 + validate_subtitle 通过 → 成功（无需 --trust-ai）"""
        with patch("bili_transcript.get_video_info",
                   return_value={"cid": 123, "title": "测试视频", "bvid": "BV1xx411x7xx", "duration": 300}), \
             patch("bili_transcript.get_subtitle_url_wbi", return_value=("/sub/1.json", True)), \
             patch("bili_transcript.download_subtitle", return_value="[0:00] 这是一个测试视频的内容"), \
             patch("builtins.print"):
            title, transcript = bt.strategy_api("BV1xx411x7xx", {})
        self.assertEqual(title, "测试视频")
        self.assertIn("测试视频", transcript)

    def test_ai_subtitle_fail_validation_retry(self):
        """AI字幕默认启用：validate_subtitle 失败 → 重试2次后降级"""
        with patch("bili_transcript.get_video_info",
                   return_value={"cid": 123, "title": "测试视频", "bvid": "BV1xx411x7xx", "duration": 300}), \
             patch("bili_transcript.get_subtitle_url_wbi", return_value=("/sub/1.json", True)), \
             patch("bili_transcript.download_subtitle", return_value="[0:00] 完全不相关的内容"), \
             patch("builtins.print"):
            result = bt.strategy_api("BV1xx411x7xx", {})
        self.assertIsNone(result)  # AI字幕不匹配 → 降级策略2

    def test_ai_subtitle_trust_ai_pass_validation(self):
        """向后兼容：--trust-ai 参数仍可传，不影响默认启用行为"""
        with patch("bili_transcript.get_video_info",
                   return_value={"cid": 123, "title": "测试视频", "bvid": "BV1xx411x7xx", "duration": 300}), \
             patch("bili_transcript.get_subtitle_url_wbi", return_value=("/sub/1.json", True)), \
             patch("bili_transcript.download_subtitle", return_value="[0:00] 这是一个测试视频的内容"), \
             patch("builtins.print"):
            title, transcript = bt.strategy_api("BV1xx411x7xx", {}, trust_ai=True)
        self.assertEqual(title, "测试视频")

    def test_wbi_unavailable_fallback_legacy(self):
        """WBI 路径不可用（返回 None）→ 回退传统 get_subtitle_url，仍可拿 CC 字幕"""
        with patch("bili_transcript.get_video_info",
                   return_value={"cid": 123, "title": "测试视频", "bvid": "BV1xx411x7xx", "duration": 300}), \
             patch("bili_transcript.get_subtitle_url_wbi", return_value=None), \
             patch("bili_transcript.get_subtitle_url", return_value=("/sub/1.json", False)), \
             patch("bili_transcript.download_subtitle", return_value="[0:00] 这是一个测试视频的内容"), \
             patch("builtins.print"):
            title, transcript = bt.strategy_api("BV1xx411x7xx", {})
        self.assertEqual(title, "测试视频")

    def test_no_subtitle_fallback(self):
        with patch("bili_transcript.get_video_info",
                   return_value={"cid": 123, "title": "t", "bvid": "BV1xx411x7xx"}), \
             patch("bili_transcript.get_subtitle_url_wbi", return_value=None), \
             patch("bili_transcript.get_subtitle_url", return_value=None), \
             patch("builtins.print"):
            result = bt.strategy_api("BV1xx411x7xx", {})
        self.assertIsNone(result)

    def test_exception_fallback(self):
        with patch("bili_transcript.get_video_info",
                   side_effect=RuntimeError("API错误")), \
             patch("builtins.print"):
            result = bt.strategy_api("BV1xx411x7xx", {})
        self.assertIsNone(result)


class TestGetVideoMeta(unittest.TestCase):
    """元数据获取测试（用于笔记 YAML frontmatter，失败不阻断流程）"""

    def test_meta_success(self):
        """正常返回 → 含 title/author/date/duration/source_url"""
        fake_resp = MagicMock()
        fake_resp.json.return_value = {
            "code": 0,
            "data": {
                "title": "测试视频",
                "owner": {"name": "测试作者"},
                "pubdate": 1721361600,  # 2024-07-19 08:00:00 UTC
                "duration": 775,
            },
        }
        with patch("bili_transcript.requests.get", return_value=fake_resp):
            meta = bt.get_video_meta("BV1xx411x7xx", {})
        self.assertEqual(meta["title"], "测试视频")
        self.assertEqual(meta["author"], "测试作者")
        self.assertEqual(meta["duration"], 775)
        self.assertEqual(meta["bvid"], "BV1xx411x7xx")
        self.assertEqual(meta["source_url"], "https://www.bilibili.com/video/BV1xx411x7xx")
        self.assertTrue(meta["date"])  # 日期非空

    def test_meta_api_error_code(self):
        """API 返回非 0 code → 返回 fallback（不抛异常）"""
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"code": -404, "message": "不存在"}
        with patch("bili_transcript.requests.get", return_value=fake_resp), \
             patch("builtins.print"):
            meta = bt.get_video_meta("BV1xx411x7xx", {})
        self.assertEqual(meta["title"], "BV1xx411x7xx")
        self.assertEqual(meta["author"], "")

    def test_meta_network_exception(self):
        """网络异常 → 返回 fallback（不抛异常，不阻断字幕流程）"""
        with patch("bili_transcript.requests.get", side_effect=Exception("timeout")), \
             patch("builtins.print"):
            meta = bt.get_video_meta("BV1xx411x7xx", {})
        self.assertEqual(meta["title"], "BV1xx411x7xx")
        self.assertEqual(meta["bvid"], "BV1xx411x7xx")


class TestSubtitleLangPriority(unittest.TestCase):
    """字幕语言优先级测试（CC优先 > AI降级）"""

    def test_cc_preferred_over_ai(self):
        """CC字幕和AI字幕同时存在 → 优先CC"""
        subtitles = [
            {"lan": "ai-zh", "subtitle_url": "/ai"},
            {"lan": "zh-Hans", "subtitle_url": "/cc"},
        ]
        with patch("bili_transcript.requests.get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = {
                "data": {"subtitle": {"subtitles": subtitles}}
            }
            mock_get.return_value = mock_resp
            result = bt.get_subtitle_url(123, "BV1xx411x7xx", {})
        self.assertEqual(result, ("/cc", False))

    def test_ai_returned_when_no_cc(self):
        """只有AI字幕 → 返回AI标记"""
        subtitles = [{"lan": "ai-zh", "subtitle_url": "/ai"}]
        with patch("bili_transcript.requests.get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = {
                "data": {"subtitle": {"subtitles": subtitles}}
            }
            mock_get.return_value = mock_resp
            result = bt.get_subtitle_url(123, "BV1xx411x7xx", {})
        self.assertEqual(result, ("/ai", True))

    def test_fallback_first_subtitle_marked_ai(self):
        """未知语言 → 兜底取第一个，标记为AI"""
        subtitles = [{"lan": "ja-JP", "subtitle_url": "/japanese"}]
        with patch("bili_transcript.requests.get") as mock_get:
            mock_resp = Mock()
            mock_resp.json.return_value = {
                "data": {"subtitle": {"subtitles": subtitles}}
            }
            mock_get.return_value = mock_resp
            result = bt.get_subtitle_url(123, "BV1xx411x7xx", {})
        self.assertEqual(result, ("/japanese", True))


class TestValidateSubtitle(unittest.TestCase):
    """字幕内容验证测试（防止 B站 AI 字幕缓存错误）"""

    def test_title_keywords_match(self):
        """标题关键词在字幕中出现 → 通过"""
        title = "RAG检索增强生成技术详解"
        transcript = "[0:00] 今天我们来聊聊RAG技术 [0:05] RAG是检索增强生成的缩写"
        self.assertTrue(bt.validate_subtitle(title, transcript))

    def test_title_keywords_mismatch(self):
        """标题关键词在字幕中均未出现 → 不通过"""
        title = "2.55万亿冲关失败，追高全是泪"
        transcript = "[0:00] 自从我加入WB以来 [0:10] 在对阵BLG的比赛"
        self.assertFalse(bt.validate_subtitle(title, transcript))

    def test_no_keywords_skip(self):
        """无法提取关键词 → 跳过检查（返回True）"""
        title = "!!！"
        transcript = "任意内容"
        self.assertTrue(bt.validate_subtitle(title, transcript))

    def test_timestamp_within_duration(self):
        """时间戳在视频时长内 → 通过"""
        title = "测试视频"
        transcript = "[0:00] 这是一个测试视频 [5:00] 结尾"
        self.assertTrue(bt.validate_subtitle(title, transcript, duration=400))

    def test_timestamp_exceeds_duration(self):
        """时间戳远超视频时长 → 不通过"""
        title = "测试视频"
        transcript = "[0:00] 这是一个测试视频 [20:00] 结尾"
        self.assertFalse(bt.validate_subtitle(title, transcript, duration=100))

    def test_english_keyword_match(self):
        """英文关键词匹配"""
        title = "Understanding Transformer Architecture"
        transcript = "[0:00] The Transformer model was introduced in 2017"
        self.assertTrue(bt.validate_subtitle(title, transcript))


class TestRefreshCookie(unittest.TestCase):
    """Cookie 自动刷新测试"""

    def test_no_ac_time_value_skip(self):
        """没有 ac_time_value → 跳过刷新"""
        cookie = {"SESSDATA": "abc", "BILI_JCT": "def"}
        result = bt.refresh_cookie(cookie)
        self.assertEqual(result, cookie)

    def test_no_sessdata_skip(self):
        """没有 SESSDATA → 跳过刷新"""
        cookie = {"BILI_JCT": "def", "ac_time_value": "xxx"}
        result = bt.refresh_cookie(cookie)
        self.assertEqual(result, cookie)

    def test_cookie_still_valid(self):
        """Cookie 仍有效（刷新接口返回 code!=0）→ 不更新"""
        cookie = {"SESSDATA": "abc", "BILI_JCT": "def", "ac_time_value": "xxx"}
        mock_resp = Mock()
        mock_resp.json.return_value = {"code": -101, "message": "not needed"}
        with patch("bili_transcript.requests.post", return_value=mock_resp), \
             patch("builtins.print"):
            result = bt.refresh_cookie(cookie)
        self.assertEqual(result["SESSDATA"], "abc")

    def test_refresh_success(self):
        """Cookie 过期 → 刷新成功 → 更新并回写"""
        cookie = {"SESSDATA": "old", "BILI_JCT": "jct", "ac_time_value": "ac"}
        # 刷新请求返回 code=0 + 新cookie
        refresh_resp = Mock()
        refresh_resp.json.return_value = {
            "code": 0,
            "data": {
                "cookie_info": {
                    "cookies": [
                        {"name": "SESSDATA", "value": "new_sessdata"},
                        {"name": "BILI_JCT", "value": "new_jct"},
                    ]
                }
            }
        }
        tmpdir = tempfile.mkdtemp()
        cookie_path = os.path.join(tmpdir, "cookie.json")
        with open(cookie_path, "w") as f:
            json.dump(cookie, f)

        with patch("bili_transcript.requests.post", return_value=refresh_resp), \
             patch("builtins.print"):
            result = bt.refresh_cookie(cookie, cookie_path)

        self.assertEqual(result["SESSDATA"], "new_sessdata")
        self.assertEqual(result["BILI_JCT"], "new_jct")
        # 验证回写
        with open(cookie_path, "r") as f:
            saved = json.load(f)
        self.assertEqual(saved["SESSDATA"], "new_sessdata")

    def test_refresh_exception_returns_original(self):
        """刷新异常 → 返回原cookie"""
        cookie = {"SESSDATA": "abc", "BILI_JCT": "def", "ac_time_value": "xxx"}
        with patch("bili_transcript.requests.get", side_effect=Exception("network")), \
             patch("builtins.print"):
            result = bt.refresh_cookie(cookie)
        self.assertEqual(result, cookie)


class TestTranscribeCache(unittest.TestCase):
    """Whisper 转写断点续传测试"""

    def test_load_cache_not_exist(self):
        """缓存文件不存在 → 返回空列表"""
        result = __import__("transcribe").load_cache("/nonexistent/path.cache")
        self.assertEqual(result, [])

    def test_load_cache_valid(self):
        """缓存文件有效 → 返回段落列表"""
        import transcribe
        tmpdir = tempfile.mkdtemp()
        cache_path = os.path.join(tmpdir, "test.cache")
        segments = [{"start": 0.0, "end": 5.0, "text": "第一句"}]
        transcribe.save_cache(cache_path, segments)
        result = transcribe.load_cache(cache_path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "第一句")

    def test_load_cache_corrupted(self):
        """缓存文件损坏 → 返回空列表"""
        import transcribe
        tmpdir = tempfile.mkdtemp()
        cache_path = os.path.join(tmpdir, "corrupt.cache")
        with open(cache_path, "w") as f:
            f.write("not json {{{")
        result = transcribe.load_cache(cache_path)
        self.assertEqual(result, [])

    def test_save_and_load_roundtrip(self):
        """保存后加载 → 数据一致"""
        import transcribe
        tmpdir = tempfile.mkdtemp()
        cache_path = os.path.join(tmpdir, "round.cache")
        segments = [
            {"start": 0.0, "end": 5.0, "text": "第一句"},
            {"start": 5.0, "end": 10.0, "text": "第二句"},
        ]
        transcribe.save_cache(cache_path, segments)
        result = transcribe.load_cache(cache_path)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1]["text"], "第二句")


class TestEnsureFfmpeg(unittest.TestCase):
    """ffmpeg 检测测试"""

    def test_system_ffmpeg(self):
        with patch("bili_transcript.shutil.which", return_value="C:/ffmpeg.exe"):
            bt.ensure_ffmpeg()  # 不应抛异常

    def test_imageio_fallback(self):
        with patch("bili_transcript.shutil.which", return_value=None), \
             patch("bili_transcript.os.environ", {"PATH": ""}), \
             patch("builtins.print"):
            bt.ensure_ffmpeg()  # 不应抛异常


class TestTranscriptCache(unittest.TestCase):
    """字幕缓存复用（`transcript_{bvid}.txt`）测试"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._old_cwd = os.getcwd()
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self._old_cwd)

    def test_miss_when_no_file(self):
        self.assertIsNone(bt.check_transcript_cache("BV1xx411x7xx"))

    def test_hit_reads_transcript(self):
        with open("transcript_BV1xx411x7xx.txt", "w", encoding="utf-8") as f:
            f.write("[0:00] 内容一\n[1:00] 内容二\n")
        title, transcript = bt.check_transcript_cache("BV1xx411x7xx")
        self.assertEqual(title, "BV1xx411x7xx")  # 无meta文件，title回退 bvid
        self.assertIn("内容一", transcript)

    def test_hit_uses_meta_title(self):
        with open("transcript_BV1xx411x7xx.txt", "w", encoding="utf-8") as f:
            f.write("[0:00] 内容\n")
        with open("meta_BV1xx411x7xx.json", "w", encoding="utf-8") as f:
            json.dump({"title": "真实标题"}, f)
        title, _ = bt.check_transcript_cache("BV1xx411x7xx")
        self.assertEqual(title, "真实标题")

    def test_empty_transcript_is_miss(self):
        with open("transcript_BV1xx411x7xx.txt", "w", encoding="utf-8") as f:
            f.write("   \n")
        self.assertIsNone(bt.check_transcript_cache("BV1xx411x7xx"))

    def test_corrupt_meta_falls_back_bvid(self):
        with open("transcript_BV1xx411x7xx.txt", "w", encoding="utf-8") as f:
            f.write("[0:00] 内容\n")
        with open("meta_BV1xx411x7xx.json", "w", encoding="utf-8") as f:
            f.write("not json{{{")
        title, _ = bt.check_transcript_cache("BV1xx411x7xx")
        self.assertEqual(title, "BV1xx411x7xx")


class TestSubtitleWbi(unittest.TestCase):
    """_pick_subtitle_url 选择逻辑 + WBI 路径降级测试"""

    def test_pick_cc_preferred_over_ai(self):
        subs = [{"lan": "ai-zh", "subtitle_url": "/ai"},
                {"lan": "zh-Hans", "subtitle_url": "/cc"}]
        self.assertEqual(bt._pick_subtitle_url(subs), ("/cc", False))

    def test_pick_ai_fallback(self):
        subs = [{"lan": "ai-zh", "subtitle_url": "/ai"}]
        self.assertEqual(bt._pick_subtitle_url(subs), ("/ai", True))

    def test_pick_unknown_lang_marks_ai(self):
        subs = [{"lan": "ja-JP", "subtitle_url": "/japanese"}]
        self.assertEqual(bt._pick_subtitle_url(subs), ("/japanese", True))

    def test_pick_empty(self):
        self.assertIsNone(bt._pick_subtitle_url([]))

    def test_wbi_missing_lib_returns_none(self):
        from unittest.mock import patch as _patch
        import builtins as _builtins
        real_import = _builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "bilibili_api":
                raise ImportError("no module named bilibili_api")
            return real_import(name, *a, **kw)

        with _patch("builtins.__import__", side_effect=fake_import), \
             _patch("builtins.print"):
            self.assertIsNone(bt.get_subtitle_url_wbi(1, "BV1xx411x7xx", {}))


class TestChannelBatch(unittest.TestCase):
    """频道级批量摄取测试（设计文档4.0 执行线#4）：枚举/批量文件/入队，全程 mock 零网络"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _fake_ytdlp(self, info):
        """构造假 yt_dlp 模块：YoutubeDL(opts) 上下文管理器，extract_info 返回 info"""
        fake = MagicMock()
        fake.YoutubeDL.return_value.__enter__.return_value.extract_info.return_value = info
        return fake

    def test_enumerate_channel_normalizes_and_skips(self):
        """① flat entries 归一化为 {bvid, title}；无 BV/无法提取的条目跳过并告警"""
        info = {"entries": [
            {"id": "BV1xx411x7xx", "title": "a"},
            {"url": "https://www.bilibili.com/video/BV1yy411x7yy", "title": "b"},
            {},
            {"id": "zzz"},
        ]}
        fake = self._fake_ytdlp(info)
        with patch.object(bt, "yt_dlp", fake), patch("builtins.print"):
            items = bt.enumerate_channel("https://space.bilibili.com/123/video")
        self.assertEqual(items, [
            {"bvid": "BV1xx411x7xx", "title": "a"},
            {"bvid": "BV1yy411x7yy", "title": "b"},
        ])
        # flat 枚举参数 + extract_info 调用契约
        self.assertEqual(
            fake.YoutubeDL.call_args[0][0],
            {"extract_flat": "in_playlist", "quiet": True,
             "no_warnings": True, "skip_download": True})
        extract_info = fake.YoutubeDL.return_value.__enter__.return_value.extract_info
        extract_info.assert_called_once_with("https://space.bilibili.com/123/video",
                                             download=False)

    def test_enumerate_channel_max_items(self):
        """max_items 截断枚举结果"""
        info = {"entries": [{"id": f"BV1aaa{i:06d}", "title": str(i)} for i in range(5)]}
        fake = self._fake_ytdlp(info)
        with patch.object(bt, "yt_dlp", fake), patch("builtins.print"):
            items = bt.enumerate_channel("https://space.bilibili.com/1/video", max_items=2)
        self.assertEqual([i["bvid"] for i in items], ["BV1aaa000000", "BV1aaa000001"])

    def test_enumerate_channel_error_propagates(self):
        """枚举异常（网络/解析失败）向上抛，由调用方处理"""
        fake = MagicMock()
        extract_info = fake.YoutubeDL.return_value.__enter__.return_value.extract_info
        extract_info.side_effect = RuntimeError("network down")
        with patch.object(bt, "yt_dlp", fake), patch("builtins.print"):
            with self.assertRaises(RuntimeError):
                bt.enumerate_channel("https://space.bilibili.com/1/video")

    def test_parse_batch_file_mixed_lines(self):
        """② 批量文件：BV/URL 归一，空行与 # 注释忽略，坏行跳过"""
        path = os.path.join(self.tmpdir, "batch.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# 频道精选\n"
                    "\n"
                    "BV1xx411x7xx\n"
                    "https://www.bilibili.com/video/BV1yy411x7yy?p=1\n"
                    "   \n"
                    "not-a-bv-line\n")
        with patch("builtins.print"):
            bvids = bt.parse_batch_file(path)
        self.assertEqual(bvids, ["BV1xx411x7xx", "BV1yy411x7yy"])

    def test_enqueue_bili_batch_writes_sqlite(self):
        """③ 入队 SQLite 字段与 inbox 队列任务同构；source 可自定义；空列表不入队"""
        queue_path = os.path.join(self.tmpdir, "sub", "queue.jsonl")
        with patch("builtins.print"):
            count = bt.enqueue_bili_batch(queue_path, ["BV1xx411x7xx", "BV1yy411x7yy"])
        self.assertEqual(count, 2)
        from queue_store import InboxQueueStore
        store = InboxQueueStore(os.path.join(self.tmpdir, "sub", "queue.db"))
        rows = store.list_tasks()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["url"], "https://www.bilibili.com/video/BV1xx411x7xx")
        self.assertEqual(rows[1]["url"], "https://www.bilibili.com/video/BV1yy411x7yy")
        for i, row in enumerate(rows, 1):
            self.assertRegex(row["id"], r"^\d{14}-\d{3}$")
            self.assertEqual(row["id"].split("-")[1], f"{i:03d}")  # 时间戳+序号
            self.assertEqual(row["type"], "bili")
            self.assertEqual(row["status"], "pending")
            self.assertEqual(row["retry_count"], 0)
            self.assertEqual(row["email_subject"], "channel-batch")
            self.assertEqual(row["source"], "bili_channel")
            self.assertIsNone(row["error"])
            self.assertTrue(row["received_at"])
        # source 参数可自定义
        queue2 = os.path.join(self.tmpdir, "q2.jsonl")
        with patch("builtins.print"):
            n = bt.enqueue_bili_batch(queue2, ["BV1zz411x7zz"], source="custom_src")
        self.assertEqual(n, 1)
        store2 = InboxQueueStore(os.path.join(self.tmpdir, "q2.db"))
        row = store2.list_tasks()[0]
        self.assertEqual(row["source"], "custom_src")
        # 跨次调用同一 BV 由数据库唯一约束去重
        with patch("builtins.print"):
            self.assertEqual(bt.enqueue_bili_batch(queue2, ["BV1zz411x7zz"]), 0)
        self.assertEqual(len(store2.list_tasks()), 1)
        # 空列表 → 0 条且不写文件
        queue3 = os.path.join(self.tmpdir, "q3.jsonl")
        with patch("builtins.print"):
            self.assertEqual(bt.enqueue_bili_batch(queue3, []), 0)
        self.assertFalse(os.path.exists(queue3))
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "q3.db")))

    def test_main_dry_run_does_not_write_queue(self):
        """④ --dry-run 只打印不入队（不触达 enqueue，也不写队列文件）"""
        queue_path = os.path.join(self.tmpdir, "queue.jsonl")
        argv = ["bili_transcript.py",
                "--channel", "https://space.bilibili.com/123/video",
                "--limit", "5", "--dry-run", "--queue-path", queue_path]
        with patch.object(sys, "argv", argv), \
             patch.object(bt, "enumerate_channel",
                          return_value=[{"bvid": "BV1xx411x7xx", "title": "t"}]) as m_enum, \
             patch.object(bt, "enqueue_bili_batch") as m_enqueue, \
             patch("builtins.print"):
            bt.main()
        m_enum.assert_called_once_with("https://space.bilibili.com/123/video", max_items=5)
        m_enqueue.assert_not_called()
        self.assertFalse(os.path.exists(queue_path))

    def test_argparse_channel_batch_flags(self):
        """⑤ --channel/--limit/--dry-run/--queue-path 组合可解析；默认队列指向仓库根 inbox"""
        parser = bt.build_arg_parser()
        args = parser.parse_args([
            "--channel", "https://space.bilibili.com/123/video",
            "--limit", "5", "--dry-run", "--queue-path", "X:/tmp/q.jsonl",
        ])
        self.assertEqual(args.channel, "https://space.bilibili.com/123/video")
        self.assertEqual(args.limit, 5)
        self.assertTrue(args.dry_run)
        self.assertEqual(args.queue_path, "X:/tmp/q.jsonl")
        self.assertIsNone(args.input)      # 批量模式下位置参数可省略
        self.assertIsNone(args.batch_file)
        # 默认队列路径 = 仓库根/inbox/queue.jsonl
        default = bt.build_arg_parser().parse_args([])
        self.assertEqual(
            os.path.normpath(default.queue_path),
            os.path.normpath(os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(bt.__file__))),
                "inbox", "queue.jsonl")))


if __name__ == "__main__":
    unittest.main()
