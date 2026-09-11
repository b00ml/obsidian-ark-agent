"""多模态功能单元测试 (video_frames + visual_analyzer + 集成)

覆盖: 智能调度、截图标记生成、时间戳容错提取、视觉分析调用、
      base64 编码、网格图拼接、配置加载、模式集成
运行: python -m unittest test_multimodal -v
"""
import base64
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

import prompt_loader
import video_frames as vf
import visual_analyzer as va
import bili_transcript as bt


def make_test_image(path: str, size=(64, 36), color=(10, 143, 117)) -> str:
    """创建测试图片（避免依赖 ffmpeg）"""
    img = Image.new("RGB", size, color)
    img.save(path, "JPEG", quality=85)
    return path


class TestPromptLoader(unittest.TestCase):
    """Prompt 加载与渲染测试（LLM 调用标准化）"""

    def setUp(self):
        prompt_loader.clear_cache()

    def test_load_prompt_visual_analyze(self):
        """visual-analyze-user.st 存在且含占位符"""
        template = prompt_loader.load_prompt("visual-analyze-user")
        self.assertIn("{{title}}", template)
        self.assertIn("{{transcript}}", template)

    def test_load_prompt_screenshot_markers(self):
        """screenshot-markers-user.st 存在且含占位符"""
        template = prompt_loader.load_prompt("screenshot-markers-user")
        self.assertIn("{{title}}", template)
        self.assertIn("{{transcript}}", template)
        self.assertIn("{{max_count}}", template)

    def test_load_prompt_missing_raises(self):
        """不存在的 prompt 抛出异常"""
        with self.assertRaises(prompt_loader.PromptNotFoundError):
            prompt_loader.load_prompt("nonexistent")

    def test_render_placeholders(self):
        """render 正确注入占位符"""
        template = "标题: {{title}}\n字幕: {{transcript}}"
        result = prompt_loader.render(template, {"title": "视频A", "transcript": "内容"})
        self.assertEqual(result, "标题: 视频A\n字幕: 内容")

    def test_render_missing_param_blank(self):
        """缺失占位符留空串，不抛错"""
        result = prompt_loader.render("a={{a}}", {"b": "1"})
        self.assertEqual(result, "a=")

    def test_render_no_format_conflict(self):
        """render 不破坏模板中的 JSON 大括号（避免 .format 冲突）"""
        template = '返回 JSON: [{"timestamp": 秒}]'
        result = prompt_loader.render(template, {})
        self.assertIn('{"timestamp": 秒}', result)

    def test_analyze_uses_prompt_loader(self):
        """analyze_with_visual 的 prompt 来自 .st 文件而非硬编码"""
        with patch.object(va.requests, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "ok"}}], "usage": {}}
            mock_post.return_value = mock_resp
            config = {"api_base": "https://api.example.com/v1",
                      "api_key": "sk-test", "model": "m"}
            va.analyze_with_visual("字幕", "b64", "标题", config)
            sent_text = mock_post.call_args.kwargs["json"]["messages"][0]["content"][0]["text"]
            # prompt 加载了模板并渲染了标题
            self.assertIn("标题", sent_text)
            self.assertIn("字幕", sent_text)


class TestShouldUseVisual(unittest.TestCase):
    """智能调度测试"""

    def test_short_video_with_keywords(self):
        transcript = "这个演示教程展示代码操作和效果对比"
        self.assertTrue(va.should_use_visual(transcript, "教程"))

    def test_short_video_without_keywords(self):
        transcript = "今天我们来聊一聊最近发生的事情"
        self.assertFalse(va.should_use_visual(transcript, "闲聊"))

    def test_long_video(self):
        # 长视频即使有关键词也不启用（成本控制）
        transcript = "演示" * 3000  # 超过 5000 字
        self.assertFalse(va.should_use_visual(transcript, "长视频"))

    def test_medium_video_low_keywords(self):
        transcript = "这是一个教程但是内容比较长" + "内容" * 500
        self.assertFalse(va.should_use_visual(transcript, "教程"))


class TestConfigLoad(unittest.TestCase):
    """配置加载测试"""

    def test_load_visual_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "visual_models.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({"default": {"api_key": "sk-xxx", "model": "deepseek-chat"}}, f)
            cfg = va.load_visual_config(cfg_path)
            self.assertEqual(cfg["default"]["model"], "deepseek-chat")

    def test_config_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            va.load_visual_config("/nonexistent/path/visual_models.json")


class TestScreenshotMarkers(unittest.TestCase):
    """截图时间点标记生成测试"""

    @patch.object(va.requests, "post")
    def test_valid_json(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '[{"timestamp": 120.5, "reason": "代码展示"}]'}}],
            "usage": {},
        }
        mock_post.return_value = mock_resp

        config = {"api_base": "https://api.example.com/v1", "api_key": "sk-test", "model": "m"}
        markers = va.generate_screenshot_markers("字幕内容", "标题", config)
        self.assertEqual(markers[0]["timestamp"], 120.5)
        self.assertEqual(markers[0]["reason"], "代码展示")

    @patch.object(va.requests, "post")
    def test_invalid_json_fallback_to_timestamps(self, mock_post):
        # 模型返回非严格 JSON，含 [M:SS] 时间戳
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "建议在 [2:05] 截图，以及 [10:30]"}}],
            "usage": {},
        }
        mock_post.return_value = mock_resp

        config = {"api_base": "https://api.example.com/v1", "api_key": "sk-test", "model": "m"}
        markers = va.generate_screenshot_markers("字幕内容", "标题", config)
        self.assertEqual(len(markers), 2)
        self.assertEqual(markers[0]["timestamp"], 125.0)  # 2:05 = 125s

    @patch.object(va.requests, "post")
    def test_empty_content(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "无截图必要"}}]}
        mock_post.return_value = mock_resp

        config = {"api_base": "https://api.example.com/v1", "api_key": "sk-test", "model": "m"}
        markers = va.generate_screenshot_markers("字幕内容", "标题", config)
        self.assertEqual(markers, [])

    def test_extract_timestamps(self):
        text = "在 [1:23] 出现图表，[12:34] 展示代码"
        markers = va._extract_timestamps_from_text(text, 5)
        self.assertEqual(len(markers), 2)
        self.assertEqual(markers[0]["timestamp"], 83.0)  # 1:23 = 83s
        self.assertEqual(markers[1]["timestamp"], 754.0)  # 12:34 = 754s

    def test_extract_timestamps_hours(self):
        text = "在 [1:02:03] 处"
        markers = va._extract_timestamps_from_text(text, 5)
        self.assertEqual(markers[0]["timestamp"], 3723.0)  # 1:02:03 = 3723s


class TestAnalyzeWithVisual(unittest.TestCase):
    """视觉模型分析测试"""

    @patch.object(va.requests, "post")
    def test_analyze_with_visual(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "画面显示操作步骤..."}}],
            "usage": {"total_tokens": 100},
        }
        mock_post.return_value = mock_resp

        config = {"api_base": "https://api.example.com/v1", "api_key": "sk-test",
                  "model": "qwen-vl-max"}
        result = va.analyze_with_visual("字幕", "base64==", "标题", config)
        self.assertEqual(result["content"], "画面显示操作步骤...")
        self.assertEqual(result["model"], "qwen-vl-max")
        self.assertEqual(result["usage"]["total_tokens"], 100)

        # 验证图片 base64 以 data:image/jpeg 前缀发送
        call_json = mock_post.call_args.kwargs["json"]
        img_content = call_json["messages"][0]["content"][1]
        self.assertIn("data:image/jpeg;base64,", img_content["image_url"]["url"])

    @patch.object(va.requests, "post")
    def test_invalid_api_key_raises(self, mock_post):
        config = {"api_base": "https://api.example.com/v1", "api_key": "sk-xxx", "model": "m"}
        with self.assertRaises(RuntimeError):
            va.analyze_with_visual("字幕", "b64", "标题", config)
        mock_post.assert_not_called()


class TestVideoFrames(unittest.TestCase):
    """video_frames 功能测试（用 PIL 生成测试图，避免依赖 ffmpeg）"""

    def test_image_to_base64(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.jpg")
            make_test_image(path)
            b64 = vf.image_to_base64(path)
            # 验证可解码回原图
            decoded = base64.b64decode(b64)
            self.assertEqual(len(decoded), os.path.getsize(path))

    def test_build_grid_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = [os.path.join(tmp, f"f{i}.jpg") for i in range(4)]
            for f in frames:
                make_test_image(f)
            out = os.path.join(tmp, "grid.jpg")
            result = vf.build_grid_image(frames, cols=2, rows=2,
                                         cell_w=64, cell_h=36, output=out)
            self.assertTrue(os.path.exists(out))
            with Image.open(out) as img:
                self.assertEqual(img.size, (128, 72))  # 2*64 x 2*36

    def test_build_grid_empty_raises(self):
        with self.assertRaises(ValueError):
            vf.build_grid_image([], output="x.jpg")

    def test_build_grid_partial_group(self):
        # 不足 cols*rows 也拼接（取可用帧）
        with tempfile.TemporaryDirectory() as tmp:
            frames = [os.path.join(tmp, f"f{i}.jpg") for i in range(2)]
            for f in frames:
                make_test_image(f)
            out = os.path.join(tmp, "grid.jpg")
            result = vf.build_grid_image(frames, cols=3, rows=3, output=out)
            self.assertTrue(os.path.exists(out))


class TestIntegration(unittest.TestCase):
    """bili_transcript 多模态模式集成测试（mock 底层依赖）"""

    def test_run_screenshot_mode(self):
        """截图模式: mock 掉 API 和视频下载"""
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                markers = [{"timestamp": 10.0, "reason": "展示代码"},
                           {"timestamp": 20.0, "reason": "图表"}]

                # capture_screenshot 写入真实假图，保证 os.path.exists(out) 为 True
                def fake_shot(video, ts, out, width=1280):
                    make_test_image(out)

                with patch.object(bt.va, "generate_screenshot_markers", return_value=markers), \
                     patch.object(bt, "download_video", return_value="video.mp4"), \
                     patch.object(bt.vf, "capture_screenshot", side_effect=fake_shot):
                    note_path = bt.run_screenshot_mode(
                        "BV1xx411x7xx", "测试视频", "字幕内容",
                        {"api_key": "sk-test"}, {"max_count": 5, "width": 1280})
                    self.assertIsNotNone(note_path)
                    # 验证笔记内容包含截图引用
                    with open(note_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    self.assertIn("![[", content)
                    self.assertIn("测试视频", content)
            finally:
                os.chdir(old_cwd)

    def test_run_screenshot_no_markers(self):
        """无截图标记时直接返回 None"""
        with patch.object(bt.va, "generate_screenshot_markers", return_value=[]):
            result = bt.run_screenshot_mode("BV1xx411x7xx", "标题", "字幕",
                                            {"api_key": "sk-test"}, {})
            self.assertIsNone(result)

    def test_run_screenshot_vault_mode(self):
        """Vault 模式: 截图存入 {vault}/raw/screenshots/，笔记写入 {vault}/Inbox/"""
        with tempfile.TemporaryDirectory() as tmp:
            vault = os.path.join(tmp, "vault")
            os.makedirs(vault)
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                markers = [{"timestamp": 10.0, "reason": "展示代码"}]

                def fake_shot(video, ts, out, width=1280):
                    make_test_image(out)

                with patch.object(bt.va, "generate_screenshot_markers", return_value=markers), \
                     patch.object(bt, "download_video", return_value="video.mp4"), \
                     patch.object(bt.vf, "capture_screenshot", side_effect=fake_shot):
                    note_path = bt.run_screenshot_mode(
                        "BV1xx411x7xx", "测试视频", "字幕内容",
                        {"api_key": "sk-test"}, {"max_count": 5, "width": 1280},
                        vault_path=vault)
                    # 笔记写入 Inbox/
                    expected_note = os.path.join(vault, "Inbox", "测试视频-screenshots.md")
                    self.assertEqual(note_path, expected_note)
                    self.assertTrue(os.path.exists(note_path))
                    # 截图存入 raw/screenshots/{title}/
                    shot_file = os.path.join(vault, "raw", "screenshots",
                                             "测试视频", "screenshot_001.jpg")
                    self.assertTrue(os.path.exists(shot_file))
                    # 笔记使用 Obsidian 嵌入语法（文件名，不含路径）
                    with open(note_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    self.assertIn("![[screenshot_001.jpg]]", content)
            finally:
                os.chdir(old_cwd)

    def test_run_visual_vault_mode(self):
        """Vault 模式: 网格图存入 raw/screenshots/，笔记嵌入网格图"""
        with tempfile.TemporaryDirectory() as tmp:
            vault = os.path.join(tmp, "vault")
            os.makedirs(vault)
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                fake_frames = [f"frame_{i:03d}.jpg" for i in range(10)]

                def fake_grid(frames, **kwargs):
                    with open(kwargs["output"], "w") as f:
                        f.write("grid")
                    return kwargs["output"]

                with patch.object(bt, "download_video", return_value="video.mp4"), \
                     patch.object(bt.vf, "extract_frames", return_value=fake_frames), \
                     patch.object(bt.vf, "build_grid_image", side_effect=fake_grid), \
                     patch.object(bt.vf, "image_to_base64", return_value="b64"), \
                     patch.object(bt.va, "analyze_grid_card",
                                  return_value={"content": "画面短卡1"}), \
                     patch.object(bt.va, "synthesize_visual_note",
                                  return_value={"content": "聚合画面总结"}), \
                     patch.object(bt.va, "summarize_transcript",
                                  return_value={"content": "总结内容"}):
                    note_path = bt.run_visual_mode(
                        "BV1xx411x7xx", "测试视频", "字幕内容",
                        {"api_key": "sk-test"}, {"cols": 3, "rows": 3},
                        vault_path=vault)
                    expected_note = os.path.join(vault, "Inbox", "测试视频-visual.md")
                    self.assertEqual(note_path, expected_note)
                    self.assertTrue(os.path.exists(note_path))
                    # 网格图存入 raw/screenshots/
                    grid_file = os.path.join(vault, "raw", "screenshots",
                                             "测试视频", "grid_001.jpg")
                    self.assertTrue(os.path.exists(grid_file))
                    # 笔记含逐帧短卡 + 单遍聚合 + 网格图嵌入
                    with open(note_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    self.assertIn("![[grid_001.jpg]]", content)
                    self.assertIn("画面短卡1", content)
                    self.assertIn("聚合画面总结", content)
                    # 三件套补全：逐字稿 + 总结 + 画面三件齐全
                    inbox = os.path.join(vault, "Inbox")
                    self.assertTrue(os.path.exists(os.path.join(inbox, "测试视频-逐字稿.md")))
                    self.assertTrue(os.path.exists(os.path.join(inbox, "测试视频-总结.md")))
            finally:
                os.chdir(old_cwd)

    def test_resolve_output_dirs_local(self):
        """本地模式: 截图存 ./screenshots/，笔记存当前目录"""
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                shot_dir, note_dir, in_vault = bt.resolve_output_dirs(None, "视频A")
                self.assertEqual(shot_dir, os.path.join("screenshots", "视频A"))
                self.assertEqual(note_dir, ".")
                self.assertFalse(in_vault)
                self.assertTrue(os.path.exists(shot_dir))
            finally:
                os.chdir(old_cwd)

    def test_resolve_output_dirs_vault(self):
        """Vault 模式: 截图存 {vault}/raw/screenshots/，笔记存 {vault}/Inbox/"""
        with tempfile.TemporaryDirectory() as tmp:
            vault = os.path.join(tmp, "vault")
            shot_dir, note_dir, in_vault = bt.resolve_output_dirs(vault, "视频A")
            self.assertEqual(shot_dir, os.path.join(vault, "raw", "screenshots", "视频A"))
            self.assertEqual(note_dir, os.path.join(vault, "Inbox"))
            self.assertTrue(in_vault)
            self.assertTrue(os.path.exists(shot_dir))
            self.assertTrue(os.path.exists(note_dir))

    def test_run_visual_mode(self):
        """网格图模式: mock 掉视频下载、截帧和分析"""
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                # 准备 10 个假帧路径
                fake_frames = [f"frame_{i:03d}.jpg" for i in range(10)]
                with patch.object(bt, "download_video", return_value="video.mp4"), \
                     patch.object(bt.vf, "extract_frames", return_value=fake_frames), \
                     patch.object(bt.vf, "build_grid_image") as mock_grid, \
                     patch.object(bt.vf, "image_to_base64", return_value="b64"), \
                     patch.object(bt.va, "analyze_grid_card",
                                  return_value={"content": "画面短卡1"}), \
                     patch.object(bt.va, "synthesize_visual_note",
                                  return_value={"content": "聚合画面总结"}), \
                     patch.object(bt.va, "summarize_transcript",
                                  return_value={"content": "总结内容"}):
                    # build_grid_image 写入假文件
                    def fake_grid(frames, **kwargs):
                        with open(kwargs["output"], "w") as f:
                            f.write("grid")
                        return kwargs["output"]
                    mock_grid.side_effect = fake_grid

                    note_path = bt.run_visual_mode(
                        "BV1xx411x7xx", "测试视频", "字幕内容",
                        {"api_key": "sk-test"}, {"cols": 3, "rows": 3})
                    self.assertIsNotNone(note_path)
                    # 10 帧 / 9 = 1 组网格图
                    self.assertEqual(mock_grid.call_count, 1)
                    with open(note_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    self.assertIn("画面短卡1", content)
                    self.assertIn("聚合画面总结", content)
                    # 本地模式三件套落当前目录
                    self.assertTrue(os.path.exists("测试视频-逐字稿.md"))
                    self.assertTrue(os.path.exists("测试视频-总结.md"))
            finally:
                os.chdir(old_cwd)

    def test_download_video_missing_raises(self):
        """视频下载失败时抛出异常"""
        with patch.object(bt, "run_ytdlp") as mock_yt:
            mock_yt.return_value = MagicMock()
            with self.assertRaises(FileNotFoundError):
                bt.download_video("BV1xx411x7xx")

    # ── OPT-096 逐字稿专名清洗 ──
    def test_clean_transcript_high_signal(self):
        out = va.clean_transcript("部署了一个 千问或三八 的本地大模型.. 然后")
        self.assertIn("3B/8B", out)
        self.assertNotIn("。。", out)   # 连续句号归一
        self.assertNotIn("  ", out.replace("\n", " "))

    # ── OPT-092/093 视觉责任分层 + 聚合接口 ──
    def test_analyze_grid_card_uses_card_prompt(self):
        # 短卡 prompt 不含 transcript（不再把整段字幕喂进单张图）
        prompt = va.load_prompt("visual-grid-card-user")
        self.assertNotIn("{{transcript}}", prompt)
        synth = va.load_prompt("visual-synth-user")
        self.assertIn("{{cards}}", synth)
        self.assertIn("帧间不一致", synth)   # 一致性标注内建

    # ── OPT-094/095 三件套幂等 + 逐字稿清洗落盘 + 死链检查 ──
    def test_write_notes_idempotent_and_deadlink_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            safe = "视频A"
            # 先写逐字稿（含待清洗内容），再调一次应不覆盖
            p1 = bt.write_transcript_note(tmp, safe, "视频A", "BV1", "部署 千问三八 的模型..")
            with open(p1, "r", encoding="utf-8") as f:
                first = f.read()
            # 落盘时已完成专名清洗：三八 → 3B/8B、连续句号归一、多余空白压缩
            self.assertIn("3B/8B", first)
            self.assertNotIn("三八", first)
            self.assertNotIn("..", first)
            # 幂等：第二次传入不同内容不应覆盖已存在文件
            bt.write_transcript_note(tmp, safe, "视频A", "BV1", "改变后的内容")
            with open(p1, "r", encoding="utf-8") as f:
                self.assertEqual(first, f.read(), "已存在的逐字稿不得被覆盖")
            # 死链：逐字稿引用 [[视频A-总结]]，暂无总结文件 → 应报死链
            dead = bt.check_note_links(tmp)
            self.assertTrue(any("视频A-总结" in m for m in dead))
            # 补齐总结后死链消失
            with open(os.path.join(tmp, f"{safe}-总结.md"), "w", encoding="utf-8") as f:
                f.write("# 视频A 总结")
            dead2 = bt.check_note_links(tmp)
            self.assertEqual(dead2, [])

    def test_check_note_links_skips_media_embeds(self):
        """媒体/附件嵌入（grid_001.jpg 等）不算笔记死链；仅追踪笔记交叉引用"""
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "视觉.md"), "w", encoding="utf-8") as f:
                f.write("![[grid_001.jpg]]\n[[视频-总结]]\n[[missing-note]]\n")
            with open(os.path.join(tmp, "视频-总结.md"), "w", encoding="utf-8") as f:
                f.write("# 视频 总结")
            dead = bt.check_note_links(tmp)
            self.assertTrue(any("missing-note" in m for m in dead), "缺失笔记应报死链")
            self.assertFalse(any("grid_001.jpg" in m for m in dead), "图片嵌入不计入笔记死链")
            self.assertFalse(any("视频-总结" in m for m in dead), "存在的笔记不报死链")


class TestStreamUrl(unittest.TestCase):
    """get_stream_url 测试（playurl API → DASH 流地址）"""

    def test_get_stream_url_success(self):
        """成功获取 DASH 视频流地址"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "code": 0,
            "data": {"dash": {"video": [
                {"id": 80, "baseUrl": "https://cdn.bilibili.com/video.m4s"},
            ]}}
        }
        with patch.object(bt.requests, "get", return_value=mock_resp):
            url = bt.get_stream_url("BV1xx411x7xx", 12345, {"SESSDATA": "abc"})
            self.assertEqual(url, "https://cdn.bilibili.com/video.m4s")

    def test_get_stream_url_protocol_relative(self):
        """// 开头的地址补全为 https://"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "code": 0,
            "data": {"dash": {"video": [
                {"id": 64, "baseUrl": "//cdn.bilibili.com/video.m4s"},
            ]}}
        }
        with patch.object(bt.requests, "get", return_value=mock_resp):
            url = bt.get_stream_url("BV1xx411x7xx", 12345, {})
            self.assertTrue(url.startswith("https://"))

    def test_get_stream_url_api_error(self):
        """API 返回错误码时返回 None"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": -400, "message": "bad request"}
        with patch.object(bt.requests, "get", return_value=mock_resp):
            url = bt.get_stream_url("BV1xx411x7xx", 12345, {})
            self.assertIsNone(url)

    def test_get_stream_url_no_dash(self):
        """无 DASH 数据时返回 None"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 0, "data": {}}
        with patch.object(bt.requests, "get", return_value=mock_resp):
            url = bt.get_stream_url("BV1xx411x7xx", 12345, {})
            self.assertIsNone(url)

    def test_get_stream_url_picks_highest_quality(self):
        """多个视频流时选最高画质（id 最大的）"""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "code": 0,
            "data": {"dash": {"video": [
                {"id": 32, "baseUrl": "https://cdn.bilibili.com/480p.m4s"},
                {"id": 80, "baseUrl": "https://cdn.bilibili.com/1080p.m4s"},
                {"id": 64, "baseUrl": "https://cdn.bilibili.com/720p.m4s"},
            ]}}
        }
        with patch.object(bt.requests, "get", return_value=mock_resp):
            url = bt.get_stream_url("BV1xx411x7xx", 12345, {})
            self.assertIn("1080p", url)


class TestStreamFrames(unittest.TestCase):
    """流式截帧功能测试"""

    def test_build_ffmpeg_headers_default(self):
        """默认包含 User-Agent 和 Referer"""
        hdr = vf._build_ffmpeg_headers()
        self.assertIn("User-Agent", hdr)
        self.assertIn("Referer", hdr)
        self.assertIn("bilibili.com", hdr)
        # 每行以 \r\n 结尾
        self.assertTrue(hdr.endswith("\r\n"))

    def test_build_ffmpeg_headers_with_cookie(self):
        """传入额外 headers 合并"""
        hdr = vf._build_ffmpeg_headers({"Cookie": "SESSDATA=abc"})
        self.assertIn("Cookie: SESSDATA=abc", hdr)
        self.assertIn("Referer", hdr)

    @patch.object(vf, "_ffmpeg")
    def test_capture_screenshot_from_stream_calls_ffmpeg(self, mock_ff):
        """流式截图调用 ffmpeg 时传入 -headers 和 stream URL"""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "shot.jpg")
            vf.capture_screenshot_from_stream(
                "https://cdn.example.com/video.m4s", 120.0, out, width=1280)
            args = mock_ff.call_args[0][0]
            self.assertIn("-headers", args)
            self.assertIn("https://cdn.example.com/video.m4s", args)
            self.assertIn("120.0", args)
            self.assertIn("-frames:v", args)

    @patch.object(vf, "_ffmpeg")
    def test_extract_frames_from_stream_multiple(self, mock_ff):
        """流式截帧按间隔生成多帧"""
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "frames")
            # 模拟 ffmpeg 实际写入文件
            def fake_ff(args, timeout=120):
                for a in args:
                    if a.endswith(".jpg"):
                        make_test_image(a)
                        break
            mock_ff.side_effect = fake_ff
            frames = vf.extract_frames_from_stream(
                "https://cdn.example.com/video.m4s", 30, interval=10,
                output_dir=out_dir)
            # 30s / 10s interval = timestamps 0,10,20 → 3 帧
            self.assertEqual(len(frames), 3)
            self.assertEqual(mock_ff.call_count, 3)

    @patch.object(vf, "_ffmpeg")
    def test_extract_frames_from_stream_skips_failed(self, mock_ff):
        """单帧失败不中断，继续下一帧"""
        call_count = [0]

        def fake_ff(args, timeout=120):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("timeout")
            for a in args:
                if a.endswith(".jpg"):
                    make_test_image(a)
                    break

        mock_ff.side_effect = fake_ff
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "frames")
            frames = vf.extract_frames_from_stream(
                "https://cdn.example.com/video.m4s", 30, interval=10,
                output_dir=out_dir)
            # 3 次调用，第 1 次失败，返回 2 帧
            self.assertEqual(len(frames), 2)
            self.assertEqual(mock_ff.call_count, 3)


class TestStreamingIntegration(unittest.TestCase):
    """流式截帧集成测试（cookie → stream 优先路径）"""

    def test_screenshot_stream_path(self):
        """有 cookie 时走流式截帧路径（不调用 download_video）"""
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                markers = [{"timestamp": 10.0, "reason": "展示代码"}]
                cookie = {"SESSDATA": "abc", "BILI_JCT": "xyz"}

                def fake_stream_shot(stream_url, ts, out, width=1280,
                                     headers=None, timeout=30):
                    make_test_image(out)

                with patch.object(bt.va, "generate_screenshot_markers",
                                  return_value=markers), \
                     patch.object(bt, "get_video_info",
                                  return_value={"cid": 123, "duration": 60}), \
                     patch.object(bt, "get_stream_url",
                                  return_value="https://cdn.example.com/v.m4s"), \
                     patch.object(bt.vf, "capture_screenshot_from_stream",
                                  side_effect=fake_stream_shot), \
                     patch.object(bt, "download_video") as mock_dl:
                    note_path = bt.run_screenshot_mode(
                        "BV1xx411x7xx", "测试视频", "字幕内容",
                        {"api_key": "sk-test"}, {"max_count": 5, "width": 1280},
                        cookie=cookie)
                    self.assertIsNotNone(note_path)
                    # 不应调用 download_video
                    mock_dl.assert_not_called()
            finally:
                os.chdir(old_cwd)

    def test_screenshot_fallback_to_download(self):
        """stream_url 获取失败时降级为 download_video"""
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                markers = [{"timestamp": 10.0, "reason": "展示代码"}]
                cookie = {"SESSDATA": "abc"}

                def fake_shot(video, ts, out, width=1280):
                    make_test_image(out)

                with patch.object(bt.va, "generate_screenshot_markers",
                                  return_value=markers), \
                     patch.object(bt, "get_video_info",
                                  return_value={"cid": 123}), \
                     patch.object(bt, "get_stream_url", return_value=None), \
                     patch.object(bt, "download_video", return_value="video.mp4"), \
                     patch.object(bt.vf, "capture_screenshot",
                                  side_effect=fake_shot):
                    note_path = bt.run_screenshot_mode(
                        "BV1xx411x7xx", "测试视频", "字幕内容",
                        {"api_key": "sk-test"}, {"max_count": 5, "width": 1280},
                        cookie=cookie)
                    self.assertIsNotNone(note_path)
            finally:
                os.chdir(old_cwd)

    def test_visual_stream_path(self):
        """视觉模式有 cookie 时走流式截帧路径"""
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                fake_frames = [f"frame_{i:03d}.jpg" for i in range(9)]
                cookie = {"SESSDATA": "abc"}

                def fake_grid(frames, **kwargs):
                    with open(kwargs["output"], "w") as f:
                        f.write("grid")
                    return kwargs["output"]

                with patch.object(bt, "get_video_info",
                                  return_value={"cid": 123, "duration": 90}), \
                     patch.object(bt, "get_stream_url",
                                  return_value="https://cdn.example.com/v.m4s"), \
                     patch.object(bt.vf, "extract_frames_from_stream",
                                  return_value=fake_frames), \
                     patch.object(bt.vf, "build_grid_image", side_effect=fake_grid), \
                     patch.object(bt.vf, "image_to_base64", return_value="b64"), \
                     patch.object(bt.va, "analyze_grid_card",
                                  return_value={"content": "画面短卡1"}), \
                     patch.object(bt.va, "synthesize_visual_note",
                                  return_value={"content": "聚合画面总结"}), \
                     patch.object(bt.va, "summarize_transcript",
                                  return_value={"content": "总结内容"}), \
                     patch.object(bt, "download_video") as mock_dl:
                    note_path = bt.run_visual_mode(
                        "BV1xx411x7xx", "测试视频", "字幕内容",
                        {"api_key": "sk-test"}, {"cols": 3, "rows": 3},
                        cookie=cookie)
                    self.assertIsNotNone(note_path)
                    mock_dl.assert_not_called()
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
