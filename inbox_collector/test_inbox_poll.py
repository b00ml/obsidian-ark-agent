"""inbox_poll 单元测试（mock 外部依赖，不依赖真实 agently-cli/网络）

运行: python -m unittest test_inbox_poll（在 inbox_collector/ 目录下）
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import inbox_poll as ip
from queue_store import InboxQueueStore


class TestClassify(unittest.TestCase):
    def test_bili_video(self):
        cls = ip.classify("https://www.bilibili.com/video/BV1xx411x7xx?p=2")
        self.assertIsNotNone(cls)
        self.assertEqual(cls[0], "bili")
        self.assertEqual(cls[1], "https://www.bilibili.com/video/BV1xx411x7xx")

    def test_bili_short(self):
        cls = ip.classify("https://b23.tv/abc123")
        self.assertEqual(cls[0], "bili")

    def test_wechat_article(self):
        cls = ip.classify("https://mp.weixin.qq.com/s/xxxxxxxxxxxxxxxx?utm=1")
        self.assertEqual(cls[0], "wechat_article")
        self.assertEqual(cls[1], "https://mp.weixin.qq.com/s/xxxxxxxxxxxxxxxx")

    def test_unknown_domain(self):
        self.assertIsNone(ip.classify("https://github.com/user/repo"))

    def test_invalid_url(self):
        self.assertIsNone(ip.classify("not a url"))


class TestNormalize(unittest.TestCase):
    def test_bili_remove_query(self):
        url = ip.normalize_url("https://www.bilibili.com/video/BV1xx411x7xx?p=1&t=10", "bili")
        self.assertEqual(url, "https://www.bilibili.com/video/BV1xx411x7xx")

    def test_bili_extract_bvid(self):
        url = ip.normalize_url("https://b23.tv/xxxx?spm=1", "bili")
        self.assertTrue(url.endswith("BV") or "b23" in url or url.startswith("https://b23"))


class TestFingerprint(unittest.TestCase):
    def test_same_bvid_dedup(self):
        a = ip.fingerprint("https://www.bilibili.com/video/BV1xx411x7xx?p=1", "bili")
        b = ip.fingerprint("https://bilibili.com/video/BV1xx411x7xx", "bili")
        self.assertEqual(a, b)

    def test_wechat_query_dedup(self):
        a = ip.fingerprint("https://mp.weixin.qq.com/s/abc?utm_source=x", "wechat_article")
        b = ip.fingerprint("https://mp.weixin.qq.com/s/abc", "wechat_article")
        self.assertEqual(a, b)


class TestWhitelist(unittest.TestCase):
    def test_b23_tv_in_whitelist(self):
        self.assertTrue(ip._in_whitelist("https://b23.tv/xxx", ["bilibili.com", "b23.tv"]))

    def test_substring_no_false_positive(self):
        # 回归：b23.tv 不能被 "bilibili.com" 子串误判（曾导致短链被过滤）
        self.assertFalse(ip._in_whitelist("https://b23.tv/xxx", ["bilibili.com"]))

    def test_empty_whitelist_allows_all(self):
        self.assertTrue(ip._in_whitelist("https://example.com/a", []))

    def test_subdomain_match(self):
        self.assertTrue(ip._in_whitelist("https://mp.weixin.qq.com/s/abc", ["mp.weixin.qq.com"]))
        self.assertTrue(ip._in_whitelist("https://sub.bilibili.com/x", ["bilibili.com"]))


class TestStripHtml(unittest.TestCase):
    def test_remove_tags(self):
        text = ip.strip_html(
            '<p>hello <a href="https://b23.tv/x">link</a></p><script>var x=1;</script>'
        )
        self.assertIn("https://b23.tv/x", text)
        self.assertNotIn("var x=1", text)

    def test_html_unescape(self):
        text = ip.strip_html("<p>a&amp;b &lt;c&gt;</p>")
        self.assertIn("a&b <c>", text)


class TestExtractUrls(unittest.TestCase):
    def test_extract(self):
        urls = ip.extract_urls("看这个 https://www.bilibili.com/video/BV1xx411x7xx 和 b23.tv 短链")
        self.assertEqual(len(urls), 1)


class TestRunAgently(unittest.TestCase):
    @mock.patch("inbox_poll.subprocess.run")
    def test_success(self, mock_run):
        mock_proc = mock.Mock()
        mock_proc.returncode = 0
        mock_proc.stdout = '{"ok": true, "data": {}}'
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc
        result = ip.run_agently(["message", "+list"])
        self.assertTrue(result["ok"])
        # 必须设置 HOME/USERPROFILE（凭证位置）
        env = mock_run.call_args.kwargs["env"]
        self.assertEqual(env["HOME"], ip.PROJECT_ROOT)
        self.assertEqual(env["USERPROFILE"], ip.PROJECT_ROOT)

    @mock.patch("inbox_poll.subprocess.run")
    def test_cli_fail(self, mock_run):
        mock_proc = mock.Mock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "boom"
        mock_run.return_value = mock_proc
        with self.assertRaises(RuntimeError) as ctx:
            ip.run_agently(["+me"])
        self.assertIn("COLLECTOR_CLI_FAIL", str(ctx.exception))

    def test_cli_missing(self):
        with mock.patch("inbox_poll.shutil.which", return_value=None), \
             mock.patch("inbox_poll.os.path.exists", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                ip.run_agently(["+me"])
            self.assertIn("COLLECTOR_CLI_MISSING", str(ctx.exception))


class TestPoll(unittest.TestCase):
    def _fake_detail(self, msg_id, body, subject="测试"):
        return {
            "message_id": msg_id,
            "subject": subject,
            "body": body,
            "body_format": "HTML",
            "created_at": "2026-08-19T10:00:00Z",
        }

    def _cfg(self, tmp: str) -> dict:
        return {
            "queue_path": os.path.join(tmp, "queue.jsonl"),
            "seen_path": os.path.join(tmp, "seen.txt"),
            "queue_db_path": os.path.join(tmp, "queue.db"),
            "max_fetch": 10,
            "whitelist_domains": ["bilibili.com", "mp.weixin.qq.com"],
        }

    @mock.patch("inbox_poll.list_messages")
    @mock.patch("inbox_poll.read_message")
    def test_bili_enqueue(self, mock_read, mock_list):
        mock_list.return_value = [{"message_id": "m1"}]
        mock_read.return_value = self._fake_detail(
            "m1", '<p>分享视频 <a href="https://www.bilibili.com/video/BV1xx411x7xx">x</a></p>')
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp)
            n = ip.poll(cfg, dry_run=False)
            self.assertEqual(n, 1)
            store = InboxQueueStore(cfg["queue_db_path"])
            task = store.list_tasks(status="pending")[0]
            self.assertEqual(task["type"], "bili")
            self.assertEqual(task["url"], "https://www.bilibili.com/video/BV1xx411x7xx")
            self.assertEqual(task["status"], "pending")

    @mock.patch("inbox_poll.list_messages")
    @mock.patch("inbox_poll.read_message")
    def test_dedup_no_double(self, mock_read, mock_list):
        """同邮件跑两遍只入队一次"""
        mock_list.return_value = [{"message_id": "m1"}]
        mock_read.return_value = self._fake_detail(
            "m1", '<p><a href="https://mp.weixin.qq.com/s/abc">x</a></p>')
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp)
            cfg["whitelist_domains"] = ["mp.weixin.qq.com"]
            n1 = ip.poll(cfg)
            n2 = ip.poll(cfg)
            self.assertEqual(n1, 1)
            self.assertEqual(n2, 0)
            store = InboxQueueStore(cfg["queue_db_path"])
            self.assertEqual(len(store.list_tasks()), 1)

    @mock.patch("inbox_poll.list_messages")
    @mock.patch("inbox_poll.read_message")
    def test_no_url_marks_seen(self, mock_read, mock_list):
        """无白名单 URL 的邮件标记已处理，不再重复扫"""
        mock_list.return_value = [{"message_id": "m1"}]
        mock_read.return_value = self._fake_detail("m1", "<p>只有文字没有链接</p>")
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp)
            cfg["whitelist_domains"] = []
            n = ip.poll(cfg)
            self.assertEqual(n, 0)
            store = InboxQueueStore(cfg["queue_db_path"])
            self.assertIn("m1", store.seen_markers())

    @mock.patch("inbox_poll.list_messages")
    @mock.patch("inbox_poll.read_message")
    def test_collection_operation_is_queryable_after_queue_commit(self, mock_read, mock_list):
        mock_list.return_value = [{"message_id": "m-op"}]
        mock_read.return_value = self._fake_detail(
            "m-op", '<a href="https://www.bilibili.com/video/BV1xx411x7xx">x</a>')
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._cfg(tmp)
            self.assertEqual(ip.poll(cfg, operation_id="op-collect-1"), 1)
            store = InboxQueueStore(cfg["queue_db_path"])
            operation = store.collection_operation("op-collect-1")
            self.assertEqual(operation["status"], "succeeded")
            self.assertEqual(operation["new_tasks"], 1)
            self.assertEqual(store.list_tasks()[0]["operation_id"], "op-collect-1")

    @mock.patch("inbox_poll.list_messages")
    @mock.patch("inbox_poll.read_message")
    def test_collection_exposes_stage_boundaries_without_affecting_queue(self, mock_read, mock_list):
        mock_list.return_value = [{"message_id": "m-stage"}]
        mock_read.return_value = self._fake_detail(
            "m-stage", '<a href="https://www.bilibili.com/video/BV1xx411x7xx">x</a>')
        with tempfile.TemporaryDirectory() as tmp:
            stages = []
            self.assertEqual(ip.poll(self._cfg(tmp), stage_callback=stages.append), 1)
            self.assertEqual(stages, ["fetched", "parsed", "completed"])


class TestQueueStore(unittest.TestCase):
    def test_duplicate_task_still_marks_message_seen(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = InboxQueueStore(os.path.join(tmp, "queue.db"))
            task = {"type": "bili", "url": "https://example.com", "status": "pending"}
            self.assertTrue(store.enqueue(task, "bili:BV1", ["bili:BV1", "m1"]))
            duplicate = dict(task, id="another")
            self.assertFalse(store.enqueue(duplicate, "bili:BV1", ["m2"]))
            self.assertEqual(len(store.list_tasks()), 1)
            self.assertIn("m2", store.seen_markers())

    def test_legacy_import_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue_path = os.path.join(tmp, "queue.jsonl")
            seen_path = os.path.join(tmp, "seen.txt")
            task = {"id": "legacy-1", "type": "bili", "url": "https://example.com",
                    "status": "done", "retry_count": 1}
            with open(queue_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(task, ensure_ascii=False) + "\n")
            with open(seen_path, "w", encoding="utf-8") as fh:
                fh.write("bili:BV1\nm1\n")
            store = InboxQueueStore(os.path.join(tmp, "queue.db"),
                                    queue_path=queue_path, seen_path=seen_path)
            self.assertEqual(store.migrate_legacy(), {"tasks": 1, "seen": 2})
            self.assertEqual(store.migrate_legacy(), {"tasks": 0, "seen": 0})
            self.assertEqual(store.stats(), {"done": 1})
            self.assertEqual(store.seen_markers(), {"bili:BV1", "m1"})


if __name__ == "__main__":
    unittest.main()
