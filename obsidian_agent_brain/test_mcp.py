"""obsidian_agent_brain MCP server 单元/集成测试（对应 TECH-OBSIDIAN-BRAIN §7）。

运行:
  .venv\\Scripts\\python.exe -m unittest obsidian_agent_brain/test_mcp.py

写操作全部落在测试临时 Vault 目录，不污染用户数据。
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import unittest
import hashlib
from types import SimpleNamespace

from unittest.mock import patch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import load_config  # noqa: E402
import tools_brain  # noqa: E402
import tools_bili  # noqa: E402
import tools_inbox  # noqa: E402
import tools_memory  # noqa: E402
import tools_obsidian  # noqa: E402
import tools_vault  # noqa: E402


# 固定工作区临时 Vault（不用 tempfile.TemporaryDirectory：其创建目录的
# chmod/写入在受限环境（如 dsh 沙箱）会被拒，导致 setUp/tearDown 报 WinError 5）
TEST_VAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".test_vault")


def make_config(tmp_vault: str) -> dict:
    cfg = load_config()
    cfg = dict(cfg)
    cfg["vault_path"] = tmp_vault
    return cfg


def fresh_vault() -> str:
    """重建固定临时 Vault 目录并返回路径。"""
    shutil.rmtree(TEST_VAULT, ignore_errors=True)
    os.makedirs(TEST_VAULT, exist_ok=True)
    return TEST_VAULT


class VaultToolsTest(unittest.TestCase):
    def setUp(self):
        self.vault = fresh_vault()
        self.cfg = make_config(self.vault)
        # 预置两篇互链笔记
        os.makedirs(os.path.join(self.vault, "Inbox"), exist_ok=True)
        os.makedirs(os.path.join(self.vault, "raw"), exist_ok=True)
        note_a = "---\ntitle: 测试A\ntype: video-summary\ntags: [rag, 检索]\n---\n\n# 测试A\n\n关于 RAG 检索的笔记。\n\n相关: [[测试B]]\n"
        note_b = "---\ntitle: 测试B\ntype: concept\ntags: [知识库]\n---\n\n# 测试B\n\n知识库概念，引用 [[测试A]]。\n"
        with open(os.path.join(self.vault, "Inbox", "测试A.md"), "w", encoding="utf-8") as f:
            f.write(note_a)
        with open(os.path.join(self.vault, "Inbox", "测试B.md"), "w", encoding="utf-8") as f:
            f.write(note_b)

    def tearDown(self):
        shutil.rmtree(self.vault, ignore_errors=True)

    def test_write_rejects_raw(self):
        with self.assertRaises(PermissionError):
            tools_vault.vault_write(self.cfg, "raw/screenshots/x.md", "内容")

    def test_search_ranks_by_relevance_not_path_order(self):
        """正文命中必须按相关性（命中次数）排，而不是按路径字母序。

        实测来源：2026-09-11 全量基线 q-memory-recall 一条任务打了 25 次搜索 + 13 次
        读回；旧排序 `(标题命中?, path)` 让相关性完全不参与，agent 只能逐条读。
        """
        # aaa-*.md 路径字母序最靠前但只命中 1 次；zzz-*.md 路径最后但命中 4 次
        with open(os.path.join(self.vault, "Inbox", "aaa-弱命中.md"), "w", encoding="utf-8") as f:
            f.write("---\ntitle: 弱命中\ntype: note\n---\n\n# 弱命中\n\n向量 一次。\n")
        with open(os.path.join(self.vault, "Inbox", "zzz-强命中.md"), "w", encoding="utf-8") as f:
            f.write("---\ntitle: 强命中\ntype: note\n---\n\n# 强命中\n\n向量 向量 向量 向量 全在这。\n")

        r = tools_vault.vault_search(self.cfg, "向量")
        paths = [h["path"] for h in r["results"]]
        self.assertEqual(paths[0], "Inbox/zzz-强命中.md", f"应按命中次数排序，实际 {paths}")
        by_path = {h["path"]: h for h in r["results"]}
        self.assertEqual(by_path["Inbox/zzz-强命中.md"]["hits"], 4, "要回传命中次数供模型判相关性")
        self.assertNotIn("rank", by_path["Inbox/zzz-强命中.md"], "排序用的中间字段不该外泄")

    def test_search_puts_title_hits_above_body_hits(self):
        with open(os.path.join(self.vault, "Inbox", "zzz-只用标题.md"), "w", encoding="utf-8") as f:
            f.write("---\ntitle: 向量\ntype: note\n---\n\n# 向量\n\n正文完全不提关键词。\n")
        with open(os.path.join(self.vault, "Inbox", "aaa-只命中正文.md"), "w", encoding="utf-8") as f:
            f.write("---\ntitle: 无关\ntype: note\n---\n\n# 无关\n\n向量 向量 向量 向量 向量。\n")
        paths = [h["path"] for h in tools_vault.vault_search(self.cfg, "向量")["results"]]
        self.assertEqual(paths[0], "Inbox/zzz-只用标题.md",
                         f"标题命中要压过正文命中，实际 {paths}")

    def test_write_default_inbox_and_read(self):
        wr = tools_vault.vault_write(self.cfg, "新笔记.md", "# 新笔记\n内容")
        self.assertEqual(wr["path"], "Inbox/新笔记.md")
        self.assertEqual(wr["revision"], hashlib.sha256("# 新笔记\n内容".encode()).hexdigest())
        self.assertIn("# 新笔记", tools_vault.vault_read(self.cfg, "Inbox/新笔记.md"))

    def test_write_cas_conflict_is_atomic_and_audited(self):
        from vault_gateway import VaultConflictError
        first = tools_vault.vault_write(self.cfg, "Inbox/cas.md", "v1")
        with self.assertRaises(VaultConflictError) as ctx:
            tools_vault.VaultGateway(self.cfg).write("Inbox/cas.md", "v2",
                                                       expected_revision="stale")
        self.assertEqual(ctx.exception.code, "VAULT_CONFLICT")
        self.assertEqual(tools_vault.vault_read(self.cfg, "Inbox/cas.md"), "v1")
        audit = os.path.join(self.vault, ".agent-brain", "audit", "vault.jsonl")
        self.assertTrue(os.path.exists(audit))
        with open(audit, encoding="utf-8") as fh:
            self.assertIn(first["revision"], fh.read())

    def test_patch_returns_revision_and_honors_cas(self):
        from vault_gateway import VaultConflictError
        first = tools_vault.vault_write(self.cfg, "Inbox/patch.md", "old")
        patched = tools_vault.vault_patch(self.cfg, "Inbox/patch.md", "old", "new",
                                           expected_revision=first["revision"])
        self.assertNotEqual(patched["revision"], first["revision"])
        with self.assertRaises(VaultConflictError):
            tools_vault.vault_patch(self.cfg, "Inbox/patch.md", "new", "bad",
                                    expected_revision=first["revision"])

    def test_patch_roundtrip_and_uniqueness(self):
        tools_vault.vault_patch(self.cfg, "Inbox/测试A.md", "关于 RAG 检索的笔记", "关于 RAG 检索的核心笔记")
        self.assertIn("核心笔记", tools_vault.vault_read(self.cfg, "Inbox/测试A.md"))
        with self.assertRaises(ValueError):
            tools_vault.vault_patch(self.cfg, "Inbox/测试A.md", "不存在的文本", "x")

    def test_search_hits_title_and_body(self):
        r = tools_vault.vault_search(self.cfg, "检索")
        self.assertGreaterEqual(r["total"], 1)
        paths = [h["path"] for h in r["results"]]
        self.assertIn("Inbox/测试A.md", paths)

    def test_graph_outbound_inbound(self):
        g = tools_vault.vault_graph(self.cfg, "Inbox/测试A.md")
        self.assertIn("测试B", g["outbound"])
        self.assertIn("Inbox/测试B.md", g["inbound"])

    def test_scan_stats(self):
        s = tools_vault.vault_scan(self.cfg)
        self.assertEqual(s["total_notes"], 2)
        self.assertIn("rag", {k.lower() for k in s["top_tags"]})
        self.assertGreaterEqual(s["linked_notes"], 2)

    def test_health_deadlink_and_orphan(self):
        # 预置：测试A 引用一个不存在的笔记 [[不存在X]]；再新增一篇孤立笔记（无出入向）
        with open(os.path.join(self.vault, "Inbox", "测试A.md"), "a", encoding="utf-8") as f:
            f.write("\n坏链: [[不存在X]]\n")
        with open(os.path.join(self.vault, "Inbox", "孤立.md"), "w", encoding="utf-8") as f:
            f.write("# 孤立笔记\n内容\n")
        h = tools_vault.vault_health(self.cfg)
        self.assertEqual(h["broken_links_count"], 1)
        self.assertEqual(h["broken_links"][0]["link"], "不存在X")
        self.assertEqual(h["broken_links"][0]["note"], "Inbox/测试A.md")
        # 孤立笔记无入向也无出向
        self.assertIn("Inbox/孤立.md", h["orphan_notes"])
        # 互链的 测试A/测试B 不判孤儿
        self.assertNotIn("Inbox/测试A.md", h["orphan_notes"])

    def test_health_payload_is_bounded_with_counts_first(self):
        """健康检查必须回传有界载荷，且计数排在清单之前。

        实测来源：真实 Vault 一次返回 135 条死链 + 48 孤儿 + 68 无引用，超出 agentlab 的
        `max_tool_result_chars=12000` 被头尾截断，计数落进被省略的中段 → agent 与评审都
        看不到，答案里正确的数字被两轮基线判成编造。
        """
        for i in range(30):
            with open(os.path.join(self.vault, "Inbox", f"孤立{i:02d}.md"), "w",
                      encoding="utf-8") as f:
                f.write(f"# 孤立{i:02d}\n无任何链接\n")
        h = tools_vault.vault_health(self.cfg)
        self.assertGreater(h["orphan_count"], 20, "计数要反映全量")
        self.assertEqual(len(h["orphan_notes"]), 20, "清单要截样")
        self.assertEqual(h["sampled"]["orphan_notes"]["total"], h["orphan_count"])
        self.assertEqual(h["sampled"]["orphan_notes"]["returned"], 20)
        keys = list(h)
        self.assertLess(keys.index("broken_links_count"), keys.index("broken_links"),
                        "计数必须排在清单前，截断时才不会先丢计数")
        self.assertLess(keys.index("orphan_count"), keys.index("orphan_notes"))
        payload = json.dumps(h, ensure_ascii=False)
        self.assertLess(len(payload), 12000, f"载荷必须低于工具结果上限，实际 {len(payload)}")

    def test_health_sample_param_can_widen_or_zero_out(self):
        for i in range(5):
            with open(os.path.join(self.vault, "Inbox", f"孤本{i}.md"), "w", encoding="utf-8") as f:
                f.write(f"# 孤本{i}\n")
        h0 = tools_vault.vault_health(self.cfg, sample=0)
        self.assertEqual(h0["orphan_notes"], [])
        self.assertGreater(h0["orphan_count"], 0, "sample=0 只影响清单，不影响计数")
        self.assertEqual(h0["sampled"]["orphan_notes"]["returned"], 0)


class MemoryTest(unittest.TestCase):
    def setUp(self):
        self.vault = fresh_vault()
        self.cfg = make_config(self.vault)

    def tearDown(self):
        shutil.rmtree(self.vault, ignore_errors=True)

    def test_commit_query_roundtrip(self):
        tools_memory.memory_commit(self.cfg, "用户偏好：多模态分析优先", ["偏好", "多模态"])
        r = tools_memory.memory_query(self.cfg, "多模态")
        self.assertEqual(r["total"], 1)
        self.assertIn("多模态", r["results"][0]["content"])
        # 不匹配返回空，不虚构
        self.assertEqual(tools_memory.memory_query(self.cfg, "不存在的主题")["total"], 0)

    def test_empty_commit_rejected(self):
        with self.assertRaises(ValueError):
            tools_memory.memory_commit(self.cfg, "  ")

    def test_missing_vault_path_rejected_instead_of_writing_to_cwd(self):
        """cfg 缺 vault_path 时必须报错，不能把 cwd 当 Vault。

        实测来源：2026-09-11 仓库里出现 `bili_summarizer/.agent-brain/memory/sessions.sqlite`。
        复现条件——cfg 无 vault_path 时 `memory_query` 把 `<cwd>/.agent-brain/...` 当记忆库且不报错，
        而 `tools_bili._run_in_bili` 恰好会把 cwd 切到 bili_summarizer。
        """
        from common import vault_root

        cfg = {"brain_dir": ".agent-brain"}
        with self.assertRaises(ValueError) as ctx:
            vault_root(cfg)
        self.assertIn("vault_path", str(ctx.exception))
        with self.assertRaises(ValueError):
            tools_memory.memory_query(cfg, "任意主题")


class BrainTest(unittest.TestCase):
    def setUp(self):
        self.vault = fresh_vault()
        self.cfg = make_config(self.vault)

    def tearDown(self):
        shutil.rmtree(self.vault, ignore_errors=True)

    def test_reindex_placeholder_not_enabled(self):
        r = tools_brain.brain_reindex(self.cfg)
        self.assertEqual(r["status"], "not_enabled")

    def test_search_empty_vault(self):
        r = tools_brain.brain_search(self.cfg, "任意词")
        self.assertEqual(r["total"], 0)


class BiliErrorPathTest(unittest.TestCase):
    def test_invalid_bvid_raises(self):
        cfg = load_config()
        from tools_bili import bili_transcribe
        with self.assertRaises(ValueError):
            bili_transcribe(cfg, "不是BV号")


class ServerRegistrationTest(unittest.TestCase):
    def test_tool_count_ge_10(self):
        import mcp_server
        tools = mcp_server.mcp._tool_manager.list_tools()
        n = len(tools)
        self.assertGreaterEqual(n, 10, "MCP server 应注册至少 10 个工具")
        names = [t.name for t in tools]
        for required in ("vault_read", "vault_write", "vault_patch", "vault_search",
                         "vault_graph", "vault_scan", "bili_transcribe", "bili_meta",
                         "bili_screenshot", "bili_visual", "article_fetch",
                         "article_summarize", "inbox_collect", "inbox_read_queue",
                         "brain_search", "brain_reindex", "brain_scan",
                         "memory_commit", "memory_query",
                         "obsidian_links", "obsidian_health", "obsidian_rename",
                         "obsidian_move", "obsidian_property_set"):
            self.assertIn(required, names, f"缺少工具 {required}")

    def test_bili_async_job_uses_durable_store_and_cancel_tool(self):
        import mcp_server
        from agentlab.runtime.state_store import TaskStore
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        old_store = mcp_server._TASK_STORE
        try:
            mcp_server._TASK_STORE = TaskStore(os.path.join(tmp.name, "state.db"))
            with patch.object(mcp_server.tools_bili, "bili_transcribe",
                              return_value={"transcript": "ok"}):
                started = mcp_server.bili_transcribe_start("BV1test")
            self.assertEqual(started["status"], "started")
            for _ in range(50):
                status = mcp_server.bili_job_status(started["job_id"])
                if status["task_status"] == "succeeded":
                    break
                import time
                time.sleep(0.01)
            self.assertEqual(status["status"], "done")
            self.assertEqual(status["result"], {"transcript": "ok"})
            self.assertEqual(mcp_server.bili_job_cancel(started["job_id"])["task_status"],
                             "succeeded")
            reopened = TaskStore(os.path.join(tmp.name, "state.db"))
            self.assertEqual(reopened.get(started["job_id"])["status"], "succeeded")
        finally:
            mcp_server._TASK_STORE = old_store
            tmp.cleanup()


class InboxQueueToolsTest(unittest.TestCase):
    def test_read_queue_migrates_legacy_jsonl_once(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            queue_path = os.path.join(tmp, "queue.jsonl")
            seen_path = os.path.join(tmp, "seen.txt")
            db_path = os.path.join(tmp, "queue.db")
            tasks = [
                {"id": "q1", "type": "bili", "url": "https://example.com/1",
                 "status": "pending", "retry_count": 0},
                {"id": "q2", "type": "wechat_article", "url": "https://example.com/2",
                 "status": "done", "retry_count": 0},
            ]
            with open(queue_path, "w", encoding="utf-8") as fh:
                for task in tasks:
                    fh.write(json.dumps(task, ensure_ascii=False) + "\n")
            with open(seen_path, "w", encoding="utf-8") as fh:
                fh.write("m1\n")

            cfg = dict(make_config(tmp))
            cfg.update({"queue_path": queue_path, "seen_path": seen_path,
                        "queue_db_path": db_path})
            result = tools_inbox.inbox_read_queue(cfg)
            self.assertEqual(result["total"], 2)
            self.assertEqual(result["by_status"], {"pending": 1, "done": 1})
            self.assertEqual(result["pending"][0]["id"], "q1")
            result2 = tools_inbox.inbox_read_queue(cfg)
            self.assertEqual(result2["total"], 2)


class BiliAsyncJobTest(unittest.TestCase):
    def test_bili_async_duplicate_reuses_task_and_cancel_discards_late_result(self):
        import mcp_server
        from agentlab.runtime.state_store import TaskStore
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        old_store = mcp_server._TASK_STORE
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        calls = []

        def slow_transcribe(*_args):
            calls.append(1)
            entered.set()
            try:
                self.assertTrue(release.wait(2), "test worker was not released")
                return {"transcript": "late"}
            finally:
                finished.set()

        try:
            mcp_server._TASK_STORE = TaskStore(os.path.join(tmp.name, "state.db"))
            with patch.object(mcp_server.tools_bili, "bili_transcribe",
                              side_effect=slow_transcribe):
                first = mcp_server.bili_transcribe_start("BV1dedupe")
                self.assertTrue(entered.wait(2), "worker did not start")
                duplicate = mcp_server.bili_transcribe_start("BV1dedupe")
                self.assertEqual(first["job_id"], duplicate["job_id"])
                self.assertEqual(duplicate["status"], "existing")
                self.assertTrue(mcp_server.bili_job_cancel(first["job_id"])["cancelled"])
                release.set()
                for _ in range(50):
                    status = mcp_server.bili_job_status(first["job_id"])
                    if status["task_status"] == "cancelled":
                        break
                    time.sleep(0.01)
                self.assertTrue(finished.wait(2), "worker did not finish before cleanup")
                worker_name = f"mcp-task-{first['job_id']}"
                for _ in range(100):
                    workers = [t for t in threading.enumerate() if t.name == worker_name]
                    if not workers:
                        break
                    time.sleep(0.01)
                self.assertFalse(workers, "task worker still holds SQLite connection")
            self.assertEqual(calls, [1])
            self.assertEqual(status["task_status"], "cancelled")
            self.assertNotIn("result", status)
        finally:
            release.set()
            mcp_server._TASK_STORE = old_store
            tmp.cleanup()


class BiliVisualTitleTest(unittest.TestCase):
    """OPT-098 回归：bili_visual 前置标题必须取真实内容标题，而非回退 BV 号。

    之前 _prepare_visual 用 `get_title_via_ytdlp or bvid`，yt-dlp 失败就把
    笔记文件名/标题写成 BV 号。修复后优先用 get_video_meta 的真实标题。
    """

    def _fake_bili(self, meta_title="真实内容标题", meta_raises=False,
                   ytdlp_title=None):
        import types

        def get_video_meta(bvid, cookie):
            if meta_raises:
                raise RuntimeError("API失败")
            return {"title": meta_title, "bvid": bvid}
        return types.SimpleNamespace(
            extract_bvid=lambda b: b,
            get_title_via_ytdlp=lambda b: ytdlp_title,
            load_cookie=lambda p: {"SESSDATA": "x"},
            get_video_meta=get_video_meta,
        )

    def test_prefers_real_title_when_meta_ok(self):
        cfg = dict(make_config(".test_vault"))
        fake = self._fake_bili(meta_title="AI应用开发的简历必须写8个项目")
        with patch.object(tools_bili, "_import_bili", return_value=fake), \
             patch.object(tools_bili, "bili_transcribe",
                          return_value={"transcript": "T"}):
            bili, bvid, title, transcript, _ = tools_bili._prepare_visual(
                cfg, "BV1xx411x7xx", "T")
        self.assertEqual(title, "AI应用开发的简历必须写8个项目")

    def test_fallback_ytdlp_when_meta_fails(self):
        cfg = dict(make_config(".test_vault"))
        fake = self._fake_bili(meta_raises=True, ytdlp_title="yt-dlp标题")
        with patch.object(tools_bili, "_import_bili", return_value=fake), \
             patch.object(tools_bili, "bili_transcribe",
                          return_value={"transcript": "T"}):
            _, _, title, _, _ = tools_bili._prepare_visual(cfg, "BV1xx411x7xx", "T")
        self.assertEqual(title, "yt-dlp标题")

    def test_fallback_bvid_when_all_fail(self):
        cfg = dict(make_config(".test_vault"))
        fake = self._fake_bili(meta_raises=True, ytdlp_title=None)
        with patch.object(tools_bili, "_import_bili", return_value=fake), \
             patch.object(tools_bili, "bili_transcribe",
                          return_value={"transcript": "T"}):
            _, _, title, _, _ = tools_bili._prepare_visual(cfg, "BV1xx411x7xx", "T")
        self.assertEqual(title, "BV1xx411x7xx")


class ObsidianCliToolsTest(unittest.TestCase):
    """Obsidian CLI 接入（优化设计文档4.0 执行线 #3）：探测 + 优雅降级 + 白名单。

    全程 mock shutil.which / subprocess.run，不依赖真实 Obsidian 安装；
    写操作参数校验落在临时 Vault 路径字符串上，不碰用户 Vault。
    """

    def setUp(self):
        self.vault = fresh_vault()
        self.cfg = make_config(self.vault)
        self.cfg["obsidian_cli"] = {"bin": "obsidian", "timeout": 15}

    def tearDown(self):
        shutil.rmtree(self.vault, ignore_errors=True)

    @staticmethod
    def _fake_proc(stdout: bytes = b"", returncode: int = 0) -> SimpleNamespace:
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=b"")

    def _all_tools(self, cfg):
        return [
            lambda: tools_obsidian.obsidian_links(cfg, "Inbox/测试A.md"),
            lambda: tools_obsidian.obsidian_health(cfg),
            lambda: tools_obsidian.obsidian_rename(cfg, "Inbox/测试A.md", "新名"),
            lambda: tools_obsidian.obsidian_move(cfg, "Inbox/测试A.md", "wiki"),
            lambda: tools_obsidian.obsidian_property_set(cfg, "Inbox/测试A.md", "status", "done"),
        ]

    # ① CLI 不可用 → 全部工具返回 degraded，绝不抛异常
    def test_unavailable_degrades_without_raise(self):
        with patch("tools_obsidian.shutil.which", return_value=None):
            avail = tools_obsidian.obsidian_available(self.cfg)
            self.assertEqual(avail, {"available": False, "version": None})
            for fn in self._all_tools(self.cfg):
                r = fn()  # 不得抛异常
                self.assertEqual(r["status"], "degraded")
                self.assertEqual(r["reason"], "obsidian cli not found")

    # ② canned JSON 输出 → 解析为结构化数据（含 backlinks/unresolved 的 format=json）
    def test_json_output_parsed(self):
        canned = json.dumps(["A.md", "B.md"]).encode("utf-8")
        with patch("tools_obsidian.shutil.which", return_value="C:/fake/obsidian.exe"), \
             patch("tools_obsidian.subprocess.run", return_value=self._fake_proc(canned)) as m_run:
            h = tools_obsidian.obsidian_health(self.cfg)
            self.assertEqual(h["status"], "ok")
            self.assertEqual(h["unresolved"], ["A.md", "B.md"])
            self.assertEqual(h["orphans"], ["A.md", "B.md"])
            self.assertEqual(h["deadends"], ["A.md", "B.md"])
            self.assertEqual(h["counts"], {"unresolved": 2, "orphans": 2, "deadends": 2})
            lk = tools_obsidian.obsidian_links(self.cfg, "测试A")
            self.assertEqual(lk["status"], "ok")
            self.assertEqual(lk["outbound"], ["A.md", "B.md"])
            self.assertEqual(lk["backlinks"], ["A.md", "B.md"])
            self.assertNotIn("errors", lk)
            # links / backlinks 子命令 argv 形态（白名单 COMMANDS 表驱动）
            argvs = [c.args[0] for c in m_run.call_args_list]
            self.assertIn(["C:/fake/obsidian.exe", "links", "file=测试A"], argvs)
            self.assertIn(["C:/fake/obsidian.exe", "backlinks", "file=测试A", "format=json"], argvs)

    # ②b 纯文本输出 → JSON 解析失败回退原文
    def test_plain_text_output_returned_as_is(self):
        fake = self._fake_proc("NoteA\nNoteB\n".encode("utf-8"))
        with patch("tools_obsidian.shutil.which", return_value="obsidian.exe"), \
             patch("tools_obsidian.subprocess.run", return_value=fake):
            h = tools_obsidian.obsidian_health(self.cfg)
            self.assertEqual(h["status"], "ok")
            self.assertEqual(h["orphans"], "NoteA\nNoteB")
            self.assertIsNone(h["counts"]["orphans"])  # 非 JSON 列表无法计数

    # ③ config 无 obsidian_cli 键 → 默认 bin/timeout 生效
    def test_defaults_without_obsidian_cli_key(self):
        cfg = dict(self.cfg)
        cfg.pop("obsidian_cli", None)
        fake = self._fake_proc(b"Obsidian 1.12.4")
        with patch("tools_obsidian.shutil.which", return_value="C:/fake/obsidian.exe") as m_which, \
             patch("tools_obsidian.subprocess.run", return_value=fake) as m_run:
            avail = tools_obsidian.obsidian_available(cfg)
            self.assertTrue(avail["available"])
            m_which.assert_called_with("obsidian")            # 默认 bin
            self.assertEqual(m_run.call_args.kwargs.get("timeout"), 10)  # 探测默认超时
            tools_obsidian.obsidian_property_set(cfg, "Inbox/测试A.md", "status", "done")
            self.assertEqual(m_run.call_args.kwargs.get("timeout"), 15)  # _run 默认超时
            self.assertEqual(m_run.call_args.args[0][1], "property:set")
            self.assertIn("name=status", m_run.call_args.args[0])
            self.assertIn("value=done", m_run.call_args.args[0])

    # ④ obsidian_available 两段逻辑：which 探测 + --version 应答
    def test_available_probe_two_stage(self):
        # which 找不到 → 直接 False，不再调 --version
        with patch("tools_obsidian.shutil.which", return_value=None), \
             patch("tools_obsidian.subprocess.run") as m_run:
            self.assertEqual(tools_obsidian.obsidian_available(self.cfg),
                             {"available": False, "version": None})
            m_run.assert_not_called()
        # which 找到 + --version 正常 → True + 版本号
        fake = self._fake_proc(b"1.12.4\n")
        with patch("tools_obsidian.shutil.which", return_value="obsidian.exe"), \
             patch("tools_obsidian.subprocess.run", return_value=fake) as m_run:
            r = tools_obsidian.obsidian_available(self.cfg)
            self.assertTrue(r["available"])
            self.assertEqual(r["version"], "1.12.4")
            self.assertEqual(m_run.call_args.args[0], ["obsidian.exe", "--version"])
        # --version 超时 → False（异常吞掉不抛）
        with patch("tools_obsidian.shutil.which", return_value="obsidian.exe"), \
             patch("tools_obsidian.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="obsidian", timeout=10)):
            self.assertFalse(tools_obsidian.obsidian_available(self.cfg)["available"])
        # --version 非零退出 → False
        bad = SimpleNamespace(returncode=1, stdout=b"", stderr=b"boom")
        with patch("tools_obsidian.shutil.which", return_value="obsidian.exe"), \
             patch("tools_obsidian.subprocess.run", return_value=bad):
            self.assertEqual(tools_obsidian.obsidian_available(self.cfg),
                             {"available": False, "version": None})

    # ⑤ 白名单护栏：raw/ 只读（rename/move/property_set 拒绝）+ rename 非法名
    def test_raw_readonly_and_rename_guards(self):
        fake = self._fake_proc(b"ok")
        with patch("tools_obsidian.shutil.which", return_value="obsidian.exe"), \
             patch("tools_obsidian.subprocess.run", return_value=fake) as m_run:
            r = tools_obsidian.obsidian_rename(self.cfg, "raw/notes/x.md", "y")
            self.assertEqual(r["status"], "error")
            self.assertIn("只读", r["reason"])
            r2 = tools_obsidian.obsidian_move(self.cfg, "Inbox/a.md", "raw/x")
            self.assertEqual(r2["status"], "error")   # 目标进 raw/ 同样拒绝
            r3 = tools_obsidian.obsidian_property_set(self.cfg, "raw/a.md", "k", "v")
            self.assertEqual(r3["status"], "error")
            r4 = tools_obsidian.obsidian_rename(self.cfg, "Inbox/a.md", "wiki/新名")
            self.assertEqual(r4["status"], "error")   # rename 不接受路径
            self.assertIn("obsidian_move", r4["reason"])
            r5 = tools_obsidian.obsidian_rename(self.cfg, "Inbox/a.md", "新名.md")
            self.assertEqual(r5["status"], "ok")      # .md 后缀剥掉后正常执行
            argv = m_run.call_args.args[0]
            self.assertEqual(argv[1], "rename")
            self.assertIn("name=新名", argv)
            self.assertIn("path=Inbox/a.md", argv)

    # ⑥ move 的 argv 组装：to= 目标 = folder + 原文件名
    def test_move_argv_targets_folder_plus_basename(self):
        fake = self._fake_proc(b"moved")
        with patch("tools_obsidian.shutil.which", return_value="obsidian.exe"), \
             patch("tools_obsidian.subprocess.run", return_value=fake) as m_run:
            r = tools_obsidian.obsidian_move(self.cfg, "Inbox/a.md", "wiki/")
            self.assertEqual(r["status"], "ok")
            self.assertEqual(r["to"], "wiki/a.md")
            argv = m_run.call_args.args[0]
            self.assertEqual(argv[1], "move")
            self.assertIn("to=wiki/a.md", argv)

    # ⑦ 可用但子命令执行失败 → status=error 返回值表达，不抛异常
    def test_run_failure_returns_error_dict(self):
        def run_fail(args, **kwargs):
            if "--version" in args:  # 探测成功=可用
                return self._fake_proc(b"1.12.4")
            return self._fake_proc(b"", returncode=127)  # 业务子命令失败

        with patch("tools_obsidian.shutil.which", return_value="obsidian.exe"), \
             patch("tools_obsidian.subprocess.run", side_effect=run_fail):
            r = tools_obsidian.obsidian_health(self.cfg)
            self.assertEqual(r["status"], "error")
            self.assertTrue(r["errors"])
            for fn in self._all_tools(self.cfg):
                result = fn()  # 不得抛异常
                self.assertIn(result["status"], ("error", "partial"))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class MemoryTagsAndRecallTest(unittest.TestCase):
    """OPT-135：tags 拆字 bug 修复 + 多关键词召回 + 历史污染行修复。"""

    def setUp(self):
        self.tmp = fresh_vault()
        self.cfg = make_config(self.tmp)

    def test_commit_normalizes_string_tags(self):
        # 旧 bug 复现路径：模型传字符串 tags → 逐字符拆开存储
        r = tools_memory.memory_commit(self.cfg, "用户偏好深色主题", tags="偏好,ui")
        self.assertEqual(r["tags"], ["偏好", "ui"])
        q = tools_memory.memory_query(self.cfg, "偏好", limit=5)
        self.assertEqual(q["results"][0]["tags"], ["偏好", "ui"])

    def test_commit_normalizes_list_and_none(self):
        self.assertEqual(
            tools_memory.memory_commit(self.cfg, "a", tags=["x", " x ", 3])["tags"],
            ["x", "3"])
        self.assertEqual(tools_memory.memory_commit(self.cfg, "b")["tags"], [])

    def test_query_bigram_recall_on_phrase(self):
        # 整句 LIKE 几乎零命中的修复验证：句子与记忆只共享"诗句"二字也能召回
        tools_memory.memory_commit(self.cfg, "用户常问关于唐诗宋词的诗句问题", tags=["诗歌"])
        q = tools_memory.memory_query(self.cfg, "今天问了什么诗句", limit=5)
        self.assertTrue(any("诗句" in r["content"] for r in q["results"]))

    def test_repair_split_tags_fixes_corrupted_rows(self):
        tools_memory.memory_commit(self.cfg, "正常记忆", tags=["rag", "检索"])
        with tools_memory._connect(self.cfg) as conn:  # 种一行旧 bug 产物
            conn.execute(
                "INSERT INTO memories (content, tags, source_session, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("被污染的记忆", ",".join("inbox,收件箱"), "", "2026-09-07T00:00:00"))
        out = tools_memory.repair_split_tags(self.cfg)
        self.assertEqual(len(out["fixed"]), 1)
        self.assertEqual(out["fixed"][0]["new"], "inbox,收件箱")
        q = tools_memory.memory_query(self.cfg, "inbox", limit=5)
        self.assertTrue(any(r["tags"] == ["inbox", "收件箱"] for r in q["results"]))
        self.assertEqual(tools_memory.repair_split_tags(self.cfg)["fixed"], [])  # 幂等
