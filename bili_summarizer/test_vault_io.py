"""OPT-213 P0-01 受控写入故障注入测试。

对应执行手册 4.2 最小故障注入用例（管线侧）：
- 拒写目录（raw/templates/.obsidian/.git）→ VAULT_WRITE_DENIED，无文件变化
- raw/screenshots 白名单可用（截图产物历史布局）
- Vault 外路径不误伤（CLI 本地输出模式）
- overwrite=False 防覆盖：已存在不写入
- 并发写同路径串行化，无半写文件
- revision 口径 = sha256；Vault 内写入留最小审计
"""
import hashlib
import os
import shutil
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vault_io import (AUDIT_REL, VaultWriteDenied, controlled_write,
                      guard_vault_path)

TEST_VAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".test_vault_io")


class VaultIoGuardTest(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(TEST_VAULT, ignore_errors=True)
        os.makedirs(os.path.join(TEST_VAULT, "Inbox"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(TEST_VAULT, ignore_errors=True)

    def test_deny_dirs_rejected_without_file_changes(self):
        for first in ("raw", "templates", ".obsidian", ".git"):
            target = os.path.join(TEST_VAULT, first, "x.md")
            with self.assertRaises(VaultWriteDenied) as ctx:
                controlled_write(target, "bad", vault_root=TEST_VAULT)
            self.assertIn(first.lstrip("."), ctx.exception.rule)
            self.assertFalse(os.path.exists(target), f"{first} 不应产生文件")

    def test_raw_screenshots_whitelisted(self):
        target = os.path.join(TEST_VAULT, "raw", "screenshots", "BVxx", "s1.jpg.txt")
        r = controlled_write(target, "fake", vault_root=TEST_VAULT)
        self.assertTrue(os.path.exists(target))
        self.assertEqual(r["revision"], hashlib.sha256(b"fake").hexdigest())

    def test_raw_subtree_other_than_screenshots_denied(self):
        target = os.path.join(TEST_VAULT, "raw", "books", "x.md")
        with self.assertRaises(VaultWriteDenied):
            controlled_write(target, "bad", vault_root=TEST_VAULT)
        self.assertFalse(os.path.exists(target))

    def test_outside_vault_paths_not_guarded(self):
        # CLI 本地输出模式：note_dir="." 或用户自定义目录 → 不做 Vault 守卫
        outside = os.path.join(os.path.dirname(TEST_VAULT), ".outside_vault_io.md")
        try:
            r = controlled_write(outside, "local", vault_root=None)
            self.assertTrue(os.path.exists(outside))
            self.assertEqual(r["overwritten"], False)  # 新文件 previous=None
        finally:
            os.unlink(outside)

    def test_no_clobber_keeps_existing_content(self):
        target = os.path.join(TEST_VAULT, "Inbox", "a.md")
        controlled_write(target, "v1", vault_root=TEST_VAULT)
        r = controlled_write(target, "v2", vault_root=TEST_VAULT, overwrite=False)
        self.assertFalse(r["overwritten"])
        self.assertEqual(r["revision"], hashlib.sha256(b"v1").hexdigest())
        with open(target, encoding="utf-8") as f:
            self.assertEqual(f.read(), "v1")

    def test_overwrite_updates_revision(self):
        target = os.path.join(TEST_VAULT, "Inbox", "b.md")
        r1 = controlled_write(target, "v1", vault_root=TEST_VAULT)
        r2 = controlled_write(target, "v2", vault_root=TEST_VAULT)
        self.assertTrue(r2["overwritten"])
        self.assertEqual(r2["previous_revision"], r1["revision"])
        self.assertNotEqual(r2["revision"], r1["revision"])

    def test_concurrent_writes_never_torn(self):
        # 同路径并发写：per-path 锁串行化，最终文件必须是某个完整版本（无半写）
        target = os.path.join(TEST_VAULT, "Inbox", "race.md")
        payloads = [f"payload-{i}-" + "x" * 20000 for i in range(8)]
        valid = {hashlib.sha256(p.encode()).hexdigest() for p in payloads}
        def worker(p):
            controlled_write(target, p, vault_root=TEST_VAULT)
        threads = [threading.Thread(target=worker, args=(p,)) for p in payloads]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with open(target, "rb") as f:
            data = f.read()
        self.assertIn(hashlib.sha256(data).hexdigest(), valid, "出现半写文件")
        self.assertFalse([n for n in os.listdir(os.path.dirname(target))
                          if n.startswith("race.md.tmp")], "临时文件残留")

    def test_audit_line_written_for_vault_writes(self):
        target = os.path.join(TEST_VAULT, "Inbox", "audited.md")
        r = controlled_write(target, "content", vault_root=TEST_VAULT, actor="test-actor")
        audit = os.path.join(TEST_VAULT, AUDIT_REL)
        self.assertTrue(os.path.exists(audit))
        with open(audit, encoding="utf-8") as f:
            lines = [json_line for json_line in f.read().splitlines() if json_line]
        import json
        rec = json.loads(lines[-1])
        self.assertEqual(rec["actor"], "test-actor")
        self.assertEqual(rec["revision"], r["revision"])
        self.assertIn("Inbox", rec["path"])

    def test_guard_skips_outside_vault(self):
        # guard 直测：Vault 外路径不抛
        guard_vault_path(TEST_VAULT, os.path.join(os.path.dirname(TEST_VAULT), "free.md"))


if __name__ == "__main__":
    unittest.main()
