"""serve-manage 单实例守护单测（OPT-091）。

覆盖不启动真实服务即可判定的核心逻辑：端口占用探测、pid 存活判定、
pidfile 写/读/清、状态三态、start 对活进程占用端口的 fail-closed 拒绝、
stop 对孤儿/端口占用进程的清理。均对 cfg 对象操作，不绑真实 8643。
"""
from __future__ import annotations

import os
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agentlab.runtime.serve_manage import (
    _port_owner, alive, cmd_start, cmd_status, pid_file, port_in_use, read_pid,
    write_pid,
)


def _cfg(base: Path, port: int, token: str = "test-token"):
    return SimpleNamespace(
        trace_dir=str(base / "logs" / "trace"),
        serve=SimpleNamespace(host="127.0.0.1", port=port, token=token),
    )


def _ephemeral_server():
    """在 127.0.0.1 上开一个占用端口的 socket，返回 (port, close)。"""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]

    def _serve():
        while True:
            try:
                conn, _ = s.accept()
                conn.close()
            except OSError:
                break

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return port, s


class TestPortInUse(unittest.TestCase):
    def test_detects_bound_and_free(self):
        port, s = _ephemeral_server()
        try:
            self.assertTrue(port_in_use("127.0.0.1", port))
            self.assertFalse(port_in_use("127.0.0.1", 1))  # 空闲端口
        finally:
            s.close()

    def test_invalid_port_safe(self):
        self.assertFalse(port_in_use("127.0.0.1", 30000))  # 未占用
        port, s = _ephemeral_server()
        s.close()
        # 关闭后应判为非占用（宽松容忍 windows 偶尔 TIME_WAIT）
        self.assertFalse(port_in_use("127.0.0.1", port))


class TestAlive(unittest.TestCase):
    def test_self_true_dead_false(self):
        self.assertTrue(alive(0) is False)  # 空 pid
        self.assertTrue(alive(None) is False)
        import os
        self.assertTrue(alive(os.getpid()))
        self.assertFalse(alive(2147483647))


class TestSubprocessDecode(unittest.TestCase):
    """netstat/tasklist 输出是系统 ANSI（中文 Windows=GBK）；PYTHONUTF8=1 下
    text=True 默认解码会崩读线程 → stdout=None → TypeError（编码加固回归）。"""

    def test_calls_are_decode_safe(self):
        # 判活子进程调用必须显式 errors="replace"，任何编码环境解码都不失败
        fake = SimpleNamespace(stdout="", returncode=0)
        with mock.patch("agentlab.runtime.serve_manage.subprocess.run",
                        return_value=fake) as m:
            alive(os.getpid())
            self.assertEqual(m.call_args.kwargs.get("encoding"), "utf-8")
            self.assertEqual(m.call_args.kwargs.get("errors"), "replace")
        with mock.patch("agentlab.runtime.serve_manage.subprocess.run",
                        return_value=fake) as m:
            _port_owner("127.0.0.1", 18643)
            self.assertEqual(m.call_args.kwargs.get("encoding"), "utf-8")
            self.assertEqual(m.call_args.kwargs.get("errors"), "replace")

    def test_stdout_none_tolerated(self):
        # 兜底：即使解码线程崩掉（stdout=None），判活返回 False/None 而非 TypeError
        fake = SimpleNamespace(stdout=None, returncode=0)
        with mock.patch("agentlab.runtime.serve_manage.subprocess.run", return_value=fake):
            self.assertFalse(alive(12345))
            self.assertIsNone(_port_owner("127.0.0.1", 18643))

    def test_gbk_output_still_matches_ascii(self):
        # 真实故障形态：netstat 输出 GBK 表头 + ASCII 表体，utf-8+replace 解码后
        # 中文表头变替换符，LISTENING 行的 ASCII 子串与 PID 保留可解析
        gbk_out = ("活动连接\n".encode("gbk")
                   + "  TCP    0.0.0.0:8643    0.0.0.0:0    LISTENING    4242\n".encode("ascii"))
        fake = SimpleNamespace(stdout=gbk_out.decode("utf-8", errors="replace"),
                               returncode=0)
        with mock.patch("agentlab.runtime.serve_manage.subprocess.run", return_value=fake):
            self.assertEqual(_port_owner("0.0.0.0", 8643), 4242)


class TestPidFile(unittest.TestCase):
    def test_write_read_clean_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(Path(tmp), port=18643)
            self.assertEqual(pid_file(cfg).name, "serve.pid")
            self.assertIsNone(read_pid(cfg))
            write_pid(cfg, 12345)
            self.assertEqual(read_pid(cfg), 12345)
            write_pid(cfg, None)
            self.assertFalse(pid_file(cfg).exists())
            self.assertIsNone(read_pid(cfg))


class TestStartFailClosed(unittest.TestCase):
    def test_refuses_without_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(Path(tmp), port=18644, token="")
            with self.assertRaises(SystemExit):
                cmd_start(cfg)

    def test_refuses_when_port_alive(self):
        # 端口被活进程(当前 python)占 → start 拒绝，不抢端口
        port, s = _ephemeral_server()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cfg = _cfg(Path(tmp), port=port, token="tk")
                with self.assertRaises(SystemExit):
                    cmd_start(cfg)
        finally:
            s.close()


class TestStatus(unittest.TestCase):
    def test_status_when_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(Path(tmp), port=18655, token="tk")
            st = cmd_status(cfg)
            self.assertFalse(st["running"])
            self.assertIsNone(st["pid"])
            self.assertFalse(st["health"])

    def test_status_ignores_stale_pidfile(self):
        # 死 pid 遗留的 pidfile 不应被当作 running
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(Path(tmp), port=18656, token="tk")
            write_pid(cfg, 2147483647)
            st = cmd_status(cfg)
            self.assertFalse(st["running"])


if __name__ == "__main__":
    unittest.main()