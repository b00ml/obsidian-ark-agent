"""OPT-135：runs_recent 自省工具单测（"今天/最近做了什么"的数据通路）。"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from agentlab.runtime.trace import Tracer
from agentlab.tools.runs_tools import build_runs_tools


class TestRunsRecent(unittest.TestCase):
    def _seed(self, tmp: str) -> None:
        tr = Tracer(tmp)
        tr.new_session()
        tr.record_run(input="今天的run", stop_reason="done", tokens=5)
        tr2 = Tracer(tmp)
        tr2.new_session()
        tr2.record_run(input="五天前的run", stop_reason="done", tokens=3)
        ts = time.time() - 5 * 86400  # 把第二条的 mtime 拨回 5 天前
        os.utime(Path(tmp) / f"{tr2.trace_id}.jsonl", (ts, ts))

    def test_days_filter_and_fields(self):
        tmp = tempfile.mkdtemp()
        self._seed(tmp)
        fn = build_runs_tools(tmp)[0].fn
        today = fn(days=1)
        self.assertEqual([r["input"] for r in today["runs"]], ["今天的run"])
        self.assertEqual(today["total"], 1)
        self.assertEqual(today["runs"][0]["stop_reason"], "done")
        week = fn(days=7)
        self.assertEqual(week["total"], 2)

    def test_empty_dir_returns_empty(self):
        fn = build_runs_tools(tempfile.mkdtemp())[0].fn
        out = fn(days=1)
        self.assertEqual(out["runs"], [])
        self.assertEqual(out["total"], 0)


if __name__ == "__main__":
    unittest.main()
