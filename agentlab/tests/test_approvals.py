from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from agentlab.runtime.approvals import ApprovalManager


class _Tool:
    name = "acp_consult"
    permission = "danger"


class _Sink:
    def __init__(self):
        self.events = []

    def event(self, payload):
        self.events.append(payload)


def _cfg(mode="risk_based"):
    return SimpleNamespace(approval_mode=mode, danger_allowlist=[], write_allowlist=None)


class TestApprovalManager(unittest.TestCase):
    def test_requested_then_allow_resolves_waiter(self):
        async def run():
            manager = ApprovalManager(timeout_seconds=3)
            sink = _Sink()
            task = asyncio.create_task(manager.confirm(
                _cfg(), _Tool(), "即将执行外部 agent", sink=sink,
                run_id="run-1", session_id="sess-1", signal=None))
            for _ in range(20):
                await asyncio.sleep(0)
                requested = next((e for e in sink.events if e["type"] == "approval.requested"), None)
                if requested:
                    break
            self.assertIsNotNone(requested)
            result = manager.resolve(requested["approval_id"], "allow")
            self.assertEqual(result["status"], "approved")
            self.assertTrue(await task)
            self.assertEqual(manager.resolve(requested["approval_id"], "deny")["status"], "approved")
            self.assertTrue(any(e["type"] == "approval.resolved" for e in sink.events))

        asyncio.run(run())

    def test_disconnect_denies(self):
        async def run():
            manager = ApprovalManager(timeout_seconds=3)
            sink = _Sink()
            signal = asyncio.Event()
            task = asyncio.create_task(manager.confirm(
                _cfg(), _Tool(), "危险操作", sink=sink, run_id="run-2",
                session_id=None, signal=signal))
            await asyncio.sleep(0)
            signal.set()
            self.assertFalse(await task)
            resolved = [e for e in sink.events if e["type"] == "approval.resolved"][-1]
            self.assertEqual(resolved["reason"], "disconnect")

        asyncio.run(run())

    def test_allow_all_has_no_waiting_request(self):
        async def run():
            manager = ApprovalManager(timeout_seconds=1)
            sink = _Sink()
            self.assertTrue(await manager.confirm(
                _cfg("allow_all"), _Tool(), "危险操作", sink=sink,
                run_id="run-3", session_id=None, signal=None))
            self.assertFalse(any(e["type"] == "approval.requested" for e in sink.events))
            self.assertIn("allow_all", sink.events[0]["decision"])

        asyncio.run(run())

    def test_invalid_mode_fails_closed_without_hitl(self):
        async def run():
            manager = ApprovalManager(timeout_seconds=1)
            sink = _Sink()
            self.assertFalse(await manager.confirm(
                _cfg("typo"), _Tool(), "危险操作", sink=sink,
                run_id="run-4", session_id=None, signal=None))
            self.assertFalse(sink.events)

        asyncio.run(run())
