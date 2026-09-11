"""typed 事件总线单测（C2 收官）。

覆盖：payload 类型不匹配 emit 即 TypeError、字段拼错构造即 ValidationError、
字符串订阅兼容（Ev 为 str enum）、订阅者异常隔离、空 payload 事件正常派发。
"""
from __future__ import annotations

import unittest

from pydantic import ValidationError

from agentlab.core.events import (
    AgentEnd, EventPayload, Ev, MessageEnd, ToolEnd, ToolStart, TurnStart,
)
from agentlab.core.loop import _RunnerEvents


class TestTypedEvents(unittest.TestCase):
    def setUp(self):
        self.bus = _RunnerEvents()

    def test_payload_instance_dispatch_and_attribute_access(self):
        seen = []
        self.bus.on(Ev.TOOL_END, seen.append)
        payload = ToolEnd(name="vault_read", result="ok")
        self.bus.emit(Ev.TOOL_END, payload)
        self.assertIs(seen[0], payload)
        self.assertEqual(seen[0].name, "vault_read")

    def test_wrong_payload_type_raises_loudly(self):
        # 事件与 payload 类型不匹配 → emit 时 TypeError，而非静默派发
        self.bus.on(Ev.TOOL_START, lambda p: None)
        with self.assertRaises(TypeError):
            self.bus.emit(Ev.TOOL_START, ToolEnd(name="x"))

    def test_typo_field_fails_at_construction(self):
        # **kw 时代的核心病灶：键拼错静默取默认。extra="forbid" 让它构造即炸
        with self.assertRaises(ValidationError):
            ToolStart(nmae="vault_read", arguments="{}")
        with self.assertRaises(ValidationError):
            MessageEnd(role="assistant", bogus="多余字段")

    def test_string_subscription_backcompat(self):
        # Ev 继承 str：按裸字符串订阅仍命中同一 handler 表（向后兼容）
        seen = []
        self.bus.on("agent_end", seen.append)
        self.bus.emit(Ev.AGENT_END, AgentEnd(stop_reason="done"))
        self.assertEqual(seen[0].stop_reason, "done")

    def test_subscriber_exception_isolated(self):
        # 前一订阅者抛错不得阻断后续订阅者，也不得中断 emit 本身（C3 语义保留）
        seen = []

        def _boom(_):
            raise RuntimeError("订阅者故障")

        self.bus.on(Ev.MESSAGE_END, _boom)
        self.bus.on(Ev.MESSAGE_END, seen.append)
        self.bus.emit(Ev.MESSAGE_END, MessageEnd(role="assistant", content="hi"))
        self.assertEqual(seen[0].content, "hi")

    def test_empty_payload_events(self):
        seen = []
        self.bus.on(Ev.TURN_START, seen.append)
        self.bus.emit(Ev.TURN_START, TurnStart())
        self.assertIsInstance(seen[0], EventPayload)


if __name__ == "__main__":
    unittest.main()
