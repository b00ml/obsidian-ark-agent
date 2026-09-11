import asyncio
import sys
import unittest

from agentlab.core.llm import LLMProvider, LLMResponse
from agentlab.core.message import Message, TokenUsage
from agentlab.core.router import RouteLLM, build_providers, classify_light_route, route


class _TierProvider(LLMProvider):
    def __init__(self, tag: str):
        self.tag = tag
        self.calls: list[str] = []

    async def chat(self, messages, tools=None, **kw):
        self.calls.append(self.tag)
        return LLMResponse(content=f"[{self.tag}]", tool_calls=[],
                           usage=TokenUsage(input_tokens=1, output_tokens=1), stop_reason="stop")


class _Factory:
    def __init__(self, tag: str):
        self.tag = tag
        self.instances = []

    def __call__(self, llm_cfg):
        p = _TierProvider(f"{self.tag}:{getattr(llm_cfg, 'model', '?')}")
        self.instances.append(p)
        return p


class _CfgLLM:
    def __init__(self, model="deepseek-chat"):
        self.model = model


class TestRoute(unittest.TestCase):
    def test_build_providers_empty_routes_creates_heavy(self):
        fact = _Factory("m")
        providers = build_providers(_CfgLLM(), {}, fact)
        self.assertEqual(list(providers.keys()), ["heavy"])
        self.assertEqual(providers["heavy"].tag, "m:deepseek-chat")

    def test_build_providers_from_routes(self):
        fact = _Factory("m")
        providers = build_providers(_CfgLLM(), {"light": "lm", "heavy": "hm"}, fact)
        # heavy 缺失时回退到已有 tier 的某个实例
        self.assertEqual(providers["light"].tag, "m:lm")
        self.assertEqual(providers["heavy"].tag, "m:hm")

    def test_route_selects_tier(self):
        provs = {"light": _TierProvider("L"), "heavy": _TierProvider("H")}
        self.assertIs(route(provs, "light"), provs["light"])
        self.assertIs(route(provs, "heavy"), provs["heavy"])
        self.assertIs(route(provs, "unknown"), provs["heavy"], "未知 tier 回退默认")

    def test_route_llm_dispatch_and_classifier(self):
        provs = {"light": _TierProvider("L"), "heavy": _TierProvider("H")}
        r = RouteLLM(provs, default_tier="heavy")

        async def go(tier):
            return await r.chat([Message(role="user", content="hi")], tier=tier)
        asyncio.run(go("light"))
        asyncio.run(go("heavy"))
        self.assertEqual(provs["light"].calls, ["L"])
        self.assertEqual(provs["heavy"].calls, ["H"])

        # 无 tier 且注入 classifier → 用 classifier 结果
        r2 = RouteLLM(provs, default_tier="heavy", classify=lambda msgs: "light")
        asyncio.run(r2.chat([Message(role="user", content="hi")]))
        self.assertEqual(provs["light"].calls, ["L", "L"])

    def test_route_llm_requires_provider(self):
        with self.assertRaises(ValueError):
            RouteLLM({})

    def test_classify_light_route_chitchat_short(self):
        # 闲聊/问候短消息 → light tier（直答省工具）
        self.assertEqual(classify_light_route([Message(role="user", content="你好")]), "light")
        self.assertEqual(classify_light_route([Message(role="user", content="hi!")]), "light")
        # 实质任务即使短也不误判
        self.assertEqual(classify_light_route([Message(role="user", content="删掉 X 笔记")]), "heavy")
        # 含闲聊词但为长消息不做快速决策（可能含实质任务）→ 回默认
        long = "你好，请帮我总结一下今天的会议，并列出三条待办、补充到记忆里供下次复用。"
        self.assertEqual(classify_light_route([Message(role="user", content=long)]), "heavy")
        # 空输入回默认
        self.assertEqual(classify_light_route([]), "heavy")


if __name__ == "__main__":
    unittest.main()