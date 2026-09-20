"""多模型路由（docs/04 §5，F17）。

- build_providers(cfg)：按 config.routes 把 model 名映射成 LLMProvider（light/heavy/…）。
- route(models, task_tier)：取指定 tier 的 provider。
- RouteLLM：一个 LLMProvider 外观，按 tier（或 classifier 判定）分发到后台 provider，
  无该 tier 时回退 default；供 loop/provider 层单一入口复用。

分类器默认不注入（用显式 tier）；需要"一次 cheap 调用判轻重"时传入 classify 回调。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentlab.core.llm import LLMProvider
from agentlab.core.message import Message


def route(providers: dict[str, LLMProvider], task_tier: str,
          default_tier: str = "heavy") -> LLMProvider:
    """docs/04 §5 route(task_tier)：按 tier 取 provider，未知 tier 回退 default。"""
    return providers.get(task_tier) or providers.get(default_tier) or next(iter(providers.values()))


# 明显的闲聊/问候信号（可直答、无需工具调用）；命中且短消息 → 走 light tier
_CHIT_CHAT = (
    "你好", "哈喽", "hello", "hi", "在吗", "谢谢", "辛苦了", "你是谁",
    "你能做什么", "再见", "拜拜", "早上好", "晚上好", "嗨",
)


def classify_light_route(messages: list[Message], default_tier: str = "heavy") -> str:
    """规则快速决策：闲聊/问候短消息 → "light"（直答省工具），否则回 default。

    对齐 open-note 三段式快速决策里"闲聊直答"一段：用纯规则代替一次 cheap LLM
    调用判轻重。未知 tier（如未配 light）由 RouteLLM 安全回退 default，无副作用。
    """
    if not messages:
        return default_tier
    last = messages[-1].content or ""
    low = last.lower().strip()
    if len(low) <= 24 and any(s in low for s in _CHIT_CHAT):
        return "light"
    return default_tier


class RouteLLM(LLMProvider):
    """多模型路由外观：把多个 tier 的 provider 封装成单一 LLMProvider。

    chat 时取 tier = 显式 tier 参数 > classifier(messages) > default_tier。
    """

    def __init__(
        self,
        providers: dict[str, LLMProvider],
        default_tier: str = "heavy",
        classify: Callable[[list[Message]], str] | None = None,
    ):
        if not providers:
            raise ValueError("RouteLLM 至少需要一个 provider")
        self.providers = dict(providers)
        self.default_tier = default_tier if default_tier in providers else next(iter(providers))
        self.classify = classify

    async def aclose(self) -> None:
        """Close each unique routed provider without double-closing aliases."""
        seen: set[int] = set()
        for provider in self.providers.values():
            marker = id(provider)
            if marker in seen:
                continue
            seen.add(marker)
            closer = getattr(provider, "aclose", None)
            if closer is not None:
                await closer()

    def resolve(self, tier: str | None, messages: list[Message]) -> LLMProvider:
        t = tier or (self.classify(messages) if self.classify else self.default_tier)
        return route(self.providers, t, self.default_tier)

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        tier: str | None = None,
        **kwargs: Any,
    ) -> Any:
        provider = self.resolve(tier, messages)
        return await provider.chat(
            messages, tools, temperature=temperature, max_tokens=max_tokens, **kwargs
        )


def build_providers(cfg_llm, routes: dict[str, str], factory) -> dict[str, LLMProvider]:
    """把 routes 的 model 名映射成 provider；routes 为空时以默认 model 建一个 heavy。

    factory(llm_config) -> LLMProvider，用于隔离 provider 构造（测试可注入 mock）。
    """
    providers: dict[str, LLMProvider] = {}
    for tier, model in (routes or {}).items():
        providers[tier] = factory(llm_with_model(cfg_llm, model))
    if not providers:
        providers["heavy"] = factory(cfg_llm)
    elif "heavy" not in providers:
        providers["heavy"] = providers.get("heavy") or next(iter(providers.values()))
    return providers


def llm_with_model(cfg_llm, model: str):
    """cfg_llm 的浅拷贝并覆盖 model（pydantic 模型用 __copy__ 或构造规避 test 复杂度）。
    """
    import copy
    from pydantic import BaseModel
    if isinstance(cfg_llm, BaseModel):
        c = cfg_llm.model_copy(deep=True)
        c.model = model
        return c
    c = copy.copy(cfg_llm)
    c.model = model
    return c
