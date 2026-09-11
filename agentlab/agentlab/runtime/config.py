"""配置加载（对齐 docs/03 §7、05 §1）。

优先级：环境变量 AGENT_LLM_API_KEY > config.json 的 llm.api_key > 默认。
只下发 `.example` 模板；api_key 永不落盘。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class LLMConfig(BaseModel):
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    model: str = "deepseek-chat"
    timeout: float = 60.0
    max_tokens: int | None = 4096

    def effective_key(self) -> str:
        return os.environ.get("AGENT_LLM_API_KEY") or self.api_key


class BrainConfig(BaseModel):
    config_path: str = "../../obsidian_agent_brain/config.json"


class MemoryConfig(BaseModel):
    dir: str = ""
    dedupe: bool = True
    decay_half_life_days: float = 30.0  # 召回时间衰减半衰期（天，0=关闭；旧记忆降权防占用召回位，OPT-101）


class RagConfig(BaseModel):
    """向量语义检索路（P0-1/OPT-105）。embed_base_url 为空 = 向量路禁用（纯关键词降级）。"""

    vector_enabled: bool = True
    embed_base_url: str = ""  # OpenAI 兼容 /embeddings，如 dashscope compatible-mode
    embed_model: str = "text-embedding-v4"
    embed_api_key: str = ""
    embed_timeout: float = 30.0
    chunk_chars: int = 600  # 分块上限（字符）：空行分段贪心合并，超长按句硬切
    vector_k: int = 6  # 向量路每次召回条数（RRF 融合前）
    auto_sync_limit: int = 8  # 检索前有界自愈：单次最多重嵌的变更文件数（防查询被大同步拖死）
    # L11/OPT-111 会话区段档案：折叠原文落 {sid}.ranges.jsonl + 入向量索引，rag_retrieve 增 session 路
    session_ranges: bool = True  # 总开关（serve 侧；关闭则折叠不可逆、rag_retrieve 无 session 路）
    range_chunk_chars: int = 2000  # 区段分块上限（对话体较粗粒度，区别于 vault 的 600）
    range_max_chunks: int = 200  # 单区段入索引块数上限：超界只索引前 N 块（jsonl 仍全量）

    def effective_key(self) -> str:
        return os.environ.get("AGENT_RAG_EMBED_API_KEY") or self.embed_api_key


class ExternalAgentConfig(BaseModel):
    """单个外部 ACP Agent（P2-1/OPT-112）。command 为空 = 未启用（工具不注册）。"""

    name: str  # agent_consult 的 agent 参数引用名
    command: str = ""  # ACP 兼容可执行文件（如 claude-code-acp / acp 适配器）
    args: list[str] = Field(default_factory=list)
    cwd: str = ""  # 外部 agent 工作目录；空 = vault_root
    env: dict[str, str] = Field(default_factory=dict)  # 额外环境变量（认证 token 等，本地 config）
    timeout: float = 300.0  # 单次 consult 每请求超时（秒）


class AgentsConfig(BaseModel):
    """外部 Agent 注册表（P2-1）。空列表 = 不注册 agent_consult（默认，零影响）。"""

    external: list[ExternalAgentConfig] = Field(default_factory=list)
    # P2-2/F5-017：单轮 multi 并行支路上限。超出名单进 summary.skipped（显式回报，
    # 不静默丢弃）；用"裁名单"而非"排队"限流——排队会让全部支都跑完，失去成本上限意义。
    max_parallel_consults: int = 3
    # 单支结果进主上下文/SSE 的字符上限；超出部分截断并全文归档到 session range，
    # 主 Agent 可经 rag_retrieve 的 session 路召回（对齐"产物落盘、对话只传引用"）。
    consult_result_max_chars: int = 6000

    @field_validator("max_parallel_consults", "consult_result_max_chars")
    @classmethod
    def _positive_int(cls, v: int, info: ValidationInfo) -> int:
        """非法值 fail-closed 回落默认，不放行 0/负数（会造成"全部 skipped"或"无上限"）。"""
        if isinstance(v, int) and not isinstance(v, bool) and v >= 1:
            return v
        return int(cls.model_fields[info.field_name].default)


class LimitsConfig(BaseModel):
    max_steps: int = 15
    timeout: float | None = 300
    context_budget: int = 600000  # 上下文预算（token）：设为 ≤ 模型窗口的 50~60%。deepseek v4 系列窗口 1M → 600000；换 128K 级模型请改回 ~64000，避免请求超窗报错
    context_window: int = 1048576  # 模型上下文窗口（OPT-110 四期，学 pi shouldCompact 挂窗口）：换模型必改；有效预算 = min(context_budget, 窗口×60%)
    compact_slice_tokens: int = 150000  # 单次折叠区段上限（token）：小步多次折叠，避免巨型摘要调用（学 pi 早折勤折）
    context_tools: bool = True  # 注入 context_status/compress_context 工具（L10/OPT-106）
    context_nudge: float = 0.75  # 用量占比达此值 → 注入一次压缩提示（L10/OPT-106）
    context_hard_trim: float = 0.95  # 用量占比达此值 → 机械硬截断最旧工具结果（L10/OPT-106）
    max_tool_result_chars: int = 12000
    max_tool_calls: int = 40  # 单轮工具调用次数上界（0=不限）：成本上界，见 core/loop.py _BUDGET_NUDGE
    tool_call_nudge: float = 0.6  # 用量占比达此值 → 注入一次"收敛"提示
    recall_item_chars: int = 1200  # recall 注入 Tier-1：单条召回正文预算（字符，OPT-089）
    memory_item_chars: int = 2000  # 捕获沉淀 Tier-1：单条记忆正文预算（字符，OPT-090）
    memorize_every: int = 5  # 记忆沉淀周期（0=关闭；每 N 轮异步批量沉淀到长期记忆）；#10①/OPT-123 起默认开启
    memory_inject_topk: int = 5  # 每轮 system prompt 自动召回注入的长期记忆条数（0=关闭注入；#10①/OPT-123）

    def effective_budget(self) -> int:
        """有效上下文预算：min(手调预算, 模型窗口×60%)——触发基准挂模型窗口（学 pi shouldCompact 挂窗口）。"""
        if self.context_window > 0:
            return min(self.context_budget, int(self.context_window * 0.6))
        return self.context_budget


class ResilienceConfig(BaseModel):
    max_retries: int = 2
    backoff: float = 1.0  # 重试退避基数（秒）
    breaker_threshold: int = 5  # 连续失败 N 次触发熔断
    breaker_recovery: float = 30.0  # 熔断恢复等待（秒）


class ServeConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8643
    token: str = ""  # fail-closed：未配置即拒绝启动；用 AGENTLAB_SERVE_TOKEN 或 serve.token 指定
    heartbeat: int = 10  # SSE 进度心跳间隔（秒；0=关闭），长任务期沿流周期推 response.heartbeat
    approval_timeout_seconds: float = 120.0  # HITL 审批等待上限；超时默认拒绝


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    routes: dict[str, str] = {}  # 多模型路由：task_tier -> model
    # Configure this explicitly for each user's Obsidian vault.
    vault_root: str = ""
    brain: BrainConfig = Field(default_factory=BrainConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    skills_dir: str | None = None
    trace_dir: str = "logs/trace"
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    rag: RagConfig = Field(default_factory=RagConfig)  # 向量语义检索路（P0-1）
    agents: AgentsConfig = Field(default_factory=AgentsConfig)  # 外部 ACP agent（P2-1）
    approval_mode: str = "risk_based"  # risk_based=危险工具走审批/白名单；allow_all=策略自动放行已注册工具
    danger_allowlist: list[str] = Field(default_factory=list)  # danger 权限白名单（默认全拒）
    write_allowlist: list[str] | None = None  # write 权限白名单；None=用代码默认集（认可的 Vault/memory 写入器），[] = 全拒


def load_config(path: str | Path | None = None) -> Config:
    """从 config.json 加载；未找到则返回全默认（可离线跑样例）。"""
    config_path = Path(path) if path else (Path(DEFAULT_CONFIG) if Path(DEFAULT_CONFIG).exists() else None)
    if config_path is None or not config_path.exists():
        return Config()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return Config.model_validate(data)


DEFAULT_CONFIG = "config/config.json"
PROJECT_CONFIG_EXAMPLE = "config/config.example.json"
