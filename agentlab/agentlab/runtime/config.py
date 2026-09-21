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
    # S1 memory governance.  User-authored commits stay compatible; automatic
    # extraction opts into candidate-first and the read gate defaults to active
    # records only.
    write_mode: str = "candidate_first"
    auto_promote: bool = False
    min_active_confidence: float = 0.85
    candidate_ttl_days: int = 60
    default_review_days: int = 90
    recall_min_confidence: float = 0.65
    allow_default_shared: bool = True
    quarantine_on_external_instruction: bool = True
    audit_path: str = ".agent-brain/memory/events.jsonl"
    retry_max_attempts: int = 3

    @field_validator("write_mode")
    @classmethod
    def _write_mode(cls, value: str) -> str:
        return value if value in {"candidate_first", "direct"} else "candidate_first"

    @field_validator("min_active_confidence", "recall_min_confidence")
    @classmethod
    def _confidence(cls, value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.65
        return max(0.0, min(1.0, value))

    @field_validator("candidate_ttl_days", "default_review_days", "retry_max_attempts")
    @classmethod
    def _positive(cls, value: int, info: ValidationInfo) -> int:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return int(cls.model_fields[info.field_name].default)


class RagConfig(BaseModel):
    """可切换、可叠加的 RAG 检索配置。

    关键词路是默认轻量基线；配置 embedding provider 后可单独启用向量路，
    也可与关键词路叠加做 RRF。provider 不可用时回退关键词路。
    """

    vector_enabled: bool = True
    embed_base_url: str = ""  # OpenAI 兼容 /embeddings，如 dashscope compatible-mode
    embed_model: str = "text-embedding-v4"
    embed_api_key: str = ""
    embed_timeout: float = 30.0
    chunk_chars: int = 600  # 分块上限（字符）：空行分段贪心合并，超长按句硬切
    vector_k: int = 6  # 向量路每次召回条数（RRF 融合前）
    auto_sync_limit: int = 8  # 检索前有界自愈：单次最多重嵌的变更文件数（防查询被大同步拖死）
    # P4 hybrid shadow controls.  These fields describe the P2 index route;
    # build_rag_tools keeps the legacy VectorIndex path until an explicit
    # production switch is approved.
    # Entry tags are part of the indexed semantics; changing this version
    # forces a derived-index rebuild instead of serving stale bucket metadata.
    chunk_strategy: str = "markdown-structure-v1-entry-tags-v1"
    index_version: str = "s1-p2-v1"
    vector_mode: str = "shadow"  # off=仅词法，shadow=词法展示+向量观测，on=词法+向量融合展示
    lexical_mode: str = "on"
    hybrid_candidate_k: int = 40
    dedupe_by: str = "entry"  # entry/file/ref
    vector_min_score: float = 0.0  # shadow 实验阈值；0=不做余弦下限过滤
    lexical_min_coverage: float = 0.0  # 词法证据最低覆盖率；0=关闭（灰度护栏）
    query_rewrite_mode: str = "off"  # off=关闭，shadow=额外观测，on=候选参与融合
    query_rewrite_max_variants: int = 1  # direct rewrite 首版最多一个候选
    rewrite_deadline_ms: int = 250
    query_expansion_mode: str = "shadow"  # off=关闭，shadow=只观测，on=候选参与融合
    query_expansion_max_variants: int = 5
    query_expansion_min_candidates: int = 8
    small_to_big_mode: str = "shadow"  # off=关闭，shadow=只观测，on=扩展注入正文
    small_to_big_neighbors: int = 1
    small_to_big_max_chars: int = 2400
    # A display-changing semantic fallback is a production gray-release
    # switch. Keep it off until its own evidence passes, even when vector
    # collection remains enabled in shadow mode.
    vector_fallback_mode: str = "off"
    query_embedding_cache_size: int = 256  # P2 查询向量进程内 LRU 上限
    shadow_log_path: str = ""  # 空=不落盘；仅保存 hybrid shadow 元数据，不含正文
    # Answer-level safety: shadow is observable-only, on fails closed after a
    # RAG call when citations are missing or outside the candidate set.
    answer_gate_mode: str = "shadow"
    answer_gate_require_citation: bool = True
    # Experimental only: allow cited, explicitly bounded partial answers when
    # rag_assess says insufficient.  Keep disabled until replay passes safety.
    answer_gate_allow_bounded_partial: bool = False
    # P2 单索引生产接线：auto 只在侧车已存在时接管，避免升级后首次启动
    # 因未完成 reconcile 而出现空召回；p2 强制使用 RagIndexStore，legacy
    # 保持旧 VectorIndex/RAGRecall 路径，便于回滚。
    index_backend: str = "auto"
    index_path: str = ""
    # L11/OPT-111 会话区段档案：折叠原文落 {sid}.ranges.jsonl + 入向量索引，rag_retrieve 增 session 路
    session_ranges: bool = True  # 总开关（serve 侧；关闭则折叠不可逆、rag_retrieve 无 session 路）
    range_chunk_chars: int = 2000  # 区段分块上限（对话体较粗粒度，区别于 vault 的 600）
    range_max_chunks: int = 200  # 单区段入索引块数上限：超界只索引前 N 块（jsonl 仍全量）

    def effective_key(self) -> str:
        # Ark deliberately sends an empty value when the user clears the key.
        # Presence, rather than truthiness, must win so a stale key in the
        # backend config cannot be silently reused after that action.
        if "AGENT_RAG_EMBED_API_KEY" in os.environ:
            return os.environ["AGENT_RAG_EMBED_API_KEY"]
        return self.embed_api_key

    @field_validator("vector_mode")
    @classmethod
    def _vector_mode(cls, value: str) -> str:
        if value not in {"off", "shadow", "on"}:
            return "shadow"
        return value

    @field_validator("lexical_mode")
    @classmethod
    def _lexical_mode(cls, value: str) -> str:
        if value not in {"off", "on"}:
            return "on"
        return value

    @field_validator("dedupe_by")
    @classmethod
    def _dedupe_by(cls, value: str) -> str:
        if value not in {"entry", "file", "ref"}:
            return "entry"
        return value

    @field_validator("hybrid_candidate_k")
    @classmethod
    def _candidate_k(cls, value: int) -> int:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return value
        return 40

    @field_validator("vector_min_score")
    @classmethod
    def _vector_min_score(cls, value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        return value if 0.0 <= value <= 1.0 else 0.0

    @field_validator("lexical_min_coverage")
    @classmethod
    def _lexical_min_coverage(cls, value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        return value if 0.0 <= value <= 1.0 else 0.0

    @field_validator("query_rewrite_mode")
    @classmethod
    def _query_rewrite_mode(cls, value: str) -> str:
        return value if value in {"off", "shadow", "on"} else "off"

    @field_validator("query_expansion_mode")
    @classmethod
    def _query_expansion_mode(cls, value: str) -> str:
        return value if value in {"off", "shadow", "on"} else "shadow"

    @field_validator("query_expansion_max_variants")
    @classmethod
    def _query_expansion_max_variants(cls, value: int) -> int:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return min(value, 5)
        return 5

    @field_validator("query_expansion_min_candidates")
    @classmethod
    def _query_expansion_min_candidates(cls, value: int) -> int:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return min(value, 40)
        return 8

    @field_validator("vector_fallback_mode")
    @classmethod
    def _vector_fallback_mode(cls, value: str) -> str:
        return value if value in {"off", "on"} else "off"

    @field_validator("query_rewrite_max_variants")
    @classmethod
    def _query_rewrite_max_variants(cls, value: int) -> int:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return min(value, 1)
        return 1

    @field_validator("rewrite_deadline_ms")
    @classmethod
    def _rewrite_deadline_ms(cls, value: int) -> int:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return min(value, 2000)
        return 250

    @field_validator("small_to_big_mode")
    @classmethod
    def _small_to_big_mode(cls, value: str) -> str:
        return value if value in {"off", "shadow", "on"} else "shadow"

    @field_validator("small_to_big_neighbors")
    @classmethod
    def _small_to_big_neighbors(cls, value: int) -> int:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return min(value, 4)
        return 1

    @field_validator("small_to_big_max_chars")
    @classmethod
    def _small_to_big_max_chars(cls, value: int) -> int:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return min(value, 8000)
        return 2400

    @field_validator("query_embedding_cache_size")
    @classmethod
    def _query_cache_size(cls, value: int) -> int:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return 256

    @field_validator("index_backend")
    @classmethod
    def _index_backend(cls, value: str) -> str:
        return value if value in {"auto", "p2", "legacy"} else "auto"

    @field_validator("answer_gate_mode")
    @classmethod
    def _answer_gate_mode(cls, value: str) -> str:
        return value if value in {"off", "shadow", "on"} else "shadow"


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


class ContextConfig(BaseModel):
    """P1 context/task-state controls; ``shadow`` is the safe default."""

    assembler_mode: str = "shadow"
    reserve_output_tokens: int = 16384
    memory_budget_tokens: int = 1200
    rag_budget_tokens: int = 4000
    history_budget_tokens: int = 8000
    task_state_path: str = ""

    @field_validator("assembler_mode")
    @classmethod
    def _assembler_mode(cls, value: str) -> str:
        return value if value in {"shadow", "on"} else "shadow"

    @field_validator("reserve_output_tokens", "memory_budget_tokens",
                     "rag_budget_tokens", "history_budget_tokens")
    @classmethod
    def _budget(cls, value: int, info: ValidationInfo) -> int:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
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
    # P0-06 run-level budgets; zero keeps the legacy unlimited semantics.
    max_llm_calls: int = 0
    max_react_rounds: int = 0
    max_plan_steps: int = 0
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


class RemoteOperationConfig(BaseModel):
    """Optional read-only endpoint for settling unknown remote operations."""

    base_url: str = ""
    status_path: str = "/operations/{operation_id}"
    token_env: str = "AGENT_REMOTE_OPERATION_TOKEN"
    timeout: float = 10.0
    source: str = "remote-operation-api"

    @field_validator("timeout")
    @classmethod
    def _timeout(cls, value: float) -> float:
        try:
            return max(0.5, min(float(value), 60.0))
        except (TypeError, ValueError):
            return 10.0


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    routes: dict[str, str] = {}  # 多模型路由：task_tier -> model
    vault_root: str = "C:/path/to/your/obsidian-vault"
    brain: BrainConfig = Field(default_factory=BrainConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    skills_dir: str | None = None
    trace_dir: str = "logs/trace"
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    remote_operation: RemoteOperationConfig = Field(default_factory=RemoteOperationConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    rag: RagConfig = Field(default_factory=RagConfig)  # 向量语义检索路（P0-1）
    agents: AgentsConfig = Field(default_factory=AgentsConfig)  # 外部 ACP agent（P2-1）
    approval_mode: str = "risk_based"  # risk_based=危险工具走审批/白名单；allow_all=策略自动放行已注册工具
    danger_allowlist: list[str] = Field(default_factory=list)  # danger 权限白名单（默认全拒）
    write_allowlist: list[str] | None = None  # write 权限白名单；None=用代码默认集（认可的 Vault/memory 写入器），[] = 全拒


def load_config(path: str | Path | None = None) -> Config:
    """从 config.json 加载，并应用运行时环境覆盖。

    Ark 设置页通过环境变量下发 RAG 选择和 provider 参数，避免把 API Key
    回写到 agentlab/config.json。未设置环境变量时保持文件配置和默认值。
    """
    config_path = Path(path) if path else (Path(DEFAULT_CONFIG) if Path(DEFAULT_CONFIG).exists() else None)
    data: dict = {}
    if config_path is not None and config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
    rag = dict(data.get("rag") or {})

    def env_text(name: str, key: str) -> None:
        value = os.environ.get(name)
        if value is not None:
            rag[key] = value

    def env_bool(name: str, key: str) -> None:
        value = os.environ.get(name)
        if value is not None:
            rag[key] = value.strip().lower() in {"1", "true", "yes", "on"}

    def env_float(name: str, key: str) -> None:
        value = os.environ.get(name)
        if value is not None:
            try:
                rag[key] = float(value)
            except ValueError:
                pass

    env_bool("AGENT_RAG_VECTOR_ENABLED", "vector_enabled")
    env_text("AGENT_RAG_EMBED_BASE_URL", "embed_base_url")
    env_text("AGENT_RAG_EMBED_MODEL", "embed_model")
    env_text("AGENT_RAG_EMBED_API_KEY", "embed_api_key")
    env_float("AGENT_RAG_EMBED_TIMEOUT", "embed_timeout")
    env_float("AGENT_RAG_VECTOR_MIN_SCORE", "vector_min_score")
    env_float("AGENT_RAG_LEXICAL_MIN_COVERAGE", "lexical_min_coverage")
    env_text("AGENT_RAG_VECTOR_MODE", "vector_mode")
    env_text("AGENT_RAG_VECTOR_FALLBACK_MODE", "vector_fallback_mode")
    env_text("AGENT_RAG_LEXICAL_MODE", "lexical_mode")
    env_text("AGENT_RAG_QUERY_REWRITE_MODE", "query_rewrite_mode")
    env_text("AGENT_RAG_QUERY_REWRITE_MAX_VARIANTS", "query_rewrite_max_variants")
    env_text("AGENT_RAG_REWRITE_DEADLINE_MS", "rewrite_deadline_ms")
    env_text("AGENT_RAG_QUERY_EXPANSION_MODE", "query_expansion_mode")
    env_text("AGENT_RAG_QUERY_EXPANSION_MAX_VARIANTS", "query_expansion_max_variants")
    env_text("AGENT_RAG_QUERY_EXPANSION_MIN_CANDIDATES", "query_expansion_min_candidates")
    env_text("AGENT_RAG_SMALL_TO_BIG_MODE", "small_to_big_mode")
    env_text("AGENT_RAG_SMALL_TO_BIG_NEIGHBORS", "small_to_big_neighbors")
    env_text("AGENT_RAG_SMALL_TO_BIG_MAX_CHARS", "small_to_big_max_chars")
    env_text("AGENT_RAG_CHUNK_STRATEGY", "chunk_strategy")
    env_text("AGENT_RAG_INDEX_VERSION", "index_version")
    env_text("AGENT_RAG_INDEX_BACKEND", "index_backend")
    env_text("AGENT_RAG_INDEX_PATH", "index_path")
    memory_enabled = os.environ.get("AGENT_MEMORY_ENABLED")
    if memory_enabled is not None:
        enabled = memory_enabled.strip().lower() in {"1", "true", "yes", "on"}
        limits = dict(data.get("limits") or {})
        if not enabled:
            # Disable only implicit behavior. Markdown memory and explicit tools
            # remain available for inspection and user-directed operations.
            limits["memory_inject_topk"] = 0
            limits["memorize_every"] = 0
        data["limits"] = limits
    if rag:
        data["rag"] = rag
    return Config.model_validate(data)


DEFAULT_CONFIG = "config/config.json"
PROJECT_CONFIG_EXAMPLE = "config/config.example.json"
