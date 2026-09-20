"""brain 连接器（docs/03 §3.3）：把 obsidian_agent_brain 的 21 工具包装成 Tool。

- 同进程直调 brain 工具函数；引入 brain 目录到 sys.path（brain 用 flat import）。
- 每个 Tool 通过闭包绑定 brain config，参数签名从工具函数自动派生（剔除 config 参数）。
- 读写分权：write 组 → HITL 确认（registry 层）；raw/ 写入 → 连接器内拒绝（纵深防御，
  与 brain 内建 assert_writable 双保险）。
- 渲染类工具（bili_screenshot/bili_visual/article_summarize）在 brain 内部完成 LLM 渲染
  并写 Inbox，框架只编排不二次渲染（AGENTS.md 能力分流）。
"""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

from agentlab.core.errors import AgentError
from agentlab.tools.base import ExecutionMode, Tool, tool, validate_contract

# brain 目录与 agentlab 项目根同父：<obsidian_agent>/obsidian_agent_brain
BRAIN_PKG_DIR = Path(__file__).resolve().parents[4] / "obsidian_agent_brain"


def _brain_file() -> Path:
    return BRAIN_PKG_DIR / "tools_vault.py"


def brain_available() -> bool:
    return _brain_file().exists()


def _add_brain_path() -> None:
    s = str(BRAIN_PKG_DIR)
    if s not in sys.path:
        sys.path.insert(0, s)


def load_brain_config(config: dict | None) -> dict | None:
    """加载并初始化 brain 配置（common.load_config + setup_paths + vault_root 注入）。

    供 brain 工具与 RAG 召回共用同一份 brain config；未就绪返回 None，不抛错。
    """
    if not brain_available():
        return None
    _add_brain_path()
    brain_cfg_path = (config or {}).get("brain", {}).get("config_path")
    if brain_cfg_path and not Path(brain_cfg_path).is_absolute():
        # config_path 相对 agentlab/config/ 目录解析（与 config.json 默认值一致）
        brain_cfg_path = str((Path(__file__).resolve().parents[3] / "config" / brain_cfg_path).resolve())
    try:
        import common
        brain_cfg = common.load_config(brain_cfg_path)
    except Exception:
        return None
    try:
        import common
        common.setup_paths(brain_cfg)
    except Exception:
        pass
    return brain_cfg


def _wrap(fn, config: dict, permission: str):
    """绑定 config，剔除 config 参数后包装；write 工具追加 raw/ 路径守卫。"""
    sig = inspect.signature(fn)
    params = [
        p for p in sig.parameters.values()
        if not (p.name == "config" or p.name in ("self", "cls"))
    ]
    wrapped_sig = sig.replace(parameters=params)

    def wrapped(*args, **kwargs):
        if permission == "write":
            raw_hint = kwargs.get("path") or kwargs.get("bvid")
            if isinstance(raw_hint, str):
                seg = raw_hint.replace("\\", "/").lstrip("./").split("/", 1)[0].lower()
                # raw/（原始素材）与 templates/（模板）均为只读，禁写入（AGENTS.md §5）
                if seg in ("raw", "templates"):
                    raise AgentError("AGENT_TOOL_PERMISSION", f"{seg}/ 目录只读，禁止写入: {raw_hint}")
        return fn(config, *args, **kwargs)

    wrapped.__signature__ = wrapped_sig
    return wrapped


# 转写结果内存缓存（按 bvid）：strategy_whisper 单次约数分钟，且工具无跨调用缓存；
# 不加缓存时 agent 因结果被截断/重试会反复拉取，每次重跑 Whisper（卡顿根因之一）。
_TRANSCRIPT_CACHE: dict[str, dict] = {}

# 视觉分析结果缓存（按 bvid，OPT-218）：bili_visual 含下载/截帧/网格图/视觉模型
# 逐格分析，实测约 4 分钟；serve 断连重试后同 bvid 重跑 = 全部重做。与转写缓存
# 同语义（进程内、重启即失效、重新分析需重启 serve）。
_VISUAL_CACHE: dict[str, dict] = {}

# 文章总结缓存（按 url，OPT-222）：article_summarize 抓取+LLM 总结约 1-2 分钟，
# "处理收件箱"重发循环中同 URL 重复渲染烧钱；同语义（进程内）。
_ARTICLE_CACHE: dict[str, dict] = {}


def _with_key_cache(fn, cache: dict, key_param: str = "bvid"):
    """把 (config, <key_param>, ...) 形态的工具包装成"同 key 命中缓存即返回"。

    registry 以 kwargs 调度工具，key 参数名必须与原函数签名一致（bili 系=bvid、
    article=url），否则 key 恒为空导致所有调用撞同一个缓存槽。
    关键：须保留原函数签名——若包装函数自带 **kw 收集，_wrap 的 inspect.signature
    会把 VAR_KEYWORD 名 `kw` 当作工具参数写入 schema，导致 registry 调度时报
    "unexpected keyword argument 'kw'"（转写直接失败，写笔记流程被阻断）。
    """
    if key_param == "url":
        def cached(config, url=None, **kw):
            key = str(url or "")
            hit = cache.get(key)
            if hit is not None:
                return {"cached": True, **hit}
            out = fn(config, url, **kw)
            cache[key] = out
            return out
    else:
        def cached(config, bvid=None, **kw):
            key = str(bvid or "")
            hit = cache.get(key)
            if hit is not None:
                return {"cached": True, **hit}
            out = fn(config, bvid, **kw)
            cache[key] = out
            return out
    cached.__signature__ = inspect.signature(fn)  # 签名对齐原函数，rw 不泄漏 kw
    return cached


def _with_transcribe_cache(fn):
    return _with_key_cache(fn, _TRANSCRIPT_CACHE, "bvid")


def build_brain_tools(vault_root: str, config: dict, eager: bool = False) -> list[Tool]:
    """构造 brain 工具集。未就绪（无 brain 目录/未配置）时返回空，不抛错。

    从 tool_registry.BRAIN_TOOL_SPECS 读取 21 工具规格（name/module/desc/permission/timeout），
    动态加载对应模块函数并包装为 Tool。
    """
    brain_cfg = load_brain_config(config)
    if brain_cfg is None:
        return []
    # vault_root 注入：允许测试指向临时 Vault
    if vault_root:
        brain_cfg["vault_path"] = str(vault_root)
    # Propagate non-secret memory governance into direct brain tools.  The
    # request scope itself is read from ContextVar by tools_memory, so a model
    # cannot widen project/session visibility through tool arguments.
    memory_cfg = (config.get("memory") if isinstance(config, dict)
                  else getattr(config, "memory", None))
    if memory_cfg is not None:
        try:
            brain_cfg["memory_policy"] = memory_cfg.model_dump()
        except AttributeError:
            if isinstance(memory_cfg, dict):
                brain_cfg["memory_policy"] = dict(memory_cfg)

    # 1) 导入工具注册表
    _add_brain_path()
    try:
        from tool_registry import BRAIN_TOOL_SPECS
    except ImportError:
        # 兼容性：若 tool_registry 不存在（旧版本），返回空
        return []

    # 2) 顶层 flat import（brain 工具模块依赖 common 同目录）
    mods: dict[str, object] = {}
    for mod_name in ("tools_vault", "tools_brain", "tools_memory",
                     "tools_bili", "tools_article", "tools_inbox"):
        mods[mod_name] = importlib.import_module(mod_name)

    # 3) 按 spec 组装 Tool（BrainToolSpec: name/module/desc/permission/timeout
    #    + side_effects/idempotent 契约字段）
    tools: list[Tool] = []
    for spec in BRAIN_TOOL_SPECS:
        name, module_suffix = spec.name, spec.module_suffix
        description, permission, timeout = spec.description, spec.permission, spec.timeout
        mod_name = f"tools_{module_suffix}"
        if mod_name not in mods or not hasattr(mods[mod_name], name):
            continue  # 该函数不存在：预留 spec（代码未实现），自动跳过
        fn = getattr(mods[mod_name], name)

        if name == "bili_transcribe":
            # 转写缓存：同 bvid 重复调用直接返回，避免重跑 Whisper（约数分钟/次）
            fn = _with_transcribe_cache(fn)
            fn.__name__ = name
            fn.__doc__ = description

        if name == "bili_visual":
            # 视觉缓存（OPT-218）：截帧+视觉模型实测约 4 分钟，断连重试/重复调用
            # 同 bvid 直接复用，不再全量重做
            fn = _with_key_cache(fn, _VISUAL_CACHE, "bvid")
            fn.__name__ = name
            fn.__doc__ = description

        if name == "article_summarize":
            # 文章总结缓存（OPT-222）：抓取+LLM 总结约 1-2 分钟；真机"处理收件箱"
            # 重发循环中同 URL 重复渲染烧钱。key=url，与 bvid 缓存同语义（进程内）。
            fn = _with_key_cache(fn, _ARTICLE_CACHE, "url")
            fn.__name__ = name
            fn.__doc__ = description

        wrapped = _wrap(fn, brain_cfg, permission)
        wrapped.__name__ = name
        wrapped.__doc__ = description
        tools.append(
            tool(
                name=name,
                description=description,
                permission=permission,  # type: ignore[arg-type]
                execution_mode="parallel",
                can_terminate=False,
                execution_timeout=timeout or 90.0,  # None → 90s 兜底
                side_effects=spec.side_effects or None,
                idempotent=spec.idempotent,
            )(wrapped)
        )

    # 契约门禁：write/danger 工具缺 side_effects/idempotent 视为装配失败（fail-closed）
    errors = validate_contract(tools)
    if errors:
        raise AgentError("AGENT_TOOL_CONTRACT", "brain 工具契约不完整: " + "; ".join(errors))
    return tools
