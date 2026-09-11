"""
统一工具注册表：21 个 brain 工具的单一真相源

结构：(tool_name, module_suffix, description, permission, timeout_seconds)
- module_suffix: "vault" → tools_vault.py 的 vault_{tool_name}()
- permission: "read" | "write"
- timeout_seconds: None（无限制）或浮点数（秒）
"""

BRAIN_TOOL_SPECS = [
    # ==== 组名: vault (7 工具) ====
    ("vault_read", "vault", "读取 Vault 内 Markdown 笔记全文（path 为相对路径）", "read", None),
    ("vault_search", "vault", "关键词检索全库笔记（标题+正文+tag），返回路径/摘要/tags", "read", None),
    ("vault_graph", "vault", "wikilink 邻接图：从指定笔记遍历 [[wikilink]] 关联", "read", None),
    ("vault_scan", "vault", "全库统计：主题分布/tag 聚合/wikilink 密度/最近笔记", "read", None),
    ("vault_health", "vault", "结构性健康检查：死链（wikilink 指向不存在笔记）+ 孤儿笔记/无引用笔记", "read", None),
    ("vault_write", "vault", "写入/覆盖 Vault 笔记；raw/ 只读拒绝", "write", None),
    ("vault_patch", "vault", "精确替换笔记中某段文本（old 须唯一）", "write", None),

    # ==== 组名: brain (3 工具) ====
    ("brain_search", "brain", "检索 brain 内部数据（索引）", "read", None),
    ("brain_scan", "brain", "brain 数据扫描/统计", "read", None),
    ("brain_reindex", "brain", "重建 brain 索引（预留）", "read", None),

    # ==== 组名: memory (2 工具) ====
    ("memory_commit", "memory", "沉淀一条可复用记忆（content+tags）到长期记忆", "write", None),
    ("memory_query", "memory", "按 topic 召回记忆（tag+内容关键词）", "read", None),

    # ==== 组名: bili (4 工具) ====
    ("bili_meta", "bili", "B站视频元信息", "read", 60.0),
    ("bili_transcribe", "bili", "B站视频转写（3 层降级）；返回文本不落库", "read", 1800.0),
    ("bili_screenshot", "bili", "B站视频截图/网格图（brain 内部渲染并写 Inbox raw/screenshots）", "write", 600.0),
    ("bili_visual", "bili", "B站视频视觉分析（brain 内部转写+视觉+渲染+写 Inbox）", "write", 1800.0),

    # ==== 组名: article (2 工具) ====
    ("article_fetch", "article", "抓取公众号文章正文（readability）", "read", 60.0),
    ("article_summarize", "article", "公众号文章总结（brain 内部渲染并写 Inbox）", "write", 300.0),

    # ==== 组名: inbox (2 工具) ====
    ("inbox_collect", "inbox", "从收件箱拉取新链接并入队", "read", 60.0),
    ("inbox_read_queue", "inbox", "读取待处理任务队列", "read", None),
]
