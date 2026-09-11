""""大脑"检索工具：全量理解 / 统计 / 索引接口。

本期用关键词 + wikilink 检索（用户已确认 #2 不做向量化）；
brain_reindex 为后期向量索引预留占位接口，调用返回"未启用"不报错。
"""
from tools_vault import vault_scan, vault_search


def brain_search(config: dict, query: str, limit: int = 20) -> dict:
    """全库关键词检索（本期实现）。

    后续语义向量化升级后，本工具将替换为向量 top-K 召回。
    """
    return vault_search(config, query, limit=limit)


def brain_scan(config: dict) -> dict:
    """全库扫描统计（主题 / tag / wikilink 密度），供产出链路分析。"""
    return vault_scan(config)


def brain_reindex(config: dict) -> dict:
    """重建向量索引（后期迭代预留接口）。

    本期未启用：不引入 embedding 依赖，返回明确状态而非报错。
    """
    return {"status": "not_enabled",
            "message": "向量索引为后期迭代（V3）能力，本期用关键词+wikilink 检索；"
                       "请使用 brain_search / brain_scan。"}
