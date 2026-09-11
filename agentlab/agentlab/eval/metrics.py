"""Ragas 类指标的轻量计算实现（docs/06 §2.2，F14）。

纯规则 / 纯计算，不依赖 LLM，可离线单测；对齐目标（>= 阈值由 run_eval 汇总判定）。
指标基于"关键词 / 引用命中"的启发式近似，避免引入 LLM 打分成本：
- faithfulness：答案对检索上下文的忠实度（答案中出现在上下文里的 token 占比，反编造）
- answer_relevancy：答案对问题的相关度（问题关键词在答案中的覆盖率）
- context_precision：检索结果对参考来源的精准度（top-k 命中 expected_refs 的比例）
"""
from __future__ import annotations

import re

# 中文/英文/数字 token；忽略单字中文与停用词以降低噪声
# 英文/数字 token（忽略单字母）；中文字符单独抽取后合成相邻 bigram
_EN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{1,}|\d+(?:\.\d+)?")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{2,}")
# 句子终止符（块尾以这些收尾才算"断句完整"的边界）
_SENT_END = "。！？…；"

_STOP = {
    "这个", "那个", "什么", "如何", "怎么", "是否", "一个", "以及", "并且",
    "the", "and", "for", "that", "with", "you", "this",
}


def _tokens(text: str) -> set[str]:
    """中文用相邻二元组、英文/数字用整 token；去停用。用于覆盖率近似。"""
    toks: set[str] = set()
    for m in _EN.findall(text or ""):
        toks.add(m.lower())
    for run in _CJK_RUN.findall(text or ""):
        for i in range(len(run) - 1):
            bigram = run[i:i + 2]
            if bigram not in _STOP:
                toks.add(bigram)
    return toks


def faithfulness(answer: str, contexts: list[str], question: str | None = None) -> float:
    """答案对检索内容的忠实度（反编造）。

    - 排除"源自问题"的 token（用户输入，重复问题词不算编造）；
    - 其余事实 token 中 % 出现在检索上下文。
    - 无可证 token（回答只复述问题）视为无编造 → 1.0。

    问题推导用"字符出现"而非"bigram 精确命中"判定：bigram 切词存在跨词
    边界伪 token（如"对比结论差异"中的"比结"），仅两字符都在问题中出现
    即视为复述问题，避免误判为新增编造事实。
    """
    ans = _tokens(answer)
    if question:
        qchars = set(question.lower())
        ans = {t for t in ans if not (t[0] in qchars and t[1] in qchars)}
    if not ans:
        return 1.0  # 无新增事实 = 无编造
    blob = " ".join(contexts).lower()
    hit = sum(1 for t in ans if t in blob)
    return hit / len(ans)


def answer_relevancy(answer: str, question: str) -> float:
    """问题关键词在答案中的覆盖率（回答是否切题）。"""
    q = _tokens(question)
    if not q:
        return 0.0
    low = (answer or "").lower()
    hit = sum(1 for t in q if t in low)
    return hit / len(q)


def context_precision(expected_refs: list[str], retrieved: list[str]) -> float:
    """检索 top-k 中命中参考来源的比例（有无噪声）。"""
    refs = {r.lower() for r in expected_refs}
    if not retrieved:
        return 0.0
    # 以 ref 的 basename 或完整路径匹配
    def _match(candidate: str) -> bool:
        c = candidate.lower()
        return c in refs or any(ref in c or c in ref for ref in refs)
    hit = sum(1 for r in retrieved if _match(r))
    return hit / len(retrieved)


def chunk_boundary_coherence(chunks: list[str]) -> float:
    """块尾断句完好度（对齐 notion-second-brain 的"句感知分块"评估判据）。

    度量现有分块是否在**语义边界**收尾：块以句子终止符（。！？…）结尾视为
    边界完整；在句中硬切则上下文被破坏（块尾停在句中）。比例越高越好。
    """
    if not chunks:
        return 0.0
    ok = sum(1 for c in chunks if c and c.rstrip().endswith(tuple(_SENT_END)))
    return ok / len(chunks)


class Metrics:
    """对 single case 聚合三个指标（用于汇总平均）。"""

    def __init__(self, context_precision: float | None = None,
                 answer_relevancy: float | None = None,
                 faithfulness: float | None = None,
                 compaction_fidelity: float | None = None,
                 chunk_boundary: float | None = None):
        self.context_precision = context_precision
        self.answer_relevancy = answer_relevancy
        self.faithfulness = faithfulness
        self.compaction_fidelity = compaction_fidelity
        self.chunk_boundary = chunk_boundary

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if v is not None else None)
                for k, v in self.__dict__.items()}