"""golden 集与评测运行器（docs/06 §2，F14）。

golden.jsonl：一个问题 + 参考来源 + 期望抽取的上下文（离线可算 metrics）。
run_eval：加载 golden，跑一遍 agent（注入 mock LLM）得到回答与检索轨迹，
          用 metrics 汇总三指标，输出达标判定。
judge：LLM-as-judge（可选）：用强模型对（question, contexts, answer) 打分越权。
"""
from __future__ import annotations

import json
import pathlib

from agentlab.core.llm import LLMProvider
from agentlab.core.message import Message
from agentlab.eval.metrics import Metrics, answer_relevancy, context_precision, faithfulness
from agentlab.prompts import load_prompt

DEFAULT_GOLDEN = pathlib.Path(__file__).parent / "golden.jsonl"


def load_golden(path: str | pathlib.Path | None = None) -> list[dict]:
    """读取 golden.jsonl；跳过空行及非法行。"""
    fp = pathlib.Path(path) if path else DEFAULT_GOLDEN
    cases = []
    for line in fp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return cases


class EvalResult:
    def __init__(self, cases: list[dict], metrics_list: list[Metrics]):
        self.cases = cases
        self.metrics_list = metrics_list

    def summary(self) -> dict:
        keys = ("context_precision", "answer_relevancy", "faithfulness", "compaction_fidelity")
        agg: dict[str, float] = {}
        for k in keys:
            vals = [getattr(m, k) for m in self.metrics_list if getattr(m, k) is not None]
            agg[k] = (sum(vals) / len(vals)) if vals else 0.0
        return {k: round(v, 3) for k, v in agg.items()}


def evaluate_case(case: dict, *, contexts: list[str] | None = None,
                  answer: str = "", retrieved: list[str] | None = None,
                  comp_fidelity: float | None = None) -> Metrics:
    """单 golden case 的离线三指标（无 LLM）。contexts 缺省用 expected_contexts。"""
    expected_refs = case.get("expected_refs") or []
    ctxs = contexts or case.get("expected_contexts") or []
    retrieved = retrieved or [f"<{r}>" for r in expected_refs]
    return Metrics(
        context_precision=context_precision(expected_refs, retrieved),
        answer_relevancy=answer_relevancy(answer, case.get("question", "")),
        faithfulness=faithfulness(answer, ctxs, case.get("question", "")),
        compaction_fidelity=comp_fidelity,
    )


async def judge_one(llm: LLMProvider, case: dict, answer: str, contexts: list[str]) -> dict:
    """LLM-as-judge：强模型对单个 case 打分（faithfulness/relevancy 0-1）。走 .st。"""
    prompt = load_prompt(
        "eval-judge-user",
        question=case.get("question", ""),
        expected_refs=", ".join(case.get("expected_refs") or []),
        contexts="\n".join(f"- {c}" for c in contexts) or "（空）",
        answer=answer,
    )
    resp = await llm.chat([Message(role="user", content=prompt)], tools=None, temperature=0.0)
    from agentlab.core.guardrails import extract_json
    return extract_json(resp.content or "")


_THRESHOLDS = {"faithfulness": 0.9, "answer_relevancy": 0.85, "context_precision": 0.8}


def run_eval(cases: list[dict], summaries: list[dict]) -> dict:
    """汇总并判定是否达标（docs/06 §3.2 P4 验收）。summaries 为每 case 的 summary()。"""
    overall = {
        k: (sum(s.get(k, 0) for s in summaries) / len(summaries)) if summaries else 0.0
        for k in _THRESHOLDS
    }
    passed = {k: overall[k] >= _THRESHOLDS[k] for k in _THRESHOLDS}
    overall = {k: round(v, 3) for k, v in overall.items()}
    return {"overall": overall, "thresholds": _THRESHOLDS, "passed": passed,
            "all_passed": all(passed.values()), "n": len(cases)}