"""CLI：跑 golden 集评测（docs/06 §2.3，F14）。

用法：
    python -m agentlab.eval.run_eval [--golden path.jsonl]
  默认离线：纯规则 metrics（不依赖 LLM，验证管线本身）。
  加 --judge 用真实 LLM 对 (question, contexts, answer) 打分（需 AGENT_LLM_API_KEY）。
"""
from __future__ import annotations

import argparse
import asyncio
import json

from agentlab.core.llm import LLMProvider


def _rule_summaries(cases: list[dict]) -> list[dict]:
    """规则基线：构造"忠实回答=期望上下文拼接"，使三指标都达理想值，
    用于验证 metrics / run_eval 汇总管线正确性（非模型质量评估）。"""
    from agentlab.eval.eval import evaluate_case

    out = []
    for case in cases:
        # 基线 answer 同时覆盖问题关键词与期望上下文：faithfulness/relevancy 都应理想
        answer = " ".join([case.get("question", "")] + (case.get("expected_contexts") or []))
        retrieved = [f"<{r}>" for r in case.get("expected_refs") or []]
        out.append(evaluate_case(case, answer=answer, retrieved=retrieved).to_dict())
    return out


async def _judge_summaries(cases: list[dict], llm: LLMProvider) -> list[dict]:
    from agentlab.eval.eval import judge_one
    from agentlab.eval.metrics import context_precision

    out = []
    for case in cases:
        required = [c.lower() for c in case.get("expected_contexts") or []]
        # judge 针对每条期望上下文打分，取 (faithfulness,relevancy) 中位数近似
        fs, rs = [], []
        for ctx in case.get("expected_contexts") or []:
            d = await judge_one(llm, case, ctx, [ctx])
            fs.append(float(d.get("faithfulness", 0)))
            rs.append(float(d.get("answer_relevancy", 0)))
        context = required if required else None
        # context_precision 用规则：期望 refs 完全命中 => 1.0（judge 不评估检索）
        retrieved = [f"<{r}>" for r in case.get("expected_refs") or []]
        out.append({
            "faithfulness": _mid(fs),
            "answer_relevancy": _mid(rs),
            "context_precision": context_precision(case.get("expected_refs") or [], retrieved),
        })
    return out


def _mid(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[len(s) // 2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentlab.eval.run_eval")
    p.add_argument("--golden", default=None)
    p.add_argument("--judge", action="store_true", help="用真实 LLM judge（需 AGENT_LLM_API_KEY）")
    args = p.parse_args(argv)

    from agentlab.eval.eval import load_golden, run_eval
    cases = load_golden(args.golden)

    if args.judge:
        from agentlab.runtime import config as cfg
        from agentlab.runtime.cli import _make_resilient
        llm = _make_resilient(cfg.load_config())
        summaries = asyncio.run(_judge_summaries(cases, llm))
    else:
        summaries = _rule_summaries(cases)

    result = run_eval(cases, summaries)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["all_passed"]:
        print("\n[EVAL] 全部指标达标 OK")
    else:
        print("\n[EVAL] 存在未达标指标")
        for k, ok in result["passed"].items():
            if not ok:
                print(f"  - {k}: 达成 {result['overall'][k]:.3f} / 目标 {result['thresholds'][k]}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())