"""CLI：跑答案质量基线（F5-024）。**花钱，需显式触发。**

用法：
    # ① 预检：不调模型，只验证任务集与 rubric 渲染（改完任务集先跑这个）
    python -m agentlab.eval.run_quality --dry-run

    # ② 真跑基线（读 agentlab/config/config.json 的模型与 key；只读工具面）
    python -m agentlab.eval.run_quality --out .ai/evals/quality-baseline-20260911.json

    # ③ 只跑前 N 条（先小样本试水，确认链路与花费）
    python -m agentlab.eval.run_quality --limit 3 --out /tmp/q3.json

    # ④ 续跑：上一轮中途失败（HTTP 402 / 网络 / 单条异常），已出分的任务不重付
    python -m agentlab.eval.run_quality --resume <上一轮报告.json> --out <同一路径>

    # ⑤ 本地对证（不调模型）：报告里被判 hallucinated 的条目，有多少其实在 Vault 里有出处
    python -m agentlab.eval.run_quality --audit <报告.json> --out <对证结果.json>

对比两次结果得到"分数从 X 到 Y"：
    python -m agentlab.eval.run_quality --compare a.json b.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def _print_summary(report: dict) -> None:
    s = report["summary"]
    print(f"[QUALITY] 任务 {s['tasks_scored']}/{s['tasks_total']} 已评分"
          f" · 无幻觉任务 {s['hallucination_free_tasks']}/{s['tasks_scored']}"
          f" · 云端 token {report.get('cloud_tokens_reported', 0)}")
    for dim, val in s["per_dimension_0_5"].items():
        print(f"  {dim:<14} {val:.2f} / 5.00   (归一 {s['per_dimension_0_1'][dim]:.3f})")
    print(f"  {'overall':<14} {s['overall_0_1']:.3f}")


def _compare(a: dict, b: dict) -> int:
    """逐维对比两份报告，输出变化值与方向；任一维变差返回非零。"""
    sa, sb = a["summary"], b["summary"]
    print(f"[COMPARE] {a.get('timestamp', '?')} -> {b.get('timestamp', '?')}")
    worse = 0
    for dim in sa["per_dimension_0_1"]:
        x, y = sa["per_dimension_0_1"][dim], sb["per_dimension_0_1"][dim]
        delta = round(y - x, 3)
        mark = "=" if delta == 0 else ("up" if delta > 0 else "down")
        if delta < 0:
            worse += 1
        print(f"  {dim:<14} {x:.3f} -> {y:.3f}  {mark} {delta:+.3f}")
    x, y = sa["overall_0_1"], sb["overall_0_1"]
    print(f"  {'overall':<14} {x:.3f} -> {y:.3f}  ({y - x:+.3f})")
    return 0 if worse == 0 else 1


def _probe() -> int:
    """自检评审链路：短证据 / 长证据各调用一次，打印 prompt 与响应长度。

    全量实测教训：评审模型返回空内容时，报告里只会看到"解析失败"，分不清是
    prompt 太长、输出被截断，还是 provider 出了问题。这个开关把差别摆出来。
    """
    from agentlab.core.message import Message
    from agentlab.prompts import load_prompt
    from agentlab.runtime import config as cfg_mod
    from agentlab.eval.quality import EVIDENCE_TOTAL_CHARS, build_judge, run_sync

    cfg = cfg_mod.load_config()
    llm = build_judge(cfg)
    inner_model = getattr(getattr(llm, "inner", None), "model", None)
    print(f"[PROBE] judge_model={inner_model or getattr(llm, 'model', '?')} "
          f"(agent_model={getattr(cfg.llm, 'model', '?')}, routes={dict(cfg.routes or {})})")
    task = "找出我的知识库里没有反向链接的笔记，列出前 10 条。"
    answer = "（模拟答案）知识库里没有反向链接的笔记有 10 条：A、B、C。"
    cases = [
        ("短证据", "### 工具 vault_scan 返回\n{\"orphans\": 10}"),
        ("长证据", "### 本轮工具调用清单（共 12 次）\n"
                   + "\n".join("- vault_scan" for _ in range(12))
                   # 用真实上限造长证据：证据预算提到 24000 后，探针必须覆盖新的量级，
                   # 否则"评审返回空内容"这类问题会等到跑完全量才暴露。
                   + "\n\n### 工具 vault_scan 返回\n" + "x" * EVIDENCE_TOTAL_CHARS),
    ]
    for label, evidence in cases:
        prompt = load_prompt("eval-quality-judge-user", task=task,
                             evidence=evidence, answer=answer)

        async def _call():
            return await llm.chat([Message(role="user", content=prompt)],
                                  tools=None, temperature=0.0)

        resp = run_sync(_call())
        content = resp.content or ""
        print(f"[PROBE] {label}: prompt_chars={len(prompt)} content_len={len(content)}"
              f" head={content[:80]!r}")
        if not content:
            print(f"        ⚠ 空响应 —— 检查 provider 是否报错/超限；该模型的响应：{resp!r}"[:300])
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentlab.eval.run_quality")
    p.add_argument("--tasks", default=None, help="任务集 JSONL（默认 .ai/evals/quality-tasks.jsonl）")
    p.add_argument("--out", default=None, help="报告输出路径")
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    p.add_argument("--only", default=None, help="只跑指定 id（逗号分隔），用于定向复跑单条")
    p.add_argument("--resume", default=None,
                   help="续跑：读入上一轮报告，已出分的任务直接复用、不重跑（省 token）")
    p.add_argument("--dry-run", action="store_true", help="不调模型，只验证任务集与 rubric")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None,
                   help="对比两份已生成的报告")
    p.add_argument("--probe", action="store_true",
                   help="自检评审链路：用短/长两种证据各发一次，打印 prompt 与响应长度")
    p.add_argument("--audit", default=None, metavar="REPORT",
                   help="本地对证：把报告里判为 hallucinated 的条目拿去 Vault 找出处（不调模型）")
    p.add_argument("--vault", default=None, help="配合 --audit：Vault 根目录（默认取 config.vault_root）")
    args = p.parse_args(argv)

    from agentlab.eval.quality import dry_run_report, load_quality_tasks

    if args.probe:
        return _probe()

    if args.audit:
        from agentlab.eval.quality import audit_claims
        from agentlab.runtime import config as cfg_mod

        report = json.loads(pathlib.Path(args.audit).read_text(encoding="utf-8"))
        vault = args.vault or cfg_mod.load_config().vault_root
        result = audit_claims(report, vault)
        print(f"[AUDIT] 扫描 {result['files_scanned']} 篇笔记；判为编造的 {result['claims_total']} 条中，"
              f"**{result['claims_found_in_vault']} 条在 Vault 里有出处**、"
              f"{result['claims_not_found']} 条查无实据")
        for row in result["claims"]:
            mark = "FOUND" if row["verdict"] == "found" else "none "
            print(f"  [{mark}] {row['task']:<20} {row['claim'][:60]!r}"
                  + (f"  ← {row['file']}" if row["file"] else ""))
        if args.out:
            pathlib.Path(args.out).write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[AUDIT] 结果已写入 {args.out}")
        # 有出处 = 评审假阳性：返回非零，便于脚本当卡口
        return 1 if result["claims_found_in_vault"] else 0

    if args.compare:
        a = json.loads(pathlib.Path(args.compare[0]).read_text(encoding="utf-8"))
        b = json.loads(pathlib.Path(args.compare[1]).read_text(encoding="utf-8"))
        return _compare(a, b)

    tasks = load_quality_tasks(args.tasks)
    only_ids: set[str] | None = None
    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        known = {t["id"] for t in tasks}
        unknown = [w for w in wanted if w not in known]
        if unknown:
            print(f"[QUALITY] 未知任务 id：{', '.join(unknown)}", file=sys.stderr)
            return 2
        only_ids = set(wanted)
        tasks = [t for t in tasks if t["id"] in wanted]

    if args.dry_run:
        print(json.dumps(dry_run_report(tasks), ensure_ascii=False, indent=2))
        return 0

    from agentlab.eval.quality import (SCHEMA, build_judge, run_quality_baseline,
                                       run_sync, write_report)
    from agentlab.runtime import config as cfg_mod

    cfg = cfg_mod.load_config()
    resume = None
    if args.resume:
        resume = json.loads(pathlib.Path(args.resume).read_text(encoding="utf-8"))
        if resume.get("schema") != SCHEMA:
            print(f"[QUALITY] 续跑报告的 schema 是 {resume.get('schema')!r}，"
                  f"当前是 {SCHEMA!r}；分数口径可能不同，已停止。", file=sys.stderr)
            return 2
        # 复用数要扣掉 --only 点名的那些（它们会被强制重跑），否则打印出来的数字会骗人：
        # 上一轮 12 条全有分时，明明要重跑 2 条，却打印"复用 12 条"。
        force = only_ids or set()
        reused = sum(1 for t in resume.get("tasks", [])
                     if t.get("scores") and not t.get("error") and t.get("id") not in force)
        print(f"[QUALITY] 续跑：复用 {reused} 条已评分任务，只补跑未完成的")
    # 评审必须用**固定模型**：走 routes 路由会随 prompt 长度切到推理模型，
    # 后者把 max_tokens 烧在推理上会返回空 content（OPT-197 根因）。
    judge = build_judge(cfg)
    report = run_sync(run_quality_baseline(cfg, tasks=tasks, judge_llm=judge,
                                           limit=args.limit, resume=resume,
                                           # 只有显式 --only 才强制重跑（配合 --resume 做"定点补跑"）；
                                           # 普通续跑必须保持复用，否则续跑就失去意义了。
                                           fresh_ids=only_ids))
    if args.out:
        write_report(report, args.out)
        print(f"[QUALITY] 报告已写入 {args.out}")
    _print_summary(report)
    failed = [t["id"] for t in report["tasks"] if t.get("error")]
    if failed:
        print(f"[QUALITY] 执行失败的任务：{', '.join(failed)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
