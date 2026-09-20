"""Build a larger, auditable RAG task set from the frozen v2 contract.

Positive tasks remain the reviewed corpus-backed rows.  Additional negatives
are deliberately typed (random token, unknown identifier, out-of-domain,
plausible absent) and must pass the corpus audit before use as live probes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NEGATIVE_SEEDS = [
    ("out_of_domain", "板球世界杯积分规则"),
    ("out_of_domain", "南极企鹅迁徙路线"),
    ("out_of_domain", "木星环维修步骤"),
    ("out_of_domain", "古罗马战车轮胎规格"),
    ("out_of_domain", "海底火山烹饪教程"),
    ("out_of_domain", "火星殖民地税务申报"),
    ("out_of_domain", "鲸鱼声呐编程课程"),
    ("out_of_domain", "莎士比亚未公开剧本清单"),
    ("out_of_domain", "量子纠缠烘焙步骤"),
    ("out_of_domain", "北欧神话股票代码"),
    ("out_of_domain", "国际象棋冠军的私人住址"),
    ("out_of_domain", "深海潜艇驾照考试"),
    ("out_of_domain", "月球矿场员工手册"),
    ("out_of_domain", "热带雨林昆虫航班时刻"),
    ("out_of_domain", "古埃及木乃伊维修协议"),
    ("random_token", "qzvplm-771-alpha"),
    ("random_token", "xk9-void-2048"),
    ("random_token", "miraq-zz-4401"),
    ("random_token", "tulip-null-991"),
    ("random_token", "n7qomega-unknown"),
    ("random_token", "vortex-absent-812"),
    ("random_token", "pluto-qx-733"),
    ("random_token", "delta-random-650"),
    ("random_token", "kappa-nohit-481"),
    ("random_token", "sierra-zzq-918"),
    ("random_token", "rho-missing-275"),
    ("random_token", "atlas-unindexed-304"),
    ("random_token", "neon-absent-552"),
    ("random_token", "orion-no-record-619"),
    ("random_token", "lyra-random-870"),
    ("unknown_identifier", "BV9QqQqQqQqQ"),
    ("unknown_identifier", "BV7ZzZzZzZzZ"),
    ("unknown_identifier", "BV1不存在编号"),
    ("unknown_identifier", "BV0000000000"),
    ("unknown_identifier", "BVabcde99999"),
    ("unknown_identifier", "BV8NoSuch1234"),
    ("unknown_identifier", "BV4Missing000"),
    ("unknown_identifier", "BV6Void888888"),
    ("unknown_identifier", "BV2Absent77777"),
    ("unknown_identifier", "BV3Unknown66666"),
    ("plausible_absent", "用户偏好荧光绿色主题"),
    ("plausible_absent", "用户偏好每次回答超过一万字"),
    ("plausible_absent", "用户决定迁移到 MongoDB 作为唯一真源"),
    ("plausible_absent", "项目决定完全删除 Markdown 真源"),
    ("plausible_absent", "用户要求默认启用全局 Hybrid"),
    ("plausible_absent", "项目已经完成 ContextAssembler 全量 on 灰度"),
    ("plausible_absent", "用户偏好所有回答都不带引用"),
    ("plausible_absent", "项目决定停止 candidate-first"),
    ("plausible_absent", "用户要求静默自动写入 Vault"),
    ("plausible_absent", "项目已接入真实邮件 operation 查询"),
    ("plausible_absent", "用户偏好使用英文回复"),
    ("plausible_absent", "项目决定 raw 目录允许修改"),
    ("plausible_absent", "系统已经通过 P3 全局灰度门禁"),
    ("plausible_absent", "项目决定取消人工 review_due"),
    ("plausible_absent", "用户偏好不进行任何测试"),
    ("plausible_absent", "项目已经启用自动 memory promote"),
    ("plausible_absent", "用户要求删除所有失败恢复屏障"),
    ("plausible_absent", "项目决定不再记录 operation audit"),
    ("plausible_absent", "用户偏好关闭所有 scope 隔离"),
    ("plausible_absent", "项目已将 answer gate 设置为全局 on"),
    ("plausible_absent", "用户决定不再使用 Obsidian Vault"),
]


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def build_rows(rows: list[dict[str, Any]], *, negative_target: int = 60) -> list[dict[str, Any]]:
    existing_negative = [row for row in rows if not row.get("expected_refs")]
    need = max(0, int(negative_target) - len(existing_negative))
    used_queries = {str(row.get("query") or "").strip() for row in rows}
    additions: list[dict[str, Any]] = []
    for kind, query in NEGATIVE_SEEDS:
        if len(additions) >= need or query in used_queries:
            continue
        task_id = f"N{len(existing_negative) + len(additions) + 1:03d}"
        additions.append({
            "id": task_id,
            "query": query,
            "expected_refs": [],
            "negative_kind": kind,
            "project_id": "default",
            "filters": {},
            "query_type": "negative",
            "notes": "扩展负例：须通过 task_audit 后才可进入 live probe",
            "dataset_schema": "rag-retrieval-task-v2",
            "corpus_scope": "full_vault",
            "route": "local_combined",
            "qrels": [],
            "graded_qrels": [],
            "answerability": "absent",
            "forbidden_refs": [],
            "allowed_refs": [],
            "expected_decision": "insufficient",
            "expected_abstention": True,
            "scope": {"project_id": "default", "session_id": "",
                      "statuses": [], "include_archive": False},
        })
        used_queries.add(query)
    if len(additions) < need:
        raise ValueError(f"only generated {len(additions)} negative rows; need {need}")
    return rows + additions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expand frozen RAG tasks with typed negative cases")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--negative-target", type=int, default=60)
    args = parser.parse_args(argv)
    rows = build_rows(load_rows(Path(args.input)), negative_target=args.negative_target)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                      encoding="utf-8")
    print(json.dumps({"output": str(output), "tasks": len(rows),
                      "positive": sum(bool(row.get("expected_refs")) for row in rows),
                      "negative": sum(not bool(row.get("expected_refs")) for row in rows)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
