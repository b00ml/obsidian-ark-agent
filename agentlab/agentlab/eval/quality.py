"""答案质量基线（F5-024）：真跑 Agent + rubric 评审，产出可比较的分数。

与 `behavior.py`（离线契约回归，不调模型、不评答案好坏）互补——本模块**会花 token**，
因此必须显式触发、不进 CI、不进 `npm run build`。

设计要点：
- **只读**：用 `_run_agent(tool_filter=...)` 把工具面收敛为 `permission == "read"`，
  评测永远不可能写用户 Vault（评测要能反复跑，副作用不可接受）。
- **真链路**：走 `_build_backend` 的完整 registry（含 brain/RAG 工具），因此测的是
  "检索 + system prompt + 记忆 + 压缩"组合后的真实表现，不是 mock。
- **可比较**：报告落 JSON（含时间戳/模型/逐任务分数与理由），下次改动后重跑即可得到
  "分数从 X 到 Y"，而不是"我觉得更好了"。
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import time
from typing import Any

from agentlab.core.message import Message
from agentlab.prompts import load_prompt

# 四维 rubric：与 prompts/eval-quality-judge-user.st 一一对应，改动必须两边同步。
QUALITY_DIMENSIONS = ("factuality", "coverage", "traceability", "actionability")
MAX_SCORE_PER_DIM = 5
SCHEMA = "f5-024.v1"

# 评测工具面：**纯查询白名单**，不是"read 权限"。
#
# 教训（首轮 smoke 实测）：`permission == "read"` 只挡 Vault 写入，挡不住外部副作用——
# `inbox_collect` 会拉邮件并把条目写进队列库、`bili_*` 会下载并写 cookie 文件、
# `*_reindex` 会重建索引、`web_search`/`article_fetch` 会打外网。评测必须可反复跑且
# 不改变任何状态，所以这里用显式白名单，宁缺勿滥。
#
# 已知局限：这是按名字匹配，新增工具若不在名单里会被静默排除（保守方向的失误，
# 不会误放行副作用）。更彻底的修法是给工具元数据加 `side_effects` 标记，见 OPT-196 边界。
EVAL_TOOL_ALLOWLIST: frozenset[str] = frozenset({
    "vault_read", "vault_search", "vault_graph", "vault_scan", "vault_health",
    "brain_search", "brain_scan",
    "memory_query",
    # rag_retrieve / rag_assess 会顺带做有界的索引自愈（产品每次检索都如此），属幂等缓存，
    # 不是用户数据；排除它们等于不测真实检索链路，故保留。
    "rag_retrieve", "rag_assess",
    "runs_recent",
})

# 证据块上限。历史沿革必须完整记下，否则很容易再退回去：
# · 总量 4000：太小——Agent 读 25 次工具时只有前几次进得去，评审把真实笔记判成"编造"。
# · 总量 12000 时"评审返回空内容"曾被归因于此，于是单工具压到 400 字。**这个归因是错的**：
#   真因是 `routes` 把长 prompt 路由到推理模型、推理 token 吃满 max_tokens 导致 content 为空
#   （OPT-197 定位，评审现已钉死 deepseek-chat + 800 token）。
# · 400 字留下的后遗症：**转述型断言**无法核验——Agent 概述它读过的长文，评审在 400 字节选里
#   找不到支撑就判 `hallucinated`（OPT-206 实测：答案引用的 `Inbox/Yuxi…文章.md`、
#   `Inbox/深度解读-Codex-Harness-源码-文章.md` 都真实存在，仍被判编造）。
# 现值：总量 24000 字，**按调用次数公平分配**（见 evidence 属性）——额度不再是"前面的调用
# 全吃、后面的喝汤"。并且**不再给单工具设硬顶**：只有 1 次调用时，硬顶会把工具返回的清单
# 从中间切开（实测 q-orphan-notes 单次 `vault_health` 返回 3,168 字被切到 1,200，评审看不到
# 答案引用的路径，又判"编造"）。调用少时就让那条结果完整进来，靠总量上限兜底。
EVIDENCE_TOTAL_CHARS = 24000
# 单条工具结果的原始留存上限（只用于内存与后续公平分配，不是最终证据长度）。
EVIDENCE_RAW_PER_TOOL_CHARS = 6000
# 证据里额外附一份"工具调用清单"（名字 + 参数摘要）：让评审判可追溯性时能看到
# "到底查过什么"，而不是只依赖正文片段是否被截进来。
TOOL_CALL_SUMMARY_CHARS = 160

# 评审模型的输出上限：rubric 已约束输出长度（三项各 ≤3 条、每条 ≤20 字），
# 800 足够，且能避免推理模型把预算烧在推理上导致 content 为空。
JUDGE_MAX_TOKENS = 800


def judge_model_name(cfg) -> str:
    """评审用哪个模型：**必须固定且可覆盖**，否则"基线"在不同 prompt 长度下会用不同模型。"""
    import os

    return os.environ.get("AGENTLAB_EVAL_JUDGE_MODEL") or str(
        getattr(getattr(cfg, "llm", None), "model", "") or "")


def build_judge(cfg):
    """构造固定模型的评审 provider（不走 routes 路由，见 OPT-197 根因）。"""
    from agentlab.runtime.cli import _resilient_for_model

    return _resilient_for_model(cfg, judge_model_name(cfg), max_tokens=JUDGE_MAX_TOKENS)


def default_tasks_path() -> pathlib.Path:
    """仓库级任务集：<repo>/.ai/evals/quality-tasks.jsonl"""
    return pathlib.Path(__file__).resolve().parents[3] / ".ai" / "evals" / "quality-tasks.jsonl"


def load_quality_tasks(path: str | pathlib.Path | None = None) -> list[dict[str, Any]]:
    """读取任务集；跳过空行与 `#` 注释，拒绝重复 id 与缺少 task 的行。"""
    fp = pathlib.Path(path) if path else default_tasks_path()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{fp}:{line_no} 不是合法 JSON：{e}") from e
        task_id = str(item.get("id", "")).strip()
        if not task_id or task_id in seen:
            raise ValueError(f"{fp}:{line_no} id 缺失或重复：{task_id!r}")
        if not str(item.get("task", "")).strip():
            raise ValueError(f"{fp}:{line_no} 缺少 task 字段")
        seen.add(task_id)
        rows.append(item)
    if not rows:
        raise ValueError(f"{fp} 没有任何任务（质量基线至少需要 1 条）")
    return rows


def eval_tool_filter(tool) -> bool:
    """评测专用工具面：只放行纯查询白名单（见 `EVAL_TOOL_ALLOWLIST`）。"""
    return getattr(tool, "name", None) in EVAL_TOOL_ALLOWLIST


def _head_tail(text: str, limit: int) -> str:
    """头 + 尾截断：工具返回的计数/结论常在末尾，只留头部会让评审把真实数字判成幻觉。

    （实测：`q-orphan-notes` 的"135 篇"出现在返回值末尾，只保头部时评审看不到。）
    截断处显式标注，避免评审误以为看到了全文。
    """
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    return f"{text[:head]}\n…（中段省略）…\n{text[-tail:]}"


class _CollectSink:
    """最小 sink：只收集回答文本，忽略工具/审批事件（评测不看过程）。"""

    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.tools: list[str] = []
        self.tool_args: list[str] = []
        self.tool_results: list[tuple[str, str]] = []

    def text(self, content: str) -> None:
        if content:
            self.text_parts.append(content)

    def tool_start(self, name: str, arguments: str = "{}") -> None:
        self.tools.append(name)
        # 参数必须留：没有参数就无法判断"它是查了什么才这么说"，评审只能凭印象判编造，
        # 事后也没法复现"到底搜过哪些关键词"（全量基线里 q-memory-recall 的 25 次搜索
        # 是从 trace 里逐条挖出来的，报告本身只存了工具名）。
        self.tool_args.append(str(arguments or "{}")[:240])

    def tool_end(self, name: str, result: str) -> None:
        """记录工具结果——这是评审判定事实性的**唯一真依据**。

        首轮 smoke 实测教训：此前这里直接 return None，评审只拿到"由 Agent 现场检索得到"
        这种空话，于是把真实结论也判成幻觉（factuality 0/1）。不喂证据的评审没有意义。
        """
        if result:
            # 这里只做"原始留存上限"，最终长度在 evidence 属性里按**公平份额**决定：
            # 若在此处就压到单工具上限，后面的公平分配就无料可分了。
            self.tool_results.append((name, _head_tail(str(result), EVIDENCE_RAW_PER_TOOL_CHARS)))

    def event(self, payload: dict) -> None:
        return None

    @property
    def answer(self) -> str:
        """注意：不能叫 text——那会与上面的 sink 方法同名，后者会被属性覆盖。"""
        return "".join(self.text_parts).strip()

    @property
    def evidence(self) -> str:
        """拼评审可核验的证据块：**调用清单**（查过什么）+ 结果节选（看到了什么）。"""
        if not self.tool_results:
            return ""
        calls = "\n".join(
            f"- {name} {self.tool_args[i] if i < len(self.tool_args) else '{}'}"
            for i, name in enumerate(self.tools[:40])
        ) or "（无）"
        blocks: list[str] = [f"### 本轮工具调用清单（共 {len(self.tools)} 次，含参数）\n{calls}"]
        sources = self._sources()
        if sources:
            blocks.append("### 本轮读到的来源（路径）\n" + "\n".join(f"- {s}" for s in sources))
        scalars = self._scalars()
        if scalars:
            blocks.append("### 工具返回的标量字段（原样，未经推断）\n" + "\n".join(f"- {s}" for s in scalars))
        # 三段的长度也要计入总量：此前它们不占额度，`EVIDENCE_TOTAL_CHARS` 名不副实
        # （实测 12000 额度下整块能到 13391）。
        used = sum(len(b) + 2 for b in blocks)
        # 公平份额：额度按**调用次数**分，而不是让前几次调用吃光、后面的整条丢失。
        # 旧实现额度耗尽就 `break`，读到 46 次工具时末尾十几条证据全被丢掉——而答案往往
        # 引用的正是后面读到的那几篇。
        n = max(1, len(self.tool_results))
        # 每段还有固定开销（"### 工具 <name> 返回\n" + 段间空行 = 14 + len(name)），
        # 必须先按实际名字长度扣掉，否则末尾几条仍会被挤出去——这正是公平分配要解决的问题。
        overhead = 14 + max((len(nm) for nm, _ in self.tool_results), default=8)
        avail = max(0, EVIDENCE_TOTAL_CHARS - used - overhead * n)
        per_tool = avail // n
        for idx, (name, result) in enumerate(self.tool_results):
            remain = EVIDENCE_TOTAL_CHARS - used
            if remain < 80:  # 连标题都放不下 → 明确标注，不静默丢
                blocks.append(f"### 其余 {len(self.tool_results) - idx} 次工具返回因证据上限未纳入")
                break
            # 末尾几条按剩余空间收缩，而不是整条丢掉：保证**每次调用都有代表**
            room = max(0, remain - (len(name) + 16))
            block = f"### 工具 {name} 返回\n{_head_tail(result, min(per_tool, room))}"
            blocks.append(block)
            used += len(block) + 2
        return "\n\n".join(blocks)

    def _sources(self) -> list[str]:
        """工具返回里出现过的**来源路径**（去重、保序、限量）。

        为什么单列：评审要判"可追溯性"，就得看到"它到底读过哪些文件"。全量基线里
        `q-memory-recall` 引用的 `RAG向量增删.md` / `AGENTS.md` 都在证据节选之外，
        评审判它编造——这是假阳性的直接来源。
        """
        import re

        seen: dict[str, None] = {}
        for _name, result in self.tool_results:
            for m in re.finditer(r"'(?:path|ref|note|source_file)':\s*'([^']{1,120})'", str(result)):
                seen.setdefault(m.group(1), None)
            if len(seen) >= 40:
                break
        return list(seen)[:40]

    def _scalars(self) -> list[str]:
        """工具返回里的**顶层标量字段**（`'key': 数字`），原样列出。

        为什么单列：计数类结论（`orphan_count=48` / `broken_links_count=135`）在长返回值里
        往往落在被省略的中段，评审看不到就把答案里的数字判成编造（q-orphan-notes 实测）。

        除数字外，还收录**短字符串事实**（时间/编号/状态/类型，值 ≤48 字）：全量基线里
        `q-daily-review` 引用的 `15:54 / 16:00` 来自 `runs_recent` 的时间字段，被判编造——
        它确实在工具返回里，只是被截断在中段。
        """
        import re

        seen: dict[str, str] = {}
        for name, result in self.tool_results:
            for m in re.finditer(r"'([a-z_][a-z0-9_]{2,30})':\s*(-?\d+(?:\.\d+)?)", str(result)):
                seen.setdefault(m.group(1), m.group(2))
            for m in re.finditer(r"'([a-z_][a-z0-9_]{2,30})':\s*'([^']{1,48})'", str(result)):
                seen.setdefault(m.group(1), m.group(2))
            if len(seen) >= 24:
                break
        return [f"{k} = {v}" for k, v in list(seen.items())[:24]]


async def run_task(cfg, build_backend, task: dict, *, project_id: str | None = None) -> dict:
    """真跑一条任务（纯查询工具面），返回 {answer, tools, evidence, stop_reason, tokens}。"""
    from agentlab.runtime.serve import _run_agent

    sink = _CollectSink()
    summary = await _run_agent(
        cfg, build_backend, [], str(task["task"]), sink,
        project_id=project_id, tool_filter=eval_tool_filter,
    )
    return {
        "answer": sink.answer or str(summary.get("final_output") or ""),
        "tools": sink.tools,
        "evidence": sink.evidence,
        "stop_reason": summary.get("stop_reason", ""),
        "tokens": int(summary.get("tokens") or 0),
    }


async def judge_answer(llm, task: dict, answer: str, evidence: str = "") -> dict:
    """评审单条答案，返回四维分数 + 幻觉/遗漏清单 + 理由。

    `evidence` 是本轮工具实际返回的内容（run_task 采集）；为空时才退回任务自带的说明。
    不喂真证据的评审会把真实结论判成幻觉——首轮 smoke 实测到的坑。

    评审模型可能输出代码块或多余文字，统一走 `extract_json` 并在失败时抛错——
    **不静默给 0 分**，否则一次解析失败会被误读成"答案质量差"。
    """
    from agentlab.core.guardrails import extract_json

    evidence_block = (evidence or "").strip() or str(task.get("evidence", "") or "（无外部依据）")
    prompt = load_prompt(
        "eval-quality-judge-user",
        task=str(task.get("task", "")),
        evidence=evidence_block,
        answer=answer or "（空回答）",
    )
    resp = await llm.chat([Message(role="user", content=prompt)], tools=None,
                          temperature=0.0, max_tokens=JUDGE_MAX_TOKENS)
    raw = resp.content or ""
    # 评审调用的元数据必须留痕：全量实测里出现"content 为空"，而报告只写"解析失败"，
    # 分不清是 provider 空响应、被 max_tokens 截断（finish_reason=length）还是别的。
    usage = getattr(resp, "usage", None)
    meta = {
        "stop_reason": str(getattr(resp, "stop_reason", "") or ""),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "raw_chars": len(raw),
        "prompt_chars": len(prompt),
    }
    try:
        data = extract_json(raw)
    except Exception as e:  # noqa: BLE001
        # 全量实测：评审模型会把 JSON 撑爆（hallucinated 里引用大段原文），失配后整条任务判失败。
        # 四个维度分数组在最前面，因此**打捞分数**比整条丢弃更有用——但要显式标记，
        # 不能假装拿到了完整评审。
        salvaged = _salvage_scores(raw)
        if salvaged is None:
            raise ValueError(
                f"评审输出无法解析为 JSON（{type(e).__name__}）：{raw[:200]!r} | meta={meta}"
            ) from e
        data = salvaged
        data["_truncated"] = True
    if not isinstance(data, dict):
        raise ValueError(f"评审输出不是 JSON 对象：{raw[:200]!r}")
    out: dict[str, Any] = {}
    for dim in QUALITY_DIMENSIONS:
        try:
            score = float(data.get(dim, 0))
        except (TypeError, ValueError):
            score = 0.0
        out[dim] = max(0.0, min(float(MAX_SCORE_PER_DIM), score))
    out["hallucinated"] = [str(x) for x in (data.get("hallucinated") or []) if str(x).strip()]
    out["unverifiable"] = [str(x) for x in (data.get("unverifiable") or []) if str(x).strip()]
    out["missing"] = [str(x) for x in (data.get("missing") or []) if str(x).strip()]
    out["reason"] = str(data.get("reason", "") or "")
    if data.get("_truncated"):
        out["judge_truncated"] = True
    out["judge_meta"] = meta
    return out


_SCORE_RE = None


def _salvage_scores(raw: str) -> dict | None:
    """从被截断的评审输出里打捞四个维度分。四个分数都在 JSON 最前面，通常完整。

    返回 None 表示连分数都没捞到（此时调用方应当抛错——不静默给 0）。
    """
    global _SCORE_RE
    if _SCORE_RE is None:
        import re
        _SCORE_RE = re.compile(r'"(\w+)"\s*:\s*(-?\d+(?:\.\d+)?)')
    found = {k: float(v) for k, v in _SCORE_RE.findall(raw)}
    if not any(d in found for d in QUALITY_DIMENSIONS):
        return None
    out: dict[str, Any] = {d: found.get(d, 0.0) for d in QUALITY_DIMENSIONS}
    reason = _salvage_reason(raw)
    if reason:
        out["reason"] = reason
    return out


def _salvage_reason(raw: str) -> str:
    """尝试从截断输出里取回 reason 文本（可能为空）。"""
    import re

    m = re.search(r'"reason"\s*:\s*"([^"]{0,200})', raw)
    return m.group(1) if m else ""


def aggregate(rows: list[dict]) -> dict:
    """按维度求均值（0-5），并归一化为 0-1；幻觉任务单列通过率。"""
    scored = [r for r in rows if r.get("scores")]
    per_dim = {
        dim: round(sum(float(r["scores"][dim]) for r in scored) / len(scored), 3)
        for dim in QUALITY_DIMENSIONS
    } if scored else {dim: 0.0 for dim in QUALITY_DIMENSIONS}
    normalized = {k: round(v / MAX_SCORE_PER_DIM, 3) for k, v in per_dim.items()}
    return {
        "per_dimension_0_5": per_dim,
        "per_dimension_0_1": normalized,
        "overall_0_1": round(sum(normalized.values()) / len(normalized), 3) if scored else 0.0,
        "tasks_scored": len(scored),
        "tasks_total": len(rows),
        "hallucination_free_tasks": sum(1 for r in scored if not r["scores"]["hallucinated"]),
        # 评审输出被截断、分数靠打捞得到的任务数：>0 说明该轮评审判定不可全信。
        "judge_truncated_tasks": sum(1 for r in scored if r["scores"].get("judge_truncated")),
    }


async def run_quality_baseline(cfg, *, tasks: list[dict], judge_llm,
                               limit: int | None = None,
                               project_id: str | None = None,
                               resume: dict | None = None,
                               fresh_ids: set[str] | None = None) -> dict:
    """跑完整基线：逐任务执行 → 评审 → 汇总。返回报告 dict（调用方负责落盘）。"""
    # resume=上一轮报告 → 续跑：已出分且无错的行直接复用（含 answer/evidence/tokens），只补跑
    # 失败或缺失的任务。全量一轮 1–3M token，中途因额度/网络/单条异常中断时，没有续跑就得
    # 整轮重付——实测已发生一次（HTTP 402 让 10/12 条作废，已花的 345k token 差点白扔）。
    from agentlab.runtime.serve import _build_backend

    build_backend, _n_brain, _gateway = _build_backend(cfg)
    resume_rows = [r for r in (resume or {}).get("tasks", []) if r.get("id")]
    by_id = {t["id"]: t for t in tasks}
    # fresh_ids：显式点名的任务即使上一轮已出分也**不复用**——否则"修了某个工具后只补跑
    # 受影响的两条"做不到（续跑会把它们原样搬回来）。没有任务定义的 id 谈不上重跑。
    fresh = set(fresh_ids or set()) & set(by_id)
    done: dict[str, dict] = {}
    for prior in resume_rows:
        rid = prior.get("id")
        if rid and prior.get("scores") and not prior.get("error") and rid not in fresh:
            done[rid] = dict(prior)

    if resume_rows:
        # 续跑/定点补跑时，**输出集合与顺序以上一轮报告为准**：被 --only 排除掉的任务没有
        # 任务定义、也无需重跑，但要原样留在报告里——否则 12 条成果会被缩成 2 条。
        order = [r["id"] for r in resume_rows]
        if limit:
            order = order[: limit]
    else:
        order = [t["id"] for t in tasks[: limit or len(tasks)]]

    rows: list[dict] = []
    for tid in order:
        if tid in done:
            reused = done[tid]
            reused["resumed"] = True
            rows.append(reused)
            continue
        task = by_id.get(tid)
        if task is None:
            # 上一轮有、本轮没有任务定义：保留分数，标明未重跑（不伪造为 resumed）
            prior = next((r for r in resume_rows if r["id"] == tid), {})
            prior = dict(prior)
            prior["resumed"] = True
            prior.setdefault("no_task_definition", True)
            rows.append(prior)
            continue
        row: dict[str, Any] = {"id": tid, "kind": task.get("kind", ""), "task": task["task"]}
        try:
            run = await run_task(cfg, build_backend, task, project_id=project_id)
            row.update(run)
            row["scores"] = await judge_answer(judge_llm, task, run["answer"], run.get("evidence", ""))
        except Exception as e:  # noqa: BLE001 —— 单条失败不能毁掉整轮基线
            row["error"] = f"{type(e).__name__}: {e}"
            row["scores"] = None
        rows.append(row)

    total_tokens = sum(int(r.get("tokens") or 0) for r in rows)
    # 评审调用的开销必须单列：`cloud_tokens_reported` 一直只统计 agent 侧，全量一轮的
    # 真实花费比它大（证据提到 24000 字后，评审每条输入 ~8–12k token）。
    judge_in = sum(int((r.get("scores") or {}).get("judge_meta", {}).get("input_tokens") or 0)
                   for r in rows)
    judge_out = sum(int((r.get("scores") or {}).get("judge_meta", {}).get("output_tokens") or 0)
                    for r in rows)
    judge_model = judge_model_name(cfg)
    # 记录两侧模型：agent 侧走 routes 路由（同一问题可能落到不同档），评审侧固定。
    # 不记录清楚的话，报告里的"基线"无法复现也无法解释。
    routes = dict(getattr(cfg, "routes", None) or {})
    return {
        "schema": SCHEMA,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent_model": getattr(getattr(cfg, "llm", None), "model", ""),
        "agent_model_routes": routes,
        "judge_model": judge_model,
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "tool_surface": sorted(EVAL_TOOL_ALLOWLIST),
        "resumed_tasks": sum(1 for r in rows if r.get("resumed")),
        "resumed_from": (resume or {}).get("timestamp"),
        "cloud_tokens_reported": total_tokens,
        "judge_tokens_reported": judge_in + judge_out,
        "cloud_tokens_total": total_tokens + judge_in + judge_out,
        "tasks": rows,
        "summary": aggregate(rows),
    }


def write_report(report: dict, path: str | pathlib.Path) -> pathlib.Path:
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def dry_run_report(tasks: list[dict]) -> dict:
    """不调模型的预检：确认任务集可解析、rubric 占位可渲染、只读工具过滤器可用。

    用途：改完任务集先 `--dry-run` 验证，避免花完 token 才发现某行 JSON 写坏。
    """
    rendered = []
    for t in tasks:
        prompt = load_prompt(
            "eval-quality-judge-user",
            task=str(t.get("task", "")),
            evidence=str(t.get("evidence", "") or "（无外部依据）"),
            answer="（dry-run 占位答案）",
        )
        missing = [p for p in ("{{", "}}") if p in prompt]
        rendered.append({"id": t["id"], "prompt_chars": len(prompt), "unrendered_placeholder": missing})
    return {
        "schema": SCHEMA, "dry_run": True, "task_count": len(tasks),
        "dimensions": list(QUALITY_DIMENSIONS),
        "tasks": rendered,
    }


def run_sync(coro):
    """同步入口（CLI 用），集中在这里便于测试替换。"""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 判词对证（纯本地，不调模型）：把评审判成"编造"的条目拿去 Vault 里找出处。
# ---------------------------------------------------------------------------

_CLAIM_TOKEN_RE = None


def _claim_tokens(claim: str) -> list[str]:
    """从判为编造的条目里抽出"可对证"的字面量（十六进制串/数字/路径/长标识符）。"""
    global _CLAIM_TOKEN_RE
    if _CLAIM_TOKEN_RE is None:
        import re
        _CLAIM_TOKEN_RE = re.compile(
            r"[0-9a-f]{7,}"                      # commit / hash
            r"|OPT-\d+"                          # 变更编号
            r"|\d{2,}(?:[.,]\d+)?"               # 两位数以上（单数字太泛，配对必假阳性）
            # 只用 ASCII 取"路径/标识符"：中文若并入这个字符类，'Hermes路径~/AppData/...'
            # 会粘成一个整串 token，反而匹配不到 Vault 里真正的 '~/AppData/Local/hermes'。
            r"|[A-Za-z0-9_./~\-]{6,}"
        )
    return list(dict.fromkeys(t for t in _CLAIM_TOKEN_RE.findall(claim) if t.strip()))


def _vault_texts(vault_root: str | pathlib.Path):
    """遍历 Vault 的 .md（跳过隐藏与系统目录），产出 (相对路径, 正文)。"""
    import os

    skip = {".obsidian", ".trash", ".agent-brain", ".git", "node_modules",
            "media-lib", ".dashboard-backup"}
    for dirpath, dirnames, filenames in os.walk(str(vault_root)):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for name in filenames:
            if not name.lower().endswith(".md"):
                continue
            full = os.path.join(dirpath, name)
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    yield os.path.relpath(full, str(vault_root)), f.read()
            except OSError:
                continue


def audit_claims(report: dict, vault_root: str | pathlib.Path) -> dict:
    """把报告里 `hallucinated` 的条目逐条拿去 Vault 对证，量化"评审假阳性"。

    **为什么需要**（OPT-198 全量基线暴露）：给评审的证据块是节选（单工具 400 字，
    头尾各留），长文档正文进不去。评审看不到出处时，会把"引用了真实来源"的结论判成
    `hallucinated`——rubric 里写了"查不到用 unverifiable"，实测挡不住。三条已核实的误判：
    `OPT-105三期、commit 6420703、91文件/86秒`（在 `Agent应用/简历拷打/obsidian+agent/RAG向量增删.md`）、
    `~/AppData/Local/hermes`（在 `02-DB/回顾/2026-08-24-周报.md`）、
    `~/.hermes/skills/obsidian-km/`（在 `AGENTS.md §5`）。

    判定是**启发式**，不是裁决：抽出的字面量在同一文件里出现 ≥2 个不同 token，
    或单个长度 ≥12 的字面量逐字命中，则记为 `found`，并给出命中文件供人工复核。
    它是**假阳性的下界**：只存在于"工具现场生成结果"里的数字（如 `vault_health` 的
    `broken_links_count=135` / `unlinked_count=68`）不在 Vault 文本里，本审计查不到，
    只能靠人工对证补上。
    结论只有两种用途：① 说明 factuality 维度有多少是度量噪声；② 把真幻觉与假阳性分开。
    """
    files = list(_vault_texts(vault_root))
    rows: list[dict] = []
    for task in report.get("tasks", []):
        scores = task.get("scores") or {}
        for claim in scores.get("hallucinated") or []:
            toks = [t for t in _claim_tokens(str(claim))]
            best = {"file": None, "matched": []}
            for rel, text in files:
                hit = [t for t in toks if t in text]
                if len(hit) > len(best["matched"]):
                    best = {"file": rel, "matched": hit}
            single = [t for t in toks if len(t) >= 12]
            if len(best["matched"]) >= 2:
                verdict = "found"
            elif single and any(t in best["matched"] for t in single):
                verdict = "found"
            else:
                verdict = "not_found"
            rows.append({"task": task.get("id"), "claim": str(claim), "verdict": verdict,
                         "file": best["file"] if verdict == "found" else None,
                         "matched": best["matched"][:5]})
    found = sum(1 for r in rows if r["verdict"] == "found")
    return {
        "schema": SCHEMA, "audit": "claims-v1", "vault_root": str(vault_root),
        "source_report": report.get("timestamp"),
        "files_scanned": len(files),
        "claims_total": len(rows),
        "claims_found_in_vault": found,
        "claims_not_found": len(rows) - found,
        "claims": rows,
    }
