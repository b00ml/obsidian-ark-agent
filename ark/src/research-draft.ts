import {
  ARK_CONTRACT_VERSION, isRetrievalResult, normalizeRetrievalResult,
  type Provenance, type RetrievalResult, type RetrievalScope,
} from "./contracts";

export interface ResearchBrief {
  question: string;
  purpose?: string;
  scope?: string;
  specifiedRefs?: string[];
}

export interface ResearchDraftMeta {
  title: string;
  strategy: RetrievalResult["strategy"];
  retrievalStatus: RetrievalResult["status"];
  warnings: string[];
  refs: string[];
  scope: RetrievalScope;
  provenance: Provenance[];
}

function cleanLine(value: unknown, max = 240): string {
  return String(value ?? "").replace(/[\r\n]+/g, " ").trim().slice(0, max);
}

function safeTitle(question: string): string {
  const compact = cleanLine(question, 48).replace(/[\\/:*?"<>|#]+/g, " ").trim();
  return compact || "未命名研究";
}

/** Keep a generated report name stable and inside the configured Vault folder. */
export function researchDraftFilename(title: string, now = new Date().toISOString()): string {
  const day = now.slice(0, 10).replace(/[^0-9]/g, "") || "undated";
  const slug = cleanLine(title, 72)
    .replace(/[\\/:*?"<>|#]+/g, " ")
    .replace(/\s+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64);
  return `${day}-${slug || "研究草稿"}`;
}

/** Remove the generated YAML envelope before passing the body to writeMarkdown. */
export function stripResearchFrontmatter(markdown: string): string {
  return String(markdown || "").replace(/^\uFEFF?---\r?\n[\s\S]*?\r?\n---\r?\n?\r?\n?/, "");
}

export function researchDraftFrontmatter(
  brief: ResearchBrief,
  meta: ResearchDraftMeta,
  now: string,
): Record<string, unknown> {
  return {
    type: "research-draft",
    status: "draft",
    title: meta.title,
    created_at: now,
    retrieval_strategy: meta.strategy,
    retrieval_status: meta.retrievalStatus,
    source_count: meta.refs.length,
    retrieval_project_id: meta.scope.project_id,
    retrieval_session_id: meta.scope.session_id,
    retrieval_statuses: meta.scope.statuses,
    retrieval_include_archive: meta.scope.include_archive,
    research_purpose: cleanLine(brief.purpose || "待补充", 300),
  };
}

function uniqueRefs(values: string[]): string[] {
  return [...new Set(values.map((value) => cleanLine(value, 300)).filter(Boolean))];
}

function safeSpecifiedRef(value: string): string {
  const ref = value.trim().replace(/\\/g, "/").replace(/^\.\//, "");
  if (!ref || ref.startsWith("/") || /^[A-Za-z]:\//.test(ref)
    || ref.split("/").some((part) => part === ".." || !part)) return "";
  return ref.toLowerCase().endsWith(".md") ? ref : "";
}

export function parseResearchRefs(raw: string): string[] {
  return uniqueRefs(raw.split(/[\n,，、;；]+/).map(safeSpecifiedRef).filter(Boolean));
}

/** Read only the complete envelope emitted by a rag_retrieve tool call. */
export function extractRetrievalEnvelope(
  outputs: Array<{ name: string; output: string }>,
): RetrievalResult | null {
  for (let index = outputs.length - 1; index >= 0; index -= 1) {
    const entry = outputs[index];
    if (!entry || !/^(?:.*[.:])?rag_retrieve$/i.test(String(entry.name || ""))) continue;
    try {
      const value = JSON.parse(String(entry.output || ""));
      if (isRetrievalResult(value) && value.contract === ARK_CONTRACT_VERSION) return value;
    } catch {
      // Truncated/legacy tool output is intentionally not promoted to a report.
    }
  }
  return null;
}

/** Build a bounded, source-first research scaffold from the frozen retrieval envelope. */
export function buildResearchDraft(
  brief: ResearchBrief,
  rawResult: unknown,
  now = new Date().toISOString(),
): { markdown: string; meta: ResearchDraftMeta } {
  const result = normalizeRetrievalResult(rawResult, {
    strategy: "lexical-only",
    status: "degraded",
  });
  const specified = parseResearchRefs((brief.specifiedRefs || []).join("\n"));
  const items = result.items
    .filter((item) => item.ref && (specified.length === 0 || specified.includes(item.ref)))
    .slice(0, 12);
  const refs = uniqueRefs([...specified, ...items.map((item) => item.ref)]);
  const title = safeTitle(brief.question);
  const evidence = items.length
    ? items.map((item, index) => {
      const excerpt = cleanLine(item.content, 600) || "（来源未提供摘要）";
      return `### 证据 ${index + 1} · ${cleanLine(item.title || item.ref, 100)}\n\n- 来源：[[${item.ref}]]\n- 路由：${item.source} · score=${Number.isFinite(item.score) ? item.score.toFixed(4) : "0.0000"}\n- 摘录：> ${excerpt}`;
    }).join("\n\n")
    : "（当前没有满足范围的检索证据；请补充资料或调整范围。）";
  const warnings = uniqueRefs(result.warnings);
  const warningBlock = warnings.length
    ? `\n> 检索提示：${warnings.join("；")}\n`
    : "";
  const meta: ResearchDraftMeta = {
    title, strategy: result.strategy, retrievalStatus: result.status,
    warnings, refs, scope: result.scope, provenance: result.provenance,
  };
  const frontmatter = researchDraftFrontmatter(brief, meta, now);
  const scopeLine = `- Project scope：${cleanLine(result.scope.project_id || "（全局）", 120)}\n`
    + `- Session scope：${cleanLine(result.scope.session_id || "（无）", 120)}\n`
    + `- 状态过滤：${result.scope.statuses.length ? result.scope.statuses.join("、") : "默认"}\n`
    + `- 包含归档：${result.scope.include_archive ? "是" : "否"}\n`;
  const provenanceLine = result.provenance.length
    ? result.provenance.slice(0, 24).map((item) => `- ${cleanLine(item.source, 80)}：${cleanLine(item.ref, 240)}`).join("\n")
    : "- （暂无 provenance）";
  const yaml = Object.entries(frontmatter)
    .map(([key, value]) => `${key}: ${Array.isArray(value) ? `[${value.map((item) => `\"${String(item)}\"`).join(", ")}]` : String(value)}`)
    .join("\n");
  const markdown = `---\n${yaml}\n---\n\n# ${title}\n\n## 研究简报\n\n- 问题：${cleanLine(brief.question, 500)}\n- 用途：${cleanLine(brief.purpose || "待补充", 300)}\n- 范围：${cleanLine(brief.scope || "按当前 Project scope", 300)}\n${warningBlock}\n## 检索元数据\n\n- 策略：${result.strategy}\n- 状态：${result.status}\n${scopeLine}- 警告：${warnings.length ? warnings.join("；") : "无"}\n\n### Provenance\n\n${provenanceLine}\n\n## 证据卡\n\n${evidence}\n\n## 证据矩阵\n\n| 子问题 | 支持证据 | 反对证据 | 未知项 |\n| --- | --- | --- | --- |\n| 待拆分 | 待填写 | 待填写 | 待填写 |\n\n## 结论草稿\n\n> 这是研究草稿。结论必须由用户或 Agent 基于上面的证据补充，不能把检索分数当作事实可信度。\n\n- 结论：待填写\n- 主要论据：待填写\n- 反方观点：待填写\n- 局限与未知：待填写\n\n## 来源清单\n\n${refs.length ? refs.map((ref) => `- [[${ref}]]`).join("\n") : "- （暂无来源）"}\n`;
  return {
    markdown,
    meta,
  };
}

export function buildResearchPrompt(brief: ResearchBrief): string {
  const refs = parseResearchRefs((brief.specifiedRefs || []).join("\n"));
  return [
    "进入快速研究草稿模式。只读检索，不要调用写入工具，不要声称已经发布正式成果。",
    `研究问题：${cleanLine(brief.question, 500)}`,
    `用途：${cleanLine(brief.purpose || "待补充", 300)}`,
    `范围：${cleanLine(brief.scope || "当前 Project scope", 300)}`,
    refs.length ? `用户指定资料：${refs.map((ref) => `[[${ref}]]`).join("、")}` : "用户未指定资料，请先用 rag_retrieve(envelope=true) 获取证据。",
    "检索要求：使用 rag_retrieve(envelope=true)，保留 strategy/status/warnings/scope/provenance；用户指定资料优先，不能把分数当作事实可信度。",
    "输出：研究简报、证据卡、支持/反对/未知矩阵、带引用的结论草稿；检索降级或证据不足时明确标记，不要补写不存在的事实。",
  ].join("\n");
}
