// M3 统一处理管线（知识信息闭环 §3.2 / P1-4 定稿：规则层确定 + AI 建议层不动路由）
// 两阶段：① 规则层（无 AI）确定性过滤(空/过短/噪声标记)；② AI 建议层仅输出
// {score, suggestedTags, action, suggestedMergeWith, keyPoints}（进程内只当建议）；
// ③ 代码按阈值决定 action=keep 才写要点卡，命中阈值以下视为忽略。
// 作用对象：已归档简报（feedArchiveFolder）中 type=feed-brief 且 status!=processed 的每条简报块，
// 产「要点卡」到 pipelineFolder，并回写原简报 frontmatter 的 tags/processed。
import { TFile } from "obsidian";
import type ArkOSPlugin from "./main";
import type { AiMessage } from "./ai";
import { runChannel, extractJson, JsonExtractError, logAiError } from "./ai-asst";
import { getToday, sanitizeFilename } from "./utils";
import processPipelineUser from "./prompts/process-pipeline-user.st";
import { capturePatch } from "./capture";

const THRESHOLD = 0.45; // score 命中阈值：AI 建议仅参考，action 是否生效由这里的阈值兜底

/** 逐级创建目录 */
async function ensureFolder(plugin: ArkOSPlugin, path: string) {
  const parts = path.split("/").filter(Boolean);
  let cur = "";
  for (const p of parts) {
    cur = cur ? `${cur}/${p}` : p;
    if (!plugin.app.vault.getAbstractFileByPath(cur)) {
      try { await plugin.app.vault.createFolder(cur); } catch { /* 并发/已存在 */ }
    }
  }
}

/** 就地补/置 frontmatter 若干键（与 archive.withFm 同构，避免跨模块拉耦合） */
function withFm(text: string, set: Record<string, string>): string {
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(text);
  if (!m) return `---\n${Object.entries(set).map(([k, v]) => `${k}: ${v}`).join("\n")}\n---\n\n${text}`;
  const lines = m[1].split("\n");
  const keyRe = (k: string) => new RegExp(`^\\s*${k}:`);
  for (const [k, v] of Object.entries(set)) {
    const i = lines.findIndex((ln) => keyRe(k).test(ln));
    if (i >= 0) lines[i] = `${k}: ${v}`;
    else lines.push(`${k}: ${v}`);
  }
  return `---\n${lines.join("\n")}\n---\n\n${text.slice(m[0].length)}`;
}

/** 从简报全文切分每条 `## 标题` 简报块 */
function splitBriefBlocks(text: string): { title: string; body: string }[] {
  const blocks: { title: string; body: string }[] = [];
  const re = /^## (.+)\n+([\s\S]*?)(?=\n## |$)/gm;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) blocks.push({ title: m[1].trim(), body: (m[2] || "").trim() });
  return blocks;
}

/**
 * 一键处理：对已归档简报逐条块跑两阶段管线，产要点卡 + 回写标签。
 * 空内容/过短等噪声在规则层直接忽略，不再消耗 AI。
 */
export async function runPipeline(plugin: ArkOSPlugin): Promise<{ scanned: number; cards: number; skipped: number; failed: number }> {
  const s = plugin.data.settings;
  const srcFolder = s.feedArchiveFolder || "02-DB/简报";
  const cardFolder = s.pipelineFolder || "02-DB/要点";
  const today = getToday();

  const files = plugin.app.vault.getFiles()
    .filter((f: TFile) => f.extension === "md" && f.path.startsWith(srcFolder + "/"));
  if (files.length === 0) return { scanned: 0, cards: 0, skipped: 0, failed: 0 };

  let cards = 0, skipped = 0, failed = 0;
  const tagAcc: string[] = []; // 累积本批所有块的 suggestedTags，统一回写各原文件

  for (const f of files) {
    const text = await plugin.app.vault.read(f);
    const fmRe = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(text);
    const fm = fmRe ? fmRe[1] : "";
    const fmKey = (k: string) => new RegExp(`(?:^|\\n)\\s*${k}:\\s*([^\\n]*)`).exec("\n" + fm)?.[1]?.trim();
    if (!/feed-brief/.test(fmKey("type") || "")) continue; // 只处理采集简报
    if (fmKey("processed") === "true") continue;             // 已处理过则跳过（幂等）
    const body = fmRe ? text.slice(fmRe[0].length) : text;

    const blocks = splitBriefBlocks(body);
    if (blocks.length === 0) { skipped++; continue; }

    const ownTags = new Set<string>();
    let fileFailed = false;
    for (const b of blocks) {
      // ----① 规则层：确定性过滤（无 AI）----
      const source = `【标题】${b.title}\n【正文】${b.body.slice(0, 800)}`;
      if (b.body.replace(/\s+/g, "").length < 30) { skipped++; continue; } // 无实质内容 → 忽略

      // ----② AI 建议层（仅建议）----
      const prompt = processPipelineUser.replace("{{source}}", source);
      let raw = "";
      let verdict: any; // AI 建议层原始返回（score/suggestedTags/suggestedMergeWith/action/keyPoints），仅作建议
      try {
        const ac = new AbortController();
        const timer = setTimeout(() => ac.abort(), 120000);
        try {
          raw = await runChannel(s, [{ role: "user", content: prompt }] as AiMessage[], "contextual", ac.signal);
          verdict = extractJson(raw);
        } finally { clearTimeout(timer); }
      } catch (e: any) {
        failed++;
        fileFailed = true;
        if (e instanceof JsonExtractError) await logAiError(plugin, "pipeline", prompt, raw || (e as any).raw || "", e);
        continue;
      }

      const score = Number(verdict?.score ?? 0);
      const tags: string[] = Array.isArray(verdict?.suggestedTags) ? verdict.suggestedTags.map(String).slice(0, 3) : [];
      const keyPoints: string[] = Array.isArray(verdict?.keyPoints) ? verdict.keyPoints.map(String).filter(Boolean).slice(0, 5) : [];
      const mergeWith = String(verdict?.suggestedMergeWith ?? "").trim();
      // ----③ 阈值决策：action 只是建议，是否生效由阈值兜底----
      const keep = verdict?.action === "keep" && score >= THRESHOLD;
      if (!keep) { skipped++; continue; }

      tags.forEach((t) => { ownTags.add(t); tagAcc.push(t); });

      // ----产出：要点卡落 pipelineFolder----
      await ensureFolder(plugin, cardFolder);
      const path = `${cardFolder}/${today}-${sanitizeFilename(b.title)}.md`;
      const lines = [
        "---",
        `type: point-card`,
        `source: [[${f.basename}]]`,
        `score: ${score}`,
        `date: ${today}`,
        tags.length ? `tags: [${tags.join(", ")}]` : "tags: []",
        "---",
        "",
        `# 要点卡 · ${b.title}`,
        "",
        "## 要点",
        "",
        ...(keyPoints.length ? keyPoints.map((p) => `- ${p}`) : ["- （无）"]),
        "",
        `> 来源：[[${f.basename}]] · 处理于 ${today}`,
      ];
      if (mergeWith) lines.push(`> 归并建议：[[${mergeWith}]]`);
      const md = lines.join("\n") + "\n";
      const existing = plugin.app.vault.getAbstractFileByPath(path);
      if (existing instanceof TFile) await plugin.app.vault.modify(existing, md);
      else await plugin.app.vault.create(path, md);
      cards++;
    }

    // ----回写原简报 frontmatter：tags + processed（幂等）----
    const retryCount = Math.max(0, Number(fmKey("retry_count") || 0) || 0);
    const newText = withFm(text, fileFailed ? {
      ...capturePatch("failed", { error_code: "PIPELINE_AI_FAILED" }),
      retry_count: String(retryCount + 1),
      last_failed_at: new Date().toISOString(),
    } : {
      ...capturePatch("linked"),
      processed: "true",
      processed_at: new Date().toISOString(),
      tags: `[${[...ownTags].join(", ")}]`,
    });
    if (newText !== text) await plugin.app.vault.modify(f, newText);
  }

  await plugin.savePluginData();
  return { scanned: files.length, cards, skipped, failed };
}
