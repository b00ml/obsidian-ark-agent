// M4 产出体系：案例库/决策备忘（§3.4）——把要点卡蒸馏成可复用的"案例条目"，沉淀进案例库。
// 输入：pipelineFolder 下 type=point-card 的要点卡；输出：AI 提炼为一组案例条目，
// 按「首标签」聚合——每个标签一个文件 `caseFolder/{首标签}-案例库.md`，新增案例追加到末尾
// （呼应"少而聚合"的文件组织偏好）。重复/浅薄条目由 AI 直接丢弃。
import { TFile } from "obsidian";
import type ArkOSPlugin from "./main";
import type { AiMessage } from "./ai";
import { runChannel, extractJson, JsonExtractError, logAiError } from "./ai-asst";
import { sanitizeFilename } from "./utils";
import caseStudyUser from "./prompts/case-study-user.st";

export interface CaseLibResult {
  cards: number;        // 扫描到的要点卡数
  cases: number;        // AI 提炼出的案例总数
  groups: number;       // 命中 的标签分组合数（含空tag）
  path: string | null;  // 若聚合到案例，返回首个分组文件路径；否则 null
}

interface CaseEntry {
  title: string;
  scenario: string;
  conclusion: string;
  action: string;
  tags: string[];
  source: string;
}

/** 逐级创建目录（与 moc/pipeline 同构，避免跨模块耦合） */
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

/** 读取一条要点卡：标题 + 要点行 + 来源，供 AI 蒸馏 */
function readPointCard(text: string): { title: string; points: string[]; source: string } {
  const fmRe = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(text);
  const fm = fmRe ? fmRe[1] : "";
  const title = /^#\s*(.+)$/m.exec(text)?.[1]?.trim() ?? "";
  const src = fm.match(/(?:^|\n)\s*source:\s*\[\[(.+?)\]\]/)?.[1] ?? "";
  const points = (text.match(/^-\s*(.+)$/gm) || []).map((l) => l.replace(/^- /, "").trim()).filter(Boolean).slice(0, 6);
  return { title, points, source: src.replace(/\.md$/, "") };
}

/**
 * 一键沉淀：扫描要点卡 → AI 蒸馏成案例条目 → 按首标签聚合落 caseFolder。
 * 无要点卡时不调用 AI（避免空跑）。
 */
export async function runCaseLib(plugin: ArkOSPlugin): Promise<CaseLibResult> {
  const s = plugin.data.settings;
  const cardFolder = s.pipelineFolder || "02-DB/要点";
  const caseFolder = s.caseFolder || "02-DB/案例";

  const files = plugin.app.vault.getFiles().filter((f: TFile) =>
    f.extension === "md" && f.path.startsWith(cardFolder + "/"));
  if (files.length === 0) return { cards: 0, cases: 0, groups: 0, path: null };

  // 组装待蒸馏的要点卡内容
  const cards: { file: TFile; title: string; points: string[]; source: string; isCard: boolean }[] = [];
  for (const f of files) {
    const text = await plugin.app.vault.read(f);
    const fmRe = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(text);
    const fm = fmRe ? fmRe[1] : "";
    if (!/point-card/.test((fm.match(/(?:^|\n)\s*type:\s*([^\n]*)/)?.[1] ?? "").trim())) continue;
    const { title, points, source } = readPointCard(text);
    cards.push({ file: f, title, points, source, isCard: true });
  }
  if (cards.length === 0) return { cards: 0, cases: 0, groups: 0, path: null };

  const body = cards.map((c, i) =>
    `【卡${i + 1}｜${c.title || c.file.basename}】\n${(c.points.join("\n") || "（无要点）")}\n【来源】${c.source || c.file.basename}`
  ).join("\n\n");

  // AI 蒸馏
  const prompt = caseStudyUser.replace("{{cases}}", body);
  let raw = "";
  let entries: CaseEntry[] = [];
  try {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), 120000);
    try {
      raw = await runChannel(s, [{ role: "user", content: prompt }] as AiMessage[], "contextual", ac.signal);
      entries = extractJson(raw);
    } finally { clearTimeout(timer); }
  } catch (e: any) {
    if (e instanceof JsonExtractError) await logAiError(plugin, "caselib", prompt, raw || (e as any).raw || "", e);
    throw new Error("案例提炼失败: " + String(e?.message ?? e));
  }
  if (!Array.isArray(entries)) throw new Error("案例提炼失败：AI 未返回数组");

  // 按首标签聚合（无标签归入"未分类"）
  const clean = entries
    .filter((c) => c && typeof c.title === "string" && c.title.trim())
    .map((c) => ({
      title: c.title.trim(),
      scenario: (c.scenario || "").trim(),
      conclusion: (c.conclusion || "").trim(),
      action: (c.action || "").trim(),
      tags: Array.isArray(c.tags) ? c.tags.map(String).map((t) => t.trim()).filter(Boolean) : [],
      source: (c.source || "").trim(),
    }));

  const groups = new Map<string, CaseEntry[]>();
  for (const c of clean) {
    const key = c.tags[0] || "未分类";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(c);
  }

  // 每个标签一个文件，追加
  await ensureFolder(plugin, caseFolder);
  let firstPath: string | null = null;
  for (const [tag, list] of groups) {
    const tagPath = `${caseFolder}/${sanitizeFilename(tag)}-案例库.md`;
    if (!firstPath) firstPath = tagPath;
    let existingText = "";
    const existing = plugin.app.vault.getAbstractFileByPath(tagPath);
    let hasFm = false;
    if (existing instanceof TFile) {
      existingText = await plugin.app.vault.read(existing);
      hasFm = /^---\r?\n/.test(existingText);
    }
    const blocks = list.map((c) => {
      const lines = [`## ${c.title}`, ""];
      if (c.scenario) lines.push(`> 适用场景：${c.scenario}`);
      if (c.conclusion) lines.push(`> 结论：${c.conclusion}`);
      if (c.action) lines.push(`> 可复用动作：${c.action}`);
      const refs: string[] = [];
      if (c.source) refs.push(`[[${c.source}]]`);
      if (c.tags.length) refs.push(c.tags.map((t) => `#${t}`).join(" "));
      if (refs.length) lines.push(refs.join("　·　"));
      return lines.join("\n");
    });
    const newMd = blocks.join("\n\n") + "\n";
    if (!hasFm) {
      existingText = [
        "---",
        `type: case-library`,
        `tag: ${tag}`,
        `updated_at: ${new Date().toISOString()}`,
        "---",
        "",
        `# 案例库 · ${tag}`,
        "",
        existingText.trim(),
      ].join("\n") + "\n";
    }
    const finalText = hasFm
      ? existingText + (existingText.endsWith("\n") ? "" : "\n") + newMd
      : existingText + "\n" + newMd;
    if (existing instanceof TFile) await plugin.app.vault.modify(existing, finalText);
    else await plugin.app.vault.create(tagPath, finalText);
  }

  await plugin.savePluginData();
  return { cards: cards.length, cases: clean.length, groups: groups.size, path: firstPath };
}