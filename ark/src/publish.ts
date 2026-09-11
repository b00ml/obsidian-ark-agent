// M4 产出体系起点：主题简报/日报（§3.4）——把当日采集简报 + M3 要点卡聚合为一页可扫读"信息日报"。
// 纯聚合、无 AI：遍历 feedArchiveFolder 今日 `{today}-*.md` 简报 → 逐条抽取"结论/要点/原文/标签"，
// 并内链到当日要点卡（若有），落 `feedArchiveFolder/{today}-信息日报.md`（覆盖当天）。
import { TFile } from "obsidian";
import type ArkOSPlugin from "./main";
import { getToday, sanitizeFilename } from "./utils";

export interface DailyBriefResult {
  sources: number;   // 参与的订阅源数
  items: number;     // 汇总的简报条目数
  path: string;      // 生成/覆盖的日报路径
}

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

/** 解析简报块内部的字段 */
function parseBlock(block: string) {
  const conclusion = /^>\s*一句话结论：\s*(.+)$/m.exec(block)?.[1]?.trim() ?? "";
  const url = /^🔗\s*原文：\s*(\S+)/m.exec(block)?.[1]?.trim() ?? "";
  const tags = /^标签：\s*(.+)$/m.exec(block)?.[1]?.trim() ?? "";
  const points = (block.match(/^- (.+)$/gm) || [])
    .map((l) => l.replace(/^- /, "").trim())
    .filter((p) => p && !p.startsWith("🔗 ") && !p.startsWith("标签："))
    .slice(0, 6);
  return { conclusion, url, tags, points };
}

/** 从简报全文切分 `## 标题` 简报块 */
function splitBlocks(body: string): { title: string; block: string }[] {
  const out: { title: string; block: string }[] = [];
  const re = /^## (.+)\n+([\s\S]*?)(?=\n## |$)/gm;
  let m: RegExpExecArray | null;
  while ((m = re.exec(body))) out.push({ title: m[1].trim(), block: (m[2] || "").trim() });
  return out;
}

/** 生成当日信息日报 */
export async function generateDailyBrief(plugin: ArkOSPlugin): Promise<DailyBriefResult> {
  const s = plugin.data.settings;
  const briefFolder = s.feedArchiveFolder || "02-DB/简报";
  const cardFolder = s.pipelineFolder || "02-DB/要点";
  const today = getToday();
  const destPath = `${briefFolder}/${today}-信息日报.md`;

  const files = plugin.app.vault.getFiles().filter((f: TFile) =>
    f.extension === "md" && f.path.startsWith(briefFolder + "/") && f.basename.startsWith(today + "-"));

  let items = 0;
  const sections: string[] = [];
  const feedOrder: string[] = [];

  for (const f of files) {
    if (f.basename.endsWith("信息日报")) continue; // 跳过日报自身
    const text = await plugin.app.vault.read(f);
    const fmRe = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(text);
    const fm = fmRe ? fmRe[1] : "";
    const fmKey = (k: string) => new RegExp(`(?:^|\\n)\\s*${k}:\\s*([^\\n]*)`).exec("\n" + fm)?.[1]?.trim();
    if (!/feed-brief/.test(fmKey("type") || "")) continue;
    const feed = fmKey("feed") || f.basename.replace(/^\d{4}-\d{2}-\d{2}-/, "").replace(/\.md$/, "");
    const body = fmRe ? text.slice(fmRe[0].length) : text;

    const blocks = splitBlocks(body);
    if (blocks.length === 0) continue;

    feedOrder.push(feed);
    const lines: string[] = [`### ${feed}`];
    for (const { title, block } of blocks) {
      const { conclusion, url, tags, points } = parseBlock(block);
      const cardName = `${today}-${sanitizeFilename(title)}`;
      const cardPath = `${cardFolder}/${cardName}.md`;
      const hasCard = plugin.app.vault.getAbstractFileByPath(cardPath) instanceof TFile;
      lines.push("");
      lines.push(`**${title}**`);
      if (conclusion) lines.push(`> ${conclusion}`);
      points.forEach((p) => lines.push(`- ${p}`));
      const refs: string[] = [];
      if (url) refs.push(`🔗 [原文](${url})`);
      if (tags) refs.push(tags);
      if (hasCard) refs.push(`[[${cardName}]]`);
      if (refs.length) lines.push(refs.join("　·　"));
      items++;
    }
    sections.push(lines.join("\n"));
  }

  await ensureFolder(plugin, briefFolder);
  const md = [
    "---",
    "type: daily-brief",
    `date: ${today}`,
    `sources: ${feedOrder.length}`,
    `items: ${items}`,
    `created_at: ${new Date().toISOString()}`,
    "---",
    "",
    `# 📰 信息日报 · ${today}`,
    "",
    items === 0 ? "今日暂无可扫读简报。" : sections.join("\n\n---\n\n") + "\n",
  ].join("\n");

  const existing = plugin.app.vault.getAbstractFileByPath(destPath);
  if (existing instanceof TFile) await plugin.app.vault.modify(existing, md);
  else await plugin.app.vault.create(destPath, md);

  return { sources: feedOrder.length, items, path: destPath };
}