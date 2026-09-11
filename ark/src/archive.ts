// 归档门控 + 一键归档（知识信息闭环 M2 二期 / §8-D4、P2-9）
// 采集简报先落 feedInboxFolder（Inbox），经用户确认后归档进知识层 feedArchiveFolder（默认 02-DB/简报）。
// 动作 = 移动文件到目标目录 + 置 frontmatter status: archived。
// 质量门槛（P2-9）：归档前必须已含 date(frontmatter) + ≥1 条非占位 [[wikilink]] 关联；不满足的留在 Inbox 或由用户强制归档。
import { TFile } from "obsidian";
import type ArkOSPlugin from "./main";
import { confirmDialog, notice } from "./ui";
import { capturePatch, captureFields } from "./capture";

interface ArchiveResult {
  archived: number;
  pending: string[]; // 未满足门槛、留在 Inbox 的简报文件名
  forceArchived?: number;
}

/** 逐级创建目录（已存在跳过） */
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

/** 解析 frontmatter + 正文 */
function splitFm(text: string): { fm: string; body: string } {
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(text);
  if (!m) return { fm: "", body: text };
  return { fm: m[1], body: text.slice(m[0].length) };
}

function fmField(fm: string, key: string): string | undefined {
  return new RegExp(`(?:^|\\n)\\s*${key}:\\s*([^\\n]*)`).exec("\n" + fm)?.[1]?.trim();
}

/** 正文是否含 ≥1 条非占位 [[wikilink]]（占位用 [[待整理/…]]，不算真实关联） */
function hasRealLink(body: string): boolean {
  const re = /\[\[(?:\/?[^\]|\s]+)?([^\]|#]*)\]\]/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(body))) {
    const target = (m[1] || "").trim();
    if (!target) continue;
    if (target.startsWith("待整理")) continue;
    return true;
  }
  return false;
}

/** 就地补/置 frontmatter 的若干键（不存在则整段补 frontmatter），返回新文本 */
function withFm(plugin: ArkOSPlugin, text: string, set: Record<string, string>): string {
  let { fm, body } = splitFm(text);
  if (fm === "") {
    const block = Object.entries(set).map(([k, v]) => `${k}: ${v}`).join("\n");
    return `---\n${block}\n---\n\n${body}`;
  }
  const lines = fm.split("\n");
  const keyRe = (k: string) => new RegExp(`^\\s*${k}:`);
  for (const [k, v] of Object.entries(set)) {
    const i = lines.findIndex((ln) => keyRe(k).test(ln));
    if (i >= 0) lines[i] = `${k}: ${v}`;
    else lines.push(`${k}: ${v}`);
  }
  return `---\n${lines.join("\n")}\n---\n\n${body}`;
}

/**
 * 一键归档：把 Inbox 简报按批搬到知识层目录并置 archived。
 * 门槛校验 + 二次确认（未满足门槛的简报由用户二选一：全部强制 / 仅归档已补齐的）。
 */
export async function archiveFeedBriefs(plugin: ArkOSPlugin): Promise<ArchiveResult> {
  const s = plugin.data.settings;
  const srcFolder = s.feedInboxFolder || "Inbox";
  const destFolder = s.feedArchiveFolder || "02-DB/简报";

  const candidates = plugin.app.vault.getFiles()
    .filter((f: TFile) => f.extension === "md" && f.path.startsWith(srcFolder + "/"));

  const ready: TFile[] = [];
  const pending: TFile[] = [];
  for (const f of candidates) {
    const text = await plugin.app.vault.read(f);
    const { fm, body } = splitFm(text);
    const isFeedBrief = /feed-brief/.test(fmField(fm, "type") || "");
    if (!isFeedBrief) continue; // 只归档采集简报，学习问答走 quizFolder
    const status = fmField(fm, "status");
    if (status === "archived" || captureFields({
      type: fmField(fm, "type"),
      status,
      state: fmField(fm, "state"),
      processed: fmField(fm, "processed"),
    }).state === "archived") continue;
    // 门槛：frontmatter 含 date + 正文有真实 wikilink
    if (fmField(fm, "date") && hasRealLink(body)) ready.push(f);
    else pending.push(f);
  }

  if (ready.length === 0 && pending.length === 0) {
    notice("Inbox 暂无待归档简报");
    return { archived: 0, pending: [] };
  }

  const pendingNameList = pending.map((f) => f.basename);
  const summary = pending.length
    ? `发现简报：已补齐关联 ${ready.length} 篇，缺关联 ${pending.length} 篇（${pendingNameList.join("、")}）。\n确定=连同缺关联的一并归档，取消=仅归档已补齐的。`
    : `将归档 ${ready.length} 篇简报到 ${destFolder}，确定？`;
  const proceedAll = await confirmDialog(plugin, summary);

  const toMove = proceedAll ? [...ready, ...pending] : ready;
  if (toMove.length === 0) {
    notice("没有满足归档门槛的简报（需补 date 和 wikilink）");
    return { archived: 0, pending: pendingNameList };
  }

  await ensureFolder(plugin, destFolder);
  let count = 0;
  for (const f of toMove) {
    const text = await plugin.app.vault.read(f);
    const next = withFm(plugin, text, {
      status: "archived", ...capturePatch("archived"), archived_at: new Date().toISOString(),
    });
    if (next !== text) await plugin.app.vault.modify(f, next); // 就地置 archived，rename 事件仍会触发导入
    await plugin.app.vault.rename(f, `${destFolder}/${f.basename}.md`); // 移动（触发 renameAt 联动）
    count++;
  }
  await plugin.savePluginData();
  notice(`已归档 ${count} 篇 → ${destFolder}${pending.length && proceedAll ? `（含 ${pending.length} 篇缺关联）` : ""}${!proceedAll && pending.length ? `，${pending.length} 篇缺关联留在 Inbox` : ""}`);
  return { archived: count, pending: pendingNameList, forceArchived: proceedAll && pending.length ? pending.length : undefined };
}
