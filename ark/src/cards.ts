// 卡片统一 CRUD：每张 UI 卡片 = 一个带 Frontmatter 的 .md 文件（1:1 双向绑定）
// 增删改一律落到 .md，由 sync 的 watcher/import 把它镜像进插件 data。
import { TFile } from "obsidian";
import type ArkOSPlugin from "./main";
import { writeMarkdown } from "./sync";
import { getToday } from "./utils";

export type CardKind = "log" | "idea" | "drawing" | "health" | "database";

export interface CardIn {
  fm: Record<string, unknown>;
  body?: string;
}

export function folderFor(plugin: ArkOSPlugin, kind: CardKind, fm?: Record<string, unknown>): string {
  const s = plugin.data.settings;
  if (kind === "log") {
    const t = String(fm?.type ?? fm?.log_type ?? "");
    return t === "fault" || t === "fault_log" ? s.faultLogFolder : s.normalLogFolder;
  }
  if (kind === "idea") return s.ideaFolder;
  if (kind === "drawing") return s.drawingFolder;
  if (kind === "health") return s.healthFolder;
  return s.databaseFolder;
}

/** 文件名安全化（导出以便单测：与 OPT-191 同类的路径规则，出过错的地方）。 */
export function safeName(kind: CardKind, fm?: Record<string, unknown>): string {
  const raw = String(fm?.title ?? fm?.db_name ?? kind);
  return raw.replace(/[\\/:*?"<>|]/g, "-").slice(0, 24);
}

/** 新增卡片 → 生成唯一 .md 文件（并导入 data） */
export async function createCard(
  plugin: ArkOSPlugin,
  kind: CardKind,
  inp: CardIn,
  opts?: { filename?: string },
): Promise<string> {
  const id = String(inp.fm.id ?? (Date.now().toString(36) + Math.random().toString(36).slice(2, 6)));
  const fm: Record<string, unknown> = { ...inp.fm, id };
  if (!fm.created_at) fm.created_at = new Date().toISOString();
  const folder = folderFor(plugin, kind, fm);
  const filename = opts?.filename ?? `${getToday()}-${safeName(kind, fm)}-${id}`;
  return writeMarkdown(plugin, folder, filename, fm, inp.body ?? "");
}

/** 更新既有卡片 → 重写同一 .md（若不存在则返回 null） */
export async function updateCard(plugin: ArkOSPlugin, path: string | undefined, inp: CardIn): Promise<string | null> {
  if (!path) return null;
  const dir = path.split("/").slice(0, -1).join("/");
  const base = path.split("/").pop()!.replace(/\.md$/, "");
  const fm = { ...inp.fm, created_at: inp.fm.created_at ?? new Date().toISOString() };
  return writeMarkdown(plugin, dir || ".", base, fm, inp.body ?? "");
}

/** 删除卡片 → 删除对应 .md（watcher 自动从 data 移除） */
export async function deleteCardAt(plugin: ArkOSPlugin, path: string | undefined): Promise<boolean> {
  if (!path) return false;
  const f = plugin.app.vault.getAbstractFileByPath(path);
  if (f instanceof TFile) {
    await plugin.app.vault.delete(f);
    return true;
  }
  // 文件不存在 → 直接从 data 移除回退
  removeFromData(plugin, path);
  await plugin.savePluginData();
  return false;
}

function removeFromData(plugin: ArkOSPlugin, path: string) {
  for (const arr of [plugin.data.logs, plugin.data.ideas, plugin.data.drawings, plugin.data.healthRecords, plugin.data.database] as any[][]) {
    const i = arr.findIndex((x) => x.notePath === path || x.filePath === path);
    if (i >= 0) arr.splice(i, 1);
  }
}
