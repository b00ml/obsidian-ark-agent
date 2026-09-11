// 数据层闭环：Vault .md ↔ 插件 data 的双向同步
// 方向：把 Vault 内对应目录下带 frontmatter 的笔记增量导入插件 data，并监听文件事件实时增删改。
import { parseYaml, TFile } from "obsidian";
import type ArkOSPlugin from "./main";
import { generateId, getToday } from "./utils";
import { parseRecurrence } from "./recurrence";
import { ensureDashboardFile } from "./dashboard";

function str(v: any, d = ""): string {
  if (v == null) return d;
  if (Array.isArray(v)) return v.map((x) => String(x)).join(", ");
  return String(v);
}
function num(v: any, d = 0): number {
  const n = Number(v);
  return isNaN(n) ? d : n;
}

/**
 * frontmatter 时间戳 → 毫秒（容错解析）。
 *
 * 必须同时容忍三种落盘形态：数字（旧数据/JSON 序列化）、`Date`（YAML 1.1 时间戳）、
 * ISO 字符串（当前写入格式，见 `taskToFm`/`createCard`/`updateCard`）。
 * 此前只用 `Number(v)`：遇到 ISO 字符串得 `NaN` → 静默回落 `Date.now()`，于是
 * `created_at`/`updated_at` 每次导入都被刷成"导入时刻"（真机验收报出：每次拖动都变一次）。
 * 这类"看起来能跑、其实一直在丢数据"的降级必须收窄到只对真正的脏值生效。
 */
function ts(v: any, d = Date.now()): number {
  if (v instanceof Date) return isNaN(v.getTime()) ? d : v.getTime();
  if (typeof v === "number") return Number.isFinite(v) ? v : d;
  if (typeof v === "string") {
    const s = v.trim();
    if (!s) return d;
    const n = Number(s);
    if (Number.isFinite(n)) return n;
    const parsed = Date.parse(s);
    if (!Number.isNaN(parsed)) return parsed;
  }
  return d;
}
function arr(v: any): string[] {
  if (v == null) return [];
  const list = Array.isArray(v) ? v : [v];
  return list.map((x) => String(x).trim()).filter(Boolean);
}

/** 读取文件 frontmatter + 正文 */
async function parseMd(plugin: ArkOSPlugin, file: TFile): Promise<{ fm: any; body: string }> {
  const text = await plugin.app.vault.read(file);
  const m = text.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  let fm: any = {};
  if (m) {
    try { fm = parseYaml(m[1]) ?? {}; } catch { /* 非法 frontmatter 忽略 */ }
  }
  const body = m ? text.slice(m[0].length) : text;
  return { fm, body };
}

/** 递归收集某目录下所有 .md */
function filesIn(plugin: ArkOSPlugin, folderPath: string): TFile[] {
  const node: any = plugin.app.vault.getAbstractFileByPath(folderPath);
  if (!node || !node.children) return [];
  const out: TFile[] = [];
  const walk = (n: any) => {
    if (n.children) n.children.forEach(walk);
    else if (n.extension === "md") out.push(n as TFile);
  };
  walk(node);
  return out;
}

/** 逐级创建目录（已存在则跳过） */
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

// ---------- 各实体解析（vault → data） ----------

interface LogLike {
  id: string; type: "normal" | "fault"; title: string; content: string;
  createdAt: number; tags: string[]; notePath?: string;
}
interface IdeaLike { id: string; title: string; content: string; tags: string[]; archived: boolean; createdAt: number; updatedAt: number; notePath?: string }
interface DbLike {
  id: string; name: string; categoryId: string; status: "pending" | "solved" | "other";
  tags: string[]; source?: string; sourceRef?: string; description?: string; filePath?: string; createdAt: number; updatedAt: number;
}
interface HealthLike { id: string; date: string; water: number; sleep: number; steps: number; weight: number; createdAt: number; notePath?: string }
interface DrawingLike { id: string; title: string; description?: string; color: string; createdAt: number; updatedAt: number; notePath?: string }

function normLogType(kind: string | undefined): "normal" | "fault" {
  return kind === "fault" || kind === "fault_log" ? "fault" : "normal";
}
function normStatus(s: string | undefined): "pending" | "solved" | "other" {
  if (s === "solved") return "solved";
  if (s === "other") return "other";
  return "pending"; // researching→pending，其余归 pending
}
function basename(path: string): string {
  return path.split("/").pop()!.replace(/\.md$/, "");
}

/** 解析任务 frontmatter → ArkTask（自动定位/创建所属清单） */
export function taskFromFm(plugin: ArkOSPlugin, path: string, fm: any): any {
  const s = plugin.data.settings;
  const listId = str(fm.list_id);
  let list = plugin.data.todoLists.find((l) => l.id === listId);
  if (!list) {
    const rel = path.replace(s.todoFolder, "").replace(/^\/?/, "");
    const listName = (fm.list_name ? str(fm.list_name) : rel.split("/").filter(Boolean)[0]) || "未命名清单";
    list = plugin.data.todoLists.find((l) => l.name === listName);
    if (!list) { list = { id: generateId(), name: listName, tasks: [] }; plugin.data.todoLists.push(list); }
  }
  const statusRaw = str(fm.status);
  const status = statusRaw === "doing" ? "doing" : statusRaw === "blocked" ? "blocked" : statusRaw === "done" ? "done" : "todo";
  const completed = !!fm.completed || status === "done";
  const priority = ["high", "medium", "low"].includes(str(fm.priority)) ? str(fm.priority) : "medium";
  const dueDate = str(fm.due_date) || undefined;
  const startDate = str(fm.start_date) || undefined;
  const dependency = arr(fm.dependency);
  const effort = num(fm.effort, 0) || undefined;
  const progress = Math.max(0, Math.min(100, num(fm.progress, completed ? 100 : 0)));
  return {
    id: str(fm.id, generateId()),
    description: str(fm.description, str(fm.title, basename(path))),
    completed,
    completedAt: completed ? ts(fm.completed_at) : undefined,
    createdAt: ts(fm.created_at),
    status,
    priority,
    dueDate,
    startDate,
    dueTime: str(fm.due_time) || undefined,
    effort,
    dependency,
    progress,
    reminderOffset: num(fm.reminder_offset, 0),
    tags: arr(fm.tags),
    recurrence: parseRecurrence(typeof fm.recurrence === "string" ? fm.recurrence : undefined),
    isRecurrenceInstance: !!fm.is_recurrence_instance,
    originalTaskId: str(fm.original_task_id) || undefined,
    notePath: path,
    listId: list.id,
  };
}

/** 判断文件所属实体类型 */
function classify(plugin: ArkOSPlugin, path: string): "log" | "db" | "idea" | "health" | "drawing" | "task" | null {
  const s = plugin.data.settings;
  if (path.startsWith(s.normalLogFolder) || path.startsWith(s.faultLogFolder)) return "log";
  if (path.startsWith(s.ideaFolder)) return "idea";
  if (path.startsWith(s.healthFolder)) return "health";
  if (path.startsWith(s.drawingFolder)) return "drawing";
  if (path.startsWith(s.todoFolder)) return "task";
  if (path.startsWith(s.databaseFolder)) return "db";
  return null;
}

/** 单文件导入 → upsert 到 data */
export async function importMarkdownAt(plugin: ArkOSPlugin, file: TFile) {
  const kind = classify(plugin, file.path);
  if (!kind) return false;
  const { fm, body } = await parseMd(plugin, file);

  if (kind === "log") {
    const s = plugin.data.settings;
    const type = normLogType(str(fm.log_type, str(fm.type)));
    const entry: LogLike = {
      id: str(fm.id, generateId()),
      type,
      title: str(fm.title, basename(file.path)),
      content: str(fm.content, body.trim()),
      createdAt: ts(fm.created_at),
      tags: arr(fm.tags),
      notePath: file.path,
    };
    upsert(plugin.data.logs as any[], "notePath", entry, file.path);
  } else if (kind === "idea") {
    const entry: IdeaLike = {
      id: str(fm.id, generateId()),
      title: str(fm.title, basename(file.path)),
      content: str(fm.content, body.trim()),
      tags: arr(fm.tags),
      archived: !!fm.archived || str(fm.status) === "archived",
      createdAt: ts(fm.created_at),
      updatedAt: ts(fm.updated_at, ts(fm.created_at)),
      notePath: file.path,
    };
    upsert(plugin.data.ideas as any[], "notePath", entry, file.path);
  } else if (kind === "health") {
    const entry: HealthLike = {
      id: str(fm.id, generateId()),
      date: str(fm.date, getToday()),
      water: num(fm.water), sleep: num(fm.sleep),
      steps: num(fm.steps), weight: num(fm.weight),
      createdAt: ts(fm.created_at),
      notePath: file.path,
    };
    upsert(plugin.data.healthRecords as any[], "notePath", entry, file.path);
  } else if (kind === "drawing") {
    const entry: DrawingLike = {
      id: str(fm.id, generateId()),
      title: str(fm.title, basename(file.path)),
      description: str(fm.description),
      color: str(fm.color, "#00e5ff"),
      createdAt: ts(fm.created_at),
      updatedAt: ts(fm.updated_at, ts(fm.created_at)),
      notePath: file.path,
    };
    upsert(plugin.data.drawings as any[], "notePath", entry, file.path);
  } else if (kind === "task") {
    const task = taskFromFm(plugin, file.path, fm);
    if (task) {
      const list = plugin.data.todoLists.find((l) => l.id === task.listId);
      if (list) upsert(list.tasks as any[], "notePath", task, file.path);
    }
  } else if (kind === "db") {
    const s = plugin.data.settings;
    const categoryId = str(fm.db_category, str(fm.category_id, inferCategory(plugin, file.path)));
    const entry: DbLike = {
      id: str(fm.id, generateId()),
      name: str(fm.db_name, str(fm.title, basename(file.path))),
      categoryId,
      status: normStatus(str(fm.status)),
      tags: fm.db_tags ? arr(fm.db_tags) : arr(fm.tags),
      source: str(fm.source, str(fm.type, "manual")),
      sourceRef: str(fm.source_ref, str(fm.url)),
      description: str(fm.description, body.trim().slice(0, 200)),
      filePath: file.path,
      createdAt: ts(fm.created_at),
      updatedAt: ts(fm.updated_at, ts(fm.created_at)),
    };
    upsertDb(plugin, entry, file.path);
  }
  return true;
}

function inferCategory(plugin: ArkOSPlugin, path: string): string {
  const s = plugin.data.settings;
  if (path.startsWith(s.reportFolder)) return "report";
  if (path.startsWith(s.battleReportFolder)) return "battle_report";
  if (path.startsWith(s.mailFolder)) return "mail";
  return s.databaseCategories[0]?.id || "reference";
}

function upsert<T extends { [k: string]: any }>(arr: T[], key: string, entry: T, match: string) {
  const i = arr.findIndex((x) => x[key] === match);
  if (i >= 0) arr[i] = entry;
  else arr.push(entry);
}

function upsertDb(plugin: ArkOSPlugin, entry: DbLike, path: string) {
  const arr = plugin.data.database as DbLike[];
  const i = arr.findIndex((x) => x.filePath === path || x.id === entry.id);
  if (i >= 0) { arr[i] = entry; return; }
  arr.push(entry);
}

/** 全量扫描受管目录 → 导入 data */
async function scanAll(plugin: ArkOSPlugin): Promise<number> {
  const s = plugin.data.settings;
  const folders = unique([s.normalLogFolder, s.faultLogFolder, s.ideaFolder, s.healthFolder, s.drawingFolder, s.todoFolder, s.databaseFolder]);
  let count = 0;
  for (const f of folders) {
    const folderNode: any = plugin.app.vault.getAbstractFileByPath(f);
    if (!folderNode || !folderNode.children) continue;
    for (const tf of filesIn(plugin, f)) {
      const ok = await importMarkdownAt(plugin, tf);
      if (ok) count++;
    }
  }
  await plugin.savePluginData();
  return count;
}

function unique(list: string[]): string[] {
  return [...new Set(list.filter((x) => x.length > 0))];
}

/** 按路径删除 data 中对应条目 */
export async function removeAt(plugin: ArkOSPlugin, path: string) {
  let changed = false;
  for (const arr of [plugin.data.logs, plugin.data.ideas, plugin.data.healthRecords, plugin.data.database] as any[][]) {
    const i = arr.findIndex((x) => x.notePath === path || x.filePath === path);
    if (i >= 0) { arr.splice(i, 1); changed = true; }
  }
  for (const list of plugin.data.todoLists) {
    const i = list.tasks.findIndex((t) => t.notePath === path);
    if (i >= 0) { list.tasks.splice(i, 1); changed = true; }
  }
  if (changed) await plugin.savePluginData();
}

/** 文件重命名 → 更新各集合中的路径并重新导入 */
export async function renameAt(plugin: ArkOSPlugin, oldPath: string, newPath: string) {
  for (const arr of [plugin.data.ideas, plugin.data.healthRecords] as any[][]) {
    const it = arr.find((x) => x.notePath === oldPath);
    if (it) it.notePath = newPath;
  }
  const db = plugin.data.database.find((x) => x.filePath === oldPath);
  if (db) db.filePath = newPath;
  const log = plugin.data.logs.find((x) => x.notePath === oldPath);
  if (log) log.notePath = newPath;
  for (const list of plugin.data.todoLists) {
    const t = list.tasks.find((x) => x.notePath === oldPath);
    if (t) t.notePath = newPath;
  }
  await plugin.savePluginData();
}

/** 挂载 Vault 文件监听（增量同步） */
export function attachWatcher(plugin: ArkOSPlugin) {
  plugin.registerEvent(
    plugin.app.vault.on("create", (f) => {
      if (f instanceof TFile) void importMarkdownAt(plugin, f).then((ok) => { if (ok) return plugin.savePluginData(); });
    }),
  );
  plugin.registerEvent(
    plugin.app.vault.on("modify", (f) => {
      if (f instanceof TFile) void importMarkdownAt(plugin, f).then((ok) => { if (ok) return plugin.savePluginData(); });
    }),
  );
  plugin.registerEvent(
    plugin.app.vault.on("delete", (f) => {
      if (f instanceof TFile) void removeAt(plugin, f.path);
    }),
  );
  plugin.registerEvent(
    plugin.app.vault.on("rename", (f, oldPath) => {
      if (f instanceof TFile) void renameAt(plugin, oldPath, f.path).then(() => importMarkdownAt(plugin, f));
    }),
  );
}

/** 一键开启闭环：建目录 + 全量扫描 + 挂监听 */
export async function setupSync(plugin: ArkOSPlugin) {
  const s = plugin.data.settings;
  for (const f of unique([s.normalLogFolder, s.faultLogFolder, s.ideaFolder, s.healthFolder, s.drawingFolder, s.todoFolder, s.databaseFolder, s.reportFolder])) {
    await ensureFolder(plugin, f);
  }
  await ensureDashboardFile(plugin); // 首页待办数据源：dashboard.md「今日待办」区块
  const n = await scanAll(plugin);
  if (!(plugin as any)._syncAttached) {
    attachWatcher(plugin);
    (plugin as any)._syncAttached = true;
  }
  // 就绪时序（§16.3）：首轮全量扫描完成视为就绪，通知 dock 等首页渲染
  (plugin as any)._dataReady = true;
  plugin.app.workspace.trigger("ark:data-ready");
  console.log(`[Ark] Vault 同步完成，导入 ${n} 条`);
}

/** 将一个内容对象写为 md（促发 watcher 进入 data），返回写入路径 */
export async function writeMarkdown(plugin: ArkOSPlugin, folder: string, filename: string, fm: Record<string, unknown>, body: string): Promise<string> {
  await ensureFolder(plugin, folder);
  const frontmatter = Object.entries(fm)
    .filter(([, v]) => v !== undefined)
    .map(([k, v]) => `${k}: ${serializeFmValue(v)}`)
    .join("\n");
  const md = `---\n${frontmatter}\n---\n\n${body}\n`;
  const path = `${folder}/${filename}.md`;
  const existing = plugin.app.vault.getAbstractFileByPath(path);
  let file: TFile;
  if (existing instanceof TFile) {
    await plugin.app.vault.modify(existing, md);
    file = existing;
  } else {
    file = await plugin.app.vault.create(path, md);
  }
  await importMarkdownAt(plugin, file);
  await plugin.savePluginData();
  return path;
}

function serializeFmValue(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map((x) => `"${String(x)}"`).join(", ")}]`;
  return String(v);
}
