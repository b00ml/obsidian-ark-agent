// 首页待办数据层：与 apex-dashboard 深度对齐的单文件数据库（支持多待办区块）。
// ark 首页「今日待办」区读写 dashboard.md 中 Todo 栏（## Todo）下的若干任务卡
// （### 标题 + id:/type: 元数据 + 嵌套 checkbox，与 apex 序列化格式一致）。
// 因此 apex 与 ark 可共享同一 dashboard.md，apex 重新启用时能直接识别/编辑这些卡。
// 保留 frontmatter 与其他栏/卡（含第三方卡片）原样不动，仅做指定卡内 checkbox 块的定点替换（surgical replace）。
import type ArkOSPlugin from "./main";
import { TFile } from "obsidian";

export interface DockTask {
  id: string;        // 内存态路径 id，如 "0"、"1"、"0-2"（apex 无逐任务 id，用路径定位）
  text: string;
  checked: boolean;
  children?: DockTask[];
  reminder?: string;  // "YYYY-MM-DD HH:mm"
  collapsed?: boolean;
}

/** 一个待办区块（dashboard.md Todo 栏下的一张 ### 卡） */
export interface DockTodoCard {
  title: string;
  tasks: DockTask[];
}

/** dashboard.md 路径（读设置，缺省 dashboard.md） */
export function dashboardPath(plugin: ArkOSPlugin): string {
  return plugin.data.settings.dashboardFile || "dashboard.md";
}

// ===== apex 语法（与 plugin/apex-dashboard/src/parser.ts 对齐）=====
const REMINDER_RE = /\s*⏰\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*$/;
const COLLAPSED_RE = /\s*<!--collapsed-->\s*$/;
const CHECKBOX_RE = /^(\s*)- \[([ xX])\]\s+(.+)$/;
const H2_RE = /^##\s+\S.*$/;
const H2_TODO_RE = /^##\s+Todo\s*$/;
const H3_RE = /^###\s+\S.*$/;
const KV_RE = /^\w+:\s*\S/;   // 卡片元数据行（id:/type:/link:…）

// ===== 解析：从任务卡块构建嵌套任务树（apex 4 空格缩进层级）=====
function indentDepth(indent: string): number {
  let n = 0;
  for (const ch of indent) {
    if (ch === "\t") n += 4;
    else if (ch === " ") n += 1;
    else break;
  }
  return Math.floor(n / 4);
}

function renumber(nodes: DockTask[], prefix = ""): DockTask[] {
  return nodes.map((n, i) => {
    const id = prefix ? `${prefix}-${i}` : String(i);
    return { ...n, id, children: n.children ? renumber(n.children, id) : n.children };
  });
}

/** 仅解析指定行区间内的 checkbox 行 -> 嵌套任务树（其余行忽略） */
function buildTasksFromLines(lines: string[], start: number, end: number): DockTask[] {
  const root: DockTask[] = [];
  const stack: { depth: number; node: DockTask }[] = [];
  for (let i = start; i < end; i++) {
    const m = CHECKBOX_RE.exec(lines[i]);
    if (!m) continue;
    let text = m[3];
    let reminder: string | undefined;
    let collapsed = false;
    const cm = text.match(COLLAPSED_RE);
    if (cm) { text = text.replace(COLLAPSED_RE, ""); collapsed = true; }
    const rm = text.match(REMINDER_RE);
    if (rm) { text = text.replace(REMINDER_RE, ""); reminder = rm[1]; }
    const depth = indentDepth(m[1]);
    const node: DockTask = { id: "", text: text.trim(), checked: /[xX]/.test(m[2]), reminder, collapsed };
    while (stack.length && stack[stack.length - 1].depth >= depth) stack.pop();
    if (stack.length === 0) root.push(node);
    else {
      const p = stack[stack.length - 1].node;
      p.children = p.children ?? [];
      p.children.push(node);
    }
    stack.push({ depth, node });
  }
  return renumber(root);
}

// ===== 定位「Todo 栏」下所有任务卡（### 标题）的文本块 =====
interface CardLoc {
  title: string;
  heading: number;    // ### 行号
  metaEnd: number;    // 元数据（id:/type:）之后的首行（无任务时的插入点）
  taskStart: number;  // 首个 checkbox 行（无任务时 = metaEnd）
  taskEnd: number;    // 最后一个 checkbox 之后的下一行
  cardEnd: number;    // 下一张卡 ### 或栏结束
}

function locateTodoCards(lines: string[]): CardLoc[] | null {
  const colIdx = lines.findIndex((l) => H2_TODO_RE.test(l));
  if (colIdx < 0) return null;
  let colEnd = lines.length;
  for (let i = colIdx + 1; i < lines.length; i++) if (H2_RE.test(lines[i])) { colEnd = i; break; }
  const cards: CardLoc[] = [];
  for (let i = colIdx + 1; i < colEnd; i++) {
    if (!H3_RE.test(lines[i])) continue;
    const c: CardLoc = {
      title: lines[i].replace(/^###\s+/, "").trim(),
      heading: i, metaEnd: i + 1, taskStart: i + 1, taskEnd: i + 1, cardEnd: colEnd,
    };
    if (cards.length) cards[cards.length - 1].cardEnd = i;
    cards.push(c);
  }
  if (cards.length === 0) return null;
  for (const c of cards) {
    // 跳过标题后的空行与元数据行，得到任务插入点
    let i = c.heading + 1;
    while (i < c.cardEnd && (lines[i].trim() === "" || KV_RE.test(lines[i].trim()))) i++;
    c.metaEnd = i;
    // 卡内 checkbox 连续块
    let ts = -1;
    for (let j = c.metaEnd; j < c.cardEnd; j++) if (CHECKBOX_RE.test(lines[j])) { ts = j; break; }
    if (ts >= 0) {
      c.taskStart = ts;
      let te = ts + 1;
      while (te < c.cardEnd && CHECKBOX_RE.test(lines[te])) te++;
      c.taskEnd = te;
    } else {
      c.taskStart = c.metaEnd;
      c.taskEnd = c.metaEnd;
    }
  }
  return cards;
}

/** 兜底：确保 Todo 栏 + 至少一张任务卡存在（不覆盖用户已有内容，仅新增缺失部分） */
function ensureTodoCard(lines: string[]): string[] {
  const colIdx = lines.findIndex((l) => H2_TODO_RE.test(l));
  if (colIdx < 0) {
    let end = lines.length;
    while (end > 0 && lines[end - 1].trim() === "") end--;
    return [...lines.slice(0, end), "", "## Todo", "", "### 今日待办", "id: todo-today", "type: task", ""];
  }
  let colEnd = lines.length;
  for (let i = colIdx + 1; i < lines.length; i++) if (H2_RE.test(lines[i])) { colEnd = i; break; }
  const hasCard = lines.slice(colIdx + 1, colEnd).some((l) => H3_RE.test(l));
  if (hasCard) return lines;
  const card = ["### 今日待办", "id: todo-today", "type: task", ""];
  return [...lines.slice(0, colIdx + 1), ...card, ...lines.slice(colIdx + 1)];
}

async function readLines(plugin: ArkOSPlugin): Promise<string[] | null> {
  const f = plugin.app.vault.getAbstractFileByPath(dashboardPath(plugin));
  if (!(f instanceof TFile)) return null;
  return (await plugin.app.vault.read(f)).split(/\r?\n/);
}

// ===== 读取：dashboard.md Todo 栏下所有任务卡 =====
export async function loadDockTodoCards(plugin: ArkOSPlugin): Promise<DockTodoCard[]> {
  const lines = await readLines(plugin);
  if (!lines) return [];
  const cards = locateTodoCards(lines);
  if (!cards) return [];
  return cards.map((c) => ({ title: c.title, tasks: buildTasksFromLines(lines, c.taskStart, c.taskEnd) }));
}

// ===== 序列化：把任务树写回指定卡（定点替换卡内 checkbox 块）=====
function serializeTasks(tasks: DockTask[]): string[] {
  const out: string[] = [];
  const write = (node: DockTask, depth: number): void => {
    const prefix = depth > 0 ? "    ".repeat(depth) : "";
    let line = `${prefix}- [${node.checked ? "x" : " "}] ${node.text}`;
    if (node.reminder) line += ` ⏰ ${node.reminder}`;
    if (node.collapsed) line += ` <!--collapsed-->`;
    out.push(line);
    for (const c of node.children ?? []) write(c, depth + 1);
  };
  for (const t of tasks) write(t, 0);
  return out;
}

/** 写回某张卡的 checkbox 块（卡缺失则无操作） */
export async function saveDockTodoCard(plugin: ArkOSPlugin, title: string, tasks: DockTask[]): Promise<void> {
  const f = plugin.app.vault.getAbstractFileByPath(dashboardPath(plugin));
  if (!(f instanceof TFile)) return;
  const lines = ensureTodoCard((await plugin.app.vault.read(f)).split(/\r?\n/));
  const c = locateTodoCards(lines)?.find((x) => x.title === title);
  if (!c) return;
  const block = serializeTasks(tasks);
  await plugin.app.vault.modify(f, [...lines.slice(0, c.taskStart), ...block, ...lines.slice(c.taskEnd)].join("\n"));
}

function genCardId(): string {
  return "todo-" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

/** 新增一个待办区块（Todo 栏末尾追加一张卡） */
export async function addDockTodoCard(plugin: ArkOSPlugin, title: string): Promise<void> {
  const f = plugin.app.vault.getAbstractFileByPath(dashboardPath(plugin));
  if (!(f instanceof TFile)) return;
  const lines = ensureTodoCard((await plugin.app.vault.read(f)).split(/\r?\n/));
  const colIdx = lines.findIndex((l) => H2_TODO_RE.test(l));
  if (colIdx < 0) return;
  let colEnd = lines.length;
  for (let i = colIdx + 1; i < lines.length; i++) if (H2_RE.test(lines[i])) { colEnd = i; break; }
  const card = ["", `### ${title}`, `id: ${genCardId()}`, "type: task", ""];
  await plugin.app.vault.modify(f, [...lines.slice(0, colEnd), ...card, ...lines.slice(colEnd)].join("\n"));
}

/** 删除一个待办区块（移除整张卡） */
export async function removeDockTodoCard(plugin: ArkOSPlugin, title: string): Promise<void> {
  const f = plugin.app.vault.getAbstractFileByPath(dashboardPath(plugin));
  if (!(f instanceof TFile)) return;
  const lines = (await plugin.app.vault.read(f)).split(/\r?\n/);
  const c = locateTodoCards(lines)?.find((x) => x.title === title);
  if (!c) return;
  await plugin.app.vault.modify(f, [...lines.slice(0, c.heading), ...lines.slice(c.cardEnd)].join("\n"));
}

// ===== 树查找 / 变更 =====
function findTask(nodes: DockTask[], id: string): DockTask | undefined {
  for (const n of nodes) {
    if (n.id === id) return n;
    const r = findTask(n.children ?? [], id);
    if (r) return r;
  }
  return undefined;
}

function setSubtreeChecked(t: DockTask, checked: boolean): void {
  t.checked = checked;
  for (const c of t.children ?? []) setSubtreeChecked(c, checked);
}

function removeById(nodes: DockTask[], id: string): DockTask[] {
  return nodes
    .filter((n) => n.id !== id)
    .map((n) => (n.children ? { ...n, children: removeById(n.children, id) } : n));
}

/** 全树开放/完成计数（apex taskCompletion 口径一致） */
export function countTree(tasks: DockTask[]): { open: number; done: number } {
  return tasks.reduce((acc, n) => {
    acc[n.checked ? "done" : "open"] += 1;
    const c = countTree(n.children ?? []);
    acc.open += c.open;
    acc.done += c.done;
    return acc;
  }, { open: 0, done: 0 });
}

// ===== 首页操作封装：读某卡 + 改 + 写（卡缺失时自动补建）=====
async function withCard(plugin: ArkOSPlugin, cardTitle: string, fn: (c: DockTodoCard) => void): Promise<void> {
  let cards = await loadDockTodoCards(plugin);
  let c = cards.find((x) => x.title === cardTitle);
  if (!c) {
    await addDockTodoCard(plugin, cardTitle);
    cards = await loadDockTodoCards(plugin);
    c = cards.find((x) => x.title === cardTitle);
    if (!c) return;
  }
  fn(c);
  await saveDockTodoCard(plugin, cardTitle, c.tasks);
}

export function addDockTask(plugin: ArkOSPlugin, cardTitle: string, text: string): Promise<void> {
  return withCard(plugin, cardTitle, (c) => { c.tasks.push({ id: "", text, checked: false }); });
}

export function flipDockTask(plugin: ArkOSPlugin, cardTitle: string, id: string, checked: boolean): Promise<void> {
  return withCard(plugin, cardTitle, (c) => {
    const t = findTask(c.tasks, id);
    if (t && t.checked !== checked) { t.checked = checked; setSubtreeChecked(t, checked); }
  });
}

export function renameDockTask(plugin: ArkOSPlugin, cardTitle: string, id: string, text: string): Promise<void> {
  return withCard(plugin, cardTitle, (c) => {
    const t = findTask(c.tasks, id);
    if (t && t.text !== text) t.text = text;
  });
}

export function removeDockTask(plugin: ArkOSPlugin, cardTitle: string, id: string): Promise<void> {
  return withCard(plugin, cardTitle, (c) => { c.tasks = removeById(c.tasks, id); });
}

// ===== 兜底：dashboard.md 模板 + 确保 Todo 栏/卡存在（setupSync 调用）=====
const DASH_PREAMBLE = [
  "---",
  "dashboard: true",
  "banner:",
  '  quote: "The mind is everything. What you think you become."',
  '  author: "Buddha"',
  "sections:",
  "  - id: todo-today",
  "    type: todo",
  "    title: 今日待办",
  "  - id: idea-notes",
  "    type: memo",
  "    source: 04-灵感",
  "    title: 灵感速记",
  "  - id: quick-notes",
  "    type: notes",
  "    source: 02-DB",
  "    title: 快捷笔记",
  "columns:",
  "  - name: Todo",
  '    color: "#6366f1"',
  "    type: todo",
  "---",
];

/** 文件缺失时创建的 apex 兼容模板（frontmatter：ark sections 布局 + apex columns 数据） */
export function defaultDashboardMarkdown(): string {
  return [
    ...DASH_PREAMBLE,
    "",
    "## Todo",
    "",
    "### 今日待办",
    "id: todo-today",
    "type: task",
    "- [ ] 双击编辑 · 勾选完成 · 回车新增",
    "",
  ].join("\n");
}

/** 确保 dashboard.md 存在，且含「## Todo / ### 任务卡」（不覆盖用户已有内容） */
export async function ensureDashboardFile(plugin: ArkOSPlugin): Promise<void> {
  const path = dashboardPath(plugin);
  const f = plugin.app.vault.getAbstractFileByPath(path);
  if (!(f instanceof TFile)) {
    await plugin.app.vault.create(path, defaultDashboardMarkdown());
    return;
  }
  const md = await plugin.app.vault.read(f);
  const nextLines = ensureTodoCard(md.split(/\r?\n/));
  const next = nextLines.join("\n");
  if (next !== md.replace(/\r?\n/g, "\n")) {
    await plugin.app.vault.modify(f, next);
  }
}
