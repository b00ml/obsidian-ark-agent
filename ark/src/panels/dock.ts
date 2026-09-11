// 助手主页（dock）Home Dashboard：卡片式聚合工作台（M-D1 + D2）+ 布局配置驱动 + 首页待办单文件数据库（深对齐 apex）
// 今日待办卡直接读写 dashboard.md「今日待办」区块（dashboard.ts）；灵感/笔记仍为只读聚合 ArkData，点卡跳对应模块。
// 区块顺序/来源由 dashboard.md 的 sections 配置（dock-layout.ts）驱动；非法配置回退默认布局。
import type { SpaceOSView } from "../view";
import { getSkin } from "../skins";
import { dockStats } from "../stats";
import { createCard } from "../cards";
import { loadDockLayout, dockLayoutPath, DEFAULT_DOCK_LAYOUT, DockLayoutSection } from "../dock-layout";
import { getGreeting, getDateLabel } from "../utils";
import { loadDockTodoCards, addDockTodoCard, removeDockTodoCard, addDockTask, flipDockTask, renameDockTask, removeDockTask, countTree, type DockTask, type DockTodoCard } from "../dashboard";
import { confirmDialog } from "../ui";

interface IdeaLite {
  title: string;
  content?: string;
  updatedAt: number;
  archived?: boolean;
  notePath?: string;
}
interface LogLite {
  title?: string;
  content?: string;
  createdAt: number;
  notePath?: string;
}

// 每个 view 只注册一次 dashboard.md 热更新监听，避免重复渲染（WeakSet 去重）
const HOT_RELOADED = new WeakSet<SpaceOSView>();

/** 渲染 Home Dashboard。 */
export function renderDockDashboard(view: SpaceOSView, mount: HTMLElement) {
  const data = view.plugin.data;
  const skin = getSkin(data.settings.skin);
  const dock = skin.panel.dock;
  const captain = data.settings.captainName || "";
  mount.addClass("dock-dashboard");

  // ===== Banner：问候 + KPI + 可选背景图 =====
  const banner = mount.createDiv({ cls: "dock-banner", attr: { "data-ready": "false" } });
  const greeting = banner.createDiv({ cls: "dock-greeting" });
  greeting.createDiv({ cls: "dock-greeting-date", text: getDateLabel() });
  greeting.createDiv({ cls: "dock-greeting-text", text: dock.bannerGreeting.replace("{greeting}", getGreeting()).replace("{captain}", captain) });
  const kpiEls: Record<string, HTMLElement> = {};
  (Object.keys(dock.kpi) as (keyof typeof dock.kpi)[]).forEach((k) => {
    const cell = kpiEls[k] = banner.createDiv({ cls: "dock-kpi-cell" });
    cell.createDiv({ cls: "dock-kpi-value", text: "…" });
    cell.createDiv({ cls: "dock-kpi-label", text: dock.kpi[k] });
  });

  // 注意：onReady / renderKpi 必须在各自引用的闭包变量（kpiEls 等）初始化之后再定义并被调用，
  // 否则同步分支（数据已就绪）会触发 TDZ：Cannot access 'renderKpi' before initialization。
  const renderKpi = () => {
    const stats = dockStats(view.plugin);
    (Object.keys(kpiEls) as (keyof typeof dock.kpi)[]).forEach((k) => {
      const v = kpiEls[k].querySelector<HTMLElement>(".dock-kpi-value")!;
      if (k === "mail") {
        v.setText("…");
        stats.mail.then((n) => { v.setText(String(n)); });
      } else {
        v.setText(String(stats[k]));
      }
    });
  };

  const onReady = () => {
    const b = mount.querySelector<HTMLElement>(".dock-banner");
    if (b) b.setAttr("data-ready", "true");
    renderKpi();
  };
  if ((view.plugin as any)._dataReady) onReady();
  else (view.plugin.app.workspace.on as any)("ark:data-ready", () => { onReady(); });

  // ===== 布局驱动的卡片区（先同步渲染默认布局保证首屏非空，布局文件就绪后再原位升级） =====
  const sections = mount.createDiv({ cls: "dock-sections" });
  for (const sec of DEFAULT_DOCK_LAYOUT.sections) {
    renderSection(view, sections, sec, dock.empty);
  }

  // 布局文件就绪后按 sections 重渲染；非法配置在 loadDockLayout 内已回退默认布局
  void loadDockLayout(view.plugin).then((layout) => {
    const bg = layout.banner?.backgroundImage || data.settings.backgroundImagePath;
    if (bg) {
      banner.setAttr("data-bg", bg);
      banner.style.backgroundImage = `url("${encodeURI(bg)}")`;
    }
    if (layout.banner && layout.banner.showStats === false) {
      banner.querySelectorAll(".dock-kpi-cell").forEach((c) => c.remove());
    }
    sections.empty();
    for (const sec of layout.sections) {
      renderSection(view, sections, sec, dock.empty);
    }
  });

  // ===== 底部最近动态 =====
  renderRecent(view, mount, dock.recentTitle, dock.empty);

  ensureHotReload(view);
}

function renderSection(view: SpaceOSView, mount: HTMLElement, sec: DockLayoutSection, emptyText: string) {
  switch (sec.type) {
    case "todo": renderTodoSection(view, mount, sec, emptyText); break;
    case "memo": renderMemoSection(view, mount, sec, emptyText); break;
    case "notes": renderNotesSection(view, mount, sec, emptyText); break;
    case "stats": renderStatsSection(view, mount, emptyText); break;
  }
}

/** 区块头部：点击标题跳对应模块；点「＋」快速生成空白文件 */
function sectionHead(el: HTMLElement, title: string, jump: () => void, onCreate: (() => void) | null, titleTip?: string) {
  const head = el.createDiv({ cls: "dock-section-head" });
  const t = head.createDiv({ cls: "dock-section-title", text: title, attr: { "data-jump": "", ...(titleTip ? { title: titleTip } : {}) } });
  t.addEventListener("click", jump);
  if (onCreate) {
    const add = head.createDiv({ cls: "dock-section-add", text: "＋", attr: { title: "快速生成空白文件", "data-jump": "" } });
    add.addEventListener("click", (e) => { e.stopPropagation(); void onCreate(); });
  }
}

/** 今日待办区：多待办区块（每张卡 = dashboard.md Todo 栏一张 ### 卡），一行放三个。
 *  区块头 ＋ = 新增待办区块；每卡可 ✕ 删除区块，卡内：勾选完成沉底 / 双击原地编辑 / 回车新增 / 进度条。
 *  短待办聚焦；长任务去任务中心排期。 */
function renderTodoSection(view: SpaceOSView, mount: HTMLElement, sec: DockLayoutSection, emptyText: string) {
  const el = mount.createDiv({ cls: "dock-section glass", attr: { "data-type": "todo", "data-sec": sec.id } });

  // 区块头部：标题点击跳任务中心；＋ 新增一个待办区块（卡）
  const head = el.createDiv({ cls: "dock-section-head" });
  const titleEl = head.createDiv({ cls: "dock-section-title", attr: { "data-jump": "", title: "前往任务中心排期" } });
  titleEl.setText(sec.title);
  titleEl.addEventListener("click", () => view.switchTab("tactical"));
  const addCard = head.createDiv({ cls: "dock-section-add", text: "＋ 区块", attr: { title: "新增待办区块（一张卡一个清单，一行可放三个）" } });
  addCard.addEventListener("click", async (e) => {
    e.stopPropagation();
    const cards = await loadDockTodoCards(view.plugin);
    await addDockTodoCard(view.plugin, `待办清单 ${cards.length + 1}`);
    await view.renderCurrentPanel();
  });

  const grid = el.createDiv({ cls: "dock-todo-grid" });
  void loadDockTodoCards(view.plugin).then((cards) => {
    if (cards.length === 0) { grid.createDiv({ cls: "dock-empty", text: emptyText }); return; }
    const openN = cards.reduce((s, cd) => s + countTree(cd.tasks).open, 0);
    titleEl.setText(`${sec.title}${openN > 0 ? ` · ${openN}` : ""}`);
    for (const card of cards) renderTodoCard(view, grid, card);
  });
}

/** 单张待办卡：头（标题 + ✕ 删区块）+ 任务列表 + 进度条 + 回车输入 */
function renderTodoCard(view: SpaceOSView, grid: HTMLElement, card: DockTodoCard) {
  const el = grid.createDiv({ cls: "dock-todo-card glass" });

  const head = el.createDiv({ cls: "dock-todo-card-head" });
  const titleEl = head.createDiv({ cls: "dock-todo-card-title", text: card.title });
  const delCard = head.createDiv({ cls: "dock-row-del", text: "✕", attr: { title: "删除该待办区块" } });
  delCard.addEventListener("click", async () => {
    if (!(await confirmDialog(view.plugin, `删除「${card.title}」区块？`))) return;
    await removeDockTodoCard(view.plugin, card.title);
    await view.renderCurrentPanel();
  });

  const body = el.createDiv({ cls: "dock-section-body" });

  // 完成度进度条（对齐 apex taskCompletion，全树统计）
  const prog = el.createDiv({ cls: "dock-progress", attr: { style: "display:none" } });
  const progBar = prog.createDiv({ cls: "dock-progress-bar" });
  const progLabel = prog.createSpan({ cls: "dock-progress-label", text: "" });

  const renderRow = (t: DockTask, isDone: boolean) => {
    const row = body.createDiv({ cls: `dock-row${isDone ? " dock-row--done" : ""}` });
    const box = row.createEl("input", { cls: "dock-row-check", type: "checkbox" });
    box.checked = isDone;
    box.addEventListener("change", async () => { await flipDockTask(view.plugin, card.title, t.id, box.checked); await view.renderCurrentPanel(); });
    row.createSpan({ cls: "dock-row-dot" });
    const label = row.createSpan({ cls: "dock-row-label", text: t.text || "（无标题）" });

    // 双击 → 原地编辑（Enter 保存 / Esc 取消 / 失焦保存）
    label.addEventListener("dblclick", () => {
      const start = label.getText();
      label.empty();
      const edit = label.createEl("input", { cls: "dock-row-edit", attr: { value: t.text || "" } });
      edit.focus();
      edit.setSelectionRange(edit.value.length, edit.value.length);
      let fired = false;
      const save = () => { if (fired) return; fired = true; const v = edit.value.trim(); if (v && v !== start && v !== t.text) void renameDockTask(view.plugin, card.title, t.id, v).then(() => view.renderCurrentPanel()); };
      const cancel = () => { if (fired) return; fired = true; void view.renderCurrentPanel(); };
      edit.addEventListener("keydown", (ke) => {
        if (ke.key === "Enter") { ke.preventDefault(); save(); }
        else if (ke.key === "Escape") { ke.preventDefault(); cancel(); }
      });
      edit.addEventListener("blur", save);
    });

    const del = row.createSpan({ cls: "dock-row-del", text: "✕", attr: { title: "删除该待办" } });
    del.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!(await confirmDialog(view.plugin, "删除该待办？"))) return;
      await removeDockTask(view.plugin, card.title, t.id);
      await view.renderCurrentPanel();
    });
  };

  const c = countTree(card.tasks);
  const open = card.tasks.filter((t) => !t.checked);
  const done = card.tasks.filter((t) => t.checked);
  titleEl.setText(`${card.title}${card.tasks.length > 0 ? ` · ${c.open}` : ""}`);
  if (card.tasks.length === 0) body.createDiv({ cls: "dock-empty", text: "暂无待办" });

  // 未完成逐条展示；已完成删除线 + 沉底（全部展示不截断，避免完成项被 slice 截掉）
  open.forEach((t) => renderRow(t, false));
  done.forEach((t) => renderRow(t, true));

  // 进度条
  const all = c.open + c.done;
  if (all > 0) {
    const pct = Math.round((c.done / all) * 100);
    progBar.setAttr("style", `width:${pct}%`);
    progLabel.setText(`${pct}%`);
    prog.removeAttribute("style");
  }

  // 卡内快捷输入：回车写待办到该卡
  const addRow = el.createDiv({ cls: "dock-addrow" });
  const inp = addRow.createEl("input", { cls: "dock-todo-input", attr: { placeholder: "回车添加" } });
  inp.addEventListener("keydown", async (e) => {
    if (e.key !== "Enter") return;
    const v = inp.value.trim();
    if (!v) return;
    inp.value = "";
    await addDockTask(view.plugin, card.title, v);
    await view.renderCurrentPanel();
  });
}

/** 灵感速记区：一个块 = 一个 .md 文件（不再内部拆列文件）。
 *  单击跳灵感草稿；双击直接打开该文件编辑。 */
function renderMemoSection(view: SpaceOSView, mount: HTMLElement, sec: DockLayoutSection, emptyText: string) {
  const el = mount.createDiv({ cls: "dock-section glass", attr: { "data-type": "memo", "data-sec": sec.id } });
  sectionHead(el, sec.title, () => view.switchTab("capture"), () => addIdeaBlank(view));
  const body = el.createDiv({ cls: "dock-section-body" });
  const ideas = (view.plugin.data.ideas as IdeaLite[])
    .filter((i) => !(i as any).archived && inSource(i.notePath, sec.source))
    .sort((a, b) => b.updatedAt - a.updatedAt).slice(0, 8);
  if (ideas.length === 0) { body.createDiv({ cls: "dock-empty", text: emptyText }); }
  ideas.forEach((i) => {
    const row = body.createDiv({ cls: "dock-row", attr: { title: "双击直接编辑该文件" } });
    row.createSpan({ cls: "dock-row-dot" });
    row.createSpan({ cls: "dock-row-label", text: i.title || "（无标题）" });
    row.createSpan({ cls: "dock-row-time", text: fmtShort(i.updatedAt) });
    row.addEventListener("click", () => view.switchTab("capture"));
    row.addEventListener("dblclick", () => openNoteFile(view, i.notePath));
  });
}

/** 直接打开笔记文件编辑（一个块 = 一个文件） */
function openNoteFile(view: SpaceOSView, path: string | undefined) {
  if (!path) return;
  const f = view.plugin.app.vault.getAbstractFileByPath(path);
  if (f) view.plugin.app.workspace.getLeaf("tab").openFile(f as any);
}

/** 灵感「＋」：生成一个空白灵感文件 */
async function addIdeaBlank(view: SpaceOSView) {
  const ts = fmtStamp();
  await createCard(view.plugin, "idea", { fm: { title: `新灵感 ${ts}`, tags: [], archived: false }, body: "" });
  await view.renderCurrentPanel();
}

/** 快捷笔记卡：头部跳 database + 快速新建空白笔记 */
function renderNotesSection(view: SpaceOSView, mount: HTMLElement, sec: DockLayoutSection, emptyText: string) {
  const el = mount.createDiv({ cls: "dock-section glass", attr: { "data-type": "notes", "data-sec": sec.id } });
  sectionHead(el, sec.title, () => view.switchTab("database"), () => addNoteBlank(view));
  const body = el.createDiv({ cls: "dock-section-body" });
  const db = (view.plugin.data.database as any[])
    .filter((n) => inSource(n.filePath || n.notePath, sec.source))
    .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0)).slice(0, 5);
  if (db.length === 0) { body.createDiv({ cls: "dock-empty", text: emptyText }); }
  db.forEach((n) => {
    const p = n.filePath || n.notePath;
    const row = body.createDiv({ cls: "dock-row" });
    row.createSpan({ cls: "dock-row-dot" });
    row.createSpan({ cls: "dock-row-label", text: n.name || n.dbName || n.title || "（未命名）" });
    const file = p ? view.plugin.app.vault.getAbstractFileByPath(p) : null;
    row.addEventListener("click", () => { if (file) view.plugin.app.workspace.getLeaf("tab").openFile(file as any); });
  });
}

/** 笔记「＋」：生成一个空白笔记文件 */
async function addNoteBlank(view: SpaceOSView) {
  const ts = fmtStamp();
  const title = `新笔记 ${ts}`;
  await createCard(view.plugin, "database", { fm: { title, db_name: title, tags: [] }, body: "" });
  await view.renderCurrentPanel();
}

/** stats 区块卡：KPI 网格（覆盖 banner.showStats=false 时的兜底，独立成卡） */
function renderStatsSection(view: SpaceOSView, mount: HTMLElement, emptyText: string) {
  const skin = getSkin(view.plugin.data.settings.skin);
  const kpi = skin.panel.dock.kpi;
  const el = mount.createDiv({ cls: "dock-section glass", attr: { "data-type": "stats" } });
  el.createDiv({ cls: "dock-section-title", text: kpi.todo });
  const body = el.createDiv({ cls: "dock-kpi-row" });
  const stats = dockStats(view.plugin);
  (Object.keys(kpi) as (keyof typeof kpi)[]).forEach((k) => {
    const cell = body.createDiv({ cls: "dock-kpi-cell" });
    const v = cell.createDiv({ cls: "dock-kpi-value" });
    cell.createDiv({ cls: "dock-kpi-label", text: kpi[k] });
    if (k === "mail") stats.mail.then((n) => v.setText(String(n)));
    else v.setText(String(stats[k]));
  });
}

/** 底部最近动态：最近日志 + 灵感 */
function renderRecent(view: SpaceOSView, mount: HTMLElement, title: string, emptyText: string) {
  const recent = mount.createDiv({ cls: "dock-recent" });
  recent.createDiv({ cls: "dock-section-title", text: title });
  const items: { text: string; time: number; target: "logs" | "capture" }[] = [];
  (view.plugin.data.logs as LogLite[]).forEach((l) => items.push({ text: l.title || "（日志）", time: l.createdAt, target: "logs" }));
  (view.plugin.data.ideas as IdeaLite[]).forEach((i) => items.push({ text: i.title || "（灵感）", time: i.updatedAt, target: "capture" }));
  items.sort((a, b) => b.time - a.time).slice(0, 6).forEach((it) => {
    const row = recent.createDiv({ cls: "dock-row" });
    row.createSpan({ cls: "dock-row-label", text: it.text });
    row.createSpan({ cls: "dock-row-time", text: fmtShort(it.time) });
    row.addEventListener("click", () => view.switchTab(it.target));
  });
  if (items.length === 0) recent.createDiv({ cls: "dock-empty", text: emptyText });
}

/** dashboard.md 被外部编辑 → 重渲当前面板（热更新）；仅注册一次 */
function ensureHotReload(view: SpaceOSView) {
  if (HOT_RELOADED.has(view)) return;
  HOT_RELOADED.add(view);
  const path = dockLayoutPath(view.plugin);
  view.plugin.registerEvent(view.plugin.app.vault.on("modify", async (f) => {
    if (f.path === path) await view.renderCurrentPanel();
  }));
  view.plugin.registerEvent(view.plugin.app.vault.on("create", async (f) => {
    if (f.path === path) await view.renderCurrentPanel();
  }));
}

function inSource(path: string | undefined, source: string | undefined): boolean {
  if (!source) return true;
  if (!path) return false;
  const norm = path.replace(/\\/g, "/");
  const s = source.replace(/\\/g, "/").replace(/^\/|\/$/g, "");
  return norm === s || norm.startsWith(s + "/") || norm.startsWith(s);
}

function fmtStamp(): string {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, "0")}${String(d.getMinutes()).padStart(2, "0")}${String(d.getSeconds()).padStart(2, "0")}`;
}
function fmtShort(ts: number): string {
  const d = new Date(ts);
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${m}.${day} ${hh}:${mm}`;
}
