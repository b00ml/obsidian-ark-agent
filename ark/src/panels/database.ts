import type { SpaceOSView } from "../view";
import { createCard, updateCard, deleteCardAt } from "../cards";
import { inputDialog, confirmDialog, notice } from "../ui";
import { sendSelectionToWorkbench } from "../agent-workbench";

interface DbItem {
  id: string;
  name: string;
  categoryId: string;
  status: "pending" | "solved" | "other";
  tags: string[];
  description?: string;
  filePath?: string;
  createdAt: number;
  updatedAt: number;
  source?: string;
  sourceRef?: string;
}

/** 知识工作台：检索、判断处理状态、打开来源、把证据交给 Agent。 */
export function renderDatabase(view: SpaceOSView, mount: HTMLElement) {
  mount.addClass("space-db");
  const d = view.plugin.data;
  const cats = view.plugin.data.settings.databaseCategories;

  const items = (d.database as DbItem[]).sort((a, b) => b.createdAt - a.createdAt);
  const selected = new Set<string>();

  mount.createDiv({ cls: "db-note", text: "三维：来源（数据从哪来）× 状态（走到哪）× 主题（讲什么）。总览用于定位流入与积压，检索用于逐条处理，主题图谱用于发现可合并的知识簇。" });
  const heading = mount.createDiv({ cls: "db-heading" });
  heading.createDiv({ cls: "tactical-title", text: "维度知识数据库" });
  heading.createDiv({ cls: "tactical-subtitle", text: "只保留可检索、可引用、可继续加工的知识对象；状态不是装饰，而是下一步动作。" });

  const dash = mount.createDiv({ cls: "log-dashboard" });
  dash.createEl("span", { cls: "log-stat", text: `全部 ${items.length}` });
  dash.createEl("span", { cls: "log-stat log-normal", text: `待处理 ${items.filter((i) => i.status !== "solved").length}` });
  dash.createEl("span", { cls: "log-stat log-fault", text: `已处理 ${items.filter((i) => i.status === "solved").length}` });

  const modebar = mount.createDiv({ cls: "db-mode-tabs" });
  const overviewPanel = mount.createDiv({ cls: "db-overview-panel" });
  const topicPanel = mount.createDiv({ cls: "db-topic-panel" });
  const searchPanel = mount.createDiv({ cls: "db-search-panel" });
  let mode: "overview" | "topic" | "search" = "overview";
  const modeButtons = new Map<string, HTMLButtonElement>();
  const showMode = (next: "overview" | "topic" | "search") => {
    mode = next;
    overviewPanel.toggleClass("active", next === "overview");
    topicPanel.toggleClass("active", next === "topic");
    searchPanel.toggleClass("active", next === "search");
    modeButtons.forEach((button, key) => button.toggleClass("active", key === next));
  };
  ([
    ["overview", "◈ 来源 × 状态总览"],
    ["topic", "✦ 主题图谱"],
    ["search", "⌕ 三维检索"],
  ] as const).forEach(([key, label]) => {
    const button = modebar.createEl("button", { cls: "db-mode-tab", text: label });
    button.addEventListener("click", () => showMode(key));
    modeButtons.set(key, button);
  });

  const dimensions = searchPanel.createDiv({ cls: "db-dimensions" });
  const sourceCounts = new Map<string, number>();
  const themeCounts = new Map<string, number>();
  items.forEach((i) => {
    const source = i.source || "manual";
    sourceCounts.set(source, (sourceCounts.get(source) ?? 0) + 1);
    (i.tags ?? []).forEach((tag) => themeCounts.set(tag, (themeCounts.get(tag) ?? 0) + 1));
  });
  const dim = (label: string, values: { text: string; run: () => void }[]) => {
    const row = dimensions.createDiv({ cls: "db-dimension-row" });
    row.createSpan({ cls: "db-dimension-label", text: label });
    values.slice(0, 8).forEach((value) => {
      const chip = row.createEl("button", { cls: "db-dimension-chip", text: value.text });
      chip.title = "点击聚焦此维度";
      chip.addEventListener("click", value.run);
    });
  };
  dim("来源", [...sourceCounts.entries()].sort((a, b) => b[1] - a[1]).map(([key, n]) => ({
    text: `${key} ${n}`, run: () => { source.value = key; showMode("search"); redraw(); },
  })));
  dim("状态", [
    { text: `待处理 ${items.filter((i) => i.status !== "solved").length}`, run: () => { status.value = "pending"; showMode("search"); redraw(); } },
    { text: `已处理 ${items.filter((i) => i.status === "solved").length}`, run: () => { status.value = "solved"; showMode("search"); redraw(); } },
  ]);
  dim("主题", [...themeCounts.entries()].sort((a, b) => b[1] - a[1]).map(([key, n]) => ({
    text: `#${key} ${n}`, run: () => { search.value = key; showMode("search"); redraw(); },
  })));

  const toolbar = searchPanel.createDiv({ cls: "log-toolbar" });
  const status = toolbar.createEl("select", { cls: "db-filter" });
  status.createEl("option", { value: "__all", text: "全部状态" });
  status.createEl("option", { value: "pending", text: "待处理" });
  status.createEl("option", { value: "solved", text: "已处理" });
  const source = toolbar.createEl("select", { cls: "db-filter" });
  source.createEl("option", { value: "__all", text: "全部来源" });
  [...sourceCounts.keys()].sort().forEach((value) => source.createEl("option", { value, text: value }));
  const category = toolbar.createEl("select", { cls: "db-filter" });
  category.createEl("option", { value: "__all", text: "全部分类" });
  category.createEl("option", { value: "__none", text: "未分类" });
  cats.forEach((c) => category.createEl("option", { value: c.id, text: c.name }));
  const search = toolbar.createEl("input", { cls: "sos-crl db-search", attr: { type: "search", placeholder: "搜索标题、正文、标签…" } });
  const selectVisible = toolbar.createEl("button", { cls: "db-batch", text: "选择当前" });
  const batch = toolbar.createEl("button", { cls: "db-batch primary", text: "批量加入 Agent" });
  batch.disabled = true;

  const updateBatch = () => {
    batch.disabled = selected.size === 0;
    batch.setText(selected.size ? `批量加入 Agent · ${selected.size}` : "批量加入 Agent");
  };

  const newBtn = toolbar.createEl("button", { cls: "log-new-btn", text: "+ 新建" });
  newBtn.addEventListener("click", async () => {
    const name = await inputDialog(view.plugin, { title: "新建资产", placeholder: "资产名称" });
    if (!name || !name.trim()) return;
    const desc = (await inputDialog(view.plugin, { title: "描述（可选）", multiline: true })) ?? "";
    const cat = (await inputDialog(view.plugin, { title: "分类 ID（留空自动）", placeholder: "可选" })) ?? "";
    await createCard(view.plugin, "database", {
      fm: {
        db_name: name.trim(),
        category_id: cat.trim() || cats[0]?.id || "reference",
        status: "pending",
        tags: [],
      },
      body: desc,
    });
    redraw();
  });

  const listEl = searchPanel.createDiv({ cls: "db-list" });

  const filteredItems = () => items.filter((i) => {
    const q = search.value.trim().toLowerCase();
    const haystack = [i.name, i.description ?? "", i.source ?? "", i.sourceRef ?? "", ...(i.tags ?? [])].join(" ").toLowerCase();
    if (q && !haystack.includes(q)) return false;
    if (status.value !== "__all" && i.status !== status.value) return false;
    if (source.value !== "__all" && (i.source || "manual") !== source.value) return false;
    if (category.value === "__none" && i.categoryId && i.categoryId !== "_none") return false;
    if (category.value !== "__all" && category.value !== "__none" && i.categoryId !== category.value) return false;
    return true;
  });

  selectVisible.addEventListener("click", () => {
    const visible = filteredItems();
    const allSelected = visible.length > 0 && visible.every((item) => selected.has(item.id));
    visible.forEach((item) => allSelected ? selected.delete(item.id) : selected.add(item.id));
    redraw();
  });
  batch.addEventListener("click", () => {
    const chosen = items.filter((item) => selected.has(item.id)).slice(0, 20);
    if (!chosen.length) return;
    const refs = chosen.map((item) =>
      `- [[${item.filePath ?? item.name}]]｜${item.name}｜来源：${item.source || "manual"}｜标签：${(item.tags ?? []).join(", ")}`,
    ).join("\n");
    const suffix = selected.size > chosen.length ? `\n其余 ${selected.size - chosen.length} 条未注入，请分批处理。` : "";
    sendSelectionToWorkbench(view.plugin,
      `请批量处理以下知识条目：提炼共同主题、指出重复或冲突、给出下一步整理建议。保留每条来源引用，不要直接覆盖原文。\n${refs}${suffix}`);
  });

  function redraw() {
    listEl.empty();
    selected.forEach((id) => { if (!items.some((item) => item.id === id)) selected.delete(id); });
    const filtered = filteredItems();
    if (filtered.length === 0) {
      listEl.createDiv({ cls: "tactical-empty", text: "暂无资产，点击上方新建" });
      updateBatch();
      return;
    }
    filtered.forEach((i) => {
      const c = cats.find((x) => x.id === i.categoryId);
      const card = listEl.createDiv({ cls: "log-entry glass" });
      const head = card.createDiv({ cls: "db-head" });
      const check = head.createEl("input", { cls: "db-check", attr: { type: "checkbox" } }) as HTMLInputElement;
      check.checked = selected.has(i.id);
      check.title = "选择后批量加入 Agent";
      check.addEventListener("change", () => {
        if (check.checked) selected.add(i.id);
        else selected.delete(i.id);
        updateBatch();
      });
      head.createSpan({ cls: "db-dot", attr: { style: `background:${c?.color || "#00c8ff"}` } });
      head.createSpan({ cls: "db-name", text: i.name });
      head.createSpan({ cls: `db-status ${i.status}`, text: statusLabel(i.status) });
      if (i.description) card.createDiv({ cls: "db-desc", text: i.description });
      const tags = i.tags?.length ? ` · ${i.tags.map((t) => `#${t}`).join(" ")}` : "";
      card.createDiv({ cls: "log-entry-time", text: `${i.source || "manual"} · ${c?.name ?? "未分类"} · ${new Date(i.createdAt).toLocaleDateString()}${tags}` });
      const act = card.createDiv({ cls: "tactical-card-actions" });
      const open = act.createEl("button", { cls: "tactical-mini-btn", text: "打开来源" });
      open.addEventListener("click", () => {
        const f = i.filePath ? view.plugin.app.vault.getAbstractFileByPath(i.filePath) : null;
        if (f) void view.plugin.app.workspace.getLeaf("tab").openFile(f as any);
        else notice("来源文件不存在，条目可能是旧数据或已被移动");
      });
      const agent = act.createEl("button", { cls: "tactical-mini-btn", text: "加入 Agent" });
      agent.title = "把这条知识连同来源路径送入主控大厅";
      agent.addEventListener("click", () => sendSelectionToWorkbench(view.plugin,
        `请基于这条知识继续处理，并保留来源引用。\n来源：[[${i.filePath ?? i.name}]]\n标题：${i.name}\n标签：${(i.tags ?? []).join(", ")}\n\n${i.description ?? "（正文请打开来源笔记读取）"}`));
      const toggle = act.createEl("button", { cls: "tactical-mini-btn", text: i.status === "solved" ? "标记待办" : "标记完成" });
      toggle.addEventListener("click", async () => {
        const newStatus = i.status === "solved" ? "pending" : "solved";
        await updateCard(view.plugin, i.filePath, {
          fm: { db_name: i.name, category_id: i.categoryId, status: newStatus, tags: i.tags, source: i.source || "manual", source_ref: i.sourceRef, created_at: new Date(i.createdAt).toISOString() },
          body: i.description ?? "",
        });
        redraw();
      });
      const del = act.createEl("button", { cls: "tactical-mini-btn", text: "✕" });
      del.addEventListener("click", async () => {
        if (!(await confirmDialog(view.plugin, "删除该资产？"))) return;
        await deleteCardAt(view.plugin, i.filePath);
        redraw();
      });
    });
    updateBatch();
  }

  search.addEventListener("input", redraw);
  status.addEventListener("change", redraw);
  source.addEventListener("change", redraw);
  category.addEventListener("change", redraw);

  renderDbOverview(overviewPanel, items, sourceCounts, (sourceName) => {
    source.value = sourceName;
    showMode("search");
    redraw();
  });
  renderDbTopics(topicPanel, themeCounts, (topic) => {
    search.value = topic;
    showMode("search");
    redraw();
  });
  showMode(mode);
  redraw();
}

function renderDbOverview(parent: HTMLElement, items: DbItem[], sourceCounts: Map<string, number>, selectSource: (source: string) => void) {
  parent.empty();
  const head = parent.createDiv({ cls: "db-panel-head" });
  head.createSpan({ cls: "db-panel-title", text: "来源 × 状态" });
  head.createSpan({ cls: "db-panel-hint", text: "各渠道流入 → 待整理 → 已入知识 → 已淘汰" });
  const grid = parent.createDiv({ cls: "db-source-grid" });
  [...sourceCounts.entries()].sort((a, b) => b[1] - a[1]).forEach(([source, total]) => {
    const card = grid.createEl("button", { cls: "db-source-card" });
    card.addEventListener("click", () => selectSource(source));
    const sourceItems = items.filter((item) => (item.source || "manual") === source);
    const solved = sourceItems.filter((item) => item.status === "solved").length;
    const pending = sourceItems.filter((item) => item.status !== "solved").length;
    card.createDiv({ cls: "db-source-name", text: source });
    const track = card.createDiv({ cls: "db-source-progress" });
    track.createDiv({ cls: "db-source-progress-fill", attr: { style: `width:${Math.round(solved / Math.max(1, total) * 100)}%` } });
    card.createDiv({ cls: "db-source-stat", text: `待整理 ${pending}` });
    card.createDiv({ cls: "db-source-stat", text: `已入知识 ${solved}` });
    card.createDiv({ cls: "db-source-stat", text: `总计 ${total}` });
  });
  if (!sourceCounts.size) grid.createDiv({ cls: "db-panel-empty", text: "暂无来源数据" });
}

function renderDbTopics(parent: HTMLElement, themeCounts: Map<string, number>, selectTopic: (topic: string) => void) {
  parent.empty();
  const head = parent.createDiv({ cls: "db-panel-head" });
  head.createSpan({ cls: "db-panel-title", text: "主题图谱" });
  head.createSpan({ cls: "db-panel-hint", text: "按 tags 聚类；点击主题进入三维检索" });
  const cloud = parent.createDiv({ cls: "db-topic-cloud" });
  [...themeCounts.entries()].sort((a, b) => b[1] - a[1]).forEach(([topic, count]) => {
    const chip = cloud.createEl("button", { cls: "db-topic-chip", text: `#${topic} ${count}` });
    chip.title = "按主题检索";
    chip.addEventListener("click", () => selectTopic(topic));
  });
  if (!themeCounts.size) parent.createDiv({ cls: "db-panel-empty", text: "暂无主题标签；可在采集暂存或知识条目中补充 tags" });
}

function statusLabel(s: string): string {
  if (s === "solved") return "已解决";
  if (s === "pending") return "待跟进";
  return "其他";
}
