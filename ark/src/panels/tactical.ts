import { Modal, Setting } from "obsidian";
import type { SpaceOSView } from "../view";
import type ArkOSPlugin from "../main";
import type { ArkTask, TaskStatus } from "../types";
import { inputDialog, confirmDialog, notice } from "../ui";
import { recurrenceEditor } from "../recurrenceEdit";
import { renderTimeline } from "./timeline";
import { getSkin } from "../skins";
import { planTaskSchedule, runTodayTodos, type ScheduleDraft } from "../ai-asst";
import { getToday } from "../utils";

const STATUS_ORDER: TaskStatus[] = ["todo", "doing", "blocked", "done"];
const COL_COLOR: Record<TaskStatus, string> = {
  todo: "var(--space-cyan)",
  doing: "var(--space-teal)",
  blocked: "var(--space-amber)",
  done: "var(--space-purple)",
};

/** 任务管理：显示待办清单 + 看板四列（文案随设定）。 */
export function renderTactical(view: SpaceOSView, mount: HTMLElement) {
  mount.addClass("space-tactical");
  const data = view.plugin.data;
  const skin = getSkin(data.settings.skin);

  const note = mount.createDiv({ cls: "tactical-note", text: "三视图（排期矩阵 / 优先级矩阵 / 看板）+ Agent 排期建议。排期矩阵展示任务跨度，负载条用于判断近期执行压力；所有写回都要经过 diff 确认。" });
  note.setAttr("role", "note");

  // 任务中枢：先让用户知道当前负载，再选择清单/视图/动作。
  const bar = mount.createDiv({ cls: "tactical-toolbar" });
  const titleWrap = bar.createDiv({ cls: "tactical-heading" });
  titleWrap.createDiv({ cls: "tactical-title", text: skin.tabs.tactical.title });
  titleWrap.createDiv({ cls: "tactical-subtitle", text: "把目标拆成可执行序列，交给 Agent 生成排期或手动推进。" });
  const addBtn = bar.createEl("button", { cls: "tactical-btn" });
  addBtn.setText("+ 清单");
  addBtn.addEventListener("click", () => createListModal(view));

  // 清单选择
  const listSelect = bar.createEl("select", { cls: "tactical-list-select" });
  if (data.todoLists.length === 0) {
    listSelect.createEl("option", { text: "（暂无清单，点击上方新建）" });
  } else {
    data.todoLists.forEach((l) => listSelect.createEl("option", { value: l.id, text: l.name }));
  }
  let activeListId = listSelect.value;

  const addTaskBtn = bar.createEl("button", { cls: "tactical-btn" });
  addTaskBtn.setText("+ 任务");
  addTaskBtn.addEventListener("click", async () => {
    if (!activeListId) { notice("请先选择一个清单"); return; }
    const desc = await inputDialog(view.plugin, { title: "添加任务", placeholder: "任务描述" });
    if (!desc || !desc.trim()) return;
    await view.plugin.addTodoTask(activeListId, desc.trim());
    view.renderCurrentPanel();
  });

  // F2 AI 生成今日待办（设计文档 DESIGN-AI-DEEP-INTEGRATION.md）
  const aiBtn = bar.createEl("button", { cls: "tactical-btn" });
  aiBtn.setText("🤖 今日计划");
  aiBtn.title = "AI 读近期进度 → 生成今日待办 → 落卡「今日」清单";
  aiBtn.addEventListener("click", async () => {
    await runTodayTodos(view.plugin);
    view.renderCurrentPanel();
  });

  const agentBtn = bar.createEl("button", { cls: "tactical-btn tactical-btn-agent" });
  agentBtn.setText("▣ 交给 Agent 排期");
  agentBtn.title = "生成结构化排期建议，预览差异后再写回任务";
  agentBtn.addEventListener("click", async () => {
    const list = data.todoLists.find((l) => l.id === listSelect.value);
    const openTasks = list?.tasks.filter((t) => t.status !== "done") ?? [];
    if (!list || !openTasks.length) { notice("当前清单没有可排期的未完成任务"); return; }
    agentBtn.setText("▣ 排期分析中…");
    agentBtn.disabled = true;
    try {
      // 传入完整清单以便 Agent 引用已完成依赖，但只允许未完成任务产生写回变更。
      const drafts = (await planTaskSchedule(view.plugin, list.name, list.tasks.map((task) => ({
        id: task.id,
        description: task.description,
        start_date: task.startDate,
        due_date: task.dueDate,
        effort: task.effort,
        dependency: task.dependency ?? [],
        progress: task.progress ?? 0,
        priority: task.priority,
        status: task.status,
      })))).filter((draft) => openTasks.some((task) => task.id === draft.task_id));
      const changes = buildScheduleChanges(list.tasks, drafts);
      if (!changes.length) { notice("Agent 未提出需要写回的排期调整"); return; }
      const accepted = await scheduleDiffModal(view.plugin, changes);
      if (!accepted?.length) return;
      await applyScheduleChanges(view, list.id, accepted);
      notice(`已确认并写回 ${accepted.length} 条排期调整`);
      view.renderCurrentPanel();
    } catch (e: any) {
      notice(`Agent 排期失败：${String(e?.message ?? e)}`, 5000);
    } finally {
      agentBtn.setText("▣ 交给 Agent 排期");
      agentBtn.disabled = false;
    }
  });

  const summary = mount.createDiv({ cls: "tactical-summary" });
  const filter = mount.createEl("select", { cls: "tactical-filter" });
  filter.createEl("option", { value: "all", text: "全部任务" });
  filter.createEl("option", { value: "today", text: "今天到期" });
  filter.createEl("option", { value: "open", text: "未完成" });
  filter.createEl("option", { value: "blocked", text: "已阻塞" });
  // 默认必须是"全部任务"：看板有独立的"已完成"列，若默认按"未完成"筛，
  // 该列永远是空的，任务一拖到已完成就"凭空消失"（真机验收报出的"自动删除任务"，OPT-191）。
  filter.value = "all";
  let activeFilter = "all";
  const layout = mount.createDiv({ cls: "tactical-layout" });
  const main = layout.createDiv({ cls: "tactical-main" });
  const board = main.createDiv({ cls: "tactical-body" });
  const load = main.createDiv({ cls: "tactical-load" });
  const backlog = layout.createDiv({ cls: "tactical-backlog" });

  let mode: "board" | "timeline" | "priority" = "board";
  const viewBtns = bar.createDiv({ cls: "log-toolbar" });
  const mk = (m: "board" | "timeline" | "priority", label: string) => {
    const b = viewBtns.createEl("button", { cls: `log-type-tab${mode === m ? " active" : ""}`, text: label });
    b.addEventListener("click", () => { mode = m; viewBtns.querySelectorAll(".log-type-tab").forEach((el) => el.toggleClass("active", (el as HTMLElement).dataset.m === m)); rerender(); });
    b.dataset.m = m;
    return b;
  };
  mk("board", "◧ 看板");
  mk("timeline", "☷ 时间线");
  mk("priority", "◇ 优先级");

  function rerender() {
    board.empty();
    backlog.empty();
    load.empty();
    board.toggleClass("tactical-board", mode === "board");
    const list = data.todoLists.find((l) => l.id === activeListId);
    renderSummary(summary, list?.tasks ?? []);
    if (mode === "board") renderBoard(view, board, activeListId, activeFilter);
    else if (mode === "timeline") renderTimeline(view, board, activeListId, activeFilter);
    else renderPriority(view, board, activeListId, activeFilter);
    renderBacklog(view, backlog, activeListId, activeFilter);
    renderLoad(load, list?.tasks ?? []);
  }
  rerender();

  listSelect.addEventListener("change", () => {
    activeListId = listSelect.value;
    rerender();
  });
  filter.addEventListener("change", () => { activeFilter = filter.value; rerender(); });
}

function renderSummary(parent: HTMLElement, tasks: ArkTask[]) {
  parent.empty();
  const today = getToday();
  const metrics: [string, string, string][] = [
    ["未完成", String(tasks.filter((t) => t.status !== "done").length), "open"],
    ["进行中", String(tasks.filter((t) => t.status === "doing").length), "doing"],
    ["阻塞", String(tasks.filter((t) => t.status === "blocked").length), "blocked"],
    ["今天", String(tasks.filter((t) => t.status !== "done" && t.dueDate === today).length), "today"],
  ];
  metrics.forEach(([label, value, tone]) => {
    const item = parent.createDiv({ cls: `tactical-summary-item ${tone}` });
    item.createDiv({ cls: "tactical-summary-value", text: value });
    item.createDiv({ cls: "tactical-summary-label", text: label });
  });
}

function renderBoard(view: SpaceOSView, parent: HTMLElement, listId: string | undefined, filter = "all") {
  const data = view.plugin.data;
  const list = data.todoLists.find((l) => l.id === listId);
  const skin = getSkin(data.settings.skin);
  if (!list) {
    parent.createDiv({ cls: "tactical-empty", text: skin.panel.taskEmpty });
    return;
  }

  const onStatusDrop = async (status: TaskStatus) => {
    await view.plugin.setTaskStatus(list.id, dragState.taskId, status);
    view.renderCurrentPanel();
  };

  STATUS_ORDER.forEach((st) => {
    const col = parent.createDiv({ cls: "tactical-col", attr: { "data-status": st } });
    col.addEventListener("dragover", (e) => { e.preventDefault(); });
    col.addEventListener("drop", (e) => { e.preventDefault(); void onStatusDrop(st); });
    const meta = skin.panel.taskCols[STATUS_ORDER.indexOf(st)];
    const color = COL_COLOR[st];
    const head = col.createDiv({ cls: "tactical-col-head", text: meta });
    head.setAttribute("style", `border-color:${color}`);
    const body = col.createDiv({ cls: "tactical-col-body" });
    const tasks = list.tasks.filter((t) => t.status === st && matchesFilter(t, filter));
    if (tasks.length === 0) {
      body.createDiv({ cls: "tactical-col-empty", text: "空" });
    } else {
      tasks.forEach((t) => body.appendChild(renderCard(view, list.id, t, st, onStatusDrop)));
    }
  });
}

function matchesFilter(task: ArkTask, filter: string): boolean {
  if (filter === "open") return task.status !== "done";
  if (filter === "blocked") return task.status === "blocked";
  if (filter === "today") return task.status !== "done" && task.dueDate === getToday();
  return true;
}

function renderPriority(view: SpaceOSView, parent: HTMLElement, listId: string | undefined, filter: string) {
  const list = view.plugin.data.todoLists.find((l) => l.id === listId);
  if (!list) { parent.createDiv({ cls: "tactical-empty", text: "请先选择一个清单" }); return; }
  const today = new Date();
  const urgent = (task: ArkTask) => {
    if (!task.dueDate) return false;
    const due = new Date(task.dueDate + "T12:00:00").getTime();
    return due <= today.getTime() + 2 * 86400000;
  };
  const groups: { title: string; key: string; test: (t: ArkTask) => boolean }[] = [
    { title: "立即处理 · 高优 / 临期", key: "q1", test: (t) => t.priority === "high" && urgent(t) },
    { title: "重要但不紧急", key: "q2", test: (t) => t.priority === "high" && !urgent(t) },
    { title: "紧急但可委派", key: "q3", test: (t) => t.priority !== "high" && urgent(t) },
    { title: "低优先级 / 待确认", key: "q4", test: (t) => t.priority !== "high" && !urgent(t) },
  ];
  const grid = parent.createDiv({ cls: "tactical-priority" });
  groups.forEach((group) => {
    const cell = grid.createDiv({ cls: `tactical-priority-cell ${group.key}` });
    cell.createDiv({ cls: "tactical-priority-head", text: group.title });
    const tasks = list.tasks.filter((t) => matchesFilter(t, filter) && group.test(t));
    if (!tasks.length) cell.createDiv({ cls: "tactical-col-empty", text: "暂无任务" });
    tasks.forEach((task) => cell.appendChild(renderCard(view, list.id, task, task.status, (status) => {
      void view.plugin.setTaskStatus(list.id, task.id, status).then(() => view.renderCurrentPanel());
    })));
  });
}

function renderBacklog(view: SpaceOSView, parent: HTMLElement, listId: string | undefined, filter: string) {
  const list = view.plugin.data.todoLists.find((candidate) => candidate.id === listId);
  parent.createDiv({ cls: "tactical-backlog-head", text: "待排池" });
  parent.createDiv({ cls: "tactical-backlog-hint", text: "未设置开始/截止日期的任务" });
  if (!list) {
    parent.createDiv({ cls: "tactical-col-empty", text: "请先选择清单" });
    return;
  }
  const tasks = list.tasks.filter((task) => task.status !== "done" && !task.dueDate && matchesFilter(task, filter));
  if (!tasks.length) {
    parent.createDiv({ cls: "tactical-col-empty", text: "暂无未排期任务" });
    return;
  }
  tasks.forEach((task) => {
    const row = parent.createDiv({ cls: "tactical-backlog-item" });
    row.createDiv({ cls: "tactical-backlog-item-title", text: task.description });
    row.createDiv({ cls: "tactical-backlog-item-meta", text: `${topLabel(task.priority)} · ${task.status === "blocked" ? "受阻" : "待执行"}` });
    const button = row.createEl("button", { cls: "tactical-mini-btn", text: "排期" });
    button.title = "快速排到今天、明天或本周；需要时再打开高级设置";
    button.addEventListener("click", () => { openSchedulePicker(view, list.id, task); });
  });
}

function renderLoad(parent: HTMLElement, tasks: ArkTask[]) {
  const title = parent.createDiv({ cls: "tactical-load-title" });
  title.createSpan({ cls: "tactical-load-label", text: "负载条（执行压力）" });
  title.createSpan({ cls: "tactical-load-hint", text: "按任务工时累计，未估工时按 1 小时计" });
  const start = dateAtLocal(getToday()) - 4 * 86400000;
  const days = Array.from({ length: 14 }, (_, index) => new Date(start + index * 86400000));
  const values = days.map((day) => {
    const key = isoDate(day);
    return tasks.reduce((sum, task) => {
      if (task.status === "done" || !task.dueDate || task.dueDate !== key) return sum;
      return sum + (task.effort && task.effort > 0 ? task.effort : 1);
    }, 0);
  });
  const max = Math.max(1, ...values);
  const chart = parent.createDiv({ cls: "tactical-load-chart" });
  days.forEach((day, index) => {
    const cell = chart.createDiv({ cls: `tactical-load-cell${isoDate(day) === getToday() ? " today" : ""}` });
    cell.createDiv({ cls: "tactical-load-day", text: `${day.getMonth() + 1}/${day.getDate()}` });
    const track = cell.createDiv({ cls: "tactical-load-track" });
    const bar = track.createDiv({ cls: "tactical-load-bar" });
    bar.setAttr("style", `height:${Math.max(values[index] ? 10 : 3, Math.round(values[index] / max * 100))}%`);
    bar.setAttr("title", `${isoDate(day)} · ${values[index]} 小时负载`);
    cell.createDiv({ cls: "tactical-load-value", text: values[index] ? String(values[index]) : "0" });
  });
}

function dateAtLocal(value: string): number {
  const [year, month, day] = value.split("-").map(Number);
  return new Date(year, month - 1, day).getTime();
}

function isoDate(value: Date): string {
  return `${value.getFullYear()}-${String(value.getMonth() + 1).padStart(2, "0")}-${String(value.getDate()).padStart(2, "0")}`;
}

const dragState: { taskId: string; fromStatus: string } = { taskId: "", fromStatus: "" };

function renderCard(view: SpaceOSView, listId: string, task: ArkTask, st: TaskStatus, onStatusDrop: (s: TaskStatus) => void): HTMLElement {
  const card = document.createElement("div");
  card.addClass("tactical-card");
  card.setAttribute("data-task-id", task.id);
  card.setAttribute("draggable", "true");
  card.addEventListener("dragstart", (e) => {
    dragState.taskId = task.id;
    dragState.fromStatus = st;
    e.dataTransfer?.setData("text/plain", task.id);
    if (e.dataTransfer) e.dataTransfer.effectAllowed = "move";
  });

  const title = document.createElement("div");
  title.addClass("tactical-card-title");
  title.setText(task.description);
  card.appendChild(title);

  const meta = document.createElement("div");
  meta.addClass("tactical-card-meta");
  const range = task.startDate && task.dueDate
    ? `${task.startDate} → ${task.dueDate}`
    : task.dueDate ? `截止 ${task.dueDate}` : "未排期";
  const ownerList = view.plugin.data.todoLists.find((l) => l.id === listId);
  const dependencies = task.dependency ?? [];
  const pendingDeps = dependencies.filter((id) => ownerList?.tasks.find((candidate) => candidate.id === id)?.status !== "done");
  const dependencyLabel = dependencies.length
    ? ` · 依赖 ${dependencies.length}${pendingDeps.length ? `（${pendingDeps.length} 未完成）` : ""}`
    : "";
  meta.setText(`${task.priority ? topLabel(task.priority) + " · " : ""}${range}${task.effort ? ` · ${task.effort}h` : ""}${task.progress ? ` · ${task.progress}%` : ""}${dependencyLabel}`);
  card.appendChild(meta);

  const actions = document.createElement("div");
  actions.addClass("tactical-card-actions");

  const repeatBtn = document.createElement("button");
  repeatBtn.addClass("tactical-mini-btn");
  repeatBtn.setText(task.recurrence?.frequency && task.recurrence.frequency !== "none" ? "🔁" : "◷");
  repeatBtn.title = "设置重复";
  repeatBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    const rec = await recurrenceEditor(view.plugin, task.recurrence);
    await view.plugin.updateTask(listId, task.id, { recurrence: rec });
    view.renderCurrentPanel();
  });
  actions.appendChild(repeatBtn);

  const scheduleBtn = document.createElement("button");
  scheduleBtn.addClass("tactical-mini-btn");
  scheduleBtn.setText("排期");
  scheduleBtn.title = "快速排期；更多字段在高级设置中编辑";
  scheduleBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    openSchedulePicker(view, listId, task);
  });
  actions.appendChild(scheduleBtn);

  const editBtn = document.createElement("button");
  editBtn.addClass("tactical-mini-btn");
  editBtn.setText("✎");
  editBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    const desc = await inputDialog(view.plugin, { title: "编辑任务", placeholder: "任务描述", initial: task.description });
    if (desc && desc.trim()) {
      await view.plugin.updateTask(listId, task.id, { description: desc.trim() });
      view.renderCurrentPanel();
    }
  });
  actions.appendChild(editBtn);

  const doneBtn = document.createElement("button");
  doneBtn.addClass("tactical-mini-btn");
  doneBtn.setText("✓");
  doneBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    await view.plugin.flipTask(listId, task.id);
    view.renderCurrentPanel();
  });
  actions.appendChild(doneBtn);
  if (st !== "blocked") {
    const blockBtn = document.createElement("button");
    blockBtn.addClass("tactical-mini-btn");
    blockBtn.setText("⊘");
    blockBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      await view.plugin.setTaskStatus(listId, task.id, "blocked");
      view.renderCurrentPanel();
    });
    actions.appendChild(blockBtn);
  }
  const delBtn = document.createElement("button");
  delBtn.addClass("tactical-mini-btn");
  delBtn.setText("✕");
  delBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    if (!(await confirmDialog(view.plugin, "删除该任务？"))) return;
    await view.plugin.deleteTask(listId, task.id);
    view.renderCurrentPanel();
  });
  actions.appendChild(delBtn);

  card.appendChild(actions);
  card.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); });
  card.addEventListener("drop", (e) => { e.preventDefault(); e.stopPropagation(); void onStatusDrop(st); });
  return card;
}

async function editSchedule(view: SpaceOSView, listId: string, task: ArkTask) {
  const start = await inputDialog(view.plugin, { title: "开始日期 (YYYY-MM-DD)", placeholder: "留空表示仅设置截止日期", initial: task.startDate ?? "" });
  if (start === null) return;
  const due = await inputDialog(view.plugin, { title: "截止日期 (YYYY-MM-DD)", placeholder: "YYYY-MM-DD", initial: task.dueDate ?? "" });
  if (due === null) return;
  const effort = await inputDialog(view.plugin, { title: "预估工时", placeholder: "小时，可留空", initial: task.effort ? String(task.effort) : "" });
  if (effort === null) return;
  const progress = await inputDialog(view.plugin, { title: "当前进度", placeholder: "0-100", initial: String(task.progress ?? (task.completed ? 100 : 0)) });
  if (progress === null) return;
  const list = view.plugin.data.todoLists.find((l) => l.id === listId);
  const dependencyInitial = (task.dependency ?? [])
    .map((id) => list?.tasks.find((candidate) => candidate.id === id)?.description ?? id)
    .join(", ");
  const dependencyText = await inputDialog(view.plugin, {
    title: "依赖任务（可选）",
    placeholder: "输入任务描述或 ID，逗号分隔",
    initial: dependencyInitial,
  });
  if (dependencyText === null) return;
  const dependency = dependencyText.split(/[,，、\n]/)
    .map((token) => token.trim())
    .filter(Boolean)
    .map((token) => list?.tasks.find((candidate) => candidate.id === token || candidate.description === token)?.id ?? token)
    .filter((id) => id !== task.id);
  const startValue = start.trim();
  const dueValue = due.trim();
  if (!validDate(startValue) || !validDate(dueValue)) {
    notice("日期格式必须为 YYYY-MM-DD");
    return;
  }
  if (validDate(startValue) && validDate(dueValue) && startValue && dueValue && startValue > dueValue) {
    notice("开始日期不能晚于截止日期");
    return;
  }
  await view.plugin.updateTask(listId, task.id, {
    startDate: validDate(startValue) ? startValue : undefined,
    dueDate: validDate(dueValue) ? dueValue : undefined,
    effort: effort.trim() ? Math.max(0, Number(effort) || 0) : undefined,
    progress: Math.max(0, Math.min(100, Number(progress) || 0)),
    dependency,
  });
  view.renderCurrentPanel();
}

/**
 * 常用排期走一键预设，避免为一个截止日期连续填写五个输入框。
 * 工时、进度和依赖只在高级设置中修改，快捷操作只写日期字段。
 */
function openSchedulePicker(view: SpaceOSView, listId: string, task: ArkTask) {
  const modal = new Modal(view.plugin.app);
  modal.titleEl.setText("快速排期");
  const body = modal.contentEl;
  body.addClass("schedule-quick-modal");
  body.createDiv({ cls: "schedule-quick-task", text: task.description });
  body.createDiv({ cls: "schedule-quick-hint", text: "快捷排期只设置开始日和截止日，不改变工时、进度或依赖。" });

  const presets = body.createDiv({ cls: "schedule-quick-grid" });
  const options: Array<[string, string]> = [
    ["今天", getToday()],
    ["明天", shiftLocalDate(getToday(), 1)],
    ["本周末", endOfWeek(getToday())],
    ["下周一", nextMonday(getToday())],
  ];
  options.forEach(([label, value]) => {
    const button = presets.createEl("button", { cls: "schedule-quick-btn" });
    button.createSpan({ cls: "schedule-quick-btn-label", text: label });
    button.createSpan({ cls: "schedule-quick-btn-date", text: value });
    button.addEventListener("click", () => {
      modal.close();
      void applyQuickSchedule(view, listId, task, value);
    });
  });

  const custom = new Setting(body)
    .setName("自定义截止日")
    .setDesc("开始日会保留已有值；未排期任务从截止日开始")
    .addText((text) => {
      text.inputEl.type = "date";
      text.setValue(task.dueDate ?? getToday());
      text.inputEl.setAttr("aria-label", "自定义截止日");
      text.inputEl.addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        const value = text.getValue().trim();
        if (!validDate(value) || !value) { notice("请选择有效的截止日期"); return; }
        modal.close();
        void applyQuickSchedule(view, listId, task, value);
      });
    });
  custom.addButton((button) => button.setButtonText("应用").setCta().onClick(() => {
    const input = custom.controlEl.querySelector("input") as HTMLInputElement | null;
    const value = input?.value.trim() ?? "";
    if (!validDate(value) || !value) { notice("请选择有效的截止日期"); return; }
    modal.close();
    void applyQuickSchedule(view, listId, task, value);
  }));

  new Setting(body)
    .addButton((button) => button.setButtonText("清除日期").onClick(() => {
      modal.close();
      void clearSchedule(view, listId, task);
    }))
    .addButton((button) => button.setButtonText("高级设置").onClick(() => {
      modal.close();
      void editSchedule(view, listId, task);
    }))
    .addButton((button) => button.setButtonText("取消").onClick(() => modal.close()));
  modal.open();
}

async function applyQuickSchedule(view: SpaceOSView, listId: string, task: ArkTask, dueDate: string) {
  const startDate = task.startDate && task.startDate <= dueDate ? task.startDate : dueDate;
  await view.plugin.updateTask(listId, task.id, { startDate, dueDate });
  view.renderCurrentPanel();
}

async function clearSchedule(view: SpaceOSView, listId: string, task: ArkTask) {
  await view.plugin.updateTask(listId, task.id, { startDate: undefined, dueDate: undefined });
  view.renderCurrentPanel();
}

function shiftLocalDate(value: string, days: number): string {
  const date = new Date(`${value}T12:00:00`);
  date.setDate(date.getDate() + days);
  return isoDate(date);
}

function endOfWeek(value: string): string {
  const date = new Date(`${value}T12:00:00`);
  const daysUntilSunday = (7 - date.getDay()) % 7;
  date.setDate(date.getDate() + daysUntilSunday);
  return isoDate(date);
}

function nextMonday(value: string): string {
  const date = new Date(`${value}T12:00:00`);
  const daysUntilNextMonday = date.getDay() === 0 ? 1 : 8 - date.getDay();
  date.setDate(date.getDate() + daysUntilNextMonday);
  return isoDate(date);
}

function validDate(value: string): boolean {
  return !value || /^\d{4}-\d{2}-\d{2}$/.test(value);
}

type SchedulePatch = Pick<ArkTask, "startDate" | "dueDate" | "effort" | "dependency" | "progress">;

interface ScheduleChange {
  task: ArkTask;
  patch: Partial<SchedulePatch>;
  before: string;
  after: string;
  reason: string;
}

function buildScheduleChanges(tasks: ArkTask[], drafts: ScheduleDraft[]): ScheduleChange[] {
  const names = new Map(tasks.map((task) => [task.id, task.description]));
  const changes: ScheduleChange[] = [];
  drafts.forEach((draft) => {
    const task = tasks.find((candidate) => candidate.id === draft.task_id);
    if (!task) return;
    const patch: Partial<SchedulePatch> = {};
    if (draft.start_date !== undefined) patch.startDate = draft.start_date ?? undefined;
    if (draft.due_date !== undefined) patch.dueDate = draft.due_date ?? undefined;
    if (draft.effort !== undefined) patch.effort = draft.effort ?? undefined;
    if (draft.dependency !== undefined) patch.dependency = [...draft.dependency];
    if (draft.progress !== undefined) patch.progress = draft.progress ?? undefined;
    const effective = { ...task, ...patch };
    if (effective.startDate && effective.dueDate && effective.startDate > effective.dueDate) {
      throw new Error(`任务“${task.description}”的建议开始日期晚于截止日期`);
    }
    const before = scheduleText(task, names);
    const after = scheduleText(effective, names);
    if (before === after) return;
    changes.push({ task, patch, before, after, reason: draft.reason || "Agent 未提供调整理由" });
  });
  return changes;
}

function scheduleText(task: Pick<ArkTask, "startDate" | "dueDate" | "effort" | "dependency" | "progress">, names: Map<string, string>): string {
  const deps = (task.dependency ?? []).map((id) => names.get(id) ?? id);
  return [
    `${task.startDate ?? "未设开始"} → ${task.dueDate ?? "未设截止"}`,
    task.effort != null ? `${task.effort}h` : "未估工时",
    `${task.progress ?? 0}%`,
    deps.length ? `依赖：${deps.join("、")}` : "无依赖",
  ].join(" · ");
}

function scheduleDiffModal(plugin: ArkOSPlugin, changes: ScheduleChange[]): Promise<ScheduleChange[] | null> {
  return new Promise((resolve) => {
    const modal = new Modal(plugin.app);
    modal.titleEl.setText("Agent 排期建议 · 写回确认");
    const body = modal.contentEl;
    body.addClass("schedule-diff-modal");
    body.createDiv({ cls: "schedule-diff-hint", text: "仅勾选项会写回任务 Markdown；取消不会修改任何任务。" });
    const selected = new Set(changes);
    changes.forEach((change) => {
      const row = body.createDiv({ cls: "schedule-diff-row" });
      const check = row.createEl("input", { cls: "schedule-diff-check", attr: { type: "checkbox" } }) as HTMLInputElement;
      check.checked = true;
      check.addEventListener("change", () => check.checked ? selected.add(change) : selected.delete(change));
      const content = row.createDiv({ cls: "schedule-diff-content" });
      content.createDiv({ cls: "schedule-diff-title", text: change.task.description });
      const values = content.createDiv({ cls: "schedule-diff-values" });
      values.createDiv({ cls: "schedule-diff-before", text: change.before });
      values.createDiv({ cls: "schedule-diff-arrow", text: "↓" });
      values.createDiv({ cls: "schedule-diff-after", text: change.after });
      content.createDiv({ cls: "schedule-diff-reason", text: change.reason });
    });
    new Setting(body)
      .addButton((button) => button.setButtonText("取消").onClick(() => { modal.close(); resolve(null); }))
      .addButton((button) => button.setButtonText("确认写回所选项").setCta().onClick(() => {
        const accepted = changes.filter((change) => selected.has(change));
        modal.close();
        resolve(accepted);
      }));
    modal.open();
  });
}

async function applyScheduleChanges(view: SpaceOSView, listId: string, changes: ScheduleChange[]): Promise<void> {
  validateScheduleGraph(view, listId, changes);
  const applied: { task: ArkTask; before: Partial<SchedulePatch> }[] = [];
  try {
    for (const change of changes) {
      const before = {
        startDate: change.task.startDate,
        dueDate: change.task.dueDate,
        effort: change.task.effort,
        dependency: [...(change.task.dependency ?? [])],
        progress: change.task.progress,
      };
      applied.push({ task: change.task, before });
      await view.plugin.updateTask(listId, change.task.id, change.patch);
    }
  } catch (error) {
    let rollbackError: unknown = null;
    for (const entry of applied.reverse()) {
      try { await view.plugin.updateTask(listId, entry.task.id, entry.before); }
      catch (rollback) { rollbackError = rollback; }
    }
    if (rollbackError) throw new Error(`排期写回失败，且回滚未完全成功：${String(rollbackError)}`);
    throw error;
  }
}

function validateScheduleGraph(view: SpaceOSView, listId: string, changes: ScheduleChange[]) {
  const list = view.plugin.data.todoLists.find((candidate) => candidate.id === listId);
  if (!list) throw new Error("任务清单已不存在");
  const changed = new Map(changes.map((change) => [change.task.id, change.patch]));
  const graph = new Map<string, string[]>();
  list.tasks.forEach((task) => {
    const deps = changed.get(task.id)?.dependency ?? task.dependency ?? [];
    graph.set(task.id, deps);
  });
  const visiting = new Set<string>();
  const visited = new Set<string>();
  const visit = (id: string) => {
    if (visiting.has(id)) throw new Error("排期建议形成循环依赖，已拒绝写回");
    if (visited.has(id)) return;
    visiting.add(id);
    (graph.get(id) ?? []).forEach((dependency) => { if (graph.has(dependency)) visit(dependency); });
    visiting.delete(id);
    visited.add(id);
  };
  graph.forEach((_, id) => visit(id));
}

function topLabel(p: string): string {
  return p === "high" ? "🔺高优" : p === "low" ? "▽低优" : "中";
}

function createListModal(view: SpaceOSView) {
  inputDialog(view.plugin, { title: "新建清单", placeholder: "清单名称" }).then((name) => {
    if (name && name.trim()) {
      view.plugin.createTodoList(name.trim().slice(0, 30));
      view.renderCurrentPanel();
    }
  });
}
