// 甘特时间线：任务按状态分组，里程碑按 dueDate 落位，today 定位，逾期线
import type { SpaceOSView } from "../view";
import type { ArkTask, TaskStatus } from "../types";
import { inputDialog, notice } from "../ui";
import { getToday } from "../utils";

const LABEL_WIDTH = 180;
const MS_PER_DAY = 86400000;
const GROUP_ORDER: TaskStatus[] = ["doing", "todo", "blocked", "done"];
const GROUP_LABEL: Record<TaskStatus, string> = { doing: "进行中", todo: "待执行", blocked: "阻塞中", done: "已完成" };
const PRI_COLOR: Record<string, string> = { high: "#ff5252", medium: "#ffc857", low: "#29e6a7" };
type TimelineRangeMode = "smart" | "week" | "14d" | "month" | "60d" | "all" | "custom";
type TimelineGranularity = "day" | "week" | "month";
interface TimelineUnit { startMs: number; endMs: number; label: string; }
interface TimelineWindow { startMs: number; endMs: number; units: TimelineUnit[]; granularity: TimelineGranularity; }
interface TimelineRange { mode: TimelineRangeMode; startDate?: string; endDate?: string; }

export function renderTimeline(
  view: SpaceOSView,
  mount: HTMLElement,
  listId: string | undefined,
  filter = "all",
  range: TimelineRange = { mode: "smart" },
) {
  const list = view.plugin.data.todoLists.find((l) => l.id === listId);
  if (!list) { mount.createDiv({ cls: "tactical-empty", text: "请先选择一个清单" }); return; }

  const now = new Date();
  const nowMs = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();

  const scheduled: { task: ArkTask; startMs: number; endMs: number }[] = [];
  const unscheduled: ArkTask[] = [];
  list.tasks.filter((t) => matchesFilter(t, filter)).forEach((t) => {
    const endMs = t.dueDate ? new Date(t.dueDate + "T12:00:00").getTime() : NaN;
    const startMs = t.startDate ? new Date(t.startDate + "T12:00:00").getTime() : endMs;
    if (!isNaN(endMs) && !isNaN(startMs)) scheduled.push({ task: t, startMs: Math.min(startMs, endMs), endMs: Math.max(startMs, endMs) });
    else unscheduled.push(t);
  });

  const timeline = buildTimelineWindow(range, scheduled, nowMs);
  const visibleScheduled = scheduled.filter((item) => item.endMs >= timeline.startMs && item.startMs <= timeline.endMs);
  const hiddenScheduled = scheduled.length - visibleScheduled.length;

  const shell = mount.createDiv({ cls: "tactical-schedule-card" });
  const header = shell.createDiv({ cls: "tactical-schedule-head" });
  const heading = header.createDiv({ cls: "tactical-schedule-heading" });
  heading.createSpan({ cls: "tactical-schedule-title", text: "排期矩阵" });
  heading.createSpan({ cls: "tactical-schedule-meta", text: `${shortDate(isoDate(new Date(timeline.startMs)))} → ${shortDate(isoDate(new Date(timeline.endMs)))} · 今天 ${shortDate(getToday())} · ${granularityLabel(timeline.granularity)}` });
  const controls = header.createDiv({ cls: "tactical-schedule-controls" });
  const rangeSelect = controls.createEl("select", { cls: "tactical-schedule-range", attr: { "aria-label": "时间范围" } });
  [
    ["smart", "智能范围"], ["week", "本周"], ["14d", "近14天"],
    ["month", "本月"], ["60d", "近60天"], ["all", "全部任务"], ["custom", "自定义范围…"],
  ].forEach(([value, label]) => rangeSelect.createEl("option", { value, text: label }));
  rangeSelect.value = range.mode;
  rangeSelect.addEventListener("change", () => {
    const nextMode = rangeSelect.value as TimelineRangeMode;
    if (nextMode !== "custom") {
      mount.empty();
      renderTimeline(view, mount, listId, filter, { mode: nextMode });
      return;
    }
    void chooseCustomRange(view, range, timeline).then((customRange) => {
      if (!customRange) {
        rangeSelect.value = range.mode;
        return;
      }
      mount.empty();
      renderTimeline(view, mount, listId, filter, customRange);
    });
  });
  const todayButton = controls.createEl("button", { cls: "tactical-schedule-today", text: "回到今天" });
  todayButton.title = "切换到智能范围并显示当前任务上下文";
  todayButton.addEventListener("click", () => {
    mount.empty();
    renderTimeline(view, mount, listId, filter, { mode: "smart" });
  });
  header.createSpan({ cls: "tactical-schedule-tip", text: "短范围按天 · 长范围按周/月 · 任务条可拖动改期" });
  if (hiddenScheduled > 0) {
    header.createSpan({ cls: "tactical-schedule-hidden", text: `${hiddenScheduled} 项任务不在当前窗口` });
  }
  const gantt = shell.createDiv({ cls: "gantt" });
  // 窄容器（Obsidian 侧栏）压缩任务名列，把宽度让给时间轴本身。
  const labelWidth = mount.clientWidth > 0 && mount.clientWidth < 620 ? 116 : LABEL_WIDTH;
  gantt.style.setProperty("--timeline-label-width", `${labelWidth}px`);
  gantt.style.setProperty("--timeline-units", String(timeline.units.length));

  // 刻度行
  const axis = gantt.createDiv({ cls: "gantt-axis" });
  axis.createDiv({ cls: "gantt-axis-label" });
  const axisTrack = axis.createDiv({ cls: "gantt-axis-track" });
  // 直接写入列模板，避免 Obsidian 内嵌 Chromium 对 CSS 变量 repeat() 的解析差异。
  axisTrack.style.gridTemplateColumns = `repeat(${timeline.units.length}, minmax(0, 1fr))`;
  timeline.units.forEach((unit) => {
    const date = new Date(unit.startMs);
    const weekend = timeline.granularity === "day" && (date.getDay() === 0 || date.getDay() === 6);
    const today = nowMs >= unit.startMs && nowMs < unit.endMs;
    axisTrack.createDiv({ cls: `gantt-day ${weekend ? "weekend" : ""} ${today ? "today" : ""}`, text: unit.label });
  });

  // 行区
  const rows = gantt.createDiv({ cls: "gantt-rows" });

  const buildGroup = (st: TaskStatus, items: { task: ArkTask; startMs: number; endMs: number }[]) => {
    if (items.length === 0) return;
    rows.createDiv({ cls: "gantt-group-header", text: `${GROUP_LABEL[st]} (${items.length})` });
    [...items].sort((a, b) => a.startMs - b.startMs).forEach((r) => {
      const row = rows.createDiv({ cls: "gantt-task-row" });
      const label = row.createDiv({ cls: "gantt-task-label", text: r.task.description + (r.task.recurrence?.frequency && r.task.recurrence.frequency !== "none" ? " 🔁" : ""), attr: { title: "点击编辑截止日" } });
      label.addEventListener("click", () => setDueDate(view, listId!, r.task));

      const track = row.createDiv({ cls: "gantt-task-track" });
      appendTodayLine(track, nowMs, timeline);
      const visibleStart = Math.max(r.startMs, timeline.startMs);
      const visibleEnd = Math.min(r.endMs, timeline.endMs);
      const left = timelineFraction(visibleStart, timeline) * 100;
      const right = timelineFraction(Math.min(timeline.endMs + MS_PER_DAY, visibleEnd + MS_PER_DAY), timeline) * 100;
      const width = Math.max(0.6, right - left);
      const bar = track.createDiv({ cls: `gantt-bar ${r.task.status === "done" ? "done" : ""}` });
      bar.setAttribute("style", `left:${left}%; width:${width}%; border-color:${PRI_COLOR[r.task.priority] ?? "#00e5ff"}`);
      const progress = Math.max(0, Math.min(100, r.task.progress ?? (r.task.completed ? 100 : 0)));
      if (progress > 0) bar.createDiv({ cls: "gantt-bar-progress", attr: { style: `width:${progress}%` } });
      bar.setAttr("title", `${shortDate(r.task.startDate ?? r.task.dueDate!)} → ${shortDate(r.task.dueDate!)} · ${progress}% · 拖动调整日期`);
      attachBarDrag(track, bar, timeline, unitDelta => {
        const daysDelta = shiftDateByGranularity(r.task.dueDate!, timeline.granularity, unitDelta);
        if (!daysDelta) return;
        const nextDue = daysDelta;
        const updates: Partial<ArkTask> = { dueDate: nextDue };
        if (r.task.startDate) updates.startDate = shiftDateByGranularity(r.task.startDate, timeline.granularity, unitDelta);
        void view.plugin.updateTask(listId!, r.task.id, updates)
          .then(() => view.renderCurrentPanel());
      });

      if (!r.task.completed && r.endMs < nowMs) {
        const over = track.createDiv({ cls: "gantt-overdue-line" });
        const overdueEnd = timelineFraction(Math.min(nowMs, timeline.endMs + MS_PER_DAY), timeline) * 100;
        const overdueStart = timelineFraction(Math.max(r.endMs, timeline.startMs), timeline) * 100;
        over.setAttribute("style", `left:${overdueStart}%; width:${Math.max(0, overdueEnd - overdueStart)}%`);
      }
    });
  };

  GROUP_ORDER.forEach((st) => buildGroup(st, visibleScheduled.filter((r) => r.task.status === st)));

  if (unscheduled.length) {
    rows.createDiv({ cls: "gantt-group-header", text: `未排期 (${unscheduled.length})` });
    unscheduled.forEach((t) => {
      const row = rows.createDiv({ cls: "gantt-task-row" });
      const label = row.createDiv({ cls: "gantt-task-label", text: t.description + (t.recurrence?.frequency && t.recurrence.frequency !== "none" ? " 🔁" : ""), attr: { title: "点击编辑截止日" } });
      label.addEventListener("click", () => setDueDate(view, listId!, t));
      const track = row.createDiv({ cls: "gantt-task-track" });
      appendTodayLine(track, nowMs, timeline);
      track.createDiv({ cls: "gantt-unscheduled", text: "未排期" });
    });
  }
}

function buildTimelineWindow(range: TimelineRange, scheduled: { startMs: number; endMs: number }[], nowMs: number): TimelineWindow {
  const { mode } = range;
  const today = dayMs(new Date(nowMs));
  // 默认窗口把今天前后各保留可读的上下文，远期任务由“全部任务”主动展开。
  let startMs = today - 7 * MS_PER_DAY;
  let endMs = today + 21 * MS_PER_DAY;
  if (mode === "week") { startMs = startOfWeek(today); endMs = startMs + 6 * MS_PER_DAY; }
  if (mode === "14d") { startMs = today - 6 * MS_PER_DAY; endMs = today + 7 * MS_PER_DAY; }
  if (mode === "month") { startMs = startOfMonth(today); endMs = endOfMonth(today); }
  if (mode === "60d") { startMs = today - 29 * MS_PER_DAY; endMs = today + 30 * MS_PER_DAY; }
  if (mode === "all" && scheduled.length) {
    startMs = Math.min(...scheduled.map((item) => item.startMs)) - MS_PER_DAY;
    endMs = Math.max(...scheduled.map((item) => item.endMs)) + MS_PER_DAY;
  }
  if (mode === "custom" && range.startDate && range.endDate) {
    startMs = parseIsoDay(range.startDate);
    endMs = parseIsoDay(range.endDate);
  }
  const rawDays = Math.max(1, Math.round((endMs - startMs) / MS_PER_DAY) + 1);
  const granularity: TimelineGranularity = rawDays <= 31 ? "day" : rawDays <= 180 ? "week" : "month";
  if (granularity === "week") { startMs = startOfWeek(startMs); endMs = endOfWeek(endMs); }
  if (granularity === "month") { startMs = startOfMonth(startMs); endMs = endOfMonth(endMs); }
  const units: TimelineUnit[] = [];
  if (granularity === "day") {
    for (let ms = startMs; ms <= endMs; ms += MS_PER_DAY) {
      const date = new Date(ms);
      units.push({ startMs: ms, endMs: ms + MS_PER_DAY, label: `${date.getMonth() + 1}/${date.getDate()}` });
    }
  } else if (granularity === "week") {
    for (let ms = startMs; ms <= endMs; ms += 7 * MS_PER_DAY) {
      const date = new Date(ms);
      units.push({ startMs: ms, endMs: ms + 7 * MS_PER_DAY, label: `${date.getMonth() + 1}/${date.getDate()}` });
    }
  } else {
    for (let ms = startMs; ms <= endMs;) {
      const date = new Date(ms);
      const next = new Date(date.getFullYear(), date.getMonth() + 1, 1).getTime();
      units.push({ startMs: ms, endMs: next, label: `${date.getFullYear()}年${date.getMonth() + 1}月` });
      ms = next;
    }
  }
  return { startMs, endMs, units, granularity };
}

function appendTodayLine(track: HTMLElement, nowMs: number, timeline: TimelineWindow) {
  if (nowMs < timeline.startMs || nowMs > timeline.endMs) return;
  track.createDiv({ cls: "gantt-today-line", attr: { style: `left:${timelineFraction(nowMs, timeline) * 100}%` } });
}

function timelineFraction(ms: number, timeline: TimelineWindow): number {
  const units = timeline.units;
  const endExclusive = units[units.length - 1]?.endMs ?? timeline.endMs + MS_PER_DAY;
  if (ms <= timeline.startMs) return 0;
  if (ms >= endExclusive) return 1;
  const index = units.findIndex((unit) => ms < unit.endMs);
  if (index < 0) return 1;
  const unit = units[index];
  const withinUnit = (ms - unit.startMs) / Math.max(1, unit.endMs - unit.startMs);
  return (index + Math.max(0, Math.min(1, withinUnit))) / units.length;
}

function granularityLabel(granularity: TimelineGranularity): string {
  return granularity === "day" ? "日视图" : granularity === "week" ? "周视图" : "月视图";
}

function dayMs(value: Date): number {
  return new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime();
}

function startOfWeek(ms: number): number {
  const date = new Date(ms);
  const offset = (date.getDay() + 6) % 7;
  date.setDate(date.getDate() - offset);
  return dayMs(date);
}

function endOfWeek(ms: number): number {
  return startOfWeek(ms) + 6 * MS_PER_DAY;
}

function startOfMonth(ms: number): number {
  const date = new Date(ms);
  return new Date(date.getFullYear(), date.getMonth(), 1).getTime();
}

function endOfMonth(ms: number): number {
  const date = new Date(ms);
  return new Date(date.getFullYear(), date.getMonth() + 1, 0).getTime();
}

async function chooseCustomRange(
  view: SpaceOSView,
  current: TimelineRange,
  timeline: TimelineWindow,
): Promise<TimelineRange | null> {
  const startDate = await inputDialog(view.plugin, {
    title: "自定义排期范围：开始日期",
    placeholder: "YYYY-MM-DD",
    initial: current.startDate ?? isoDate(new Date(timeline.startMs)),
  });
  if (startDate === null) return null;
  const endDate = await inputDialog(view.plugin, {
    title: "自定义排期范围：结束日期",
    placeholder: "YYYY-MM-DD",
    initial: current.endDate ?? isoDate(new Date(timeline.endMs)),
  });
  if (endDate === null) return null;
  const start = startDate.trim();
  const end = endDate.trim();
  if (!isValidIsoDay(start) || !isValidIsoDay(end)) {
    notice("日期必须使用有效的 YYYY-MM-DD 格式");
    return null;
  }
  if (parseIsoDay(start) > parseIsoDay(end)) {
    notice("结束日期不能早于开始日期");
    return null;
  }
  return { mode: "custom", startDate: start, endDate: end };
}

function parseIsoDay(value: string): number {
  return new Date(value + "T12:00:00").getTime();
}

function isValidIsoDay(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const date = new Date(value + "T12:00:00");
  return !isNaN(date.getTime()) && isoDate(date) === value;
}

function matchesFilter(task: ArkTask, filter: string): boolean {
  if (filter === "open") return task.status !== "done";
  if (filter === "blocked") return task.status === "blocked";
  if (filter === "today") return task.status !== "done" && task.dueDate === getToday();
  return true;
}

function setDueDate(view: SpaceOSView, listId: string, task: ArkTask) {
  inputDialog(view.plugin, { title: "设置到期日 (YYYY-MM-DD)", placeholder: "YYYY-MM-DD", initial: task.dueDate ?? "" }).then(async (v) => {
    if (v === null) return;
    const val = v.trim();
    await view.plugin.updateTask(listId, task.id, { dueDate: val || undefined });
    view.renderCurrentPanel();
  });
}

/** 任务条整体平移：按当前时间粒度吸附，保留任务持续时间。 */
function attachBarDrag(
  track: HTMLElement,
  bar: HTMLElement,
  timeline: TimelineWindow,
  onDrop: (unitDelta: number) => void,
) {
  let startX = 0;
  let dragging = false;
  let lastDelta = 0;
  let unitWidth = 1;
  bar.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    event.stopPropagation();
    startX = event.clientX;
    dragging = true;
    lastDelta = 0;
    unitWidth = Math.max(1, track.getBoundingClientRect().width / Math.max(1, timeline.units.length));
    bar.addClass("is-dragging");
    bar.setPointerCapture(event.pointerId);
  });
  bar.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    lastDelta = event.clientX - startX;
    bar.style.transform = `translateX(${Math.round(lastDelta / unitWidth) * unitWidth}px)`;
  });
  const finish = (event: PointerEvent) => {
    if (!dragging) return;
    dragging = false;
    bar.removeClass("is-dragging");
    bar.style.transform = "";
    if (bar.hasPointerCapture(event.pointerId)) bar.releasePointerCapture(event.pointerId);
    const unitDelta = Math.round(lastDelta / unitWidth);
    if (unitDelta) onDrop(unitDelta);
  };
  bar.addEventListener("pointerup", finish);
  bar.addEventListener("pointercancel", finish);
  bar.addEventListener("click", (event) => {
    // 拖动结束后不要误触发任务标签/日期编辑；任务条本身没有点击动作。
    event.stopPropagation();
  });
}

function shiftDateByGranularity(value: string, granularity: TimelineGranularity, units: number): string {
  const date = new Date(value + "T12:00:00");
  if (granularity === "month") date.setMonth(date.getMonth() + units);
  else date.setDate(date.getDate() + units * (granularity === "week" ? 7 : 1));
  return isoDate(date);
}

function shortDate(s: string): string {
  const d = new Date(s + "T12:00:00");
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

function isoDate(value: Date): string {
  return `${value.getFullYear()}-${String(value.getMonth() + 1).padStart(2, "0")}-${String(value.getDate()).padStart(2, "0")}`;
}
