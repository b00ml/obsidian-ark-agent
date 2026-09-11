import { App } from "obsidian";
import type { SpaceOSView } from "../view";
import type ArkOSPlugin from "../main";
import { fetchAgentRuns, fetchAgentRunDetail, probeAgent, type AgentRunSummary } from "../ai";
import { getToday } from "../utils";
import { openSearch } from "../search";
import { serveStart } from "../serve-control";

const INBOX_DIR = "Inbox";
const DAY_MS = 86400000;

/**
 * 运行健康：只回答三个问题：Agent 是否可用、最近一次执行为何结束、当前是否有需要处理的积压。
 * Vault 统计保留为诊断依据，不再和热力图拼成装饰性首页。
 */
export function renderDashboard(view: SpaceOSView, mount: HTMLElement) {
  mount.addClass("ark-health-host");
  const heading = mount.createDiv({ cls: "ark-health-heading" });
  heading.createDiv({ cls: "tactical-title", text: "运行健康" });
  heading.createDiv({ cls: "tactical-subtitle", text: "只展示会影响 Agent 与知识流转的状态，并提供下一步处理入口。" });

  const overview = mount.createDiv({ cls: "ark-health-overview" });
  const agentState = overview.createDiv({ cls: "ark-health-state pending" });
  agentState.createDiv({ cls: "ark-health-state-label", text: "AGENT" });
  agentState.createDiv({ cls: "ark-health-state-value", text: "检测中…" });
  const inbox = inboxStats(view.plugin.app);
  renderState(overview, "INBOX", inbox.count ? `${inbox.count} 条积压` : "已清空", inbox.count ? "warn" : "ok");
  const tasks = taskStats(view.plugin);
  renderState(overview, "逾期", tasks.overdue ? `${tasks.overdue} 条` : "无", tasks.overdue ? "warn" : "ok");
  const today = todayTaskStats(view.plugin);
  renderState(overview, "今日完成率", today.total ? `${today.done}/${today.total}` : "无计划", today.total && today.done < today.total ? "warn" : "ok");
  const errorState = renderState(overview, "AGENT 错误率", "读取中…", "pending");
  const health = healthStats(view.plugin.app);
  renderState(overview, "VAULT", `${health.score} 分`, health.score < 70 ? "warn" : "ok");

  const grid = mount.createDiv({ cls: "ark-health-grid" });
  renderRunsCard(view, grid);
  renderAttentionCard(view, grid, inbox, tasks, health);
  renderActivityCard(view, mount);

  void probeAgent(view.plugin.data.settings).then((result) => {
    agentState.removeClass("pending");
    agentState.addClass(result.ok ? "ok" : "warn");
    const value = agentState.querySelector<HTMLElement>(".ark-health-state-value");
    value?.setText(result.ok ? `在线${result.version ? ` · ${result.version}` : ""}` : "不可用");
    agentState.setAttr("title", result.ok ? "Agent 内核健康检查通过" : (result.error ?? "Agent 未连接"));
    if (!result.ok && view.plugin.data.settings.agentProvider === "agentlab") {
      const start = agentState.createEl("button", { cls: "ark-health-start", text: "启动服务" });
      start.addEventListener("click", async () => {
        start.disabled = true;
        start.setText("启动中…");
        try { await serveStart(view.plugin.data.settings); } finally { view.renderCurrentPanel(); }
      });
    }
  });
  void fetchAgentRuns(view.plugin.data.settings, 100).then((runs) => {
    if (runs === null) {
      updateState(errorState, "不可用", "warn");
      return;
    }
    const errors = runs.filter((run) => !!run.error).length;
    const rate = runs.length ? `${(errors * 100 / runs.length).toFixed(1)}%` : "无运行";
    updateState(errorState, rate, errors ? (errors / runs.length > 0.1 ? "warn" : "ok") : "ok");
  });
}

function renderState(parent: HTMLElement, label: string, value: string, tone: string): HTMLElement {
  const item = parent.createDiv({ cls: `ark-health-state ${tone}` });
  item.createDiv({ cls: "ark-health-state-label", text: label });
  item.createDiv({ cls: "ark-health-state-value", text: value });
  updateState(item, value, tone);
  return item;
}

function updateState(item: HTMLElement, value: string, tone: string): void {
  ["ok", "warn", "pending"].forEach((name) => item.removeClass(name));
  item.addClass(tone);
  item.querySelector<HTMLElement>(".ark-health-state-value")?.setText(value);
  const label = item.querySelector<HTMLElement>(".ark-health-state-label")?.textContent ?? "";
  const hint = stateHint(label, tone);
  if (hint) item.setAttr("title", hint);
}

function stateHint(label: string, tone: string): string {
  const hints: Record<string, string> = {
    INBOX: "有积压时打开采集暂存，按来源和状态批量交给 Agent。",
    任务: "先处理阻塞任务，再在时序调度矩阵补齐日期和依赖。",
    逾期: "打开时序调度矩阵查看逾期条目并重新排期。",
    今日完成率: "完成率低时检查今日清单是否拆分过大或缺少排期。",
    "AGENT 错误率": "错误率来自最近 100 次 Agent run；升高时展开最近执行查看脱敏诊断。",
    VAULT: "健康分受死链和缺失 frontmatter 影响，打开搜索定位问题。",
  };
  if (label === "AGENT") return tone === "ok" ? "Agent 探活通过，可在主控大厅执行任务。" : "Agent 未就绪，检查服务状态或启动 agentlab serve。";
  return hints[label] ?? "";
}

function renderRunsCard(view: SpaceOSView, grid: HTMLElement) {
  const card = grid.createDiv({ cls: "ark-health-card ark-health-runs" });
  const head = card.createDiv({ cls: "ark-health-card-head" });
  head.createDiv({ cls: "ark-health-card-title", text: "最近执行" });
  head.createDiv({ cls: "ark-health-card-meta", text: "用于定位失败、取消和工具调用异常" });
  const body = card.createDiv({ cls: "ark-health-runs-body", text: "读取运行记录…" });
  void fetchAgentRuns(view.plugin.data.settings, 8).then((runs) => {
    body.empty();
    if (runs === null) { body.setText("当前 Agent 后端不提供运行记录"); body.addClass("muted"); return; }
    if (!runs.length) { body.setText("暂无执行记录"); body.addClass("muted"); return; }
    const table = body.createEl("table", { cls: "ark-health-table" });
    const row = table.createEl("tr");
    ["时间", "输入", "结果", "工具", "诊断"].forEach((h) => row.createEl("th", { text: h }));
    runs.forEach((r) => renderRunRow(view, table, r));
  });
}

function renderRunRow(view: SpaceOSView, table: HTMLTableElement, run: AgentRunSummary) {
  const row = table.createEl("tr");
  row.createEl("td", { text: (run.time || "").slice(11, 16) || "—" });
  const input = row.createEl("td", { cls: "ark-health-input", text: run.input || "（无摘要）" });
  input.setAttr("title", run.input || "");
  const reason = run.error ? "失败" : (run.cancel_reason ? "已取消" : (run.stop_reason || "完成"));
  row.createEl("td", { cls: run.error ? "bad" : run.cancel_reason ? "warn" : "ok", text: reason });
  row.createEl("td", { text: String(run.tools ?? 0) });
  const detail = row.createEl("td");
  const details = detail.createEl("details");
  details.createEl("summary", { text: "查看" });
  const content = details.createDiv({ cls: "ark-health-detail", text: "加载中…" });
  details.addEventListener("toggle", () => {
    if (!details.open || details.dataset.loaded === "1") return;
    details.dataset.loaded = "1";
    void fetchAgentRunDetail(view.plugin.data.settings, run.trace_id).then((d) => {
      if (!d) { content.setText("诊断不可用"); return; }
      const r = d.run || {};
      content.setText([
        `run ${r.run_id || "-"}`,
        `session ${r.session_id || "-"}`,
        `project ${r.project_id || "-"}`,
        `events ${d.events?.length ?? 0}`,
        r.error ? `error ${r.error}` : "",
      ].filter(Boolean).join("\n"));
    });
  });
}

function renderAttentionCard(view: SpaceOSView, grid: HTMLElement, inbox: { count: number; oldestDays: number | null }, tasks: { open: number; blocked: number; overdue: number }, health: { dead: number; orphans: number; missingFm: number; score: number }) {
  const card = grid.createDiv({ cls: "ark-health-card ark-health-attention" });
  const head = card.createDiv({ cls: "ark-health-card-head" });
  head.createDiv({ cls: "ark-health-card-title", text: "需要处理" });
  head.createDiv({ cls: "ark-health-card-meta", text: "按风险排序，不显示无行动价值的统计" });
  const list = card.createDiv({ cls: "ark-health-attention-list" });
  const items: { label: string; detail: string; action: string; run: () => void; tone: string }[] = [];
  if (tasks.blocked) items.push({ label: "任务阻塞", detail: `${tasks.blocked} 条任务处于 blocked`, action: "打开任务", tone: "bad", run: () => view.switchTab("tactical") });
  if (tasks.overdue) items.push({ label: "任务逾期", detail: `${tasks.overdue} 条任务已过期`, action: "查看排期", tone: "warn", run: () => view.switchTab("tactical") });
  if (inbox.count) items.push({ label: "捕获积压", detail: `${inbox.count} 条 Inbox${inbox.oldestDays != null ? `，最早 ${inbox.oldestDays} 天前` : ""}`, action: "打开捕获队列", tone: "warn", run: () => view.switchTab("capture") });
  if (health.dead || health.missingFm) items.push({ label: "Vault 需要整理", detail: `死链 ${health.dead} · 缺 frontmatter ${health.missingFm}`, action: "打开搜索", tone: "warn", run: () => openSearch(view.plugin) });
  if (!items.length) list.createDiv({ cls: "ark-health-clear", text: "当前没有需要人工处理的异常" });
  items.forEach((item) => {
    const row = list.createDiv({ cls: `ark-health-attention-row ${item.tone}` });
    const text = row.createDiv({ cls: "ark-health-attention-text" });
    text.createDiv({ cls: "ark-health-attention-label", text: item.label });
    text.createDiv({ cls: "ark-health-attention-detail", text: item.detail });
    row.createEl("button", { cls: "tactical-mini-btn", text: item.action }).addEventListener("click", item.run);
  });
}

function renderActivityCard(view: SpaceOSView, mount: HTMLElement) {
  const card = mount.createDiv({ cls: "ark-health-card ark-health-activity" });
  const head = card.createDiv({ cls: "ark-health-card-head" });
  head.createDiv({ cls: "ark-health-card-title", text: "实时活动流" });
  head.createDiv({ cls: "ark-health-card-meta", text: "日志与系统事件倒序；运行详情见最近执行" });
  const list = card.createDiv({ cls: "ark-health-activity-list" });
  const logs = [...view.plugin.data.logs].sort((a, b) => b.createdAt - a.createdAt).slice(0, 8);
  if (!logs.length) {
    list.createDiv({ cls: "ark-health-clear", text: "暂无活动记录" });
    return;
  }
  logs.forEach((log) => {
    const row = list.createDiv({ cls: `ark-health-activity-row ${log.type === "fault" ? "bad" : ""}` });
    row.createSpan({ cls: "ark-health-activity-time", text: new Date(log.createdAt).toLocaleString() });
    row.createSpan({ cls: "ark-health-activity-label", text: log.title });
    row.createSpan({ cls: "ark-health-activity-detail", text: log.content || "无附加说明" });
  });
}

function inboxStats(app: App): { count: number; oldestDays: number | null } {
  const files = app.vault.getMarkdownFiles().filter((f) => f.path.startsWith(INBOX_DIR + "/"));
  let oldestDays: number | null = null;
  files.forEach((f) => {
    const days = Math.floor((Date.now() - f.stat.mtime) / DAY_MS);
    oldestDays = oldestDays === null ? days : Math.max(oldestDays, days);
  });
  return { count: files.length, oldestDays };
}

function taskStats(plugin: ArkOSPlugin): { open: number; blocked: number; overdue: number } {
  const today = getToday();
  const tasks = plugin.data.todoLists.flatMap((l) => l.tasks);
  return {
    open: tasks.filter((t) => t.status !== "done").length,
    blocked: tasks.filter((t) => t.status === "blocked").length,
    overdue: tasks.filter((t) => t.status !== "done" && !!t.dueDate && t.dueDate < today).length,
  };
}

function todayTaskStats(plugin: ArkOSPlugin): { done: number; total: number } {
  const today = getToday();
  const tasks = plugin.data.todoLists.flatMap((list) => list.tasks)
    .filter((task) => task.dueDate === today || new Date(task.createdAt).toISOString().slice(0, 10) === today);
  return { done: tasks.filter((task) => task.status === "done").length, total: tasks.length };
}

function healthStats(app: App): { dead: number; orphans: number; missingFm: number; score: number } {
  const mc = app.metadataCache;
  let dead = 0;
  for (const missing of Object.values(mc.unresolvedLinks ?? {})) dead += Object.keys(missing ?? {}).length;
  const referenced = new Set<string>();
  for (const targets of Object.values(mc.resolvedLinks ?? {})) for (const target of Object.keys(targets ?? {})) referenced.add(target);
  let orphans = 0;
  let missingFm = 0;
  for (const file of app.vault.getMarkdownFiles()) {
    if (!referenced.has(file.path)) orphans++;
    if (!mc.getCache(file.path)?.frontmatter) missingFm++;
  }
  const score = Math.max(0, Math.round(100 - dead - orphans * 0.5 - missingFm * 0.1));
  return { dead, orphans, missingFm, score };
}
