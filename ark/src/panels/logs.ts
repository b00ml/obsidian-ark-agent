import { TFile } from "obsidian";
import type { SpaceOSView } from "../view";
import { getToday } from "../utils";
import { promoteToDatabase, generateReport } from "../report";
import { createCard, deleteCardAt } from "../cards";
import { inputDialog, confirmDialog, notice } from "../ui";
import { generateDailyReview } from "../daily";
import { generateWeeklyReview } from "../weekly";
import { generateQuiz, gradeQuiz } from "../learning-quiz";
import { scanFeeds } from "../feeds";
import { archiveFeedBriefs } from "../archive";
import { runPipeline } from "../pipeline";
import { generateDailyBrief } from "../publish";
import { generateMoc } from "../moc";
import { runCaseLib } from "../caselib";
import { getSkin } from "../skins";
import { copyText, generatePeriodReview, previewGeneratedModal } from "../ai-asst";
import { fetchAgentRuns } from "../ai";
import { writeMarkdown } from "../sync";

/** 工作记录：周期趋势与 AI 复盘；原有采集/生成动作收进可展开操作区。 */
export function renderLogs(view: SpaceOSView, mount: HTMLElement) {
  mount.addClass("space-logs");
  const skin = getSkin(view.plugin.data.settings.skin);
  const logs = view.plugin.data.logs;
  const today = getToday();
  const total = logs.length;
  const normal = logs.filter((l) => l.type === "normal").length;
  const fault = logs.filter((l) => l.type === "fault").length;
  const todayCount = logs.filter((l) => new Date(l.createdAt).toISOString().split("T")[0] === today).length;

  const dash = mount.createDiv({ cls: "log-dashboard" });
  dash.createEl("span", { cls: "log-stat", text: `${skin.panel.logStats[0]} ${total}` });
  dash.createEl("span", { cls: "log-stat log-normal", text: `${skin.panel.logStats[1]} ${normal}` });
  dash.createEl("span", { cls: "log-stat log-fault", text: `${skin.panel.logStats[2]} ${fault}` });
  dash.createEl("span", { cls: "log-stat", text: `${skin.panel.logStats[3]} ${todayCount}` });

  renderPeriodSummary(view, mount);

  const operations = mount.createEl("details", { cls: "log-operations" });
  operations.createEl("summary", { text: "操作入口 · 采集 / 处理 / 产出" });
  const opBody = operations.createDiv({ cls: "log-operations-body" });
  const modebar = opBody.createDiv({ cls: "log-mode-tabs" });
  const actionArea = opBody.createDiv({ cls: "log-action-area" });
  const activity = actionArea.createDiv({ cls: "log-action-group", attr: { "data-group": "activity" } });
  const processing = actionArea.createDiv({ cls: "log-action-group", attr: { "data-group": "processing" } });
  const output = actionArea.createDiv({ cls: "log-action-group", attr: { "data-group": "output" } });
  let activeGroup = "activity";
  const showGroup = (group: string) => {
    activeGroup = group;
    actionArea.querySelectorAll<HTMLElement>(".log-action-group").forEach((el) => {
      el.toggleClass("active", el.dataset.group === group);
    });
    modebar.querySelectorAll<HTMLElement>(".log-mode-tab").forEach((el) => {
      el.toggleClass("active", el.dataset.group === group);
    });
  };
  ([
    ["activity", "活动"], ["processing", "处理"], ["output", "产出"],
  ] as const).forEach(([group, label]) => {
    const b = modebar.createEl("button", { cls: "log-mode-tab", text: label, attr: { "data-group": group } });
    b.addEventListener("click", () => showGroup(group));
  });
  showGroup(activeGroup);
  let active = "normal" as "normal" | "fault";
  const makeTab = (kv: "normal" | "fault", label: string) => {
    const b = activity.createEl("button", { cls: `log-type-tab${active === kv ? " active" : ""}` });
    b.setText(label);
    b.addEventListener("click", () => {
      active = kv;
      activity.querySelectorAll<HTMLElement>(".log-type-tab").forEach((el) => el.toggleClass("active", el.dataset.kv === kv));
      list.empty();
      renderList(list, kv);
    });
    b.dataset.kv = kv;
    return b;
  };
  makeTab("normal", skin.panel.logNormal);
  makeTab("fault", skin.panel.logFault);

  const newBtn = activity.createEl("button", { cls: "log-new-btn" });
  newBtn.setText("+ 新建日志");
  newBtn.addEventListener("click", async () => {
    const title = await inputDialog(view.plugin, { title: "新建日志", placeholder: "标题" });
    if (!title || !title.trim()) return;
    const content = (await inputDialog(view.plugin, { title: "日志内容", multiline: true })) ?? "";
    await createCard(view.plugin, "log", {
      fm: { type: active === "fault" ? "fault_log" : "captain_log", log_type: active, title: title.trim(), tags: [] },
      body: content,
    });
    view.renderCurrentPanel();
  });

  const reportBtn = output.createEl("button", { cls: "log-type-tab", text: "📊 生成日报" });
  reportBtn.addEventListener("click", async () => {
    try {
      await generateReport(view.plugin, "daily");
      notice("日报已生成，可在数据库/报告文件夹查看");
      view.renderCurrentPanel();
    } catch (e: any) {
      notice("日报生成失败: " + String(e?.message ?? e));
    }
  });
  const weekBtn = output.createEl("button", { cls: "log-type-tab", text: "📅 生成周报" });
  weekBtn.addEventListener("click", async () => {
    try {
      await generateReport(view.plugin, "weekly");
      notice("周报已生成");
      view.renderCurrentPanel();
    } catch (e: any) {
      notice("周报生成失败: " + String(e?.message ?? e));
    }
  });

  // 手动触发生成每日回顾（走 /v1 agent 路径，不经 Hermes cron——本环境 cron agent 模式空转，见 DEV-020）
  const aiReviewBtn = output.createEl("button", { cls: "log-type-tab", text: "🤖 每日回顾" });
  aiReviewBtn.addEventListener("click", async () => {
    aiReviewBtn.setText("🤖 生成中…");
    try {
      const reply = await generateDailyReview(view.plugin);
      notice("每日回顾已生成");
      view.renderCurrentPanel();
    } catch (e: any) {
      notice("每日回顾生成失败: " + String(e?.message ?? e));
    } finally {
      aiReviewBtn.setText("🤖 每日回顾");
    }
  });

  // 手动触发生成 AI 周报回顾（同每日回顾路径，扫描本周笔记）
  const aiWeeklyBtn = output.createEl("button", { cls: "log-type-tab", text: "🤖 周报回顾" });
  aiWeeklyBtn.addEventListener("click", async () => {
    aiWeeklyBtn.setText("🤖 生成中…");
    try {
      const reply = await generateWeeklyReview(view.plugin);
      notice("周报回顾已生成");
      view.renderCurrentPanel();
    } catch (e: any) {
      notice("周报回顾生成失败: " + String(e?.message ?? e));
    } finally {
      aiWeeklyBtn.setText("🤖 周报回顾");
    }
  });

  // 学习问答：读今日学习笔记 → 出题（M1）
  const quizBtn = output.createEl("button", { cls: "log-type-tab", text: "🤖 学习问答" });
  quizBtn.addEventListener("click", async () => {
    quizBtn.setText("🤖 出题中…");
    try {
      const path = await generateQuiz(view.plugin);
      notice(path === "NOTHING" ? "今日无学习笔记可出题（learningFolder 下今日修改的笔记）" : `学习问答已生成：${path}`);
      view.renderCurrentPanel();
    } catch (e: any) {
      notice("学习问答生成失败: " + String(e?.message ?? e));
    } finally {
      quizBtn.setText("🤖 学习问答");
    }
  });

  // 学习批改：读最新未批改问答 → 判定（M1）
  const gradeBtn = output.createEl("button", { cls: "log-type-tab", text: "🤖 批改" });
  gradeBtn.addEventListener("click", async () => {
    gradeBtn.setText("🤖 批改中…");
    try {
      const path = await gradeQuiz(view.plugin);
      notice(path === "NONE" ? "尚未生成学习问答" : path === "DONE" ? "学习问答均已批改" : `已批改：${path}`);
      view.renderCurrentPanel();
    } catch (e: any) {
      notice("批改失败: " + String(e?.message ?? e));
    } finally {
      gradeBtn.setText("🤖 批改");
    }
  });

  // 主动获取：扫描所有启用订阅源（M2）
  const feedBtn = processing.createEl("button", { cls: "log-type-tab", text: "📡 获取订阅" });
  feedBtn.addEventListener("click", async () => {
    feedBtn.setText("📡 获取中…");
    try {
      const r = await scanFeeds(view.plugin);
      notice(`订阅完成：新增 ${r.written} 篇，跳过 ${r.skipped} 篇${r.feedsFailed ? `，失败 ${r.feedsFailed} 个源` : ""}`);
      view.renderCurrentPanel();
    } catch (e: any) {
      notice("订阅获取失败: " + String(e?.message ?? e));
    } finally {
      feedBtn.setText("📡 获取订阅");
    }
  });

  // 归档门控：把 Inbox 简报确认后搬到知识层（M2 二期 / D4）
  const archiveBtn = processing.createEl("button", { cls: "log-type-tab", text: "🗂 归档简报" });
  archiveBtn.addEventListener("click", async () => {
    archiveBtn.setText("🗂 归档中…");
    try {
      await archiveFeedBriefs(view.plugin);
      view.renderCurrentPanel();
    } catch (e: any) {
      notice("归档失败: " + String(e?.message ?? e));
    } finally {
      archiveBtn.setText("🗂 归档简报");
    }
  });

  // M3 处理管线：把已归档简报批量跑规则层+AI建议层，产要点卡（§3.2 / P1-4）
  const pipelineBtn = processing.createEl("button", { cls: "log-type-tab", text: "⚙️ 处理简报" });
  pipelineBtn.addEventListener("click", async () => {
    pipelineBtn.setText("⚙️ 处理中…");
    try {
      const r = await runPipeline(view.plugin);
      notice(`处理简报：产要点卡 ${r.cards} 篇，忽略 ${r.skipped}，失败 ${r.failed}`);
      view.renderCurrentPanel();
    } catch (e: any) {
      notice("处理失败: " + String(e?.message ?? e));
    } finally {
      pipelineBtn.setText("⚙️ 处理简报");
    }
  });

  // M4 产出：把当日简报+要点卡聚合成一页信息日报（§3.4）
  const briefBtn = output.createEl("button", { cls: "log-type-tab", text: "📰 信息日报" });
  briefBtn.addEventListener("click", async () => {
    briefBtn.setText("📰 生成中…");
    try {
      const r = await generateDailyBrief(view.plugin);
      notice(`信息日报已${r.sources ? "更新" : "生成"}：${r.sources} 个来源、${r.items} 条`);
      view.renderCurrentPanel();
    } catch (e: any) {
      notice("生成失败: " + String(e?.message ?? e));
    } finally {
      briefBtn.setText("📰 信息日报");
    }
  });

  // M4 产出：按主题聚合成知识地图导航页（§3.4 MOC，D3 手动触发）
  const mocBtn = output.createEl("button", { cls: "log-type-tab", text: "🗺 MOC" });
  mocBtn.addEventListener("click", async () => {
    const topic = await inputDialog(view.plugin, { title: "生成主题 MOC", placeholder: "输入主题，如 AI agent" });
    if (!topic || !topic.trim()) return;
    mocBtn.setText("🗺 生成中…");
    try {
      const r = await generateMoc(view.plugin, topic);
      notice(`MOC 已生成：命中 ${r.match} 条 → ${r.path}`);
      view.renderCurrentPanel();
    } catch (e: any) {
      notice("生成失败: " + String(e?.message ?? e));
    } finally {
      mocBtn.setText("🗺 MOC");
    }
  });

  // M4 产出：从要点卡蒸馏可复用案例 → 案例库（§3.4，按首标签聚合）
  const caseBtn = output.createEl("button", { cls: "log-type-tab", text: "📚 案例库" });
  caseBtn.addEventListener("click", async () => {
    caseBtn.setText("📚 沉淀中…");
    try {
      const r = await runCaseLib(view.plugin);
      if (r.cards === 0) notice("暂无要点卡可直接沉淀");
      else notice(`案例库沉淀：扫 ${r.cards} 张要点卡 → 提炼 ${r.cases} 条案例，聚合 ${r.groups} 组`);
      view.renderCurrentPanel();
    } catch (e: any) {
      notice("沉淀失败: " + String(e?.message ?? e));
    } finally {
      caseBtn.setText("📚 案例库");
    }
  });

  const list = mount.createDiv({ cls: "log-list" });
  const renderList = (parent: HTMLElement, kv: "normal" | "fault") => {
    const items = logs.filter((l) => l.type === kv).sort((a, b) => b.createdAt - a.createdAt);
    if (items.length === 0) {
      parent.createDiv({ cls: "tactical-empty", text: "暂无记录" });
      return;
    }
    items.forEach((l) => {
      const entry = parent.createDiv({ cls: "log-entry glass" });
      entry.createDiv({ cls: "log-entry-title", text: l.title });
      entry.createDiv({ cls: "log-entry-time", text: new Date(l.createdAt).toLocaleString() });
      if (l.content) entry.createDiv({ cls: "log-entry-content", text: l.content });
      const act = entry.createDiv({ cls: "tactical-card-actions" });
      const promo = act.createEl("button", { cls: "tactical-mini-btn", text: "💎 沉淀到数据库" });
      promo.addEventListener("click", async () => {
        try {
          await promoteToDatabase(view.plugin, { name: l.title, tags: l.tags, description: l.content || l.title });
          notice("已沉淀到数据库");
        } catch (e: any) {
          notice("沉淀失败: " + String(e?.message ?? e));
        }
      });
      const del = act.createEl("button", { cls: "tactical-mini-btn", text: "🗑" });
      del.addEventListener("click", async () => {
        if (!(await confirmDialog(view.plugin, "删除该日志？"))) return;
        await deleteCardAt(view.plugin, l.notePath);
        view.renderCurrentPanel();
      });
    });
  };
  renderList(list, active);
}

function renderPeriodSummary(view: SpaceOSView, mount: HTMLElement) {
  const section = mount.createDiv({ cls: "log-period-summary" });
  const head = section.createDiv({ cls: "log-period-head" });
  head.createDiv({ cls: "tactical-title", text: "周期推演" });
  head.createDiv({ cls: "tactical-subtitle", text: "用可验证的增量、任务和捕获吞吐判断系统是否在向前推进。" });
  const period = head.createEl("select", { cls: "log-period-select" });
  period.createEl("option", { value: "day", text: "今天" });
  period.createEl("option", { value: "week", text: "本周" });
  period.createEl("option", { value: "month", text: "本月" });
  period.createEl("option", { value: "quarter", text: "本季度" });
  period.value = "week";
  const cards = section.createDiv({ cls: "log-trend-grid" });
  const insight = section.createDiv({ cls: "log-period-insight" });
  let renderVersion = 0;
  const reviewBtn = head.createEl("button", { cls: "tactical-btn-agent", text: "🤖 AI 周期复盘" });
  reviewBtn.title = "按当前周期生成回顾，结果写入 Vault 前由现有生成流程确认";
  reviewBtn.addEventListener("click", async () => {
    reviewBtn.disabled = true;
    reviewBtn.setText("🤖 复盘中…");
    try {
      const range = periodRange(period.value);
      const logs = view.plugin.data.logs.filter((l) => l.createdAt >= range.start && l.createdAt <= range.end);
      const knowledge = (view.plugin.data.database as any[]).filter((i) => i.createdAt >= range.start && i.createdAt <= range.end);
      const tasks = view.plugin.data.todoLists.flatMap((l) => l.tasks);
      const done = tasks.filter((t) => t.status === "done" && (t.completedAt ?? t.createdAt) >= range.start && (t.completedAt ?? t.createdAt) <= range.end).length;
      const inboxFiles = view.plugin.app.vault.getMarkdownFiles().filter((f) => f.path.startsWith("Inbox/") && f.stat.mtime >= range.start && f.stat.mtime <= range.end);
      const evidence = [
        ...logs.slice(-12).map((l) => `- 日志：${l.title}${l.content ? `｜${l.content.slice(0, 180)}` : ""}`),
        ...knowledge.slice(-12).map((i) => `- 知识：[[${i.filePath ?? i.name}]]｜${i.name}`),
        ...inboxFiles.slice(-12).map((f) => `- 捕获：[[${f.path}]]`),
        ...tasks.filter((t) => t.status === "done").slice(-12).map((t) => `- 任务完成：${t.description}`),
      ];
      const runs = await fetchAgentRuns(view.plugin.data.settings, 100);
      const draft = await generatePeriodReview(view.plugin, {
        periodLabel: periodLabel(period.value),
        startDate: localDate(new Date(range.start)),
        endDate: localDate(new Date(range.end)),
        metrics: {
          "知识增量": knowledge.length,
          "活动记录": logs.length,
          "任务完成": done,
          "捕获流入": inboxFiles.length,
          "Agent运行量": runs?.filter((run) => {
            const time = Date.parse(run.time);
            return Number.isFinite(time) && time >= range.start && time <= range.end;
          }).length ?? "不可用",
        },
        evidence,
      });
      const folder = view.plugin.data.settings.reviewFolder || "02-DB/回顾";
      const filename = `${localDate(new Date(range.end))}-${period.value}-复盘`;
      const existing = view.plugin.app.vault.getAbstractFileByPath(`${folder}/${filename}.md`);
      const previousRaw = existing instanceof TFile ? await view.plugin.app.vault.read(existing) : "";
      const previous = previousRaw.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, "").trim();
      const action = await previewGeneratedModal(view.plugin, `AI ${periodLabel(period.value)}复盘预览`, previous, draft);
      if (action === "copy") {
        await copyText(view.plugin, draft);
        notice("复盘草稿已复制，未写入 Vault");
      } else if (action === "save") {
        const path = await writeMarkdown(view.plugin, folder, filename, {
          title: `${periodLabel(period.value)}复盘`, type: "period-review", period: period.value,
          period_start: localDate(new Date(range.start)), period_end: localDate(new Date(range.end)),
          date: localDate(new Date()), tags: ["period-review", "回顾"],
        }, draft);
        notice(`复盘已写入：${path}`);
      }
    } catch (e: any) {
      notice(`AI 周期复盘失败：${String(e?.message ?? e)}`);
    } finally {
      reviewBtn.disabled = false;
      reviewBtn.setText("🤖 AI 周期复盘");
    }
  });
  const render = () => {
    const version = ++renderVersion;
    cards.empty();
    insight.empty();
    const { start, end } = periodRange(period.value);
    const logs = view.plugin.data.logs.filter((l) => l.createdAt >= start && l.createdAt <= end);
    const knowledge = (view.plugin.data.database as any[]).filter((i) => i.createdAt >= start && i.createdAt <= end);
    const tasks = view.plugin.data.todoLists.flatMap((l) => l.tasks);
    const done = tasks.filter((t) => t.status === "done" && (t.completedAt ?? t.createdAt) >= start).length;
    const open = tasks.filter((t) => t.status !== "done").length;
    const inboxFiles = view.plugin.app.vault.getMarkdownFiles().filter((f) => f.path.startsWith("Inbox/") && f.stat.mtime >= start && f.stat.mtime <= end);
    addTrend(cards, "知识增量", String(knowledge.length), "条目在此周期进入知识层", knowledge.length ? "ok" : "dim", bucketValues(knowledge.map((item) => item.createdAt), start, end));
    addTrend(cards, "任务完成率", taskRate(tasks, start, end), `当前仍有 ${open} 条未完成`, done ? "ok" : "warn", bucketValues(tasks.filter((t) => t.status === "done").map((task) => task.completedAt ?? task.createdAt).filter((time) => time >= start && time <= end), start, end));
    addTrend(cards, "捕获流入", String(inboxFiles.length), "Inbox 新增文件", inboxFiles.length ? "warn" : "ok", bucketValues(inboxFiles.map((file) => file.stat.mtime), start, end));
    const runCard = addTrend(cards, "Agent 运行量", "读取中…", "最近 100 次运行", "dim", []);
    void fetchAgentRuns(view.plugin.data.settings, 100).then((runs) => {
      if (version !== renderVersion) return;
      if (runs === null) {
        updateTrend(runCard, "不可用", "当前后端未提供运行记录", "dim", []);
        return;
      }
      const periodRuns = runs.filter((run) => {
        const time = Date.parse(run.time);
        return Number.isFinite(time) && time >= start && time <= end;
      });
      const runErrors = periodRuns.filter((run) => !!run.error).length;
      updateTrend(runCard, String(periodRuns.length), `错误 ${runErrors} · 最近 100 次`, periodRuns.length && runErrors ? "warn" : "ok", bucketValues(periodRuns.map((run) => Date.parse(run.time)), start, end));
    });
    insight.createDiv({ cls: "log-period-insight-title", text: "周期结论草稿" });
    insight.createDiv({ cls: "log-period-insight-row", text: `知识新增 ${knowledge.length} 条 · 活动记录 ${logs.length} 条 · 捕获流入 ${inboxFiles.length} 条` });
    insight.createDiv({ cls: "log-period-insight-row", text: `任务完成 ${done} 条 · 当前未完成 ${open} 条${done === 0 && open > 0 ? " · 建议先打开调度矩阵处理阻塞项" : ""}` });
  };
  period.addEventListener("change", render);
  render();
}

function addTrend(parent: HTMLElement, label: string, value: string, hint: string, tone: string, samples: number[]): HTMLElement {
  const card = parent.createDiv({ cls: `log-trend-card ${tone}` });
  card.createDiv({ cls: "log-trend-label", text: label });
  card.createDiv({ cls: "log-trend-value", text: value });
  const chart = card.createDiv({ cls: "log-trend-chart" });
  const max = Math.max(1, ...samples);
  samples.forEach((sample) => {
    const bar = chart.createDiv({ cls: "log-trend-bar" });
    bar.setAttr("style", `height:${sample ? Math.max(12, Math.round(sample / max * 100)) : 3}%`);
  });
  card.createDiv({ cls: "log-trend-hint", text: hint });
  return card;
}

function updateTrend(card: HTMLElement, value: string, hint: string, tone: string, samples: number[]): void {
  ["ok", "warn", "dim"].forEach((name) => card.removeClass(name));
  card.addClass(tone);
  card.querySelector<HTMLElement>(".log-trend-value")?.setText(value);
  card.querySelector<HTMLElement>(".log-trend-hint")?.setText(hint);
  const chart = card.querySelector<HTMLElement>(".log-trend-chart");
  if (!chart) return;
  chart.empty();
  const max = Math.max(1, ...samples);
  samples.forEach((sample) => {
    const bar = chart.createDiv({ cls: "log-trend-bar" });
    bar.setAttr("style", `height:${sample ? Math.max(12, Math.round(sample / max * 100)) : 3}%`);
  });
}

function taskRate(tasks: any[], start: number, end: number): string {
  const touched = tasks.filter((task) => {
    const created = Number(task.createdAt ?? 0);
    const completed = Number(task.completedAt ?? 0);
    return (created >= start && created <= end) || (completed >= start && completed <= end);
  });
  const done = touched.filter((task) => {
    const completed = Number(task.completedAt ?? task.createdAt ?? 0);
    return task.status === "done" && completed >= start && completed <= end;
  }).length;
  return touched.length ? `${Math.round(done * 100 / touched.length)}%` : "—";
}

function bucketValues(times: number[], start: number, end: number): number[] {
  const count = 8;
  const span = Math.max(1, end - start);
  const bucket = span / count;
  const values = Array.from({ length: count }, () => 0);
  times.forEach((time) => {
    const index = Math.max(0, Math.min(count - 1, Math.floor((time - start) / bucket)));
    values[index]++;
  });
  return values;
}

function periodStart(period: string): number {
  const now = new Date();
  if (period === "day") return new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  if (period === "month") return new Date(now.getFullYear(), now.getMonth(), 1).getTime();
  if (period === "quarter") return new Date(now.getFullYear(), Math.floor(now.getMonth() / 3) * 3, 1).getTime();
  const day = now.getDay() || 7;
  return new Date(now.getFullYear(), now.getMonth(), now.getDate() - day + 1).getTime();
}

function periodRange(period: string): { start: number; end: number } {
  return { start: periodStart(period), end: Date.now() };
}

function periodLabel(period: string): string {
  if (period === "day") return "今日";
  if (period === "month") return "本月";
  if (period === "quarter") return "本季度";
  return "本周";
}

function localDate(date: Date): string {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}
