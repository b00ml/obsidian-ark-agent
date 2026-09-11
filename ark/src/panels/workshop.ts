import { TFile } from "obsidian";
import type { SpaceOSView } from "../view";
import { createCard } from "../cards";
import { generateReport } from "../report";
import { generateDailyReview } from "../daily";
import { generateWeeklyReview } from "../weekly";
import { generateQuiz, gradeQuiz } from "../learning-quiz";
import { generateDailyBrief } from "../publish";
import { generateMoc } from "../moc";
import { runCaseLib } from "../caselib";
import { inputDialog, notice } from "../ui";
import { sendSelectionToWorkbench } from "../agent-workbench";

type WorkshopKind =
  | "report"
  | "review"
  | "quiz"
  | "brief"
  | "moc"
  | "case"
  | "drawing";

interface OutputSpec {
  id: WorkshopKind;
  icon: string;
  label: string;
  description: string;
  folder: (view: SpaceOSView) => string;
  matches: (type: string, file: TFile) => boolean;
  run: (view: SpaceOSView) => Promise<string>;
  alternatives?: { label: string; run: (view: SpaceOSView) => Promise<string> }[];
}

interface OutputItem {
  file: TFile;
  title: string;
  type: string;
  date: string;
  sources: number | null;
}

/**
 * 产出工坊：统一浏览和触发知识产物生成。
 * 真源仍是 Vault Markdown；本面板只读取 frontmatter 投影，不维护第二套产物索引。
 */
export function renderWorkshop(view: SpaceOSView, mount: HTMLElement) {
  mount.addClass("space-workshop");
  const specs = makeSpecs();
  let selected = specs[0].id;
  let busy = false;

  const head = mount.createDiv({ cls: "workshop-head" });
  const heading = head.createDiv({ cls: "workshop-heading" });
  heading.createDiv({ cls: "tactical-title", text: "产出工坊" });
  heading.createDiv({ cls: "tactical-subtitle", text: "集中管理日报、回顾、问答、知识地图和案例等可复用产物。" });
  const agent = head.createEl("button", { cls: "tactical-btn-agent", text: "🤖 交给 Agent 规划" });
  agent.title = "让 Agent 根据当前产出类型和已有产物给出下一步建议";
  agent.addEventListener("click", () => {
    const spec = specs.find((x) => x.id === selected)!;
    const items = listItems(view, spec).slice(0, 8);
    const refs = items.map((x) => `[[${x.file.path}]]`).join(" ");
    sendSelectionToWorkbench(view.plugin,
      `请规划“${spec.label}”的下一步产出，先分析已有文件再给出可执行建议。\n已有产物：${refs || "（暂无）"}`);
  });

  const layout = mount.createDiv({ cls: "workshop-layout" });
  const nav = layout.createDiv({ cls: "workshop-nav" });
  const content = layout.createDiv({ cls: "workshop-content" });
  const actions = layout.createDiv({ cls: "workshop-actions" });
  const contentHead = content.createDiv({ cls: "workshop-content-head" });
  const contentTitle = contentHead.createDiv({ cls: "workshop-content-title" });
  const contentMeta = contentHead.createDiv({ cls: "workshop-content-meta" });
  const list = content.createDiv({ cls: "workshop-list" });

  const render = () => {
    nav.empty();
    actions.empty();
    list.empty();
    const spec = specs.find((x) => x.id === selected)!;

    specs.forEach((candidate) => {
      const button = nav.createEl("button", {
        cls: `workshop-nav-item${candidate.id === selected ? " active" : ""}`,
      });
      button.createSpan({ cls: "workshop-nav-label", text: `${candidate.icon} ${candidate.label}` });
      button.createSpan({ cls: "workshop-nav-count", text: String(listItems(view, candidate).length) });
      button.addEventListener("click", () => {
        selected = candidate.id;
        render();
      });
    });

    const items = listItems(view, spec);
    contentTitle.setText(spec.label);
    contentMeta.setText(`${items.length} 个产物 · ${spec.description}`);
    if (!items.length) {
      list.createDiv({ cls: "workshop-empty", text: "暂无产物，可从右侧生成" });
    } else {
      items.forEach((item) => renderItem(view, list, item));
    }

    const actionHead = actions.createDiv({ cls: "workshop-action-head" });
    actionHead.createDiv({ cls: "workshop-action-title", text: "生成产物" });
    actionHead.createDiv({ cls: "workshop-action-hint", text: spec.description });
    const runAction = (button: HTMLButtonElement, label: string, action: (view: SpaceOSView) => Promise<string>) => {
      button.disabled = busy;
      button.addEventListener("click", async () => {
        if (busy) return;
        busy = true;
        button.disabled = true;
        button.setText(`${spec.icon} 生成中…`);
        try {
          const result = await action(view);
          if (result) notice(result);
          view.renderCurrentPanel();
        } catch (e: any) {
          notice(`生成失败：${String(e?.message ?? e)}`);
        } finally {
          busy = false;
          button.setText(`${spec.icon} ${label}`);
        }
      });
    };
    const primaryLabel = spec.id === "report" ? "生成日报" : spec.id === "review" ? "生成每日回顾" : spec.id === "quiz" ? "生成问答" : `生成 ${spec.label}`;
    const run = actions.createEl("button", { cls: "workshop-run", text: `${spec.icon} ${primaryLabel}` });
    runAction(run, primaryLabel, spec.run);
    spec.alternatives?.forEach((alternative) => {
      const alt = actions.createEl("button", { cls: "workshop-run secondary", text: `${spec.icon} ${alternative.label}` });
      runAction(alt, alternative.label, alternative.run);
    });
    const note = actions.createDiv({ cls: "workshop-action-note", text: "产物会直接写入对应 Vault 文件夹；生成失败不会创建空文件。" });
    note.setAttr("aria-live", "polite");
    renderWorkshopHistory(view, actions, specs);
  };

  render();
}

function renderWorkshopHistory(view: SpaceOSView, parent: HTMLElement, specs: OutputSpec[]) {
  const history = parent.createDiv({ cls: "workshop-history" });
  history.createDiv({ cls: "workshop-history-title", text: "最近生成" });
  const items = specs.flatMap((spec) => listItems(view, spec).map((item) => ({ ...item, spec })));
  items.sort((a, b) => b.file.stat.mtime - a.file.stat.mtime).slice(0, 4).forEach((item) => {
    const row = history.createDiv({ cls: "workshop-history-item" });
    row.createSpan({ cls: "workshop-history-date", text: item.date });
    row.createSpan({ cls: "workshop-history-label", text: item.spec.label });
    row.createSpan({ cls: "workshop-history-sources", text: item.sources != null ? `${item.sources} 来源` : "已落盘" });
  });
  if (!items.length) history.createDiv({ cls: "workshop-history-empty", text: "尚无生成记录" });
}

function makeSpecs(): OutputSpec[] {
  return [
    {
      id: "report", icon: "📊", label: "日报 / 周报", description: "按日志汇总工作与故障记录",
      folder: (v) => v.plugin.data.settings.reportFolder,
      matches: (type) => type === "report",
      run: async (v) => `日报已生成：${await generateReport(v.plugin, "daily")}`,
      alternatives: [{ label: "生成周报", run: async (v) => `周报已生成：${await generateReport(v.plugin, "weekly")}` }],
    },
    {
      id: "review", icon: "🧭", label: "AI 回顾", description: "让 Agent 总结近期学习、产出与下一步",
      folder: (v) => v.plugin.data.settings.reviewFolder,
      matches: (type) => type === "daily-review" || type === "weekly-review" || type === "period-review",
      run: async (v) => { await generateDailyReview(v.plugin); return "AI 每日回顾已生成"; },
      alternatives: [{ label: "生成周报回顾", run: async (v) => { await generateWeeklyReview(v.plugin); return "AI 周报回顾已生成"; } }],
    },
    {
      id: "quiz", icon: "🧠", label: "学习问答", description: "从学习笔记出题并生成批改结果",
      folder: (v) => v.plugin.data.settings.quizFolder,
      matches: (type) => type === "learning-quiz" || type === "learning-quiz-graded",
      run: async (v) => {
        const path = await generateQuiz(v.plugin);
        return path === "NOTHING" ? "今日没有可出题的新学习笔记" : `学习问答已生成：${path}`;
      },
      alternatives: [{ label: "批改最新问答", run: async (v) => {
        const path = await gradeQuiz(v.plugin);
        return path === "NONE" ? "尚未生成学习问答" : path === "DONE" ? "学习问答均已批改" : `批改已生成：${path}`;
      } }],
    },
    {
      id: "brief", icon: "📰", label: "信息日报", description: "把已归档订阅简报聚合成可扫读日报",
      folder: (v) => v.plugin.data.settings.feedArchiveFolder,
      matches: (type) => type === "daily-brief",
      run: async (v) => { const r = await generateDailyBrief(v.plugin); return `信息日报已生成：${r.items} 条`; },
    },
    {
      id: "moc", icon: "🗺", label: "主题知识地图", description: "按主题把知识层产物聚合为 MOC",
      folder: (v) => v.plugin.data.settings.mocFolder,
      matches: (type) => type === "moc",
      run: async (v) => {
        const topic = await inputDialog(v.plugin, { title: "生成主题 MOC", placeholder: "输入主题，如 AI agent" });
        if (!topic?.trim()) return "";
        const r = await generateMoc(v.plugin, topic.trim());
        return `MOC 已生成：命中 ${r.match} 条 → ${r.path}`;
      },
    },
    {
      id: "case", icon: "📚", label: "案例库", description: "把要点卡蒸馏成可复用案例",
      folder: (v) => v.plugin.data.settings.caseFolder,
      matches: (type) => type === "case-library",
      run: async (v) => {
        const r = await runCaseLib(v.plugin);
        return r.cards ? `案例库已更新：${r.cases} 条案例` : "暂无要点卡可沉淀";
      },
    },
    {
      id: "drawing", icon: "🖥", label: "创作草稿", description: "保留创作草稿并交给 Agent 提炼",
      folder: (v) => v.plugin.data.settings.drawingFolder,
      matches: (type, file) => type === "drawing" || file.path.endsWith("-画板.md"),
      run: async (v) => {
        const title = await inputDialog(v.plugin, { title: "新建创作草稿", placeholder: "标题" });
        if (!title?.trim()) return "";
        const body = (await inputDialog(v.plugin, { title: "草稿内容（可选）", multiline: true })) ?? "";
        const path = await createCard(v.plugin, "drawing", { fm: { title: title.trim(), color: "#00e5ff", description: body }, body });
        return `创作草稿已保存：${path}`;
      },
    },
  ];
}

function listItems(view: SpaceOSView, spec: OutputSpec): OutputItem[] {
  const folder = spec.folder(view).replace(/\/$/, "");
  return view.plugin.app.vault.getMarkdownFiles()
    .filter((file) => file.path.startsWith(folder + "/"))
    .map((file) => {
      const fm = view.plugin.app.metadataCache.getFileCache(file)?.frontmatter as Record<string, unknown> | undefined;
      const type = String(fm?.type ?? "").trim();
      return {
        file,
        title: String(fm?.title ?? file.basename),
        type,
        date: formatDate(fm?.date ?? fm?.generated_at ?? fm?.created_at, file.stat.mtime),
        sources: sourceCount(fm),
      };
    })
    .filter((item) => spec.matches(item.type, item.file))
    .sort((a, b) => b.file.stat.mtime - a.file.stat.mtime);
}

function renderItem(view: SpaceOSView, parent: HTMLElement, item: OutputItem) {
  const row = parent.createDiv({ cls: "workshop-item" });
  const main = row.createDiv({ cls: "workshop-item-main" });
  main.createDiv({ cls: "workshop-item-title", text: item.title });
  const meta = [item.date, item.sources != null ? `来源 ${item.sources}` : ""].filter(Boolean).join(" · ");
  main.createDiv({ cls: "workshop-item-meta", text: meta });
  const actions = row.createDiv({ cls: "workshop-item-actions" });
  actions.createEl("button", { text: "打开" }).addEventListener("click", () => {
    void view.plugin.app.workspace.getLeaf("tab").openFile(item.file);
  });
  const agent = actions.createEl("button", { text: "交给 Agent" });
  agent.title = "把产物和来源路径交给主控大厅继续处理";
  agent.addEventListener("click", () => sendSelectionToWorkbench(view.plugin,
    `请基于这份产物继续处理，保留来源引用。\n来源：[[${item.file.path}]]\n标题：${item.title}`));
}

function sourceCount(fm?: Record<string, unknown>): number | null {
  if (!fm) return null;
  const value = fm.sources ?? fm.count ?? fm.items ?? fm.log_count;
  if (Array.isArray(value)) return value.length;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function formatDate(value: unknown, fallback: number): string {
  const raw = String(value ?? "");
  const date = raw && !Number.isNaN(Date.parse(raw)) ? new Date(raw) : new Date(fallback);
  return date.toLocaleDateString();
}
