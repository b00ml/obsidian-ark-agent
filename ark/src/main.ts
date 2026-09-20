import { Plugin, WorkspaceLeaf, TFile, Menu, setIcon } from "obsidian";
import { DEFAULT_SETTINGS, ArkSettings, emptyData, ArkData, normalizeSettings } from "./settings";
import type { TaskStatus } from "./types";
import { generateId } from "./utils";
import { SpaceOSView, VIEW_TYPE_SPACE_OS } from "./view";
import { setupSync } from "./sync";
import { writeTask, createNextRecurrence } from "./task";
import { openSearch } from "./search";
import { generateDailyReview } from "./daily";
import { generateWeeklyReview } from "./weekly";
import { generateQuiz, gradeQuiz } from "./learning-quiz";
import { scanFeeds } from "./feeds";
import { archiveFeedBriefs } from "./archive";
import { runPipeline } from "./pipeline";
import { generateDailyBrief } from "./publish";
import { generateMoc } from "./moc";
import { runCaseLib } from "./caselib";
import { inputDialog, notice } from "./ui";
import { createProject } from "./projects";
import { getSkin } from "./skins";
import { installEditorAiMenu, runTodayTodos } from "./ai-asst";
import { serveStart, serveStop, serveStatusText, serveUp } from "./serve-control";
import { restoreProjectArtifacts } from "./artifacts";
import { notifyMemoryReviewDue, openMemoryLifecycle, openMemoryReview } from "./memory-review";

export default class ArkOSPlugin extends Plugin {
  data: ArkData = emptyData();
  private serveRibbonEl?: HTMLElement;
  private serveBusy = false;

  async onload() {
    await this.loadArkData();
    // F3：ArkData 可丢弃，Project 成果侧车可在重启后恢复当前项目的结果索引。
    if (this.data.settings.activeProjectId) {
      await restoreProjectArtifacts(this, this.data.settings.activeProjectId);
    }

    this.registerView(VIEW_TYPE_SPACE_OS, (leaf: WorkspaceLeaf) => new SpaceOSView(leaf, this));

    const brand = getSkin(this.data.settings.skin).brand;
    this.addRibbonIcon("orbit", `${brand} 工作台`, () => this.activateView());
    // OPT-113：agentlab serve 服务管理按钮——图标/颜色随健康状态切换（30s 轮询 + 操作后刷新）
    this.serveRibbonEl = this.addRibbonIcon(
      "server-off", "agentlab serve：未运行（点击拉起）", (evt) => void this.onServeRibbonClick(evt));
    void this.refreshServeRibbon();
    this.registerInterval(window.setInterval(() => void this.refreshServeRibbon(), 30_000));
    this.addCommand({
      id: "serve-start",
      name: "启动 agentlab 服务",
      callback: () => void this.serveAction("start"),
    });
    this.addCommand({
      id: "serve-stop",
      name: "停止 agentlab 服务",
      callback: () => void this.serveAction("stop"),
    });
    this.addCommand({
      id: "review-long-term-memory",
      name: "复核到期长期记忆",
      callback: () => openMemoryReview(this),
    });
    this.addCommand({
      id: "manage-long-term-memory",
      name: "管理长期记忆",
      callback: () => openMemoryLifecycle(this),
    });
    void notifyMemoryReviewDue(this);
    this.registerInterval(window.setInterval(
      () => void notifyMemoryReviewDue(this), 12 * 60 * 60 * 1000,
    ));
    this.addCommand({
      id: "open-ark",
      name: `打开 ${brand} 工作台`,
      callback: () => this.activateView(),
    });
    this.addCommand({
      id: "sync-ark-vault",
      name: "同步 Vault → Ark",
      callback: () => setupSync(this),
    });
    this.addCommand({
      id: "search-ark",
      name: "知识库搜索",
      callback: () => openSearch(this),
    });
    this.addCommand({
      id: "new-project",
      name: "新建 Project（长期任务空间）",
      callback: () => {
        void (async () => {
          const name = (await inputDialog(this, { title: "Project 名称", placeholder: "如：视频入库流水线" }))?.trim();
          if (!name) return;
          const created = await createProject(this, name);
          if (!created) return;
          // 实时刷新：SpaceOSView 若开着，当前面板（含工作台项目下拉）立即重渲染，无需重载插件
          const leaf = this.app.workspace.getLeavesOfType(VIEW_TYPE_SPACE_OS)[0];
          const osView = leaf?.view as SpaceOSView | undefined;
          if (osView) await osView.renderCurrentPanel();
          notice(`Project 已创建并激活：${created.name}（ark/projects/${created.id}/，可直接在 Obsidian 编辑规则）`);
        })();
      },
    });
    this.addCommand({
      id: "generate-today-todos",
      name: "AI 生成今日待办",
      callback: () => {
        void (async () => {
          try {
            await runTodayTodos(this);
          } catch (e: any) {
            notice("今日待办生成失败: " + String(e?.message ?? e));
          }
        })();
      },
    });
    this.addCommand({
      id: "generate-daily-review",
      name: "生成每日回顾",
      callback: () => {
        void (async () => {
          try {
            await generateDailyReview(this);
            notice("每日回顾已生成");
          } catch (e: any) {
            notice("每日回顾生成失败: " + String(e?.message ?? e));
          }
        })();
      },
    });

    this.addCommand({
      id: "generate-weekly-review",
      name: "生成周报回顾",
      callback: () => {
        void (async () => {
          try {
            await generateWeeklyReview(this);
            notice("周报回顾已生成");
          } catch (e: any) {
            notice("周报回顾生成失败: " + String(e?.message ?? e));
          }
        })();
      },
    });

    this.addCommand({
      id: "generate-learning-quiz",
      name: "生成学习问答",
      callback: () => {
        void (async () => {
          try {
            const path = await generateQuiz(this);
            notice(path === "NOTHING" ? "今日无学习笔记可出题" : `学习问答已生成`);
          } catch (e: any) {
            notice("学习问答生成失败: " + String(e?.message ?? e));
          }
        })();
      },
    });
    this.addCommand({
      id: "grade-learning-quiz",
      name: "批改学习问答",
      callback: () => {
        void (async () => {
          try {
            const path = await gradeQuiz(this);
            notice(path === "NONE" ? "尚未生成学习问答" : path === "DONE" ? "学习问答均已批改" : "已批改");
          } catch (e: any) {
            notice("批改失败: " + String(e?.message ?? e));
          }
        })();
      },
    });
    this.addCommand({
      id: "scan-feeds",
      name: "获取订阅简报",
      callback: () => {
        void (async () => {
          try {
            const r = await scanFeeds(this);
            notice(`订阅新增 ${r.written} 篇${r.feedsFailed ? `，失败 ${r.feedsFailed} 个源` : ""}`);
          } catch (e: any) {
            notice("订阅获取失败: " + String(e?.message ?? e));
          }
        })();
      },
    });
    this.addCommand({
      id: "archive-feed-briefs",
      name: "一键归档简报",
      callback: () => {
        void (async () => {
          try {
            await archiveFeedBriefs(this);
          } catch (e: any) {
            notice("归档失败: " + String(e?.message ?? e));
          }
        })();
      },
    });
    this.addCommand({
      id: "run-pipeline",
      name: "处理简报（M3 管线）",
      callback: () => {
        void (async () => {
          try {
            const r = await runPipeline(this);
            notice(`处理简报：产要点卡 ${r.cards} 篇，忽略 ${r.skipped}，失败 ${r.failed}`);
          } catch (e: any) {
            notice("处理失败: " + String(e?.message ?? e));
          }
        })();
      },
    });
    this.addCommand({
      id: "daily-brief",
      name: "生成信息日报（M4）",
      callback: () => {
        void (async () => {
          try {
            const r = await generateDailyBrief(this);
            notice(`信息日报已${r.sources ? "更新" : "生成"}：${r.sources} 个来源、${r.items} 条`);
          } catch (e: any) {
            notice("生成失败: " + String(e?.message ?? e));
          }
        })();
      },
    });
    this.addCommand({
      id: "generate-moc",
      name: "生成主题 MOC 知识地图",
      callback: () => {
        void (async () => {
          const topic = await inputDialog(this, { title: "生成主题 MOC", placeholder: "输入主题，如 AI agent" });
          if (!topic || !topic.trim()) return;
          try {
            const r = await generateMoc(this, topic);
            notice(`MOC 已生成：命中 ${r.match} 条 → ${r.path}`);
          } catch (e: any) {
            notice("生成失败: " + String(e?.message ?? e));
          }
        })();
      },
    });
    this.addCommand({
      id: "run-case-lib",
      name: "沉淀案例库（M4）",
      callback: () => {
        void (async () => {
          try {
            const r = await runCaseLib(this);
            if (r.cards === 0) notice("暂无要点卡可直接沉淀");
            else notice(`案例库沉淀：扫 ${r.cards} 张要点卡 → 提炼 ${r.cases} 条案例，聚合 ${r.groups} 组`);
          } catch (e: any) {
            notice("沉淀失败: " + String(e?.message ?? e));
          }
        })();
      },
    });

    // 启动数据层闭环：确保目录 + 全量扫描 + 文件监听
    void setupSync(this);

    // F1 编辑器右键「AI 润色」
    installEditorAiMenu(this);
  }

  // ---- agentlab serve 服务管理（OPT-113） ----

  /** 健康探测 → 切 ribbon 图标/颜色/提示（探测失败静默按未运行处理） */
  private async refreshServeRibbon(): Promise<void> {
    if (!this.serveRibbonEl) return;
    const up = await serveUp(this.data.settings);
    setIcon(this.serveRibbonEl, up ? "server" : "server-off");
    this.serveRibbonEl.setAttribute("aria-label",
      up ? "agentlab serve：运行中（点击管理）" : "agentlab serve：未运行（点击拉起）");
    (this.serveRibbonEl as HTMLElement).style.color =
      up ? "var(--text-success)" : "var(--text-faint)";
  }

  /** ribbon 点击：未运行 → 直接拉起；运行中 → 菜单（停止 / 查看状态） */
  private async onServeRibbonClick(evt: MouseEvent): Promise<void> {
    if (this.serveBusy) {
      notice("已有服务操作在进行，请稍候…");
      return;
    }
    const s = this.data.settings;
    if (!(await serveUp(s))) {
      await this.serveAction("start");
      return;
    }
    const menu = new Menu();
    menu.addItem((mi) => mi
      .setTitle("停止 agentlab serve")
      .setIcon("server-off")
      .onClick(() => void this.serveAction("stop")));
    menu.addItem((mi) => mi
      .setTitle("查看运行状态")
      .setIcon("activity")
      .onClick(() => {
        void serveStatusText(s)
          .then((t) => notice(t || "serve_manage 无输出"))
          .catch((e: any) => notice("状态查询失败: " + String(e?.message ?? e)));
      }));
    menu.showAtMouseEvent(evt);
  }

  /** 启动/停止执行器：Notice 反馈 serve_manage 输出，完成后刷新按钮状态 */
  private async serveAction(action: "start" | "stop"): Promise<void> {
    if (this.serveBusy) {
      notice("已有服务操作在进行，请稍候…");
      return;
    }
    this.serveBusy = true;
    notice(action === "start" ? "正在拉起 agentlab serve…" : "正在停止 agentlab serve…");
    try {
      const out = action === "start"
        ? await serveStart(this.data.settings)
        : await serveStop(this.data.settings);
      notice(out || (action === "start" ? "agentlab serve 已启动" : "agentlab serve 已停止"));
    } catch (e: any) {
      notice(`${action === "start" ? "拉起" : "停止"} agentlab serve 失败：${e?.message ?? e}`);
    } finally {
      this.serveBusy = false;
      await this.refreshServeRibbon();
    }
  }

  async onunload() {
    this.app.workspace.detachLeavesOfType(VIEW_TYPE_SPACE_OS);
  }

  async loadArkData() {
    const stored = (await this.loadData()) as Partial<ArkData> | null;
    const base = emptyData();
    if (stored) {
      this.data = {
        ...base,
        ...stored,
        settings: normalizeSettings({ ...DEFAULT_SETTINGS, ...(stored.settings ?? {}) }),
      };
      // dashboard.md 强制入检索黑名单（M-D2：布局文件不参与搜索/回顾）
      const black = this.data.settings.scanBlacklist || [];
      if (!black.includes("dashboard.md")) {
        black.push("dashboard.md");
        this.data.settings.scanBlacklist = black;
      }
    } else {
      this.data = base;
    }
    // 历史 skin ID 只作为兼容输入，加载后统一使用 Ark 中性界面。
    this.data.settings = normalizeSettings(this.data.settings);
  }

  /** 持久化整个 data（settings + 对象集合） */
  async savePluginData() {
    await this.saveData(this.data);
  }

  async getSettings(): Promise<ArkSettings> {
    return this.data.settings;
  }

  async updateSettings(patch: Partial<ArkSettings>) {
    this.data.settings = normalizeSettings({ ...this.data.settings, ...patch });
    await this.savePluginData();
  }

  async activateView() {
    const { workspace } = this.app;
    let leaf: WorkspaceLeaf | null = null;
    const leaves = workspace.getLeavesOfType(VIEW_TYPE_SPACE_OS);
    if (leaves.length > 0) {
      leaf = leaves[0];
    } else {
      leaf = workspace.getLeaf(true);
      await leaf.setViewState({ type: VIEW_TYPE_SPACE_OS, active: true });
    }
    workspace.revealLeaf(leaf);
  }

  // ---- 待办数据操作（任务卡片 = .md，双向） ----
  async createTodoList(name: string): Promise<string> {
    const id = "list_" + Date.now().toString(36);
    this.data.todoLists.push({ id, name, tasks: [] });
    await this.savePluginData();
    return id;
  }

  async addTodoTask(listId: string, description: string, opts: {
    priority?: string; startDate?: string; dueDate?: string; effort?: number;
    dependency?: string[]; progress?: number; tags?: string[];
  } = {}) {
    const list = this.data.todoLists.find((l) => l.id === listId);
    if (!list) return;
    const task = {
      id: generateId(), description, completed: false, status: "todo" as TaskStatus,
      priority: (opts.priority ?? "medium") as any, tags: opts.tags ?? [], createdAt: Date.now(),
      startDate: opts.startDate, dueDate: opts.dueDate, effort: opts.effort,
      dependency: opts.dependency ?? [], progress: opts.progress ?? 0, listId,
    };
    await writeTask(this, list, task); // 写 md → import 进 data.todoLists
    await this.savePluginData();
  }

  async flipTask(listId: string, taskId: string) {
    const task = this.findTask(listId, taskId);
    const list = this.data.todoLists.find((l) => l.id === listId);
    if (!task || !list) return;
    task.completed = !task.completed;
    task.completedAt = task.completed ? Date.now() : undefined;
    task.status = task.completed ? "done" : "todo";
    await writeTask(this, list, task);
    if (task.completed) await createNextRecurrence(this, list, task);
    await this.savePluginData();
  }

  async setTaskStatus(listId: string, taskId: string, status: TaskStatus) {
    const task = this.findTask(listId, taskId);
    const list = this.data.todoLists.find((l) => l.id === listId);
    if (!task || !list) return;
    const becomingDone = status === "done" && task.status !== "done";
    task.status = status;
    task.completed = status === "done";
    task.completedAt = status === "done" ? Date.now() : undefined;
    await writeTask(this, list, task);
    if (becomingDone) await createNextRecurrence(this, list, task);
    await this.savePluginData();
  }

  async updateTask(listId: string, taskId: string, patch: Partial<any>) {
    const task = this.findTask(listId, taskId);
    const list = this.data.todoLists.find((l) => l.id === listId);
    if (!task || !list) return;
    Object.assign(task, patch);
    await writeTask(this, list, task);
    await this.savePluginData();
  }

  async deleteTask(listId: string, taskId: string) {
    const task = this.findTask(listId, taskId);
    const list = this.data.todoLists.find((l) => l.id === listId);
    if (!task || !list) return;
    if (task.notePath) {
      const f = this.app.vault.getAbstractFileByPath(task.notePath);
      if (f instanceof TFile) {
        await this.app.vault.delete(f); // watcher 触发 removeAt
        return;
      }
    }
    const i = list.tasks.indexOf(task);
    if (i >= 0) list.tasks.splice(i, 1);
    await this.savePluginData();
  }

  private findTask(listId: string, taskId: string) {
    const list = this.data.todoLists.find((l) => l.id === listId);
    if (!list) return undefined;
    return list.tasks.find((t) => t.id === taskId);
  }
}
