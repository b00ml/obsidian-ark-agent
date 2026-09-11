import { ItemView, WorkspaceLeaf, setIcon } from "obsidian";
import type ArkOSPlugin from "./main";
import { renderArk } from "./panels/ark";
import { renderTactical } from "./panels/tactical";
import { renderLogs } from "./panels/logs";
import { renderPanelsPlaceholder } from "./panels/placeholder";
import { renderDatabase } from "./panels/database";
import { renderIdeas } from "./panels/ideas";
import { renderDrawing } from "./panels/drawing";
import { renderComm } from "./panels/comm";
import { renderWeaponry } from "./panels/weaponry";
import { renderMedical } from "./panels/medical";
import { renderDockDashboard } from "./panels/dock";
import { renderDashboard } from "./panels/dashboard";
import { renderCapture } from "./panels/capture";
import { renderWorkshop } from "./panels/workshop";
import { launchAI } from "./crt";
import { settleAllLive } from "./agent-live";
import { openSettings } from "./settings-modal";
import { openSearch } from "./search";
import { getSkin } from "./skins";

export const VIEW_TYPE_SPACE_OS = "ark-view";

/** 顶层 Tab 顺序（含首页）。 */
export const TABS = [
  "dock",
  "dashboard",
  "ark",
  "tactical",
  "logs",
  "database",
  "weaponry",
  "medical",
  "ideas",
  "drawing",
  "comm",
  "capture",
  "workshop",
] as const;

export type TabName = (typeof TABS)[number];

/** 主导航。低频能力仍保留在“更多”菜单，不删除任何路由。 */
const ZERO_PRIMARY_TABS: readonly TabName[] = ["dock", "ark", "capture", "tactical", "database", "dashboard"];
// D4：生命体征/算力训练不再进入产品导航；旧 Tab 路由保留兼容，避免历史深链失效。
const ZERO_SECONDARY_TABS: readonly TabName[] = ["workshop", "logs", "drawing", "comm"];
/** 需要"整页不滚动、只有内部功能区滚动"的面板：主控大厅（Agent 工作台）。 */
const FIT_TABS: readonly TabName[] = ["ark"];

/** 面板标签（按当前设定取 icon+title；未知 tab 回退 tab 名） */
function tabLabel(skin: ReturnType<typeof getSkin>, tab: TabName): string {
  const t = skin.tabs[tab];
  return t ? `${t.icon} ${t.title}` : tab;
}

/** 是否对某 Tab 生效的渲染入口（内置各面板） */
function renderPanel(view: SpaceOSView, tab: TabName, mount: HTMLElement) {
  switch (tab) {
    case "dock":
      renderDockDashboard(view, mount);
      break;
    case "dashboard":
      renderDashboard(view, mount);
      break;
    case "ark":
      renderArk(view, mount);
      break;
    case "tactical":
      renderTactical(view, mount);
      break;
    case "logs":
      renderLogs(view, mount);
      break;
    case "database":
      renderDatabase(view, mount);
      break;
    case "ideas":
      renderIdeas(view, mount);
      break;
    case "drawing":
      renderDrawing(view, mount);
      break;
    case "comm":
      renderComm(view, mount);
      break;
    case "capture":
      renderCapture(view, mount);
      break;
    case "workshop":
      renderWorkshop(view, mount);
      break;
    case "weaponry":
      renderWeaponry(view, mount);
      break;
    case "medical":
      renderMedical(view, mount);
      break;
    default:
      renderPanelsPlaceholder(view, tab, mount);
  }
}

export class SpaceOSView extends ItemView {
  plugin: ArkOSPlugin;
  currentTab: TabName = "dock";
  initializedFor: Set<TabName> = new Set();

  constructor(leaf: WorkspaceLeaf, plugin: ArkOSPlugin) {
    super(leaf);
    this.plugin = plugin;
    // Ark 以首页作为第一页；AI 工作台保留为第二个高频入口。
    this.currentTab = "dock";
  }

  getViewType(): string {
    return VIEW_TYPE_SPACE_OS;
  }

  getDisplayText(): string {
    return getSkin(this.plugin.data.settings.skin).hints.displayText;
  }

  getIcon(): string {
    return "orbit";
  }

  async onOpen() {
    const root = this.containerEl.children[1] as HTMLElement;
    root.empty();
    root.addClass("space-os-root");
    root.setAttr("data-skin", getSkin(this.plugin.data.settings.skin).id); // 设定 → CSS 变量换色
    this.buildShell(root);
    // 直接渲染初始面板（currentTab 已为 "dock"，switchTab 会被卫语句拦截导致空白）
    await this.renderCurrentPanel();
  }

  async onClose() {
    // OPT-109：视图关闭兜底——所有在途流的半截回答结算进源会话并持久化（与 CRT onClose 同语义）
    settleAllLive(this.plugin.data.crtSessions);
    await this.plugin.savePluginData();
    this.contentEl.empty();
  }

  private buildShell(root: HTMLElement) {
    this.rootEl = root;
    const skin = getSkin(this.plugin.data.settings.skin);
    // 顶栏
    const tabbar = root.createDiv({ cls: "space-tabbar" });
    const primaryTabs = skin.id === "zero" ? ZERO_PRIMARY_TABS : TABS;
    primaryTabs.forEach((t) => {
      const btn = tabbar.createDiv({ cls: `space-tab-btn${t === this.currentTab ? " active" : ""}`, attr: { "data-tab": t } });
      btn.setText(tabLabel(skin, t));
    });
    if (skin.id === "zero") {
      const moreWrap = tabbar.createDiv({ cls: "space-tab-more" });
      const moreBtn = moreWrap.createDiv({ cls: "space-tab-btn space-tab-more-btn", text: "··· 更多" });
      const menu = moreWrap.createDiv({ cls: "space-more-menu", attr: { "aria-hidden": "true" } });
      ZERO_SECONDARY_TABS.forEach((t) => {
        const item = menu.createDiv({ cls: "space-more-item", text: tabLabel(skin, t), attr: { "data-tab": t } });
        item.addEventListener("click", (e) => {
          e.stopPropagation();
          this.switchTab(t);
          moreWrap.removeClass("open");
          menu.setAttr("aria-hidden", "true");
        });
      });
      moreBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const open = moreWrap.classList.toggle("open");
        menu.setAttr("aria-hidden", open ? "false" : "true");
      });
      if (ZERO_SECONDARY_TABS.includes(this.currentTab)) moreWrap.addClass("has-active");
    }
    const gear = tabbar.createDiv({ cls: "space-tab-btn space-tab-gear", text: "⚙ 设置" });
    gear.addEventListener("click", (e) => { e.stopPropagation(); openSettings(this.plugin); });
    const searchBtn = tabbar.createDiv({ cls: "space-tab-btn space-tab-gear", text: "⌕ 搜索" });
    searchBtn.addEventListener("click", (e) => { e.stopPropagation(); openSearch(this.plugin); });
    tabbar.addEventListener("click", (e) => {
      const el = (e.target as HTMLElement).closest(".space-tab-btn") as HTMLElement | null;
      if (!el?.dataset.tab) return;
      this.switchTab(el.dataset.tab as TabName);
    });
    this.tabbarEl = tabbar;

    // 面板容器
    const panel = root.createDiv({ cls: "space-panel-container" });
    this.panelEl = panel;
  }

  private tabbarEl!: HTMLElement;
  private panelEl!: HTMLElement;
  private rootEl!: HTMLElement;

  switchTab(tab: TabName) {
    if (this.currentTab === tab) return;
    this.currentTab = tab;
    this.tabbarEl.querySelectorAll(".space-tab-btn").forEach((el) => {
      el.toggleClass("active", (el as HTMLElement).dataset.tab === tab);
    });
    const more = this.tabbarEl.querySelector<HTMLElement>(".space-tab-more");
    if (more) {
      const secondary = ZERO_SECONDARY_TABS.includes(tab);
      more.classList.toggle("has-active", secondary);
      more.querySelectorAll(".space-more-item").forEach((el) => {
        el.classList.toggle("active", (el as HTMLElement).dataset.tab === tab);
      });
      if (!secondary) {
        more.classList.remove("open");
        more.setAttr("aria-hidden", "true");
      }
    }
    this.renderCurrentPanel();
  }

  async renderCurrentPanel() {
    const mount = this.panelEl;
    // Each panel owns its root class; clear the previous panel marker before
    // reusing the shared mount so page-level layout rules cannot leak across tabs.
    mount.className = "space-panel-container";
    mount.empty();
    renderPanel(this, this.currentTab, mount);
    // 主控大厅整页不收滚动条：根节点固定高度、内部功能区各自滚动。
    const fit = FIT_TABS.includes(this.currentTab);
    this.rootEl?.toggleClass("ark-fit-panel", fit);
    // 同时把叶子容器变成纵向 flex，让根节点通过 flex 拿到确定高度，
    // 不依赖 height:100% 在宿主主题下能否解析。
    this.containerEl.toggleClass("ark-fit-shell", fit);
    this.initializedFor.add(this.currentTab);
  }

  /** 唤出 AI 助手：优先 Hermes Desktop，回退 CRT */
  showAI() {
    launchAI(this.plugin);
  }
}
