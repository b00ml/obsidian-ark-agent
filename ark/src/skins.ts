// Ark 的用户界面文案与视觉主题。
// 旧版主题 ID 仍由 settings.ts 兼容读取，但产品只保留一套中性文案。

export type SkinId = "zero";

export interface SkinTab {
  icon: string;
  title: string;
  desc: string;
}

export interface SkinHints {
  ready: string;        // 首页副标题（点击唤醒 AI）
  crtTitle: string;     // CRT 终端标题
  settingsTitle: string; // 设置页标题
  displayText: string;  // 视图/ribbon 显示名
  captainLabel: string; // 设置页「称呼」字段标签
}

/** 各面板页内文案组。 */
export interface SkinPanel {
  taskCols: [string, string, string, string]; // 任务看板四列（待执行/进行中/阻塞/已完成）
  taskEmpty: string;                           // 任务面板空态
  logNormal: string;                           // 日志面板：普通记录 tab
  logFault: string;                            // 日志面板：故障/问题 tab
  logStats: [string, string, string, string];  // 日志面板统计（总/普通/故障/今日）
  commLoading: string;                         // 通信面板加载占位
  weaponRecord: string;                        // 训练面板：战绩/成绩记录标题
  dock: {
    bannerGreeting: string;                    // 问候语模板（含 {greeting} 时段 {captain} 称呼）
    kpi: { todo: string; log: string; idea: string; mail: string }; // KPI 卡片标签（对象键定址）
    todoTitle: string;                         // 内嵌「今日待办」标题
    memoTitle: string;                         // 内嵌「灵感速记」标题
    notesTitle: string;                        // 内嵌「快捷笔记」标题
    recentTitle: string;                       // 底部「最近动态」标题
    loading: string;                           // 数据未就绪加载占位文案
    empty: string;                             // 空态文案
  };
}

/** AI 工作台首页文案组。 */
export interface ArkCopy {
  hudLabel: string;      // HUD 顶部小标签
  hudValue: string;      // HUD 大标题
  tagline: string;       // 底部标语
  stats: [string, string, string, string]; // 统计标签（待办/通讯/资产/灵感）
  cardTodo: string;      // 左卡标题
  cardTodoBody: string;  // 左卡正文
  cardTodoLink: string;  // 左卡跳转
  cardComm: string;      // 右卡标题
  cardCommBody: string;  // 右卡正文
  cardCommLink: string;  // 右卡跳转
  radarLabel: string;    // 底部雷达标签
  enter: string;         // 首页：欢迎/进入语
}

export interface Skin {
  id: SkinId;
  name: string;
  brand: string;
  systemPrompt: string; // AI 助手人设（Hermes/CRT 会话注入）
  tabs: Record<string, SkinTab>;
  hints: SkinHints;
  ark: ArkCopy;
  panel: SkinPanel;
}

const ark: Skin = {
  id: "zero",
  name: "Ark",
  brand: "Ark",
  systemPrompt:
    "你是 Ark 知识工作台中的 AI 助手。你的职责是协助用户处理任务、整理资料、管理知识库、沉淀灵感与复盘工作。回答简洁、专业、直接，使用与用户相同的语言。",
  tabs: {
    dock: { icon: "🏠", title: "首页", desc: "工作区首页" },
    dashboard: { icon: "📈", title: "运行状态", desc: "Agent 运行与知识库状态" },
    ark: { icon: "💬", title: "AI 工作台", desc: "AI 对话与任务协作" },
    tactical: { icon: "✅", title: "任务管理", desc: "任务清单与排期计划" },
    logs: { icon: "📋", title: "工作记录", desc: "工作记录、日报与周报" },
    database: { icon: "📚", title: "知识库", desc: "资料存储、检索与引用" },
    weaponry: { icon: "🎯", title: "训练记录", desc: "训练与结果记录" },
    medical: { icon: "💚", title: "健康记录", desc: "健康数据记录" },
    ideas: { icon: "💡", title: "灵感草稿", desc: "灵感收集与归档" },
    drawing: { icon: "📝", title: "随笔", desc: "随笔与画板记录" },
    comm: { icon: "✉️", title: "消息与邮箱", desc: "消息收发与联系人" },
    capture: { icon: "📥", title: "采集箱", desc: "统一输入与处理状态" },
    workshop: { icon: "🧰", title: "内容产出", desc: "生成、浏览与复用知识产物" },
  },
  hints: {
    ready: "系统已就绪 · 开始处理你的知识与任务 ▸_",
    crtTitle: "Ark · AI 工作台",
    settingsTitle: "⚙ Ark 设置",
    displayText: "Ark",
    captainLabel: "用户称呼",
  },
  ark: {
    hudLabel: "WORKSPACE OVERVIEW",
    hudValue: "工作台概览",
    tagline: "READY TO WORK",
    stats: ["任务", "消息", "知识", "灵感"],
    cardTodo: "◈ 任务队列",
    cardTodoBody: "正在加载任务…",
    cardTodoLink: "查看任务 ▸",
    cardComm: "✉ 最近消息",
    cardCommBody: "正在加载消息…",
    cardCommLink: "查看消息 ▸",
    radarLabel: "工作区状态",
    enter: "打开工作台",
  },
  panel: {
    taskCols: ["待处理", "进行中", "已阻塞", "已完成"],
    taskEmpty: "创建任务以开始工作 ／ 暂无任务",
    logNormal: "工作记录",
    logFault: "问题记录",
    logStats: ["总记录", "工作", "问题", "今日"],
    commLoading: "正在加载消息…",
    weaponRecord: "训练记录",
    dock: {
      bannerGreeting: "{greeting}，{captain}",
      kpi: { todo: "任务", log: "记录", idea: "灵感", mail: "消息" },
      todoTitle: "今日任务",
      memoTitle: "灵感速记",
      notesTitle: "快捷笔记",
      recentTitle: "最近动态",
      loading: "正在同步数据…",
      empty: "暂无数据",
    },
  },
};

const SKINS: Record<SkinId, Skin> = { zero: ark };

export const SKIN_IDS: SkinId[] = ["zero"];

/** 历史配置中的旧主题 ID 统一回退到 Ark 的中性界面。 */
export function getSkin(_id?: string): Skin {
  return SKINS.zero;
}
