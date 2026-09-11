import type { Contact, TodoList, LogEntry, DbCategory, AutoContact } from "./types";

/** 邮件账户 */
export interface MailAccountConfig {
  id: string;
  name: string;
  email: string;
  password?: string;
  type: "imap" | "local";
  imapHost?: string;
  imapPort: number;
  imapSsl: boolean;
  smtpHost?: string;
  smtpPort: number;
  smtpSsl: boolean;
  color: string;
}

/** 提醒预设 */
export interface ReminderPreset {
  value: number; // 提前分钟数
  label: string;
}

/** 信息订阅源（知识信息闭环 ① 主动获取） */
export interface FeedConfig {
  id: string;
  name: string;                    // 订阅名，如 "每日AI"
  source: "rss" | "url" | "bilibili" | "wechat" | "keyword_search";
  target: string;                  // 源地址（RSS/URL）；bilibili=UP主/BV前缀；wechat=主页；keyword_search=搜索关键词
  keywords: string[];              // 内容过滤与热度匹配关键词
  weight: number;                  // 排序权重，越大越靠前
  enabled: boolean;
  lastScan?: number;               // 上次扫描时间(ms)，供增量
  lastError?: string;
}

/** 插件设置（对应原插件 settings / DEFAULT_SETTINGS） */
export interface ArkSettings {
  captainName: string;
  captainHeight: number;

  faultLogFolder: string;
  normalLogFolder: string;
  databaseFolder: string;
  reportFolder: string;
  healthFolder: string;
  todoFolder: string;
  ideaFolder: string;
  drawingFolder: string;
  battleReportFolder: string;
  weaponFolder: string;
  mailFolder: string;

  // AI 助手
  aiProvider: string;
  aiApiUrl: string;
  aiApiKey: string;
  aiModel: string;
  aiTemperature: number;
  aiFolder: string;
  aiContextLength: number;
  aiMaxRounds: number;
  reviewFolder: string;    // 每日回顾 / 周报回顾 生成文件的目录

  // 学习问答（知识信息闭环 M1，§3.3）
  learningFolder: string;  // 今日修改笔记的来源目录（从其出题）
  quizFolder: string;      // 学习问答/批改 文件目录
  quizQuestionCount: number;
  quizDifficulty: string;

  // 信息订阅（知识信息闭环 M2，§3.1）
  feeds: FeedConfig[];
  feedSeen: string[];      // 已处理 URL 指纹（去重）
  feedInboxFolder: string; // 简述+原链接 产物落盘目录
  feedArchiveFolder: string; // 简报归档目录（一键归档目标，默认 02-DB/简报）
  pipelineFolder: string; // M3 处理管线产出要点卡片目录（默认 02-DB/要点）
  mocFolder: string; // M4 MOC 知识地图页目录（默认 02-DB/主题）
  caseFolder: string; // M4 案例库/决策备忘 沉淀目录（默认 02-DB/案例）

  // Agent 内核（外部常驻）
  agentProvider: string;   // "" = 直连；"hermes" = Hermes Agent 外部常驻；"agentlab" = 自研 agentlab serve
  hermesUrl: string;       // 如 http://127.0.0.1:8642/v1
  hermesToken: string;     // API_SERVER_KEY
  hermesModel: string;     // profile 名（/v1/models 实测为 hermes-agent）
  // agentlab serve（OPT-067：Hermes 兼容 SSE，ark 解析零改动复用 agentChat）
  agentlabUrl: string;     // 如 http://127.0.0.1:8643/v1
  agentlabToken: string;   // serve.token / AGENTLAB_SERVE_TOKEN
  agentlabModel: string;   // agentlab 内 model 名
  agentlabExePath: string;  // 拉起用的 python 可执行文件（留空自动探测项目 .venv）
  agentlabWorkdir: string;  // serve 启动目录（须含 config/config.json；留空默认项目 agentlab 目录）
  // P2-2/F5-021：审批策略。risk_based=只有 danger 工具（含 @ 多 Agent 门禁）需要人工确认；
  // allow_all=策略自动放行已注册工具，不再弹卡（Bearer/工具注册/Vault 路径守卫仍然生效）。
  // 启动 serve 时以 AGENTLAB_APPROVAL_MODE 环境变量下发，不写后端 config.json（该文件含 API Key）。
  approvalMode: "risk_based" | "allow_all";
  desktopExePath: string;   // Hermes Desktop：仅 hermes 模式"AI 助手"一键拉起用，留空自动探测

  // Project 长期任务空间（P0-2/OPT-107）：Vault 内 ark/projects/<id>/ 存 AGENTS.md + project.md
  projects: ProjectInfo[];
  activeProjectId: string;  // 当前激活项目（"" = 全局，不注入项目上下文）

  // 兼容字段：产品界面统一使用默认中性文案，历史 skin ID 仍可读取。
  skin: string;

  // 通信 / 邮件
  mailAccountsConfig: MailAccountConfig[];
  defaultMailAccountId?: string;
  autoSyncInterval: number; // 分钟，0=关闭
  reminderEmail: string;
  defaultReminderOffset: number;
  calendarInviteEnabled: boolean;
  reminderEmailEnabled: boolean;
  reminderPresets: ReminderPreset[];

  // 数据库分类
  databaseCategories: DbCategory[];

  // 背景（首页）
  backgroundImagePath: string;
  backgroundVideoPath: string;
  backgroundMode: "image" | "video";

  // 首页 dashboard 布局文件（§16.1 M-D2，Note-as-Layout）
  dashboardFile: string;

  scanBlacklist: string[];

  // 通讯录自动采集轨道
  contacts: AutoContact[];
}

/** Project 长期任务空间（P0-2/OPT-107）：id 即 Vault 目录名 ark/projects/<id>/ */
export interface ProjectInfo {
  id: string;      // slug（与后端白名单一致：中英文/数字/_/-，1~64）
  name: string;    // 显示名
  createdAt: number;
}

/** CRT 单条会话记录（前端持久化，落 data.json；与 serve 解耦） */
export interface CrtSession {
  id: string;                // 短ID（generateId），路由/删除键
  title: string;             // 首条用户消息 ≤24 字
  createdAt: number;         // 毫秒
  updatedAt: number;         // 毫秒，淘汰排序用
  projectId?: string;        // 所属 Project（P0-2/OPT-107）；undefined = 全局会话
  messages: { role: "system" | "user" | "assistant" | "tool"; content: string; name?: string }[];
}

/** 插件运行时数据（对应原插件 this.data） */
export interface ArkData {
  settings: ArkSettings;
  crtSessions: CrtSession[];      // CRT 会话列表（上限 50，按 updatedAt 淘汰最旧）
  crtActiveId: string | null;     // 续聊指针：最近会话
  todoLists: TodoList[];
  logs: LogEntry[];
  database: any[];
  ideas: any[];
  drawings: any[];
  healthReports: any[];
  healthRecords: any[];
  weapons: any[];
  weaponResults: any[];
  contactList: Contact[];
}

export const DEFAULT_REMINDER_PRESETS: ReminderPreset[] = [
  { value: 0, label: "即时" },
  { value: 5, label: "提前 5 分钟" },
  { value: 15, label: "提前 15 分钟" },
  { value: 30, label: "提前 30 分钟" },
  { value: 60, label: "提前 1 小时" },
  { value: 120, label: "提前 2 小时" },
  { value: 1440, label: "提前 1 天" },
  { value: 4320, label: "提前 3 天" },
];

export const DEFAULT_SETTINGS: ArkSettings = {
  captainName: "用户",
  captainHeight: 170,

  faultLogFolder: "01-日志/问题",
  normalLogFolder: "01-日志/工作",
  databaseFolder: "02-DB",
  reportFolder: "02-DB/报告",
  healthFolder: "02-DB/健康",
  todoFolder: "03-待办",
  ideaFolder: "04-灵感",
  drawingFolder: "05-随笔",
  battleReportFolder: "02-DB/结果",
  weaponFolder: "06-模板",
  mailFolder: "02-DB/消息",

  aiProvider: "",
  aiApiUrl: "https://api.deepseek.com/chat/completions",
  aiApiKey: "",
  aiModel: "deepseek-chat",
  aiTemperature: 0.3,
  aiFolder: "02-DB/AI对话",
  aiContextLength: 4096,
  aiMaxRounds: 10,
  reviewFolder: "02-DB/回顾",

  learningFolder: "02-DB/学习",
  quizFolder: "02-DB/问答",
  quizQuestionCount: 5,
  quizDifficulty: "中等",

  feeds: [],
  feedSeen: [],
  feedInboxFolder: "Inbox",
  feedArchiveFolder: "02-DB/简报",
  pipelineFolder: "02-DB/要点",
  mocFolder: "02-DB/主题",
  caseFolder: "02-DB/案例",

  agentProvider: "",
  hermesUrl: "http://127.0.0.1:8642/v1",
  hermesToken: "",
  hermesModel: "hermes-agent",
  agentlabUrl: "http://127.0.0.1:8643/v1",
  agentlabToken: "",
  agentlabModel: "agentlab-demo",
  agentlabExePath: "",
  agentlabWorkdir: "",
  approvalMode: "risk_based",

  projects: [],
  activeProjectId: "",

  desktopExePath: "",

  // 内部 ID 保留用于兼容旧配置；用户界面使用中性产品文案。
  skin: "zero",

  mailAccountsConfig: [],
  autoSyncInterval: 5,
  reminderEmail: "",
  defaultReminderOffset: 1440,
  calendarInviteEnabled: true,
  reminderEmailEnabled: true,
  reminderPresets: [...DEFAULT_REMINDER_PRESETS],

  databaseCategories: [],
  backgroundImagePath: "",
  backgroundVideoPath: "",
  backgroundMode: "image",
  dashboardFile: "dashboard.md",
  scanBlacklist: ["dashboard.md"],

  contacts: [],
};

/** 初始运行时数据 */
export function emptyData(): ArkData {
  return { ...structuredCloneBase(DEFAULT_SETTINGS) };
}

/** 将历史界面配置归一为产品唯一的中性界面。 */
export function normalizeSettings(settings: ArkSettings): ArkSettings {
  // approvalMode 缺失/非法一律回落 risk_based（fail-closed）：宁可多弹一次确认，
  // 不能因为配置字段拼错而让危险工具与 @ 多 Agent 门禁静默放开。
  const mode = settings.approvalMode === "allow_all" ? "allow_all" : "risk_based";
  const captainName = settings.captainName === "舰长" ? "用户" : settings.captainName;
  return { ...settings, skin: "zero", captainName, approvalMode: mode };
}

function structuredCloneBase(s: ArkSettings): ArkData {
  return {
    settings: structuredClone(s),
    crtSessions: [],
    crtActiveId: null,
    todoLists: [],
    logs: [],
    database: [],
    ideas: [],
    drawings: [],
    healthReports: [],
    healthRecords: [],
    weapons: [],
    weaponResults: [],
    contactList: [],
  };
}

/** AI provider 预设（对应原 getAIProviderDefaults） */
export function aiProviderDefaults(provider: string): { url: string; model: string; key: string } {
  switch (provider || "deepseek") {
    case "openai":
      return { url: "https://api.openai.com/v1/chat/completions", model: "gpt-4o-mini", key: "" };
    case "doubao":
      return { url: "https://ark.cn-beijing.volces.com/api/v3/chat/completions", model: "doubao-pro-32k", key: "" };
    case "ollama":
      return { url: "http://localhost:11434/v1/chat/completions", model: "qwen2.5:7b", key: "ollama" };
    case "custom":
      return { url: "", model: "", key: "" };
    default:
      return { url: "https://api.deepseek.com/chat/completions", model: "deepseek-chat", key: "" };
  }
}
