// P2-2/F5-019：@ 多 Agent 指派的共享解析层。
//
// 设计约束（见 docs/roadmap/P2-2多Agent实施规划.md §2 决策 D3 与 §6 风险表）：
// · `@name` 只在**行首或空白之后**识别，避免邮箱（a@b）与双链里的 @ 被误判；
// · name 必须命中 `GET /v1/agents` 返回的名单才转入 multi，否则按普通文本发送；
// · 命中的 mention 从提问正文中**剥离**——答者只收到裸问题，名单单独走 multi 字段；
// · 去重且保持出现顺序（服务端按顺序裁到并发上限，多余进 skipped）。

import type { ArkSettings } from "./settings";
import { agentEndpoint, getJsonWithFallback } from "./ai";

export interface AgentInfo {
  name: string;
  kind: "internal" | "external";
  enabled: boolean;
  description?: string;
}

export interface AgentCatalog {
  agents: AgentInfo[];
  maxParallelConsults: number;
  consultResultMaxChars: number;
  /** 拿不到名单时的原因（离线/未授权/旧版服务端），供 UI 明示而非静默降级。 */
  error?: string;
}

export interface MentionParse {
  /** 命中的 agent 名（去重、保序）。 */
  multi: string[];
  /** 剥离 mention 后的提问正文。 */
  cleaned: string;
  /** 出现但不在名单里的 @token，供 UI 提示"未识别，按普通文本发送"。 */
  unknown: string[];
}

/** 拉取可 @ 的 agent 名单（`GET /v1/agents`，P2-2/F5-019 契约）。 */
export async function fetchAgentCatalog(settings: ArkSettings): Promise<AgentCatalog> {
  const ep = agentEndpoint(settings);
  const empty: AgentCatalog = {
    agents: [], maxParallelConsults: 3, consultResultMaxChars: 6000,
  };
  if (!ep.token) return { ...empty, error: "未配置 Agent 内核 Token" };
  // 走 ai.ts 的统一出口：fetch 失败自动回退 requestUrl（不经 CORS 预检、不受 CSP 限制）
  const got = await getJsonWithFallback(`${ep.origin}/v1/agents`,
    { Authorization: `Bearer ${ep.token}` });
  if (!got) {
    return { ...empty, error: "无法连接 Agent 内核（fetch 与 requestUrl 均失败，请确认 serve 在运行）" };
  }
  if (got.status < 200 || got.status >= 300) {
    return { ...empty, error: `名单接口返回 ${got.status}（旧版 serve 可能没有 /v1/agents，请重启 Agent 内核）` };
  }
  const data = (got.json ?? {}) as {
    agents?: AgentInfo[]; max_parallel_consults?: number; consult_result_max_chars?: number;
  };
  return {
    agents: Array.isArray(data.agents) ? data.agents.filter((a) => a?.name) : [],
    maxParallelConsults: Number(data.max_parallel_consults) || 3,
    consultResultMaxChars: Number(data.consult_result_max_chars) || 6000,
  };
}

// 名单缓存：@ 补全与发送前解析都会用，避免每次输入都打一次 HTTP（60s TTL，可强制刷新）。
let catalogCache: { at: number; catalog: AgentCatalog } | null = null;
const CATALOG_TTL_MS = 60_000;

export async function getAgentCatalog(settings: ArkSettings, force = false): Promise<AgentCatalog> {
  const now = Date.now();
  if (!force && catalogCache && now - catalogCache.at < CATALOG_TTL_MS) {
    return catalogCache.catalog;
  }
  const catalog = await fetchAgentCatalog(settings);
  catalogCache = { at: now, catalog };
  return catalog;
}

/** 测试/设置变更后清缓存（改端点或 token 时需要重新拉）。 */
export function resetAgentCatalogCache(): void {
  catalogCache = null;
}

/**
 * 解析正文里的 @mention。
 * 只认行首或空白后的 `@name`；name 由 `[A-Za-z0-9_.-]` 组成（与 serve 的 sid/agent 命名同字符集）。
 */
export function parseMentions(text: string, known: string[]): MentionParse {
  const allowed = new Set(known);
  const seen = new Set<string>();
  const multi: string[] = [];
  const unknown: string[] = [];
  // 末尾的 `[ \t]?` 顺手吃掉 mention 后紧跟的一个空格，避免剥离后残留 " 问 B" 这类前导空白；
  // 只吃空格/制表符、不吃换行——换行属于用户正文结构。
  const cleaned = text.replace(/(^|\s)@([A-Za-z0-9_.-]+)[ \t]?/g, (whole, lead: string, name: string) => {
    if (!allowed.has(name)) {
      if (!unknown.includes(name)) unknown.push(name);
      return whole; // 未知名保持原样：它是用户正文的一部分，不该被吞掉
    }
    if (!seen.has(name)) {
      seen.add(name);
      multi.push(name);
    }
    return lead; // 剥离 mention，仅保留前导空白以维持词间距
  });
  // 只去掉 mention 本身与首尾空白；**不得**压缩内部空白——用户用 Shift+Enter 写的
  // 多行正文（贴代码、列清单）必须原样保留，之前这里做过 \s+ → " " 的压缩，属于数据损失。
  return { multi, cleaned: cleaned.trim(), unknown };
}

/** 消息流里的分支进度（由 SSE `response.branch.*` 驱动）。 */
export interface BranchProgress {
  agent: string;
  kind: string;
  status: "running" | "ok" | "failed";
  elapsed?: number;
  chars?: number;
  truncated?: boolean;
  ref?: string;
  error?: string;
}

/** 汇总统计（由 SSE `response.multi.summary` 驱动）。 */
export interface MultiSummary {
  okCount: number;
  failed: string[];
  skipped: string[];
  denied: boolean;
  synthesized: boolean;
  synthError?: string;
}

/** 把 SSE 事件映射为分支进度增量；不认识的事件返回 null（未知事件一律跳过）。 */
export function branchEventToProgress(ev: Record<string, unknown>): { agent: string; patch: BranchProgress } | null {
  const type = String(ev.type || "");
  const agent = String(ev.agent || "");
  if (!agent) return null;
  if (type === "response.branch.started") {
    return { agent, patch: { agent, kind: String(ev.kind || ""), status: "running" } };
  }
  if (type === "response.branch.done") {
    return {
      agent,
      patch: {
        agent, kind: String(ev.kind || ""), status: "ok",
        elapsed: Number(ev.elapsed) || 0,
        chars: Number(ev.chars) || 0,
        truncated: Boolean(ev.truncated),
        ref: typeof ev.ref === "string" ? ev.ref : undefined,
      },
    };
  }
  if (type === "response.branch.failed") {
    return {
      agent,
      patch: {
        agent, kind: String(ev.kind || ""), status: "failed",
        elapsed: Number(ev.elapsed) || 0,
        error: String(ev.error || "未知错误"),
      },
    };
  }
  return null;
}
