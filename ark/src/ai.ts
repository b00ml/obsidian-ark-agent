import { requestUrl } from "obsidian";
import type { ArkSettings } from "./settings";
import { aiProviderDefaults } from "./settings";

export interface AiMessage {
  // "tool"（W2/OPT-110）：工作台历史含工具卡片行；服务端 history() 解析时自动忽略 tool 行
  role: "system" | "user" | "assistant" | "tool";
  content: string;
}

export interface ChatOptions {
  temperature?: number;
  maxTokens?: number;
  /** 超时兜底（对齐 daily.ts 的 AbortController 做法） */
  signal?: AbortSignal;
}

/** 归一化的 POST 响应（fetch 与 requestUrl 兜底两种来源） */
interface ChatResp {
  ok: boolean;
  status: number;
  text(): Promise<string>;
  json(): Promise<any>;
}

/**
 * POST 到 OpenAI 兼容 /chat/completions。
 * 个别 Obsidian CSP 会拦原生 fetch（旧壳 OPT-038 改 requestUrl）：
 * fetch 抛 CSP/网络错误时自动换 requestUrl() 重试一次；AbortError 直接上抛，不降级。
 */
async function postChat(
  url: string,
  headers: Record<string, string>,
  body: string,
  signal?: AbortSignal,
): Promise<ChatResp> {
  try {
    const resp = await fetch(url, { method: "POST", headers, body, signal });
    return {
      ok: resp.ok,
      status: resp.status,
      text: () => resp.text(),
      json: () => resp.json(),
    };
  } catch (err: any) {
    if (err?.name === "AbortError") throw err; // 超时取消，不降级
    const r: any = await requestUrl({ url, method: "POST", headers, body, contentType: "application/json" });
    const text = typeof r.text === "function" ? await r.text() : String(r.text ?? "");
    let json: any = r.json;
    if (typeof r.json === "function") json = await r.json();
    const ok = r.status >= 200 && r.status < 300;
    return { ok, status: r.status, text: async () => text, json: async () => json };
  }
}

/** 通用 OpenAI 兼容 /chat/completions 调用 */
export async function chat(
  settings: ArkSettings,
  messages: AiMessage[],
  opts: ChatOptions = {},
): Promise<string> {
  const provider = settings.aiProvider || "deepseek";
  const defaults = aiProviderDefaults(provider);
  const url = settings.aiApiUrl || defaults.url;
  const apiKey = settings.aiApiKey;
  const model = settings.aiModel || defaults.model;
  const temperature = opts.temperature ?? settings.aiTemperature ?? 0.3;

  if (!apiKey) {
    throw new Error("未配置 API Key。请在 AI 助手中用 /config key 或用设置中心填写。");
  }
  if (!url) {
    throw new Error("未配置 API 地址。");
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${apiKey}`,
  };
  // Ollama 本地无需 Bearer
  if (provider === "ollama") {
    delete headers.Authorization;
  }

  const resp = await postChat(url, headers, JSON.stringify({
    model,
    messages,
    temperature,
    max_tokens: opts.maxTokens ?? settings.aiContextLength ?? 4096,
  }), opts.signal);

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`AI 调用失败 ${resp.status}: ${text.slice(0, 300)}`);
  }

  const json: any = await resp.json();
  const content = json?.choices?.[0]?.message?.content;
  if (typeof content !== "string") {
    throw new Error("AI 响应缺少内容字段。");
  }
  return content;
}

// ── Hermes / agentlab Agent（外部常驻）──────────────────────────────

/** Hermes 回调：onText 流式文本；onTool 工具轨迹（start/done）；onStatus 心跳（agentlab serve 进度） */
export interface HermesCallbacks {
  onText?: (delta: string) => void;
  onTool?: (name: string, phase: "start" | "done", detail?: string) => void;
  onStatus?: (s: { elapsed: number }) => void;
  /** F5-018：risk_based 下服务端等待人工审批；requested/resolved 均透传给工作台。 */
  onApproval?: (event: { phase: "requested" | "resolved"; approval: any }) => void;
  /** W2/OPT-110：completed 事件携带折叠后的最终历史快照（去 system）——
   *  调用方用它替换本地历史，让 nudge/compress/硬截断产生的压缩跨轮持久。 */
  onCompleted?: (info: { summary?: any; history?: { role: string; content: string }[] }) => void;
  /** P2-2/F5-019：多 Agent 分支进度（response.branch.*）。 */
  onBranch?: (ev: Record<string, unknown>) => void;
  /** P2-2/F5-019：多 Agent 汇总统计（response.multi.summary）。 */
  onMultiSummary?: (ev: Record<string, unknown>) => void;
}

/** 「Agent 内核」连接信息：由 settings 按 agentProvider 解析（hermes / agentlab 共用同一套 /v1/responses SSE） */
export interface AgentEndpoint {
  label: string;              // 展示名
  base: string;               // 如 http://127.0.0.1:8642/v1
  origin: string;             // origin(去 /v1)，拼 /health 用
  responses: string;          // /v1/responses
  health: string;             // /health
  token: string;
  model: string;
}

/** 解析当前启用的 Agent 内核端点：agentlab → agentlab*，否则回退 hermes*。 */
export function agentEndpoint(settings: ArkSettings): AgentEndpoint {
  const isLab = settings.agentProvider === "agentlab";
  const base = (isLab ? settings.agentlabUrl : settings.hermesUrl || "http://127.0.0.1:8642/v1").replace(/\/+$/, "");
  const origin = base.replace(/\/v1.*$/, "");
  return {
    label: isLab ? "agentlab" : "Hermes",
    base,
    origin,
    responses: `${origin}/v1/responses`,
    health: `${origin}/health`,
    token: isLab ? settings.agentlabToken : settings.hermesToken,
    model: isLab ? (settings.agentlabModel || "agentlab-demo")
                  : (settings.hermesModel || "hermes-agent"),
  };
}

/** GET 探活：同 postChat 的 CSP 兜底——个别 Obsidian CSP 拦原生 fetch（含 localhost），
 *  fetch 抛错时换 requestUrl() 重试一次；否则 CRT 开屏探测恒报 "Failed to fetch"
 *  而对话（走 postChat 兜底）正常，出现"未运行但能聊"的错位。 */
async function getJson(url: string): Promise<ChatResp> {
  try {
    const resp = await fetch(url, { method: "GET" });
    return {
      ok: resp.ok,
      status: resp.status,
      text: () => resp.text(),
      json: () => resp.json(),
    };
  } catch {
    const r: any = await requestUrl({ url, method: "GET" });
    const text = typeof r.text === "function" ? await r.text() : String(r.text ?? "");
    let json: any = r.json;
    if (typeof r.json === "function") json = await r.json();
    const ok = r.status >= 200 && r.status < 300;
    return { ok, status: r.status, text: async () => text, json: async () => json };
  }
}

/** 探测当前 Agent 内核存活（M2 手动拉起判定；agentlab serve 同走 /health） */
export async function probeAgent(settings: ArkSettings): Promise<{ ok: boolean; version?: string; error?: string }> {
  const { health, label } = agentEndpoint(settings);
  try {
    const resp = await getJson(health);
    if (!resp.ok) return { ok: false, error: `HTTP ${resp.status}` };
    const j: any = await resp.json();
    return { ok: true, version: j?.version };
  } catch (err: any) {
    return { ok: false, error: `${label} 未运行（${String(err?.message ?? err)}）` };
  }
}

/** 兼容旧名：探测 Hermes gateway 存活 */
export async function probeHermes(settings: ArkSettings): Promise<{ ok: boolean; version?: string; error?: string }> {
  return probeAgent(settings);
}

/** /v1/runs 单条 run 概览（serve 概览字段，OPT-126 契约） */
export interface AgentRunSummary {
  trace_id: string;
  run_id?: string;
  session_id?: string;
  project_id?: string;
  time: string;
  input: string;
  stop_reason: string;
  tokens: number | null;
  steps: number;
  tools: number;
  error?: string;
  cancel_reason?: string;
}

export interface AgentRunDetail {
  trace_id: string;
  run: Record<string, any>;
  events: Record<string, any>[];
}

/** #6/OPT-133：拉取最近 run 概览（Dashboard Runs 卡数据源）。
 *  非 agentlab 后端（无此端点）返回 null；网络/非 2xx/解析失败一律 null，由调用方展示降级文案。 */
export async function fetchAgentRuns(settings: ArkSettings, limit = 20): Promise<AgentRunSummary[] | null> {
  if (settings.agentProvider !== "agentlab") return null;
  const ep = agentEndpoint(settings);
  const url = `${ep.origin}/v1/runs?limit=${limit}`;
  const headers: Record<string, string> = ep.token ? { Authorization: `Bearer ${ep.token}` } : {};
  const got = await getJsonWithFallback(url, headers);
  if (!got || got.status < 200 || got.status >= 300) return null;
  return Array.isArray(got.json?.runs) ? (got.json.runs as AgentRunSummary[]) : null;
}

/**
 * 带鉴权头的 GET：fetch 优先，失败回退 Obsidian 的 `requestUrl()`。
 *
 * 两个失败面都要兜：① Obsidian CSP 拦原生 fetch（旧壳 OPT-038 起沿用此惯例）；
 * ② 浏览器对带 Authorization 的跨域请求先发 OPTIONS 预检，服务端未注册该路由时
 * 预检 405 → fetch 直接抛 "Failed to fetch"（OPT-187 修的就是这一类）。
 * `requestUrl` 走 Electron 主进程，既不受 CSP 限制也不做 CORS 预检。
 * 返回非 2xx 不抛，交给调用方给业务文案；彻底连不上返回 null。
 */
export async function getJsonWithFallback(
  url: string, headers: Record<string, string>,
): Promise<{ status: number; json: any } | null> {
  try {
    const resp = await fetch(url, { method: "GET", headers });
    const json = await resp.json().catch(() => null);
    return { status: resp.status, json };
  } catch {
    try {
      const r: any = await requestUrl({ url, method: "GET", headers });
      const json = typeof r.json === "function" ? await r.json() : r.json;
      return { status: r.status, json };
    } catch {
      return null;
    }
  }
}

/** DELETE 请求：同 getJson 的 CSP 兜底——fetch 抛错时换 requestUrl() 重试一次 */
async function deleteJson(url: string, headers: Record<string, string>): Promise<ChatResp> {
  try {
    const resp = await fetch(url, { method: "DELETE", headers });
    return {
      ok: resp.ok,
      status: resp.status,
      text: () => resp.text(),
      json: () => resp.json(),
    };
  } catch {
    const r: any = await requestUrl({ url, method: "DELETE", headers });
    const text = typeof r.text === "function" ? await r.text() : String(r.text ?? "");
    let json: any = r.json;
    if (typeof r.json === "function") json = await r.json();
    const ok = r.status >= 200 && r.status < 300;
    return { ok, status: r.status, text: async () => text, json: async () => json };
  }
}

/** serve 端会话删除结果（4.0 执行线 #8 契约：{"ok":true,"deleted":{...}}） */
export interface SessionDeleteResult {
  ok?: boolean;
  deleted?: { session: boolean; ranges: boolean; index: boolean };
}

/** 删除 serve 端会话（4.0 执行线 #8）：DELETE /v1/sessions/{sid}，Bearer 鉴权。
 *  仅 agentlab 后端支持（老 Hermes 无此能力 → 直接返回 null）；
 *  网络 / HTTP 非 2xx / 解析任一失败均返回 null，由调用方 console.warn 并继续本地清理。 */
export async function sessionDelete(settings: ArkSettings, sid: string): Promise<SessionDeleteResult | null> {
  if (settings.agentProvider !== "agentlab" || !sid) return null;
  const ep = agentEndpoint(settings);
  try {
    const resp = await deleteJson(`${ep.origin}/v1/sessions/${encodeURIComponent(sid)}`,
      ep.token ? { Authorization: `Bearer ${ep.token}` } : {});
    if (!resp.ok) return null;
    return (await resp.json()) as SessionDeleteResult;
  } catch {
    return null;
  }
}

/** 获取单次 run 的诊断元数据；服务端只返回工具/LLM 事件摘要，不含正文。 */
export async function fetchAgentRunDetail(settings: ArkSettings, traceId: string): Promise<AgentRunDetail | null> {
  if (settings.agentProvider !== "agentlab" || !traceId) return null;
  const ep = agentEndpoint(settings);
  const url = `${ep.origin}/v1/runs/${encodeURIComponent(traceId)}`;
  const headers: Record<string, string> = ep.token ? { Authorization: `Bearer ${ep.token}` } : {};
  try {
    const resp = await fetch(url, { method: "GET", headers });
    if (!resp.ok) return null;
    return (await resp.json())?.detail ?? null;
  } catch {
    try {
      const r: any = await requestUrl({ url, method: "GET", headers });
      if (r.status < 200 || r.status >= 300) return null;
      const j = typeof r.json === "function" ? await r.json() : r.json;
      return j?.detail ?? null;
    } catch { return null; }
  }
}

/** 解析 agentlab HITL 审批；终态重复提交由服务端按 approval_id 幂等返回。 */
export async function resolveApproval(settings: ArkSettings, approvalId: string,
                                      decision: "allow" | "deny" | "cancel"): Promise<any | null> {
  if (settings.agentProvider !== "agentlab" || !approvalId) return null;
  const ep = agentEndpoint(settings);
  const url = `${ep.origin}/v1/approvals/${encodeURIComponent(approvalId)}`;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(ep.token ? { Authorization: `Bearer ${ep.token}` } : {}),
  };
  try {
    const resp = await fetch(url, { method: "POST", headers,
      body: JSON.stringify({ decision }) });
    if (!resp.ok) return null;
    return (await resp.json())?.approval ?? null;
  } catch {
    try {
      const r: any = await requestUrl({ url, method: "POST", headers,
        body: JSON.stringify({ decision }), contentType: "application/json" });
      if (r.status < 200 || r.status >= 300) return null;
      const j = typeof r.json === "function" ? await r.json() : r.json;
      return j?.approval ?? null;
    } catch { return null; }
  }
}

/**
 * Hermes / agentlab /v1/responses 流式 agent 调用（SSE）。
 * 事件形态见开发记录 DEV-008：工具轨迹在 response.output_item.added/done（data.item.type=function_call/function_call_output），
 * 文本在 response.output_text.delta。按 event/data 分流，未识别事件跳过。
 * agentlab serve（OPT-067）刻意复刻 Hermes 兼容事件，故此解析零改动复用。
 * 返回完整回复文本。
 */
export async function agentChat(
  settings: ArkSettings,
  messages: AiMessage[],
  cb: HermesCallbacks = {},
  signal?: AbortSignal,
  opts: { tools?: unknown[]; projectId?: string; multi?: string[] } = {},
): Promise<string> {
  const ep = agentEndpoint(settings);
  const { responses } = ep;
  const token = ep.token;
  const model = ep.model;
  if (!token) throw new Error(`未配置 ${ep.label} Token（agentProvider=${settings.agentProvider}）。请在设置中心 → Agent 内核 填写。`);

  const body: Record<string, unknown> = { model, input: messages, stream: true };
  // contextual 兜底强制禁工具（外部审查 P0-1：防止 Hermes 直写库破坏 1:1）
  if (opts.tools) body.tools = opts.tools;
  // P0-2/OPT-107：Project 长期任务空间——带激活项目 id，服务端注入项目规则/背景
  // 以发起会话的 projectId 为准，避免用户切换项目后在途请求读取到全局/新项目设置。
  const pid = (opts.projectId ?? settings.activeProjectId ?? "").trim();
  if (pid) body.project_id = pid;
  // P2-2/F5-019：@ 多 Agent 指派名单（服务端据此并行作答 + 主 agent 汇总）
  if (opts.multi && opts.multi.length) body.multi = opts.multi;

  const resp = await fetch(responses, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(body),
    signal,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Hermes 调用失败 ${resp.status}: ${text.slice(0, 300)}`);
  }
  if (!resp.body) throw new Error("Hermes 响应无流");

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let full = "";
  let activeTool = "";

  const dispatch = (data: string) => {
    if (!data.trim()) return;
    let obj: any;
    try {
      obj = JSON.parse(data);
    } catch {
      console.warn("[SSE] JSON parse failed:", data.slice(0, 100));
      return;
    }
    const t = obj?.type;
    if (t === "response.output_item.added" || t === "response.output_item.done") {
      const item = obj?.item;
      if (item?.type === "function_call") {
        activeTool = item.name || "tool";
        cb.onTool?.(activeTool, "start", JSON.stringify(item.arguments ?? ""));
      } else if (item?.type === "function_call_output") {
        const out = typeof item.output === "string" ? item.output : JSON.stringify(item.output ?? "");
        cb.onTool?.(activeTool || "tool", "done", out.slice(0, 500));
        activeTool = "";
      }
    } else if (t === "response.output_item.done" && obj?.item?.type === "message") {
      // 兜底：部分 gateway 以整条 message 而非 output_text.delta 吐文本，防 full 为空
      const content = obj.item.content;
      if (Array.isArray(content)) {
        const txt = content
          .map((c: any) => (c?.type === "output_text" || c?.type === "text") && typeof c.text === "string" ? c.text : "")
          .join("");
        if (txt) full += txt;
      }
    } else if (t === "response.output_text.delta") {
      const d = typeof obj?.delta === "string" ? obj.delta : "";
      if (d) {
        full += d;
        cb.onText?.(d);
      }
    } else if (t === "response.heartbeat") {
      // agentlab serve 进度心跳：不并入正文，仅供 UI 刷新"运行中 N 秒"
      cb.onStatus?.({ elapsed: Number(obj?.elapsed) || 0 });
    } else if (t === "approval.requested" || t === "approval.resolved") {
      cb.onApproval?.({
        phase: t === "approval.requested" ? "requested" : "resolved",
        approval: obj,
      });
    } else if (t === "response.branch.started" || t === "response.branch.done"
               || t === "response.branch.failed") {
      // P2-2/F5-019：多 Agent 分支进度（字段见 agentlab docs/03 §8.3）
      cb.onBranch?.(obj);
    } else if (t === "response.multi.summary") {
      cb.onMultiSummary?.(obj);
    } else if (t === "response.completed") {
      // W2/OPT-110：completed 携带最终历史快照（可选，旧版 serve 无此字段 → history undefined）
      // 快照在 response.summary.history（后端把 history 收进 summary 字典；response.history 为兼容读法）
      const resp: any = obj?.response ?? {};
      const sum: any = resp?.summary ?? {};
      const hist = Array.isArray(sum.history) ? sum.history
        : Array.isArray(resp?.history) ? resp.history : undefined;
      cb.onCompleted?.({ summary: resp?.summary ?? sum, history: hist });
    } else if (t) {
      // 未知事件类型：记录日志但不崩溃
      console.warn("[SSE] Unknown event type:", t, obj);
    }
  };


  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, idx).replace(/\r$/, "");
      buf = buf.slice(idx + 1);
      if (line.startsWith("data:")) dispatch(line.slice(5).trim());
    }
  }
  return full;
}
