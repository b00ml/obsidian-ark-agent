// AI 深度集成能力层：quick/contextual 通道自适应 + 统一 prompt 解析/二次校验 + 预览确认。
// 设计文档：docs/DESIGN-AI-DEEP-INTEGRATION.md
// 硬约束：只产出结构化数据/文本，绝不越权直写任务/卡片 .md（1:1 由 main.addTodoTask / cards.updateCard 保证）。
import { Modal, Notice, Setting, TFile, type Editor, type MarkdownFileInfo } from "obsidian";
import type ArkOSPlugin from "./main";
import type { ArkSettings } from "./settings";
import { chat, agentChat, type AiMessage } from "./ai";
import { inputDialog, confirmDialog } from "./ui";
import { renderWordDiff } from "./word-diff";
import { sendSelectionToWorkbench } from "./agent-workbench";
import { writeMarkdown } from "./sync";
import { sanitizeFilename } from "./utils";
import polishUser from "./prompts/polish-user.st";
import todosUser from "./prompts/todos-user.st";
import scheduleUser from "./prompts/schedule-user.st";
import periodReviewUser from "./prompts/period-review-user.st";

// ── D1 通道自适应路由 ───────────────────────────────────────────────
export type ChannelKind = "quick" | "contextual";

/** 通道路由：直连 chat（无工具、秒回）优先，无直连 key 才兜底 Hermes agentChat */
export function selectChannel(settings: ArkSettings, kind: ChannelKind): "chat" | "agentChat" | null {
  if (settings.aiApiKey) return "chat";
  if (settings.hermesToken || settings.agentlabToken) return "agentChat";
  return null;
}

export async function runChannel(
  settings: ArkSettings,
  messages: AiMessage[],
  kind: ChannelKind,
  signal: AbortSignal,
  onText?: (d: string) => void,
): Promise<string> {
  const ch = selectChannel(settings, kind);
  if (ch === "chat") return chat(settings, messages, { signal });
  if (ch === "agentChat") {
    // contextual 兜底强制禁工具（外部审查 P0-1：防 Hermes 直写库）；quick 兜底允许工具
    return agentChat(settings, messages, { onText }, signal, kind === "contextual" ? { tools: [] } : {});
  }
  throw new Error("未配置 AI 与 Hermes。请在设置中心 → AI 助手 / Agent 内核 配置后使用。");
}

// ── §3.1 extractJson：剥壳 + 括号配对截取 + 清洗 + 二次校验 ───────────
export class JsonExtractError extends Error {}

function bracketBounds(s: string, from: number): number {
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = from; i < s.length; i++) {
    const c = s[i];
    if (inStr) {
      if (esc) esc = false;
      else if (c === "\\") esc = true;
      else if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') { inStr = true; continue; }
    if (c === "{" || c === "[") depth++;
    else if (c === "}" || c === "]") { depth--; if (depth === 0) return i; }
  }
  return -1;
}

function cleanJson(s: string): string {
  return s
    .replace(/^\uFEFF/, "")
    .replace(/[\u200b\u200c\ufeff\u2028\u2029]/g, "")   // 零宽/行分隔符
    // 中文引号被模型写成未转义 ASCII " → 若两侧均为汉字/中文标点/全角标点，判为内容引号，还原为左弯引号（对 JSON 无害）。
    // 否则像「并引用"知之者…"」这种裸 ASCII " 会提前截断字符串、破坏解析（OPT-043 实测）。
    .replace(/(?<=[\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF])"(?=[\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF])/g, "\u201C")
    .replace(/,\s*([}\]])/g, "$1")                       // 尾随逗号
    .trim();
}

export function extractJson(text: string): any {
  const t0 = (text ?? "").trim();
  const candidates: string[] = [];
  // 优先取 ```json``` 围栏内容
  const fence = t0.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fence) candidates.push(fence[1].trim());
  // 整段文本优先：畸形引号时 bracketBounds 可能判定数组"不闭合"，靠 cleanJson 修复后整体直接解析（OPT-043）
  candidates.push(t0);
  // 再按配对括号截取多个最可能的 JSON 结构（限前 15 个防 O(n^2)），提升容错
  let examined = 0;
  for (let i = 0; i < t0.length && examined < 15; i++) {
    const c = t0[i];
    if (c === "[" || c === "{") {
      const end = bracketBounds(t0, i);
      if (end >= 0) { candidates.push(t0.slice(i, end + 1)); examined++; }
    }
  }
  if (candidates.length === 0) candidates.push(t0);
  for (const cand of candidates) {
    try {
      return JSON.parse(cleanJson(cand));
    } catch { /* 尝试下一个候选 */ }
  }
  const e = new JsonExtractError(`无法从AI响应中解析出合法JSON（原文前置：${t0.slice(0, 120)}…）`);
  (e as any).raw = t0; // 供调用方落 _errors 日志
  throw e;
}

/** P1-7：记录一次 AI 结构化失败到 02-DB/_errors/（错误不落 Inbox、不污染知识层）；日志失败不阻断主流程 */
export async function logAiError(
  plugin: ArkOSPlugin,
  kind: string,
  prompt: string,
  raw: string,
  err: unknown,
): Promise<void> {
  try {
    const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const record = {
      kind,
      at: new Date().toISOString(),
      error: err instanceof Error ? err.message : String(err),
      rawLen: (raw || "").length,
      prompt,
      raw,
    };
    await writeMarkdown(plugin, "02-DB/_errors", `${ts}-${sanitizeFilename(kind)}`, {
      type: "ai-error",
      kind,
      created_at: new Date().toISOString(),
    }, "```json\n" + JSON.stringify(record, null, 2) + "\n```");
  } catch { /* 忽略日志失败 */ }
}

// ── F1/F3 共用：润色变体生成（quick） ────────────────────────────────
/** 润色模式：F1 用 正式/口语/精简/提炼要点；F3 用 润色/扩写/提炼要点 */
export async function generateVariant(
  plugin: ArkOSPlugin,
  text: string,
  mode: string,
  onChunk?: (d: string) => void,
): Promise<string> {
  const prompt = polishUser.replace("{{mode}}", mode).replace("{{text}}", text);
  const s = plugin.data.settings;
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), 60000); // quick 60s
  try {
    return await runChannel(s, [{ role: "user", content: prompt }], "quick", ac.signal, onChunk);
  } finally {
    clearTimeout(timer);
  }
}

// ── F2：今日待办生成（contextual，单数据路径） ────────────────────────
export interface TodoDraft {
  description: string;
  priority: "high" | "medium" | "low";
  due_date?: string;
  tags: string[];
}

/** Agent 排期建议：只允许触及任务排期字段，写回前必须经过 UI diff 确认。 */
export interface ScheduleDraft {
  task_id: string;
  start_date?: string | null;
  due_date?: string | null;
  effort?: number | null;
  dependency?: string[];
  progress?: number | null;
  reason?: string;
}

export interface ScheduleTaskInput {
  id: string;
  description: string;
  start_date?: string;
  due_date?: string;
  effort?: number;
  dependency?: string[];
  progress?: number;
  priority: string;
  status: string;
}

export interface PeriodReviewInput {
  periodLabel: string;
  startDate: string;
  endDate: string;
  metrics: Record<string, string | number>;
  evidence: string[];
}

/** 结构化排期：模型只提出建议，不在此函数中写入任务。 */
export async function planTaskSchedule(
  plugin: ArkOSPlugin,
  listName: string,
  tasks: ScheduleTaskInput[],
): Promise<ScheduleDraft[]> {
  if (!tasks.length) return [];
  const prompt = scheduleUser
    .replace("{{list_name}}", listName)
    .replace("{{tasks}}", JSON.stringify(tasks));
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), 180000);
  let raw = "";
  try {
    raw = await runChannel(plugin.data.settings, [{ role: "user", content: prompt }], "contextual", ac.signal);
    const parsed = extractJson(raw);
    if (!Array.isArray(parsed)) throw new JsonExtractError("AI 排期结果不是数组");
    return validateScheduleDrafts(parsed, tasks);
  } catch (e: any) {
    if (raw || e instanceof JsonExtractError) await logAiError(plugin, "schedule-drafts", prompt, raw || (e as any).raw || "", e);
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

function validateScheduleDrafts(raw: any[], tasks: ScheduleTaskInput[]): ScheduleDraft[] {
  const ids = new Set(tasks.map((t) => t.id));
  const seen = new Set<string>();
  const date = (value: unknown, field: string): string | null | undefined => {
    if (value == null || value === "") return value === null ? null : undefined;
    const text = String(value);
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(text);
    if (!match) throw new Error(`排期字段 ${field} 日期格式非法`);
    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = Number(match[3]);
    const parsed = new Date(Date.UTC(year, month - 1, day));
    parsed.setUTCFullYear(year);
    if (parsed.getUTCFullYear() !== year || parsed.getUTCMonth() !== month - 1 || parsed.getUTCDate() !== day) {
      throw new Error(`排期字段 ${field} 不是有效日期`);
    }
    return text;
  };
  return raw.map((item, index) => {
    if (!item || typeof item !== "object" || typeof item.task_id !== "string" || !ids.has(item.task_id)) {
      throw new Error(`排期建议第 ${index + 1} 条 task_id 无效`);
    }
    const allowed = new Set(["task_id", "start_date", "due_date", "effort", "dependency", "progress", "reason"]);
    const unknown = Object.keys(item).filter((key) => !allowed.has(key));
    if (unknown.length) throw new Error(`排期建议第 ${index + 1} 条包含未知字段：${unknown.join(", ")}`);
    if (seen.has(item.task_id)) throw new Error(`排期建议重复修改任务 ${item.task_id}`);
    seen.add(item.task_id);
    const start = date(item.start_date, "start_date");
    const due = date(item.due_date, "due_date");
    if (start && due && start > due) throw new Error(`任务 ${item.task_id} 的开始日期晚于截止日期`);
    const effort = item.effort == null || item.effort === "" ? item.effort === null ? null : undefined : Number(item.effort);
    if (effort != null && (!Number.isFinite(effort) || effort < 0 || effort > 10000)) {
      throw new Error(`任务 ${item.task_id} 的 effort 超出范围`);
    }
    const progress = item.progress == null || item.progress === "" ? item.progress === null ? null : undefined : Number(item.progress);
    if (progress != null && (!Number.isFinite(progress) || progress < 0 || progress > 100)) {
      throw new Error(`任务 ${item.task_id} 的 progress 超出范围`);
    }
    const dependency = item.dependency === undefined ? undefined : Array.isArray(item.dependency) ? item.dependency.map(String) : null;
    if (dependency === null) throw new Error(`任务 ${item.task_id} 的 dependency 必须是数组`);
    if (dependency?.some((id: string) => !ids.has(id) || id === item.task_id)) {
      throw new Error(`任务 ${item.task_id} 的 dependency 包含未知任务或自身`);
    }
    if (dependency && new Set(dependency).size !== dependency.length) {
      throw new Error(`任务 ${item.task_id} 的 dependency 存在重复项`);
    }
    return {
      task_id: item.task_id,
      start_date: start,
      due_date: due,
      effort,
      dependency,
      progress,
      reason: typeof item.reason === "string" ? item.reason.slice(0, 240) : "",
    };
  });
}

/** 生成周期复盘草稿。该函数禁用工具且不写 Vault，落盘必须由调用方展示预览后执行。 */
export async function generatePeriodReview(
  plugin: ArkOSPlugin,
  input: PeriodReviewInput,
): Promise<string> {
  const prompt = periodReviewUser
    .replace("{{period_label}}", input.periodLabel)
    .replace("{{period_start}}", input.startDate)
    .replace("{{period_end}}", input.endDate)
    .replace("{{metrics}}", JSON.stringify(input.metrics, null, 2))
    .replace("{{evidence}}", input.evidence.length ? input.evidence.join("\n") : "（本周期暂无可引用条目）");
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), 180000);
  try {
    const raw = await runChannel(
      plugin.data.settings,
      [{ role: "user", content: prompt }],
      "contextual",
      ac.signal,
    );
    const body = raw.trim()
      .replace(/^```(?:markdown|md)?\s*/i, "")
      .replace(/\s*```$/, "")
      .replace(/^---[\s\S]*?---\s*/, "")
      .trim();
    if (!body) throw new Error("AI 周期复盘返回空内容");
    return body.slice(0, 30000);
  } finally {
    clearTimeout(timer);
  }
}

/** 插件本地拼近况：最近 5 篇每日回顾 + 未完成任务（确定性、可复现） */
async function buildContext(plugin: ArkOSPlugin): Promise<string> {
  const s = plugin.data.settings;
  const parts: string[] = [];
  try {
    const reviewFiles = plugin.app.vault
      .getFiles()
      .filter((f: TFile) => s.reviewFolder && f.path.startsWith(s.reviewFolder))
      .sort((a, b) => b.stat.mtime - a.stat.mtime)
      .slice(0, 5);
    for (const f of reviewFiles) {
      const body = (await plugin.app.vault.read(f)).trim();
      if (body) parts.push(`【回顾 ${f.basename}】\n${body.slice(0, 800)}`);
    }
  } catch { /* 读库失败不阻断（空目录=空近况，非错误） */ }
  const unfinished: string[] = [];
  for (const list of plugin.data.todoLists) {
    for (const t of list.tasks) {
      if (!t.completed) unfinished.push(`${t.description}${t.dueDate ? `（due ${t.dueDate}）` : ""}`);
    }
  }
  if (unfinished.length) parts.push(`【未完成待办】\n${unfinished.slice(0, 20).join("\n")}`);
  const ctx = parts.join("\n\n");
  return ctx.length > 6000 ? ctx.slice(0, 6000) : ctx; // charCount/2 ≈ 3000 token
}

/**
 * 纯函数：仅把用户口述拆解为结构化待办数组，不带 list、不落卡。
 * 清单定位与落卡由调用方（runTodayTodos）做。
 */
export async function planTodayTodos(plugin: ArkOSPlugin, userInput: string): Promise<TodoDraft[]> {
  const s = plugin.data.settings;
  const context = await buildContext(plugin);
  const prompt = todosUser.replace("{{context}}", context).replace("{{input}}", userInput.trim());
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), 180000); // contextual 180s（抬手对齐 daily.ts）
  try {
    const raw = await runChannel(s, [{ role: "user", content: prompt }], "contextual", ac.signal);
    const arr = extractJson(raw);
    if (!Array.isArray(arr)) throw new JsonExtractError("AI 返回的不是数组");
    return cleanTodoDrafts(arr);
  } finally {
    clearTimeout(timer);
  }
}

function cleanTodoDrafts(arr: any[]): TodoDraft[] {
  const out: TodoDraft[] = [];
  for (const it of arr) {
    if (!it || typeof it !== "object") continue;
    const description = String(it.description ?? it.title ?? "").trim();
    if (!description) continue;
    const priority: TodoDraft["priority"] = it.priority === "high" || it.priority === "low" ? it.priority : "medium";
    const due = typeof it.due_date === "string" && /^\d{4}-\d{2}-\d{2}$/.test(it.due_date) ? it.due_date : undefined;
    const tags = Array.isArray(it.tags)
      ? it.tags.map((x: any) => String(x).trim()).filter(Boolean).slice(0, 5)
      : [];
    out.push({ description, priority, due_date: due, tags });
  }
  return out;
}

/** 收敛到「今日」清单（存在复用，否则新建），杜绝 AI 幻构清单名 */
export async function findOrCreateTodayList(plugin: ArkOSPlugin): Promise<string> {
  const list = plugin.data.todoLists.find((l) => l.name === "今日");
  if (list) return list.id;
  return plugin.createTodoList("今日");
}

/** F2 完整流程：input → 拆解 → 预览确认 → 落卡「今日」清单 */
export async function runTodayTodos(plugin: ArkOSPlugin): Promise<void> {
  const userInput = await inputDialog(plugin, {
    title: "AI 生成今日待办",
    placeholder: "例如：下午写 PRD，晚上跑步",
    multiline: true,
  });
  if (userInput === null || !userInput.trim()) return;
  const loading = new Notice("AI 拆解中…", 0);
  let drafts: TodoDraft[];
  try {
    drafts = await planTodayTodos(plugin, userInput.trim());
  } catch (e: any) {
    loading.hide();
    new Notice("AI 待办生成失败: " + (e?.message ?? e), 4000);
    return;
  }
  loading.hide();
  if (drafts.length === 0) {
    new Notice("未能解析出有效待办，请调整输入后重试", 4000);
    return;
  }
  const ok = await confirmDialog(plugin, `将在「今日」清单新增 ${drafts.length} 条待办，确认？`);
  if (!ok) return;
  const listId = await findOrCreateTodayList(plugin);
  for (const d of drafts) {
    await plugin.addTodoTask(listId, d.description, { priority: d.priority, dueDate: d.due_date, tags: d.tags });
  }
  new Notice(`已在「今日」清单新增 ${drafts.length} 条待办`);
}

// ── 预览 / 模式选择 / 剪贴板 ────────────────────────────────────────
/** 原文 vs 优化 对比 Modal：返回 apply / copy / null */
export function previewDiffModal(plugin: ArkOSPlugin, original: string, optimized: string): Promise<"apply" | "copy" | null> {
  return new Promise((resolve) => {
    const modal = new Modal(plugin.app);
    modal.titleEl.setText("AI 优化 · 逐词 Diff");
    const body = modal.contentEl;
    body.addClass("ai-preview");
    // W3/OPT-067：逐词 Diff（删除=红删除线，新增=绿底）——替代左右对照
    const box = body.createDiv({ cls: "ai-preview-text wdiff" });
    renderWordDiff(box, original, optimized);
    new Setting(body)
      .addButton((b) => b.setButtonText("✖ 拒绝").onClick(() => { modal.close(); resolve(null); }))
      .addButton((b) => b.setButtonText("📋 仅复制").onClick(() => { modal.close(); resolve("copy"); }))
      .addButton((b) => b.setButtonText("✅ 接受替换").setCta().onClick(() => { modal.close(); resolve("apply"); }));
    modal.open();
  });
}

/** 新文件草稿预览：已有文件时显示逐词差异，无已有文件时显示完整草稿。 */
export function previewGeneratedModal(
  plugin: ArkOSPlugin,
  title: string,
  previous: string,
  draft: string,
): Promise<"save" | "copy" | null> {
  return new Promise((resolve) => {
    const modal = new Modal(plugin.app);
    modal.titleEl.setText(title);
    const body = modal.contentEl;
    body.addClass("ai-preview");
    const box = body.createDiv({ cls: "ai-preview-text wdiff" });
    if (previous.trim()) renderWordDiff(box, previous, draft);
    else box.createEl("pre", { cls: "ai-generated-preview", text: draft });
    new Setting(body)
      .addButton((b) => b.setButtonText("取消").onClick(() => { modal.close(); resolve(null); }))
      .addButton((b) => b.setButtonText("仅复制").onClick(() => { modal.close(); resolve("copy"); }))
      .addButton((b) => b.setButtonText("确认写入").setCta().onClick(() => { modal.close(); resolve("save"); }));
    modal.open();
  });
}

/** 模式选择小 Modal（F3 用）：返回选中模式或 null */
export function pickVariantMode(plugin: ArkOSPlugin, modes: string[]): Promise<string | null> {
  return new Promise((resolve) => {
    const modal = new Modal(plugin.app);
    modal.titleEl.setText("选择优化方式");
    const body = modal.contentEl;
    const row = body.createDiv({ cls: "ai-mode-row" });
    for (const m of modes) {
      const btn = row.createEl("button", { cls: "mod-cta", text: m });
      btn.addEventListener("click", () => { modal.close(); resolve(m); });
    }
    modal.open();
  });
}

export async function copyText(plugin: ArkOSPlugin, text: string): Promise<void> {
  try {
    await (plugin.app.vault.adapter as any).writeClipboard(text);
  } catch {
    await navigator.clipboard.writeText(text);
  }
}

// ── F1 编辑器右键「AI 润色」 ─────────────────────────────────────────
export function installEditorAiMenu(plugin: ArkOSPlugin) {
  plugin.registerEvent(
    plugin.app.workspace.on("editor-menu", (menu, editor, view) => {
      const sel = editor.getSelection();
      if (!sel) return;
      // 当前 obsidian API 无 addSubmenu，用「置灰标题 + 同 section 选项」模拟分组
      menu.addItem((item) => item.setIcon("wand-2").setTitle("AI 润色").setDisabled(true));
      for (const style of ["正式", "口语", "精简", "提炼要点"]) {
        menu.addItem((item) =>
          item.setSection("ai-polish").setTitle(style)
            .onClick(() => void runPolish(plugin, editor, view, sel, style)),
        );
      }
      menu.addItem((item) =>
        item.setSection("ai-polish").setIcon("pencil").setTitle("自定义指令…")
          .onClick(() => void askCustomPolish(plugin, editor, view, sel)),
      );
      // W3/OPT-067：选区直送工作台（预填即时生效 + 跳转主控大厅；逻辑在 agent-workbench 内）
      menu.addItem((item) =>
        item.setSection("ai-polish").setIcon("rocket").setTitle("发送到 Agent 工作台")
          .onClick(() => sendSelectionToWorkbench(plugin, sel)),
      );
    }),
  );
}

/** 自定义指令润色：输入任意润色要求，复用预览替换/复制流程 */
async function askCustomPolish(
  plugin: ArkOSPlugin,
  editor: Editor,
  view: MarkdownFileInfo,
  selected: string,
) {
  const instruction = await inputDialog(plugin, {
    title: "AI 润色指令",
    placeholder: "例如：改成正式汇报口吻 / 精简为五条要点 / 语气更坚定",
    multiline: true,
  });
  if (instruction === null || !instruction.trim()) return;
  await runPolish(plugin, editor, view, selected, instruction.trim());
}

async function runPolish(
  plugin: ArkOSPlugin,
  editor: Editor,
  view: MarkdownFileInfo,
  selected: string,
  style: string,
) {
  const loading = new Notice("AI 润色中…", 0);
  let out: string;
  try {
    out = await generateVariant(plugin, selected, style);
  } catch (e: any) {
    loading.hide();
    new Notice("AI 润色失败: " + (e?.message ?? e), 4000);
    return;
  }
  loading.hide();
  if (!out) return;

  const action = await previewDiffModal(plugin, selected, out);
  if (!action) return;
  if (action === "copy") {
    await copyText(plugin, out);
    new Notice("已复制到剪贴板");
    return;
  }
  // apply：校验编辑器仍存活且仍是同一文件，否则回退剪贴板（外部审查 P0-3）
  if (view && view.file != null) {
    try {
      editor.replaceSelection(out);
      new Notice("已应用替换");
      return;
    } catch {
      await copyText(plugin, out);
      new Notice("原选区已不可用，已复制到剪贴板");
      return;
    }
  }
  await copyText(plugin, out);
  new Notice("原选区已不可用，已复制到剪贴板");
}
