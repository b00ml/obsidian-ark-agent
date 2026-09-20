import { execFile } from "child_process";
import { Modal, Notice, Setting } from "obsidian";
import type ArkOSPlugin from "./main";
import { resolvePython, serveWorkdir } from "./serve-control";
import { confirmDialog, inputDialog } from "./ui";

export interface MemoryReviewItem {
  id: string;
  type?: string;
  project_id?: string;
  session_id?: string;
  source?: string;
  source_ref?: string;
  content_hash: string;
  review_due_at?: string;
  path?: string;
}

export interface MemoryLifecycleItem extends MemoryReviewItem {
  status: string;
  stored_status?: string;
  valid_from?: string;
  valid_until?: string;
  updated_at?: string;
  content_preview?: string;
  project_id?: string;
  session_id?: string;
}

function vaultPath(plugin: ArkOSPlugin): string {
  const adapter: any = plugin.app.vault.adapter as any;
  return String((typeof adapter?.getBasePath === "function"
    ? adapter.getBasePath() : adapter?.basePath) || "").trim();
}

export function memoryReviewListArgs(vault: string): string[] {
  return ["-m", "agentlab.eval.memory_review_due", "--vault", vault];
}

export function memoryLifecycleListArgs(vault: string): string[] {
  return ["-m", "agentlab.eval.memory_lifecycle", "--vault", vault];
}

export function memoryReviewApplyArgs(
  vault: string, item: Pick<MemoryReviewItem, "id" | "content_hash">,
  reviewer: string, reason: string, decision: "confirm" | "defer",
  deferUntil = "",
): string[] {
  const args = [
    ...memoryReviewListArgs(vault), "--apply", "--memory-id", item.id,
    "--decision", decision, "--reviewer", reviewer,
    "--expected-content-hash", item.content_hash, "--reason", reason,
  ];
  if (decision === "defer" && deferUntil) args.push("--defer-until", deferUntil);
  return args;
}

export function memoryLifecycleActionArgs(
  vault: string, item: Pick<MemoryLifecycleItem, "id" | "content_hash" | "project_id" | "session_id">,
  reviewer: string, reason: string, action: "promote" | "correct" | "revoke",
  content = "", tags: string[] = [],
): string[] {
  const args = [
    "-m", "agentlab.eval.memory_lifecycle_action", "--vault", vault, "--apply",
    "--memory-id", item.id, "--action", action,
    "--expected-content-hash", item.content_hash, "--reviewer", reviewer,
    "--reason", reason,
  ];
  if (item.project_id) args.push("--project-id", item.project_id);
  if (item.session_id) args.push("--session-id", item.session_id);
  if (action === "correct") {
    args.push("--content", content);
    if (tags.length) args.push("--tags-json", JSON.stringify(tags));
  }
  return args;
}

export function parseMemoryReviewReport(output: string): MemoryReviewItem[] {
  try {
    const parsed = JSON.parse(output);
    if (parsed?.passed === false || !Array.isArray(parsed?.items)) return [];
    return parsed.items.filter((item: any) => item && typeof item.id === "string"
      && typeof item.content_hash === "string");
  } catch {
    return [];
  }
}

export function parseMemoryLifecycleReport(output: string): MemoryLifecycleItem[] {
  try {
    const parsed = JSON.parse(output);
    if (parsed?.passed === false || !Array.isArray(parsed?.items)) return [];
    return parsed.items.filter((item: any) => item && typeof item.id === "string"
      && typeof item.status === "string" && typeof item.content_hash === "string");
  } catch { return []; }
}

function runReviewCommand(plugin: ArkOSPlugin, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(resolvePython(plugin.data.settings), args, {
      cwd: serveWorkdir(plugin.data.settings), timeout: 30_000, encoding: "utf8",
      windowsHide: true, env: { ...process.env, PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8" },
    }, (error, stdout, stderr) => {
      const output = `${stdout || ""}${stderr || ""}`.trim();
      if (error) reject(new Error(output || error.message));
      else resolve(output);
    });
  });
}

export async function listMemoryReviewDue(plugin: ArkOSPlugin): Promise<MemoryReviewItem[]> {
  const vault = vaultPath(plugin);
  if (!vault) throw new Error("当前 Vault 路径不可用");
  return parseMemoryReviewReport(await runReviewCommand(plugin, memoryReviewListArgs(vault)));
}

export async function listMemoryLifecycle(plugin: ArkOSPlugin): Promise<MemoryLifecycleItem[]> {
  const vault = vaultPath(plugin);
  if (!vault) throw new Error("当前 Vault 路径不可用");
  return parseMemoryLifecycleReport(await runReviewCommand(plugin, memoryLifecycleListArgs(vault)));
}

async function applyMemoryReview(plugin: ArkOSPlugin, item: MemoryReviewItem,
                                 reviewer: string, reason: string,
                                 decision: "confirm" | "defer", deferUntil = ""): Promise<void> {
  const vault = vaultPath(plugin);
  if (!vault) throw new Error("当前 Vault 路径不可用");
  await runReviewCommand(plugin, memoryReviewApplyArgs(
    vault, item, reviewer, reason, decision, deferUntil,
  ));
}

async function applyMemoryLifecycleAction(
  plugin: ArkOSPlugin, item: MemoryLifecycleItem, action: "promote" | "correct" | "revoke",
  reviewer: string, reason: string, content = "", tags: string[] = [],
): Promise<void> {
  const vault = vaultPath(plugin);
  if (!vault) throw new Error("当前 Vault 路径不可用");
  await runReviewCommand(plugin, memoryLifecycleActionArgs(
    vault, item, reviewer, reason, action, content, tags,
  ));
}

class MemoryReviewModal extends Modal {
  private items: MemoryReviewItem[] = [];

  constructor(private readonly plugin: ArkOSPlugin) {
    super(plugin.app);
  }

  async onOpen(): Promise<void> {
    this.titleEl.setText("长期记忆复核");
    await this.refresh();
  }

  onClose(): void {
    this.contentEl.empty();
  }

  private async refresh(): Promise<void> {
    this.contentEl.empty();
    try {
      this.items = await listMemoryReviewDue(this.plugin);
    } catch (error: any) {
      this.contentEl.createDiv({ cls: "sos-hint", text: `无法读取复核队列：${String(error?.message ?? error)}` });
      return;
    }
    if (!this.items.length) {
      this.contentEl.createDiv({ cls: "sos-hint", text: "当前没有到期记忆。" });
      return;
    }
    this.contentEl.createDiv({ cls: "sos-hint", text: `待人工复核 ${this.items.length} 条；确认或延期均会校验内容哈希。` });
    for (const item of this.items) this.renderItem(item);
  }

  private renderItem(item: MemoryReviewItem): void {
    const row = this.contentEl.createDiv({ cls: "sos-mail-row" });
    row.createDiv({ cls: "sos-mail-name", text: `${item.type || "memory"} · ${item.id}` });
    row.createDiv({ cls: "sos-hint", text: `${item.project_id || "default"} · ${item.review_due_at || ""}` });
    const actions = row.createDiv({ cls: "sos-row-actions" });
    const confirm = actions.createEl("button", { cls: "sos-mini", text: "确认" });
    confirm.addEventListener("click", () => void this.review(item, "confirm"));
    const defer = actions.createEl("button", { cls: "sos-mini", text: "延期" });
    defer.addEventListener("click", () => void this.review(item, "defer"));
  }

  private async review(item: MemoryReviewItem, decision: "confirm" | "defer"): Promise<void> {
    const reviewer = this.plugin.data.settings.captainName || "user";
    const reason = await inputDialog(this.plugin, { title: "复核原因", placeholder: "填写人工核对结论" });
    if (!reason?.trim()) return;
    let deferUntil = "";
    if (decision === "defer") {
      const value = await inputDialog(this.plugin, {
        title: "延期至", placeholder: "ISO 时间，例如 2026-10-01T00:00:00+00:00",
      });
      if (!value?.trim()) return;
      deferUntil = value.trim();
    }
    const accepted = await confirmDialog(this.plugin, decision === "confirm"
      ? `确认复核 ${item.id}？` : `延期复核 ${item.id}？`);
    if (!accepted) return;
    try {
      await applyMemoryReview(this.plugin, item, reviewer, reason.trim(), decision, deferUntil);
      new Notice(decision === "confirm" ? "记忆已确认" : "记忆已延期复核");
      await this.refresh();
    } catch (error: any) {
      new Notice(`复核未执行：${String(error?.message ?? error)}`, 6000);
    }
  }
}

class MemoryLifecycleModal extends Modal {
  constructor(private readonly plugin: ArkOSPlugin) { super(plugin.app); }

  async onOpen(): Promise<void> {
    this.titleEl.setText("长期记忆管理");
    this.contentEl.empty();
    try {
      const items = await listMemoryLifecycle(this.plugin);
      if (!items.length) {
        this.contentEl.createDiv({ cls: "sos-hint", text: "当前没有可管理的长期记忆。" });
        return;
      }
      this.contentEl.createDiv({ cls: "sos-hint", text: `已读取 ${items.length} 条 Markdown 真源状态。` });
      for (const item of items) {
        const row = this.contentEl.createDiv({ cls: "sos-mail-row" });
        row.createDiv({ cls: "sos-mail-name", text: `${item.status} · ${item.type || "memory"} · ${item.id}` });
        row.createDiv({ cls: "sos-hint", text: item.content_preview || item.source_ref || item.path || "" });
        row.createDiv({ cls: "sos-hint", text: `${item.project_id || "default"} · ${item.session_id || ""} · ${item.updated_at || item.review_due_at || ""}` });
        this.renderActions(row, item);
      }
    } catch (error: any) {
      this.contentEl.createDiv({ cls: "sos-hint", text: `无法读取长期记忆：${String(error?.message ?? error)}` });
    }
  }

  onClose(): void { this.contentEl.empty(); }

  private renderActions(row: HTMLElement, item: MemoryLifecycleItem): void {
    const status = String(item.status || "").toLowerCase();
    if (!item.content_hash || ["expired", "not_yet_valid", "revoked", "superseded"].includes(status)) return;
    const actions = row.createDiv({ cls: "sos-row-actions" });
    if (status === "candidate" || status === "quarantine") {
      const promote = actions.createEl("button", { cls: "sos-mini", text: "人工确认晋升" });
      promote.addEventListener("click", () => void this.runAction(item, "promote"));
    }
    if (status === "active" || status === "conflict" || status === "review_due") {
      const correct = actions.createEl("button", { cls: "sos-mini", text: "纠正" });
      correct.addEventListener("click", () => void this.runAction(item, "correct"));
      const revoke = actions.createEl("button", { cls: "sos-mini", text: "撤销" });
      revoke.addEventListener("click", () => void this.runAction(item, "revoke"));
    }
  }

  private async runAction(item: MemoryLifecycleItem, action: "promote" | "correct" | "revoke"): Promise<void> {
    const reviewer = this.plugin.data.settings.captainName || "user";
    const reason = await inputDialog(this.plugin, { title: `${action === "promote" ? "晋升" : action === "correct" ? "纠正" : "撤销"}原因`, placeholder: "填写人工核对结论" });
    if (!reason?.trim()) return;
    let content = "";
    if (action === "correct") {
      const value = await inputDialog(this.plugin, { title: "纠正后的记忆内容", placeholder: "输入新的完整事实内容" });
      if (!value?.trim()) return;
      content = value.trim();
    }
    const accepted = await confirmDialog(this.plugin, `确认对 ${item.id} 执行${action === "promote" ? "晋升" : action === "correct" ? "纠正" : "撤销"}？\n当前内容哈希：${item.content_hash}`);
    if (!accepted) return;
    try {
      await applyMemoryLifecycleAction(this.plugin, item, action, reviewer, reason.trim(), content);
      new Notice("长期记忆操作已完成");
      await this.onOpen();
    } catch (error: any) {
      new Notice(`长期记忆操作未执行：${String(error?.message ?? error)}`, 6000);
    }
  }
}

export function openMemoryReview(plugin: ArkOSPlugin): void {
  new MemoryReviewModal(plugin).open();
}

export function openMemoryLifecycle(plugin: ArkOSPlugin): void {
  new MemoryLifecycleModal(plugin).open();
}

export async function notifyMemoryReviewDue(plugin: ArkOSPlugin): Promise<number> {
  try {
    const items = await listMemoryReviewDue(plugin);
    if (items.length) new Notice(`长期记忆有 ${items.length} 条待复核`, 6000);
    return items.length;
  } catch {
    return 0;
  }
}
