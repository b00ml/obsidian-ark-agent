// 邮件后端：通过 agently-cli（agent.qq.com 邮箱）真实收发
import { execFile } from "child_process";
import type ArkOSPlugin from "./main";

function run(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(
      "cmd.exe",
      ["/c", "agently-cli", ...args],
      { maxBuffer: 64 * 1024 * 1024, windowsHide: true },
      (err, stdout) => {
        if (err) {
          reject(new Error(String((err as Error).message ?? err)));
          return;
        }
        resolve(stdout);
      },
    );
  });
}

export interface MailSummary {
  message_id: string;
  subject: string;
  snippet?: string;
  is_read: boolean;
  created_at: string;
  from?: { name?: string; email?: string };
}

export interface MailFull extends MailSummary {
  body: string;
  body_format: string;
  to?: { name?: string; email?: string }[];
  cc?: { name?: string; email?: string }[];
  bcc?: { name?: string; email?: string }[];
  attachments?: any[];
  calendar_ics?: string;
}

/** 校验授权，未授权返回错误信息字符串，成功返回 null */
export async function checkAuth(): Promise<string | null> {
  try {
    const j = JSON.parse(await run(["+me"]));
    return j.ok ? null : String(j.error?.message ?? "未授权");
  } catch (e: any) {
    return String(e?.message ?? e);
  }
}

export async function listMails(opts: { dir?: string; limit?: number } = {}): Promise<MailSummary[]> {
  const args = ["message", "+list", "--limit", String(opts.limit ?? 30)];
  if (opts.dir && opts.dir !== "all") args.push("--dir", opts.dir);
  const j = JSON.parse(await run(args));
  if (!j.ok) throw new Error("拉取邮件失败: " + (j?.error?.message ?? ""));
  return (j.data?.data ?? []) as MailSummary[];
}

export async function readMail(id: string): Promise<MailFull> {
  const j = JSON.parse(await run(["message", "+read", "--id", id]));
  if (!j.ok) throw new Error("读取邮件失败: " + (j?.error?.message ?? ""));
  return j.data as MailFull;
}

export async function sendMail(to: string[], subject: string, body: string, cc: string[] = [], bcc: string[] = []): Promise<string> {
  const args = ["message", "+send", "--confirmed"];
  to.forEach((t) => args.push("--to", t));
  cc.forEach((t) => args.push("--cc", t));
  bcc.forEach((t) => args.push("--bcc", t));
  args.push("--subject", subject, "--body", body);
  const j = JSON.parse(await run(args));
  if (!j.ok) throw new Error("发送失败: " + (j?.error?.message ?? ""));
  return j.data?.message_id ?? "";
}

export async function replyMail(id: string, body: string): Promise<string> {
  const j = JSON.parse(await run(["message", "+reply", "--id", id, "--body", body]));
  if (!j.ok) throw new Error("回复失败: " + (j?.error?.message ?? ""));
  return j.data?.message_id ?? "";
}

export async function deleteMail(id: string): Promise<boolean> {
  const j = JSON.parse(await run(["message", "+delete", "--id", id]));
  return !!j.ok;
}

/** 简易 HTML→纯文本 */
export function toPlain(html: string): string {
  const div = document.createElement("div");
  div.innerHTML = html;
  return div.textContent ?? div.innerText ?? html;
}

/** 逐级创建 Vault 文件夹（已存在忽略） */
async function ensureVaultFolder(plugin: ArkOSPlugin, path: string) {
  const parts = path.split("/").filter(Boolean);
  let cur = "";
  for (const p of parts) {
    cur = cur ? `${cur}/${p}` : p;
    if (!plugin.app.vault.getAbstractFileByPath(cur)) {
      try { await plugin.app.vault.createFolder(cur); } catch { /* 并发/已存在 */ }
    }
  }
}

/**
 * 下载邮件附件到 Vault mailFolder/附件。
 * 有 attachment_id 用 CLI 下载；仅 download_url 的大附件返回链接由调用方打开。
 */
export async function downloadAttachment(plugin: ArkOSPlugin, msgId: string, att: any): Promise<string> {
  if (att.attachment_id) {
    const relFolder = (plugin.data.settings.mailFolder || "02-DB/消息") + "/附件";
    await ensureVaultFolder(plugin, relFolder);
    const adapter: any = plugin.app.vault.adapter;
    let absDir = relFolder;
    try { absDir = adapter?.getFullPath ? adapter.getFullPath(relFolder) : relFolder; } catch { absDir = relFolder; }
    const j = JSON.parse(await run(["attachment", "+download", "--msg", msgId, "--att", att.attachment_id, "--output", absDir]));
    if (!j.ok) throw new Error("下载失败: " + (j?.error?.message ?? ""));
    return `已保存到 ${relFolder}/${att.filename ?? ""}`;
  }
  if (att.download_url) return att.download_url; // 大附件：返回直链
  return "";
}

/** 保存 .ics 日历邀请到 Vault mailFolder/日历 */
export async function saveIcs(plugin: ArkOSPlugin, mail: MailFull): Promise<string | null> {
  if (!mail.calendar_ics) return null;
  const folder = (plugin.data.settings.mailFolder || "02-DB/消息") + "/日历";
  await ensureVaultFolder(plugin, folder);
  const file = `${folder}/${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}.ics`;
  await plugin.app.vault.create(file, mail.calendar_ics);
  return file;
}
