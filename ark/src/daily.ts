import type ArkOSPlugin from "./main";
import { agentChat, chat, type AiMessage } from "./ai";

/**
 * 手动触发"每日回顾"生成：让 agent 扫描当天笔记 → 生成回顾 → 写 Vault reviewFolder（默认 02-DB/回顾）。
 * 走已验证的 /v1 agent 路径（Hermes 模式）或直连 chat（非 Hermes 模式）；
 * 不走 Hermes cron agent 模式（本环境会空转烧 CPU，见 DEV-020）。
 * 返回 agent 的回复（含笔记路径）。
 */
/** 本地日期 YYYY-MM-DD（勿用 toISOString——那是 UTC，会比北京时间慢一天） */
function localDate(d = new Date()): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

export async function generateDailyReview(plugin: ArkOSPlugin): Promise<string> {
  const s = plugin.data.settings;
  const today = localDate();
  const prompt =
    "你是每日回顾助手。生成今天的每日回顾笔记：\n" +
    "1) 调 mcp_obsidian_brain_vault_scan 或 mcp_obsidian_brain_vault_search 扫描 Vault 中今天新增/修改的笔记；\n" +
    "2) 概述今天学习了什么、产出了什么、值得注意的观点；\n" +
    "3) 按结构化 Markdown 生成（frontmatter 含 title、type: daily-review、date、tags）；\n" +
    `4) 用 mcp_obsidian_brain_vault_write 写入 Vault 的 ${s.reviewFolder || "02-DB/回顾"}/${today}-每日回顾.md；若当天无新增笔记，写简短说明"今日无新增"。\n` +
    "完成后回复笔记路径和字数。";
  const messages: AiMessage[] = [{ role: "user", content: prompt }];
  // 3 分钟超时兜底，避免 Hermes 挂起时按钮永远"生成中…"
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), 180000);
  try {
    if (s.agentProvider === "hermes") {
      return await agentChat(s, messages, {}, ac.signal);
    }
    return await chat(s, messages);
  } finally {
    clearTimeout(timer);
  }
}
