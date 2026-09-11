import type ArkOSPlugin from "./main";
import { agentChat, chat, type AiMessage } from "./ai";

/**
 * 手动触发"周报回顾"生成：让 agent 扫描本周笔记 → 生成深度周报 → 写 Vault reviewFolder（默认 02-DB/回顾）。
 * 走已验证的 /v1 agent 路径（Hermes 模式）或直连 chat（非 Hermes 模式）；
 * 不走 Hermes cron agent 模式（本环境会空转烧 CPU，见 DEV-020）。
 * 返回 agent 的回复（含笔记路径）。
 */
/** 本地日期 YYYY-MM-DD（勿用 toISOString——那是 UTC，会比北京时间慢一天） */
function localDate(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

/** 本周一（周日=7）的本地日期 */
function mondayDate(d = new Date()): Date {
  const day = d.getDay() || 7; // 周日={7}
  const monday = new Date(d.getFullYear(), d.getMonth(), d.getDate() - (day - 1));
  return monday;
}

export async function generateWeeklyReview(plugin: ArkOSPlugin): Promise<string> {
  const s = plugin.data.settings;
  const now = new Date();
  const monday = mondayDate(now);
  const mondayStr = localDate(monday);
  const todayStr = localDate(now);
  const filename = `${mondayStr}-周报.md`;

  const prompt =
    "你是周报回顾助手。生成本周的周报笔记：\n" +
    `1) 本周范围：${mondayStr}（周一）到 ${todayStr}（今天）。调 mcp_obsidian_brain_vault_scan 或 mcp_obsidian_brain_vault_search 扫描 Vault 中本周新增/修改的笔记（按 frontmatter 的 created/updated 或文件修改时间过滤）；\n` +
    "2) 概述本周学习了什么、产出了什么、值得注意的观点、本周趋势与下周计划；\n" +
    `3) 按结构化 Markdown 生成（frontmatter 含 title、type: weekly-review、week_start: ${mondayStr}、week_end: ${todayStr}、date、tags: [weekly-review, 回顾]）；\n` +
    `4) 用 mcp_obsidian_brain_vault_write 写入 Vault 的 ${s.reviewFolder || "02-DB/回顾"}/${filename}；若本周无新增笔记，写简短说明"本周无新增"。\n` +
    "完成后回复笔记路径和字数。";
  const messages: AiMessage[] = [{ role: "user", content: prompt }];
  // 5 分钟超时兜底（周报比日报扫的笔记更多，给更长时间），避免 Hermes 挂起时按钮永远"生成中…"
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), 300000);
  try {
    if (s.agentProvider === "hermes") {
      return await agentChat(s, messages, {}, ac.signal);
    }
    return await chat(s, messages);
  } finally {
    clearTimeout(timer);
  }
}
