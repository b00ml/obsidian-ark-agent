// 统一 KPI 统计口径（B3 / §17.3）：首页与各模块页取数同源，保证数字一致。
// 取数原则：待办/日志/灵感统一读 ArkData（内存集合，sync.ts 填充）；仅邮件因不在
// ArkData 而走 mail 模块异步计数（comm.ts 维护）。无第三套机制。
import type ArkOSPlugin from "./main";
import { isBlacklisted, getToday } from "./utils";
import { listMails } from "./mail";

interface TodoTaskLite {
  completed?: boolean;
  status?: string;
  notePath?: string;
}

/** 待办：未完成任务数 = 各清单未完成任务去重后的总数（notePath 唯一） */
export function todoCount(plugin: ArkOSPlugin): number {
  const seen = new Set<string>();
  let count = 0;
  for (const list of plugin.data.todoLists) {
    for (const t of list.tasks as TodoTaskLite[]) {
      if (t.completed || (t.status && t.status === "done")) continue;
      const key = t.notePath || "id-" + count;
      if (seen.has(key)) continue;
      seen.add(key);
      count++;
    }
  }
  return count;
}

/** 日志：今日新增日志条数（createdAt 落在今天） */
export function todayLogCount(plugin: ArkOSPlugin): number {
  const today = getToday();
  return plugin.data.logs.filter((l) => {
    const d = new Date(l.createdAt);
    if (isNaN(d.getTime())) return false;
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${d.getFullYear()}-${m}-${day}` === today;
  }).length;
}

/** 灵感：受条数（黑名单过滤后，不含归档） */
export function ideaCount(plugin: ArkOSPlugin): number {
  const black = plugin.data.settings.scanBlacklist || [];
  return plugin.data.ideas.filter((i) => !i.archived && !isBlacklisted(i.notePath, black)).length;
}

/** 邮件：未读邮件数（异步，inbox）；失败/未授权返回 0（不阻塞首页渲染） */
export async function unreadMailCount(): Promise<number> {
  try {
    const ms = await listMails({ dir: "inbox", limit: 50 });
    return ms.filter((m) => !m.is_read).length;
  } catch {
    return 0;
  }
}

/** 首页 Banner KPI 一次取全（同步部分 + 邮件 Promise） */
export function dockStats(plugin: ArkOSPlugin) {
  return {
    todo: todoCount(plugin),
    log: todayLogCount(plugin),
    idea: ideaCount(plugin),
    mail: unreadMailCount(), // Promise<number>
  };
}