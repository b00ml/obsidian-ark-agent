// 数据库沉淀 + 日/周报告生成
import type ArkOSPlugin from "./main";
import { generateId, getToday } from "./utils";
import { writeMarkdown } from "./sync";

/** 把一段内容沉淀为数据库条目（写 md 到 databaseFolder 触发 watcher 入库） */
export async function promoteToDatabase(
  plugin: ArkOSPlugin,
  opts: { name: string; tags?: string[]; categoryId?: string; description?: string },
): Promise<string> {
  const s = plugin.data.settings;
  const today = getToday();
  const fm: Record<string, unknown> = {
    id: generateId(),
    db_name: opts.name,
    status: "pending",
    tags: opts.tags ?? [],
    category_id: opts.categoryId ?? s.databaseCategories[0]?.id ?? "reference",
    created_at: new Date().toISOString(),
    description: opts.description ?? "",
  };
  return writeMarkdown(plugin, s.databaseFolder, `${today}-${Date.now().toString(36)}`, fm, opts.description ?? "");
}

type Period = "daily" | "weekly";

/** 生成日报/周报（日志四维分组 → 落 reportFolder → 同步 database） */
export async function generateReport(plugin: ArkOSPlugin, period: Period): Promise<string> {
  const s = plugin.data.settings;
  const now = new Date();
  const today = getToday();

  let start: Date;
  if (period === "daily") {
    start = now;
  } else {
    const day = now.getDay() || 7; // 周日={7}
    start = new Date(now.getFullYear(), now.getMonth(), now.getDate() - (day - 1));
  }

  const startStr = start.toISOString().slice(0, 10);
  const inRange = (d: number) => !(d < start.setHours(0, 0, 0, 0) || d > now.getTime());
  // start.setHours 会改 start；重算 startStr 用副本
  const startMs = new Date(start.getFullYear(), start.getMonth(), start.getDate()).getTime();
  const logs = plugin.data.logs.filter((l) => l.createdAt >= startMs && l.createdAt <= now.getTime());

  // 四维分组
  const group = (pred: (t: string[]) => boolean) => logs.filter((l) => pred(l.tags));
  const tWork = (t: string[]) => t.includes("工作") || t.includes("job");
  const tPersonal = (t: string[]) => t.includes("个人") || t.includes("personal");
  const tDone = (t: string[]) => t.includes("已完成");
  const work = group(tWork);
  const personal = group(tPersonal);
  const done = group(tDone);
  const other = logs.filter((l) => !tWork(l.tags) && !tPersonal(l.tags) && !tDone(l.tags));

  const section = (title: string, list: typeof logs): string => {
    if (list.length === 0) return "";
    const lines = list.map((l) => `- ${l.title}：${l.content}`).join("\n");
    return `\n**${title}** (${list.length}条):\n${lines}`;
  };

  const body = [
    `日期：${today}`,
    `共 ${logs.length} 条记录`,
    section("工作", work),
    section("个人", personal),
    section("已完成", done),
    section("其他", other),
  ].join("\n");

  const safe = today.replace(/-/g, "");
  const filename = `${safe}${period === "weekly" ? "-周报" : "-日报"}`;
  const fm: Record<string, unknown> = {
    type: "report",
    report_type: period,
    date: today,
    period_start: startStr,
    log_count: logs.length,
    work_count: work.length,
    personal_count: personal.length,
    generated_at: new Date().toISOString(),
  };
  const path = await writeMarkdown(plugin, s.reportFolder, filename, fm, `# ${period === "weekly" ? "周报" : "日报"}\n` + body);
  return path;
}