// 循环任务工具：recurrence 序列化 / 解析 / 下一到期日推算
import type { Recurrence, Frequency } from "./types";

/** 序列化 recurrence → 存入 frontmatter 的字符串（JSON，none→"none"） */
export function serializeRecurrence(rec?: Recurrence): string {
  if (!rec || !rec.frequency || rec.frequency === "none") return "none";
  const o: Record<string, unknown> = { frequency: rec.frequency };
  if (rec.daysOfWeek?.length) o.days_of_week = rec.daysOfWeek;
  if (rec.daysOfMonth?.length) o.days_of_month = rec.daysOfMonth;
  if (rec.month) o.month = rec.month;
  if (rec.day) o.day = rec.day;
  if (rec.count != null) o.count = rec.count;
  if (rec.until) o.until = rec.until;
  return JSON.stringify(o);
}

/** 解析 frontmatter 中的 recurrence 字符串 → Recurrence | undefined */
export function parseRecurrence(s?: string | undefined): Recurrence | undefined {
  if (!s || s === "none" || s === "undefined") return undefined;
  try {
    const o = JSON.parse(s);
    const f = o.frequency as Frequency;
    if (!f || f === "none") return undefined;
    return {
      frequency: f,
      daysOfWeek: o.days_of_week,
      daysOfMonth: o.days_of_month,
      month: o.month,
      day: o.day,
      count: o.count,
      until: o.until,
    };
  } catch {
    return undefined;
  }
}

function addDays(d: Date, n: number): Date {
  const x = new Date(d);
  x.setDate(x.getDate() + n);
  return x;
}

function fmt(d: Date): string {
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${m}-${day}`;
}

/** 从 base 起推算下一个满足 recurrence 的日期（YYYY-MM-DD），base 当日不算（至少 +1 天） */
export function getNextDueDate(rec: Recurrence, base: Date): string {
  let d = addDays(base, 1);
  const freq = rec.frequency;
  let guard = 0;
  while (guard++ < 1200) {
    if (freq === "daily") return fmt(d);
    if (freq === "weekly") {
      const days = rec.daysOfWeek?.length ? rec.daysOfWeek : [1];
      const dow = d.getDay() === 0 ? 7 : d.getDay(); // 周日归一为 7
      if (days.includes(dow)) return fmt(d);
    } else if (freq === "monthly") {
      const days = rec.daysOfMonth?.length ? rec.daysOfMonth : [1];
      if (days.includes(d.getDate())) return fmt(d);
    } else if (freq === "yearly") {
      const m = rec.month || 1;
      const day = rec.day || 1;
      if (d.getMonth() + 1 === m && d.getDate() === day) return fmt(d);
    }
    d = addDays(d, 1);
  }
  return fmt(d);
}

/** 今天是否落在当前 recurrence 内 */
export function matchesRecurrence(dateStr: string, rec: Recurrence, base: Date): boolean {
  const want = getNextDueDate(rec, base);
  return want === dateStr;
}