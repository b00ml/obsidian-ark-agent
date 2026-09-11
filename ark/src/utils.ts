// 通用工具函数

/** 生成唯一 ID（同原实现风格） */
export function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
}

/** 联系人 ID */
export function contactId(): string {
  return "ct_" + Date.now().toString(16) + Math.random().toString(16).substr(2, 9);
}

/** 本地 YYYY-MM-DD */
export function getToday(): string {
  const d = new Date();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${m}-${day}`;
}

/** 本地 YYYY.MM.DD 与星期 */
export function getDateLabel(date?: Date): string {
  const d = date ?? new Date();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  const week = "日一二三四五六".charAt(d.getDay());
  return `${d.getFullYear()}.${m}.${day} 周${week}`;
}

/** 当天问候语 */
export function getGreeting(): string {
  const h = new Date().getHours();
  if (h < 5) return "夜深了";
  if (h < 9) return "早上好";
  if (h < 12) return "上午好";
  if (h < 14) return "中午好";
  if (h < 18) return "下午好";
  return "晚上好";
}

/** 文件名清洗 */
export function sanitizeFilename(name: string): string {
  return name.replace(/[\\/:*?"<>|]/g, "-");
}

/** 本地 HH-MM-SS（冒号转连字符，用于文件名） */
export function timeToFile(d: Date): string {
  const t = d.toTimeString().split(" ")[0];
  return t.replace(/:/g, "-");
}

/** 联系人头像色板 */
export const AVATAR_PALETTE = [
  "#00e5ff", "#b34dff", "#00ff99", "#ffcc00", "#ff6699", "#ff8800", "#0066ff",
];

export function randomPalette(): string {
  return AVATAR_PALETTE[Math.floor(Math.random() * AVATAR_PALETTE.length)];
}

/**
 * 黑名单 glob → 正则
 * ** 任意、* 非 "/" 字符、目录以 "/" 结尾加 .*、非 * 结尾加 (?:/|$)
 */
export function blacklistToRegExp(pattern: string): RegExp {
  let re = pattern
    .replace(/\*\*/g, "\u0000") // 占位
    .replace(/\*/g, "[^/]*")
    .replace(/\u0000/g, ".*");
  if (pattern.endsWith("/")) re += ".*";
  else if (!pattern.endsWith("*")) re += "(?:/|$)";
  return new RegExp(re);
}

export function isBlacklisted(path: string | undefined, patterns: string[]): boolean {
  if (!path) return false;
  for (const p of patterns) {
    if (p.length > 0 && blacklistToRegExp(p).test(path)) return true;
  }
  return false;
}
