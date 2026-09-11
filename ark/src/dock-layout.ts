// dock 首页布局解析（§16.1 M-D2）：dashboard.md frontmatter `sections` 驱动区块顺序/来源（Note-as-Layout）。
// 解析失败（无 sections / 结构非法）回退内置默认布局，绝不抛错中断首页。
import type ArkOSPlugin from "./main";
import { parseYaml } from "obsidian";

export type DockSectionType = "todo" | "memo" | "notes" | "stats";

export interface DockLayoutSection {
  id: string;
  type: DockSectionType;
  source?: string; // 目录路径前缀（空=全量）
  title: string;
}

export interface DockLayout {
  dashboard: boolean;
  banner?: {
    backgroundImage?: string; // Vault 内相对路径；空则回退 settings.backgroundImagePath
    showStats?: boolean;
  };
  sections: DockLayoutSection[];
}

const KNOWN_TYPES: DockSectionType[] = ["todo", "memo", "notes", "stats"];

/** 内置默认布局（与 M-D1 固定三卡 + KPI 一致） */
export const DEFAULT_DOCK_LAYOUT: DockLayout = {
  dashboard: true,
  banner: { showStats: true },
  sections: [
    { id: "todo-today", type: "todo", title: "今日待办" },
    { id: "idea-notes", type: "memo", title: "灵感速记" },
    { id: "quick-notes", type: "notes", title: "快捷笔记" },
  ],
};

/** 从 markdown 解析 frontmatter 布局；非法输入抛错由调用方回退默认 */
export function parseDockLayout(markdown: string): DockLayout {
  const fm = parseFrontmatter(markdown);
  const sections: DockLayoutSection[] = [];
  if (Array.isArray(fm.sections)) {
    for (const raw of fm.sections) {
      if (!raw || typeof raw !== "object") continue;
      const o = raw as Record<string, unknown>;
      const type = o.type as DockSectionType;
      if (!KNOWN_TYPES.includes(type) || typeof o.title !== "string") continue;
      sections.push({
        id: typeof o.id === "string" && o.id ? o.id : `sec-${sections.length}`,
        type,
        title: o.title,
        source: typeof o.source === "string" && o.source ? o.source : undefined,
      });
    }
  }
  if (sections.length === 0) throw new Error("no sections");
  let banner: DockLayout["banner"] = Object.assign({}, DEFAULT_DOCK_LAYOUT.banner);
  const b = (fm.banner ?? {}) as Record<string, unknown>;
  if (typeof b.backgroundImage === "string" && b.backgroundImage) banner = { ...banner, backgroundImage: b.backgroundImage };
  if (typeof b.showStats === "boolean") banner = { ...banner, showStats: b.showStats };
  return { dashboard: true, banner, sections };
}

/** 读取布局；文件缺失/非法回退默认 */
export async function loadDockLayout(plugin: ArkOSPlugin): Promise<DockLayout> {
  const path = plugin.data.settings.dashboardFile || "dashboard.md";
  try {
    const f = plugin.app.vault.getAbstractFileByPath(path);
    if (!f) return DEFAULT_DOCK_LAYOUT;
    const content = await plugin.app.vault.cachedRead(f as any);
    return parseDockLayout(content);
  } catch {
    return DEFAULT_DOCK_LAYOUT;
  }
}

/** 当前布局文件路径（view.ts / dock.ts 监听热更新用） */
export function dockLayoutPath(plugin: ArkOSPlugin): string {
  return plugin.data.settings.dashboardFile || "dashboard.md";
}

function parseFrontmatter(markdown: string): Record<string, unknown> {
  const m = /^---\r?\n([\s\S]*?)^---\r?\n?/m.exec(markdown);
  if (!m) return {};
  try {
    const y = parseYaml(m[1]);
    return y && typeof y === "object" && !Array.isArray(y) ? (y as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}