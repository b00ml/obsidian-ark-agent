import type { SpaceOSView } from "../view";
import type { TabName } from "../view";

const PLACEHOLDER_DESC: Partial<Record<TabName, string>> = {
  database: "数据库沉淀（frontmatter 映射 / 分类 / 提醒 / 统计）",
  weaponry: "训练记录（Shell 工具 + 训练结果）",
  medical: "健康记录（健康数据 / 记录 / 评分）",
  ideas: "灵感草稿（想法收集 / 归档 / 统计）",
  drawing: "随笔（画板创建 / 描述管理）",
  comm: "消息与邮箱（IMAP/SMTP 邮件 + 联系人 + AI 归档）",
};

/** 尚未实现模块的占位 */
export function renderPanelsPlaceholder(view: SpaceOSView, tab: TabName, mount: HTMLElement) {
  mount.addClass("space-placeholder");
  mount.createDiv({ cls: "space-placeholder-title", text: `◈ ${tab.toUpperCase()} · 建设中` });
  mount.createDiv({ cls: "space-placeholder-body", text: PLACEHOLDER_DESC[tab] ?? "模块规划中" });
  view.plugin.data; // 保持引用
}
