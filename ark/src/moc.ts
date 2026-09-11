// M4 产出体系：MOC 知识地图（§3.4 + 决策 D3：手动按钮触发）
// 输入一个主题词，扫描知识层各产物目录（简报/要点卡/问答/回顾），按「frontmatter tags 命中
// 或 文件名+正文含主题词」聚合命中笔记，按来源分节、带一句话摘要，落 `mocFolder/{主题}-MOC.md`。
// 纯本地聚合、无 AI：产物本身已带 tag/结构，直接内链 `[[...]]` 即成反链导航页。
import { TFile } from "obsidian";
import type ArkOSPlugin from "./main";
import { sanitizeFilename } from "./utils";

export interface MocResult {
  topic: string;
  match: number;
  path: string;
}

async function ensureFolder(plugin: ArkOSPlugin, path: string) {
  const parts = path.split("/").filter(Boolean);
  let cur = "";
  for (const p of parts) {
    cur = cur ? `${cur}/${p}` : p;
    if (!plugin.app.vault.getAbstractFileByPath(cur)) {
      try { await plugin.app.vault.createFolder(cur); } catch { /* 并发/已存在 */ }
    }
  }
}

function splitFm(text: string): { fm: string; body: string } {
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(text);
  if (!m) return { fm: "", body: text };
  return { fm: m[1], body: text.slice(m[0].length) };
}

const TYPE_LABEL: Record<string, string> = {
  "point-card": "要点卡",
  "feed-brief": "订阅简报",
  "daily-brief": "信息日报",
  "learning-quiz": "学习问答",
  "daily-review": "每日回顾",
  "weekly-review": "周报回顾",
};

/** 提取一句摘要：一句话结论 > 首条要点 > 首段非空纯文本 */
function summary(body: string): string {
  const conclusion = /^>\s*一句话结论：\s*(.+)$/m.exec(body)?.[1]?.trim();
  if (conclusion) return conclusion.slice(0, 80);
  const point = /^-\s*(.+)$/m.exec(body)?.[1]?.trim();
  if (point) return point.slice(0, 80);
  const para = body
    .split(/\n+/)
    .map((l) => l.replace(/^[#>-]\s*/, "").trim())
    .find((l) => l.length > 4);
  return (para ?? "").slice(0, 80);
}

/** 生成主题 MOC 导航页 */
export async function generateMoc(plugin: ArkOSPlugin, topic: string): Promise<MocResult> {
  const s = plugin.data.settings;
  const mocFolder = s.mocFolder || "02-DB/主题";
  const topicLower = topic.trim().toLowerCase();
  const today = new Date();

  // 聚合待索引的知识层目录（去除 _errors 等内部目录）
  const dirs = new Set<string>([
    s.feedArchiveFolder || "02-DB/简报",
    s.pipelineFolder || "02-DB/要点",
    s.quizFolder || "02-DB/问答",
    s.reviewFolder || "02-DB/回顾",
  ]);
  const files = plugin.app.vault.getFiles().filter((f: TFile) => {
    if (f.extension !== "md") return false;
    for (const d of dirs) if (f.path.startsWith(d + "/")) return true;
    return false;
  });

  const hits: { file: TFile; type: string; label: string; note: string; tags: string[] }[] = [];
  for (const f of files) {
    const text = await plugin.app.vault.read(f);
    const { fm, body } = splitFm(text);
    const tags = (fm.match(/(?:^|\n)\s*tags:\s*\[([^\]]*)\]/)?.[1] ?? "")
      .split(",").map((t: string) => t.trim().replace(/^["']|["']$/g, "")).filter(Boolean);
    // 命中：tags 任一含主题词，或 文件名/正文含主题词（前面 500 字足够代表主题）
    const hitByTag = tags.some((t) => t.toLowerCase().includes(topicLower));
    const hay = `${f.basename} ${body.slice(0, 500)}`.toLowerCase();
    if (!hitByTag && !hay.includes(topicLower)) continue;
    const type = (fm.match(/(?:^|\n)\s*type:\s*([^\n]+)/)?.[1] ?? "").trim().replace(/^["']|["']$/g, "") || "other";
    hits.push({
      file: f,
      type,
      label: TYPE_LABEL[type] ?? "其他",
      note: summary(body),
      tags,
    });
  }

  // 按 type 分组排序，保持导航可读
  const groups = new Map<string, typeof hits>();
  for (const h of hits) {
    if (!groups.has(h.label)) groups.set(h.label, []);
    groups.get(h.label)!.push(h);
  }

  const mocPath = `${mocFolder}/${sanitizeFilename(topic.trim())}-MOC.md`;
  await ensureFolder(plugin, mocFolder);
  const lines: string[] = [
    "---",
    "type: moc",
    `topic: ${topic.trim()}`,
    `date: ${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`,
    `count: ${hits.length}`,
    hits.some((h) => h.tags.length) ? `tags: [${topic.trim()}]` : "tags: []",
    "---",
    "",
    `# ${topic.trim()} · 知识地图`,
    "",
    `聚合 ${hits.length} 条相关笔记：`,
    "",
  ];
  if (hits.length === 0) {
    lines.push("_该主题暂未命中任何笔记，可先采集/归档/处理后重试。_");
  } else {
    for (const [label, list] of groups) {
      lines.push(`## ${label}（${list.length}）`);
      lines.push("");
      for (const h of list) {
        const tag = h.tags.length ? ` · ${h.tags.map((t) => `#${t}`).join(" ")}` : "";
        lines.push(`- [[${h.file.basename}]] — ${h.note || "（无摘要）"}${tag}`);
      }
      lines.push("");
    }
  }
  await plugin.app.vault.create(mocPath, lines.join("\n"));

  return { topic: topic.trim(), match: hits.length, path: mocPath };
}