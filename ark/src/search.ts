// 知识库搜索：本地 Vault 分词检索 + 全网（DuckDuckGo）
import { Modal, Setting, TFile, requestUrl, Notice } from "obsidian";
import type ArkOSPlugin from "./main";
import { sendSelectionToWorkbench } from "./agent-workbench";

const STOPWORDS = new Set([
  "的", "了", "在", "是", "和", "与", "及", "或", "有", "也", "都", "而", "以及", "把",
  "被", "从", "到", "对", "为", "关于", "这个", "那个", "一个", "我们", "你们", "他们",
  "但", "并", "你", "我", "他", "它", "a", "an", "the", "and", "or", "of", "to", "in",
  "is", "are", "for", "on", "with", "as", "by", "at", "from",
]);

/** 查询分词（导出以便单测：停用词与分隔符规则是检索质量的地基）。 */
export function tokenize(q: string): string[] {
  return q
    .toLowerCase()
    .split(/[\s,，。.;；:：!?！？"'“”‘’()（）[\]【】{}+=*\/\\\-–—_#@&%$~^`]+/)
    .map((t) => t.trim())
    .filter((t) => t.length > 0 && !STOPWORDS.has(t));
}

export interface VaultHit {
  file: TFile;
  score: number;
  snippet: string;
  summary: string;
  sourceRef: string;
  sourceType: "vault";
  lineStart: number;
  lineEnd: number;
}

/**
 * 扫描黑名单 glob → 正则。
 *
 * 必须**单次扫描**替换：分两次 replace 时，第一步先把双星号换成正则的点星号，第二步针对星号的
 * 全局替换会把这个结果里的星号再替换一次，于是双星号（本意跨目录）退化成了同层匹配——
 * 例如 `03-待办/**.md` 匹配不到 `03-待办/子/a.md`，会静默影响 scanBlacklist 的跨目录模式。
 *
 * 已知限制：模式里的正则元字符（如点号）未转义，`dashboard.md` 中的点号会当通配符用。
 * 这是既有行为，改动会让现存模式突然不再匹配，故本次不动。
 */
export function globToRegExp(pattern: string): RegExp {
  return new RegExp(pattern.replace(/\*\*|\*/g, (m) => (m === "**" ? ".*" : "[^/]*")));
}

/** 路径是否命中扫描黑名单（空模式忽略）。 */
export function isBlacklisted(path: string, patterns: string[]): boolean {
  return patterns.some((p) => !!p && globToRegExp(p).test(path));
}

export interface TextScore {
  /** 全部检索词都出现才算命中（AND 语义）。 */
  matched: boolean;
  score: number;
  firstIdx: number;
  lineStart: number;
  snippet: string;
}

/**
 * 单文件打分（纯函数，便于单测）。
 * 规则：文件名命中 +6，仅正文命中 +4（两者互斥），标题行含首词额外 +3；命中位置取最早出现。
 */
export function scoreText(path: string, text: string, terms: string[]): TextScore {
  const lower = text.toLowerCase();
  if (!terms.every((t) => lower.includes(t))) {
    return { matched: false, score: 0, firstIdx: -1, lineStart: 1, snippet: "" };
  }
  const base = basename(path).toLowerCase();
  let score = 0;
  let firstIdx = -1;
  for (const t of terms) {
    if (base.includes(t)) score += 6;
    else if (lower.includes(t)) score += 4;
    const i = lower.indexOf(t);
    if (i >= 0 && (firstIdx === -1 || i < firstIdx)) firstIdx = i;
  }
  // 标题行加权（以 # 开始）
  const headIdx = text.split("\n").findIndex((l) => l.trimStart().startsWith("#") && l.toLowerCase().includes(terms[0]));
  if (headIdx >= 0) score += 3;
  return {
    matched: true,
    score,
    firstIdx,
    lineStart: firstIdx >= 0 ? text.slice(0, firstIdx).split("\n").length : 1,
    snippet: makeSnippet(text, firstIdx),
  };
}

/** 本地 Vault 分词检索（全库 .md），按命中项打分，返回 top N */
export async function vaultSearch(plugin: ArkOSPlugin, query: string, limit = 20): Promise<VaultHit[]> {
  const terms = tokenize(query);
  if (terms.length === 0) return [];
  const files = plugin.app.vault.getMarkdownFiles();
  const hits: VaultHit[] = [];

  for (const file of files) {
    let text: string;
    try { text = await plugin.app.vault.cachedRead(file); } catch { continue; }
    if (isBlacklisted(file.path, plugin.data.settings.scanBlacklist)) continue;

    const s = scoreText(file.path, text, terms);
    if (!s.matched) continue;
    hits.push({ file, score: s.score, snippet: s.snippet, summary: s.snippet,
      sourceRef: file.path, sourceType: "vault",
      lineStart: s.lineStart, lineEnd: s.lineStart });
  }

  return hits.sort((a, b) => b.score - a.score).slice(0, limit);
}

function basename(p: string): string {
  return p.split("/").pop()?.replace(/\.md$/, "") ?? p;
}

/** 命中位置附近的片段裁剪（导出以便单测）。 */
export function makeSnippet(text: string, idx: number): string {
  if (idx < 0) return text.replace(/\s+/g, " ").slice(0, 80);
  const start = Math.max(0, idx - 24);
  return (start > 0 ? "…" : "") + text.slice(start, idx + 60).replace(/\s+/g, " ") + "…";
}

export interface WebHit { title: string; url: string; snippet: string }

/** 全网搜索（DuckDuckGo HTML），返回结果列表 */
export async function webSearch(query: string, limit = 10): Promise<WebHit[]> {
  const url = "https://html.duckduckgo.com/html/?q=" + encodeURIComponent(query);
  const resp = await requestUrl({ url });
  const doc = new DOMParser().parseFromString(resp.text, "text/html");
  const out: WebHit[] = [];
  const nodes = doc.querySelectorAll(".result");
  for (const node of Array.from(nodes)) {
    if (out.length >= limit) break;
    const a = node.querySelector(".result__a");
    const snip = node.querySelector(".result__snippet");
    if (!a) continue;
    const href = a.getAttribute("href") || "";
    let real = href;
    if (href.startsWith("//duckduckgo.com/l/")) {
      const m = href.match(/uddg=([^&]+)/);
      if (m) real = decodeURIComponent(m[1]);
    }
    out.push({ title: a.textContent?.trim() ?? "", url: real, snippet: snip?.textContent?.trim() ?? "" });
  }
  if (out.length === 0) throw new Error("DuckDuckGo 未返回结果（可能被反爬）");
  return out;
}

/** 搜索模态：本地 / 全网 两页 */
export function openSearch(plugin: ArkOSPlugin): void {
  new SearchModal(plugin).open();
}

class SearchModal extends Modal {
  private plugin: ArkOSPlugin;
  private mode: "local" | "web" = "local";
  private resultsEl!: HTMLElement;

  constructor(plugin: ArkOSPlugin) {
    super(plugin.app);
    this.plugin = plugin;
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.addClass("search-modal");
    contentEl.createDiv({ cls: "sos-settings-title", text: "⌕ 知识库搜索" });

    const tabs = contentEl.createDiv({ cls: "sos-settings-tabs" });
    const mkTab = (m: typeof this.mode, label: string) => {
      const b = tabs.createEl("button", { cls: `sos-tab-btn${this.mode === m ? " active" : ""}`, text: label });
      b.addEventListener("click", () => { this.mode = m; tabs.querySelectorAll(".sos-tab-btn").forEach((el) => el.toggleClass("active", (el as HTMLElement).dataset.m === m)); });
      b.dataset.m = m;
      return b;
    };
    mkTab("local", "本库检索");
    mkTab("web", "全网检索");

    let q = "";
    new Setting(contentEl).setName("查询").addText((t) => { t.inputEl.setAttribute("type", "search"); t.setValue(""); t.onChange((v) => { q = v; }); t.inputEl.addEventListener("keydown", (e) => { if (e.key === "Enter") void doSearch(q); }); }).addButton((b) => b.setButtonText("搜索").setCta().onClick(() => void doSearch(q)));

    this.resultsEl = contentEl.createDiv({ cls: "search-results" });

    const doSearch = async (query: string) => {
      if (!query.trim()) return;
      this.resultsEl.empty();
      this.resultsEl.createDiv({ cls: "tactical-empty", text: "搜索中…" });
      try {
        if (this.mode === "local") {
          const hits = await vaultSearch(this.plugin, query);
          this.resultsEl.empty();
          if (hits.length === 0) { this.resultsEl.createDiv({ cls: "tactical-empty", text: "本库无匹配" }); return; }
          hits.forEach((h) => {
            const row = this.resultsEl.createDiv({ cls: "search-row" });
            const title = row.createEl("button", { cls: "search-row-title", text: h.file.basename });
            title.addEventListener("click", () => { void this.app.workspace.getLeaf(true).openFile(h.file); });
            row.createDiv({ cls: "search-row-meta", text: `本库 · 第 ${h.lineStart} 行 · 得分 ${h.score} · ${h.file.path}` });
            row.createDiv({ cls: "search-row-snippet", text: h.snippet });
            const actions = row.createDiv({ cls: "search-row-actions" });
            actions.createEl("button", { cls: "search-row-context", text: "加入 Agent" }).addEventListener("click", () => {
              sendSelectionToWorkbench(this.plugin,
                `来源：[[${h.file.path}]]（第 ${h.lineStart} 行）\n\n${h.snippet}`);
              this.close();
            });
          });
        } else {
          const hits = await webSearch(query);
          this.resultsEl.empty();
          hits.forEach((h) => {
            const row = this.resultsEl.createDiv({ cls: "search-row" });
            row.createEl("a", { cls: "search-row-title", text: h.title, href: h.url, attr: { target: "_blank" } });
            row.createDiv({ cls: "search-row-snippet", text: h.snippet });
          });
        }
      } catch (e: any) {
        this.resultsEl.empty();
        this.resultsEl.createDiv({ cls: "tactical-empty", text: "搜索失败: " + String(e?.message ?? e) });
        new Notice("搜索失败: " + String(e?.message ?? e));
      }
    };
  }

  onClose() { this.contentEl.empty(); }
}
