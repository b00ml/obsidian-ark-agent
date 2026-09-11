// 信息订阅（知识信息闭环 M2 / §3.1）：主动抓取订阅源 → 增量去重 → AI 简述+原链接 → 落 feedInboxFolder。
// 复用 search.webSearch（keyword_search）、ai-asst contextual 通道、writeMarkdown 落库。
import { requestUrl, Notice, TFile } from "obsidian";
import type ArkOSPlugin from "./main";
import type { FeedConfig } from "./settings";
import type { AiMessage } from "./ai";
import { runChannel, extractJson, JsonExtractError, logAiError } from "./ai-asst";
import { webSearch } from "./search";
import { writeMarkdown } from "./sync";
import { getToday, sanitizeFilename } from "./utils";
import topicNote from "./prompts/topic-note.st";
import { capturePatch } from "./capture";

export interface FeedItem {
  title: string;
  url: string;
  content: string;   // 摘要/描述/正文片段
  pubDate?: string;
}

interface Brief {
  title: string;
  conclusion: string;
  points: string[];
  url: string;
  tags: string[];
}

/** URL 指纹（去重）：去协议 + 去尾斜杠 + 去查询锚点 */
export function feedFingerprint(url: string): string {
  try {
    const u = new URL(url);
    return `${u.hostname}${u.pathname}`.replace(/\/+$/, "").toLowerCase();
  } catch {
    return url.replace(/^[a-z]+:\/\//i, "").replace(/[?#].*$/, "").replace(/\/+$/, "").toLowerCase();
  }
}

/** 抓取单个订阅源的原始条目 */
export async function fetchFeedItems(plugin: ArkOSPlugin, feed: FeedConfig): Promise<FeedItem[]> {
  switch (feed.source) {
    case "rss": {
      if (!feed.target) return [];
      const resp = await requestUrl({ url: feed.target });
      const doc = new DOMParser().parseFromString(resp.text, "text/xml");
      const items: FeedItem[] = [];
      const nodes = doc.querySelectorAll("item");
      for (const node of Array.from(nodes)) {
        const title = node.querySelector("title")?.textContent?.trim() ?? "";
        const link = node.querySelector("link")?.textContent?.trim() ?? "";
        if (!title || !link) continue;
        items.push({
          title,
          url: link,
          content: (node.querySelector("description")?.textContent ?? "").replace(/<[^>]+>/g, "").trim().slice(0, 500),
          pubDate: node.querySelector("pubDate")?.textContent?.trim() ?? undefined,
        });
      }
      // Atom 兜底
      if (items.length === 0) {
        for (const node of Array.from(doc.querySelectorAll("entry"))) {
          const title = node.querySelector("title")?.textContent?.trim() ?? "";
          const link = node.querySelector("link")?.getAttribute("href") ?? "";
          if (!title || !link) continue;
          items.push({ title, url: link, content: (node.querySelector("summary")?.textContent ?? "").slice(0, 500) });
        }
      }
      return items;
    }
    case "url": {
      // 单条直达链接：抓取真实页面标题+正文摘要，供 AI 简述该页本身的实质内容；
      // 抓不到正文也要至少带真实 <title>，避免只剩订阅名让 AI 脑补成"每日聚合清单"
      if (!feed.target) return [];
      let title = feed.name || "链接";
      let content = "";
      try {
        const resp = await requestUrl({ url: feed.target, headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" } });
        const text = resp.text || "";
        const doc = new DOMParser().parseFromString(text, "text/html");
        const t = doc.querySelector("title")?.textContent?.trim();
        if (t) title = t;
        const desc = doc.querySelector('meta[name="description"]')?.getAttribute("content")?.trim()
          || doc.querySelector('meta[property="og:description"]')?.getAttribute("content")?.trim();
        content = (desc || text.replace(/<script[\s\S]*?<\/script>|<style[\s\S]*?<\/style>/gi, "").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim()).slice(0, 600);
      } catch { /* 抓不到则保留单条标题，AI 至少能简述标题 */ }
      return [{ title, url: feed.target, content }];
    }
    case "keyword_search": {
      const hits = await webSearch(feed.target || feed.name, 15);
      return hits.map((h) => ({ title: h.title, url: h.url, content: h.snippet, pubDate: undefined }));
    }
    case "bilibili":
    case "wechat":
      throw new Error("B站/公众号订阅请走既有收件箱采集（inbox_collector），本订阅器暂不直采");
  }
}

/** 增量扫描 + AI 简述，写简述笔记到 feedInboxFolder，返回写入条数 */
export async function scanFeeds(plugin: ArkOSPlugin): Promise<{ written: number; skipped: number; feedsDone: number; feedsFailed: number }> {
  const s = plugin.data.settings;
  const feeds = (s.feeds || []).filter((f) => f.enabled && f.target);
  if (feeds.length === 0) throw new Error("尚未配置启用的订阅源，请在设置中心 → 基础 → 信息订阅 添加");

  const seen = new Set(s.feedSeen || []);
  const stat = { written: 0, skipped: 0, feedsDone: 0, feedsFailed: 0 };
  const today = getToday();

  for (const feed of feeds) {
    let items: FeedItem[];
    try {
      items = await fetchFeedItems(plugin, feed);
    } catch (e: any) {
      feed.lastError = String(e?.message ?? e);
      feed.lastScan = Date.now();
      stat.feedsFailed++;
      continue;
    }

    // 增量去重：只在未见过指纹里留新的（每源限 15 条，控成本）
    // 注：url 单条源链接恒定，若走全局去重会永远"已见"导致无新增，故 url 源不过滤，靠写入阶段按"当天文件已含"去重。
    const fresh: FeedItem[] = [];
    for (const it of items) {
      const fp = feedFingerprint(it.url);
      if (feed.source !== "url" && fp && seen.has(fp)) continue;
      if (fp) seen.add(fp);
      fresh.push(it);
      if (fresh.length >= 15) break;
    }
    feed.lastScan = Date.now();
    feed.lastError = "";
    stat.feedsDone++;

    if (fresh.length === 0) { stat.skipped++; continue; }

    // AI 简述
    const itemBlock = fresh.map((it, i) => `[${i}] ${it.title}\n${it.url}\n${it.content}`).join("\n\n");
    const keywords = (feed.keywords || []).join("、") || "（无）";
    const prompt = topicNote.replace("{{keywords}}", keywords).replace("{{items}}", itemBlock);

    let briefs: Brief[];
    let raw = "";
    try {
      const ac = new AbortController();
      const timer = setTimeout(() => ac.abort(), 180000);
      try {
        raw = await runChannel(s, [{ role: "user", content: prompt }] as AiMessage[], "contextual", ac.signal);
        briefs = extractJson(raw);
      } finally {
        clearTimeout(timer);
      }
    } catch (e: any) {
      feed.lastError = String(e?.message ?? e);
      stat.feedsFailed++;
      if (e instanceof JsonExtractError) {
        await logAiError(plugin, "feed-brief", prompt, raw || (e as any).raw || "", e);
      }
      continue;
    }
    if (!Array.isArray(briefs)) {
      const e = new JsonExtractError("AI 返回的不是数组");
      await logAiError(plugin, "feed-brief", prompt, raw, e);
      throw e;
    }

    const blocks: string[] = [];
    // 简报条数封顶=输入条目数：prompt 契约是「每元素对应一条资讯」，防止 AI 对单条 url 脑补成一整份清单
    for (const b of briefs.filter((x) => x && x.title && x.conclusion).slice(0, fresh.length)) {
      if (feedFingerprint(b.url || "") && !seen.has(feedFingerprint(b.url || ""))) seen.add(feedFingerprint(b.url || ""));
      const points = Array.isArray(b.points) ? b.points.filter((p) => typeof p === "string").slice(0, 5) : [];
      const block = [
        `## ${b.title}`,
        "",
        `> 一句话结论：${b.conclusion}`,
        "",
        ...points.map((p) => `- ${p}`),
        "",
        `🔗 原文：${b.url}`,
        `标签：#${(Array.isArray(b.tags) ? b.tags : []).slice(0, 3).join(" #")}`,
        "关联：[[待整理/…]]",
      ].join("\n");
      blocks.push(block);
    }
    if (blocks.length > 0) {
      const written = await appendFeedNote(plugin, s.feedInboxFolder, feed.name, today, {
        type: "feed-brief",
        feed: feed.name,
        source: feed.source,
        source_ref: feed.target || feed.name,
        retry_count: "0",
        ...capturePatch("captured"),
        date: today,
        created_at: new Date().toISOString(),
      }, blocks);
      stat.written += written;
    } else {
      stat.skipped++;
    }
  }

  s.feedSeen = [...seen].slice(-2000);
  await plugin.savePluginData();
  return stat;
}

/** 追加简报到「一天一源一个文件」，不存在则创建，存在则追加新简报到末尾 */
async function appendFeedNote(
  plugin: ArkOSPlugin,
  folder: string,
  feedName: string,
  today: string,
  frontmatter: Record<string, unknown>,
  newBlocks: string[]
): Promise<number> {
  const filename = `${today}-${sanitizeFilename(feedName)}`;
  const path = `${folder}/${filename}.md`;
  const existing = plugin.app.vault.getAbstractFileByPath(path);

  let fullBody: string;
  if (existing instanceof TFile) {
    // 文件已存在：读取原有内容，去掉末尾空行，追加新简报块
    const oldContent = await plugin.app.vault.read(existing);
    // 按原文链接指纹做「当天文件内」去重，避免同一天重复抓同一条链接产生重复块
    const existingFps = new Set<string>();
    for (const m of oldContent.matchAll(/🔗\s*原文：\s*(\S+)/g)) {
      const fp = feedFingerprint(m[1]);
      if (fp) existingFps.add(fp);
    }
    const kept = newBlocks.filter((b) => {
      const m = b.match(/🔗\s*原文：\s*(\S+)/);
      if (!m) return true;
      const fp = feedFingerprint(m[1]);
      return !(fp && existingFps.has(fp));
    });
    if (kept.length === 0) { await plugin.savePluginData(); return 0; }
    const trimmed = oldContent.trimEnd();
    const hasHr = /---\s*$/.test(trimmed);
    fullBody = trimmed + (hasHr ? "\n\n" : "\n\n---\n\n") + kept.join("\n\n") + "\n";
    await plugin.app.vault.modify(existing, fullBody);
    return kept.length;
  } else {
    // 文件不存在：新建，带完整标题和 frontmatter 格式
    await ensureFolder(plugin, folder);
    fullBody = `# 📬 ${feedName} · ${today}\n\n` + newBlocks.join("\n\n") + "\n";
    const frontmatterStr = Object.entries(frontmatter)
      .filter(([, v]) => v !== undefined)
      .map(([k, v]) => `${k}: ${serializeFmValue(v)}`)
      .join("\n");
    const md = `---\n${frontmatterStr}\n---\n\n${fullBody}`;
    const file = await plugin.app.vault.create(path, md);
    await importMarkdownAt(plugin, file);
  }

  await plugin.savePluginData();
  return newBlocks.length;
}

/** 序列化单个 frontmatter 值（复用 sync.ts 逻辑，这里局部拷贝避免循环依赖） */
function serializeFmValue(v: unknown): string {
  if (v == null) return "";
  if (typeof v === "string") {
    // 含空格/冒号/换行会 yaml 解析错，加引号
    if (/[:\s"']/.test(v)) return `"${v.replace(/"/g, '\\"')}"`;
    return v;
  }
  if (Array.isArray(v)) {
    return `[${v.map((i) => serializeFmValue(i)).join(", ")}]`;
  }
  return String(v);
}

/** 确保目录存在（局部拷贝避免循环依赖） */
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

/** 导入文件到插件 data 同步（局部拷贝避免循环依赖） */
async function importMarkdownAt(plugin: ArkOSPlugin, file: TFile) {
  // 触发 sync.ts 的监听钩子自然导入，这里仅占位保持接口一致
  plugin.app.vault.trigger("vault-modify", file);
}

/** 特例：手动拉取一个订阅源（供按钮），返回写入条数 */
export async function scanFeedNow(plugin: ArkOSPlugin, feedId: string): Promise<number> {
  const s = plugin.data.settings;
  const feed = (s.feeds || []).find((f) => f.id === feedId);
  if (!feed) return 0;
  const seen = new Set(s.feedSeen || []);
  const today = getToday();
  let items: FeedItem[];
  try {
    items = await fetchFeedItems(plugin, feed);
  } catch (e: any) {
    new Notice(`订阅抓取失败: ${String((e as any)?.message ?? e)}`, 4000);
    return 0;
  }
  // url 源链接恒定，不做全局去重；其余聚合源按全局指纹过滤
  const fresh = items.filter((it) => feed.source === "url" || !seen.has(feedFingerprint(it.url)));
  if (fresh.length === 0) { new Notice("该源无新内容"); return 0; }

  const itemBlock = fresh.slice(0, 15).map((it, i) => `[${i}] ${it.title}\n${it.url}\n${it.content}`).join("\n\n");
  const keywords = (feed.keywords || []).join("、") || "（无）";
  const prompt = topicNote.replace("{{keywords}}", keywords).replace("{{items}}", itemBlock);
  let raw = "";
  let briefs: Brief[];
  try {
    raw = await runChannel(s, [{ role: "user", content: prompt }] as AiMessage[], "contextual", new AbortController().signal);
    briefs = extractJson(raw);
  } catch (e: any) {
    if (e instanceof JsonExtractError) {
      await logAiError(plugin, "feed-brief", prompt, raw || (e as any).raw || "", e);
    }
    throw e;
  }
  if (!Array.isArray(briefs)) {
    const e = new JsonExtractError("AI 返回的不是数组");
    await logAiError(plugin, "feed-brief", prompt, raw, e);
    throw e;
  }

  const blocks: string[] = [];
  for (const b of briefs.filter((x) => x && x.title).slice(0, fresh.length)) {
    const url = String(b.url || "");
    if (url && !seen.has(feedFingerprint(url))) seen.add(feedFingerprint(url));
    const points = Array.isArray(b.points) ? (b.points.filter((p: unknown) => typeof p === "string") as string[]).slice(0, 5) : [];
    const block = [
      `## ${b.title}`,
      "",
      `> 一句话结论：${b.conclusion ?? ""}`,
      "",
      ...points.map((p) => `- ${p}`),
      "",
      `🔗 原文：${url}`,
    ].join("\n");
    blocks.push(block);
  }
  let idx = 0;
  if (blocks.length > 0) {
    idx = await appendFeedNote(plugin, s.feedInboxFolder, feed.name, today, {
      type: "feed-brief", feed: feed.name, source: feed.source, date: today, created_at: new Date().toISOString(),
    }, blocks);
  }
  s.feedSeen = [...seen].slice(-2000);
  await plugin.savePluginData();
  return idx;
}
