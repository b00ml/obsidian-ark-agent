// 学习问答（知识信息闭环 M1 / §3.3）：读今日学习笔记 → 出题 → 批改 → 反链。
// 复用 ai-asst 的 contextual 通道路由 + extractJson；笔记用 writeMarkdown 落库（普通笔记，非卡片，不破坏 1:1）。
import { Notice, TFile } from "obsidian";
import type ArkOSPlugin from "./main";
import type { AiMessage } from "./ai";
import { runChannel, extractJson, JsonExtractError, logAiError } from "./ai-asst";
import { writeMarkdown } from "./sync";
import { getToday, sanitizeFilename } from "./utils";
import quizUser from "./prompts/quiz-user.st";
import quizGraderUser from "./prompts/quiz-grader-user.st";

interface Question {
  type: string;
  difficulty: string;
  question: string;
  reference: string;
  answer: string;
}

interface GradedItem {
  index: number;
  verdict: string;
  reason: string;
  correct_answer: string;
  suggestion: string;
}

function localDate(): string {
  return getToday();
}

function isToday(ts: number): boolean {
  const d = new Date(ts);
  const now = new Date();
  return d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth() && d.getDate() === now.getDate();
}

/** learningFolder 下、今日修改的 .md 笔记（排除目录自身） */
function todayLearningNotes(plugin: ArkOSPlugin): TFile[] {
  const s = plugin.data.settings;
  const folder = s.learningFolder || "02-DB/学习";
  return plugin.app.vault.getFiles().filter(
    (f: TFile) => f.extension === "md" && f.path.startsWith(folder + "/") && isToday(f.stat.mtime),
  );
}

/** 读取今日问答文件 frontmatter 的来源列表 */
async function readQuizzedSources(plugin: ArkOSPlugin): Promise<string[]> {
  const s = plugin.data.settings;
  const path = `${s.quizFolder}/${localDate()}-学习问答.md`;
  const file = plugin.app.vault.getAbstractFileByPath(path);
  if (!(file instanceof TFile)) return [];
  const text = await plugin.app.vault.read(file);
  const m = /^---\r?\n([\s\S]*?)\r?\n---/.exec(text);
  if (!m) return [];
  const src = /(?:^|\n)\s*sources:\s*\[([^\]]*)\]/.exec(m[1])?.[1];
  if (!src) return [];
  return src.split(",").map((x) => x.trim().replace(/[\[\]"]/g, "")).filter(Boolean);
}

/** 出题：合并今日未出题笔记 → 生成 JSON 题目 → 写问答 md，返回写入路径 */
export async function generateQuiz(plugin: ArkOSPlugin): Promise<string> {
  const s = plugin.data.settings;
  const today = localDate();
  const quizzed = await readQuizzedSources(plugin);
  const candidates = todayLearningNotes(plugin).filter((f) => !quizzed.includes(f.basename));
  if (candidates.length === 0) {
    return "NOTHING";
  }

  // 拼近况上下文（标题 + 正文截断，控制 token）
  const parts: string[] = [];
  for (const f of candidates) {
    const body = (await plugin.app.vault.read(f)).trim();
    parts.push(`【${f.basename}】\n${body.slice(0, 1500)}`);
  }
  const ctx = parts.join("\n\n").slice(0, 8000);

  const count = s.quizQuestionCount || 5;
  const difficulty = s.quizDifficulty || "中等";
  const prompt = quizUser
    .replace("{{notes}}", ctx)
    .replace("{{count}}", String(Math.min(count, 10)))
    .replace("{{difficulty}}", difficulty);

  const loaded = new Notice("AI 出题中…", 0);
  let arr: Question[];
  let raw = "";
  try {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), 180000);
    try {
      raw = await runChannel(s, [{ role: "user", content: prompt }] as AiMessage[], "contextual", ac.signal);
      arr = extractJson(raw);
    } finally {
      clearTimeout(timer);
    }
  } catch (e: any) {
    loaded.hide();
    if (e instanceof JsonExtractError) {
      await logAiError(plugin, "quiz-questions", prompt, raw || (e as any).raw || "", e);
    }
    throw e;
  }
  loaded.hide();

  if (!Array.isArray(arr)) {
    const e = new JsonExtractError("AI 返回的不是数组");
    await logAiError(plugin, "quiz-questions", prompt, raw, e);
    throw e;
  }
  const clean = arr.filter((q) => q && typeof q.question === "string" && q.question.trim()).slice(0, count);

  // 拼问答 md：来源 + 逐题题干 + 留白作答区 + 参考答案统一收尾（作答前不展示，避免剧透）
  const sourceLinks = candidates.map((f) => `[[${f.basename}]]`).join(" ");
  const lines: string[] = [];
  lines.push(`# 学习问答 · ${today}`);
  lines.push("");
  lines.push(`来源笔记：${sourceLinks}`);
  lines.push("");
  lines.push("---");
  lines.push("");
  clean.forEach((q, i) => {
    lines.push(`### Q${i + 1} [${q.type ?? "理解"}/${q.difficulty ?? "medium"}]`);
    lines.push(`- 题干：${q.question}`);
    lines.push(`- 来源：[[${sanitizeFilename(String(q.reference ?? ""))}]]`);
    lines.push("> 作答：");
    lines.push("");
    lines.push("");
    lines.push("");
  });
  lines.push("---");
  lines.push("");
  lines.push("## 参考答案");
  lines.push("");
  clean.forEach((q, i) => {
    lines.push(`- **Q${i + 1}**：${q.answer ?? "（无）"}`);
  });

  const path = await writeMarkdown(plugin, s.quizFolder, `${today}-学习问答`, {
    type: "learning-quiz",
    date: today,
    sources: candidates.map((f) => f.basename),
    question_count: clean.length,
    status: "pending",
    generated_at: new Date().toISOString(),
  }, lines.join("\n"));
  return path;
}

/** 批改：读最新未批改的问答文件 + 源笔记 → 生成判定 → 另存 -批改，并标记原文件已批改 */
export async function gradeQuiz(plugin: ArkOSPlugin): Promise<string> {
  const s = plugin.data.settings;
  const quizDir = s.quizFolder || "02-DB/问答";

  const quizFiles = plugin.app.vault.getFiles()
    .filter((f: TFile) => f.extension === "md" && f.path.startsWith(quizDir + "/") && /-学习问答\.md$/.test(f.path))
    .sort((a, b) => b.stat.mtime - a.stat.mtime);
  if (quizFiles.length === 0) return "NONE";

  // 取第一个 status 仍为 pending 的文件（最新未批改）
  let target: TFile | null = null;
  for (const f of quizFiles) {
    const text = await plugin.app.vault.read(f);
    const m = /^---\r?\n([\s\S]*?)\r?\n---/.exec(text);
    const status = m && /status:\s*(\w+)/.exec(m[1])?.[1];
    if (status !== "graded") { target = f; break; }
  }
  if (!target) return "DONE";

  const quizText = await plugin.app.vault.read(target);
  const m = /^---\r?\n([\s\S]*?)\r?\n---/.exec(quizText);
  const fmSrc = m && /sources:\s*\[([^\]]*)\]/.exec(m[1])?.[1];
  const sources = fmSrc ? fmSrc.split(",").map((x) => x.trim().replace(/[\[\]"]/g, "")).filter(Boolean) : [];

  const body = m ? quizText.slice(m[0].length) : quizText;
  // 拆题块：### Q1 ... ### Q2 ...（只取题干区；末题作答在 `---\n## 参考答案` 前结束，不把参考答案误当作答）
  const blocks = body.split(/\n### Q/).slice(1);
  const qas: string[] = [];
  blocks.forEach((b, i) => {
    const ans = (b.match(/>\s*作答：\s*([\s\S]*?)(?=\n### |\n---|$)/)?.[1] ?? "").trim();
    qas.push(`Q${i + 1} ${ans ? ans : "（未作答）"}`);
  });

  // 源笔记原文
  const srcParts: string[] = [];
  for (const name of sources) {
    const file = plugin.app.vault.getFiles().find((f: TFile) => f.basename === name);
    if (file) srcParts.push(`【${name}】\n${(await plugin.app.vault.read(file)).slice(0, 1500)}`);
  }
  const srcCtx = srcParts.join("\n\n").slice(0, 8000);
  const ansCtx = qas.join("\n\n");

  const prompt = quizGraderUser.replace("{{notes}}", srcCtx).replace("{{answers}}", `${quizText}\n\n${ansCtx}`);
  const loaded = new Notice("AI 批改中…", 0);
  let arr: GradedItem[];
  let raw = "";
  try {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), 180000);
    try {
      raw = await runChannel(s, [{ role: "user", content: prompt }] as AiMessage[], "contextual", ac.signal);
      arr = extractJson(raw);
    } finally {
      clearTimeout(timer);
    }
  } catch (e: any) {
    loaded.hide();
    if (e instanceof JsonExtractError) {
      await logAiError(plugin, "quiz-graded", prompt, raw || (e as any).raw || "", e);
    }
    throw e;
  }
  loaded.hide();

  if (!Array.isArray(arr)) {
    const e = new JsonExtractError("AI 返回的不是数组");
    await logAiError(plugin, "quiz-graded", prompt, raw, e);
    throw e;
  }

  // 拼批改 md
  const today = localDate();
  const graded: string[] = [];
  graded.push(`# 批改 · ${today}`);
  graded.push("");
  graded.push(`原文：[[${target.basename}]]`);
  graded.push("");
  graded.push("---");
  graded.push("");
  const verdictLabel: Record<string, string> = { 对: "✅ 对", 偏: "🔄 偏", 错: "❌ 错" };
  arr.forEach((g) => {
    const i = Number(g.index) || graded.length;
    const v = verdictLabel[g.verdict] ?? g.verdict;
    graded.push(`### Q${i}`);
    graded.push(`**判定**：${v}`);
    if (g.reason) graded.push(`**原因**：${g.reason}`);
    if (g.correct_answer) graded.push(`**标准答案**：${g.correct_answer}`);
    if (g.suggestion) graded.push(`**建议**：${g.suggestion}`);
    graded.push("");
  });

  const gradedName = sanitizeFilename(target.basename.replace(/学习问答/, "学习问答-批改"));
  const path = await writeMarkdown(plugin, s.quizFolder, gradedName, {
    type: "learning-quiz-graded",
    date: today,
    original: `[[${target.basename}]]`,
    graded_at: new Date().toISOString(),
  }, graded.join("\n"));

  // 标记原文件 status → graded
  await updateFrontmatterStatus(plugin, target, "graded");
  return path;
}

/** 就地更新问答文件 frontmatter 的 status 字段 */
async function updateFrontmatterStatus(plugin: ArkOSPlugin, file: TFile, status: string) {
  const text = await plugin.app.vault.read(file);
  if (!/^---\r?\n/.test(text)) return;
  const replaced = text.replace(/(^---\r?\n)([\s\S]*?)(\r?\n---)/, (whole, head, body, tail) => {
    const next = body.replace(/(status:\s*)\w+/, `$1${status}`);
    return `${head}${next}${tail}`;
  });
  if (replaced !== text) await plugin.app.vault.modify(file, replaced);
}