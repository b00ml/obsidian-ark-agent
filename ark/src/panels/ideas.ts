import { Notice } from "obsidian";
import type { SpaceOSView } from "../view";
import { createCard, updateCard, deleteCardAt } from "../cards";
import { inputDialog, confirmDialog } from "../ui";
import { generateVariant, previewDiffModal, pickVariantMode, copyText } from "../ai-asst";
import { sendSelectionToWorkbench } from "../agent-workbench";

interface Idea {
  id: string;
  title: string;
  content: string;
  tags: string[];
  archived: boolean;
  createdAt: number;
  updatedAt: number;
  notePath?: string;
}

/** 捕获草稿：保留旧 ideas 数据格式，但每条草稿都能直接交给 Agent 归纳。 */
export function renderIdeas(view: SpaceOSView, mount: HTMLElement) {
  mount.addClass("space-ideas");
  const data = view.plugin.data;
  const ideas = data.ideas as Idea[];
  let showArchived = false;

  const dash = mount.createDiv({ cls: "log-dashboard" });
  const active = ideas.filter((i) => !i.archived);
  const archivedN = ideas.length - active.length;
  dash.createEl("span", { cls: "log-stat", text: `灵感 ${active.length}` });
  dash.createEl("span", { cls: "log-stat", text: `已归档 ${archivedN}` });

  const toolbar = mount.createDiv({ cls: "log-toolbar" });
  const archiveBtn = toolbar.createEl("button", { cls: "log-type-tab", text: showArchived ? "隐藏归档" : "显示归档" });
  archiveBtn.addEventListener("click", () => { showArchived = !showArchived; archiveBtn.setText(showArchived ? "隐藏归档" : "显示归档"); redraw(); });
  toolbar.createEl("button", { cls: "log-new-btn", text: "+ 新增灵感" }).addEventListener("click", () => newIdea());

  const grid = mount.createDiv({ cls: "idea-gallery" });

  function redraw() {
    grid.empty();
    const list = ideas
      .filter((i) => showArchived || !i.archived)
      .sort((a, b) => b.updatedAt - a.updatedAt);
    if (list.length === 0) {
      grid.createDiv({ cls: "tactical-empty", text: "暂无灵感，点击上方新增" });
      return;
    }
    list.forEach((i) => {
      const card = grid.createDiv({ cls: "idea-card glass" });
      const head = card.createDiv({ cls: "idea-head" });
      head.createSpan({ cls: "idea-title", text: i.title });
      if (i.archived) head.createSpan({ cls: "idea-archived", text: "已归档" });
      if (i.content) card.createDiv({ cls: "idea-content", text: i.content });
      if (i.tags.length) card.createDiv({ cls: "idea-tags", text: i.tags.map((t) => `#${t}`).join(" ") });
      const act = card.createDiv({ cls: "tactical-card-actions" });
      const agent = act.createEl("button", { cls: "tactical-mini-btn", text: "交给 Agent" });
      agent.title = "让 Agent 提炼要点、判断是否值得沉淀到知识库";
      agent.addEventListener("click", () => sendSelectionToWorkbench(view.plugin,
        `请处理这条捕获草稿：提炼核心观点、补充缺失上下文，并判断它是否应该进入知识库。保留来源引用 [[${i.notePath ?? i.title}]]。\n\n标题：${i.title}\n标签：${i.tags.join(", ")}\n\n${i.content || "（正文为空，请先提出澄清问题）"}`));
      const ai = act.createEl("button", { cls: "tactical-mini-btn", text: "AI 优化" });
      ai.title = "润色 / 扩写 / 提炼要点，预览后回写卡片正文";
      ai.addEventListener("click", () => void optimizeIdea(view, i));
      const edit = act.createEl("button", { cls: "tactical-mini-btn", text: "编辑" });
      edit.addEventListener("click", () => editIdea(i));
      const arc = act.createEl("button", { cls: "tactical-mini-btn", text: i.archived ? "恢复" : "归档" });
      arc.addEventListener("click", async () => {
        await saveIdea(view, i, { archived: !i.archived });
        redraw();
      });
      const del = act.createEl("button", { cls: "tactical-mini-btn", text: "✕" });
      del.addEventListener("click", async () => {
        if (!(await confirmDialog(view.plugin, "删除该灵感？"))) return;
        await deleteCardAt(view.plugin, i.notePath);
        redraw();
      });
    });
  }

  async function newIdea() {
    const title = await inputDialog(view.plugin, { title: "新增灵感", placeholder: "灵感标题" });
    if (!title || !title.trim()) return;
    const content = (await inputDialog(view.plugin, { title: "内容（可选）", multiline: true })) ?? "";
    const tagsRaw = (await inputDialog(view.plugin, { title: "标签（逗号分隔）", placeholder: "可选" })) ?? "";
    const tags = tagsRaw.split(/[,，]/).map((t) => t.trim()).filter(Boolean);
    try {
      await createCard(view.plugin, "idea", { fm: { title: title.trim(), tags, archived: false }, body: content });
      redraw();
    } catch (e: any) {
      new Notice("灵感新建失败: " + (e?.message ?? e));
    }
  }

  async function editIdea(i: Idea) {
    const title = await inputDialog(view.plugin, { title: "编辑标题", initial: i.title });
    if (title === null) return;
    const content = (await inputDialog(view.plugin, { title: "编辑内容", multiline: true, initial: i.content })) ?? "";
    i.title = title.trim() || i.title;
    await saveIdea(view, i, { content });
    redraw();
  }
}

async function saveIdea(view: SpaceOSView, i: Idea, patch: { archived?: boolean; content?: string }) {
  const archived = patch.archived ?? i.archived;
  const content = patch.content ?? i.content;
  const fm = { title: i.title, tags: i.tags, archived, created_at: new Date(i.createdAt).toISOString() };
  await updateCard(view.plugin, i.notePath, { fm, body: content });
}

/** F3 AI 优化：选模式 → 生成变体 → 预览 → 回写卡片正文或复制（1:1 经 cards.updateCard） */
async function optimizeIdea(view: SpaceOSView, idea: Idea) {
  if (!idea.content || !idea.content.trim()) {
    new Notice("该灵感无正文，无法优化");
    return;
  }
  const mode = await pickVariantMode(view.plugin, ["润色", "扩写", "提炼要点"]);
  if (!mode) return;
  const loading = new Notice("AI 优化中…", 0);
  let out: string;
  try {
    out = await generateVariant(view.plugin, idea.content, mode);
  } catch (e: any) {
    loading.hide();
    new Notice("AI 优化失败: " + (e?.message ?? e), 4000);
    return;
  }
  loading.hide();
  if (!out) return;

  const action = await previewDiffModal(view.plugin, idea.content, out);
  if (!action) return;
  if (action === "copy") {
    await copyText(view.plugin, out);
    new Notice("已复制到剪贴板");
    return;
  }
  await saveIdea(view, idea, { content: out });
  new Notice("已应用替换");
}
