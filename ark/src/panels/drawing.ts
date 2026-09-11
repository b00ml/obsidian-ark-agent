import type { SpaceOSView } from "../view";
import { createCard, updateCard, deleteCardAt } from "../cards";
import { inputDialog, confirmDialog } from "../ui";
import { getSkin } from "../skins";
import { sendSelectionToWorkbench } from "../agent-workbench";

interface Drawing {
  id: string;
  title: string;
  description?: string;
  color: string;
  createdAt: number;
  updatedAt: number;
  notePath?: string;
}

/** 创作草稿：兼容旧画板数据；真正有价值的动作是把草稿交给 Agent 形成可检索笔记。 */
export function renderDrawing(view: SpaceOSView, mount: HTMLElement) {
  mount.addClass("space-drawing");
  const data = view.plugin.data;
  const drawings = data.drawings as Drawing[];

  const toolbar = mount.createDiv({ cls: "log-toolbar" });
  toolbar.createDiv({ cls: "tactical-title", text: getSkin(data.settings.skin).tabs.drawing.title });
  toolbar.createEl("button", { cls: "log-new-btn", text: "+ 新建画板" }).addEventListener("click", async () => {
    const title = await inputDialog(view.plugin, { title: "新建画板", placeholder: "画板标题" });
    if (!title || !title.trim()) return;
    await createCard(view.plugin, "drawing", { fm: { title: title.trim(), color: randomColor(), description: "" }, body: "" });
    redraw();
  });

  const grid = mount.createDiv({ cls: "drawing-grid" });

  function redraw() {
    grid.empty();
    const list = [...drawings].sort((a, b) => b.updatedAt - a.updatedAt);
    if (list.length === 0) {
      grid.createDiv({ cls: "tactical-empty", text: "暂无画板，点击上方新建" });
      return;
    }
    list.forEach((dr) => {
      const card = grid.createDiv({ cls: "drawing-card glass" });
      // 缩略画布：色块 + 标题首字
      const canvas = card.createDiv({ cls: "drawing-canvas", attr: { style: `background:${dr.color}` } });
      canvas.setText(dr.title.charAt(0) || "·");
      const info = card.createDiv({ cls: "drawing-card-info" });
      info.createDiv({ cls: "drawing-title", text: dr.title });
      if (dr.description) info.createDiv({ cls: "db-desc", text: dr.description });
      info.createDiv({ cls: "log-entry-time", text: new Date(dr.createdAt).toLocaleDateString() });
      const act = card.createDiv({ cls: "tactical-card-actions" });
      const agent = act.createEl("button", { cls: "tactical-mini-btn", text: "提炼为笔记" });
      agent.title = "让 Agent 将创作草稿整理为结构化 Markdown 笔记";
      agent.addEventListener("click", () => sendSelectionToWorkbench(view.plugin,
        `请把这份创作草稿整理成可检索的 Markdown 笔记，给出标题、摘要、要点和后续行动；不要丢失来源。\n来源：[[${dr.notePath ?? dr.title}]]\n标题：${dr.title}\n\n${dr.description || "（草稿暂无描述）"}`));
      const edit = act.createEl("button", { cls: "tactical-mini-btn", text: "描述" });
      edit.addEventListener("click", async () => {
        const desc = await inputDialog(view.plugin, { title: "画板描述", multiline: true, initial: dr.description ?? "" });
        if (desc === null) return;
        await updateCard(view.plugin, dr.notePath, {
          fm: { title: dr.title, color: dr.color, created_at: new Date(dr.createdAt).toISOString() },
          body: desc ?? "",
        });
        redraw();
      });
      const del = act.createEl("button", { cls: "tactical-mini-btn", text: "✕" });
      del.addEventListener("click", async () => {
        if (!(await confirmDialog(view.plugin, "删除该画板？"))) return;
        await deleteCardAt(view.plugin, dr.notePath);
        redraw();
      });
    });
  }

  redraw();
}

const DRAW_COLORS = ["#00e5ff", "#b34dff", "#00ff99", "#ffcc00", "#ff6699", "#ff8800", "#0066ff"];
function randomColor(): string {
  return DRAW_COLORS[Math.floor(Math.random() * DRAW_COLORS.length)];
}
