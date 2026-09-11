import { TFile } from "obsidian";
import type { SpaceOSView } from "../view";
import { captureFields, type CaptureState } from "../capture";
import { notice } from "../ui";
import { sendSelectionToWorkbench } from "../agent-workbench";

const STATES: CaptureState[] = ["captured", "processing", "needs_review", "approved", "archived", "linked", "failed"];
const LABELS: Record<CaptureState, string> = {
  captured: "已捕获", processing: "处理中", needs_review: "待审核", approved: "已批准",
  archived: "已归档", linked: "已入知识", failed: "失败",
};

/** F5-021：统一 Capture/Inbox 投影视图。真源仍是 Vault 文件，data 只作缓存。 */
export function renderCapture(view: SpaceOSView, mount: HTMLElement) {
  mount.addClass("ark-capture-host");
  mount.createDiv({ cls: "capture-note", text: "唯一未加工输入入口：灵感、邮件、订阅、视频和剪藏统一进入状态机；Agent 只提出提炼建议，原文和来源引用保持不变。" });
  const heading = mount.createDiv({ cls: "capture-heading" });
  heading.createDiv({ cls: "tactical-title", text: "采集暂存" });
  heading.createDiv({ cls: "tactical-subtitle", text: "先筛选和批处理，再决定哪些内容进入维度知识库。" });
  const files = view.plugin.app.vault.getMarkdownFiles().filter((f) => {
    const fm = view.plugin.app.metadataCache.getFileCache(f)?.frontmatter as Record<string, any> | undefined;
    const isIdea = f.path.startsWith(view.plugin.data.settings.ideaFolder + "/");
    return !!fm && (isIdea || String(fm.type || "").includes("feed-brief") || fm.state || fm.status === "archived" || fm.processed != null);
  });
  const fieldsFor = (file: TFile) => captureFields(view.plugin.app.metadataCache.getFileCache(file)?.frontmatter || {});
  const stats = mount.createDiv({ cls: "capture-stats" });
  const stateCounts = new Map<CaptureState, number>();
  files.forEach((file) => {
    const state = fieldsFor(file).state;
    stateCounts.set(state, (stateCounts.get(state) ?? 0) + 1);
  });
  ([("captured" as CaptureState), ("processing" as CaptureState), ("needs_review" as CaptureState), ("linked" as CaptureState), ("failed" as CaptureState)]).forEach((state) => {
    stats.createSpan({ cls: `capture-stat ${state}`, text: `${LABELS[state]} ${stateCounts.get(state) ?? 0}` });
  });
  const toolbar = mount.createDiv({ cls: "ark-capture-toolbar" });
  const select = toolbar.createEl("select", { cls: "ark-capture-filter" });
  select.createEl("option", { value: "all", text: "全部状态" });
  STATES.forEach((s) => select.createEl("option", { value: s, text: LABELS[s] }));
  const sources = [...new Set(files.map((file) => fieldsFor(file).source || "manual"))].sort();
  const sourceSelect = toolbar.createEl("select", { cls: "ark-capture-filter", attr: { "aria-label": "来源筛选" } });
  sourceSelect.createEl("option", { value: "all", text: "全部来源" });
  sources.forEach((source) => sourceSelect.createEl("option", { value: source, text: source }));
  const selected = new Set<string>();
  const selectVisible = toolbar.createEl("button", { cls: "ark-capture-batch", text: "选择当前" });
  const batch = toolbar.createEl("button", { cls: "ark-capture-batch primary", text: "批量交给 Agent" });
  batch.disabled = true;
  const list = mount.createDiv({ cls: "ark-capture-list" });

  const updateBatch = () => {
    batch.disabled = selected.size === 0;
    batch.setText(selected.size ? `批量交给 Agent · ${selected.size}` : "批量交给 Agent");
  };

  selectVisible.addEventListener("click", () => {
    const wanted = select.value;
    const visible = files.filter((file) => {
      const fields = fieldsFor(file);
      return (wanted === "all" || fields.state === wanted) && (sourceSelect.value === "all" || fields.source === sourceSelect.value);
    });
    const allSelected = visible.length > 0 && visible.every((file) => selected.has(file.path));
    visible.forEach((file) => allSelected ? selected.delete(file.path) : selected.add(file.path));
    render();
  });

  batch.addEventListener("click", () => {
    const chosen = files.filter((file) => selected.has(file.path)).slice(0, 20);
    if (!chosen.length) return;
    const refs = chosen.map((file) => `- [[${file.path}]]（${file.basename}）`).join("\n");
    const suffix = selected.size > chosen.length ? `\n其余 ${selected.size - chosen.length} 条未注入，请分批处理。` : "";
    sendSelectionToWorkbench(view.plugin,
      `请批量处理以下采集项：提取可复用要点、补充标签、保留原始来源，并逐条给出建议状态（needs_review/linked）。不要直接覆盖原文。\n${refs}${suffix}`);
  });

  const render = () => {
    list.empty();
    const wanted = select.value;
    const rows = files
      .map((file) => ({ file, fields: fieldsFor(file) }))
      .filter((x) => wanted === "all" || x.fields.state === wanted)
      .filter((x) => sourceSelect.value === "all" || x.fields.source === sourceSelect.value)
      .sort((a, b) => b.file.stat.mtime - a.file.stat.mtime);
    if (!rows.length) {
      list.createDiv({ cls: "ark-capture-empty", text: "暂无符合条件的捕获" });
      updateBatch();
      return;
    }
    rows.forEach(({ file, fields }) => {
      const row = list.createDiv({ cls: `ark-capture-row state-${fields.state}` });
      const checkbox = row.createEl("input", { cls: "ark-capture-check", attr: { type: "checkbox" } }) as HTMLInputElement;
      checkbox.checked = selected.has(file.path);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) selected.add(file.path);
        else selected.delete(file.path);
        updateBatch();
      });
      const main = row.createDiv({ cls: "ark-capture-main" });
      main.createDiv({ cls: "ark-capture-title", text: file.basename });
      main.createDiv({ cls: "ark-capture-meta", text: `${LABELS[fields.state]} · ${fields.source}${fields.retryCount ? ` · 重试 ${fields.retryCount}` : ""}` });
      const actions = row.createDiv({ cls: "ark-capture-actions" });
      actions.createEl("button", { text: "打开" }).addEventListener("click", () => {
        void view.plugin.app.workspace.getLeaf("tab").openFile(file);
      });
      if (fields.state === "failed") {
        actions.createEl("button", { text: "重试" }).addEventListener("click", async () => {
          await resetForRetry(view, file);
          notice(`已重置：${file.basename}，可重新运行处理简报`);
          view.renderCurrentPanel();
        });
      }
      if (fields.state !== "linked") {
        actions.createEl("button", { text: "交给 Agent" }).addEventListener("click", () => {
          const text = `请处理这条采集项并判断是否进入知识库。保留来源引用，不要直接覆盖原文。\n来源：[[${file.path}]]\n标题：${file.basename}`;
          sendSelectionToWorkbench(view.plugin, text);
        });
      }
    });
    updateBatch();
  };
  select.addEventListener("change", render);
  sourceSelect.addEventListener("change", render);
  const legend = mount.createDiv({ cls: "capture-state-legend" });
  STATES.forEach((state, index) => {
    if (index) legend.createSpan({ cls: "capture-state-arrow", text: "→" });
    legend.createSpan({ cls: `capture-state-chip ${state}`, text: LABELS[state] });
    if (state === "failed") legend.createSpan({ cls: "capture-state-hint", text: "可重试" });
  });
  render();
}

async function resetForRetry(view: SpaceOSView, file: TFile): Promise<void> {
  const text = await view.plugin.app.vault.read(file);
  const m = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(text);
  if (!m) return;
  const lines = m[1].split("\n");
  const set = (key: string, value: string) => {
    const re = new RegExp(`^\\s*${key}:`);
    const i = lines.findIndex((line) => re.test(line));
    if (i >= 0) lines[i] = `${key}: ${value}`;
    else lines.push(`${key}: ${value}`);
  };
  set("state", "archived");
  set("error_code", "");
  await view.plugin.app.vault.modify(file, `---\n${lines.join("\n")}\n---\n\n${text.slice(m[0].length)}`);
}
