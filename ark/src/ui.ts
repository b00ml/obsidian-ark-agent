// Obsidian 原生 UI 替代 window.prompt / alert / confirm（后者在插件中不可用）
import { Modal, Setting, Notice } from "obsidian";
import type ArkOSPlugin from "./main";

export function notice(msg: string, timeout?: number): void {
  new Notice(msg, timeout ?? 3000);
}

export function confirmDialog(plugin: ArkOSPlugin, message: string): Promise<boolean> {
  return new Promise((resolve) => {
    const modal = new Modal(plugin.app);
    modal.titleEl.setText("确认操作");
    modal.contentEl.createDiv({ cls: "sos-hint", text: message });
    new Setting(modal.contentEl)
      .addButton((b) => b.setButtonText("取消").onClick(() => { modal.close(); resolve(false); }))
      .addButton((b) => b.setButtonText("确定").setCta().onClick(() => { modal.close(); resolve(true); }));
    modal.open();
  });
}

export interface InputOptions {
  title: string;
  placeholder?: string;
  multiline?: boolean;
  initial?: string;
}

export async function inputDialog(plugin: ArkOSPlugin, opts: InputOptions): Promise<string | null> {
  return new Promise((resolve) => {
    let val = opts.initial ?? "";
    const modal = new Modal(plugin.app);
    modal.titleEl.setText(opts.title);

    if (opts.multiline) {
      const ta = modal.contentEl.createEl("textarea", { attr: { rows: "6" } });
      ta.value = opts.initial ?? "";
      ta.style.width = "100%";
      ta.addEventListener("input", () => { val = ta.value; });
      new Setting(modal.contentEl)
        .addButton((b) => b.setButtonText("取消").onClick(() => { modal.close(); resolve(null); }))
        .addButton((b) => b.setButtonText("确定").setCta().onClick(() => { modal.close(); resolve(val); }));
    } else {
      new Setting(modal.contentEl)
        .setName(opts.placeholder ?? "")
        .addText((t) => {
          t.setValue(opts.initial ?? "");
          val = t.getValue();
          t.onChange((v) => { val = v; });
          t.inputEl.addEventListener("keydown", (e) => {
            if (e.key === "Enter") { modal.close(); resolve(val); }
          });
        })
        .addButton((b) => b.setButtonText("取消").onClick(() => { modal.close(); resolve(null); }))
        .addButton((b) => b.setButtonText("确定").setCta().onClick(() => { modal.close(); resolve(val); }));
    }
    modal.open();
  });
}
/** 写入冲突对话框：提供"重试（覆盖）"、"另存为新会话"、"取消"三个选项 */
export function conflictDialog(plugin: ArkOSPlugin, message: string): Promise<"retry" | "save-as" | "cancel"> {
  return new Promise((resolve) => {
    const modal = new Modal(plugin.app);
    modal.titleEl.setText("写入冲突");
    modal.contentEl.createDiv({ cls: "sos-hint", text: message });
    modal.contentEl.createDiv({ 
      cls: "sos-hint", 
      text: "可能的原因：其他窗口正在修改此会话，或并发执行了多个 Agent 任务。" 
    });
    new Setting(modal.contentEl)
      .addButton((b) => b.setButtonText("取消").onClick(() => { modal.close(); resolve("cancel"); }))
      .addButton((b) => b.setButtonText("另存为新会话").onClick(() => { modal.close(); resolve("save-as"); }))
      .addButton((b) => b.setButtonText("重试（覆盖）").setCta().onClick(() => { modal.close(); resolve("retry"); }));
    modal.open();
  });
}
