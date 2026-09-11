// 重复规则编辑器（Modal）：返回 Recurrence | undefined（无重复）
import { Modal, Setting } from "obsidian";
import type ArkOSPlugin from "./main";
import type { Recurrence, Frequency } from "./types";

const FREQ_LABEL: Record<Frequency, string> = {
  none: "不重复", daily: "每天", weekly: "每周", monthly: "每月", yearly: "每年",
};

export function recurrenceEditor(plugin: ArkOSPlugin, current?: Recurrence): Promise<Recurrence | undefined> {
  return new Promise((resolve) => {
    const m = new Modal(plugin.app);
    m.titleEl.setText("重复规则");
    const body = m.contentEl;

    let freq: Frequency = current?.frequency ?? "none";
    const daysOfWeek = new Set<number>(current?.daysOfWeek ?? [1]);
    const daysOfMonth = new Set<number>(current?.daysOfMonth ?? [1]);
    let month = current?.month ?? 1;
    let day = current?.day ?? 1;
    let endMode: "never" | "count" | "until" = current?.count ? "count" : current?.until ? "until" : "never";
    let count = current?.count ?? 1;
    let until = current?.until ?? "";

    const freqSetting = new Setting(body).setName("频率").addDropdown((dd) => {
      (Object.keys(FREQ_LABEL) as Frequency[]).forEach((f) => dd.addOption(f, FREQ_LABEL[f]));
      dd.setValue(freq).onChange((v) => { freq = v as Frequency; renderFreq(); });
    });
    const extra = body.createDiv();
    const endSetting = new Setting(body).setName("结束").addDropdown((dd) => {
      dd.addOption("never", "从不").addOption("count", "限定次数").addOption("until", "直到日期");
      dd.setValue(endMode).onChange((v) => { endMode = v as typeof endMode; renderEnd(); });
    });
    const endExtra = body.createDiv();

    function renderFreq() {
      extra.empty();
      if (freq === "weekly") {
        for (let d = 0; d <= 6; d++) {
          const day0 = new Date(2026, 0, 4 + d).toLocaleDateString("zh-CN", { weekday: "short" });
          const cb = extra.createEl("label", { cls: "sos-chk", text: ` ${day0} ` });
          const inp = cb.createEl("input", { attr: { type: "checkbox" } });
          inp.checked = daysOfWeek.has(d);
          inp.addEventListener("change", () => { if (inp.checked) daysOfWeek.add(d); else daysOfWeek.delete(d); });
        }
      } else if (freq === "monthly") {
        new Setting(extra).setName("每月几日（逗号分隔）").addText((t) => {
          t.setValue([...daysOfMonth].join(","));
          t.onChange((v) => { daysOfMonth.clear(); v.split(/[,，]/).forEach((s) => { const n = Number(s); if (n >= 1 && n <= 31) daysOfMonth.add(n); }); });
        });
      } else if (freq === "yearly") {
        new Setting(extra).setName("月份 (1-12)").addText((t) => { t.setValue(String(month)); t.onChange((v) => { month = Number(v) || 1; }); });
        new Setting(extra).setName("日期 (1-31)").addText((t) => { t.setValue(String(day)); t.onChange((v) => { day = Number(v) || 1; }); });
      }
    }
    function renderEnd() {
      endExtra.empty();
      if (endMode === "count") {
        new Setting(endExtra).setName("重复次数").addText((t) => { t.setValue(String(count)); t.onChange((v) => { count = Number(v) || 1; }); });
      } else if (endMode === "until") {
        new Setting(endExtra).setName("截至日期 (YYYY-MM-DD)").addText((t) => { t.setValue(until); t.onChange((v) => { until = v; }); });
      }
    }
    renderFreq();
    renderEnd();

    new Setting(body)
      .addButton((b) => b.setButtonText("取消").onClick(() => { m.close(); resolve(undefined); }))
      .addButton((b) => b.setButtonText("确定").setCta().onClick(() => {
        m.close();
        if (freq === "none") { resolve(undefined); return; }
        const rec: Recurrence = { frequency: freq };
        if (freq === "weekly" && daysOfWeek.size) rec.daysOfWeek = [...daysOfWeek].sort((a, b) => a - b);
        if (freq === "monthly" && daysOfMonth.size) rec.daysOfMonth = [...daysOfMonth].sort((a, b) => a - b);
        if (freq === "yearly") { rec.month = month; rec.day = day; }
        if (endMode === "count") rec.count = count;
        if (endMode === "until" && until) rec.until = until;
        resolve(rec);
      }));
    m.open();
  });
}

export { FREQ_LABEL };