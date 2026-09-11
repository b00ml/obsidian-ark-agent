import type { SpaceOSView } from "../view";
import { getToday } from "../utils";
import { createCard } from "../cards";

interface HealthRecord {
  id: string;
  date: string;
  water: number;
  sleep: number;
  steps: number;
  weight: number;
  createdAt: number;
  notePath?: string;
}

/** 可选个人指标：不参与 Agent 主流程，只提供手工记录与回顾。 */
export function renderMedical(view: SpaceOSView, mount: HTMLElement) {
  mount.addClass("space-medical");
  const data = view.plugin.data;
  const records = data.healthRecords as HealthRecord[];
  const today = getToday();
  const rec = records.find((r) => r.date === today);

  const heading = mount.createDiv({ cls: "medical-heading" });
  heading.createDiv({ cls: "tactical-title", text: "个人指标" });
  heading.createDiv({ cls: "tactical-subtitle", text: "可选的手工健康记录；不会被自动发送给 Agent，也不作为系统运行健康评分。" });

  const dash = mount.createDiv({ cls: "log-dashboard" });
  dash.createEl("span", { cls: "log-stat", text: `今日健康分 ${rec ? score(rec).total : "未记录"}` });

  // 录入表单
  const form = mount.createDiv({ cls: "glass health-form" });
  form.createDiv({ cls: "bic-header", text: "今日健康记录" });
  const fields = [
    { k: "water", label: "饮水 (ml)", def: 2000 },
    { k: "sleep", label: "睡眠 (h)", def: 7 },
    { k: "steps", label: "步数", def: 7000 },
    { k: "weight", label: "体重 (kg)", def: 60 },
  ] as const;
  const inputs: Record<string, HTMLInputElement> = {};
  fields.forEach((f) => {
    const row = form.createDiv({ cls: "health-field" });
    row.createSpan({ cls: "comm-contacts-label", text: f.label });
    const inp = row.createEl("input", { cls: "sos-crl", attr: { type: "number", value: String(rec ? rec[f.k] : "") } });
    inputs[f.k] = inp;
  });

  const saveBtn = form.createEl("button", { cls: "log-new-btn", text: rec ? "更新记录" : "保存记录" });
  saveBtn.addEventListener("click", async () => {
    // 健康卡按日期 1 卡 1 md（写入 healthFolder/日期.md）
    await createCard(view.plugin, "health", {
      fm: {
        id: rec?.id,
        date: today,
        water: Number(inputs.water.value) || 0,
        sleep: Number(inputs.sleep.value) || 0,
        steps: Number(inputs.steps.value) || 0,
        weight: Number(inputs.weight.value) || 0,
      },
      body: `# ${today} 健康记录`,
    }, { filename: today });
    view.renderCurrentPanel();
  });

  // 评分卡
  const scoreCard = mount.createDiv({ cls: "glass health-score" });
  if (rec) {
    const s = score(rec);
    const color = s.total >= 90 ? "#00e676" : s.total >= 60 ? "#ffc107" : "#ff5252";
    scoreCard.createDiv({ cls: "bic-header", text: "健康评分" });
    const big = scoreCard.createDiv({ cls: "health-score-num", attr: { style: `color:${color}` }, text: String(s.total) });
    void big;
    scoreCard.createDiv({ cls: "health-score-host", text: listFactors(s) });
  } else {
    scoreCard.createDiv({ cls: "bic-header", text: "健康评分" });
    scoreCard.createDiv({ cls: "health-score-num", text: "—" });
  }

  // 历史
  const history = mount.createDiv({ cls: "comm-list health-history" });
  history.createDiv({ cls: "tactical-title", text: "历史记录" });
  const sorted = records.slice().sort((a, b) => b.date.localeCompare(a.date)).slice(0, 10);
  if (sorted.length === 0) {
    history.createDiv({ cls: "tactical-empty", text: "暂无历史记录" });
  } else {
    sorted.forEach((r) => {
      const row = history.createDiv({ cls: "log-entry glass" });
      row.createDiv({ cls: "db-name", text: `${r.date} · 健康分 ${score(r).total}` });
      row.createDiv({ cls: "log-entry-time", text: `水${r.water}ml · 睡${r.sleep}h · 步${r.steps} · 重${r.weight}kg` });
    });
  }
}

function listFactors(s: { water: number; sleep: number; steps: number; bmi: number }) {
  return [
    `饮水 ${s.water}/30`,
    `睡眠 ${s.sleep}/30`,
    `步数 ${s.steps}/30`,
    `BMI ${s.bmi} 分`,
  ].join(" · ");
}

function score(rec: HealthRecord): { total: number; water: number; sleep: number; steps: number; bmi: number } {
  const cap = { water: 30, sleep: 30, steps: 30, bmi: 10 };
  void cap;
  const water = Math.min(30, Math.round((rec.water / 2000) * 25));
  const sleep = Math.min(30, Math.round((rec.sleep / 7) * 25));
  const steps = Math.min(30, Math.round((rec.steps / 7000) * 25));
  // BMI 加分（用默认身高 170cm）
  const h = 1.7;
  const bmi = rec.weight > 0 ? rec.weight / (h * h) : 0;
  const bmiBonus = bmi >= 18.5 && bmi <= 24 ? 6 : 0;
  return { total: water + sleep + steps + bmiBonus, water, sleep, steps, bmi: bmiBonus };
}
