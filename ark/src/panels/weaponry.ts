import type { SpaceOSView } from "../view";
import { generateId } from "../utils";
import { getSkin } from "../skins";

interface Target {
  x: number; y: number; r: number; hp: number; maxHp: number;
  vx: number; vy: number; life: number; maxLife: number; color: string;
}

interface WeaponGame {
  /** 开始/暂停切换，返回新状态（true=运行中） */
  toggle(): boolean;
}

/** 训练记录：训练射击小游戏（canvas）+ 结果记录。 */
export function renderWeaponry(view: SpaceOSView, mount: HTMLElement) {
  mount.addClass("space-weaponry");
  const data = view.plugin.data;
  const results = data.weaponResults as any[];

  const dash = mount.createDiv({ cls: "log-dashboard" });
  const best = results.reduce((m, r) => Math.max(m, r.score ?? 0), 0);
  dash.createEl("span", { cls: "log-stat", text: `训练 ${results.length} 次` });
  dash.createEl("span", { cls: "log-stat log-normal", text: `最佳 ${best}` });

  const stage = mount.createDiv({ cls: "weapon-stage" });
  const canvas = stage.createEl("canvas", { cls: "weapon-canvas" });
  const ctx = canvas.getContext("2d")!;
  canvas.width = 640;
  canvas.height = 380;

  const hint = stage.createDiv({ cls: "weapon-hint", text: "点击目标射击 · 连击加成 · 漏掉/射空扣生命" });
  const control = stage.createDiv({ cls: "log-toolbar" });
  const startBtn = control.createEl("button", { cls: "log-new-btn", text: "▶ 开始训练" });

  const listEl = mount.createDiv({ cls: "comm-list weapon-records" });

  const game = createGame(view, canvas, ctx, () => redrawRecords(), results);

  startBtn.addEventListener("click", () => {
    const runningNow = game.toggle();
    startBtn.setText(runningNow ? "⏸ 暂停" : "▶ 开始训练");
  });

  function redrawRecords() {
    listEl.empty();
    listEl.createDiv({ cls: "tactical-title", text: getSkin(view.plugin.data.settings.skin).panel.weaponRecord });
    const sorted = results.slice().sort((a, b) => b.executedAt - a.executedAt).slice(0, 10);
    if (sorted.length === 0) return;
    sorted.forEach((r) => {
      const row = listEl.createDiv({ cls: "log-entry glass" });
      row.createDiv({ cls: "db-name", text: `得分 ${r.score} · 命中率 ${r.accuracy ?? 0}%` });
      row.createDiv({ cls: "log-entry-time", text: new Date(r.executedAt).toLocaleString() });
    });
  }
  redrawRecords();
}

function createGame(
  view: SpaceOSView,
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  onSave: () => void,
  results: any[],
): WeaponGame {
  const W = canvas.width, H = canvas.height;
  let raf = 0;
  let running = false;
  let paused = false;
  let score = 0, best = 0, combo = 0, lives = 5;
  let shots = 0, hits = 0;
  let targets: Target[] = [];
  let mouse = { x: -100, y: -100 };
  const particles: { x: number; y: number; vx: number; vy: number; life: number; color: string }[] = [];
  let last = 0;

  const colors = ["#00e5ff", "#b34dff", "#00ff99", "#ffcc00", "#ff6699"];

  function reset() {
    score = 0; combo = 0; lives = 5; shots = 0; hits = 0; targets = [];
  }

  function spawn() {
    if (targets.length >= 6) return;
    const r = 14 + Math.random() * 14;
    const hp = Math.max(1, Math.round(2 + Math.random() * 2));
    targets.push({
      x: r + Math.random() * (W - 2 * r),
      y: r + 20 + Math.random() * (H - 2 * r - 20),
      r, hp, maxHp: hp,
      vx: (Math.random() - 0.5) * 40, vy: (Math.random() - 0.5) * 40,
      life: 2 + Math.random() * 3, maxLife: 5, color: colors[Math.floor(Math.random() * colors.length)],
    });
  }

  function shoot() {
    shots++;
    const hit = targets.find((t) => {
      const dx = mouse.x - t.x, dy = mouse.y - t.y;
      return Math.hypot(dx, dy) <= t.r + 6;
    });
    // 命中判定：优先打最近的命中目标
    const hittable = targets
      .map((t) => ({ t, d: Math.hypot(mouse.x - t.x, mouse.y - t.y) - t.r }))
      .filter((o) => o.d <= 6)
      .sort((a, b) => a.d - b.d);
    const targetObj = hittable[0]?.t;
    if (hit) { /* 兼容 fallback */ }
    if (targetObj) {
      targetObj.hp--;
      hits++;
      combo++;
      score += 10 + combo * 2;
      burst(targetObj.x, targetObj.y, targetObj.color);
      if (targetObj.hp <= 0) {
        score += 50;
        targets = targets.filter((t) => t !== targetObj);
      }
      if (combo > best) best = combo;
    } else {
      combo = 0;
      lives--;
      flashMiss();
      if (lives <= 0) end();
    }
  }

  function burst(x: number, y: number, color: string) {
    for (let i = 0; i < 10; i++) {
      const a = Math.random() * Math.PI * 2;
      const s = 40 + Math.random() * 80;
      particles.push({ x, y, vx: Math.cos(a) * s, vy: Math.sin(a) * s, life: 0.4 + Math.random() * 0.3, color });
    }
  }

  let missFlash = 0;
  function flashMiss() { missFlash = 0.4; }

  function end() {
    running = false;
    cancelAnimationFrame(raf);
    const accuracy = shots ? Math.round((hits / shots) * 100) : 0;
    // 存战绩
    best = results.reduce((m, r) => Math.max(m, r.score ?? 0), 0);
    results.unshift({
      id: generateId(), score, best: Math.max(best, score), accuracy, shots, hits,
      executedAt: Date.now(),
    });
    view.plugin.savePluginData().then(() => onSave());
    // 覆写全局 best 展示由外部 refresh 处理
    ctx.clearRect(0, 0, W, H);
    drawBanner(`训练结束 · 得分 ${score} · 命中率 ${accuracy}%`);
  }

  function drawBanner(text: string) {
    ctx.fillStyle = "rgba(0,0,0,.7)";
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#00e5ff";
    ctx.font = "18px Consolas, monospace";
    ctx.textAlign = "center";
    ctx.fillText(text, W / 2, H / 2);
  }

  function tick(now: number) {
    const dt = Math.min(0.05, (now - last) / 1000 || 0.016);
    last = now;
    if (!running) return;

    ctx.clearRect(0, 0, W, H);
    // 背景网格
    ctx.strokeStyle = "rgba(0,229,255,.08)";
    ctx.lineWidth = 1;
    for (let x = 0; x <= W; x += 32) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y <= H; y += 32) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

    // spawn
    if (Math.random() < dt * 1.2) spawn();

    // 粒子
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.x += p.vx * dt; p.y += p.vy * dt;
      p.vx *= 0.9; p.vy *= 0.9; p.life -= dt;
      if (p.life <= 0) { particles.splice(i, 1); continue; }
      ctx.fillStyle = p.color;
      ctx.globalAlpha = Math.max(0, p.life);
      ctx.beginPath(); ctx.arc(p.x, p.y, 2, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1;
    }

    // 目标
    for (let i = targets.length - 1; i >= 0; i--) {
      const t = targets[i];
      t.x += t.vx * dt; t.y += t.vy * dt;
      t.life -= dt;
      if (t.life <= 0) { targets.splice(i, 1); lives--; combo = 0; if (lives <= 0) { end(); return; } continue; }
      const lowHp = t.hp / t.maxHp < 0.35;
      ctx.strokeStyle = lowHp ? "#ff5252" : t.color;
      ctx.fillStyle = lowHp ? "rgba(255,82,82,.25)" : "rgba(0,229,255,.08)";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(t.x, t.y, t.r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.arc(t.x, t.y, 3, 0, Math.PI * 2); ctx.fillStyle = t.color; ctx.fill();
      // 生命条
      ctx.strokeStyle = "rgba(255,255,255,.5)";
      ctx.lineWidth = 1;
      ctx.strokeRect(t.x - t.r, t.y - t.r - 8, t.r * 2, 3);
      ctx.fillStyle = lowHp ? "#ff5252" : t.color;
      ctx.fillRect(t.x - t.r, t.y - t.r - 8, t.r * 2 * (t.hp / t.maxHp), 3);
    }

    // miss 闪光
    if (missFlash > 0) {
      missFlash -= dt;
      ctx.fillStyle = `rgba(255,82,82,${Math.max(0, missFlash)})`;
      ctx.fillRect(0, 0, W, H);
    }

    // HUD
    ctx.textAlign = "left";
    ctx.font = "14px Consolas, monospace";
    ctx.fillStyle = "#00e5ff";
    ctx.fillText(`SCORE: ${score}`, 10, 18);
    ctx.fillStyle = combo >= 2 ? "#ff9800" : "#cfe7f5";
    ctx.fillText(`COMBO: x${combo}`, 150, 18);
    ctx.fillStyle = "#cfe7f5";
    ctx.fillText(`BEST: ${best}`, 280, 18);
    let lifeBar = "LIVES: ";
    for (let i = 0; i < lives; i++) lifeBar += "■";
    for (let i = lives; i < 5; i++) lifeBar += "□";
    ctx.fillStyle = lives <= 1 ? "#ff5252" : "#29e6a7";
    ctx.fillText(lifeBar, 400, 18);

    // 准星
    ctx.strokeStyle = "#00e5ff";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(mouse.x, mouse.y, 7, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(mouse.x - 12, mouse.y); ctx.lineTo(mouse.x - 4, mouse.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(mouse.x + 4, mouse.y); ctx.lineTo(mouse.x + 12, mouse.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(mouse.x, mouse.y - 12); ctx.lineTo(mouse.x, mouse.y - 4); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(mouse.x, mouse.y + 4); ctx.lineTo(mouse.x, mouse.y + 12); ctx.stroke();

    raf = requestAnimationFrame(tick);
  }

  canvas.addEventListener("mousemove", (e) => {
    const r = canvas.getBoundingClientRect();
    mouse.x = ((e.clientX - r.left) / r.width) * W;
    mouse.y = ((e.clientY - r.top) / r.height) * H;
  });
  canvas.addEventListener("click", () => { if (running) shoot(); });

  return {
    toggle() {
      if (running) { running = false; cancelAnimationFrame(raf); return false; }
      reset(); running = true; last = performance.now(); targets = []; particles.length = 0;
      for (let i = 0; i < 3; i++) spawn();
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(tick);
      return true;
    },
  };
}
