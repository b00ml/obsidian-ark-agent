import { Modal, Notice } from "obsidian";
import { spawn } from "child_process";
import { existsSync } from "fs";
import { homedir } from "os";
import type ArkOSPlugin from "./main";
import type { CrtSession } from "./settings";
import { chat, agentChat, agentEndpoint, probeAgent, sessionDelete } from "./ai";
import { branchEventToProgress, getAgentCatalog, parseMentions } from "./multi-mention";
import { serveStart } from "./serve-control";
import { sanitizeFilename } from "./utils";
import { getSkin } from "./skins";
import { agentLive, notifyLive, removeLive, settleLiveStream, updateLive } from "./agent-live";
import { commitSession, newSession, sessionList, sessionsForProject, touchSession } from "./session-service";

interface Cmd {
  name: string;
  help: string;
  run: (args: string) => Promise<void>;
}

/** agentlab serve 默认启动目录（须含 config/config.json）与 python（项目 .venv 优先）
 *  —— OPT-113 起常量与进程控制统一收编至 serve-control.ts（serve_manage 单实例守护）。 */

/** 一键拉起 agentlab serve（自研）：已运行则跳过；否则经 serve_manage 单实例拉起
 *  （pidfile 守护 + 等健康检查；fail-closed token 缺失时 start 显式报错）。 */
export async function launchAgentlab(plugin: ArkOSPlugin): Promise<void> {
  const s = plugin.data.settings;
  const r = await probeAgent(s);
  if (r.ok) { new Notice(`agentlab serve 已在运行 v${r.version ?? "?"}`); return; }

  try {
    const out = await serveStart(s);
    new Notice(out || "agentlab serve 已启动，正拉起 CRT 等待就绪…");
  } catch (e: any) {
    new Notice(`启动 agentlab serve 失败：${e?.message ?? e}`);
    return;
  }
}

/** 打开 AI 助手：按 agentProvider 分派。
 *  - agentlab：拉起/复用 serve 后落 CRT 会话；
 *  - hermes：优先拉起 Hermes Desktop（桌面 GUI），未配置/失败则回退 CRT 终端（DEV-028/029）；
 *  - 直连：直接开 CRT。 */
export function launchAI(plugin: ArkOSPlugin): void {
  const s = plugin.data.settings;
  if (s.agentProvider === "agentlab") {
    void launchAgentlab(plugin).then(() => openCRT(plugin));
    return;
  }
  let exe = (s.desktopExePath || "").trim();
  // 未配置时自动探测常见安装位置（高可自定义：设置页可改；留空也能自动找；都没有才回退 CRT）
  if (!exe) {
    try {
      const cand = `${homedir()}\\AppData\\Local\\hermes\\hermes-agent\\apps\\desktop\\release\\win-unpacked\\Hermes.exe`;
      if (existsSync(cand)) exe = cand;
    } catch { /* 探测失败则走回退分支 */ }
  }
  if (exe) {
    try {
      const child = spawn(exe, [], { detached: true, stdio: "ignore" });
      child.unref();
      child.on("error", (e) => {
        console.warn("启动 Hermes Desktop 失败，回退 CRT：", e);
        openCRT(plugin);
      });
      new Notice("已启动 AI 助手（Hermes Desktop）");
      return;
    } catch (e) {
      console.warn("启动 Hermes Desktop 异常，回退 CRT：", e);
    }
  }
  openCRT(plugin);
}

/** 打开 AI 助手 CRT 终端模态。可选 sessionId：精确恢复某会话；缺省续聊上次会话（crtActiveId）。 */
export function openCRT(plugin: ArkOSPlugin, sessionId?: string): void {
  new CRTTerminal(plugin, sessionId).open();
}

class CRTTerminal extends Modal {
  private plugin: ArkOSPlugin;
  private stdout!: HTMLElement;
  private input!: HTMLElement;
  private sessionSelect!: HTMLSelectElement;
  private projectSelect!: HTMLSelectElement;
  private history: string[] = [];
  private cursor = -1;
  private messages: CrtSession["messages"];
  private session!: CrtSession;   // 当前工作会话（messages 与 session.messages 同引用）
  private inList = false;         // 是否已纳入 data.crtSessions（首条消息才纳入，空会话不落盘）
  private dirty = false;          // 有未落盘变更

  // —— 跨模态在途流登记：流绑定"发起会话"而非当前帧，杜绝切会话污染 / 误判被打断（DEV 修复）——
  // 关窗/切会话不取消流（长任务继续，重绘时按会话重挂载直播文本）；仅删除会话才真正取消。
  // OPT-109：注册表抽到 agent-live.ts 模块单例——CRT 与 Agent 工作台共享同一批在途流。
  private static readonly live = agentLive;
  private liveSessId: string | null = null;   // 当前视图正直播的会话 id（重绘时设置 / 切走后清空以丢弃旧锚点）
  private liveTextEl: HTMLElement | null = null;
  private liveStatEl: HTMLElement | null = null;

  constructor(plugin: ArkOSPlugin, sessionId?: string) {
    super(plugin.app);
    this.plugin = plugin;
    // data.crtSessions 兜底（老 data.json 无该字段）
    const list = sessionList(this.plugin);
    let target: CrtSession | undefined;
    const activePid = (this.plugin.data.settings.activeProjectId || "").trim();
    if (sessionId) {
      target = list.find((s) => s.id === sessionId && (s.projectId || "") === activePid);
    } else if (this.plugin.data.crtActiveId) {
      // crtActiveId 是全局续聊指针，不能覆盖当前 Project 的隔离边界。
      target = list.find((s) => s.id === this.plugin.data.crtActiveId
        && (s.projectId || "") === activePid);
    }
    if (!target) {
      target = sessionsForProject(this.plugin, activePid)
        .sort((a, b) => b.updatedAt - a.updatedAt)[0];
    }
    if (target) {
      this.session = target;
      this.inList = true;
    } else {
      // 续聊/指定会话不存在时，新建工作会话（草稿，首条消息才提交）
      this.session = this.makeSession();
    }
    this.messages = this.session.messages;
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.addClass("crt-modal");

    const wrap = contentEl.createDiv({ cls: "crt-wrap" });
    const skin = getSkin(this.plugin.data.settings.skin);
    const head = wrap.createDiv({ cls: "crt-head" });
    head.createDiv({ cls: "crt-title", text: skin.hints.crtTitle });
    const mode = this.plugin.data.settings.agentProvider === "hermes" ? "Hermes Agent"
      : this.plugin.data.settings.agentProvider === "agentlab" ? "Agentlab serve"
      : "直连";
    wrap.createDiv({ cls: "crt-sub", text: `${mode} · ${this.plugin.data.settings.aiProvider || "deepseek"} · ${this.plugin.data.settings.aiModel || "deepseek-chat"} · 输入 /help 查看命令` });

    // 会话栏：项目下拉（P0-2/OPT-107）+ 历史下拉 + 新会话按钮（OPT-073）
    const bar = wrap.createDiv({ cls: "crt-session-bar" });
    this.projectSelect = bar.createEl("select", { cls: "crt-session-select", attr: { title: "Project 长期任务空间：切换后历史隔离、注入项目规则/背景" } });
    this.renderProjectSelect();
    this.projectSelect.addEventListener("change", () => void this.onSelectProject());
    this.sessionSelect = bar.createEl("select", { cls: "crt-session-select" });
    this.renderSessionSelect();
    this.sessionSelect.addEventListener("change", () => this.onSelectSession());
    const newBtn = bar.createEl("button", { cls: "crt-btn", text: "＋ 新会话" });
    newBtn.addEventListener("click", () => void this.commands.find((c) => c.name === "new")?.run(""));
    const delBtn = bar.createEl("button", { cls: "crt-btn crt-btn-danger", text: "🗑 删除选中" });
    delBtn.addEventListener("click", () => this.deleteSelectedSession());

    this.stdout = wrap.createDiv({ cls: "crt-stdout" });
    this.repaint();
    const activePid = (this.plugin.data.settings.activeProjectId || "").trim();
    if (activePid) {
      const pname = this.plugin.data.settings.projects.find((p) => p.id === activePid)?.name || activePid;
      this.print("PROJECT", `当前项目：${pname}（历史与规则已隔离，空提示词输入 /projects 查看）`, "dim");
    }
    if (this.plugin.data.settings.agentProvider === "hermes"
        || this.plugin.data.settings.agentProvider === "agentlab") {
      void this.checkAgent();
    }

    const row = wrap.createDiv({ cls: "crt-input-row" });
    const sign = row.createDiv({ cls: "crt-prompt", text: ">" });
    this.input = row.createEl("input", { cls: "crt-input", placeholder: "下达指令… (Enter 发送)" });
    this.input.addEventListener("keydown", (e) => this.handleKey(e as KeyboardEvent));
    this.input.focus();
  }

  onClose() {
    this.settleLive();                       // 关窗先结算在途流到的发言（pi 式 teardown）
    void this.persistOnClose();
    // 关窗丢弃直播锚点（后续 onText 不再写已销毁 DOM），但不中止在途流本身——
    // 长任务继续在后台运行并写入源会话，重开该会话时由 repaint 重新挂载。
    this.liveSessId = null; this.liveTextEl = null; this.liveStatEl = null;
    this.contentEl.empty();
  }

  private handleKey(e: KeyboardEvent) {
    const ip = this.input as HTMLInputElement;
    if (e.key === "Enter") {
      e.preventDefault();
      const raw = ip.value.trim();
      if (!raw) return;
      ip.value = "";
      this.history.push(raw);
      this.cursor = this.history.length;
      this.submit(raw);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (this.cursor > 0) {
        this.cursor--;
        ip.value = this.history[this.cursor];
      }
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (this.cursor < this.history.length - 1) {
        this.cursor++;
        ip.value = this.history[this.cursor];
      } else {
        this.cursor = this.history.length;
        ip.value = "";
      }
    }
  }

  private get commands(): Cmd[] {
    const s = this.plugin.data.settings;
    const self = this;
    return [
      { name: "help", help: "列出命令", run: async () => this.commands.forEach((c) => this.print("USAGE", `/${c.name} ${c.help}`, "dim")) },
      { name: "new", help: "新开会话（当前会先保存）", run: async () => this.startNew() },
      { name: "sessions", help: "列出历史会话", run: async () => this.listSessions() },
      { name: "projects", help: "列出 Project 与当前激活项", run: async () => {
          const ps = this.plugin.data.settings;
          const cur = (ps.activeProjectId || "").trim();
          this.print("PROJECT", `当前激活：${cur || "全局"}（顶栏下拉切换；新建用命令面板「新建 Project」）`, "dim");
          (ps.projects || []).forEach((p) => this.print("PROJECT", `${p.id} · ${p.name}`, "dim"));
      } },
      { name: "load", help: "<id|序号> 载入会话", run: async (a) => this.loadSession(a) },
      { name: "rm", help: "<id|序号> 删除会话", run: async (a) => this.removeSession(a) },
      { name: "agent", help: "on|off|agentlab 切换 Agent 模式", run: async (a) => { const v = a.trim().toLowerCase(); const next = v === "on" ? "hermes" : v === "agentlab" ? "agentlab" : v === "off" ? "" : null; if (next !== null) { s.agentProvider = next; await this.persist(); this.print("CONFIG", `Agent 模式：${next === "hermes" ? "Hermes（外部常驻）" : next === "agentlab" ? "Agentlab serve（自研）" : "直连"}`, "dim"); if (next) void this.checkAgent(); } else this.print("INFO", "用法 /agent on|off|agentlab", "dim"); } },
      { name: "key", help: "<key> 设置 API Key", run: async (a) => { if (a) { s.aiApiKey = a; await this.persist(); this.print("CONFIG", "API Key 已更新", "dim"); } else this.print("INFO", "用法 /key <key>", "dim"); } },
      { name: "url", help: "<url> 设置 API 地址", run: async (a) => { if (a) { s.aiApiUrl = a; await this.persist(); this.print("CONFIG", "API 地址已更新", "dim"); } else this.print("INFO", "用法 /url <url>", "dim"); } },
      { name: "model", help: "<model> 设置模型", run: async (a) => { if (a) { s.aiModel = a; await this.persist(); this.print("CONFIG", "模型已更新", "dim"); } else this.print("INFO", "用法 /model <model>", "dim"); } },
      { name: "status", help: "查看当前配置", run: async () => { this.print("STATUS", `agent=${s.agentProvider || "direct"}\nprovider=${s.aiProvider || "deepseek"}\nurl=${s.aiApiUrl || "未设置"}\nkey=${s.aiApiKey ? "已设置" : "未设置"}\nmodel=${s.aiModel || "deepseek-chat"}\nhermes=${s.hermesUrl || "未设置"}\nhermesModel=${s.hermesModel || "hermes-agent"}\nhermesToken=${s.hermesToken ? "已设置" : "未设置"}\nagentlab=${s.agentlabUrl || "未设置"}\nagentlabModel=${s.agentlabModel || "agentlab-demo"}\nagentlabToken=${s.agentlabToken ? "已设置" : "未设置"}`, "dim"); } },
      { name: "save", help: "将本会话保存为笔记", run: async () => { await this.saveConversation(); } },
    ];
  }

  private async submit(raw: string) {
    this.print("YOU", raw, "user");
    if (raw.startsWith("/")) {
      const [cmd, ...rest] = raw.slice(1).split(" ");
      const c = this.commands.find((x) => x.name === cmd);
      if (c) await c.run(rest.join(" "));
      else this.print("ERROR", `未知命令 /${cmd}，输入 /help 查看`, "err");
      return;
    }
    this.messages.push({ role: "user", content: raw });
    if (!this.inList) this.commitIfNew();   // 首条消息才把会话纳入列表并更新续聊指针
    touchSession(this.plugin, this.session);
    this.dirty = true;
    if (this.plugin.data.settings.agentProvider === "hermes"
        || this.plugin.data.settings.agentProvider === "agentlab") {
      await this.submitAgent();
      return;
    }
    this.print("[🧠]", "思考中…", "dim");
    try {
      const reply = await chat(this.plugin.data.settings, this.messages);
      this.messages.push({ role: "assistant", content: reply });
      this.print("AI", reply, "ai");
    } catch (err: any) {
      this.print("ERROR", String(err?.message ?? err), "err");
    }
  }

  /** Agent 内核模式（Hermes / agentlab）：/v1/responses 流式 + 工具轨迹渲染
   *
   * 关键：流绑定发起会话（sessId/sessMsgs 在提交瞬间捕获），而非可变的 this.messages/DOM。
   * 这样关窗/切会话不会中断长任务（src 持续运行），也不会把旧会话回答写进新会话。
   * 当前视图若正在展示该会话，则把直播文本实时挂到视图上；否则继续在后台累积，
   * 待重绘该会话时再挂载（live 登记表由 repaint 读取）。
   */
  private async submitAgent() {
    const s = this.plugin.data.settings;
    const sessId = this.session.id;
    const sessMsgs = this.messages;          // 源会话消息数组（稳定引用，杜绝跨会话污染）
    const abr = new AbortController();
    const startedAt = Date.now();
    CRTTerminal.live.set(sessId, {
      text: "", abort: abr, started: startedAt, elapsed: 0,
      tools: [], status: "running",
    });
    notifyLive(sessId);
    const stream = () => CRTTerminal.live.get(sessId)!;

    const line = this.stdout.createDiv({ cls: "crt-line ai" });
    line.createSpan({ cls: "crt-tag", text: "AI" });
    const body = line.createSpan({ cls: "crt-text" });
    const statusLine = this.stdout.createDiv({ cls: "crt-line" });
    statusLine.createSpan({ cls: "crt-tag", text: "RUNNING" });
    const statusText = statusLine.createSpan({ cls: "crt-text", text: "运行中 0s" });
    const stopBtn = statusLine.createEl("button", { cls: "crt-btn crt-btn-danger crt-stop", text: "⏹ 停止" });
    let stopped = false;

    // 本视图正在展示发起会话 → 登记直播锚点；否则丢弃旧锚点（流在后台继续）
    this.liveSessId = sessId; this.liveTextEl = body; this.liveStatEl = statusText;
    stopBtn.addEventListener("click", () => { abr.abort(); stopped = true; statusText.setText("已停止"); stopBtn.remove(); });

    const syncLive = () => {
      if (this.liveSessId !== sessId || !this.liveTextEl) return;
      this.liveTextEl.setText(stream().text);
      this.stdout.scrollTop = this.stdout.scrollHeight;
    };
    try {
      // P2-2/F5-019：CRT 入口同样支持 @ 指派（与工作台共用同一解析层，避免两套实现）
      let crtMulti: string[] = [];
      const lastUser = [...sessMsgs].reverse().find((m) => m.role === "user");
      if (lastUser && /@[A-Za-z0-9_.-]/.test(lastUser.content)) {
        const cat = await getAgentCatalog(s);
        const parsed = parseMentions(lastUser.content, cat.agents.map((a) => a.name));
        if (parsed.unknown.length) {
          this.print("WARN", `未识别的 Agent：${parsed.unknown.join("、")}（按普通文本发送）`, "dim");
        }
        if (parsed.multi.length) {
          crtMulti = parsed.multi;
          lastUser.content = parsed.cleaned || lastUser.content;
          this.print("MULTI", `并行咨询 ${crtMulti.map((n) => "@" + n).join("、")}，由主 Agent 汇总`, "dim");
        }
      }
      const reply = await agentChat(s, sessMsgs, {
        onText: (d) => { stream().text += d; notifyLive(sessId); syncLive(); },
        onTool: (name, phase, detail) => {
          stream().tools.push({ name, phase, output: detail });
          notifyLive(sessId);
          this.print(phase === "start" ? "TOOL" : "DONE", phase === "start" ? `调用 ${name}…` : `${name} ✓`, "dim");
        },
        onStatus: (st) => {
          updateLive(sessId, { elapsed: st.elapsed });
          if (!stopped && this.liveSessId === sessId && this.liveStatEl) this.liveStatEl.setText(`运行中 ${st.elapsed}s`);
        },
        onBranch: (ev) => {
          const hit = branchEventToProgress(ev);
          if (!hit) return;
          const p = hit.patch;
          const tail = p.status === "running" ? "运行中…"
            : p.status === "ok"
              ? `完成 ${p.elapsed ?? 0}s${p.chars ? ` · ${p.chars} 字` : ""}${p.truncated ? " · 已截断归档" : ""}`
              : `失败：${p.error ?? "未知错误"}`;
          this.print("BRANCH", `@${p.agent} ${tail}`, "dim");
          updateLive(sessId, {});
        },
        onMultiSummary: (ev) => {
          const ok = Number(ev.ok_count) || 0;
          const failed = Array.isArray(ev.failed) ? ev.failed.join("、") : "";
          const skipped = Array.isArray(ev.skipped) ? ev.skipped.join("、") : "";
          this.print("MULTI", `${ok} 支成功`
            + (failed ? ` · 失败 ${failed}` : "")
            + (skipped ? ` · 跳过 ${skipped}` : "")
            + (ev.denied ? " · 未获批准，已降级单 Agent 直答" : ""), "dim");
        },
      }, abr.signal, { projectId: this.session.projectId, multi: crtMulti });
      const lvSettled = CRTTerminal.live.get(sessId);
      if (lvSettled?.settled && lvSettled.settleIdx != null) {
        // 切走时已结算过半截 → 完成回答替换该半截消息，避免重复
        sessMsgs[lvSettled.settleIdx].content = reply;
      } else {
        sessMsgs.push({ role: "assistant", content: reply });   // 只写入源会话
      }
      if (this.liveSessId === sessId && this.liveTextEl) this.liveTextEl.setText(reply || "（无文本回复）");
      updateLive(sessId, { status: "completed", elapsed: Math.floor((Date.now() - startedAt) / 1000) });
      if (this.liveStatEl && !stopped) this.liveStatEl.setText("完成");
      if (!stopped) statusLine.addClass("dim");
    } catch (err: any) {
      // 主动停止 / 断流：保留已流式文本到源会话，避免半途回答丢失；
      // 未结算（从未被切走/view）也一并结算，保证半截回答必落盘
      const lv = stream();
      if (lv && !lv.settled && lv.text) {
        sessMsgs.push({ role: "assistant", content: lv.text });
        lv.settled = true;
        if (this.liveSessId === sessId && this.liveTextEl) this.liveTextEl.setText(lv.text);
      } else if (!stopped && this.liveSessId === sessId) {
        this.print("ERROR", String(err?.message ?? err), "err");
      }
      updateLive(sessId, { status: "failed", error: String(err?.message ?? err) });
      if (this.liveStatEl && !stopped) this.liveStatEl.setText("失败");
      if (!stopped) statusLine.addClass("dim");
    } finally {
      removeLive(sessId);
      if (this.liveSessId === sessId) { this.liveSessId = null; this.liveTextEl = null; this.liveStatEl = null; }
    }
  }

  /** 探测当前 Agent 内核（Hermes / agentlab serve；未运行则提示拉起） */
  private async checkAgent() {
    const s = this.plugin.data.settings;
    const ep = agentEndpoint(s);
    // 刚经 launchAgentlab 拉起 / /agent on 时 serve 需 1~2s 冷启动：给几次重试吃下时序，避免假"未连接"
    let r = await probeAgent(s);
    for (let i = 0; !r.ok && i < 4; i++) {
      await new Promise((res) => setTimeout(res, 700));
      r = await probeAgent(s);
      if (r.ok) break;
    }
    if (r.ok) {
      this.print("[" + ep.label.toUpperCase() + "]", `已连接 v${r.version ?? "?"} · ${ep.model} · ${ep.base || "未设置地址"}`, "dim");
    } else {
      this.print("ERROR", r.error ?? "连接失败", "err");
      const row = this.stdout.createDiv({ cls: "crt-line" });
      row.createSpan({ cls: "crt-tag", text: "ACTION" });
      const btn = row.createEl("button", { cls: "crt-btn",
        text: ep.label === "agentlab" ? "重试拉起 agentlab" : "启动 Hermes" });
      btn.addEventListener("click", () => {
        if (ep.label === "agentlab") {
          // 复用一键拉起：真实 spawn；失败信息由 launchAgentlab 内的 Notice 兜底
          void launchAgentlab(this.plugin).then(() => void this.checkAgent());
        } else {
          new Notice("请在系统 PowerShell 运行：hermes gateway start，然后输入 /agent on 重试");
        }
      });
    }
  }

  private print(tag: string, text: string, kind: string) {
    const line = this.stdout.createDiv({ cls: `crt-line ${kind}` });
    line.createSpan({ cls: "crt-tag", text: tag });
    const body = line.createSpan({ cls: "crt-text" });
    body.setText(text);
    this.stdout.scrollTop = this.stdout.scrollHeight;
  }

  private async persist() {
    await this.plugin.savePluginData();
  }

  private async saveConversation() {
    const s = this.plugin.data.settings;
    const vault = this.plugin.app.vault;
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const path = `${s.aiFolder || "02-DB/AI对话"}/${sanitizeFilename(stamp) + "-" + sanitizeFilename(this.messages[1]?.content?.slice(0, 20) || "conversation")}.md`;
    try {
      await vault.adapter.mkdir((s.aiFolder || "02-DB/AI对话"));
    } catch { /* 目录已存在 */ }
    const lines = this.messages
      .filter((m) => m.role !== "system")
      .map((m) => `${m.role === "user" ? "### 用户" : "### AI 助手"}\n\n${m.content}`)
      .join("\n\n---\n\n");
    try {
      await vault.create(path, `---\ntype: ai_conversation\n---\n\n${lines}\n`);
      this.print("SAVED", path, "dim");
    } catch (err: any) {
      this.print("ERROR", "保存失败: " + (err?.message ?? err), "err");
    }
  }

  // ── 会话持久化 / 切换（OPT-073）────────────────────────────
  private makeSession(): CrtSession {
    return newSession(this.plugin);
  }

  /** 首次发言才把会话纳入 data.crtSessions 并更新续聊指针（避免空会话成为 active） */
  private commitIfNew() {
    if (this.inList) return;
    this.inList = commitSession(this.plugin, this.session);
  }

  private async saveSession() {
    if (!this.dirty || !this.inList) return;
    await this.persistNow();
  }

  /** 统一落盘：刷 updatedAt → 容量治理（上限 50，按 updatedAt 保留最新）→ saveData */
  private async persistNow() {
    touchSession(this.plugin, this.session);
    this.dirty = false;
    await this.plugin.savePluginData();
  }

  /** 关窗落盘：已纳入列表的会话记录最新 updatedAt */
  private async persistOnClose() {
    if (!this.inList) return;
    await this.persistNow();
  }

  /** 结算在途流（pi 式 teardown）：切走/关窗前把已产生的部分回答写入源会话。
   *  幂等：已 settled 则跳过；返回后即使不再打开该会话，半途回答也会随 dirty 落盘可见。 */
  private settleLive() {
    if (settleLiveStream(this.session)) this.dirty = true;  // OPT-109：结算逻辑收敛到 agent-live.ts
  }

  private async startNew() {
    this.settleLive();                          // 切走先结算，再保存/切换
    await this.saveSession();
    this.session = this.makeSession();
    this.inList = false;
    this.dirty = false;
    this.messages = this.session.messages;
    this.renderSessionSelect();
    this.repaint();
    this.print("[SYSTEM]", "已开始新会话。", "dim");
  }

  /** 清空 stdout 并按当前会话完整重绘；若该会话有在途流，则把直播文本重新挂载（长任务切走再返回仍可见） */
  private repaint() {
    this.stdout.empty();
    // 重绘丢弃旧的直播锚点，避免后续 onText 写到被清空的 DOM
    this.liveSessId = null; this.liveTextEl = null; this.liveStatEl = null;
    const skin = getSkin(this.plugin.data.settings.skin);
    this.print("[SYSTEM]", `${skin.brand} 已就绪。输入消息开始对话，输入 /help 查看命令。`, "dim");
    if (this.inList) this.print("INFO", `[会话] ${this.session.title}`, "dim");
    for (const m of this.session.messages) {
      if (m.role === "system") continue;
      this.print(m.role === "user" ? "YOU" : "AI", m.content, m.role === "user" ? "user" : "ai");
    }
    // 在途流重挂载：切回来仍能看到进行中的回答与运行秒数；
    // 已 settled 的半截回答已在 messages 里（作为普通 AI 消息），不重复挂载直播块
    const live = CRTTerminal.live.get(this.session.id);
    if (live && !live.settled) {
      const lline = this.stdout.createDiv({ cls: "crt-line ai" });
      lline.createSpan({ cls: "crt-tag", text: "AI" });
      const lbody = lline.createSpan({ cls: "crt-text", text: live.text });
      for (const tool of live.tools) {
        this.print(tool.phase === "start" ? "TOOL" : "DONE",
          tool.phase === "start" ? `调用 ${tool.name}…` : `${tool.name} ✓`, "dim");
      }
      const stline = this.stdout.createDiv({ cls: "crt-line" });
      stline.createSpan({ cls: "crt-tag", text: "RUNNING" });
      const stText = stline.createSpan({ cls: "crt-text", text: `运行中 ${live.elapsed}s` });
      this.liveSessId = this.session.id; this.liveTextEl = lbody; this.liveStatEl = stText;
    }
    this.stdout.scrollTop = this.stdout.scrollHeight;
  }

  /** P0-2/OPT-107：Project 下拉（全局 + 已建项目）；切换后历史按项目过滤 */
  private renderProjectSelect() {
    const sel = this.projectSelect;
    if (!sel) return;
    sel.empty();
    const s = this.plugin.data.settings;
    const cur = (s.activeProjectId || "").trim();
    const g = document.createElement("option");
    g.value = "";
    g.textContent = "🌐 全局（无项目上下文）";
    if (!cur) g.selected = true;
    sel.appendChild(g);
    (s.projects || []).forEach((p) => {
      const o = document.createElement("option");
      o.value = p.id;
      o.textContent = `📁 ${p.name}`;
      if (p.id === cur) o.selected = true;
      sel.appendChild(o);
    });
  }

  private async onSelectProject() {
    const v = this.projectSelect?.value || "";
    const s = this.plugin.data.settings;
    if ((s.activeProjectId || "") === v) return;
    const live = CRTTerminal.live.get(this.session.id);
    if (live && !live.settled) {
      // 在途请求仍绑定旧会话；切换后继续输入会把旧项目任务误认为当前项目任务。
      this.projectSelect.value = (s.activeProjectId || "").trim();
      new Notice("当前项目有在途任务，请停止或等待完成后再切换");
      return;
    }
    this.settleLive();
    await this.saveSession();
    s.activeProjectId = v;
    const next = this.plugin.data.crtSessions
      .filter((x) => (x.projectId || "") === v)
      .sort((a, b) => b.updatedAt - a.updatedAt)[0];
    if (next) {
      this.session = next;
      this.inList = true;
    } else {
      this.session = this.makeSession();
      this.inList = false;
    }
    this.plugin.data.crtActiveId = next?.id ?? null;
    this.messages = this.session.messages;
    this.dirty = false;
    await this.persist();
    this.renderProjectSelect();
    this.renderSessionSelect();
    this.repaint();
    const pname = s.projects.find((p) => p.id === v)?.name || "全局";
    this.print("PROJECT", `已切换：${v ? pname : "全局"}——会话列表已按项目过滤，新会话将绑定该项目`, "dim");
  }

  private renderSessionSelect() {
    const sel = this.sessionSelect;
    if (!sel) return;
    sel.empty();
    const pid = (this.plugin.data.settings.activeProjectId || "").trim();
    const list = this.plugin.data.crtSessions.filter((x) => (x.projectId || "") === pid);
    const curId = this.inList && (this.session.projectId || "") === pid ? this.session.id : "";
    if (!curId) {
      const o = document.createElement("option");
      o.value = "";
      o.textContent = "（新会话）";
      o.selected = true;
      sel.appendChild(o);
    }
    list.forEach((s) => {
      const o = document.createElement("option");
      o.value = s.id;
      const n = Math.max(0, s.messages.length - 1);
      const d = new Date(s.updatedAt);
      const hm = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
      o.textContent = `${s.title}（${n}条 · ${hm}）`;
      if (s.id === curId) o.selected = true;
      sel.appendChild(o);
    });
  }

  private onSelectSession() {
    const v = this.sessionSelect?.value;
    if (!v || v === this.session.id) return;
    void this.loadSession(v);
  }

  private async loadSession(q: string) {
    const list = this.plugin.data.crtSessions;
    const query = q.trim();
    if (!query) { this.print("INFO", "用法 /load <id|序号>（用 /sessions 查看）", "dim"); return; }
    let target: CrtSession | undefined;
    if (/^\d+$/.test(query)) target = list[Number(query) - 1];
    else target = list.find((s) => s.id === query);
    if (!target) { this.print("ERROR", "未找到会话。", "err"); return; }
    if (target.id === this.session.id) { this.print("INFO", "已在该会话。", "dim"); return; }
    this.settleLive();                          // 切走先结算在途流，再保存/切换
    await this.saveSession();
    this.session = target;
    this.inList = true;
    this.messages = this.session.messages;
    this.dirty = false;
    this.renderSessionSelect();
    this.repaint();
    this.print("INFO", `已载入会话：${target.title}`, "dim");
  }

  private listSessions() {
    const list = this.plugin.data.crtSessions;
    if (!list.length) { this.print("INFO", "暂无历史会话。开始对话后自动保存。", "dim"); return; }
    const items = list.map((s, i) => {
      const cur = s.id === this.session.id ? " ▶" : "";
      const n = Math.max(0, s.messages.length - 1);
      const d = new Date(s.updatedAt);
      const hm = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
      return `${i + 1}. ${s.title}（${n}条 · ${hm}）${cur}`;
    });
    this.print("INFO", items.join("\n"), "dim");
  }

  /** 顶栏「🗑 删除选中」：删除下拉当前选中的会话（删除当前会话后自动切到新会话） */
  private deleteSelectedSession() {
    const id = this.sessionSelect?.value;
    if (!id) { this.print("INFO", "当前为新会话，暂无可删除的历史会话。", "dim"); return; }
    void this.removeSession(id);
  }

  private async removeSession(q: string) {
    const query = q.trim();
    const list = this.plugin.data.crtSessions;
    if (!query) { this.print("INFO", "用法 /rm <id|序号>（用 /sessions 查看）", "dim"); return; }
    let idx = -1;
    if (/^\d+$/.test(query)) idx = Number(query) - 1;
    else idx = list.findIndex((s) => s.id === query);
    if (idx < 0 || idx >= list.length) { this.print("ERROR", "未找到会话。", "err"); return; }
    const removed = list[idx];
    // 4.0 执行线 #8：serve 端同步删除（仅 agentlab；失败不阻断本地清理）——
    // 顶栏「删除选中」与 /rm 命令都汇入本函数，serve 接线放这一处即可覆盖两路；
    // 回显一行简讯（stdout 可能已随关窗卸载，脱管 DOM 写入无副作用）
    const s = this.plugin.data.settings;
    if (s.agentProvider === "agentlab") {
      void sessionDelete(s, removed.id).then((r) => {
        if (r?.ok) this.print("DONE", `serve 端已同步清理：${removed.title}`, "dim");
        else this.print("INFO", `serve 端删除未生效（本地已删除）：${removed.title}`, "dim");
      });
    }
    list.splice(idx, 1);
    // 删除会话 → 终止其仍在运行的流（会话已不复存在，无需再后台续跑）
    const live = CRTTerminal.live.get(removed.id);
    if (live) live.abort.abort();
    if (removed.id === this.session.id) {
      if (this.plugin.data.crtActiveId === removed.id) this.plugin.data.crtActiveId = null;
      this.session = this.makeSession();
      this.inList = false;
      this.dirty = false;
      this.messages = this.session.messages;
      this.renderSessionSelect();
      this.repaint();
    } else {
      this.renderSessionSelect();
    }
    await this.plugin.savePluginData();
    this.print("DONE", `已删除会话：${removed.title}`, "dim");
  }
}
