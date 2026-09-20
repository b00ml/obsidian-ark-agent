/** Agent 工作台（OPT-109 W1，二期修切换）：Ark 内嵌 Agent 前端，替代 CRT 终端正态。
 *
 * 三栏：会话侧栏（项目切换 + 按项目分组的会话列表）｜消息流（Markdown + 工具轨迹）｜项目上下文条。
 * 设计依据：docs/Agent工作台设计.md；宿主 = ark Tab（主控大厅）原地升级，HUD 保留为头部。
 *
 * 二期修复（用户验收反馈"无法切换会话/无法选项目"）：渲染闭包曾捕获挂载时的旧会话
 * 对象——切会话/切项目后 wbSession 已变而重渲染仍画旧引用。现约定：**一切渲染点
 * 经 currentSession(view) 现取现用，禁止跨渲染捕获 sess**；项目切换器进侧栏顶部。
 *
 * 复用：会话同源 data.crtSessions；在途流 agent-live.ts（与 CRT 互通）；结算 settleLiveStream。
 * W1 边界：工具轨迹仅流式期间可见；上下文用量条 W2。
 */
import { FuzzySuggestModal, MarkdownRenderer, Notice, requestUrl, TFile } from "obsidian";
import {
  branchEventToProgress, getAgentCatalog, parseMentions,
  type AgentInfo, type BranchProgress,
} from "./multi-mention";
import type { SpaceOSView } from "./view";
import type ArkOSPlugin from "./main";
import type { CrtSession, ProjectInfo } from "./settings";
import { agentChat, agentEndpoint, resolveApproval, sessionDelete } from "./ai";
import { copyText, previewGeneratedModal } from "./ai-asst";
import { agentLive, notifyLive, removeLive, settleLiveStream, subscribeLive, updateLive } from "./agent-live";
import { commitSession, newSession, touchSession } from "./session-service";
import { createProject } from "./projects";
import { confirmDialog, inputDialog, conflictDialog } from "./ui";
import { writeMarkdown } from "./sync";
import {
  artifactsForSession, extractSessionArtifacts, upsertArtifacts, validateArtifactPaths,
  persistProjectArtifactIndex, restoreProjectArtifacts,
  type ArtifactToolOutput,
} from "./artifacts";
import { skillCatalog, SKILL_CATALOG, type ArkSkill } from "./skill-catalog";
import {
  buildResearchDraft, buildResearchPrompt, extractRetrievalEnvelope,
  parseResearchRefs, researchDraftFilename, researchDraftFrontmatter,
  stripResearchFrontmatter, type ResearchBrief,
} from "./research-draft";

// 最近一次 context-status 结果（W2 用量条数据源；refreshUsage 刷新，renderCtx 绘制）
let wbUsage: {
  tokens: number; budget: number; usage_pct: number; nudge_pct: number;
  hard_trim_pct: number; messages: number; scope?: string;
  project_context_loaded?: boolean; project_context_tokens?: number;
} | null = null;
let wbUsageAt = 0;        // 节流：同会话 2s 内不重复请求
let wbUsageSess = "";     // 上次取数的会话 id：切换会话强制刷新（修"用量条不随会话更新"）

/** @ 笔记引用选择器：vault 全量 md 模糊搜索 → 插入 [[链接]] */
class NotePickModal extends FuzzySuggestModal<TFile> {
  constructor(app: any, private onPick: (f: TFile) => void) {
    super(app);
    this.setPlaceholder("选择要引用的笔记（插入 [[链接]]）");
  }
  getItems(): TFile[] { return this.app.vault.getMarkdownFiles(); }
  getItemText(f: TFile): string { return f.path; }
  onChooseItem(f: TFile): void { this.onPick(f); }
}

// 工作台当前会话草稿/覆盖（模块级：Tab 切走再回来不丢；与会话列表同源）
let wbSession: CrtSession | null = null;
// A structured brief is kept only for the session that launched the research
// flow.  It is consumed after the first successful retrieval-backed response;
// ordinary Agent turns never become reports by accident.
const wbResearchBriefs = new Map<string, ResearchBrief>();
// 编辑器联动（W3/OPT-067）：右键「发送到 Agent 工作台」的选区暂存，下一次渲染注入输入框
let wbPrefill: string | null = null;
// 当前工作台输入框引用（面板渲染时更新；prefill 在面板已渲染时走立即写入路径）
let wbInput: HTMLTextAreaElement | null = null;
const wbLiveSubscriptions = new WeakMap<HTMLElement, () => void>();

/** 把选区注入工作台输入框：面板已渲染 → 立即写入并聚焦（切 Tab 卫语句拦不住）；
 *  面板未渲染 → 暂存，渲染时消费。 */
export function prefillWorkbench(text: string): void {
  const payload = `请基于以下选区内容进行处理：\n\n${text}`;
  if (wbInput && wbInput.isConnected) {
    wbInput.value = payload;
    wbInput.focus();
    wbPrefill = null;
    return;
  }
  wbPrefill = payload;
}

/** 编辑器右键「发送到 Agent 工作台」总入口：预填 + 定位到主控大厅 Tab */
export function sendSelectionToWorkbench(plugin: ArkOSPlugin, selection: string): void {
  prefillWorkbench(selection);
  // 遍历所有布局区找 SpaceOS 视图（类型串与 view.VIEW_TYPE_SPACE_OS 一致，避免运行时循环导入）
  for (const leaf of plugin.app.workspace.getLeavesOfType("ark-view")) {
    const v = leaf.view as { switchTab?: (t: string) => void };
    if (v?.switchTab) {
      v.switchTab("ark");
      return;
    }
  }
  void plugin.activateView();  // 视图未开 → 激活（onOpen 渲染时消费 prefill）
}

function activePid(view: SpaceOSView): string {
  return (view.plugin.data.settings.activeProjectId || "").trim();
}

function draftSession(plugin: ArkOSPlugin, pid: string): CrtSession {
  // 通过共享生命周期服务创建，工作台仍保留无 system 消息的旧 UI 草稿形态。
  const session = newSession(plugin, pid);
  session.messages = [];
  return session;
}

/** Turn one explicit research run into a confirmed, traceable Vault artifact. */
async function confirmResearchDraft(
  plugin: ArkOSPlugin,
  session: CrtSession,
  brief: ResearchBrief,
  toolOutputs: ArtifactToolOutput[],
): Promise<void> {
  const envelope = extractRetrievalEnvelope(toolOutputs);
  if (!envelope) {
    new Notice("研究草稿未捕获完整 Retrieval envelope，已保留会话，不自动写入。", 5000);
    return;
  }
  const now = new Date().toISOString();
  const built = buildResearchDraft(brief, envelope, now);
  const folder = plugin.data.settings.reportFolder || "02-DB/报告";
  const filename = researchDraftFilename(built.meta.title, now);
  const path = `${folder}/${filename}.md`;
  let previous = "";
  const existing = plugin.app.vault.getAbstractFileByPath(path);
  if (existing instanceof TFile) {
    previous = await plugin.app.vault.read(existing);
  }
  const action = await previewGeneratedModal(
    plugin,
    `快速研究草稿 · ${built.meta.title}`,
    previous,
    built.markdown,
  );
  if (action === "copy") {
    await copyText(plugin, built.markdown);
    new Notice("研究草稿已复制，未写入 Vault");
    return;
  }
  if (action !== "save") return;

  const writtenPath = await writeMarkdown(
    plugin,
    folder,
    filename,
    researchDraftFrontmatter(brief, built.meta, now),
    stripResearchFrontmatter(built.markdown),
  );
  // Feed the exact generated Markdown back through the existing Artifact
  // extractor so source refs and provenance stay on the shared F3 path.
  const artifactSession: CrtSession = {
    ...session,
    messages: [...session.messages, { role: "assistant", content: built.markdown }],
  };
  const detected = extractSessionArtifacts(artifactSession, [{
    name: "vault_write",
    output: JSON.stringify({ ok: true, path: writtenPath, operation: existing ? "updated" : "created" }),
  }], now);
  if (detected.length) {
    upsertArtifacts(plugin, validateArtifactPaths(plugin, detected));
    if (session.projectId) await persistProjectArtifactIndex(plugin, session.projectId);
  }
  await plugin.savePluginData();
  new Notice(`研究草稿已写入：${writtenPath}`);
}

/** 解析"当前应显示的会话"：显式草稿（且项目匹配）→ crtActiveId（项目匹配）→ 该项目最近会话 → 新草稿。
 *  每次渲染现取现用——这是二期切换修复的核心约定。 */
function currentSession(view: SpaceOSView): CrtSession {
  const pid = activePid(view);
  const data = view.plugin.data;
  if (wbSession && (wbSession.projectId || "") === pid) return wbSession;
  wbSession = null;
  const active = data.crtActiveId ? data.crtSessions.find((s) => s.id === data.crtActiveId) : undefined;
  if (active && (active.projectId || "") === pid) return active;
  const recent = data.crtSessions.find((s) => (s.projectId || "") === pid);
  if (recent) return recent;
  wbSession = draftSession(view.plugin, pid);
  return wbSession;
}

/** 快捷技能按钮行（#5/OPT-132）：与 skills/ 权威目录对应；内置清单，后续可配置化 */
/** F4：快捷按钮与技能目录共用同一份定义，避免出现两套入口语义。 */
const SKILL_BUTTONS = SKILL_CATALOG;

/** 渲染入口：SpaceOSView 的 ark Tab 主体（HUD 之下） */
export function renderAgentWorkbench(view: SpaceOSView, mount: HTMLElement) {
  wbLiveSubscriptions.get(mount)?.();
  mount.addClass("agent-wb-host");
  const grid = mount.createDiv({ cls: "agent-wb" });
  const side = grid.createDiv({ cls: "agent-wb-side" });
  const flow = grid.createDiv({ cls: "agent-wb-flow" });
  const ctx = grid.createDiv({ cls: "agent-wb-ctx" });
  // #5/OPT-132：快捷技能按钮行（skill 即按钮，点击预填；内置清单，后续可配置化）
  const skillRow = grid.createDiv({ cls: "ark-skill-row" });
  const inputWrap = grid.createDiv({ cls: "agent-wb-inputrow" });
  let renderQueued = false;

  const ta = inputWrap.createEl("textarea", {
    cls: "agent-wb-input",
    attr: { placeholder: "下达指令…（Enter 发送 / Shift+Enter 换行）", rows: "2" },
  });
  const sendBtn = inputWrap.createEl("button", { cls: "agent-wb-send", text: "发送 ▸" });
  const stopBtn = inputWrap.createEl("button", { cls: "agent-wb-stop", text: "■ 停止" });
  stopBtn.style.display = "none";
  ta.addEventListener("keydown", (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send(ta.value.trim());
    }
  });
  sendBtn.addEventListener("click", () => void send(ta.value.trim()));
  stopBtn.addEventListener("click", () => {
    const lv = agentLive.get(currentSession(view).id);
    if (lv) lv.abort.abort();
  });
  // @ 笔记引用（W2）：模糊选择 → 光标处插入 [[链接]]
  const atBtn = inputWrap.createEl("button", { cls: "agent-wb-btn", text: "📎",
    attr: { title: "引用笔记（模糊搜索，插入 [[链接]]）" } });
  // P2-2/F5-019：@ 指派多 Agent。与"行首/空白后输入 @"同一条路径。
  const mentionBtn = inputWrap.createEl("button", { cls: "agent-wb-btn", text: "＠",
    attr: { title: "指派 Agent（多 Agent 并行作答，主 Agent 汇总；需 agentlab serve 在线）" } });
  const INSERT = (a: AgentInfo) => {
    const pos = ta.selectionStart ?? ta.value.length;
    const before = ta.value.slice(0, pos);
    const sep = before && !/\s$/.test(before) ? " " : "";
    ta.setRangeText(`${sep}@${a.name} `, pos, pos, "end");
    ta.focus();
  };
  const openAgentPicker = async (force = false) => {
    const cat = await getAgentCatalog(view.plugin.data.settings, force);
    if (!cat.agents.length) {
      new Notice(cat.error ? `无法读取 Agent 名单：${cat.error}` : "没有可指派的 Agent");
      return;
    }
    new AgentPickModal(view.plugin.app, cat.agents, INSERT).open();
  };
  mentionBtn.addEventListener("click", () => void openAgentPicker());
  ta.addEventListener("keydown", (e: KeyboardEvent) => {
    // 只在"行首或空白后的 @"弹出：邮箱 a@b、双链 [[a@b]] 里的 @ 不触发
    if (e.key !== "@" || e.ctrlKey || e.metaKey || e.altKey) return;
    const pos = ta.selectionStart ?? 0;
    const before = ta.value.slice(0, pos);
    if (before && !/\s$/.test(before)) return;
    e.preventDefault();
    void openAgentPicker();
  });
  atBtn.addEventListener("click", () => {
    new NotePickModal(view.plugin.app, (f: TFile) => {
      const pos = ta.selectionStart ?? ta.value.length;
      ta.setRangeText(`[[${f.path}]]`, pos, pos, "end");
      ta.focus();
    }).open();
  });
  // W3：编辑器「发送到工作台」的选区注入（一次性消费）
  if (wbPrefill) {
    ta.value = wbPrefill;
    wbPrefill = null;
    ta.focus();
  }
  wbInput = ta;
  for (const b of SKILL_BUTTONS) {
    const btn = skillRow.createEl("button", { cls: "ark-skill-btn",
      text: `${b.icon} ${b.label}`, attr: { title: b.prompt } });
    btn.addEventListener("click", () => {
      ta.value = b.prompt;
      ta.focus();
    });
  }
  const catalogBtn = skillRow.createEl("button", { cls: "ark-skill-btn ark-skill-catalog", text: "技能目录",
    attr: { title: "浏览全部技能并填入试用指令" } });
  catalogBtn.addEventListener("click", () => {
    new SkillPickModal(view.plugin.app, (skill) => {
      ta.value = skill.prompt;
      ta.focus();
    }).open();
  });
  const researchBtn = skillRow.createEl("button", { cls: "ark-skill-btn ark-research-btn", text: "研究简报",
    attr: { title: "填写问题、用途、范围和指定资料，生成只读研究 prompt" } });
  researchBtn.addEventListener("click", () => {
    void (async () => {
      const targetSessionId = currentSession(view).id;
      const question = await inputDialog(view.plugin, { title: "研究问题", placeholder: "你要比较、判断或解释什么？", multiline: true });
      if (!question?.trim()) return;
      const purpose = await inputDialog(view.plugin, { title: "研究用途", placeholder: "技术选型、决策记录、学习准备……" });
      const scope = await inputDialog(view.plugin, { title: "资料范围", placeholder: "例如 wiki/ 或当前 Project scope" });
      const refsRaw = await inputDialog(view.plugin, { title: "指定 Markdown 资料（可选）", placeholder: "多个路径用逗号或换行分隔" });
      const refs = parseResearchRefs(refsRaw || "");
      const invalidCount = (refsRaw || "").split(/[\n,，、;；]+/).map((item) => item.trim()).filter(Boolean).length - refs.length;
      if (invalidCount > 0) new Notice(`已忽略 ${invalidCount} 个非安全 Markdown 路径`);
      const brief: ResearchBrief = {
        question: question.trim(), purpose: purpose?.trim(), scope: scope?.trim(), specifiedRefs: refs,
      };
      wbResearchBriefs.set(targetSessionId, brief);
      ta.value = buildResearchPrompt(brief);
      ta.focus();
    })();
  });

  function renderAll() {
    const sess = currentSession(view);
    renderSide(view, side, sess, renderAll);
    renderFlowInto(flow, view, sess);
    renderCtx(view, ctx, (msg) => void send(msg), sess);
    void refreshUsage(view, ctx);
    const running = agentLive.has(sess.id);
    sendBtn.style.display = running ? "none" : "";
    stopBtn.style.display = running ? "" : "none";
  }

  function syncLiveView(sessionId: string) {
    if (!grid.isConnected) return;
    const sess = currentSession(view);
    if (sessionId !== sess.id) return;
    const live = agentLive.get(sessionId);
    if (!live) {
      renderAll();
      return;
    }
    sendBtn.style.display = "none";
    stopBtn.style.display = "";
    let textEl = flow.querySelector<HTMLElement>(".agent-wb-livetext");
    if (!textEl) {
      textEl = renderFlowInto(flow, view, sess);
      return;
    }
    textEl.textContent = live.text;
    const wrap = textEl.parentElement;
    if (!wrap) return;
    let chips = wrap.querySelector<HTMLElement>(".agent-wb-chips");
    if (!chips) chips = wrap.createDiv({ cls: "agent-wb-chips" });
    const rendered = Array.from(chips.children).filter((el) => el.classList.contains("agent-wb-chip")).length;
    for (const tool of live.tools.slice(rendered)) {
      chips.createSpan({
        cls: `agent-wb-chip ${tool.phase === "start" ? "run" : "done"}`,
        text: tool.phase === "start" ? `🔧 ${tool.name}…` : `${tool.name} ✓`,
      });
    }
    let stat = wrap.querySelector<HTMLElement>(".agent-wb-status");
    if (!stat) stat = wrap.createDiv({ cls: "agent-wb-status" });
    stat.setText(`运行中 ${live.elapsed}s`);
    flow.scrollTop = flow.scrollHeight;
  }

  function scheduleRender(sessionId: string) {
    if (renderQueued || !grid.isConnected) return;
    renderQueued = true;
    window.setTimeout(() => {
      renderQueued = false;
      syncLiveView(sessionId);
    }, 100);
  }

  const unsubscribeLive = subscribeLive((sessionId) => {
    if (!grid.isConnected) return;
    if (sessionId !== currentSession(view).id) return;
    scheduleRender(sessionId);
  });
  wbLiveSubscriptions.set(mount, unsubscribeLive);
  const detachObserver = new MutationObserver(() => {
    if (grid.isConnected) return;
    unsubscribeLive();
    detachObserver.disconnect();
  });
  detachObserver.observe(mount, { childList: true });

  /** W2：向 serve 查询上下文用量（与 loop 同口径 estimate），节流 2s；旧版 serve 静默跳过 */
  async function refreshUsage(v: SpaceOSView, ctxEl: HTMLElement) {
    const s = v.plugin.data.settings;
    if (s.agentProvider !== "agentlab") return;
    const sess = currentSession(v);
    const now = Date.now();
    const sessionChanged = wbUsageSess !== sess.id;
    if (!sessionChanged && now - wbUsageAt < 2000) return;  // 同会话节流；切会话强制刷新
    wbUsageAt = now;
    wbUsageSess = sess.id;
    const ep = agentEndpoint(s);
    try {
      const r: any = await requestUrl({
        url: `${ep.origin}/v1/context-status`, method: "POST",
        contentType: "application/json",
        headers: ep.token ? { Authorization: `Bearer ${ep.token}` } : {},
         body: JSON.stringify({
           messages: currentSession(v).messages,
           project_id: currentSession(v).projectId || "",
         }),
      });
      const text = typeof r.text === "function" ? await r.text() : String(r.text ?? "");
      const data = JSON.parse(text);
      if (data?.ok) {
        wbUsage = data;
        renderCtx(v, ctxEl, (msg: string) => void send(msg), sess);
      }
    } catch { /* 旧版 serve 无此端点 → 静默 */ }
  }

  async function send(text: string) {
    if (!text) return;
    const sess = currentSession(view);
    if (agentLive.has(sess.id)) { new Notice("当前会话有在途任务，请先停止或等待完成"); return; }
    const s = view.plugin.data.settings;
    if (s.agentProvider !== "agentlab" && s.agentProvider !== "hermes") {
      new Notice("当前 Agent 内核为直连模式，工作台仅支持 agentlab / Hermes。请到设置切换。");
      return;
    }
    // P2-2/F5-019：@ 多 Agent 指派——名单命中才转 multi；mention 从正文剥离（决策 D3），
    // 因此会话历史里不留路由语法，答者只收到裸问题；未知名原样保留为普通文本。
    let multiNames: string[] = [];
    let outgoing = text;
    if (/@[A-Za-z0-9_.-]/.test(text)) {
      const cat = await getAgentCatalog(s);
      const parsed = parseMentions(text, cat.agents.map((a) => a.name));
      if (parsed.unknown.length) {
        new Notice(`未识别的 Agent：${parsed.unknown.join("、")}（按普通文本发送）`);
      }
      if (parsed.multi.length) {
        multiNames = parsed.multi;
        outgoing = parsed.cleaned || text;
        if (multiNames.length > cat.maxParallelConsults) {
          new Notice(`最多并行 ${cat.maxParallelConsults} 支，超出的会被服务端跳过并计入 skipped`);
        }
      }
    }
    sess.messages.push({ role: "user", content: outgoing });
    if (sess.title === "新会话") sess.title = outgoing.slice(0, 24);
    commitIfNew(view, sess);
    touchSession(view.plugin, sess);
    await view.plugin.savePluginData();
    ta.value = "";
    // OPT-116 三期：先注册 live 再切按钮——setRunning 以 agentLive.has() 判运行中，
    // 此前注册晚于调用，stopBtn 整个运行期间从不显示（发送键照旧）
    const abr = new AbortController();
    agentLive.set(sess.id, {
      text: "", abort: abr, started: Date.now(), elapsed: 0,
      tools: [], status: "running",
    });
    notifyLive(sess.id);
    setRunning(true, sess);
    renderAll();

    const liveTextEl = renderFlowInto(flow, view, sess);
    const liveWrap = liveTextEl?.parentElement ?? null;
    const chips = liveWrap?.querySelector<HTMLElement>(".agent-wb-chips") ?? null;
    const liveStat = liveWrap?.querySelector<HTMLElement>(".agent-wb-status") ?? null;
    const approvalCards = new Map<string, HTMLElement>();
    const toolOutputs: ArtifactToolOutput[] = [];
    // P2-2/F5-019：分支进度卡（response.branch.*）+ 汇总行（response.multi.summary）
    const branches = new Map<string, BranchProgress>();
    const branchBox = multiNames.length && liveWrap?.isConnected
      ? liveWrap.createDiv({ cls: "agent-wb-branches" }) : null;
    const branchStatus = branchBox ? branchBox.createDiv({ cls: "agent-wb-branch-status" }) : null;
    const paintBranches = () => {
      if (!branchBox || !branchBox.isConnected) return;
      const list = branchBox.querySelector(".agent-wb-branch-list") as HTMLElement | null;
      const host = list ?? branchBox.createDiv({ cls: "agent-wb-branch-list" });
      host.empty();
      branches.forEach((b) => {
        const row = host.createDiv({ cls: `agent-wb-branch ${b.status}` });
        row.createSpan({ cls: "agent-wb-branch-name", text: `@${b.agent}` });
        row.createSpan({ cls: "agent-wb-branch-kind", text: b.kind === "internal" ? "内置" : "外部" });
        const tail = b.status === "running" ? "运行中…"
          : b.status === "ok"
            ? `完成 ${b.elapsed ?? 0}s${b.chars ? ` · ${b.chars} 字` : ""}${b.truncated ? " · 已截断并归档" : ""}`
            : `失败：${b.error ?? "未知错误"}`;
        row.createSpan({ cls: "agent-wb-branch-tail", text: tail });
        if (b.status === "ok" && b.ref) {
          row.createSpan({ cls: "agent-wb-branch-ref", text: `召回 ${b.ref}`,
            attr: { title: "被截断的全文已归档到会话区段，可让 Agent 经 rag_retrieve 召回" } });
        }
      });
      flow.scrollTop = flow.scrollHeight;
    };
    const startedAt = Date.now();
    const tick = window.setInterval(() => {
      const elapsed = Math.floor((Date.now() - startedAt) / 1000);
      updateLive(sess.id, { elapsed });
      if (liveStat) liveStat.setText(`运行中 ${elapsed}s`);
    }, 1000);
    // W2：completed 历史快照 → 替换本地历史（压缩跨轮持久）
    const done = { hist: null as { role: string; content: string }[] | null };  // 回调赋值：用对象属性避免 tsc 跨闭包窄化成 never
    // 在途流锚点若被用户切走（重渲染替换 DOM），放弃直写 DOM——文本仍在 live 注册表累积，
    // 完成后统一 renderAll 重绘；与 CRT 的 liveSessId 丢弃语义一致。
    const anchorAlive = () => agentLive.get(sess.id) != null && liveTextEl?.isConnected;
    const paint = () => {
      if (!anchorAlive()) return;
      liveTextEl!.textContent = agentLive.get(sess.id)?.text ?? "";
      flow.scrollTop = flow.scrollHeight;
    };

    let researchToFinalize: ResearchBrief | null = null;
    try {
      const reply = await agentChat(s, sess.messages, {
        onText: (d: string) => {
          const lv = agentLive.get(sess.id);
          if (lv) {
            lv.text += d;
            notifyLive(sess.id);
          }
          paint();
        },
        onTool: (name: string, phase: "start" | "done", detail?: string) => {
          const lv = agentLive.get(sess.id);
          if (lv) {
            lv.tools.push({ name, phase, output: detail });
            notifyLive(sess.id);
          }
          if (phase === "done" && detail) toolOutputs.push({ name, output: detail });
          if (!chips || !anchorAlive()) return;
          chips.createSpan({ cls: `agent-wb-chip ${phase === "start" ? "run" : "done"}`,
            text: phase === "start" ? `🔧 ${name}…` : `${name} ✓` });
          flow.scrollTop = flow.scrollHeight;
        },
        onStatus: (st: { elapsed: number }) => {
          updateLive(sess.id, { elapsed: st.elapsed });
          if (liveStat && anchorAlive()) liveStat.setText(`运行中 ${st.elapsed}s`);
        },
        onApproval: (event: { phase: "requested" | "resolved"; approval: any }) => {
          const a = event.approval || {};
          const id = String(a.approval_id || "");
          if (!id) return;
          if (id === "policy") {
            // OPT-222：策略自动放行必须用户可见——此前静默 return，write 工具在
            // allowlist 内自动执行时用户看不到任何提示，长任务期间误判"卡住"而反复重发
            // （真机 4 次重试 3 次断连白跑的根因之一）。
            if (anchorAlive() && chips) {
              chips.createSpan({
                cls: "agent-wb-chip done",
                text: `⚡ ${a.tool_name || "工具"} 已按策略自动放行`,
              });
              flow.scrollTop = flow.scrollHeight;
            }
            return;
          }
          if (event.phase === "requested" && chips && anchorAlive()) {
            const card = chips.createDiv({ cls: "agent-wb-approval" });
            card.createDiv({ cls: "agent-wb-approval-title", text: `需要确认：${a.tool_name || "危险工具"}` });
            card.createDiv({ cls: "agent-wb-approval-summary", text: String(a.summary || "该工具将执行受控操作") });
            const actions = card.createDiv({ cls: "agent-wb-approval-actions" });
            const status = card.createSpan({ cls: "agent-wb-approval-status", text: "等待确认" });
            const decide = (d: "allow" | "deny") => {
              void resolveApproval(s, id, d).then((result) => {
                if (!result) { new Notice("审批请求已失效或服务不可用"); return; }
                status.setText(result.status === "approved" ? "已允许" : "已拒绝");
                actions.querySelectorAll("button").forEach((b) => (b as HTMLButtonElement).disabled = true);
              });
            };
            actions.createEl("button", { text: "允许" }).addEventListener("click", () => decide("allow"));
            actions.createEl("button", { cls: "danger", text: "拒绝" }).addEventListener("click", () => decide("deny"));
            approvalCards.set(id, card);
            flow.scrollTop = flow.scrollHeight;
          } else if (event.phase === "resolved") {
            const card = approvalCards.get(id);
            if (card) {
              const st = card.querySelector(".agent-wb-approval-status");
              if (st) st.textContent = a.status === "approved" ? "已允许" : `已拒绝${a.reason ? `（${a.reason}）` : ""}`;
              card.querySelectorAll("button").forEach((b) => (b as HTMLButtonElement).disabled = true);
            }
          }
        },
        onCompleted: (info: { history?: { role: string; content: string }[] }) => {
          if (info.history && info.history.length) done.hist = info.history;
        },
        onBranch: (ev: Record<string, unknown>) => {
          if (!branchBox) return;
          const hit = branchEventToProgress(ev);
          if (!hit) return;
          const prev = branches.get(hit.agent);
          branches.set(hit.agent, { ...(prev ?? hit.patch), ...hit.patch });
          paintBranches();
        },
        onMultiSummary: (ev: Record<string, unknown>) => {
          if (!branchStatus) return;
          const ok = Number(ev.ok_count) || 0;
          const failed = Array.isArray(ev.failed) ? ev.failed.map(String) : [];
          const skipped = Array.isArray(ev.skipped) ? ev.skipped.map(String) : [];
          const parts = [`${ok}/${branches.size || ok + failed.length} 支成功`];
          if (failed.length) parts.push(`失败 ${failed.join("、")}`);
          if (skipped.length) parts.push(`超并发上限跳过 ${skipped.join("、")}`);
          if (ev.denied) parts.push("外部支未获批准，已降级单 Agent 直答");
          else if (ev.synthesized === false) parts.push("汇总不可用，已直接返回得分最高的一支");
          if (ev.synth_error) parts.push(`汇总出错：${String(ev.synth_error)}`);
          branchStatus.setText(parts.join(" · "));
        },
      }, abr.signal, { projectId: sess.projectId, multi: multiNames });
      window.clearInterval(tick);
      if (done.hist && done.hist.length) {
        // W2：服务端折叠后的历史快照替换本地历史（保留首条本地 system）
        // —— 快照已含最终回复，live 半截结算作废
        const keepSys = sess.messages.length && sess.messages[0].role === "system" ? [sess.messages[0]] : [];
        // 服务端历史行 role 为宽 string，但值域即本联合类型（own backend 保证）
        sess.messages = [...keepSys, ...done.hist] as CrtSession["messages"];
      } else {
        const lv = agentLive.get(sess.id);
        if (lv?.settled && lv.settleIdx != null) sess.messages[lv.settleIdx].content = reply;
        else sess.messages.push({ role: "assistant", content: reply });
      }
      const detected = extractSessionArtifacts(sess, toolOutputs);
      if (detected.length) {
        upsertArtifacts(view.plugin, validateArtifactPaths(view.plugin, detected));
        if (sess.projectId) void persistProjectArtifactIndex(view.plugin, sess.projectId);
        void view.plugin.savePluginData();
      }
      researchToFinalize = wbResearchBriefs.get(sess.id) ?? null;
      if (researchToFinalize) wbResearchBriefs.delete(sess.id);
      // 完成路径也必须清理 live；否则工作台会一直显示运行中且后续发送被拦截。
      updateLive(sess.id, { status: "completed", elapsed: Math.floor((Date.now() - startedAt) / 1000) });
      removeLive(sess.id);
    } catch (e: any) {
      window.clearInterval(tick);
      if (e?.name === "AbortError") {
        // 必须先结算再删除 live；否则 settleLiveStream 找不到半截文本，停止后 UI 会把已流式内容清空。
        settleLiveStream(sess);  // 用户主动停止：半截回答结算落盘
        updateLive(sess.id, { status: "cancelled" });
        removeLive(sess.id);
        new Notice("已停止");
      } else if (e?.message?.includes("409") || e?.message?.includes("冲突")) {
        updateLive(sess.id, { status: "failed", error: String(e?.message ?? e) });
        removeLive(sess.id);
        // F5-009: 写入冲突处理（可能是并发执行或其他窗口修改会话）
        const choice = await conflictDialog(view.plugin, "Agent 写入冲突，可能是并发执行导致");
        if (choice === "retry") {
          // 重试（覆盖）：重新发送相同消息
          new Notice("正在重试...");
          return send(text);
        } else if (choice === "save-as") {
          // 另存为新会话：创建新会话并复制消息后重试
          const oldMsgs = [...sess.messages];
          const newSess = draftSession(view.plugin, sess.projectId || "");
          newSess.messages = oldMsgs;
          view.plugin.data.crtSessions.push(newSess);
          view.plugin.data.crtActiveId = newSess.id;
          wbSession = newSess;
          await view.plugin.savePluginData();
          renderAll();
          new Notice("已另存为新会话，正在重试...");
          return send(text);
        }
        // choice === "cancel": 用户取消，静默退出
      } else {
        updateLive(sess.id, { status: "failed", error: String(e?.message ?? e) });
        removeLive(sess.id);
        new Notice("agent 调用失败: " + String(e?.message ?? e));
      }
    }

    touchSession(view.plugin, sess);
    await view.plugin.savePluginData();
    setRunning(false, sess);
    renderAll();
    if (researchToFinalize) {
      try {
        await confirmResearchDraft(view.plugin, sess, researchToFinalize, toolOutputs);
      } catch (error: any) {
        new Notice(`研究草稿处理失败，未自动重试：${String(error?.message ?? error)}`, 6000);
      }
    }
  }

  function setRunning(on: boolean, sess: CrtSession) {
    const running = on && agentLive.has(sess.id);
    sendBtn.style.display = running ? "none" : "";
    stopBtn.style.display = running ? "" : "none";
  }

  function commitIfNew(v: SpaceOSView, session: CrtSession) {
    commitSession(v.plugin, session);
  }

  renderAll();
}

// ── 会话区段引用差异化展示（4.0 执行线 #8）────────────────────────────
// OPT-111 的 rag_retrieve 返回 ref 形如 session/<sid>#r<seq>（ranges.py 生成），被模型引用进正文后
// 原样渲染成无效链接。三形态（[[…|别名]] / [文字](…) / 裸引用）统一替换为不可点击的行内徽标，
// 其余 markdown 原样保留交给 MarkdownRenderer；徽标为行内 HTML（渲染器原生支持），样式内联
// （styles.css 未纳入本次改动面，留 class=ark-session-ref 供后续收编）。
const SESSION_REF_RE =
  /\[\[session\/([a-z0-9-]{4,40})#r(\d+)(?:\|[^\]]*)?\]\]|\[[^\]]*\]\(session\/([a-z0-9-]{4,40})#r(\d+)\)|(?<![\w:\/\-.])(session\/[a-z0-9-]{4,40}#r(\d+))/g;

/** 会话区段徽标：小圆角 / 淡底色 / 字号略小 */
function sessionRefBadge(seq: string): string {
  return `<span class="ark-session-ref" style="display:inline-block;border-radius:8px;`
    + `background:var(--background-modifier-hover,rgba(127,127,127,0.15));`
    + `padding:0 6px;font-size:0.85em;">🗓 会话区段 r${seq}</span>`;
}

/** 把文本中的 session/<sid>#r<seq> 引用替换为徽标（捕获组：wikilink 1/2、md 链接 3/4、裸引用 5/6） */
function rewriteSessionRefs(text: string): string {
  return text.replace(SESSION_REF_RE,
    (_m: string, s1: string, q1: string, s2: string, q2: string, bare: string, q3: string) =>
      sessionRefBadge(q1 ?? q2 ?? q3 ?? ""));
}

/** 渲染消息流；若该会话有在途流则重挂直播锚，返回直播文本元素（无在途流返回 null） */
function renderFlowInto(flow: HTMLElement, view: SpaceOSView, sess: CrtSession): HTMLElement | null {
  flow.empty();
  flow.addClass("agent-wb-flow");
  const lv = agentLive.get(sess.id);
  const msgs = sess.messages.filter((m: CrtSession["messages"][number]) => m.role !== "system");
  if (!msgs.length && !lv) {
    flow.createDiv({ cls: "agent-wb-empty",
      text: "开始对话。检索、写笔记、查资料——工具调用会显示为轨迹卡片；回答里的笔记引用可点击打开。" });
    return null;
  }
  for (const m of msgs) {
    const row = flow.createDiv({
      cls: `agent-wb-msg ${m.role === "user" ? "user" : m.role === "tool" ? "tool" : "ai"}`,
    });
    row.createDiv({ cls: "agent-wb-who",
      text: m.role === "user" ? "你" : m.role === "tool" ? `🔧 ${m.name || "tool"} 结果` : "Agent" });
    const body = row.createDiv({ cls: "agent-wb-body" });
    if (m.role === "user") body.setText(m.content);
    else if (m.role === "tool") {
      // W2：历史工具卡片持久化——默认折叠，点击展开前 2000 字
      const det = body.createEl("details", { cls: "agent-wb-toolcard" });
      det.createEl("summary", { text: `${(m.content || "").length} 字结果（点击展开）` });
      det.createDiv({ cls: "agent-wb-toolbody", text: (m.content || "").slice(0, 2000) });
    } else void MarkdownRenderer.render(view.plugin.app, rewriteSessionRefs(m.content || "（空）"), body, "", view);
  }
  if (!lv || lv.settled) return null;
  // 在途流：重挂直播锚（切 Tab 回来仍可见；CRT 发起的流同样在此显示）
  const live = flow.createDiv({ cls: "agent-wb-msg ai live" });
  live.createDiv({ cls: "agent-wb-who", text: "Agent" });
  const wrap = live.createDiv({ cls: "agent-wb-body" });
  const textEl = wrap.createDiv({ cls: "agent-wb-livetext", text: lv.text });
  if (lv.tools.length) {
    const chips = wrap.createDiv({ cls: "agent-wb-chips" });
    for (const tool of lv.tools) {
      chips.createSpan({
        cls: `agent-wb-chip ${tool.phase === "start" ? "run" : "done"}`,
        text: tool.phase === "start" ? `🔧 ${tool.name}…` : `${tool.name} ✓`,
      });
    }
  }
  wrap.createDiv({ cls: "agent-wb-status", text: `运行中 ${lv.elapsed}s` });
  flow.scrollTop = flow.scrollHeight;
  return textEl;
}

/** 左侧会话栏：顶部项目切换（全局 + 各项目），下面按项目分组的会话列表，底部新建/删除 */
function renderSide(view: SpaceOSView, side: HTMLElement, sess: CrtSession, rerender: () => void) {
  side.empty();
  const data = view.plugin.data;
  const pid = activePid(view);

  // —— 项目切换器 ＋ 一步新建（OPT-064 二期）：切上下文与历史视图；当前会话先结算 ——
  side.createDiv({ cls: "agent-wb-side-title", text: "项目" });
  const projselRow = side.createDiv({ cls: "agent-wb-projsel" });
  const projSel = projselRow.createEl("select", { cls: "agent-wb-btn",
    attr: { title: "切换项目：历史与上下文随之隔离" } });
  const gOpt = document.createElement("option");
  gOpt.value = "";
  gOpt.textContent = "🌐 全局";
  if (!pid) gOpt.selected = true;
  projSel.appendChild(gOpt);
  for (const p of (data.settings.projects || []) as ProjectInfo[]) {
    const o = document.createElement("option");
    o.value = p.id;
    o.textContent = `📁 ${p.name}`;
    if (p.id === pid) o.selected = true;
    projSel.appendChild(o);
  }
  projSel.addEventListener("change", () => void onProjectChange(view, projSel.value, rerender));
  const newProjBtn = projselRow.createEl("button", { cls: "agent-wb-btn", text: "＋",
    attr: { title: "新建 Project（输入名称即建即激活）" } });
  newProjBtn.addEventListener("click", () => void onNewProject(view, rerender));
  const delProjBtn = projselRow.createEl("button", { cls: "agent-wb-btn danger", text: "🗑",
    attr: { title: "删除当前 Project（规则/背景目录删除，会话转为全局）" } });
  delProjBtn.disabled = !pid;
  delProjBtn.addEventListener("click", () => void onDeleteProject(view, pid, rerender));

  side.createDiv({ cls: "agent-wb-side-title", text: "会话" });
  const groups: { label: string; pid: string; items: CrtSession[] }[] = [
    { label: "🌐 全局", pid: "", items: [] },
    ...(data.settings.projects || []).map((p: ProjectInfo) => ({ label: `📁 ${p.name}`, pid: p.id, items: [] })),
  ];
  for (const s of data.crtSessions) {
    const grp = groups.find((x) => x.pid === (s.projectId || ""));
    (grp ?? groups[0]).items.push(s);
  }
  for (const grp of groups) {
    if (!grp.items.length) continue;
    side.createDiv({ cls: "agent-wb-group", text: grp.label });
    for (const it of grp.items.slice(0, 15)) {
      const item = side.createDiv({
        cls: `agent-wb-sess${it.id === sess.id ? " active" : ""}`,
        text: `${agentLive.has(it.id) ? "⏳ " : ""}${it.title}`,
        attr: { title: `${it.messages.length - 1} 条 · ${new Date(it.updatedAt).toLocaleString()}` },
      });
      item.addEventListener("click", () => {
        if (it.id === sess.id) return;
        settleLiveStream(sess);          // 切走先结算当前会话的半截回答
        wbSession = it;
        data.crtActiveId = it.id;
        void view.plugin.savePluginData();
        rerender();
      });
    }
  }
  const btns = side.createDiv({ cls: "agent-wb-side-btns" });
  btns.createEl("button", { cls: "agent-wb-btn", text: "＋ 新会话" }).addEventListener("click", () => {
    settleLiveStream(sess);
    wbSession = draftSession(view.plugin, activePid(view));
    data.crtActiveId = wbSession.id;
    void view.plugin.savePluginData();
    rerender();
  });
  if (sess.messages.length) {
    btns.createEl("button", { cls: "agent-wb-btn danger", text: "🗑 删除当前" }).addEventListener("click", () => {
      // 4.0 执行线 #8：serve 端同步删除（仅 agentlab；失败不阻断本地清理，仅 console.warn）
      const s = data.settings;
      if (s.agentProvider === "agentlab") {
        void sessionDelete(s, sess.id).then((r) => {
          if (!r) console.warn("[ARK] serve 端会话删除失败（本地已删除）：", sess.id);
        });
      }
      const list = data.crtSessions;
      const i = list.findIndex((x: CrtSession) => x.id === sess.id);
      if (i >= 0) list.splice(i, 1);
      agentLive.get(sess.id)?.abort.abort();
      removeLive(sess.id);
      if (data.crtActiveId === sess.id) data.crtActiveId = list[0]?.id ?? null;
      wbSession = null;
      void view.plugin.savePluginData();
      rerender();
    });
  }
}

/** 项目切换：结算当前会话 → 改激活项目 → wbSession 清空按新项目重新解析（该项目最近会话或新草稿） */
async function onProjectChange(view: SpaceOSView, v: string, rerender: () => void) {
  const s = view.plugin.data.settings;
  if ((s.activeProjectId || "").trim() === v.trim()) return;
  settleLiveStream(currentSession(view));
  s.activeProjectId = v;
  wbSession = null;
  if (v.trim()) await restoreProjectArtifacts(view.plugin, v.trim());
  await view.plugin.savePluginData();
  rerender();
  const pname = (s.projects || []).find((p: ProjectInfo) => p.id === v)?.name || "全局";
  new Notice(`已切换：${v ? pname : "全局"}（会话列表与上下文已切换）`);
}

/** 一步新建（OPT-064 二期）：输入名称 → 建目录+模板 → 自动激活 → 工作台立即刷新 */
async function onNewProject(view: SpaceOSView, rerender: () => void) {
  const name = (await inputDialog(view.plugin, { title: "Project 名称", placeholder: "如：视频入库流水线" }))?.trim();
  if (!name) return;
  const created = await createProject(view.plugin, name);
  if (!created) return;
  wbSession = null;  // 按新项目重新解析（createProject 已激活新项目 → 新草稿）
  rerender();
  new Notice(`Project 已创建并激活：${created.name}（ark/projects/${created.id}/）`);
}

/** P2-2/F5-019：@ 指派 agent 的选择器（数据源 = GET /v1/agents）。 */
class AgentPickModal extends FuzzySuggestModal<AgentInfo> {
  constructor(app: any, private agents: AgentInfo[], private onPick: (a: AgentInfo) => void) {
    super(app);
    this.setPlaceholder("选择要 @ 的 Agent（多 Agent 并行作答后由主 Agent 汇总）");
  }
  getItems(): AgentInfo[] { return this.agents; }
  getItemText(a: AgentInfo): string { return `${a.name} — ${a.description ?? a.kind}`; }
  onChooseItem(a: AgentInfo): void { this.onPick(a); }
}

/** 删除 Project：规则/背景目录与元数据删除；历史会话保留并解除项目归属，避免误删对话资产。 */
async function onDeleteProject(view: SpaceOSView, pid: string, rerender: () => void) {
  if (!pid) return;
  const busy = view.plugin.data.crtSessions.some((s: CrtSession) =>
    s.projectId === pid && agentLive.has(s.id));
  if (busy) {
    new Notice("当前 Project 有在途任务，请停止或等待完成后再删除");
    return;
  }
  const project = (view.plugin.data.settings.projects || []).find((p: ProjectInfo) => p.id === pid);
  if (!project) return;
  const ok = await confirmDialog(
    view.plugin,
    `确认删除项目“${project.name}”？\n\n将删除 ark/projects/${pid}/ 下的 AGENTS.md、project.md 及其他项目文件；历史会话保留并转为全局。`,
  );
  if (!ok) return;
  const { deleteProject } = await import("./projects");
  try {
    const detached = await deleteProject(view.plugin, pid);
    wbSession = null;
    rerender();
    new Notice(`项目已删除：${project.name}；保留并转为全局的会话 ${detached} 条`);
  } catch (e: any) {
    new Notice(`删除项目失败：${String(e?.message ?? e)}`);
  }
}

/** 右侧项目上下文条（W2）：项目/规则入口 + 用量可视化 + 手动压缩（onCompress 走对话管线） */
function renderCtx(view: SpaceOSView, ctx: HTMLElement,
                   onCompress: (msg: string) => void, sess: CrtSession) {
  ctx.empty();
  const s = view.plugin.data.settings;
  const pid = activePid(view);
  ctx.createDiv({ cls: "agent-wb-side-title", text: "上下文" });

  // —— 用量条（context-status 数据；旧版 serve 无端点则隐藏） ——
  if (wbUsage && s.agentProvider === "agentlab") {
    const pct = Math.min(100, Math.max(0, wbUsage.usage_pct));
    const tone = pct >= wbUsage.hard_trim_pct ? "danger" : pct >= wbUsage.nudge_pct ? "warn" : "ok";
    const usage = ctx.createDiv({ cls: "agent-wb-usage" });
    usage.createDiv({ cls: "agent-wb-usage-text",
      text: `历史消息 ${wbUsage.tokens} / ${wbUsage.budget} tok（${wbUsage.usage_pct}%）` });
    const track = usage.createDiv({ cls: `agent-wb-usage-track ${tone}` });
    track.createDiv({ cls: "agent-wb-usage-fill", attr: { style: `width:${pct}%` } });
    usage.createDiv({ cls: "agent-wb-usage-hint",
      text: `提示线 ${wbUsage.nudge_pct}% · 硬截断 ${wbUsage.hard_trim_pct}% · 仅历史估算；运行时另含 system、项目规则、记忆和工具结果` });
    const cbtn = ctx.createEl("button", { cls: "agent-wb-btn", text: "🗜 压缩上下文" });
    cbtn.addEventListener("click", () => {
      const sess = currentSession(view);
      if (agentLive.has(sess.id)) { new Notice("有在途任务，结束后再压缩"); return; }
      // 走正常对话管线：模型调用 compress_context 工具 → completed 历史快照回写本地（压缩持久）
      onCompress("[系统请求] 请调用 compress_context 工具压缩上下文，完成后用一句话确认。");
    });
  } else if (s.agentProvider === "agentlab") {
    ctx.createDiv({ cls: "agent-wb-muted", text: "上下文用量：发起一次对话后显示（需 agentlab serve ≥ 今日版本）。" });
  }

  renderArtifactSection(view, ctx, sess);

  // —— 项目规则入口 ——
  if (!pid) {
    ctx.createDiv({ cls: "agent-wb-muted",
      text: "全局模式（未选项目）。左侧顶部下拉可切换或建项目（＋按钮一步到位）。" });
    return;
  }
  const pname = (s.projects || []).find((x: ProjectInfo) => x.id === pid)?.name || pid;
  ctx.createDiv({ cls: "agent-wb-muted", text: `📁 ${pname}` });
  for (const f of ["AGENTS.md", "project.md"]) {
    const link = ctx.createDiv({ cls: "agent-wb-link crt-link", text: `打开 ${f}` });
    link.addEventListener("click", () => {
      void view.plugin.app.workspace.openLinkText(`ark/projects/${pid}/${f}`, "", false);
    });
  }
  ctx.createDiv({ cls: "agent-wb-muted", text: "规则优先于全局规范。" });
}

class SkillPickModal extends FuzzySuggestModal<ArkSkill> {
  constructor(app: any, private onPick: (skill: ArkSkill) => void) {
    super(app);
    this.setPlaceholder("选择技能（只填入指令，不会自动发送）");
  }
  getItems() { return skillCatalog(); }
  getItemText(skill: ArkSkill): string { return `${skill.icon} ${skill.label}`; }
  onChooseItem(skill: ArkSkill): void { this.onPick(skill); }
}

/** F3：展示本轮已识别的成果文件，并将已校验路径直接打开。 */
function renderArtifactSection(view: SpaceOSView, ctx: HTMLElement, sess: CrtSession): void {
  const artifacts = artifactsForSession(view.plugin, sess.id);
  const title = ctx.createDiv({ cls: "agent-wb-side-title", text: "成果" });
  if (!artifacts.length) {
    title.insertAdjacentElement("afterend", ctx.createDiv({
      cls: "agent-wb-muted", text: "本轮尚未识别到 Agent 创建或修改的笔记。",
    }));
    return;
  }
  const list = ctx.createDiv({ cls: "agent-wb-artifacts" });
  artifacts.forEach((artifact) => {
    const card = list.createDiv({ cls: `agent-wb-artifact ${artifact.status}` });
    const head = card.createDiv({ cls: "agent-wb-artifact-head" });
    head.createSpan({ cls: "agent-wb-artifact-kind", text: artifact.kind === "note" ? "笔记" : artifact.kind });
    head.createSpan({ cls: "agent-wb-artifact-status", text: artifact.status === "validated" ? "已校验" : artifact.status === "missing" ? "路径不存在" : artifact.status });
    if (artifact.version && artifact.version > 1) {
      head.createSpan({ cls: "agent-wb-artifact-version", text: `v${artifact.version}` });
    }
    const link = card.createDiv({ cls: "agent-wb-link", text: artifact.path, attr: { title: "打开成果文件" } });
    link.addEventListener("click", () => {
      const file = view.plugin.app.vault.getAbstractFileByPath(artifact.path);
      if (file) void view.plugin.app.workspace.getLeaf("tab").openFile(file as any);
      else new Notice(`成果文件不存在：${artifact.path}`);
    });
    if (artifact.source_refs.length) {
      card.createDiv({ cls: "agent-wb-artifact-sources", text: `来源 ${artifact.source_refs.length} 条` });
    }
    if (artifact.previous_path) {
      card.createDiv({ cls: "agent-wb-artifact-sources", text: `由 ${artifact.previous_path} 重命名` });
    }
  });
}
