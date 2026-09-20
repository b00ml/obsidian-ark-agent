import { Modal, Setting, ButtonComponent, Notice } from "obsidian";
import type ArkOSPlugin from "./main";
import type { ArkSettings, MailAccountConfig, ReminderPreset } from "./settings";
import type { DbCategory } from "./types";
import { DEFAULT_REMINDER_PRESETS, DEFAULT_SETTINGS, aiProviderDefaults, normalizeSettings } from "./settings";
import type { FeedConfig } from "./settings";
import { probeHermes } from "./ai";
import { getSkin } from "./skins";
import { selectChannel } from "./ai-asst";
import { diagnoseRagSettings, ragStrategyLabel } from "./rag-diagnostics";

const AI_PROVIDERS: Record<string, string> = {
  "": "未启用",
  deepseek: "DeepSeek",
  openai: "OpenAI",
  doubao: "豆包（火山引擎）",
  ollama: "Ollama（本地）",
  custom: "自定义 OpenAI 兼容",
};

type SettingsTab = "basic" | "ai" | "rag" | "comm";

/** 打开设置中心 */
export function openSettings(plugin: ArkOSPlugin): void {
  new SettingsModal(plugin).open();
}

export class SettingsModal extends Modal {
  private plugin: ArkOSPlugin;
  private s: ArkSettings;
  private tab: SettingsTab = "basic";
  private content!: HTMLElement;

  constructor(plugin: ArkOSPlugin) {
    super(plugin.app);
    this.plugin = plugin;
    // 设置表单只编辑草稿；点击取消必须丢弃所有未保存修改。
    this.s = normalizeSettings(structuredClone(plugin.data.settings));
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.addClass("sos-settings-modal");

    contentEl.createDiv({ cls: "sos-settings-title", text: getSkin(this.s.skin).hints.settingsTitle });

    const tabbar = contentEl.createDiv({ cls: "sos-settings-tabs" });
    const defs: { k: SettingsTab; label: string }[] = [
      { k: "basic", label: "基础" },
      { k: "ai", label: "AI 助手" },
      { k: "rag", label: "检索" },
      { k: "comm", label: "通信" },
    ];
    defs.forEach((d) => {
      const btn = tabbar.createEl("button", { cls: `sos-tab-btn${this.tab === d.k ? " active" : ""}`, text: d.label });
      btn.addEventListener("click", () => {
        this.tab = d.k;
        this.render();
      });
    });

    this.content = contentEl.createDiv({ cls: "sos-settings-body" });
    this.render();

    const foot = contentEl.createDiv({ cls: "sos-settings-foot" });
    new ButtonComponent(foot)
      .setButtonText("取消")
      .onClick(() => this.close());
    new ButtonComponent(foot)
      .setButtonText("保存")
      .setCta()
      .onClick(async () => {
        const next = normalizeSettings(this.s);
        next.backgroundMode = next.backgroundImagePath
          ? "image"
          : next.backgroundVideoPath
            ? "video"
            : "image";
        this.plugin.data.settings = next;
        await this.plugin.savePluginData();
        this.close();
      });
  }

  onClose() {
    this.contentEl.empty();
  }

  private render() {
    this.content.empty();
    if (this.tab === "basic") this.renderBasic();
    else if (this.tab === "ai") this.renderAI();
    else if (this.tab === "rag") this.renderRag();
    else this.renderComm();
  }

  // ===== 基础 =====
  private renderBasic() {
    const { contentEl } = this;
    this.section("基础");
    new Setting(this.content)
      .setName("产品界面")
      .setDesc("Ark 工作台 · 中性、清晰的知识管理界面");
    new Setting(this.content)
      .setName(getSkin(this.s.skin).hints.captainLabel)
      .addText((t) => { t.setValue(this.s.captainName).onChange((v) => { this.s.captainName = v || "用户"; }); });

    this.section("文件夹配置");
    const folders: { key: keyof ArkSettings; label: string }[] = [
      { key: "normalLogFolder", label: "工作记录文件夹" },
      { key: "faultLogFolder", label: "问题记录文件夹" },
      { key: "databaseFolder", label: "数据库文件夹" },
      { key: "reportFolder", label: "日报周报文件夹" },
      { key: "healthFolder", label: "健康记录文件夹" },
      { key: "todoFolder", label: "任务文件夹" },
      { key: "ideaFolder", label: "灵感文件夹" },
      { key: "drawingFolder", label: "随笔文件夹" },
      { key: "battleReportFolder", label: "结果记录文件夹" },
      { key: "weaponFolder", label: "训练模板文件夹" },
      { key: "learningFolder", label: "学习笔记文件夹" },
      { key: "quizFolder", label: "学习问答文件夹" },
      { key: "feedInboxFolder", label: "订阅简报文件夹" },
    ];
    folders.forEach((f) => {
      new Setting(this.content)
        .setName(f.label)
        .addText((t) => t.setPlaceholder(String(DEFAULT_SETTINGS[f.key])).setValue(String(this.s[f.key] ?? "")).onChange((v) => { (this.s[f.key] as string) = v; }));
    });

    this.section("扫描黑名单");
    const blacklistCtx = this.content.createDiv({ cls: "sos-bl-list" });
    const drawBlacklist = () => {
      blacklistCtx.empty();
      this.s.scanBlacklist.forEach((item, i) => {
        const row = blacklistCtx.createDiv({ cls: "sos-bl-row" });
        const inp = row.createEl("input", { attr: { type: "text", placeholder: "例: 02-DB/归档/**", value: item } });
        inp.addEventListener("change", () => { this.s.scanBlacklist[i] = inp.value; });
        const del = row.createEl("button", { cls: "sos-mini", text: "✕" });
        del.addEventListener("click", () => { this.s.scanBlacklist.splice(i, 1); drawBlacklist(); });
      });
      const add = blacklistCtx.createEl("button", { cls: "sos-mini", text: "+ 添加黑名单规则" });
      add.addEventListener("click", () => { this.s.scanBlacklist.push(""); drawBlacklist(); });
    };
    drawBlacklist();

    this.section("数据库分类");
    const catCtx = this.content.createDiv({ cls: "sos-cat-list" });
    const drawCats = () => {
      catCtx.empty();
      this.s.databaseCategories.forEach((c, i) => {
        const row = catCtx.createDiv({ cls: "sos-bl-row" });
        const color = row.createEl("input", { attr: { type: "color", value: c.color || "#00c8ff" } });
        color.addEventListener("change", () => { this.s.databaseCategories[i].color = color.value; });
        const name = row.createEl("input", { attr: { type: "text", value: c.name } });
        name.addEventListener("change", () => { this.s.databaseCategories[i].name = name.value; });
        const del = row.createEl("button", { cls: "sos-mini", text: "✕" });
        del.addEventListener("click", () => { this.s.databaseCategories.splice(i, 1); drawCats(); });
      });
      const add = catCtx.createEl("button", { cls: "sos-mini", text: "+ 添加分类" });
      add.addEventListener("click", () => {
        const c: DbCategory = { id: "cat_" + Date.now().toString(36), name: "新分类", color: "#00c8ff" };
        this.s.databaseCategories.push(c);
        drawCats();
      });
    };
    drawCats();

    this.section("学习问答");
    new Setting(this.content).setName("题目数量").addText((t) => { t.inputEl.setAttribute("type", "number"); t.inputEl.setAttribute("min", "1"); t.inputEl.setAttribute("max", "10"); t.setValue(String(this.s.quizQuestionCount || 5)).onChange((v) => { this.s.quizQuestionCount = Number(v) || 5; }); });
    new Setting(this.content).setName("题目难度").addDropdown((dd) => {
      ["简单", "中等", "偏难"].forEach((d) => dd.addOption(d, d));
      dd.setValue(this.s.quizDifficulty || "中等").onChange((v) => { this.s.quizDifficulty = v; });
    });

    this.section("信息订阅");
    this.renderFeeds();

    void contentEl;
  }

  private renderFeeds() {
    const ctx = this.content.createDiv({ cls: "sos-mail-list" });
    const draw = () => {
      ctx.empty();
      if (this.s.feeds.length === 0) {
        ctx.createDiv({ cls: "sos-hint", text: "暂无订阅源。示例：新增一个 source=rss、name=每日AI、target=某 RSS 地址。" });
      }
      this.s.feeds.forEach((f, i) => {
        const row = ctx.createDiv({ cls: "sos-mail-row" });
        row.createSpan({ cls: "sos-mail-name", text: `${f.enabled === false ? "⏸ " : ""}${f.name ?? "未命名"} [${f.source}] ${f.target}` });
        if (f.lastError) row.createSpan({ cls: "sos-hint", text: ` ⚠ ${f.lastError}` });
        const edit = row.createEl("button", { cls: "sos-mini", text: "编辑" });
        edit.addEventListener("click", () => this.editFeed(i));
        const del = row.createEl("button", { cls: "sos-mini", text: "✕" });
        del.addEventListener("click", () => { this.s.feeds.splice(i, 1); draw(); });
      });
      const add = ctx.createEl("button", { cls: "sos-mini", text: "+ 添加订阅源" });
      add.addEventListener("click", () => {
        this.s.feeds.push({ id: "feed_" + Date.now().toString(36), name: "新订阅", source: "rss", target: "", keywords: [], weight: 1, enabled: true });
        this.editFeed(this.s.feeds.length - 1);
        draw();
      });
    };
    draw();
  }

  private editFeed(i: number) {
    const f = this.s.feeds[i];
    const modal = new Modal(this.app);
    modal.titleEl.setText(`编辑订阅源`);
    const body = modal.contentEl;
    body.addClass("sos-mail-edit");
    new Setting(body).setName("名称").addText((t) => t.setValue(f.name).onChange((v) => { f.name = v; }));
    new Setting(body).setName("类型").addDropdown((dd) => {
      (["rss", "url", "keyword_search", "bilibili", "wechat"] as const).forEach((s) => dd.addOption(s, s));
      dd.setValue(f.source).onChange((v) => { f.source = v as FeedConfig["source"]; });
    });
    const hint = body.createDiv({ cls: "sos-hint", text: "rss=订阅地址；url=单条直达链接；keyword_search=全网关键词；bilibili/wechat 请走既有收件箱采集" });
    void hint;
    new Setting(body).setName("目标地址").addText((t) => t.setPlaceholder("https://…").setValue(f.target).onChange((v) => { f.target = v; }));
    new Setting(body).setName("过滤关键词").setDesc("逗号分隔；留空=全部保留").addText((t) => t.setValue((f.keywords || []).join(",")).onChange((v) => { f.keywords = v.split(/[,，]/).map((s) => s.trim()).filter(Boolean); }));
    new Setting(body).setName("权重").addText((t) => { t.inputEl.setAttribute("title", "越大越靠前"); t.setValue(String(f.weight ?? 1)).onChange((v) => { f.weight = Number(v) || 1; }); });
    new Setting(body).setName("启用").addToggle((t) => t.setValue(f.enabled !== false).onChange((v) => { f.enabled = v; }));
    new Setting(body).addButton((b) => b.setButtonText("完成").setCta().onClick(() => modal.close()));
    modal.open();
  }

  // ===== AI 助手 =====
  private renderAI() {
    this.section("AI 助手");
    new Setting(this.content)
      .setName("AI 服务")
      .setDesc("选择供应商，或本地 Ollama")
      .addDropdown((dd) => {
        Object.entries(AI_PROVIDERS).forEach(([k, v]) => dd.addOption(k, v));
        dd.setValue(this.s.aiProvider || "").onChange(async (v) => {
          this.s.aiProvider = v;
          const d = aiProviderDefaults(v);
          if (v !== "custom") {
            if (d.url) this.s.aiApiUrl = d.url;
            if (d.model) this.s.aiModel = d.model;
          }
          /* 重渲染该节较复杂，仅更新提示 */
          providerHint.setText(d.url ? `预设地址：${d.url} ｜ 模型：${d.model}` : "自定义：请自填 URL 与模型");
        });
      });
    const providerHint = this.content.createDiv({ cls: "sos-hint", text: "" });

    // §7 深度集成说明 + 通道状态只读指示器（设计文档 DESIGN-AI-DEEP-INTEGRATION.md）
    new Setting(this.content)
      .setName("深度集成说明")
      .setDesc("编辑器选中润色 / 灵感优化走快通道（直连）；今日待办由插件读库、AI 拆解。未配直连 key 时自动兜底 Hermes。");
    const stRow = new Setting(this.content).setName("通道状态");
    const stText = stRow.descEl.createDiv({ cls: "sos-hint" });
    const refresh = (hermesAlive?: { ok: boolean; version?: string; error?: string }) => {
      const ch = (k: "quick" | "contextual") => selectChannel(this.s, k) ?? "无";
      if (hermesAlive) {
        stText.setText(`直连: ${this.s.aiApiKey ? "✅" : "❌"} · Hermes: ${this.s.hermesToken ? (hermesAlive.ok ? `✅ 在线 v${hermesAlive.version ?? ""}` : `❌ 离线(${hermesAlive.error ?? "?"})`) : "未配置"} · 当前: quick→${ch("quick")} · contextual→${ch("contextual")}`);
      } else {
        stText.setText(`直连: ${this.s.aiApiKey ? "✅" : "❌"} · Hermes: ${this.s.hermesToken ? "已配置(点刷新探测在线)" : "❌"} · 当前: quick→${ch("quick")} · contextual→${ch("contextual")}`);
      }
    };
    refresh();
    stRow.addButton((b) => b.setButtonText("刷新").onClick(async () => {
      const p = await probeHermes(this.s);
      refresh(p);
    }));

    new Setting(this.content).setName("API 地址").addText((t) => t.setPlaceholder("https://api.deepseek.com/chat/completions").setValue(this.s.aiApiUrl).onChange((v) => { this.s.aiApiUrl = v; }));
    new Setting(this.content).setName("API Key").addText((t) => t.setPlaceholder("sk-...").setValue(this.s.aiApiKey).onChange((v) => { this.s.aiApiKey = v; }));
    new Setting(this.content).setName("模型名称").addText((t) => t.setPlaceholder("deepseek-chat").setValue(this.s.aiModel).onChange((v) => { this.s.aiModel = v; }));
    new Setting(this.content).setName("温度")
      .addSlider((sl) => sl.setLimits(0, 1, 0.1).setValue(this.s.aiTemperature).onChange((v) => { this.s.aiTemperature = +v.toFixed(1); }));
    new Setting(this.content).setName("AI 对话保存路径").addText((t) => t.setValue(this.s.aiFolder).onChange((v) => { this.s.aiFolder = v; }));
    new Setting(this.content).setName("上下文长度 (tokens)").addText((t) => { t.inputEl.setAttribute("type", "number"); t.inputEl.setAttribute("min", "1024"); t.inputEl.setAttribute("max", "32768"); t.inputEl.setAttribute("step", "1024"); t.setValue(String(this.s.aiContextLength)).onChange((v) => { this.s.aiContextLength = Number(v) || 4096; }); });
    new Setting(this.content).setName("最大对话轮数").addText((t) => { t.inputEl.setAttribute("type", "number"); t.inputEl.setAttribute("min", "2"); t.inputEl.setAttribute("max", "50"); t.setValue(String(this.s.aiMaxRounds)).onChange((v) => { this.s.aiMaxRounds = Number(v) || 10; }); });
    new Setting(this.content).setName("回顾文件路径").setDesc("每日回顾 / 周报回顾 生成文件的目录").addText((t) => t.setPlaceholder("02-DB/回顾").setValue(this.s.reviewFolder || "02-DB/回顾").onChange((v) => { this.s.reviewFolder = v; }));

    this.section("Agent 内核");
    new Setting(this.content)
      .setName("Agent 模式")
      .setDesc("hermes = 外部常驻 Hermes Agent；agentlab = 自研 agentlab serve（Hermes 兼容 SSE）；ark 只做交互")
      .addDropdown((dd) => {
        dd.addOption("", "直连（现状，/chat/completions）");
        dd.addOption("hermes", "Hermes Agent（外部常驻 /v1/responses）");
        dd.addOption("agentlab", "Agentlab serve（自研 /v1/responses）");
        dd.setValue(this.s.agentProvider || "").onChange((v) => { this.s.agentProvider = v; });
      });
    new Setting(this.content).setName("Hermes 地址").setDesc("gateway API 基础地址，如 http://127.0.0.1:8642/v1").addText((t) => t.setPlaceholder("http://127.0.0.1:8642/v1").setValue(this.s.hermesUrl || "http://127.0.0.1:8642/v1").onChange((v) => { this.s.hermesUrl = v; }));
    new Setting(this.content).setName("Hermes Token").setDesc("即 Hermes .env 的 API_SERVER_KEY").addText((t) => t.setPlaceholder("API_SERVER_KEY").setValue(this.s.hermesToken || "").onChange((v) => { this.s.hermesToken = v; }));
    new Setting(this.content).setName("Hermes 模型").setDesc("profile 名，/v1/models 实测为 hermes-agent").addText((t) => t.setValue(this.s.hermesModel || "hermes-agent").onChange((v) => { this.s.hermesModel = v; }));
    new Setting(this.content).setName("Agentlab 地址").setDesc("自研 serve 基础地址，如 http://127.0.0.1:8643/v1").addText((t) => t.setPlaceholder("http://127.0.0.1:8643/v1").setValue(this.s.agentlabUrl || "http://127.0.0.1:8643/v1").onChange((v) => { this.s.agentlabUrl = v; }));
    new Setting(this.content).setName("Agentlab Token").setDesc("agentlab config.json 的 serve.token / 环境变量 AGENTLAB_SERVE_TOKEN").addText((t) => t.setPlaceholder("serve token").setValue(this.s.agentlabToken || "").onChange((v) => { this.s.agentlabToken = v; }));
    // P2-2/F5-021：审批策略（产品收敛设计里要求的"可调节"，此前只有后端 config 底座）
    new Setting(this.content)
      .setName("审批策略")
      .setDesc("风险分级：仅危险操作（含 @ 多 Agent 门禁）弹卡确认。全部允许：不再弹卡，"
        + "但 Bearer 鉴权、工具注册与 Vault 路径守卫仍然生效。保存后需重启 Agent 内核才生效。")
      .addDropdown((dd) => {
        dd.addOption("risk_based", "风险分级（默认，危险操作需确认）");
        dd.addOption("allow_all", "全部允许（自动放行已注册工具）");
        dd.setValue(this.s.approvalMode === "allow_all" ? "allow_all" : "risk_based")
          .onChange((v) => { this.s.approvalMode = v === "allow_all" ? "allow_all" : "risk_based"; });
      });
    new Setting(this.content).setName("Agentlab 模型").setDesc("agentlab 内 model 名，默认 agentlab-demo").addText((t) => t.setValue(this.s.agentlabModel || "agentlab-demo").onChange((v) => { this.s.agentlabModel = v; }));
    new Setting(this.content).setName("Agentlab Python").setDesc("「AI 助手」一键拉起 serve 的 python 可执行文件；留空自动探测项目 .venv").addText((t) => t.setPlaceholder("留空自动（项目 .venv）").setValue(this.s.agentlabExePath || "").onChange((v) => { this.s.agentlabExePath = v; }));
    new Setting(this.content).setName("Agentlab 目录").setDesc("serve 启动目录（须含 config/config.json）；留空使用当前项目的 agentlab 目录").addText((t) => t.setPlaceholder("agentlab").setValue(this.s.agentlabWorkdir || "").onChange((v) => { this.s.agentlabWorkdir = v; }));
    new Setting(this.content)
      .setName("Hermes Desktop 路径")
      .setDesc("点「AI 助手」优先拉起桌面 GUI（Hermes.exe），留空则回退 CRT 终端")
      .addText((t) => t.setPlaceholder("…\\release\\win-unpacked\\Hermes.exe").setValue(this.s.desktopExePath || "").onChange((v) => { this.s.desktopExePath = v; }));
    new Setting(this.content).setName("探测 Agent 内核").addButton((b) => b.setButtonText("探测").onClick(async () => {
      const r = await probeHermes(this.s);
      const label = this.s.agentProvider === "agentlab" ? "agentlab" : "Hermes";
      new Notice(r.ok ? `${label} 已连接 v${r.version ?? "?"}` : `${label} 未运行：${r.error ?? ""}`);
    }));
  }

  // ===== 知识库检索 =====
  private renderRag() {
    this.section("检索方式");
    const diagnostics = diagnoseRagSettings(this.s);
    const diag = this.content.createDiv({ cls: "sos-rag-diagnostics" });
    diag.createDiv({ cls: "sos-rag-diagnostics-title", text: "当前生效配置" });
    diag.createDiv({ cls: "sos-rag-diagnostics-line", text:
      `策略：${ragStrategyLabel(diagnostics.effectiveStrategy)} · 关键词：${diagnostics.lexicalEnabled ? "启用" : "关闭"} · 向量：${diagnostics.vectorEnabled ? "启用" : "关闭"}` });
    diag.createDiv({ cls: "sos-rag-diagnostics-line", text:
      `Provider：${diagnostics.providerConfigured ? "已配置地址" : "未配置地址"} · Key：${diagnostics.apiKeyConfigured ? "已填写" : "未填写"} · 全局门禁：${diagnostics.productionGate === "blocked" ? "未开放" : "仅手工灰度"}` });
    diagnostics.warnings.forEach((warning) => diag.createDiv({ cls: "sos-rag-diagnostics-warning", text: `提示：${warning}` }));
    new Setting(this.content)
      .setName("召回模式")
      .setDesc("关键词适合标题、BV 号和代码；向量适合同义表达；混合模式会同时召回并用 RRF 融合。")
      .addDropdown((dd) => {
        dd.addOption("keyword", "仅关键词（轻量，不调用 embedding）");
        dd.addOption("shadow", "关键词展示 + 向量观测（推荐先用）");
        dd.addOption("hybrid", "关键词 + 向量（RRF 融合）");
        dd.addOption("vector", "仅向量（实验）");
        dd.setValue(this.s.ragMode || "shadow")
          .onChange((v) => { this.s.ragMode = v as ArkSettings["ragMode"]; });
      });
    new Setting(this.content)
      .setName("Embedding API 地址")
      .setDesc("OpenAI 兼容 /embeddings 地址；留空则不启用向量路。")
      .addText((t) => t
        .setPlaceholder("https://dashscope.aliyuncs.com/compatible-mode/v1")
        .setValue(this.s.ragEmbedBaseUrl || "")
        .onChange((v) => { this.s.ragEmbedBaseUrl = v.trim(); }));
    new Setting(this.content)
      .setName("Embedding 模型")
      .setDesc("例如 text-embedding-v4；需要与 API 地址匹配。")
      .addText((t) => t
        .setPlaceholder("text-embedding-v4")
        .setValue(this.s.ragEmbedModel || "text-embedding-v4")
        .onChange((v) => { this.s.ragEmbedModel = v.trim(); }));
    new Setting(this.content)
      .setName("Embedding API Key")
      .setDesc("仅在 provider 要求鉴权时填写；不会写入 agentlab/config.json。")
      .addText((t) => {
        t.inputEl.setAttribute("type", "password");
        t.setPlaceholder("sk-...").setValue(this.s.ragEmbedApiKey || "")
          .onChange((v) => { this.s.ragEmbedApiKey = v; });
      });
    new Setting(this.content)
      .setName("Embedding 超时（秒）")
      .setDesc("网络异常时会走有限重试；建议 10～60 秒。")
      .addText((t) => {
        t.inputEl.setAttribute("type", "number");
        t.inputEl.setAttribute("min", "1");
        t.inputEl.setAttribute("max", "300");
        t.setValue(String(this.s.ragEmbedTimeout || 30))
          .onChange((v) => { this.s.ragEmbedTimeout = Number(v) || 30; });
      });
    this.content.createDiv({
      cls: "sos-hint",
      text: "保存后重启 Agentlab serve 生效。未填写 API 地址时，即使选择混合/向量模式也会安全回退到关键词。",
    });
    new Setting(this.content)
      .setName("启用长期记忆")
      .setDesc("开启后自动召回并沉淀长期记忆；关闭只停止隐式使用，不删除已有 Markdown 记忆，也不影响人工管理和显式 memory 工具。")
      .addToggle((toggle) => toggle
        .setValue(this.s.memoryEnabled !== false)
        .onChange((value) => { this.s.memoryEnabled = value; }));
  }

  // ===== 通信 =====
  private renderComm() {
    this.section("邮箱账户");
    const ctx = this.content.createDiv({ cls: "sos-mail-list" });
    const draw = () => {
      ctx.empty();
      if (this.s.mailAccountsConfig.length === 0) {
        ctx.createDiv({ cls: "sos-hint", text: "暂无邮箱账户，请添加 IMAP 账户以收发邮件" });
      }
      this.s.mailAccountsConfig.forEach((acc, i) => {
        const row = ctx.createDiv({ cls: "sos-mail-row" });
        row.createSpan({ cls: "sos-mail-dot", attr: { style: `background:${acc.color || "#00e5ff"}` } });
        row.createSpan({ cls: "sos-mail-name", text: `${acc.name ?? "未命名"} (${acc.email})` });
        const edit = row.createEl("button", { cls: "sos-mini", text: "编辑" });
        edit.addEventListener("click", () => this.editAccount(i));
        const del = row.createEl("button", { cls: "sos-mini", text: "✕" });
        del.addEventListener("click", () => { this.s.mailAccountsConfig.splice(i, 1); draw(); });
      });
      const add = ctx.createEl("button", { cls: "sos-mini", text: "+ 添加邮箱账户" });
      add.addEventListener("click", () => {
        const acc: MailAccountConfig = {
          id: "mail_" + Date.now().toString(36), name: "新邮箱", email: "", type: "imap",
          imapPort: 993, imapSsl: true, smtpPort: 465, smtpSsl: true, color: "#00e5ff",
        };
        this.s.mailAccountsConfig.push(acc);
        this.editAccount(this.s.mailAccountsConfig.length - 1);
        draw();
      });
    };
    draw();

    this.section("自动同步");
    new Setting(this.content).setName("自动同步间隔（分钟）")
      .setDesc("0 = 关闭自动同步")
      .addDropdown((dd) => {
        ([0, 3, 5, 10, 15, 30] as const).forEach((m) => dd.addOption(String(m), m === 0 ? "不自动同步" : `每 ${m} 分`));
        dd.setValue(String(this.s.autoSyncInterval ?? 5)).onChange((v) => { this.s.autoSyncInterval = Number(v); });
      });

    this.section("提醒与日历");
    new Setting(this.content).setName("提醒接收邮箱").addText((t) => t.setPlaceholder("your@email.com").setValue(this.s.reminderEmail).onChange((v) => { this.s.reminderEmail = v; }));
    new Setting(this.content).setName("默认提醒时间")
      .setDesc("提前分钟数")
      .addDropdown((dd) => {
        this.s.reminderPresets.forEach((p) => dd.addOption(String(p.value), `${p.value} 分钟 ${p.label}`));
        dd.setValue(String(this.s.defaultReminderOffset)).onChange((v) => { this.s.defaultReminderOffset = Number(v); });
      });
    new Setting(this.content).setName("自动发送日历邀请 (.ics)").addToggle((t) => t.setValue(this.s.calendarInviteEnabled).onChange((v) => { this.s.calendarInviteEnabled = v; }));
    new Setting(this.content).setName("自动发送到期提醒邮件").addToggle((t) => t.setValue(this.s.reminderEmailEnabled).onChange((v) => { this.s.reminderEmailEnabled = v; }));

    this.section("提醒预设管理");
    const preCtx = this.content.createDiv({ cls: "sos-bl-list" });
    const drawPresets = () => {
      preCtx.empty();
      if (this.s.reminderPresets.length === 0) this.s.reminderPresets = [...DEFAULT_REMINDER_PRESETS];
      this.s.reminderPresets.forEach((p, i) => {
        const row = preCtx.createDiv({ cls: "sos-bl-row" });
        const val = row.createEl("input", { attr: { type: "number", min: "0", title: "提前分钟数（0=不提醒）", value: String(p.value) } });
        val.addEventListener("change", () => { this.s.reminderPresets[i].value = Number(val.value) || 0; });
        const label = row.createEl("input", { attr: { type: "text", value: p.label } });
        label.addEventListener("change", () => { this.s.reminderPresets[i].label = label.value; });
        const del = row.createEl("button", { cls: "sos-mini", text: "✕" });
        del.addEventListener("click", () => { this.s.reminderPresets.splice(i, 1); drawPresets(); });
      });
      const add = preCtx.createEl("button", { cls: "sos-mini", text: "+ 添加预设" });
      add.addEventListener("click", () => { const p: ReminderPreset = { value: 60, label: "提前 1 小时" }; this.s.reminderPresets.push(p); drawPresets(); });
    };
    drawPresets();
  }

  private section(name: string) {
    this.content.createDiv({ cls: "sos-settings-section", text: name });
  }

  private editAccount(i: number) {
    const acc = this.s.mailAccountsConfig[i];
    const modal = new Modal(this.app);
    modal.titleEl.setText(`编辑邮箱账户`);
    const body = modal.contentEl;
    body.addClass("sos-mail-edit");
    const s = new Setting(body).setName("名称").addText((t) => t.setValue(acc.name).onChange((v) => { acc.name = v; }));
    void s;
    new Setting(body).setName("邮箱").addText((t) => t.setValue(acc.email).onChange((v) => { acc.email = v; }));
    new Setting(body).setName("密码").addText((t) => { t.inputEl.setAttribute("type", "password"); t.setValue(acc.password ?? "").onChange((v) => { acc.password = v; }); });
    new Setting(body).setName("IMAP Host").addText((t) => t.setValue(acc.imapHost ?? "").onChange((v) => { acc.imapHost = v; }));
    new Setting(body).setName("IMAP 端口").addText((t) => { t.inputEl.setAttribute("type", "number"); t.setValue(String(acc.imapPort)).onChange((v) => { acc.imapPort = Number(v) || 993; }); });
    new Setting(body).setName("SMTP Host").addText((t) => t.setValue(acc.smtpHost ?? "").onChange((v) => { acc.smtpHost = v; }));
    new Setting(body).setName("SMTP 端口").addText((t) => { t.inputEl.setAttribute("type", "number"); t.setValue(String(acc.smtpPort)).onChange((v) => { acc.smtpPort = Number(v) || 465; }); });
    new Setting(body).setName("SSL").addToggle((t) => t.setValue(acc.imapSsl).onChange((v) => { acc.imapSsl = v; acc.smtpSsl = v; }));
    new Setting(body).addButton((b) => b.setButtonText("完成").setCta().onClick(() => modal.close()));
    modal.open();
  }
}
