import { Modal, Setting } from "obsidian";
import type { SpaceOSView } from "../view";
import { checkAuth, listMails, readMail, sendMail, replyMail, deleteMail, toPlain, downloadAttachment, saveIcs, type MailFull } from "../mail";
import { confirmDialog, notice, inputDialog } from "../ui";
import { generateId } from "../utils";
import { getSkin } from "../skins";

/** 消息与邮箱：收件箱/发件箱 + 读信 + 发送/回复 + 最近联系人。 */
export function renderComm(view: SpaceOSView, mount: HTMLElement) {
  mount.addClass("space-comm");
  const data = view.plugin.data;
  const skin = getSkin(data.settings.skin);

  const toolbar = mount.createDiv({ cls: "log-toolbar" });
  toolbar.createDiv({ cls: "tactical-title", text: skin.tabs.comm.title });

  const dirSel = toolbar.createEl("select", { cls: "tactical-list-select" });
  (["inbox", "sent", "trash", "spam"] as const).forEach((d) => dirSel.createEl("option", { value: d, text: dLabel(d) }));
  let activeDir = "inbox" as string;

  const authStatus = toolbar.createEl("span", { cls: "comm-auth", text: "…" });

  toolbar.createEl("button", { cls: "log-new-btn", text: "✉ 写邮件" }).addEventListener("click", () => composeModal(view, []));
  const refresh = toolbar.createEl("button", { cls: "log-type-tab", text: "🔄 刷新" });
  refresh.addEventListener("click", () => load());

  const contactsBtn = toolbar.createEl("button", { cls: "log-type-tab", text: "📒 通讯录" });
  contactsBtn.addEventListener("click", () => {
    contactsBtn.toggleClass("active", true);
    listEl.empty(); detailEl.empty();
    renderContactManager(view, listEl);
  });

  const listEl = mount.createDiv({ cls: "comm-list" });
  const detailEl = mount.createDiv({ cls: "comm-detail" });
  const contactBar = mount.createDiv({ cls: "comm-contacts" });

  // 最近联系人
  function renderContacts() {
    contactBar.empty();
    contactBar.createSpan({ cls: "comm-contacts-label", text: "联系人: " });
    const seen: string[] = [];
    data.settings.contacts.slice(0, 12).forEach((c) => {
      if (seen.includes(c.email)) return;
      seen.push(c.email);
      const chip = contactBar.createEl("button", { cls: "sos-mini", text: c.name ?? c.email });
      chip.addEventListener("click", () => composeModal(view, [c.email]));
    });
    if (seen.length === 0) contactBar.createSpan({ cls: "comm-contacts-label", text: "（发送邮件后自动累积）" });
  }

  function load() {
    listEl.empty();
    listEl.createDiv({ cls: "tactical-empty", text: skin.panel.commLoading });
    detailEl.empty();
    checkAuth().then((err) => {
      if (err) {
        authStatus.setText("⚠ " + err);
        listEl.empty();
        listEl.createDiv({ cls: "tactical-empty", text: `未授权：请在终端先执行 \`agently-cli auth login\`（打开浏览器登录 agent.qq.com）。\n提示：${err}` });
        return;
      }
      authStatus.setText("已连接");
      listMails({ dir: activeDir, limit: 30 })
        .then((ms) => {
          listEl.empty();
          if (ms.length === 0) {
            listEl.createDiv({ cls: "tactical-empty", text: "暂无邮件" });
            return;
          }
          ms.forEach((m) => listEl.appendChild(renderRow(m, detailEl, view, load)));
        })
        .catch((e: any) => {
          listEl.empty();
          listEl.createDiv({ cls: "tactical-empty", text: "拉取失败: " + String(e?.message ?? e) });
        });
    });
  }

  dirSel.addEventListener("change", () => { activeDir = dirSel.value; load(); });
  renderContacts();
  load();
}

function dLabel(d: string): string {
  return d === "inbox" ? "收件箱" : d === "sent" ? "已发送" : d === "trash" ? "回收站" : "垃圾邮件";
}

function renderRow(m: { message_id: string; subject: string; snippet?: string; is_read: boolean; created_at: string; from?: { name?: string; email?: string } }, detailEl: HTMLElement, view: SpaceOSView, refresh: () => void): HTMLElement {
  const row = document.createElement("div");
  row.addClass("comm-row");
  if (!m.is_read) row.addClass("unread");
  const from = m.from?.name || m.from?.email?.split("@")[0] || "未知";
  row.createDiv({ cls: "comm-row-from", text: from });
  row.createDiv({ cls: "comm-row-subject", text: m.subject || "（无主题）" });
  row.createDiv({ cls: "comm-row-time", text: fmtTime(m.created_at) });
  row.addEventListener("click", () => openDetail(detailEl, m.message_id, view, refresh));
  return row;
}

function openDetail(detailEl: HTMLElement, id: string, view: SpaceOSView, refresh: () => void) {
  detailEl.empty();
  detailEl.createDiv({ cls: "tactical-empty", text: "加载中…" });
  readMail(id).then((m: MailFull) => {
    detailEl.empty();
    const card = detailEl.createDiv({ cls: "log-entry glass" });
    card.createDiv({ cls: "db-head" }).createSpan({ cls: "db-name", text: m.subject || "（无主题）" });
    const from = m.from?.name ? `${m.from.name} <${m.from.email}>` : m.from?.email || "未知";
    card.createDiv({ cls: "log-entry-time", text: `发件人：${from}` });
    if (m.to?.length) card.createDiv({ cls: "log-entry-time", text: `收件人：${m.to.map((t) => t.email).join(", ")}` });
    card.createDiv({ cls: "log-entry-time", text: fmtTime(m.created_at) });

    let body = m.body || "";
    const fmt = (m.body_format || "").toLowerCase();
    if (fmt.includes("html")) body = toPlain(body);
    const bodyEl = card.createDiv({ cls: "comm-body" });
    bodyEl.setText(body);

    // 附件
    if (m.attachments?.length) {
      const attWrap = card.createDiv({ cls: "comm-body", attr: { style: "color:var(--space-yellow)" } });
      attWrap.setText("附件：");
      m.attachments.forEach((a) => {
        const b = attWrap.createEl("button", { cls: "tactical-mini-btn", text: `⬇ ${a.filename ?? a.attachment_id ?? "file"}` });
        b.addEventListener("click", async () => {
          try {
            const r = await downloadAttachment(view.plugin, id, a);
            if (r.startsWith("http")) { window.open(r, "_blank"); notice("已打开下载链接"); }
            else notice(r || "下载完成");
          } catch (e: any) { notice("附件下载失败: " + (e?.message ?? e)); }
        });
      });
    }
    // 日历邀请
    if (m.calendar_ics) {
      const icsBtn = card.createEl("button", { cls: "tactical-mini-btn", text: "📅 存为日历邀请 (.ics)" });
      icsBtn.addEventListener("click", async () => {
        try {
          const p = await saveIcs(view.plugin, m);
          notice(p ? "已保存: " + p : "无日历内容");
        } catch (e: any) { notice("保存失败: " + (e?.message ?? e)); }
      });
    }

    const act = card.createDiv({ cls: "tactical-card-actions" });
    const reply = act.createEl("button", { cls: "tactical-mini-btn", text: "↩ 回复" });
    reply.addEventListener("click", () => composeModal(view, [m.from?.email ?? ""], undefined, id));
    const del = act.createEl("button", { cls: "tactical-mini-btn", text: "🗑 删除" });
    del.addEventListener("click", async () => {
      if (!(await confirmDialog(view.plugin, "删除该邮件？"))) return;
      await deleteMail(id).catch(() => {});
      detailEl.empty();
      refresh();
    });

    // 保存发件人到通讯录
    if (m.from?.email) {
      const already = view.plugin.data.settings.contacts.some((c) => c.email.toLowerCase() === m.from!.email!.toLowerCase());
      if (!already) {
        view.plugin.data.settings.contacts.push({ email: m.from.email, name: m.from.name || m.from.email.split("@")[0], lastUsed: Date.now() });
        view.plugin.savePluginData();
      }
    }
  }).catch((e: any) => {
    detailEl.empty();
    detailEl.createDiv({ cls: "tactical-empty", text: "读取失败: " + String(e?.message ?? e) });
  });
}

function composeModal(view: SpaceOSView, to: string[], preSubject?: string, replyId?: string) {
  const modal = new Modal(view.plugin.app);
  modal.titleEl.setText(replyId ? "回复邮件" : "写邮件");
  const body = modal.contentEl;

  let tos = to.join(", ");
  let subject = preSubject ?? "";
  let mailBody = "";

  new Setting(body).setName("收件人").addText((t) => t.setValue(tos).onChange((v) => { tos = v; }));
  if (!replyId) {
    new Setting(body).setName("主题").addText((t) => t.setValue(subject).onChange((v) => { subject = v; }));
  }
  const area = body.createDiv({ cls: "comm-body-input" });
  const ta = area.createEl("textarea", { attr: { rows: "10", placeholder: "邮件正文（支持 Markdown）…" } });
  ta.style.width = "100%";
  ta.addEventListener("input", () => { mailBody = ta.value; });

  const btns = body.createDiv({ cls: "sos-settings-foot" });
  new Setting(body).addButton((b) => b.setButtonText("取消").onClick(() => modal.close()));
  new Setting(body).addButton((b) => b.setButtonText("发送").setCta().onClick(async () => {
    const recipients = tos.split(/[,，;]/).map((s) => s.trim()).filter(Boolean);
    if (recipients.length === 0) { notice("请填写收件人"); return; }
    try {
      const id = replyId ? await replyMail(replyId, mailBody) : await sendMail(recipients, subject.trim() || "(无主题)", mailBody);
      void id;
      // 累积到通讯录
      recipients.forEach((t) => {
        const already = view.plugin.data.settings.contacts.some((c) => c.email.toLowerCase() === t.toLowerCase());
        if (!already) view.plugin.data.settings.contacts.push({ email: t, name: t.split("@")[0], lastUsed: Date.now() });
      });
      await view.plugin.savePluginData();
      modal.close();
      notice("已发送");
      view.renderCurrentPanel();
    } catch (e: any) {
      notice("发送失败: " + String(e?.message ?? e));
    }
  }));
  void btns;
  modal.open();
}

function fmtTime(iso: string): string {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

/** 通讯录管理（独立双轨：正式 contactList + 自动采集 settings.contacts） */
function renderContactManager(view: SpaceOSView, mount: HTMLElement) {
  const data = view.plugin.data;
  const formal = data.contactList;
  const auto = data.settings.contacts;

  const dash = mount.createDiv({ cls: "log-dashboard" });
  dash.createEl("span", { cls: "log-stat", text: `正式 ${formal.length}` });
  dash.createEl("span", { cls: "log-stat", text: `自动 ${auto.length}` });

  const bar = mount.createDiv({ cls: "log-toolbar" });
  bar.createEl("button", { cls: "log-new-btn", text: "+ 新建联系人" }).addEventListener("click", async () => {
    const name = await inputDialog(view.plugin, { title: "姓名" });
    if (!name || !name.trim()) return;
    const email = (await inputDialog(view.plugin, { title: "邮箱", placeholder: "xxx@example.com" })) ?? "";
    const phone = (await inputDialog(view.plugin, { title: "电话（可选）", placeholder: "可选" })) ?? "";
    data.contactList.push({
      id: generateId(), name: name.trim(), email: email.trim() || undefined,
      phone: phone.trim() || undefined, tags: [], createdAt: Date.now(), avatarColor: randomColor(),
    });
    await view.plugin.savePluginData();
    redrawFormal();
  });

  const formalList = mount.createDiv({ cls: "comm-list" });
  function redrawFormal() {
    formalList.empty();
    if (formal.length === 0) { formalList.createDiv({ cls: "tactical-empty", text: "暂无正式联系人" }); return; }
    formal.forEach((c) => {
      const row = formalList.createDiv({ cls: "comm-row" });
      row.createDiv({ cls: "comm-row-from", text: c.name || c.email || "未命名" });
      if (c.email) row.createDiv({ cls: "comm-row-subject", text: c.email });
      if (c.phone) row.createDiv({ cls: "comm-row-time", text: c.phone });
      const act = row.createDiv({ cls: "tactical-card-actions" });
      const mail = act.createEl("button", { cls: "tactical-mini-btn", text: "✉" });
      mail.addEventListener("click", () => composeModal(view, c.email ? [c.email] : []));
      const del = act.createEl("button", { cls: "tactical-mini-btn", text: "✕" });
      del.addEventListener("click", async () => {
        if (!(await confirmDialog(view.plugin, `删除联系人 ${c.name}？`))) return;
        const i = formal.indexOf(c);
        if (i >= 0) formal.splice(i, 1);
        await view.plugin.savePluginData();
        redrawFormal();
      });
    });
  }
  redrawFormal();

  // 自动采集联系人（只读 + 一键填收件人）
  mount.createDiv({ cls: "tactical-title", text: "自动采集（发送/读信累积）" });
  const autoList = mount.createDiv({ cls: "comm-list" });
  if (auto.length === 0) {
    autoList.createDiv({ cls: "tactical-empty", text: "暂无自动采集联系人" });
  } else {
    [...auto].sort((a, b) => b.lastUsed - a.lastUsed).slice(0, 40).forEach((c) => {
      const chip = autoList.createEl("button", { cls: "sos-mini", text: `${c.name ?? c.email}` });
      chip.style.margin = "3px";
      chip.addEventListener("click", () => composeModal(view, [c.email]));
    });
  }
}

const CONTACT_COLORS = ["#00e5ff", "#b34dff", "#00ff99", "#ffcc00", "#ff6699", "#ff8800", "#0066ff"];
function randomColor(): string {
  return CONTACT_COLORS[Math.floor(Math.random() * CONTACT_COLORS.length)];
}
