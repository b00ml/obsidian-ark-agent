// @ 解析规则（OPT-185/188）与审批模式归一（OPT-187）的不变量。
// 两者都是"规则密集 + 我刚写完"的纯函数，是下一步最值得锁的目标。
import assert from "node:assert/strict";
import { test } from "node:test";

import { parseMentions } from "../../src/multi-mention";
import { normalizeSettings } from "../../src/settings";
import { getSkin } from "../../src/skins";

const AGENTS = ["agentlab", "claude-code"];

test("行首/空白后的 @name 命中名单 → 转 multi 并从正文剥离", () => {
  const r = parseMentions("@claude-code 帮我核对这段设计", AGENTS);
  assert.deepEqual(r.multi, ["claude-code"]);
  assert.equal(r.cleaned, "帮我核对这段设计", "答者只应收裸问题，mention 必须剥离");
  assert.deepEqual(r.unknown, []);
});

test("邮箱与正文中的 @ 不触发（必须行首或空白后）", () => {
  const mail = parseMentions("发到 someone@claude-code.com 就行", AGENTS);
  assert.deepEqual(mail.multi, [], "邮箱里的 @ 不得被当成指派");
  assert.equal(mail.cleaned, "发到 someone@claude-code.com 就行");
  const inline = parseMentions("讨论a@claude-code 的用法", AGENTS);
  assert.deepEqual(inline.multi, [], "紧贴文字的 @ 不触发（要求前置空白）");
});

test("未知名原样保留并回报 unknown（不得吞掉用户正文）", () => {
  const r = parseMentions("@nobody 你好", AGENTS);
  assert.deepEqual(r.multi, []);
  assert.deepEqual(r.unknown, ["nobody"]);
  assert.equal(r.cleaned, "@nobody 你好");
});

test("重复 mention 去重且保持出现顺序", () => {
  const r = parseMentions("@claude-code 问 A\n@agentlab 问 B\n@claude-code 再问", AGENTS);
  assert.deepEqual(r.multi, ["claude-code", "agentlab"]);
  assert.equal(r.cleaned, "问 A\n问 B\n再问", "换行必须保留");
});

test("多行正文不被压缩（贴代码/列清单场景）", () => {
  const code = "@claude-code 看这段\n\n```ts\nconst a = 1;\n```\n";
  const r = parseMentions(code, AGENTS);
  assert.deepEqual(r.multi, ["claude-code"]);
  assert.ok(r.cleaned.includes("```ts\nconst a = 1;\n```"),
    `代码块必须原样保留，实际=${JSON.stringify(r.cleaned)}`);
});

test("OPT-187：approvalMode 缺失或非法一律回落 risk_based（fail-closed）", () => {
  assert.equal(normalizeSettings({ skin: "star" } as any).approvalMode, "risk_based");
  assert.equal(normalizeSettings({ approvalMode: "ALLOW" } as any).approvalMode, "risk_based");
  assert.equal(normalizeSettings({ approvalMode: "allow_all" } as any).approvalMode, "allow_all");
  assert.equal(normalizeSettings({} as any).skin, "zero", "皮肤归一仍然生效");
});

test("OPT-210：历史 skin ID 统一返回 Ark 中性界面文案", () => {
  const star = getSkin("star");
  const cultivation = getSkin("cultivation");
  const zero = getSkin("zero");
  assert.strictEqual(star, zero, "历史主题应复用同一套界面文案");
  assert.strictEqual(cultivation, zero, "历史主题应复用同一套界面文案");
  assert.equal(zero.brand, "Ark");
  assert.equal(zero.tabs.tactical.title, "任务管理");
  assert.equal(zero.tabs.logs.title, "工作记录");
  assert.equal(zero.tabs.comm.title, "消息与邮箱");

  const forbidden = ["舰桥", "航行志", "战术台", "专业版", "VIP", "零号中枢", "星舰"];
  const copy = JSON.stringify(zero);
  forbidden.forEach((word) => assert.equal(copy.includes(word), false, `界面文案不应包含 ${word}`));
});

test("OPT-210：旧称呼配置归一为用户", () => {
  assert.equal(normalizeSettings({ captainName: "舰长" } as any).captainName, "用户");
  assert.equal(normalizeSettings({ captainName: "小明" } as any).captainName, "小明");
});
