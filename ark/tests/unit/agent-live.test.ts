// 在途流结算语义（OPT-174）：用户报过"点停止后半截回答被清空"。
// agent-live.ts 零 Obsidian 依赖（只 import type），因此可直接单测。
import assert from "node:assert/strict";
import { test } from "node:test";

import { agentLive, removeLive, settleAllLive, settleLiveStream, updateLive } from "../../src/agent-live";

function makeLive(text = "半截回答") {
  return { text, abort: new AbortController(), started: 0, tools: [], elapsed: 3, status: "running" as const };
}

function makeSession(id: string) {
  return { id, title: "t", createdAt: 0, updatedAt: 0, messages: [] as { role: string; content: string }[] };
}

test.beforeEach(() => agentLive.clear());

test("停止时把半截回答结算进源会话（不得丢内容）", () => {
  const session = makeSession("s1");
  agentLive.set("s1", makeLive());
  assert.equal(settleLiveStream(session as any), true);
  assert.deepEqual(session.messages, [{ role: "assistant", content: "半截回答" }]);
  assert.equal(agentLive.get("s1")!.settled, true, "必须打标，供完成时改为替换而非追加");
  assert.equal(agentLive.get("s1")!.settleIdx, 0);
});

test("重复结算幂等：同一会话不得追加第二条半截回答", () => {
  const session = makeSession("s1");
  agentLive.set("s1", makeLive());
  settleLiveStream(session as any);
  assert.equal(settleLiveStream(session as any), false, "已结算应返回 false");
  assert.equal(session.messages.length, 1, "不得重复追加（否则停止一次就多一条回答）");
});

test("无在途流 / 无文本 / 无会话时结算返回 false 且不写脏数据", () => {
  const session = makeSession("s1");
  assert.equal(settleLiveStream(session as any), false, "没有 live 记录");
  agentLive.set("s1", makeLive(""));
  assert.equal(settleLiveStream(session as any), false, "空文本不该结算成空回答");
  assert.equal(session.messages.length, 0);
});

test("settleAllLive 只结算存在于会话列表里的 id（其余不误写）", () => {
  const s1 = makeSession("s1");
  agentLive.set("s1", makeLive("A"));
  agentLive.set("ghost", makeLive("B"));
  settleAllLive([s1 as any]);
  assert.deepEqual(s1.messages, [{ role: "assistant", content: "A" }]);
  assert.equal(agentLive.get("ghost")!.settled, undefined, "不在列表里的会话不得被结算");
});

test("removeLive 之后结算不再生效（运行态已收口）", () => {
  const session = makeSession("s1");
  agentLive.set("s1", makeLive());
  removeLive("s1");
  assert.equal(settleLiveStream(session as any), false);
  assert.equal(session.messages.length, 0);
});

test("updateLive 在无记录时返回 undefined（不凭空造状态）", () => {
  assert.equal(updateLive("nope", { elapsed: 9 }), undefined);
  agentLive.set("s1", makeLive());
  assert.equal(updateLive("s1", { elapsed: 9 })?.elapsed, 9);
});
