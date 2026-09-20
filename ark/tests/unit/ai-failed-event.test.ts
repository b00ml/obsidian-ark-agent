// OPT-217：SSE 终态事件（response.failed/cancelled）不得被静默吞掉。
// 回归背景：LLM 402（DeepSeek 余额不足）时 serve 发 response.failed，
// dispatch 落入"未知事件"仅打 console，agentChat 返回空串被结算成空消息，
// 用户在 CRT/工作台看到"（空）"而不知失败原因。
import assert from "node:assert/strict";
import { test } from "node:test";

import { agentChat } from "../../src/ai";

type Settings = Parameters<typeof agentChat>[0];

function makeSettings(): Settings {
  return {
    agentProvider: "agentlab",
    agentlabUrl: "http://127.0.0.1:1/v1",
    agentlabToken: "t",
    agentlabModel: "agentlab-demo",
  } as unknown as Settings;
}

function sseResponse(events: object[]): Response {
  const body = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("");
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

function withFetch(events: object[], calls: string[] = []) {
  const original = globalThis.fetch;
  globalThis.fetch = (async (_url: any, _init?: any) => {
    calls.push(String(_url));
    return sseResponse(events);
  }) as typeof fetch;
  return () => { globalThis.fetch = original; };
}

test("response.failed 事件必须抛错且携带服务端错误文本", async () => {
  const restore = withFetch([
    { type: "response.output_text.delta", delta: "部分" },
    { type: "response.failed", error: "[AGENT_LLM_AUTH] LLM 返回 HTTP 402: Insufficient Balance" },
  ]);
  try {
    await assert.rejects(
      agentChat(makeSettings(), [{ role: "user", content: "hi" }]),
      /Insufficient Balance/,
    );
  } finally {
    restore();
  }
});

test("response.cancelled 事件抛出取消语义错误", async () => {
  const restore = withFetch([{ type: "response.cancelled" }]);
  try {
    await assert.rejects(
      agentChat(makeSettings(), [{ role: "user", content: "hi" }]),
      /已取消/,
    );
  } finally {
    restore();
  }
});

test("回归：正常 delta 流仍拼接全文，无 failed 时不误报", async () => {
  const restore = withFetch([
    { type: "response.output_text.delta", delta: "Vault 中存在 " },
    { type: "response.output_text.delta", delta: "README.md" },
    { type: "response.completed" },
  ]);
  try {
    const reply = await agentChat(makeSettings(), [{ role: "user", content: "hi" }]);
    assert.equal(reply, "Vault 中存在 README.md");
  } finally {
    restore();
  }
});
