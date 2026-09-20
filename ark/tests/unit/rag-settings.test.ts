import assert from "node:assert/strict";
import { test } from "node:test";

import { DEFAULT_SETTINGS } from "../../src/settings";
import { ragEnvironment } from "../../src/serve-control";

test("RAG 设置映射：关键词模式关闭向量且不依赖后端旧配置", () => {
  const env = ragEnvironment({ ...DEFAULT_SETTINGS, ragMode: "keyword", ragEmbedBaseUrl: "" });
  assert.equal(env.AGENT_RAG_VECTOR_ENABLED, "false");
  assert.equal(env.AGENT_RAG_VECTOR_MODE, "off");
  assert.equal(env.AGENT_RAG_LEXICAL_MODE, "on");
  assert.equal(env.AGENT_RAG_EMBED_BASE_URL, "");
  assert.equal(env.AGENT_MEMORY_ENABLED, "true");
});

test("RAG 设置映射：混合模式同时开启关键词和向量", () => {
  const env = ragEnvironment({
    ...DEFAULT_SETTINGS,
    ragMode: "hybrid",
    ragEmbedBaseUrl: "https://embed.example/v1",
    ragEmbedModel: "embed-v1",
    ragEmbedApiKey: "secret",
    ragEmbedTimeout: 12,
  });
  assert.equal(env.AGENT_RAG_VECTOR_ENABLED, "true");
  assert.equal(env.AGENT_RAG_VECTOR_MODE, "on");
  assert.equal(env.AGENT_RAG_LEXICAL_MODE, "on");
  assert.equal(env.AGENT_RAG_EMBED_BASE_URL, "https://embed.example/v1");
  assert.equal(env.AGENT_RAG_EMBED_MODEL, "embed-v1");
  assert.equal(env.AGENT_RAG_EMBED_API_KEY, "secret");
  assert.equal(env.AGENT_RAG_EMBED_TIMEOUT, "12");
  assert.equal(env.AGENT_MEMORY_ENABLED, "true");
});

test("RAG 设置映射：无 API 地址时安全回退关键词", () => {
  const env = ragEnvironment({ ...DEFAULT_SETTINGS, ragMode: "vector", ragEmbedBaseUrl: "" });
  assert.equal(env.AGENT_RAG_VECTOR_ENABLED, "false");
  assert.equal(env.AGENT_RAG_VECTOR_MODE, "off");
  assert.equal(env.AGENT_RAG_LEXICAL_MODE, "on");
  assert.equal(env.AGENT_RAG_EMBED_API_KEY, "");
});

test("长期记忆开关关闭隐式召回与沉淀", () => {
  const env = ragEnvironment({ ...DEFAULT_SETTINGS, memoryEnabled: false });
  assert.equal(env.AGENT_MEMORY_ENABLED, "false");
});
