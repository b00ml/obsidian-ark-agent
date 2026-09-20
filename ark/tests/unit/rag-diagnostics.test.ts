import assert from "node:assert/strict";
import { test } from "node:test";
import { diagnoseRagSettings, ragStrategyLabel } from "../../src/rag-diagnostics";
import { DEFAULT_SETTINGS } from "../../src/settings";

test("RAG diagnostics explains keyword-only mode without provider", () => {
  const d = diagnoseRagSettings({ ...DEFAULT_SETTINGS, ragMode: "keyword", ragEmbedBaseUrl: "", ragEmbedApiKey: "" });
  assert.equal(d.effectiveStrategy, "keyword");
  assert.equal(d.lexicalEnabled, true);
  assert.equal(d.vectorEnabled, false);
  assert.equal(d.warnings.length, 0);
});

test("RAG diagnostics explains shadow mode and provider state", () => {
  const d = diagnoseRagSettings({ ...DEFAULT_SETTINGS, ragMode: "shadow", ragEmbedBaseUrl: "https://example.test/v1", ragEmbedApiKey: "" });
  assert.equal(d.effectiveStrategy, "shadow");
  assert.equal(d.vectorEnabled, true);
  assert.equal(d.lexicalEnabled, true);
  assert.equal(d.warnings.some((w) => w.includes("API Key")), true);
  assert.equal(ragStrategyLabel(d.effectiveStrategy), "关键词展示 + 向量观测");
});

test("RAG diagnostics fails closed to keyword when vector provider is absent", () => {
  const d = diagnoseRagSettings({ ...DEFAULT_SETTINGS, ragMode: "vector", ragEmbedBaseUrl: "", ragEmbedApiKey: "" });
  assert.equal(d.effectiveStrategy, "keyword");
  assert.equal(d.lexicalEnabled, true);
  assert.equal(d.vectorEnabled, false);
  assert.equal(d.warnings.some((w) => w.includes("回退")), true);
});
