import assert from "node:assert/strict";
import { test } from "node:test";

import { DEFAULT_SETTINGS } from "../../src/settings";
import {
  isRagReconcilePath,
  ragReconcileArgs,
  ragReconcileEnabled,
} from "../../src/rag-reconcile";

test("RAG reconcile 仅在 embedding provider 配置后启用", () => {
  assert.equal(ragReconcileEnabled({ ...DEFAULT_SETTINGS, ragMode: "keyword" }), false);
  assert.equal(ragReconcileEnabled({ ...DEFAULT_SETTINGS, ragMode: "shadow" }), false);
  assert.equal(ragReconcileEnabled({
    ...DEFAULT_SETTINGS, ragMode: "shadow", ragEmbedBaseUrl: "https://embed.example/v1",
  }), true);
});

test("RAG reconcile 只接收 Markdown，忽略隐藏索引和二进制文件", () => {
  assert.equal(isRagReconcilePath("wiki/笔记.md"), true);
  assert.equal(isRagReconcilePath("wiki\\笔记.MD"), true);
  assert.equal(isRagReconcilePath(".agent-brain/rag-index.sqlite"), false);
  assert.equal(isRagReconcilePath("assets/image.png"), false);
});

test("RAG reconcile CLI 参数通过独立 vault 参数传递", () => {
  assert.deepEqual(ragReconcileArgs("E:/vault with spaces"), [
    "-m", "agentlab.rag_reindex", "--reconcile", "--vault", "E:/vault with spaces",
  ]);
});
