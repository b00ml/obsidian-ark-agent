import assert from "node:assert/strict";
import { test } from "node:test";

import {
  artifactsForSession, extractSessionArtifacts, upsertArtifacts, validateArtifactPaths,
  persistProjectArtifactIndex, projectArtifactIndexPath, readProjectArtifactIndex,
} from "../../src/artifacts";
import type { CrtSession } from "../../src/settings";
import { buildResearchDraft } from "../../src/research-draft";

const session: CrtSession = {
  id: "sess-1", title: "研究", createdAt: 1, updatedAt: 2, projectId: "proj-1",
  messages: [
    { role: "user", content: "整理资料" },
    { role: "assistant", content: "已完成，来源见 [[wiki/source.md]]。" },
  ],
};

test("extractSessionArtifacts extracts created/updated paths and preserves provenance", () => {
  const items = extractSessionArtifacts(session, [{
    name: "vault_write",
    output: JSON.stringify({ ok: true, path: "02-DB/result.md", status: "created" }),
  }], "2026-09-15T00:00:00.000Z");
  assert.equal(items.length, 1);
  assert.equal(items[0].path, "02-DB/result.md");
  assert.equal(items[0].project_id, "proj-1");
  assert.deepEqual(items[0].source_refs, ["wiki/source.md"]);
  assert.equal(items[0].provenance[0].ref, "session/sess-1");
  assert.equal(items[0].status, "unvalidated");
});

test("extractSessionArtifacts rejects traversal and deduplicates paths", () => {
  const items = extractSessionArtifacts(session, [
    { name: "vault_write", output: JSON.stringify({ path: "../secret.md" }) },
    { name: "vault_patch", output: JSON.stringify({ updated: ["02-DB/result.md", "02-DB/result.md"] }) },
  ]);
  assert.deepEqual(items.map((item) => item.path), ["02-DB/result.md"]);
});

test("validateArtifactPaths marks files without treating folders as files", () => {
  const plugin = { app: { vault: {
    getAbstractFileByPath: (path: string) => path === "ok.md" ? { extension: "md" } : path === "folder" ? {} : null,
  } }, data: { artifacts: [] } } as any;
  const input = extractSessionArtifacts({ ...session, messages: [] }, [{ name: "vault_write", output: JSON.stringify({ path: "ok.md" }) }]);
  input.push({ ...input[0], id: "missing", path: "missing.md" });
  assert.deepEqual(validateArtifactPaths(plugin, input).map((item) => item.status), ["validated", "missing"]);
});

test("upsertArtifacts is idempotent and session-scoped lookup is stable", () => {
  const plugin = { data: { artifacts: [] } } as any;
  const items = extractSessionArtifacts(session, [{ name: "vault_write", output: JSON.stringify({ path: "ok.md" }) }]);
  upsertArtifacts(plugin, items);
  upsertArtifacts(plugin, items);
  assert.equal(plugin.data.artifacts.length, 1);
  assert.equal(artifactsForSession(plugin, "sess-1").length, 1);
  assert.equal(artifactsForSession(plugin, "other").length, 0);
});

test("same path updates create a version chain while replay stays idempotent", () => {
  const plugin = { data: { artifacts: [] } } as any;
  const first = extractSessionArtifacts(session, [{ name: "vault_write", output: JSON.stringify({ path: "ok.md", status: "created" }) }], "2026-09-15T00:00:00.000Z");
  const second = extractSessionArtifacts(session, [{ name: "vault_patch", output: JSON.stringify({ path: "ok.md", status: "updated" }) }], "2026-09-15T00:01:00.000Z");
  upsertArtifacts(plugin, first);
  upsertArtifacts(plugin, second);
  upsertArtifacts(plugin, second);
  const rows = plugin.data.artifacts.filter((item: any) => item.path === "ok.md").sort((a: any, b: any) => a.version - b.version);
  assert.equal(rows.length, 2);
  assert.equal(rows[0].version, 1);
  assert.equal(rows[1].version, 2);
  assert.equal(rows[1].supersedes, rows[0].id);
});

test("structured rename links old and new paths", () => {
  const items = extractSessionArtifacts(session, [{
    name: "vault_rename",
    output: JSON.stringify({ old_path: "draft.md", new_path: "final.md" }),
  }], "2026-09-15T00:02:00.000Z");
  const oldItem = items.find((item) => item.path === "draft.md");
  const newItem = items.find((item) => item.path === "final.md");
  assert.equal(oldItem?.operation, "deleted");
  assert.equal(newItem?.operation, "renamed");
  assert.equal(newItem?.previous_path, "draft.md");
});

test("delete tool output preserves deleted operation", () => {
  const items = extractSessionArtifacts(session, [{
    name: "vault_delete",
    output: JSON.stringify({ path: "old.md", operation: "deleted" }),
  }], "2026-09-15T00:02:30.000Z");
  assert.equal(items[0]?.path, "old.md");
  assert.equal(items[0]?.operation, "deleted");
  assert.equal(items[0]?.status, "deleted");
});

test("project artifact sidecar is rebuildable and rejects unsafe project ids", async () => {
  assert.equal(projectArtifactIndexPath("../escape"), null);
  const writes = new Map<string, string>();
  const plugin = { data: { artifacts: [] }, app: { vault: { adapter: {
    exists: async () => true,
    mkdir: async () => {},
    write: async (path: string, content: string) => writes.set(path, content),
    read: async (path: string) => writes.get(path) || "",
  } } } } as any;
  const item = extractSessionArtifacts(session, [{ name: "vault_write", output: JSON.stringify({ path: "ok.md" }) }], "2026-09-15T00:03:00.000Z");
  upsertArtifacts(plugin, item);
  assert.equal(await persistProjectArtifactIndex(plugin, "proj-1"), true);
  assert.deepEqual((await readProjectArtifactIndex(plugin, "proj-1")).map((row) => row.path), ["ok.md"]);
});

test("research draft fixture feeds its source refs into the shared artifact result", () => {
  const built = buildResearchDraft({ question: "研究问题", specifiedRefs: ["wiki/source-one.md"] }, {
    contract: "ark-contract-v1", strategy: "lexical-only", status: "available", warnings: [],
    scope: { project_id: "proj-1", session_id: "sess-1", statuses: [], include_archive: false },
    provenance: [{ source: "vault", ref: "wiki/source-one.md" }],
    items: [{ title: "来源", content: "证据", ref: "wiki/source-one.md", source: "vault", score: 1,
      status: "active", project_id: "proj-1", session_id: "sess-1", provenance: [] }],
  });
  const items = extractSessionArtifacts({ ...session, messages: [
    ...session.messages, { role: "assistant", content: built.markdown },
  ] }, [{ name: "vault_write", output: JSON.stringify({ path: "02-DB/研究.md", operation: "created" }) }]);
  assert.equal(items[0]?.path, "02-DB/研究.md");
  assert.equal(items[0]?.operation, "created");
  assert.ok(items[0]?.source_refs.includes("wiki/source-one.md"));
});
