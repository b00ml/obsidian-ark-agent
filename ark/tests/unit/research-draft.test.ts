import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildResearchDraft, buildResearchPrompt, extractRetrievalEnvelope,
  parseResearchRefs, researchDraftFilename, stripResearchFrontmatter,
} from "../../src/research-draft";

const envelope = {
  contract: "ark-contract-v1",
  strategy: "shadow",
  status: "degraded",
  warnings: ["vector_provider_timeout"],
  scope: { project_id: "p1", session_id: "s1", statuses: [], include_archive: false },
  items: [{
    title: "来源一", content: "支持该判断的摘录\n不要把换行撑坏布局。", ref: "wiki/source-one.md",
    source: "vault", score: 0.81234, status: "active", project_id: "p1", session_id: "s1", provenance: [],
  }],
  provenance: [],
};

test("research draft preserves envelope strategy, warning and citations", () => {
  const result = buildResearchDraft({ question: "如何比较两个方案？", purpose: "技术选型", scope: "wiki/", specifiedRefs: ["wiki/source-one.md"] }, envelope, "2026-09-15T00:00:00.000Z");
  assert.equal(result.meta.strategy, "shadow");
  assert.equal(result.meta.retrievalStatus, "degraded");
  assert.deepEqual(result.meta.refs, ["wiki/source-one.md"]);
  assert.match(result.markdown, /retrieval_strategy: shadow/);
  assert.match(result.markdown, /vector_provider_timeout/);
  assert.match(result.markdown, /\[\[wiki\/source-one\.md\]\]/);
  assert.match(result.markdown, /Project scope：p1/);
  assert.match(result.markdown, /Provenance/);
  assert.match(result.markdown, /结论草稿/);
});

test("research draft filters unrequested retrieval items and stays bounded", () => {
  const result = buildResearchDraft({ question: "Q", specifiedRefs: ["allowed.md"] }, {
    ...envelope,
    items: [envelope.items[0], { ...envelope.items[0], ref: "other.md", content: "secret" }],
  });
  assert.equal(result.meta.refs.includes("other.md"), false);
  assert.equal(result.markdown.includes("secret"), false);
});

test("research prompt explicitly keeps the flow read-only", () => {
  const prompt = buildResearchPrompt({ question: "Q", specifiedRefs: ["a.md"] });
  assert.match(prompt, /只读检索/);
  assert.match(prompt, /rag_retrieve\(envelope=true\)/);
  assert.match(prompt, /\[\[a\.md\]\]/);
});

test("research refs accept relative Markdown only", () => {
  assert.deepEqual(parseResearchRefs("wiki/a.md, ./wiki/a.md\n../secret.md, C:/secret.md, note.txt"), ["wiki/a.md"]);
});

test("research envelope extraction rejects truncated output and keeps the latest valid result", () => {
  const truncated = JSON.stringify(envelope).slice(0, 100);
  assert.equal(extractRetrievalEnvelope([{ name: "rag_retrieve", output: truncated }]), null);
  const found = extractRetrievalEnvelope([
    { name: "rag_retrieve", output: JSON.stringify({ ...envelope, scope: { ...envelope.scope, project_id: "old" } }) },
    { name: "rag_retrieve", output: JSON.stringify(envelope) },
  ]);
  assert.equal(found?.scope.project_id, "p1");
});

test("research write helpers stay bounded and do not duplicate YAML", () => {
  const built = buildResearchDraft({ question: "路径/注入: 研究", purpose: "p" }, envelope, "2026-09-15T00:00:00.000Z");
  assert.equal(researchDraftFilename(built.meta.title, "2026-09-15T00:00:00.000Z"), "20260915-路径-注入-研究");
  assert.match(stripResearchFrontmatter(built.markdown), /^# /);
  assert.doesNotMatch(stripResearchFrontmatter(built.markdown), /^---/);
});
