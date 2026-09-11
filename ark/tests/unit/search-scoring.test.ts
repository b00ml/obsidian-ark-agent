// 本库检索的打分与过滤规则（此前内联在 vaultSearch 里、依赖 TFile，因此没被覆盖）。
// 抽成 scoreText / isBlacklisted 后可直接单测；vaultSearch 的排序与截断用假 vault 覆盖。
import assert from "node:assert/strict";
import { test } from "node:test";

import { globToRegExp, isBlacklisted, scoreText, tokenize, vaultSearch } from "../../src/search";

const terms = (q: string) => tokenize(q);

test("AND 语义：少一个词就不算命中", () => {
  const s = scoreText("a/笔记.md", "这里只有 知识库", terms("知识库 检索"));
  assert.equal(s.matched, false, "两个词必须都出现，否则不应进入结果");
  assert.equal(s.score, 0);
});

test("加权：文件名 +6、仅正文 +4，且不叠加", () => {
  const inName = scoreText("a/知识库.md", "正文提到 知识库", terms("知识库"));
  assert.equal(inName.score, 6, "文件名命中加 6，不再另加正文的 4");
  const inBody = scoreText("a/other.md", "正文提到 知识库", terms("知识库"));
  assert.equal(inBody.score, 4);
});

test("多词按词累加，标题行含首词再 +3", () => {
  const text = "# 知识库 入门\n正文 检索 说明";
  // 知识库：文件名命中(+6)；检索：正文命中(+4)；标题行含首词(+3)
  const s = scoreText("a/知识库.md", text, terms("知识库 检索"));
  assert.equal(s.score, 6 + 4 + 3);
});

test("大小写不敏感，且命中位置取最早出现处", () => {
  const s = scoreText("a/x.md", "ABC 在前\nabc 在后", terms("abc"));
  assert.equal(s.matched, true);
  assert.equal(s.firstIdx, 0, "应取最早出现位置");
  assert.equal(s.lineStart, 1);
});

test("lineStart 按命中前的换行数计算", () => {
  const s = scoreText("a/x.md", "第一行\n第二行\n第三行有 知识库", terms("知识库"));
  assert.equal(s.lineStart, 3);
  assert.ok(s.snippet.includes("知识库"), "片段必须包含命中词");
});

test("glob：** 跨目录、* 只在同层（顺序敏感：** 必须先于 * 替换）", () => {
  assert.equal(globToRegExp("**.md").test("深/层/文件.md"), true);
  assert.equal(globToRegExp("03-待办/*.md").test("03-待办/a.md"), true);
  assert.equal(globToRegExp("03-待办/*.md").test("03-待办/子/a.md"), false,
    "单星号不得跨目录，否则黑名单会误伤整棵子树");
  assert.equal(globToRegExp("03-待办/**.md").test("03-待办/子/a.md"), true);
});

test("isBlacklisted：空模式忽略，命中任一即拦", () => {
  assert.equal(isBlacklisted("a/b.md", ["", "  ", "a/**"]), true);
  assert.equal(isBlacklisted("a/b.md", ["", "x/**"]), false);
});

// ── vaultSearch：假 vault 覆盖排序 / 截断 / 黑名单跳过 ──

function fakePlugin(files: { path: string; text: string }[], blacklist: string[] = []) {
  return {
    app: { vault: {
      getMarkdownFiles: () => files,
      cachedRead: async (f: { text: string }) => f.text,
    } },
    data: { settings: { scanBlacklist: blacklist } },
  } as any;
}

test("vaultSearch：按分数降序返回，并遵守 limit", async () => {
  const plugin = fakePlugin([
    { path: "a/other.md", text: "知识库" },        // 仅正文 → 4
    { path: "a/知识库.md", text: "知识库" },        // 文件名 → 6
    { path: "a/知识库 手册.md", text: "知识库" },   // 文件名 → 6（并列）
  ]);
  const hits = await vaultSearch(plugin, "知识库", 2);
  assert.equal(hits.length, 2, "limit 必须生效");
  assert.equal(hits[0].score, 6);
  assert.ok(hits.every((h) => h.score >= hits[hits.length - 1].score), "必须降序");
  assert.equal(hits[0].sourceType, "vault");
  assert.equal(hits[0].sourceRef.startsWith("a/"), true, "必须带来源引用");
});

test("vaultSearch：黑名单命中直接跳过，AND 语义与空查询都要正确", async () => {
  const plugin = fakePlugin([
    { path: "a/dashboard.md", text: "知识库" },
    { path: "a/keep.md", text: "知识库" },
  ], ["**/dashboard.md"]);
  const hits = await vaultSearch(plugin, "知识库");
  assert.deepEqual(hits.map((h) => h.sourceRef), ["a/keep.md"], "黑名单文件不得进入结果");

  const andMiss = await vaultSearch(plugin, "知识库 不存在的词");
  assert.deepEqual(andMiss, [], "AND 语义：缺少任一词应无结果");
  assert.deepEqual(await vaultSearch(plugin, "   "), [], "空查询不得返回全库");
});
