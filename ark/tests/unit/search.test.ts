// 检索地基：分词与片段裁剪。它们是"本库搜索"的质量前提（打分公式尚未抽出，见 OPT-192 边界）。
import assert from "node:assert/strict";
import { test } from "node:test";

import { makeSnippet, tokenize } from "../../src/search";

test("tokenize：中英标点与空白都作为分隔符", () => {
  assert.deepEqual(tokenize("知识库, 检索。agent/hub"), ["知识库", "检索", "agent", "hub"]);
  assert.deepEqual(tokenize("alpha-beta_gamma"), ["alpha", "beta", "gamma"], "连字符与下划线也分词");
});

test("tokenize：大小写归一 + 过滤停用词 + 去掉空片段", () => {
  assert.deepEqual(tokenize("The AGENT"), ["agent"], "停用词与大小写都要处理");
  // 停用词表是显式枚举（中英混合），不是"所有单字母"：a/an/the/and… 会被丢，b/c 不会
  assert.deepEqual(tokenize("a an the"), []);
  assert.deepEqual(tokenize("a b c"), ["b", "c"], "不在表里的短词必须保留");
  assert.deepEqual(tokenize("   "), []);
  assert.deepEqual(tokenize("，，。"), []);
});

test("makeSnippet：未命中时返回开头摘要，且压平换行", () => {
  const s = makeSnippet("第一行\n第二行".repeat(20), -1);
  assert.equal(s.length <= 80, true);
  assert.equal(s.includes("\n"), false, "片段里不能带原始换行，否则列表会串行");
});

test("makeSnippet：命中时截取上下文并加省略号标记", () => {
  const text = "A".repeat(60) + "命中" + "B".repeat(60);
  const s = makeSnippet(text, 60);
  assert.ok(s.startsWith("…"), "前方被截断要标省略号");
  assert.ok(s.endsWith("…"), "后方被截断要标省略号");
  assert.ok(s.includes("命中"), "命中词必须留在片段里");
});
