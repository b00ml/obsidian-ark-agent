// 卡片落盘路径规则：与 OPT-191（任务分叉）同类的"路径/文件名"逻辑。
// folderFor 只读 settings → 用一个假 plugin 就能测，不需要真实 Vault。
import assert from "node:assert/strict";
import { test } from "node:test";

import { folderFor, safeName } from "../../src/cards";

const fakePlugin = {
  data: {
    settings: {
      normalLogFolder: "02-DB/日志",
      faultLogFolder: "02-DB/故障",
      ideaFolder: "Inbox/灵感",
      drawingFolder: "02-DB/画板",
      healthFolder: "02-DB/体检",
      databaseFolder: "02-DB/知识",
    },
  },
} as any;

test("folderFor 按类型返回配置目录（日志按 type 分流）", () => {
  assert.equal(folderFor(fakePlugin, "log", { type: "normal" }), "02-DB/日志");
  assert.equal(folderFor(fakePlugin, "log", { type: "fault" }), "02-DB/故障");
  assert.equal(folderFor(fakePlugin, "log", { log_type: "fault_log" }), "02-DB/故障");
  assert.equal(folderFor(fakePlugin, "log", {}), "02-DB/日志", "缺 type 时回落普通日志目录");
  assert.equal(folderFor(fakePlugin, "idea"), "Inbox/灵感");
  assert.equal(folderFor(fakePlugin, "drawing"), "02-DB/画板");
  assert.equal(folderFor(fakePlugin, "health"), "02-DB/体检");
  assert.equal(folderFor(fakePlugin, "database"), "02-DB/知识");
});

test("safeName 替换文件名非法字符并截断到 24 字", () => {
  assert.equal(safeName("idea", { title: 'a/b:c*d?e"f<g>h|i' }), "a-b-c-d-e-f-g-h-i");
  const long = safeName("idea", { title: "字".repeat(50) });
  assert.equal(long.length, 24, "超长标题必须截断，避免生成超长文件名");
});

test("safeName 取标题的优先级与兜底：title > db_name > 类型名", () => {
  assert.equal(safeName("database", { title: "标题优先", db_name: "库名" }), "标题优先");
  assert.equal(safeName("database", { db_name: "库名" }), "库名");
  assert.equal(safeName("idea", {}), "idea", "都没有时回落类型名，不能是空文件名");
});
