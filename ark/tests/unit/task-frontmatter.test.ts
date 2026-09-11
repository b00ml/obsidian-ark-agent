// 时间戳持久化（用户真机报出："每次拖动 created_at 都会变一次"）。
//
// 根因：写入侧统一写 ISO 字符串（taskToFm/createCard/updateCard 都用 toISOString），
// 读取侧却只做 Number(v) → NaN → 静默回落 Date.now()。于是每次导入都把
// created_at/updated_at 刷成"导入时刻"，所有按时间排序/统计的功能都在用错数据。
import assert from "node:assert/strict";
import { test } from "node:test";

import { taskToFm } from "../../src/task";
import { taskFromFm } from "../../src/sync";

const LIST_ID = "l1";
const fakePlugin = {
  data: {
    settings: { todoFolder: "03-待办" },
    todoLists: [{ id: LIST_ID, name: "1", tasks: [] }],
  },
} as any;

const PATH = "03-待办/1/2026-09-10-abc.md";

function fmOf(extra: Record<string, unknown> = {}) {
  return {
    id: "abc", description: "数学", list_id: LIST_ID, list_name: "1",
    status: "doing", completed: false, priority: "medium", created_at: "2026-09-10T05:47:36.002Z",
    ...extra,
  };
}

test("ISO 字符串 created_at 必须保真（这正是每次拖动漂移的那条）", () => {
  const t = taskFromFm(fakePlugin, PATH, fmOf());
  assert.equal(t.createdAt, Date.parse("2026-09-10T05:47:36.002Z"),
    "解析成 NaN 再回落 Date.now() 就是漂移 bug 本身");
});

test("兼容另两种落盘形态：YAML Date 对象与数字（旧数据）", () => {
  const asDate = taskFromFm(fakePlugin, PATH, fmOf({ created_at: new Date("2026-09-10T05:47:36.002Z") }));
  assert.equal(asDate.createdAt, Date.parse("2026-09-10T05:47:36.002Z"));
  const asNum = taskFromFm(fakePlugin, PATH, fmOf({ created_at: 1757483256002 }));
  assert.equal(asNum.createdAt, 1757483256002);
});

test("只有真正的脏值才回落当前时间", () => {
  const before = Date.now();
  // null 特别列出来：旧实现走 Number(null)=0，会把 created_at 变成 1970 年
  for (const bad of ["", "   ", "not-a-date", null, undefined, {}, []]) {
    const t = taskFromFm(fakePlugin, PATH, fmOf({ created_at: bad }));
    assert.ok(t.createdAt >= before && t.createdAt <= Date.now(), `脏值 ${JSON.stringify(bad)} 应回落 now`);
  }
});

test("completed_at 持久化：完成后有值，未完成时无值", () => {
  const done = taskFromFm(fakePlugin, PATH, fmOf({
    status: "done", completed: true, completed_at: "2026-09-11T01:02:03.000Z",
  }));
  assert.equal(done.completedAt, Date.parse("2026-09-11T01:02:03.000Z"),
    "完成时间丢了会让无 dueDate 的循环任务算错下一期");

  const doing = taskFromFm(fakePlugin, PATH, fmOf());
  assert.equal(doing.completedAt, undefined);
});

test("taskToFm 写出 completed_at（ISO），未完成时不写该字段", () => {
  const base = { id: "t1", description: "x", completed: true, createdAt: 0,
    status: "done", priority: "medium", tags: [], completedAt: 1757483256002 } as any;
  const fm = taskToFm(base, "1");
  assert.equal(fm.completed_at, new Date(1757483256002).toISOString());

  const fm2 = taskToFm({ ...base, completed: false, completedAt: undefined }, "1");
  assert.equal(fm2.completed_at, undefined, "未完成不得残留完成时间");
});

test("往返一致性：写盘→重读后 createdAt / completedAt 不漂移", () => {
  const task = {
    id: "t1", description: "往返", completed: true, createdAt: Date.parse("2026-09-10T05:47:36.002Z"),
    completedAt: Date.parse("2026-09-11T01:02:03.000Z"),
    status: "done", priority: "medium", tags: [],
  } as any;
  const back = taskFromFm(fakePlugin, PATH, taskToFm(task, "1"));
  assert.equal(back.createdAt, task.createdAt);
  assert.equal(back.completedAt, task.completedAt);
});
