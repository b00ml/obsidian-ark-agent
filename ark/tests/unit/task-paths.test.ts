// 归档已踩过的两个真机缺陷，锁成不变量：
//  · OPT-191：任务文件跨天分叉（拖动后凭空多出一个任务）
//  · OPT-176：任务描述没写进 frontmatter（标题退回"日期+随机 ID"）
//
// 只测纯函数——不需要 Obsidian 运行时不碰 DOM，因此可以 node --test 直接跑。
import assert from "node:assert/strict";
import { test } from "node:test";

import { resolveTaskPath, taskFolderFor, taskToFm } from "../../src/task";

const TODO_FOLDER = "03-待办";
const LIST = "1";

test("OPT-191：已有 notePath 时，跨天写入必须复用同一路径", () => {
  const task = { id: "abc123", notePath: `${TODO_FOLDER}/${LIST}/2026-09-10-abc123.md` };

  const day1 = resolveTaskPath(task, LIST, TODO_FOLDER, "2026-09-10");
  const day2 = resolveTaskPath(task, LIST, TODO_FOLDER, "2026-09-11");
  const day90 = resolveTaskPath(task, LIST, TODO_FOLDER, "2026-12-31");

  assert.deepEqual(day2, day1, "次日写回不得算出新路径（否则会新建第二个文件 → 任务分叉）");
  assert.deepEqual(day90, day1, "跨月写回同样必须稳定");
  assert.equal(day2.filename, "2026-09-10-abc123", "文件名日期保持首次落盘那天");
});

test("全新任务：没有 notePath 时才按当天日期生成路径", () => {
  const fresh = { id: "new001" };
  assert.deepEqual(resolveTaskPath(fresh, LIST, TODO_FOLDER, "2026-09-11"),
    { folder: `${TODO_FOLDER}/${LIST}`, filename: "2026-09-11-new001" });
});

test("清单名里的非法文件名字符被替换为短横线", () => {
  const { folder } = taskFolderFor('我/的:清单*?', "x1", TODO_FOLDER, "2026-09-11");
  assert.equal(folder, `${TODO_FOLDER}/我-的-清单--`);
});

test("OPT-176：frontmatter 必须带 description 与 id", () => {
  const fm = taskToFm({
    id: "t1", description: "数学", completed: false, createdAt: 0,
    status: "todo", priority: "medium", tags: [],
  } as any, LIST);
  assert.equal(fm.description, "数学", "缺 description 会让 watcher 退回文件名，标题变成日期+随机 ID");
  assert.equal(fm.id, "t1", "id 必须在 frontmatter 里，否则无法与内存对象对应");
  assert.equal(fm.list_name, LIST);
});
