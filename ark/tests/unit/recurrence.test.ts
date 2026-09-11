// 循环任务规则（OP T-049 引入 until/count 双终止 + 防重；逻辑密集，历史出过问题）。
// 关键语义：getNextDueDate 的 base **当日不算**（至少 +1 天），周日的 getDay 归一到 7。
import assert from "node:assert/strict";
import { test } from "node:test";

import { getNextDueDate, matchesRecurrence, parseRecurrence, serializeRecurrence } from "../../src/recurrence";

const FRI = new Date("2026-09-11T12:00:00");  // 2026-09-11 是周五（js getDay=5）

test("serialize → parse 往返保真", () => {
  const rec = { frequency: "weekly" as const, daysOfWeek: [1, 5], count: 3, until: "2026-12-31" };
  const back = parseRecurrence(serializeRecurrence(rec));
  assert.deepEqual(back, {
    frequency: "weekly", daysOfWeek: [1, 5], daysOfMonth: undefined,
    month: undefined, day: undefined, count: 3, until: "2026-12-31",
  });
});

test("none / 空 / 脏值解析为 undefined", () => {
  assert.equal(serializeRecurrence(undefined), "none");
  assert.equal(serializeRecurrence({ frequency: "none" }), "none");
  assert.equal(parseRecurrence("none"), undefined);
  assert.equal(parseRecurrence("undefined"), undefined);
  assert.equal(parseRecurrence(""), undefined);
  assert.equal(parseRecurrence("{不是 JSON"), undefined, "脏值不得抛异常");
  assert.equal(parseRecurrence('{"frequency":"none"}'), undefined);
});

test("daily：base 当日不算，返回次日", () => {
  assert.equal(getNextDueDate({ frequency: "daily" }, FRI), "2026-09-12");
});

test("weekly：默认周一；指定周五时返回下一周的周五而非当天", () => {
  assert.equal(getNextDueDate({ frequency: "weekly" }, FRI), "2026-09-14");
  assert.equal(getNextDueDate({ frequency: "weekly", daysOfWeek: [5] }, FRI), "2026-09-18",
    "base 当天即使是命中日也不能返回当天，否则完成即等于没滚动");
});

test("weekly：7 表示周日（getDay=0 的归一）", () => {
  assert.equal(getNextDueDate({ frequency: "weekly", daysOfWeek: [7] }, FRI), "2026-09-13");
});

test("monthly：默认每月 1 日；[31] 会跳过没有 31 号的月份", () => {
  assert.equal(getNextDueDate({ frequency: "monthly" }, FRI), "2026-10-01");
  assert.equal(getNextDueDate({ frequency: "monthly", daysOfMonth: [31] }, FRI), "2026-10-31");
});

test("yearly：默认 1/1；指定月日按年滚动", () => {
  assert.equal(getNextDueDate({ frequency: "yearly" }, FRI), "2027-01-01");
  assert.equal(getNextDueDate({ frequency: "yearly", month: 3, day: 15 }, FRI), "2027-03-15");
});

test("matchesRecurrence 与 getNextDueDate 口径一致", () => {
  const rec = { frequency: "weekly" as const, daysOfWeek: [1] };
  assert.equal(matchesRecurrence("2026-09-14", rec, FRI), true);
  assert.equal(matchesRecurrence("2026-09-15", rec, FRI), false);
});
