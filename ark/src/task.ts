// 任务卡片 ↔ md 写入：任务落盘到 todoFolder/<清单名>/<日期>-<id>.md
import type ArkOSPlugin from "./main";
import type { ArkTask, TodoList } from "./types";
import { writeMarkdown } from "./sync";
import { generateId, getToday } from "./utils";
import { serializeRecurrence, getNextDueDate } from "./recurrence";

/** 新建任务的落盘位置（清单名子目录 + 当天日期前缀）。纯函数，便于单测路径规则。 */
export function taskFolderFor(listName: string, taskId: string, todoFolder: string,
                              today: string): { folder: string; filename: string } {
  const dir = todoFolder + "/" + listName.replace(/[/\\:*?"<>|]/g, "-");
  return { folder: dir, filename: `${today}-${taskId}` };
}

/**
 * 决定一次写入该落在哪个文件——**已有 notePath 时一律复用**。
 *
 * 这是 OPT-191 的回归点：`taskFolderFor` 的文件名带当天日期，若每次写入都重算，
 * 隔天写回就会算出新路径 → 新建第二个文件 → watcher 按 notePath 去重失配 → 任务分叉。
 * 抽成纯函数是为了让"跨天路径必须稳定"这条不变量能被单测直接锁住（见 tests/unit）。
 */
export function resolveTaskPath(task: Pick<ArkTask, "id" | "notePath">, listName: string,
                                todoFolder: string, today: string): { folder: string; filename: string } {
  const existing = (task.notePath ?? "").trim();
  if (existing) {
    const parts = existing.split("/");
    const base = parts.pop()!.replace(/\.md$/, "");
    return { folder: parts.join("/") || ".", filename: base };
  }
  return taskFolderFor(listName, task.id, todoFolder, today);
}

export function taskToFm(task: ArkTask, listName: string): Record<string, unknown> {
  return {
    id: task.id,
    // 任务描述必须落在 frontmatter；否则 watcher 重新导入时只能退回文件名。
    description: task.description,
    list_id: task.listId,
    list_name: listName,
    status: task.status,
    completed: task.completed,
    // 完成时间必须落盘：此前没写，导致重载后 completedAt 被回落成"导入时刻"。
    // createNextRecurrence 在任务没有 dueDate 时会拿它当基准，丢了这个值循环任务会算错下一期。
    completed_at: task.completedAt ? new Date(task.completedAt).toISOString() : undefined,
    priority: task.priority,
    start_date: task.startDate,
    due_date: task.dueDate,
    due_time: task.dueTime,
    effort: task.effort,
    dependency: task.dependency ?? [],
    progress: task.progress ?? (task.completed ? 100 : 0),
    reminder_offset: task.reminderOffset ?? 0,
    tags: task.tags ?? [],
    recurrence: serializeRecurrence(task.recurrence),
    is_recurrence_instance: task.isRecurrenceInstance ?? false,
    original_task_id: task.originalTaskId,
    created_at: new Date(task.createdAt).toISOString(),
  };
}

/** 写任务 md（新增即建新文件，更新即重写同文件）——writeMarkdown 会 import 进 data */
export async function writeTask(plugin: ArkOSPlugin, list: TodoList, task: ArkTask): Promise<string> {
  // 路径决策全部交给 resolveTaskPath（纯函数、有单测）；这里只负责落盘与回填。
  const { folder, filename } = resolveTaskPath(
    task, list.name, plugin.data.settings.todoFolder, getToday());
  const path = await writeMarkdown(plugin, folder, filename, taskToFm(task, list.name), "");
  // 立即回填：不等 watcher 导入完成，后续写入就已稳定（否则同一 tick 内连写两次仍会分叉）。
  task.notePath = path;
  return path;
}

/**
 * 任务完成时生成下一循环实例（until/count 双终止 + 防重）。
 * 返回生成的下一任务（原任务不再被修改）；不满足条件返回 null。
 */
export async function createNextRecurrence(plugin: ArkOSPlugin, list: TodoList, task: ArkTask): Promise<ArkTask | null> {
  const rec = task.recurrence;
  if (!rec || rec.frequency === "none") return null;

  const base = task.dueDate ? new Date(task.dueDate + "T12:00:00") : task.completedAt ? new Date(task.completedAt) : new Date();
  if (isNaN(base.getTime())) return null;
  const next = getNextDueDate(rec, base);

  // until 终止：下一期超过 until → 不再生成
  if (rec.until) {
    const untilMs = new Date(rec.until + "T23:59:59").getTime();
    if (isNaN(untilMs)) return null;
    if (new Date(next + "T12:00:00").getTime() > untilMs) return null;
  }

  // 防重：已存在"未完成 + 同描述 + 同 next dueDate"的任务则复用
  const dup = list.tasks.find((t) => !t.completed && t.description === task.description && t.dueDate === next);
  if (dup) return dup;

  // count 递减：存在 count 时 -1；减到 0 移除循环
  let nextRec = rec;
  if (typeof rec.count === "number" && rec.count > 0) {
    const c = rec.count - 1;
    nextRec = c > 0 ? { ...rec, count: c } : undefined as any;
  }

  const nt: ArkTask = {
    id: generateId(),
    description: task.description,
    completed: false,
    createdAt: Date.now(),
    status: "todo",
    priority: task.priority ?? "medium",
    dueDate: next,
    startDate: next,
    dueTime: task.dueTime,
    effort: task.effort,
    dependency: task.dependency ?? [],
    progress: 0,
    reminderOffset: task.reminderOffset,
    tags: task.tags ?? [],
    listId: task.listId,
    recurrence: nextRec,
    isRecurrenceInstance: true,
    originalTaskId: task.id,
  };
  await writeTask(plugin, list, nt);
  return nt;
}
