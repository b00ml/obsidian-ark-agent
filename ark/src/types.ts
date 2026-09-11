// 全局共享类型定义

/** 重复频率 */
export type Frequency = "none" | "daily" | "weekly" | "monthly" | "yearly";

/** 任务状态（即看板四列） */
export type TaskStatus = "todo" | "doing" | "blocked" | "done";

/** 任务优先级 */
export type Priority = "high" | "medium" | "low";

/** 循环任务规则 */
export interface Recurrence {
  frequency: Frequency;
  daysOfWeek?: number[];
  daysOfMonth?: number[];
  month?: number;
  day?: number;
  count?: number;
  until?: string;
}

/** 待办任务 */
export interface ArkTask {
  id: string;
  description: string;
  completed: boolean;
  completedAt?: number;
  createdAt: number;
  status: TaskStatus;
  priority: Priority;
  dueDate?: string; // YYYY-MM-DD
  startDate?: string; // YYYY-MM-DD，向后兼容可选
  dueTime?: string; // HH:MM
  effort?: number; // 预估工时
  dependency?: string[]; // 依赖任务 id
  progress?: number; // 0-100
  recurrence?: Recurrence;
  reminderOffset?: number;
  tags: string[];
  notePath?: string;
  listId?: string;
  logType?: "normal" | "fault";
  isRecurrenceInstance?: boolean;
  originalTaskId?: string;
}

/** 待办清单 */
export interface TodoList {
  id: string;
  name: string;
  tasks: ArkTask[];
}

/** 日志类型 */
export type LogKind = "normal" | "fault";

/** 工作记录条目 */
export interface LogEntry {
  id: string;
  type: LogKind;
  title: string;
  content: string;
  createdAt: number;
  tags: string[];
  notePath?: string;
}

/** 联系人（正式轨道） */
export interface Contact {
  id: string;
  name: string;
  email?: string;
  phone?: string;
  company?: string;
  notes?: string;
  tags: string[];
  createdAt: number;
  avatarColor: string;
}

/** 通讯录自动采集地址（settings 轨道） */
export interface AutoContact {
  email: string;
  name?: string;
  lastUsed: number;
}

/** 数据库分类 */
export interface DbCategory {
  id: string;
  name: string;
  color: string;
}
