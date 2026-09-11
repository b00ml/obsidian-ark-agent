import type ArkOSPlugin from "./main";
import type { CrtSession } from "./settings";
import { generateId } from "./utils";
import { getSkin } from "./skins";

/** Agent 工作台与 CRT 共用的会话生命周期服务；不包含 UI、网络或 Modal 逻辑。 */
export function sessionList(plugin: ArkOSPlugin): CrtSession[] {
  if (!Array.isArray(plugin.data.crtSessions)) plugin.data.crtSessions = [];
  return plugin.data.crtSessions;
}

export function newSession(plugin: ArkOSPlugin, projectId?: string): CrtSession {
  return {
    id: generateId(),
    title: "新会话",
    createdAt: Date.now(),
    updatedAt: Date.now(),
    projectId: (projectId ?? plugin.data.settings.activeProjectId ?? "").trim() || undefined,
    messages: [{ role: "system", content: getSkin(plugin.data.settings.skin).systemPrompt }],
  };
}

/** 首条用户消息才提交，避免空会话污染历史。返回是否发生提交。 */
export function commitSession(plugin: ArkOSPlugin, session: CrtSession): boolean {
  const list = sessionList(plugin);
  if (list.some((item) => item.id === session.id)) return false;
  const firstUser = session.messages.find((message) => message.role === "user");
  if (firstUser && session.title === "新会话") session.title = firstUser.content.slice(0, 24);
  list.unshift(session);
  plugin.data.crtActiveId = session.id;
  return true;
}

/** 统一更新时间和历史容量治理；调用方随后执行 savePluginData。 */
export function touchSession(plugin: ArkOSPlugin, session: CrtSession, limit = 50): void {
  session.updatedAt = Date.now();
  const list = sessionList(plugin);
  list.sort((a, b) => b.updatedAt - a.updatedAt);
  if (list.length > limit) list.length = limit;
}

export function sessionsForProject(plugin: ArkOSPlugin, projectId = ""): CrtSession[] {
  const pid = projectId.trim();
  return sessionList(plugin).filter((session) => (session.projectId || "") === pid);
}
