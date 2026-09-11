/** Project 创建共享逻辑（OPT-064 二期）：命令面板与 Agent 工作台侧栏共用，一步到位。
 *
 * slug 规则与后端白名单一致（中英文/数字/_/-），目录约定 ark/projects/<id>/。
 * 创建即激活（settings.activeProjectId）；调用方负责创建后的界面刷新。
 */
import { Notice } from "obsidian";
import type ArkOSPlugin from "./main";

const SAFE_PROJECT_ID = /^[A-Za-z0-9_\-\u4e00-\u9fff]{1,64}$/;

export function slugifyProject(name: string): string {
  return name.replace(/[^\w\u4e00-\u9fff-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 64)
    || `p-${Date.now()}`;
}

export async function createProject(
  plugin: ArkOSPlugin, rawName: string,
): Promise<{ id: string; name: string } | null> {
  const name = rawName.trim();
  if (!name) return null;
  const id = slugifyProject(name);
  const s = plugin.data.settings;
  if ((s.projects || []).some((p) => p.id === id)) {
    new Notice(`Project 已存在：${id}`);
    return null;
  }
  const base = `ark/projects/${id}`;
  try {
    await plugin.app.vault.adapter.mkdir("ark");
    await plugin.app.vault.adapter.mkdir("ark/projects");
    await plugin.app.vault.adapter.mkdir(base);
    await plugin.app.vault.adapter.write(`${base}/AGENTS.md`,
      `# ${name} · 项目规则\n\n（项目规则优先于全局规范。写本项目专属约定：命名规范、输出模板、禁止动作……每次对话注入 AI 的 system 上下文）\n`);
    await plugin.app.vault.adapter.write(`${base}/project.md`,
      `# ${name} · 项目背景\n\n（长期上下文：目标、关键决策、常用路径。每次对话注入）\n`);
  } catch (e: any) {
    new Notice("创建项目目录失败: " + String(e?.message ?? e));
    return null;
  }
  s.projects = [...(s.projects || []), { id, name, createdAt: Date.now() }];
  s.activeProjectId = id;
  await plugin.savePluginData();
  return { id, name };
}

/** 删除 Project 目录和元数据；会话正文保留，解除 projectId 后回到全局分组。 */
export async function deleteProject(plugin: ArkOSPlugin, projectId: string): Promise<number> {
  const id = projectId.trim();
  if (!SAFE_PROJECT_ID.test(id) || id === "." || id === "..") {
    throw new Error("非法 Project id");
  }
  const s = plugin.data.settings;
  if (!(s.projects || []).some((p) => p.id === id)) {
    throw new Error(`Project 不存在：${id}`);
  }
  const base = `ark/projects/${id}`;
  if (await plugin.app.vault.adapter.exists(base)) {
    await plugin.app.vault.adapter.rmdir(base, true);
  }
  let detached = 0;
  for (const session of plugin.data.crtSessions || []) {
    if (session.projectId === id) {
      delete session.projectId;
      detached += 1;
    }
  }
  s.projects = (s.projects || []).filter((p) => p.id !== id);
  if (s.activeProjectId === id) s.activeProjectId = "";
  await plugin.savePluginData();
  return detached;
}
