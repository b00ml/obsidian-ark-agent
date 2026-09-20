/**
 * Vault 事件 -> P3 RAG 增量索引触发器。
 *
 * 这是一个故意很薄的运维桥：索引的扫描、hash 判定、队列和 embedding
 * 仍由 agentlab.rag_reindex 负责。Ark 只负责把短时间内的一批 Vault 事件
 * 合并成一次 `--reconcile`，并保证同一时刻最多只有一个 CLI 子进程。
 */
import { execFile, type ChildProcess } from "child_process";
import type ArkOSPlugin from "./main";
import type { ArkSettings } from "./settings";
import { ragEnvironment, resolvePython, serveWorkdir } from "./serve-control";

export const RAG_RECONCILE_DEBOUNCE_MS = 1500;

interface ReconcileState {
  timer?: ReturnType<typeof setTimeout>;
  running: boolean;
  pending: boolean;
  child?: ChildProcess;
}

function getState(plugin: ArkOSPlugin): ReconcileState {
  const holder = plugin as ArkOSPlugin & { _ragReconcileState?: ReconcileState };
  if (!holder._ragReconcileState) {
    holder._ragReconcileState = { running: false, pending: false };
  }
  return holder._ragReconcileState;
}

/** 只有真正配置了 embedding provider 的非关键词模式才需要扫描 P3 索引。 */
export function ragReconcileEnabled(settings: ArkSettings): boolean {
  return settings.ragMode !== "keyword" && Boolean((settings.ragEmbedBaseUrl || "").trim());
}

/** P3 索引当前只收录 Markdown；隐藏索引目录不应触发自身递归重建。 */
export function isRagReconcilePath(path: string): boolean {
  const normalized = String(path || "").replace(/\\/g, "/");
  if (!normalized.toLowerCase().endsWith(".md")) return false;
  return !normalized.split("/").some((part) => part.toLowerCase() === ".agent-brain");
}

/** CLI 参数单独导出，便于单元测试也锁住跨平台的路径传递。 */
export function ragReconcileArgs(vaultRoot: string): string[] {
  return ["-m", "agentlab.rag_reindex", "--reconcile", "--vault", vaultRoot];
}

function vaultRoot(plugin: ArkOSPlugin): string {
  const adapter: any = plugin.app.vault.adapter as any;
  try {
    // Desktop FileSystemAdapter exposes getBasePath(); the property fallback
    // keeps the bridge testable with lightweight adapters and older Obsidian.
    return String((typeof adapter?.getBasePath === "function"
      ? adapter.getBasePath() : adapter?.basePath) || "").trim();
  } catch {
    return "";
  }
}

async function runReconcile(plugin: ArkOSPlugin, state: ReconcileState): Promise<void> {
  if (state.running) {
    state.pending = true;
    return;
  }
  const settings = plugin.data.settings;
  const root = vaultRoot(plugin);
  if (!ragReconcileEnabled(settings) || !root) return;

  state.running = true;
  state.pending = false;
  await new Promise<void>((resolve) => {
    const child = execFile(
      resolvePython(settings),
      ragReconcileArgs(root),
      {
        cwd: serveWorkdir(settings),
        timeout: 300_000,
        windowsHide: true,
        encoding: "utf8",
        env: {
          ...process.env,
          PYTHONUTF8: "1",
          PYTHONIOENCODING: "utf-8",
          // 与 serve 启动使用同一组 UI 覆盖，避免 CLI 误读旧 config 启用向量。
          ...ragEnvironment(settings),
        },
      },
      (error, stdout, stderr) => {
        const output = `${stdout || ""}${stderr || ""}`.trim();
        if (error) {
          console.warn(`[Ark] RAG 增量 reconcile 失败：${error.message}${output ? `\n${output}` : ""}`);
        } else if (output) {
          console.debug(`[Ark] RAG 增量 reconcile 完成：${output.slice(-1200)}`);
        }
        resolve();
      },
    );
    state.child = child;
  });
  state.child = undefined;
  state.running = false;
  if (state.pending) {
    state.pending = false;
    scheduleRagReconcile(plugin, "coalesced");
  }
}

/** 由 Vault watcher 调用；1500ms 内的多次 modify/delete/rename 合并为一次。 */
export function scheduleRagReconcile(plugin: ArkOSPlugin, _reason = "vault-change"): void {
  const settings = plugin.data.settings;
  if (!ragReconcileEnabled(settings) || !vaultRoot(plugin)) return;
  const state = getState(plugin);
  state.pending = true;
  if (state.timer) clearTimeout(state.timer);
  state.timer = setTimeout(() => {
    state.timer = undefined;
    void runReconcile(plugin, state);
  }, RAG_RECONCILE_DEBOUNCE_MS);
}

/** 插件卸载时取消未触发的 timer，并终止正在运行的本地 CLI。 */
export function stopRagReconcile(plugin: ArkOSPlugin): void {
  const holder = plugin as ArkOSPlugin & { _ragReconcileState?: ReconcileState };
  const state = holder._ragReconcileState;
  if (!state) return;
  if (state.timer) clearTimeout(state.timer);
  state.timer = undefined;
  state.pending = false;
  state.child?.kill();
  state.child = undefined;
  state.running = false;
}
