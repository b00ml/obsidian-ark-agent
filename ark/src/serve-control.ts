/** agentlab serve 进程控制（OPT-113）：ribbon 按钮 / 命令面板 / CRT 一键拉起统一走这里。
 *
 *  - start/stop/status 全部经 `python -m agentlab.runtime.serve_manage <cmd>`——
 *    pidfile 单实例守护 + 端口反查 + 连树终止，替代此前裸 spawn `python -m agentlab
 *    serve`（无守护、只进不出、重启电脑后无法从插件收尸）；
 *  - 健康探测复用 ai.probeAgent（GET /health），状态图标随之切换；
 *  - 子进程输出强制 UTF-8（PYTHONUTF8/PYTHONIOENCODING）：serve_manage 打印中文，
 *    不注入则走 GBK 控制台页码，Node 侧按 utf8 解码必碎（对齐 OPT-099 编码红线）；
 *  - 端口/token 以 agentlab config.json 为准（serve_manage 无 --port 位，fail-closed
 *    token 缺失时 start 会显式报错），ark 侧 agentlabUrl 仅用于健康探测。
 */
import { execFile } from "child_process";
import { existsSync } from "fs";
import { homedir } from "os";
import { requestUrl } from "obsidian";
import { ArkSettings, RagMode } from "./settings";

/** 将 Ark 设置页的友好模式映射为 agentlab RagConfig 环境覆盖。 */
export function ragEnvironment(settings: ArkSettings): Record<string, string> {
  const mode: RagMode = settings.ragMode || "shadow";
  const endpoint = (settings.ragEmbedBaseUrl || "").trim();
  const vectorMode = mode === "keyword" ? "off"
    : mode === "shadow" ? "shadow"
      : "on";
  // 没有 endpoint 时强制关闭 provider，避免后端 config.json 中旧配置意外启用向量。
  const enabled = Boolean(endpoint) && mode !== "keyword";
  const lexicalMode = !enabled || mode !== "vector" ? "on" : "off";
  return {
    AGENT_RAG_VECTOR_ENABLED: enabled ? "true" : "false",
    AGENT_RAG_VECTOR_MODE: enabled ? vectorMode : "off",
    // Shadow collects vector evidence only. A display-changing fallback needs
    // its own production gate and must never be implied by selecting shadow.
    AGENT_RAG_VECTOR_FALLBACK_MODE: "off",
    AGENT_RAG_LEXICAL_MODE: lexicalMode,
    // Ark owns the migration point: once a provider is configured, serve uses
    // the versioned P2 single index and can initialize it on first reconcile.
    // Keyword-only remains legacy and never creates an embedding index.
    AGENT_RAG_INDEX_BACKEND: enabled ? "p2" : "legacy",
    AGENT_RAG_EMBED_BASE_URL: endpoint,
    AGENT_RAG_EMBED_MODEL: (settings.ragEmbedModel || "text-embedding-v4").trim() || "text-embedding-v4",
    AGENT_RAG_EMBED_API_KEY: settings.ragEmbedApiKey || "",
    AGENT_RAG_EMBED_TIMEOUT: String(Number(settings.ragEmbedTimeout) > 0 ? settings.ragEmbedTimeout : 30),
    // One user-facing switch controls implicit long-term-memory behavior.
    // Existing Markdown memories remain intact when this is disabled.
    AGENT_MEMORY_ENABLED: settings.memoryEnabled === false ? "false" : "true",
  };
}

/** agentlab serve 健康探测（GET {agentlabUrl 去掉 /v1}/health）——
 *  始终盯 agentlabUrl，不随 agentProvider 切换（ribbon/按钮是 agentlab 专属）。 */
export async function serveUp(settings: ArkSettings): Promise<boolean> {
  const base = (settings.agentlabUrl || "http://127.0.0.1:8643/v1")
    .replace(/\/v1\/?$/, "").replace(/\/+$/, "");
  try {
    const r: any = await requestUrl({ url: `${base}/health`, method: "GET" });
    return r.status === 200;
  } catch {
    return false;
  }
}

/** serve 启动目录（须含 config/config.json）：设置覆盖 > 项目默认 */
export function serveWorkdir(settings: ArkSettings): string {
  return (settings.agentlabWorkdir || "").trim() || "agentlab";
}

/** python 可执行文件：设置覆盖 > 项目 .venv > PATH 上的 python */
export function resolvePython(settings: ArkSettings): string {
  const configured = (settings.agentlabExePath || "").trim();
  if (configured) return configured;
  const candidates = process.platform === "win32"
    ? [`${process.cwd()}\\.venv\\Scripts\\python.exe`, `${homedir()}\\.venv\\Scripts\\python.exe`]
    : [`${process.cwd()}/.venv/bin/python`, `${homedir()}/.venv/bin/python`];
  return candidates.find((p) => existsSync(p)) || "python";
}

function runManage(settings: ArkSettings, action: "start" | "stop" | "status",
                   timeoutMs: number): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(
      resolvePython(settings),
      ["-m", "agentlab.runtime.serve_manage", action],
      {
        cwd: serveWorkdir(settings),
        timeout: timeoutMs,
        encoding: "utf8",
        windowsHide: true,
        env: {
          ...process.env, PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8",
          // P2-2/F5-021：审批策略由 Ark 设置下发，不写后端 config.json（内含 API Key）。
          // 服务端 serve_config() 读取该变量覆盖 cfg.approval_mode。
          AGENTLAB_APPROVAL_MODE:
            settings.approvalMode === "allow_all" ? "allow_all" : "risk_based",
          ...(action === "start" ? ragEnvironment(settings) : {}),
        },
      },
      (err, stdout, stderr) => {
        const out = `${stdout || ""}${stderr || ""}`.trim();
        if (err && !out) {
          reject(err);
          return;
        }
        // serve_manage 的业务结果以输出为准（业务失败也带说明文本，比裸退出码可读）
        resolve(out);
      },
    );
  });
}

/** 拉起 serve（内部等健康检查，返回 serve_manage 输出文本）。
 *  先探测：已在运行则免重复调 serve_manage（防双启残留进程）。 */
export async function serveStart(settings: ArkSettings): Promise<string> {
  if (await serveUp(settings)) {
    return "agentlab serve 已在运行，无需重复启动";
  }
  return runManage(settings, "start", 90_000);
}

/** 停止 serve（pidfile + 端口反查连树终止） */
export function serveStop(settings: ArkSettings): Promise<string> {
  return runManage(settings, "stop", 30_000);
}

/** 状态 JSON 文本（running/pid/port/health/token 掩码由 serve_manage 负责） */
export function serveStatusText(settings: ArkSettings): Promise<string> {
  return runManage(settings, "status", 15_000);
}
