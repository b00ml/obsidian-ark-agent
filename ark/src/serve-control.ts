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
import { requestUrl } from "obsidian";
import { ArkSettings } from "./settings";

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

/** serve 启动目录（须含 config/config.json）：由用户设置，空值使用当前进程目录 */
export function serveWorkdir(settings: ArkSettings): string {
  return (settings.agentlabWorkdir || "").trim();
}

/** python 可执行文件：设置覆盖 > PATH 上的 python */
export function resolvePython(settings: ArkSettings): string {
  const configured = (settings.agentlabExePath || "").trim();
  if (configured) return configured;
  return "python";
}

function runManage(settings: ArkSettings, action: "start" | "stop" | "status",
                   timeoutMs: number): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(
      resolvePython(settings),
      ["-m", "agentlab.runtime.serve_manage", action],
      {
        cwd: serveWorkdir(settings) || undefined,
        timeout: timeoutMs,
        encoding: "utf8",
        windowsHide: true,
        env: {
          ...process.env, PYTHONUTF8: "1", PYTHONIOENCODING: "utf-8",
          // P2-2/F5-021：审批策略由 Ark 设置下发，不写后端 config.json（内含 API Key）。
          // 服务端 serve_config() 读取该变量覆盖 cfg.approval_mode。
          AGENTLAB_APPROVAL_MODE:
            settings.approvalMode === "allow_all" ? "allow_all" : "risk_based",
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
