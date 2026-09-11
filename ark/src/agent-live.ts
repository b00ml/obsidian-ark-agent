/** Agent 直播注册表（OPT-109 W1）：CRT 终端与 Agent 工作台共用的在途流状态。
 *
 * 从 CRTTerminal 的 private static live 抽出为模块级单例——两处 UI（终端/工作台）
 * 看到同一批在途流：一边发起、另一边重绘时能重新挂载直播文本；结算语义同源。
 */
import type { CrtSession } from "./settings";

export interface LiveStream {
  text: string;
  abort: AbortController;
  started: number;
  /** 已观测到的工具调用，跨页面重绘时用于恢复轨迹。 */
  tools: { name: string; phase: "start" | "done"; output?: string }[];
  /** 最近一次心跳/本地时钟计算出的运行秒数。 */
  elapsed: number;
  /** 运行结束前保留的终态；从 Map 移除后发送按钮恢复。 */
  status: "running" | "completed" | "failed" | "cancelled";
  error?: string;
  settled?: boolean;   // 已被"切走结算"写入过源会话（幂等标记，完成时改为替换而非追加）
  settleIdx?: number;  // 结算插入源会话 messages 的索引，供完成时替换半截内容
}

export const agentLive = new Map<string, LiveStream>();

type LiveListener = (sessionId: string) => void;
const liveListeners = new Set<LiveListener>();

/** 订阅共享运行态。返回取消订阅函数；面板卸载时应调用它。 */
export function subscribeLive(listener: LiveListener): () => void {
  liveListeners.add(listener);
  return () => liveListeners.delete(listener);
}

/** 通知所有面板重新读取指定会话的共享快照。 */
export function notifyLive(sessionId: string): void {
  for (const listener of [...liveListeners]) {
    try {
      listener(sessionId);
    } catch (err) {
      // UI 重绘失败不能反向中断后台 Agent 流；下一次通知仍会尝试恢复页面。
      console.warn("[ARK] Agent live listener failed", err);
    }
  }
}

/** 更新运行态并通知订阅者，避免调用方直接改 Map 后遗漏 UI 刷新。 */
export function updateLive(
  sessionId: string,
  patch: Partial<Pick<LiveStream, "elapsed" | "status" | "error">>,
): LiveStream | undefined {
  const live = agentLive.get(sessionId);
  if (!live) return undefined;
  Object.assign(live, patch);
  notifyLive(sessionId);
  return live;
}

/** 统一移除运行态并通知当前页面恢复发送按钮。 */
export function removeLive(sessionId: string): void {
  if (!agentLive.delete(sessionId)) return;
  notifyLive(sessionId);
}

/** 把在途半截回答结算进源会话（幂等，学 pi teardown）：已结算/无文本 → false。
 *  只改会话数据不落盘——调用方负责持久化（CRT 走 dirty+persist，工作台走 savePluginData）。 */
export function settleLiveStream(session: CrtSession): boolean {
  const live = agentLive.get(session.id);
  if (!live || live.settled || !live.text) return false;
  live.settled = true;
  session.messages.push({ role: "assistant", content: live.text });
  live.settleIdx = session.messages.length - 1;
  session.updatedAt = Date.now();
  notifyLive(session.id);
  return true;
}

/** 结算所有在途流（App/视图关闭时兜底持久化半截回答）。 */
export function settleAllLive(sessions: CrtSession[]): void {
  const byId = new Map(sessions.map((s) => [s.id, s]));
  for (const [id] of agentLive) {
    const sess = byId.get(id);
    if (sess) settleLiveStream(sess);
  }
}
