/** Capture/Inbox 生命周期契约（F5-016）。
 *
 * 新字段采用向后兼容读取：旧 feed-brief 的 status/processed 仍可被识别，
 * 新写入统一使用 state/source_ref/retry_count；不负责搬迁或删除旧文件。
 */
export type CaptureState =
  | "captured" | "processing" | "needs_review" | "approved"
  | "archived" | "linked" | "failed";

export interface CaptureFields {
  state: CaptureState;
  source: string;
  sourceRef: string;
  retryCount: number;
  errorCode?: string;
}

export function normalizeCaptureState(fm: Record<string, any>): CaptureState {
  const state = String(fm.state || "").trim();
  if (["captured", "processing", "needs_review", "approved", "archived", "linked", "failed"].includes(state)) {
    return state as CaptureState;
  }
  if (String(fm.status || "").trim() === "archived") return "archived";
  if (String(fm.processed || "").trim().toLowerCase() === "true") return "linked";
  return "captured";
}

export function captureFields(fm: Record<string, any>): CaptureFields {
  return {
    state: normalizeCaptureState(fm),
    source: String(fm.source || fm.type || "manual"),
    sourceRef: String(fm.source_ref || fm.url || fm.source || ""),
    retryCount: Math.max(0, Number(fm.retry_count || 0) || 0),
    errorCode: fm.error_code ? String(fm.error_code) : undefined,
  };
}

/** 生成统一状态字段，旧字段由调用方按兼容需要继续保留。 */
export function capturePatch(state: CaptureState, extra: Record<string, string> = {}): Record<string, string> {
  return { state, ...extra };
}
