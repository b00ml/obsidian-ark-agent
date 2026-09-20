import type { ArkSettings, RagMode } from "./settings";

export interface RagDiagnostics {
  configuredMode: RagMode;
  effectiveStrategy: "keyword" | "shadow" | "hybrid" | "vector";
  lexicalEnabled: boolean;
  vectorEnabled: boolean;
  providerConfigured: boolean;
  apiKeyConfigured: boolean;
  productionGate: "blocked" | "manual-only";
  warnings: string[];
}

/**
 * Explain the effective local routing without contacting the provider.
 * This is intentionally pure so the settings page and tests share one truth.
 */
export function diagnoseRagSettings(settings: Pick<ArkSettings,
  "ragMode" | "ragEmbedBaseUrl" | "ragEmbedApiKey">): RagDiagnostics {
  const configuredMode: RagMode = ["keyword", "shadow", "hybrid", "vector"].includes(settings.ragMode)
    ? settings.ragMode : "shadow";
  const providerConfigured = Boolean(String(settings.ragEmbedBaseUrl || "").trim());
  const vectorEnabled = providerConfigured && configuredMode !== "keyword";
  const lexicalEnabled = configuredMode !== "vector" || !vectorEnabled;
  const effectiveStrategy = !vectorEnabled && configuredMode !== "keyword"
    ? "keyword" : configuredMode;
  const warnings: string[] = [];
  if (configuredMode !== "keyword" && !providerConfigured) {
    warnings.push("未配置 Embedding 地址，当前实际回退为关键词");
  }
  if (vectorEnabled && !String(settings.ragEmbedApiKey || "").trim()) {
    warnings.push("未填写 API Key；若 provider 要求鉴权，向量请求会降级");
  }
  if (configuredMode === "shadow") {
    warnings.push("Shadow 只观测向量，不改变用户看到的关键词结果");
  }
  if (configuredMode === "hybrid" || configuredMode === "vector") {
    warnings.push("已按当前设置启用；全局生产门禁仍未通过，建议保留关键词回退并观察负例与延迟");
  }
  return {
    configuredMode,
    effectiveStrategy,
    lexicalEnabled,
    vectorEnabled,
    providerConfigured,
    apiKeyConfigured: Boolean(String(settings.ragEmbedApiKey || "").trim()),
    productionGate: configuredMode === "hybrid" || configuredMode === "vector" ? "manual-only" : "blocked",
    warnings,
  };
}

export function ragStrategyLabel(strategy: RagDiagnostics["effectiveStrategy"]): string {
  return { keyword: "关键词", shadow: "关键词展示 + 向量观测", hybrid: "关键词 + 向量 RRF", vector: "仅向量" }[strategy];
}
