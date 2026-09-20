/**
 * S0 product/runtime contract shared by Ark UI and agentlab.
 *
 * This file contains the wire shape only.  Persistence, RAG implementation
 * and rendering remain in their existing modules so the contract can migrate
 * callers without creating a second source of truth.
 */

export const ARK_CONTRACT_VERSION = "ark-contract-v1" as const;

export type RetrievalStrategy = "lexical-only" | "vector-only" | "hybrid" | "shadow";
export type RetrievalStatus = "available" | "degraded" | "unavailable";

export interface RetrievalScope {
  project_id: string;
  session_id: string;
  /** Empty means the route's documented default, not an inherited filter. */
  statuses: string[];
  include_archive: boolean;
}

export interface Provenance {
  source: string;
  ref: string;
  source_id?: string;
  project_id?: string;
  session_id?: string;
  captured_at?: string;
  version?: string;
}

export interface RetrievalItem {
  title: string;
  content: string;
  ref: string;
  source: string;
  score: number;
  status: string;
  project_id: string;
  session_id: string;
  provenance: Provenance[];
  /** Optional Small-to-Big neighbors; the primary ref remains unchanged. */
  context_of?: string[];
}

export interface RetrievalResult {
  contract: typeof ARK_CONTRACT_VERSION;
  items: RetrievalItem[];
  strategy: RetrievalStrategy;
  status: RetrievalStatus;
  warnings: string[];
  scope: RetrievalScope;
  provenance: Provenance[];
}

export interface ProjectContract {
  id: string;
  title: string;
  name?: string;
  mode: string;
  goal: string;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface SourceContract {
  ref: string;
  title: string;
  kind: string;
  scope: string;
  captured_at?: string;
  status: string;
  provenance: Provenance[];
}

export interface SessionContract {
  id: string;
  project_id: string;
  goal: string;
  status: string;
  events: Record<string, unknown>[];
  created_at: string;
  updated_at: string;
}

export interface ArtifactContract {
  id: string;
  project_id: string;
  kind: string;
  path: string;
  source_refs: string[];
  status: string;
  created_by: string;
  session_id: string;
  provenance: Provenance[];
  created_at: string;
  updated_at: string;
  /** Version chain metadata. Optional so older persisted ArkData remains valid. */
  version?: number;
  supersedes?: string;
  previous_path?: string;
  operation?: "created" | "updated" | "deleted" | "renamed" | "referenced";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

/** Runtime guard at the network boundary; unknown fields are tolerated for forward compatibility. */
export function isRetrievalResult(value: unknown): value is RetrievalResult {
  if (!isRecord(value)) return false;
  const strategy = value.strategy;
  const status = value.status;
  const scope = value.scope;
  return value.contract === ARK_CONTRACT_VERSION
    && (strategy === "lexical-only" || strategy === "vector-only" || strategy === "hybrid" || strategy === "shadow")
    && (status === "available" || status === "degraded" || status === "unavailable")
    && Array.isArray(value.items)
    && Array.isArray(value.warnings)
    && isRecord(scope)
    && typeof scope.project_id === "string"
    && typeof scope.session_id === "string"
    && Array.isArray(scope.statuses)
    && typeof scope.include_archive === "boolean"
    && value.items.every((item) => isRecord(item)
      && stringValue(item.ref)
      && stringValue(item.source)
      && typeof item.score === "number"
      && Number.isFinite(item.score)
      && Array.isArray(item.provenance));
}

export function emptyRetrievalScope(): RetrievalScope {
  return { project_id: "", session_id: "", statuses: [], include_archive: false };
}

/**
 * Convert a legacy bare-list response to a labelled result while preserving
 * the old route as an explicitly degraded compatibility path.
 */
export function normalizeRetrievalResult(
  value: unknown,
  fallback: Partial<RetrievalResult> = {},
): RetrievalResult {
  if (isRetrievalResult(value)) return value;
  const rows = Array.isArray(value) ? value : [];
  const items: RetrievalItem[] = rows
    .filter(isRecord)
    .filter((row) => stringValue(row.ref))
    .map((row) => ({
      title: typeof row.title === "string" ? row.title : "",
      content: typeof row.content === "string" ? row.content : "",
      ref: String(row.ref),
      source: typeof row.source === "string" && row.source ? row.source : "vault",
      score: typeof row.score === "number" && Number.isFinite(row.score) ? row.score : 0,
      status: typeof row.status === "string" && row.status ? row.status : "active",
      project_id: typeof row.project_id === "string" ? row.project_id : "",
      session_id: typeof row.session_id === "string" ? row.session_id : "",
      provenance: Array.isArray(row.provenance) ? row.provenance as Provenance[] : [{
        source: typeof row.source === "string" && row.source ? row.source : "vault",
        ref: String(row.ref),
      }],
      context_of: Array.isArray(row.context_of) ? row.context_of.map(String) : [],
    }));
  return {
    contract: ARK_CONTRACT_VERSION,
    items,
    strategy: fallback.strategy ?? "lexical-only",
    status: fallback.status ?? "degraded",
    warnings: [...(fallback.warnings ?? []), "legacy_unlabelled_retrieval"],
    scope: fallback.scope ?? emptyRetrievalScope(),
    provenance: fallback.provenance ?? items.flatMap((item) => item.provenance),
  };
}
