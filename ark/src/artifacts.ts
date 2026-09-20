import type { ArtifactContract, Provenance } from "./contracts";
import type ArkOSPlugin from "./main";
import type { CrtSession } from "./settings";

export interface ArtifactToolOutput {
  name: string;
  output: string;
}

const PATH_KEYS = new Set([
  "path", "file", "file_path", "target_path", "output_path", "artifact_path",
  "created_path", "updated_path", "written_path", "modified_path",
]);
const COLLECTION_KEYS = new Set([
  "created", "updated", "modified", "written", "artifacts", "files", "outputs",
]);
const RENAME_OLD_KEYS = new Set(["old_path", "from", "source_path", "previous_path"]);
const RENAME_NEW_KEYS = new Set(["new_path", "to", "destination_path", "target_path"]);

function normalisePath(value: unknown): string {
  return String(value ?? "").trim().replace(/\\/g, "/").replace(/^\/+/, "");
}

function validVaultPath(value: string): boolean {
  return Boolean(value) && value.toLowerCase().endsWith(".md")
    && !value.split("/").some((part) => !part || part === "." || part === "..");
}

function stableHash(value: string): string {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function artifactId(projectId: string, sessionId: string, path: string): string {
  return `artifact-${stableHash(`${projectId}\0${sessionId}\0${path}`)}`;
}

function versionedArtifactId(projectId: string, path: string, version: number, timestamp: string): string {
  return `artifact-${stableHash(`${projectId}\0${path}\0${version}\0${timestamp}`)}`;
}

function collectPaths(value: unknown, key = "", out: Map<string, string> = new Map()): Map<string, string> {
  if (typeof value === "string") {
    const path = normalisePath(value);
    if ((PATH_KEYS.has(key) || COLLECTION_KEYS.has(key)) && validVaultPath(path)) {
      const operation = /^(created|written)$/i.test(key) || /_path$/i.test(key) && /created|written/i.test(key)
        ? "created" : /deleted|removed/i.test(key) ? "deleted" : "updated";
      out.set(path, operation);
    }
    return out;
  }
  if (Array.isArray(value)) {
    value.forEach((item) => collectPaths(item, key, out));
    return out;
  }
  if (value && typeof value === "object") {
    Object.entries(value as Record<string, unknown>).forEach(([childKey, child]) =>
      collectPaths(child, childKey, out));
  }
  return out;
}

function parseToolPaths(output: string, toolName: string): Map<string, string> {
  const paths = new Map<string, string>();
  try {
    collectPaths(JSON.parse(output || ""), "", paths);
  } catch {
    // Some legacy tools return prose. Only accept explicit .md path tokens.
    const matches = output.match(/[A-Za-z0-9_\-./\u4e00-\u9fff][^\n<>"']*\.md/gi) || [];
    matches.forEach((raw) => {
      const path = normalisePath(raw).replace(/[),.;:，。；：]+$/, "");
      if (validVaultPath(path)) paths.set(path, /write|create|patch|modify/i.test(toolName) ? "updated" : "referenced");
    });
  }
  return paths;
}

interface ParsedToolDetails { paths: Map<string, string>; renames: Map<string, string>; }

function parseToolDetails(output: string, toolName: string): ParsedToolDetails {
  const paths = parseToolPaths(output, toolName);
  const renames = new Map<string, string>();
  try {
    const root = JSON.parse(output || "");
    const visit = (value: unknown) => {
      if (Array.isArray(value)) { value.forEach(visit); return; }
      if (!value || typeof value !== "object") return;
      const record = value as Record<string, unknown>;
      let oldPath = "";
      let newPath = "";
      const explicitOperation = String(record.operation ?? record.status ?? "").toLowerCase();
      const toolOperation = /delete|remove/i.test(toolName) ? "deleted"
        : /rename|move/i.test(toolName) ? "renamed" : "";
      Object.entries(record).forEach(([key, child]) => {
        const path = normalisePath(child);
        if (RENAME_OLD_KEYS.has(key.toLowerCase()) && validVaultPath(path)) oldPath = path;
        if (RENAME_NEW_KEYS.has(key.toLowerCase()) && validVaultPath(path)) newPath = path;
        if ((PATH_KEYS.has(key.toLowerCase()) || COLLECTION_KEYS.has(key.toLowerCase())) && validVaultPath(path)) {
          const operation = /delete|remove/i.test(explicitOperation) || toolOperation === "deleted"
            ? "deleted" : /create|write/i.test(explicitOperation) ? "created" : "";
          if (operation) paths.set(path, operation);
        }
        visit(child);
      });
      if (oldPath && newPath && oldPath !== newPath) {
        renames.set(newPath, oldPath);
        paths.set(oldPath, "deleted");
        paths.set(newPath, "created");
      }
    };
    visit(root);
  } catch {
    // Prose outputs are handled by parseToolPaths; rename relations need structured fields.
  }
  return { paths, renames };
}

function answerRefs(messages: CrtSession["messages"]): string[] {
  const refs = new Set<string>();
  const re = /\[\[([^\]|\n]+)(?:#[^\]|\n]+)?(?:\|[^\]]*)?\]\]|(?:^|[\s(`])((?:[^\s<>()[\]{}]+\/)?[^\s<>()[\]{}]+\.md)/giu;
  messages.filter((message) => message.role === "assistant").forEach((message) => {
    let match: RegExpExecArray | null;
    while ((match = re.exec(message.content || ""))) {
      const value = normalisePath(match[1] || match[2]);
      if (validVaultPath(value)) refs.add(value);
    }
  });
  return [...refs];
}

function provenance(session: CrtSession, toolName: string, path: string): Provenance[] {
  return [{
    source: "agent",
    ref: `session/${session.id}`,
    session_id: session.id,
    project_id: session.projectId || "",
    captured_at: new Date().toISOString(),
    version: toolName || "agent",
  }, { source: "vault", ref: path }];
}

export function extractSessionArtifacts(
  session: CrtSession,
  toolOutputs: ArtifactToolOutput[] = [],
  now = new Date().toISOString(),
): ArtifactContract[] {
  const refs = answerRefs(session.messages);
  const byPath = new Map<string, { operation: string; tool: string }>();
  const previousPaths = new Map<string, string>();
  toolOutputs.forEach(({ name, output }) => {
    const parsed = parseToolDetails(output, name);
    parsed.paths.forEach((operation, path) => byPath.set(path, { operation, tool: name }));
    parsed.renames.forEach((oldPath, newPath) => previousPaths.set(newPath, oldPath));
  });
  return [...byPath.entries()].map(([path, meta]) => ({
    id: artifactId(session.projectId || "", session.id, path),
    project_id: session.projectId || "",
    kind: "note",
    path,
    source_refs: refs.filter((ref) => ref !== path),
    status: meta.operation === "deleted" ? "deleted" : "unvalidated",
    created_by: meta.tool ? `tool:${meta.tool}` : "agent",
    session_id: session.id,
    provenance: provenance(session, meta.tool, path),
    created_at: now,
    updated_at: now,
    operation: meta.operation as ArtifactContract["operation"],
    ...(previousPaths.has(path) ? { previous_path: previousPaths.get(path), operation: "renamed" as const } : {}),
  }));
}

export function validateArtifactPaths(plugin: ArkOSPlugin, artifacts: ArtifactContract[]): ArtifactContract[] {
  return artifacts.map((artifact) => {
    if (artifact.status === "deleted") return artifact;
    const file = plugin.app.vault.getAbstractFileByPath(artifact.path);
    const exists = Boolean(file && typeof (file as any).extension === "string");
    return { ...artifact, status: exists ? "validated" : "missing" };
  });
}

export function upsertArtifacts(plugin: ArkOSPlugin, artifacts: ArtifactContract[]): void {
  if (!Array.isArray(plugin.data.artifacts)) plugin.data.artifacts = [];
  const byId = new Map(plugin.data.artifacts.map((artifact) => [artifact.id, artifact]));
  artifacts.forEach((artifact) => {
    const existingById = byId.get(artifact.id);
    if (existingById && existingById.created_at === artifact.created_at
      && existingById.session_id === artifact.session_id
      && existingById.operation === artifact.operation) {
      byId.set(artifact.id, { ...existingById, ...artifact });
      return;
    }
    const sameEvent = plugin.data.artifacts.find((existing) =>
      existing.project_id === artifact.project_id && existing.path === artifact.path
      && existing.created_at === artifact.created_at && existing.session_id === artifact.session_id
      && existing.operation === artifact.operation);
    if (sameEvent) { byId.set(sameEvent.id, { ...sameEvent, ...artifact }); return; }
    const history = plugin.data.artifacts
      .filter((existing) => existing.project_id === artifact.project_id && existing.path === artifact.path)
      .sort((a, b) => (b.version || 1) - (a.version || 1));
    const previous = history[0];
    const version = (previous?.version || 0) + 1;
    const next = previous
      ? { ...artifact, id: versionedArtifactId(artifact.project_id, artifact.path, version, artifact.created_at), version, supersedes: previous.id }
      : { ...artifact, version };
    byId.set(next.id, next);
  });
  plugin.data.artifacts = [...byId.values()].sort((a, b) => b.updated_at.localeCompare(a.updated_at)).slice(0, 200);
}

export function artifactsForSession(plugin: ArkOSPlugin, sessionId: string): ArtifactContract[] {
  return (plugin.data.artifacts || []).filter((artifact) => artifact.session_id === sessionId);
}

const SAFE_PROJECT_ID = /^[A-Za-z0-9_\-\u4e00-\u9fff]{1,64}$/;

export function projectArtifactIndexPath(projectId: string): string | null {
  const id = String(projectId || "").trim();
  return SAFE_PROJECT_ID.test(id) ? `ark/projects/${id}/artifacts.json` : null;
}

export function serializeProjectArtifactIndex(artifacts: ArtifactContract[]): string {
  return JSON.stringify({ contract: "ark-artifacts-v1", generated_at: new Date().toISOString(), artifacts }, null, 2);
}

/** Write a rebuildable sidecar. Index failures never make an Agent run fail. */
export async function persistProjectArtifactIndex(plugin: ArkOSPlugin, projectId: string): Promise<boolean> {
  const path = projectArtifactIndexPath(projectId);
  if (!path) return false;
  try {
    const base = path.slice(0, path.lastIndexOf("/"));
    if (!(await plugin.app.vault.adapter.exists(base))) await plugin.app.vault.adapter.mkdir(base);
    await plugin.app.vault.adapter.write(path,
      serializeProjectArtifactIndex((plugin.data.artifacts || []).filter((item) => item.project_id === projectId)));
    return true;
  } catch {
    return false;
  }
}

export async function readProjectArtifactIndex(plugin: ArkOSPlugin, projectId: string): Promise<ArtifactContract[]> {
  const path = projectArtifactIndexPath(projectId);
  if (!path) return [];
  try {
    const raw = await plugin.app.vault.adapter.read(path);
    const parsed = JSON.parse(raw || "{}");
    return Array.isArray(parsed?.artifacts) ? parsed.artifacts.filter((item: unknown) => {
      const value = item as Partial<ArtifactContract>;
      return Boolean(value && typeof value.id === "string" && typeof value.path === "string"
        && value.project_id === projectId && validVaultPath(value.path));
    }) as ArtifactContract[] : [];
  } catch {
    return [];
  }
}

/** Restore project artifacts from its sidecar, keeping newer in-memory records. */
export async function restoreProjectArtifacts(plugin: ArkOSPlugin, projectId: string): Promise<number> {
  const loaded = await readProjectArtifactIndex(plugin, projectId);
  if (!loaded.length) return 0;
  upsertArtifacts(plugin, loaded);
  return loaded.length;
}
