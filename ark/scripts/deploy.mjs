// Deploy the built plugin files to an Obsidian vault.
import { copyFileSync, mkdirSync, existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const vaultRoot = (process.env.OBSIDIAN_VAULT_PATH || "").trim();
const files = ["main.js", "manifest.json", "styles.css"];

if (!vaultRoot) {
  console.error("[fail] OBSIDIAN_VAULT_PATH is not set.");
  console.error("[hint] Set it to the root of your Obsidian vault and retry.");
  process.exitCode = 1;
} else {
  const vaultPluginDir = join(vaultRoot, ".obsidian", "plugins", "ark");
  for (const file of files) {
    const source = join(scriptDir, "..", file);
    const destination = join(vaultPluginDir, file);
    if (!existsSync(source)) {
      console.warn("[skip] Missing " + file + "; run npm run build first.");
      continue;
    }
    try {
      mkdirSync(dirname(destination), { recursive: true });
      if (existsSync(destination) &&
          readFileSync(source).equals(readFileSync(destination))) {
        console.log("[same] " + file + " is unchanged.");
        continue;
      }
      copyFileSync(source, destination);
      console.log("[deploy] " + file + " -> " + destination);
    } catch (error) {
      console.error("[fail] Could not deploy " + file + ".");
      console.error("       Copy manually: " + source + " -> " + destination);
    }
  }
}
