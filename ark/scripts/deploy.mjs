// 部署构建产物到 Obsidian Vault（E:/peik1_books/.obsidian/plugins/ark）
import { copyFileSync, mkdirSync, existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { homedir } from "node:os";

const __dirname = dirname(fileURLToPath(import.meta.url));
const vaultRoot = process.env.OBSIDIAN_VAULT_PATH || join(homedir(), "Obsidian", "Default");
const vaultPluginDir = join(vaultRoot, ".obsidian", "plugins", "ark");
const files = ["main.js", "manifest.json", "styles.css"];

for (const f of files) {
  const src = join(__dirname, "..", f);
  const dest = join(vaultPluginDir, f);
  if (!existsSync(src)) {
    console.warn(`[skip] 缺少 ${f}（先执行 npm run build）`);
    continue;
  }
  try {
    mkdirSync(dirname(dest), { recursive: true });
    // 内容相同则跳过，避免无谓写操作（按内容而非字节大小比较，防同长度不同内容漏部署）
    if (existsSync(dest) && readFileSync(src).equals(readFileSync(dest))) {
      console.log(`[same] ${f} 未变化，跳过`);
      continue;
    }
    copyFileSync(src, dest);
    console.log(`[deploy] ${f} -> ${dest}`);
  } catch (err) {
    console.error(`[fail] ${f} 部署失败（文件可能被 Obsidian 占用或沙箱限制）`);
    console.error(`       请手动复制: ${src}  ->  ${dest}`);
  }
}
console.log("[deploy] 结束。若上方有 [fail]，请在 Obsidian 里重载插件后再试，或手动复制。");
